#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset

from relmeasure3d.data import NPZRelationDataset, save_json
from relmeasure3d.external_baselines import (
    MonaiMedNeXtBaseline,
    MonaiSegResNetBaseline,
    NNUNetResEncBaseline,
)
from relmeasure3d.geometry import critical_band_voxels


def surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)


def surface_metrics(predicted: np.ndarray, target: np.ndarray, tolerance_mm: float = 1.0) -> tuple[float, float]:
    pred_surface = surface(predicted)
    target_surface = surface(target)
    if not pred_surface.any() or not target_surface.any():
        return float("nan"), float("nan")
    distance_to_pred = distance_transform_edt(~pred_surface)
    distance_to_target = distance_transform_edt(~target_surface)
    pred_to_target = distance_to_target[pred_surface]
    target_to_pred = distance_to_pred[target_surface]
    hd95 = float(np.quantile(np.concatenate([pred_to_target, target_to_pred]), 0.95))
    nsd = float(
        (np.count_nonzero(pred_to_target <= tolerance_mm) + np.count_nonzero(target_to_pred <= tolerance_mm))
        / (len(pred_to_target) + len(target_to_pred))
    )
    return nsd, hd95


def critical_boundary_error(predicted: np.ndarray, critical: np.ndarray) -> float:
    pred_surface = surface(predicted)
    if not pred_surface.any() or not np.asarray(critical, dtype=bool).any():
        return float("nan")
    return float(distance_transform_edt(~pred_surface)[np.asarray(critical, dtype=bool)].mean())


def correlation(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    x = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["measurement_error_mm"]) for row in rows], dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    rho, p_value = spearmanr(x[valid], y[valid])
    return {"spearman_rho": float(rho), "p_value": float(p_value), "n": int(valid.sum())}


def patient_bootstrap_correlations(
    rows: list[dict[str, object]], key: str, seed: int, repetitions: int
) -> dict[str, float]:
    patient_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        patient_to_indices.setdefault(str(row["patient_id"]), []).append(index)
    patients = sorted(patient_to_indices)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(repetitions):
        selected = rng.choice(patients, size=len(patients), replace=True)
        indices = [index for patient in selected for index in patient_to_indices[patient]]
        x = np.asarray([float(rows[index][key]) for index in indices], dtype=np.float64)
        y = np.asarray([float(rows[index]["measurement_error_mm"]) for index in indices], dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= 3:
            rho = float(spearmanr(x[valid], y[valid]).statistic)
            if np.isfinite(rho):
                draws.append(rho)
    array = np.asarray(draws, dtype=np.float64)
    return {
        "ci95_low": float(np.quantile(array, 0.025)),
        "ci95_high": float(np.quantile(array, 0.975)),
        "bootstrap_draws": len(draws),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", choices=("monai_segresnet", "monai_mednext", "nnunet_resenc"), required=True)
    parser.add_argument("--base", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    model_class = {
        "monai_segresnet": MonaiSegResNetBaseline,
        "monai_mednext": MonaiMedNeXtBaseline,
        "nnunet_resenc": NNUNetResEncBaseline,
    }[args.model]
    model = model_class(base=args.base)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device).eval()

    dataset = NPZRelationDataset(args.data, crop_size=args.crop_size)
    by_id = {path.stem: index for index, path in enumerate(dataset.files)}
    split = json.loads(Path(args.split_json).read_text())
    ids = list(split["samples"][args.split])
    loader = DataLoader(
        Subset(dataset, [by_id[sample_id] for sample_id in ids]),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    rows: list[dict[str, object]] = []
    failures: list[str] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True)).segmentation_logits
            predicted = (torch.sigmoid(logits)[0].float().cpu().numpy() >= args.threshold)
            target = batch["mask"][0].numpy().astype(bool)
            critical = batch["critical"][0].numpy().astype(bool)
            sample_id = batch["sample_id"][0]
            gt_mm = float(batch["distance_mm"].item())
            channel_dice: list[float] = []
            channel_nsd: list[float] = []
            channel_hd95: list[float] = []
            channel_critical_error: list[float] = []
            for channel in range(2):
                intersection = float(np.count_nonzero(predicted[channel] & target[channel]))
                denominator = float(np.count_nonzero(predicted[channel]) + np.count_nonzero(target[channel]))
                channel_dice.append((2.0 * intersection + 1.0) / (denominator + 1.0))
                nsd, hd95 = surface_metrics(predicted[channel], target[channel])
                channel_nsd.append(nsd)
                channel_hd95.append(hd95)
                channel_critical_error.append(critical_boundary_error(predicted[channel], critical[channel]))
            try:
                _, _, pred_mm = critical_band_voxels(predicted[0], predicted[1], (1.0, 1.0, 1.0))
                measurement_error = abs(float(pred_mm) - gt_mm)
            except ValueError:
                pred_mm = float("nan")
                measurement_error = float("nan")
                failures.append(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "patient_id": sample_id.split("__", 1)[0],
                    "gt_mm": gt_mm,
                    "pred_mm": float(pred_mm),
                    "measurement_error_mm": float(measurement_error),
                    "pair_dice": float(np.mean(channel_dice)),
                    "dice_error": float(1.0 - np.mean(channel_dice)),
                    "nsd_1mm": float(np.nanmean(channel_nsd)),
                    "nsd_error": float(1.0 - np.nanmean(channel_nsd)),
                    "hd95_mm": float(np.nanmean(channel_hd95)),
                    "critical_boundary_error_mm": float(np.nanmean(channel_critical_error)),
                }
            )

    metric_keys = ("dice_error", "nsd_error", "hd95_mm", "critical_boundary_error_mm")
    correlations = {}
    for key in metric_keys:
        correlations[key] = {
            **correlation(rows, key),
            **patient_bootstrap_correlations(rows, key, args.seed, args.bootstrap),
        }
    payload = {
        "status": "complete",
        "model": args.model,
        "threshold": args.threshold,
        "split": args.split,
        "patients": len({row["patient_id"] for row in rows}),
        "relations": len(rows),
        "failure_count": len(failures),
        "failure_rate": len(failures) / max(1, len(rows)),
        "correlations_with_measurement_error": correlations,
        "rows": rows,
    }
    save_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
