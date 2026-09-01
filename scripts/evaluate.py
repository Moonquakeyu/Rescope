#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from relmeasure3d.data import NPZRelationDataset, save_json
from relmeasure3d.model import (
    CriticalAttentionRegression,
    CriticalRefinementRegression,
    DirectRegressionSmoke,
    SurfaceCriticalAttentionRegression,
)
from relmeasure3d.publication_model import (
    PublicationAnalyticCriticalPooling,
    PublicationAnalyticSoftmin,
    PublicationDirectRegression,
    PublicationRelMeasure3D,
    PublicationSurfaceGlobalRegression,
)


def describe(errors: np.ndarray, signed: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_mm": float(errors.mean()),
        "median_ae_mm": float(np.median(errors)),
        "p90_ae_mm": float(np.quantile(errors, 0.90)),
        "p95_ae_mm": float(np.quantile(errors, 0.95)),
        "bias_mm": float(signed.mean()),
        "pearson_r": float(np.corrcoef(gt, pred)[0, 1]),
    }


def patient_bootstrap(
    patient_ids: list[str],
    direct_errors: np.ndarray,
    refined_errors: np.ndarray,
    seed: int,
    repetitions: int = 10000,
) -> dict[str, float]:
    by_patient: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        by_patient[patient_id].append(index)
    patients = sorted(by_patient)
    rng = np.random.default_rng(seed)
    observed = float((direct_errors - refined_errors).mean())
    draws = np.empty(repetitions, dtype=np.float64)
    for draw in range(repetitions):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([np.asarray(by_patient[patient], dtype=int) for patient in sampled])
        draws[draw] = float((direct_errors[indices] - refined_errors[indices]).mean())
    return {
        "direct_minus_refined_mae_mm": observed,
        "patient_bootstrap_ci95_low_mm": float(np.quantile(draws, 0.025)),
        "patient_bootstrap_ci95_high_mm": float(np.quantile(draws, 0.975)),
        "improvement_probability": float((draws > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--direct-checkpoint", required=True)
    parser.add_argument("--refined-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--base", type=int, default=12)
    parser.add_argument(
        "--direct-model",
        choices=("direct", "publication_direct"),
        default="direct",
    )
    parser.add_argument(
        "--refined-model",
        choices=(
            "critical_refinement", "critical_attention", "search_measure",
            "publication_multitask", "publication_relmeasure",
            "publication_analytic_softmin", "publication_analytic_pool",
        ),
        default="critical_refinement",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "calibration", "test"),
        default=("validation", "calibration", "test"),
    )
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--fixed-offset", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--include-rows", action="store_true")
    args = parser.parse_args()

    dataset = NPZRelationDataset(
        args.data,
        crop_size=args.crop_size,
        fixed_offset=tuple(args.fixed_offset),
    )
    path_by_id = {path.stem: index for index, path in enumerate(dataset.files)}
    split = json.loads(Path(args.split_json).read_text())
    device = torch.device("cuda")

    if args.direct_model == "publication_direct":
        direct = PublicationDirectRegression(base=args.base).to(device)
    else:
        direct = DirectRegressionSmoke(base=args.base).to(device)
    direct_checkpoint = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=False)
    direct.load_state_dict(direct_checkpoint["model"])
    direct.eval()
    if args.refined_model == "critical_attention":
        refined = CriticalAttentionRegression(base=args.base).to(device)
    elif args.refined_model == "search_measure":
        refined = SurfaceCriticalAttentionRegression(base=args.base).to(device)
    elif args.refined_model == "publication_multitask":
        refined = PublicationSurfaceGlobalRegression(base=args.base).to(device)
    elif args.refined_model == "publication_relmeasure":
        refined = PublicationRelMeasure3D(base=args.base).to(device)
    elif args.refined_model == "publication_analytic_softmin":
        refined = PublicationAnalyticSoftmin(base=args.base).to(device)
    elif args.refined_model == "publication_analytic_pool":
        refined = PublicationAnalyticCriticalPooling(base=args.base).to(device)
    else:
        refined = CriticalRefinementRegression(base=args.base).to(device)
    refined_checkpoint = torch.load(args.refined_checkpoint, map_location="cpu", weights_only=False)
    refined.load_state_dict(refined_checkpoint["model"])
    refined.eval()

    payload: dict[str, object] = {
        "status": "complete",
        "base": args.base,
        "direct_model": args.direct_model,
        "refined_model": args.refined_model,
        "crop_size": args.crop_size,
        "fixed_offset": args.fixed_offset,
        "splits": {},
    }
    for split_name in args.splits:
        sample_ids = split["samples"][split_name]
        indices = [path_by_id[sample_id] for sample_id in sample_ids]
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        gt_rows: list[float] = []
        direct_rows: list[float] = []
        refined_rows: list[float] = []
        ordered_ids: list[str] = []
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                gt_rows.extend(batch["distance_mm"].cpu().numpy().astype(float).tolist())
                direct_rows.extend(direct(image).distance_mm.float().cpu().numpy().astype(float).tolist())
                refined_rows.extend(refined(image).distance_mm.float().cpu().numpy().astype(float).tolist())
                ordered_ids.extend(batch["sample_id"])
        gt = np.asarray(gt_rows)
        direct_pred = np.asarray(direct_rows)
        refined_pred = np.asarray(refined_rows)
        direct_signed = direct_pred - gt
        refined_signed = refined_pred - gt
        direct_errors = np.abs(direct_signed)
        refined_errors = np.abs(refined_signed)
        patients = [sample_id.split("__", 1)[0] for sample_id in ordered_ids]
        result = {
            "patients": len(set(patients)),
            "relations": len(gt),
            "direct": describe(direct_errors, direct_signed, gt, direct_pred),
            "refined": describe(refined_errors, refined_signed, gt, refined_pred),
            "paired": patient_bootstrap(patients, direct_errors, refined_errors, args.seed),
        }
        if args.include_rows:
            result["rows"] = [
                {
                    "sample_id": sample_id,
                    "patient_id": patient,
                    "gt_mm": float(gt_value),
                    "direct_mm": float(direct_value),
                    "refined_mm": float(refined_value),
                }
                for sample_id, patient, gt_value, direct_value, refined_value in zip(
                    ordered_ids, patients, gt, direct_pred, refined_pred
                )
            ]
        payload["splits"][split_name] = result  # type: ignore[index]
        print(json.dumps({"split": split_name, **result}), flush=True)
    save_json(args.output, payload)


if __name__ == "__main__":
    main()
