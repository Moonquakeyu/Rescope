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
from relmeasure3d.geometry import critical_band_voxels
from relmeasure3d.model import SegmentationDistanceBaseline
from relmeasure3d.publication_model import PublicationSegmentationBaseline


def describe(errors: np.ndarray) -> dict[str, float]:
    return {
        "mae_mm": float(errors.mean()),
        "median_ae_mm": float(np.median(errors)),
        "p90_ae_mm": float(np.quantile(errors, 0.90)),
        "p95_ae_mm": float(np.quantile(errors, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", type=int, default=24)
    parser.add_argument(
        "--model",
        choices=("smoke", "publication", "monai_segresnet", "monai_dynunet", "monai_mednext", "nnunet_resenc"),
        default="smoke",
    )
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--fixed-offset", nargs=3, type=int, default=(0, 0, 0))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test"), default=("validation", "test"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dataset = NPZRelationDataset(
        args.data,
        crop_size=args.crop_size,
        fixed_offset=tuple(args.fixed_offset),
    )
    path_by_id = {path.stem: index for index, path in enumerate(dataset.files)}
    split = json.loads(Path(args.split_json).read_text())
    device = torch.device("cuda")
    if args.model == "publication":
        model = PublicationSegmentationBaseline(base=args.base).to(device)
    elif args.model in ("monai_segresnet", "monai_dynunet", "monai_mednext", "nnunet_resenc"):
        from relmeasure3d.external_baselines import (
            MonaiDynUNetBaseline,
            MonaiMedNeXtBaseline,
            MonaiSegResNetBaseline,
            NNUNetResEncBaseline,
        )

        model_class = {
            "monai_segresnet": MonaiSegResNetBaseline,
            "monai_dynunet": MonaiDynUNetBaseline,
            "monai_mednext": MonaiMedNeXtBaseline,
            "nnunet_resenc": NNUNetResEncBaseline,
        }[args.model]
        model = model_class(base=args.base).to(device)
    else:
        model = SegmentationDistanceBaseline(base=args.base).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    payload: dict[str, object] = {
        "status": "complete",
        "threshold": args.threshold,
        "model": args.model,
        "crop_size": args.crop_size,
        "fixed_offset": args.fixed_offset,
        "splits": {},
    }
    for split_name in args.splits:
        indices = [path_by_id[sample_id] for sample_id in split["samples"][split_name]]
        loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=args.workers)
        predicted_errors: list[float] = []
        oracle_errors: list[float] = []
        dice_rows: list[float] = []
        failures: list[str] = []
        by_patient: dict[str, list[float]] = defaultdict(list)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch in loader:
                logits = model(batch["image"].to(device)).segmentation_logits
                predicted = (torch.sigmoid(logits) >= args.threshold).cpu().numpy()[0]
                target = batch["mask"].numpy()[0].astype(bool)
                sample_id = batch["sample_id"][0]
                gt = float(batch["distance_mm"].item())
                intersection = (predicted & target).sum(axis=(1, 2, 3))
                denominator = predicted.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
                dice_rows.append(float(np.mean((2 * intersection + 1) / (denominator + 1))))
                try:
                    _, _, predicted_distance = critical_band_voxels(predicted[0], predicted[1], (1.0, 1.0, 1.0))
                    predicted_errors.append(abs(predicted_distance - gt))
                    by_patient[sample_id.split("__", 1)[0]].append(abs(predicted_distance - gt))
                except ValueError:
                    failures.append(sample_id)
                _, _, oracle_distance = critical_band_voxels(target[0], target[1], (1.0, 1.0, 1.0))
                oracle_errors.append(abs(oracle_distance - gt))
        pred_array = np.asarray(predicted_errors, dtype=float)
        oracle_array = np.asarray(oracle_errors, dtype=float)
        result = {
            "patients": len(set(sample_id.split("__", 1)[0] for sample_id in split["samples"][split_name])),
            "relations": len(indices),
            "successful_measurements": len(predicted_errors),
            "failure_count": len(failures),
            "failure_rate": len(failures) / max(1, len(indices)),
            "mean_pair_dice": float(np.mean(dice_rows)),
            "segmentation_to_distance": describe(pred_array) if len(pred_array) else None,
            "oracle_resampling_floor": describe(oracle_array),
        }
        payload["splits"][split_name] = result  # type: ignore[index]
        print(json.dumps({"split": split_name, **result}), flush=True)
    save_json(args.output, payload)


if __name__ == "__main__":
    main()
