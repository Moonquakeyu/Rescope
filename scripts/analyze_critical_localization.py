#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Subset

from relmeasure3d.data import NPZRelationDataset, save_json
from relmeasure3d.publication_model import (
    PublicationAnalyticCriticalPooling,
    PublicationRelMeasure3D,
)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
    }


def channel_metrics(attention: np.ndarray, critical: np.ndarray) -> dict[str, float]:
    attention = np.asarray(attention, dtype=np.float64)
    attention /= max(float(attention.sum()), 1e-12)
    critical = np.asarray(critical, dtype=bool)
    critical_count = int(critical.sum())
    if critical_count == 0:
        return {
            "critical_mass": float("nan"),
            "topk_recall": float("nan"),
            "expected_distance_vox": float("nan"),
            "argmax_distance_vox": float("nan"),
        }
    flat_attention = attention.reshape(-1)
    flat_critical = critical.reshape(-1)
    k = min(critical_count, flat_attention.size)
    topk = np.argpartition(flat_attention, -k)[-k:]
    distance = distance_transform_edt(~critical)
    argmax_index = np.unravel_index(int(np.argmax(attention)), attention.shape)
    return {
        "critical_mass": float(attention[critical].sum()),
        "topk_recall": float(flat_critical[topk].mean()),
        "expected_distance_vox": float((attention * distance).sum()),
        "argmax_distance_vox": float(distance[argmax_index]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model",
        choices=("publication_relmeasure", "publication_analytic_pool"),
        required=True,
    )
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--split", choices=("validation", "calibration", "test"), default="test")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

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

    if args.model == "publication_analytic_pool":
        model = PublicationAnalyticCriticalPooling(base=args.base)
    else:
        model = PublicationRelMeasure3D(base=args.base)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device).eval()

    rows: list[dict[str, object]] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            output = model(batch["image"].to(device, non_blocking=True))
            attentions = torch.cat([output.attention_a, output.attention_b], dim=1)[0].float().cpu().numpy()
            critical = batch["critical"][0].numpy().astype(bool)
            channel_rows = [channel_metrics(attentions[channel], critical[channel]) for channel in range(2)]
            row: dict[str, object] = {
                "sample_id": batch["sample_id"][0],
                "patient_id": batch["sample_id"][0].split("__", 1)[0],
                "gt_mm": float(batch["distance_mm"].item()),
                "pred_mm": float(output.distance_mm.item()),
                "absolute_error_mm": abs(float(output.distance_mm.item()) - float(batch["distance_mm"].item())),
            }
            for key in channel_rows[0]:
                row[key] = float(np.nanmean([channel_rows[0][key], channel_rows[1][key]]))
                row[f"{key}_a"] = channel_rows[0][key]
                row[f"{key}_b"] = channel_rows[1][key]
            rows.append(row)

    metric_names = ("critical_mass", "topk_recall", "expected_distance_vox", "argmax_distance_vox")
    payload = {
        "status": "complete",
        "model": args.model,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "patients": len({row["patient_id"] for row in rows}),
        "relations": len(rows),
        "metrics": {name: summarize([float(row[name]) for row in rows]) for name in metric_names},
        "rows": rows,
    }
    save_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
