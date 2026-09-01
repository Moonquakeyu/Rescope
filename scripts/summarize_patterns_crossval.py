#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from relmeasure3d.data import save_json


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p95": float(np.quantile(array, 0.95)),
        "n": len(values),
    }


def patient_bootstrap(
    patient_rows: dict[str, list[tuple[float, float, float]]], seed: int, repetitions: int
) -> dict[str, dict[str, float]]:
    patients = sorted(patient_rows)
    rng = np.random.default_rng(seed)
    ours_minus_direct: list[float] = []
    ours_minus_segmentation: list[float] = []
    for _ in range(repetitions):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        rows = [row for patient in sampled for row in patient_rows[patient]]
        direct = np.asarray([row[0] for row in rows])
        ours = np.asarray([row[1] for row in rows])
        segmentation = np.asarray([row[2] for row in rows])
        ours_minus_direct.append(float(np.mean(direct - ours)))
        ours_minus_segmentation.append(float(np.mean(segmentation - ours)))

    def interval(values: list[float]) -> dict[str, float]:
        array = np.asarray(values)
        return {
            "mean_improvement_mm": float(array.mean()),
            "ci95_low_mm": float(np.quantile(array, 0.025)),
            "ci95_high_mm": float(np.quantile(array, 0.975)),
            "improvement_probability": float(np.mean(array > 0)),
        }

    return {"ours_vs_direct": interval(ours_minus_direct), "ours_vs_segmentation": interval(ours_minus_segmentation)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    root = Path(args.root)
    seed_summaries: list[dict[str, object]] = []
    for seed in args.seeds:
        direct_errors: list[float] = []
        ours_errors: list[float] = []
        segmentation_errors: list[float] = []
        patient_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        seen_samples: set[str] = set()
        for fold in range(args.folds):
            measurement_path = root / f"fold{fold}_seed{seed}_measurement.json"
            segmentation_path = root / f"fold{fold}_seed{seed}_segmentation_rows.json"
            measurement = json.loads(measurement_path.read_text())["splits"]["test"]["rows"]
            segmentation = json.loads(segmentation_path.read_text())["rows"]
            segmentation_by_id = {row["sample_id"]: row for row in segmentation}
            for row in measurement:
                sample_id = str(row["sample_id"])
                if sample_id in seen_samples:
                    raise RuntimeError(f"duplicate out-of-fold sample {sample_id} for seed {seed}")
                seen_samples.add(sample_id)
                segmentation_row = segmentation_by_id[sample_id]
                gt = float(row["gt_mm"])
                direct_error = abs(float(row["direct_mm"]) - gt)
                ours_error = abs(float(row["refined_mm"]) - gt)
                segmentation_error = float(segmentation_row["measurement_error_mm"])
                if not np.isfinite(segmentation_error):
                    continue
                direct_errors.append(direct_error)
                ours_errors.append(ours_error)
                segmentation_errors.append(segmentation_error)
                patient_rows[str(row["patient_id"])].append((direct_error, ours_error, segmentation_error))
        seed_summaries.append(
            {
                "seed": seed,
                "patients": len(patient_rows),
                "relations": len(ours_errors),
                "direct": describe(direct_errors),
                "relmeasure3d": describe(ours_errors),
                "segmentation": describe(segmentation_errors),
                "paired_patient_bootstrap": patient_bootstrap(patient_rows, seed, args.bootstrap),
            }
        )

    payload = {
        "status": "complete",
        "protocol": "patient-level 5-fold out-of-fold evaluation; rotating validation fold",
        "seeds": args.seeds,
        "folds": args.folds,
        "per_seed": seed_summaries,
        "across_seed_mae": {
            method: describe([float(row[method]["mean"]) for row in seed_summaries])
            for method in ("direct", "relmeasure3d", "segmentation")
        },
    }
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
