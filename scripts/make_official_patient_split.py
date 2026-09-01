#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from relmeasure3d.data import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    args = parser.parse_args()

    payload = json.loads(Path(args.index).read_text())
    cached = {path.stem for path in Path(args.cache).glob("*.npz")}
    by_patient: dict[str, list[str]] = defaultdict(list)
    official: dict[str, str] = {}
    for relation in payload["relations"]:
        sample_id = str(relation["sample_id"])
        if sample_id not in cached:
            continue
        patient_id = str(relation["patient_id"])
        by_patient[patient_id].append(sample_id)
        official[patient_id] = str(relation.get("official_split", "unknown"))

    train_candidates = sorted(patient for patient in by_patient if official[patient] == "train")
    validation = sorted(patient for patient in by_patient if official[patient] == "val")
    test = sorted(patient for patient in by_patient if official[patient] == "test")
    if not train_candidates or not validation or not test:
        raise RuntimeError("official train/val/test groups are incomplete after cache filtering")
    rng = random.Random(args.seed)
    rng.shuffle(train_candidates)
    calibration_count = max(1, round(len(train_candidates) * args.calibration_fraction))
    calibration = sorted(train_candidates[:calibration_count])
    train = sorted(train_candidates[calibration_count:])
    patient_splits = {"train": train, "validation": validation, "calibration": calibration, "test": test}
    sample_splits = {
        name: sorted(sample for patient in patients for sample in by_patient[patient])
        for name, patients in patient_splits.items()
    }
    all_patients = [patient for patients in patient_splits.values() for patient in patients]
    all_samples = [sample for samples in sample_splits.values() for sample in samples]
    if len(all_patients) != len(set(all_patients)) or len(all_samples) != len(set(all_samples)):
        raise RuntimeError("patient or sample leakage across splits")
    result = {
        "seed": args.seed,
        "protocol": "TotalSegmentator official train/val/test; calibration held out from official train",
        "source_index": args.index,
        "cache": args.cache,
        "patients": patient_splits,
        "samples": sample_splits,
        "counts": {
            name: {"patients": len(patient_splits[name]), "samples": len(sample_splits[name])}
            for name in patient_splits
        },
    }
    save_json(args.output, result)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
