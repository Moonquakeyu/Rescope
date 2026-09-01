#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from relmeasure3d.data import save_json


def allocate_counts(total: int) -> tuple[int, int, int, int]:
    if total < 10:
        raise ValueError("at least 10 patients are required for a four-way split")
    validation = max(1, round(total * 0.1))
    calibration = max(1, round(total * 0.1))
    test = max(1, round(total * 0.1))
    train = total - validation - calibration - test
    return train, validation, calibration, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    payload = json.loads(Path(args.index).read_text())
    cached = {path.stem for path in Path(args.cache).glob("*.npz")}
    by_patient: dict[str, list[str]] = defaultdict(list)
    distances_by_patient: dict[str, list[float]] = defaultdict(list)
    for relation in payload["relations"]:
        sample_id = str(relation["sample_id"])
        if sample_id in cached:
            patient_id = str(relation["patient_id"])
            by_patient[patient_id].append(sample_id)
            distances_by_patient[patient_id].append(float(relation["distance_mm"]))

    patients_by_distance = sorted(by_patient, key=lambda patient: sum(distances_by_patient[patient]) / len(distances_by_patient[patient]))
    strata = [
        patients_by_distance[index * len(patients_by_distance) // 4 : (index + 1) * len(patients_by_distance) // 4]
        for index in range(4)
    ]
    rng = random.Random(args.seed)
    for stratum in strata:
        rng.shuffle(stratum)
    patients = [
        stratum[index]
        for index in range(max(map(len, strata)))
        for stratum in strata
        if index < len(stratum)
    ]
    n_train, n_val, n_cal, n_test = allocate_counts(len(patients))
    boundaries = (n_train, n_train + n_val, n_train + n_val + n_cal)
    patient_splits = {
        "train": patients[: boundaries[0]],
        "validation": patients[boundaries[0] : boundaries[1]],
        "calibration": patients[boundaries[1] : boundaries[2]],
        "test": patients[boundaries[2] :],
    }
    sample_splits = {
        name: sorted(sample for patient in ids for sample in by_patient[patient])
        for name, ids in patient_splits.items()
    }
    all_samples = [sample for values in sample_splits.values() for sample in values]
    if len(all_samples) != len(set(all_samples)):
        raise RuntimeError("sample leakage across splits")
    all_patients = [patient for values in patient_splits.values() for patient in values]
    if len(all_patients) != len(set(all_patients)):
        raise RuntimeError("patient leakage across splits")

    result = {
        "seed": args.seed,
        "stratification": "patient mean relation distance quartiles",
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
