#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from relmeasure3d.data import save_json


def interleaved_distance_strata(
    patients: list[str], distances: dict[str, list[float]], seed: int
) -> list[str]:
    ordered = sorted(patients, key=lambda patient: sum(distances[patient]) / len(distances[patient]))
    strata = [ordered[i * len(ordered) // 4 : (i + 1) * len(ordered) // 4] for i in range(4)]
    rng = random.Random(seed)
    for stratum in strata:
        rng.shuffle(stratum)
    return [
        stratum[index]
        for index in range(max(map(len, strata)))
        for stratum in strata
        if index < len(stratum)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    index = json.loads(Path(args.index).read_text())
    cached = {path.stem for path in Path(args.cache).glob("*.npz")}
    samples: dict[str, list[str]] = defaultdict(list)
    distances: dict[str, list[float]] = defaultdict(list)
    for relation in index["relations"]:
        sample_id = str(relation["sample_id"])
        if sample_id not in cached:
            continue
        patient = str(relation["patient_id"])
        samples[patient].append(sample_id)
        distances[patient].append(float(relation["distance_mm"]))
    target_patients = sorted(patient for patient in samples if patient.startswith("ToothFairy3S_"))
    source_patients = [
        patient for patient in samples
        if patient.startswith("ToothFairy3P_") or patient.startswith("ToothFairy3F_")
    ]
    ordered = interleaved_distance_strata(source_patients, distances, args.seed)
    validation_count = round(len(ordered) * 0.10)
    calibration_count = round(len(ordered) * 0.10)
    train_count = len(ordered) - validation_count - calibration_count
    patient_splits = {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "calibration": ordered[train_count + validation_count :],
        "test": target_patients,
    }
    sample_splits = {
        name: sorted(sample for patient in patients for sample in samples[patient])
        for name, patients in patient_splits.items()
    }
    all_patients = [patient for values in patient_splits.values() for patient in values]
    all_samples = [sample for values in sample_splits.values() for sample in values]
    if len(all_patients) != len(set(all_patients)) or len(all_samples) != len(set(all_samples)):
        raise RuntimeError("patient or sample leakage in scanner holdout split")
    result = {
        "seed": args.seed,
        "protocol": "train/validation/calibration on ToothFairy3 Sets A+B (P/F); held-out test on Set C (S)",
        "scanner_metadata_source": "https://ditto.ing.unimore.it/toothfairy3/",
        "scanner_note": "Official documentation states P/F share one acquisition machine and S uses a different machine.",
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
