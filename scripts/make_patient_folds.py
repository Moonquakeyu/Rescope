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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    payload = json.loads(Path(args.index).read_text())
    cached = {path.stem for path in Path(args.cache).glob("*.npz")}
    samples_by_patient: dict[str, list[str]] = defaultdict(list)
    distances_by_patient: dict[str, list[float]] = defaultdict(list)
    for relation in payload["relations"]:
        sample_id = str(relation["sample_id"])
        if sample_id not in cached:
            continue
        patient = str(relation["patient_id"])
        samples_by_patient[patient].append(sample_id)
        distances_by_patient[patient].append(float(relation["distance_mm"]))

    patients = sorted(
        samples_by_patient,
        key=lambda patient: sum(distances_by_patient[patient]) / len(distances_by_patient[patient]),
    )
    rng = random.Random(args.seed)
    strata = [patients[i * len(patients) // 5 : (i + 1) * len(patients) // 5] for i in range(5)]
    fold_patients: list[list[str]] = [[] for _ in range(args.folds)]
    for stratum in strata:
        rng.shuffle(stratum)
        for index, patient in enumerate(stratum):
            fold_patients[index % args.folds].append(patient)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"seed": args.seed, "folds": args.folds, "items": []}
    all_patients = set(samples_by_patient)
    for fold in range(args.folds):
        test_patients = sorted(fold_patients[fold])
        validation_patients = sorted(fold_patients[(fold + 1) % args.folds])
        train_patients = sorted(all_patients - set(test_patients) - set(validation_patients))
        patients_for_split = {
            "train": train_patients,
            "validation": validation_patients,
            "calibration": [],
            "test": test_patients,
        }
        samples_for_split = {
            name: sorted(sample for patient in ids for sample in samples_by_patient[patient])
            for name, ids in patients_for_split.items()
        }
        result = {
            "fold": fold,
            "seed": args.seed,
            "stratification": "patient mean distance quintiles; rotating validation fold",
            "patients": patients_for_split,
            "samples": samples_for_split,
            "counts": {
                name: {"patients": len(patients_for_split[name]), "samples": len(samples_for_split[name])}
                for name in patients_for_split
            },
        }
        path = output / f"fold_{fold}.json"
        save_json(path, result)
        manifest["items"].append({"fold": fold, "path": str(path), "counts": result["counts"]})  # type: ignore[index]
    save_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
