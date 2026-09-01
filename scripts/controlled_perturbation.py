#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon

from relmeasure3d.data import save_json
from relmeasure3d.geometry import binary_surface, critical_band_voxels


def nearest_mask_voxels(mask: np.ndarray, center: np.ndarray, count: int, spacing: float) -> np.ndarray:
    coords = np.argwhere(mask)
    distance_sq = np.square((coords - center[None]) * spacing).sum(axis=1)
    chosen = np.argpartition(distance_sq, count - 1)[:count]
    selected = np.zeros_like(mask, dtype=bool)
    selected[tuple(coords[chosen].T)] = True
    return selected


def perturb_pair(mask_a: np.ndarray, mask_b: np.ndarray, spacing: float, radius_mm: float) -> dict[str, float]:
    surf_a = binary_surface(mask_a)
    surf_b = binary_surface(mask_b)
    if not surf_a.any() or not surf_b.any():
        raise ValueError("empty surface")
    dt_b = distance_transform_edt(~surf_b, sampling=(spacing,) * 3)
    critical_center = np.argwhere(surf_a)[int(np.argmin(dt_b[surf_a]))]
    critical_seed = np.zeros_like(mask_a, dtype=bool)
    critical_seed[tuple(critical_center)] = True
    distance_from_critical = distance_transform_edt(~critical_seed, sampling=(spacing,) * 3)
    noncritical_center = np.argwhere(surf_a)[int(np.argmax(distance_from_critical[surf_a]))]

    coordinates = np.indices(mask_a.shape).transpose(1, 2, 3, 0)
    critical_ball = np.linalg.norm((coordinates - critical_center) * spacing, axis=-1) <= radius_mm
    noncritical_ball = np.linalg.norm((coordinates - noncritical_center) * spacing, axis=-1) <= radius_mm
    count = min(int(np.count_nonzero(mask_a & critical_ball)), int(np.count_nonzero(mask_a & noncritical_ball)))
    if count < 4 or count >= int(mask_a.sum()):
        raise ValueError(f"invalid matched perturbation size: {count}")

    critical_remove = nearest_mask_voxels(mask_a, critical_center, count, spacing)
    noncritical_remove = nearest_mask_voxels(mask_a, noncritical_center, count, spacing)
    critical_mask = mask_a & ~critical_remove
    noncritical_mask = mask_a & ~noncritical_remove
    _, _, base_distance = critical_band_voxels(mask_a, mask_b, (spacing,) * 3)
    _, _, critical_distance = critical_band_voxels(critical_mask, mask_b, (spacing,) * 3)
    _, _, noncritical_distance = critical_band_voxels(noncritical_mask, mask_b, (spacing,) * 3)
    denominator = 2 * int(mask_a.sum()) - count
    matched_dice = 2 * (int(mask_a.sum()) - count) / denominator
    return {
        "base_distance_mm": base_distance,
        "critical_distance_mm": critical_distance,
        "noncritical_distance_mm": noncritical_distance,
        "critical_delta_mm": abs(critical_distance - base_distance),
        "noncritical_delta_mm": abs(noncritical_distance - base_distance),
        "paired_delta_mm": abs(critical_distance - base_distance) - abs(noncritical_distance - base_distance),
        "critical_dice": matched_dice,
        "noncritical_dice": matched_dice,
        "removed_voxels": count,
    }


def bootstrap_patient_difference(rows: list[dict[str, object]], seed: int, replicates: int) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["patient_id"])].append(float(row["paired_delta_mm"]))
    values = np.asarray([np.mean(items) for items in grouped.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        estimates[index] = rng.choice(values, size=len(values), replace=True).mean()
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius-mm", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for path in sorted(Path(args.cache).glob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as data:
                masks = data["mask"].astype(bool)
                spacing = float(data["spacing_mm"])
                sample_id = str(data["sample_id"])
            values = perturb_pair(masks[0], masks[1], spacing=spacing, radius_mm=args.radius_mm)
            rows.append({"sample_id": sample_id, "patient_id": sample_id.split("__", 1)[0], **values})
        except Exception as exc:
            failures.append({"sample_id": path.stem, "error": repr(exc)})

    if len(rows) < 10:
        raise SystemExit("too few valid perturbation pairs")
    critical = np.asarray([float(row["critical_delta_mm"]) for row in rows])
    noncritical = np.asarray([float(row["noncritical_delta_mm"]) for row in rows])
    paired = critical - noncritical
    ci_low, ci_high = bootstrap_patient_difference(rows, args.seed, args.bootstrap)
    try:
        statistic, p_value = wilcoxon(critical, noncritical, alternative="greater")
    except ValueError:
        statistic, p_value = float("nan"), float("nan")
    summary = {
        "status": "complete",
        "radius_mm": args.radius_mm,
        "valid_relations": len(rows),
        "patients": len({str(row["patient_id"]) for row in rows}),
        "failures": failures,
        "critical_delta_mean_mm": float(critical.mean()),
        "critical_delta_median_mm": float(np.median(critical)),
        "noncritical_delta_mean_mm": float(noncritical.mean()),
        "noncritical_delta_median_mm": float(np.median(noncritical)),
        "paired_mean_difference_mm": float(paired.mean()),
        "patient_bootstrap_95ci_mm": [ci_low, ci_high],
        "critical_greater_fraction": float(np.mean(critical > noncritical)),
        "wilcoxon_greater_statistic": float(statistic),
        "wilcoxon_greater_p": float(p_value),
        "max_matched_dice_difference": float(
            max(abs(float(row["critical_dice"]) - float(row["noncritical_dice"])) for row in rows)
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "summary.json", summary)
    with (output / "per_relation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
