#!/usr/bin/env python3
"""Create a publication-facing statistical summary for the Patterns manuscript.

The script consumes completed per-case evaluation JSON files. It does not train,
select, or alter a model. All uncertainty resampling is clustered by patient so
that multiple anatomical relations from one scan are never treated as
independent observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon


SEEDS = (20260821, 20260822, 20260823)


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    path_template: str
    split: str = "test"
    role: str = "in_domain"


SETTINGS = (
    Setting(
        "tooth_ian",
        "ToothFairy3 tooth–IAN",
        "artifacts/publication_evidence_v1/seed{seed}_evaluation.json",
    ),
    Setting(
        "tooth_sinus",
        "ToothFairy3 tooth–sinus",
        "artifacts/cvpr_extension_v1/tooth_sinus_seed{seed}_evaluation.json",
    ),
    Setting(
        "totalseg_pancreas_aorta",
        "TotalSegmentator pancreas–aorta",
        "artifacts/cvpr_extension_v1/totalseg_pancreas_aorta_seed{seed}_evaluation.json",
    ),
    Setting(
        "amos_pancreas_aorta",
        "AMOS22 pancreas–aorta",
        "artifacts/amos22_external_v1/amos_indomain_seed{seed}_evaluation.json",
    ),
    Setting(
        "totalseg_to_amos",
        "TotalSegmentator→AMOS22 locked transfer",
        "artifacts/amos22_external_v1/totalseg_to_amos_seed{seed}_evaluation.json",
        role="locked_transfer",
    ),
    Setting(
        "msd_tumour_vessel",
        "MSD Task08 tumour–hepatic vessel",
        "artifacts/cvpr_extension_v1/msd_tumour_vessel_seed{seed}_evaluation.json",
        role="degenerate_negative_control",
    ),
)


DISTANCE_BINS = (
    ("contact_0", -math.inf, 1e-8),
    ("0_to_2_mm", 1e-8, 2.0),
    ("2_to_5_mm", 2.0, 5.0),
    ("5_to_10_mm", 5.0, 10.0),
    ("over_10_mm", 10.0, math.inf),
)


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def percentile(values: np.ndarray, q: float) -> float | None:
    return finite(np.quantile(values, q)) if len(values) else None


def rows_from(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload["splits"][split]["rows"]
    required = {"sample_id", "patient_id", "gt_mm", "direct_mm", "refined_mm"}
    for row in rows:
        missing = required.difference(row)
        if missing:
            raise KeyError(f"{path}: row missing {sorted(missing)}")
    return rows


def patient_vectors(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        gt = float(row["gt_mm"])
        grouped[str(row["patient_id"])].append(
            (abs(float(row["direct_mm"]) - gt), abs(float(row["refined_mm"]) - gt))
        )
    patients = sorted(grouped)
    direct = np.asarray([np.mean([x[0] for x in grouped[p]]) for p in patients], dtype=float)
    refined = np.asarray([np.mean([x[1] for x in grouped[p]]) for p in patients], dtype=float)
    return patients, direct, refined


def bootstrap_patient_delta(
    rows: list[dict[str, Any]], seed: int, repetitions: int
) -> dict[str, Any]:
    patients, direct, refined = patient_vectors(rows)
    delta = direct - refined
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for idx in range(repetitions):
        sampled = rng.integers(0, len(patients), len(patients))
        draws[idx] = float(np.mean(delta[sampled]))
    try:
        w = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(w.statistic), float(w.pvalue)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    return {
        "patients": len(patients),
        "relations": len(rows),
        "direct_patient_mean_mae_mm": float(np.mean(direct)),
        "relmeasure3d_patient_mean_mae_mm": float(np.mean(refined)),
        "direct_minus_relmeasure3d_mm": float(np.mean(delta)),
        "relative_improvement_percent": float(100.0 * np.mean(delta) / np.mean(direct)),
        "patient_bootstrap_ci95_mm": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "improvement_probability": float(np.mean(draws > 0.0)),
        "median_patient_delta_mm": float(np.median(delta)),
        "wilcoxon_patient_mean_error": {"statistic": statistic, "two_sided_p": p_value},
    }


def hierarchical_bootstrap(
    seed_rows: dict[int, list[dict[str, Any]]], rng_seed: int, repetitions: int
) -> dict[str, Any]:
    vectors = {seed: patient_vectors(rows)[1:] for seed, rows in seed_rows.items()}
    seed_ids = sorted(vectors)
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(repetitions, dtype=float)
    for idx in range(repetitions):
        sampled_seeds = rng.choice(seed_ids, size=len(seed_ids), replace=True)
        seed_deltas = []
        for seed in sampled_seeds:
            direct, refined = vectors[int(seed)]
            sampled_patients = rng.integers(0, len(direct), len(direct))
            seed_deltas.append(float(np.mean((direct - refined)[sampled_patients])))
        draws[idx] = float(np.mean(seed_deltas))
    observed = [float(np.mean(d - r)) for d, r in vectors.values()]
    return {
        "estimand": "mean across seeds of patient-mean Direct minus RelMeasure3D absolute error",
        "observed_mean_delta_mm": float(np.mean(observed)),
        "observed_seed_std_mm": float(np.std(observed, ddof=1)),
        "hierarchical_bootstrap_ci95_mm": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "improvement_probability": float(np.mean(draws > 0.0)),
        "repetitions": repetitions,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def bin_name(gt: float) -> str:
    for name, low, high in DISTANCE_BINS:
        if gt > low and gt <= high:
            return name
    raise RuntimeError(f"No bin for {gt}")


def summarize_distance_regimes(
    seed_rows: dict[int, list[dict[str, Any]]], repetitions: int, rng_seed: int
) -> dict[str, Any]:
    bins: dict[str, dict[int, list[dict[str, Any]]]] = {
        name: {seed: [] for seed in seed_rows} for name, _, _ in DISTANCE_BINS
    }
    for seed, rows in seed_rows.items():
        for row in rows:
            bins[bin_name(float(row["gt_mm"]))][seed].append(row)

    output: dict[str, Any] = {}
    for name, _, _ in DISTANCE_BINS:
        available = {seed: rows for seed, rows in bins[name].items() if rows}
        if not available:
            output[name] = {"relations": 0, "patients": 0}
            continue
        per_seed = []
        for seed, rows in available.items():
            gt = np.asarray([float(r["gt_mm"]) for r in rows])
            direct = np.asarray([float(r["direct_mm"]) for r in rows])
            refined = np.asarray([float(r["refined_mm"]) for r in rows])
            direct_mae = float(np.mean(np.abs(direct - gt)))
            refined_mae = float(np.mean(np.abs(refined - gt)))
            per_seed.append(
                {
                    "seed": seed,
                    "direct_mae_mm": direct_mae,
                    "relmeasure3d_mae_mm": refined_mae,
                    "delta_mm": direct_mae - refined_mae,
                }
            )
        first_rows = next(iter(available.values()))
        direct_values = np.asarray([entry["direct_mae_mm"] for entry in per_seed])
        refined_values = np.asarray([entry["relmeasure3d_mae_mm"] for entry in per_seed])
        delta_values = direct_values - refined_values
        result: dict[str, Any] = {
            "patients": len({str(r["patient_id"]) for r in first_rows}),
            "relations": len(first_rows),
            "direct_mae_mm_mean_across_seeds": float(np.mean(direct_values)),
            "relmeasure3d_mae_mm_mean_across_seeds": float(np.mean(refined_values)),
            "direct_minus_relmeasure3d_mm": float(np.mean(delta_values)),
            "relative_improvement_percent": float(100.0 * np.mean(delta_values) / np.mean(direct_values)),
            "per_seed": per_seed,
        }
        if len(available) == len(seed_rows):
            hierarchical = hierarchical_bootstrap(available, rng_seed + len(output), repetitions)
            result["hierarchical_bootstrap_ci95_mm"] = hierarchical["hierarchical_bootstrap_ci95_mm"]
            result["improvement_probability"] = hierarchical["improvement_probability"]
        output[name] = result
    return output


def ground_truth_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gt = np.asarray([float(row["gt_mm"]) for row in rows], dtype=float)
    return {
        "patients": len({str(row["patient_id"]) for row in rows}),
        "relations": len(rows),
        "zero_distance_count": int(np.sum(gt <= 1e-8)),
        "zero_distance_percent": float(100.0 * np.mean(gt <= 1e-8)),
        "mean_mm": float(np.mean(gt)),
        "std_mm": float(np.std(gt, ddof=1)) if len(gt) > 1 else 0.0,
        "quantiles_mm": {
            "min": float(np.min(gt)),
            "p25": percentile(gt, 0.25),
            "median": percentile(gt, 0.5),
            "p75": percentile(gt, 0.75),
            "p95": percentile(gt, 0.95),
            "max": float(np.max(gt)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/patterns_statistics_v1"))
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()

    output_dir = args.root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_payload: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    flat_rows: list[dict[str, Any]] = []

    for setting_index, setting in enumerate(SETTINGS):
        seed_rows: dict[int, list[dict[str, Any]]] = {}
        per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            path = args.root / setting.path_template.format(seed=seed)
            rows = rows_from(path, setting.split)
            seed_rows[seed] = rows
            stats = bootstrap_patient_delta(rows, seed + setting_index * 1000, args.bootstrap)
            per_seed[str(seed)] = stats
            raw_p_values[f"{setting.key}:{seed}"] = stats["wilcoxon_patient_mean_error"]["two_sided_p"]
            flat_rows.append(
                {
                    "setting": setting.key,
                    "role": setting.role,
                    "seed": seed,
                    "patients": stats["patients"],
                    "relations": stats["relations"],
                    "direct_patient_mean_mae_mm": stats["direct_patient_mean_mae_mm"],
                    "relmeasure3d_patient_mean_mae_mm": stats["relmeasure3d_patient_mean_mae_mm"],
                    "direct_minus_relmeasure3d_mm": stats["direct_minus_relmeasure3d_mm"],
                    "ci95_low_mm": stats["patient_bootstrap_ci95_mm"][0],
                    "ci95_high_mm": stats["patient_bootstrap_ci95_mm"][1],
                    "improvement_probability": stats["improvement_probability"],
                    "relative_improvement_percent": stats["relative_improvement_percent"],
                    "wilcoxon_p": stats["wilcoxon_patient_mean_error"]["two_sided_p"],
                }
            )
        settings_payload[setting.key] = {
            "label": setting.label,
            "role": setting.role,
            "ground_truth_profile": ground_truth_profile(seed_rows[SEEDS[0]]),
            "per_seed_patient_statistics": per_seed,
            "hierarchical_across_seed_patient_bootstrap": hierarchical_bootstrap(
                seed_rows, 20260828 + setting_index, args.bootstrap
            ),
            "distance_regimes": summarize_distance_regimes(
                seed_rows, args.bootstrap, 20261828 + setting_index * 10
            ),
        }

    adjusted = holm_adjust(raw_p_values)
    for row in flat_rows:
        key = f"{row['setting']}:{row['seed']}"
        row["wilcoxon_holm_p_across_all_tests"] = adjusted[key]
        settings_payload[row["setting"]]["per_seed_patient_statistics"][str(row["seed"])][
            "wilcoxon_patient_mean_error"
        ]["holm_p_across_all_tests"] = adjusted[key]

    scanner_summary_path = args.root / "artifacts/cvpr_gap_v1/scanner_holdout_summary.json"
    scanner_summary = json.loads(scanner_summary_path.read_text())
    payload = {
        "status": "complete",
        "generated_from": "completed per-case evaluation JSON; no model selection or test-set tuning",
        "statistical_unit": "patient (relations within a scan are clustered)",
        "seeds": list(SEEDS),
        "bootstrap_repetitions": args.bootstrap,
        "multiple_testing": f"Holm adjustment across {len(SETTINGS)} settings × {len(SEEDS)} seeds for patient-level Wilcoxon tests",
        "settings": settings_payload,
        "summary_only_settings": {
            "scanner_holdout": {
                "label": "ToothFairy3 P/F→S scanner holdout",
                "reason": "Legacy evaluation retained patient-clustered bootstrap summaries but not per-case rows; no retrospective reconstruction was used.",
                "reported_summary": scanner_summary,
            }
        },
    }
    json_path = output_dir / "patterns_unified_statistics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    csv_path = output_dir / "patterns_patient_statistics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"status": "complete", "json": str(json_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
