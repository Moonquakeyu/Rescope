#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from relmeasure3d.data import NPZRelationDataset, save_json
from relmeasure3d.publication_model import PublicationRelMeasure3D


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def finite_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def predict(
    model: torch.nn.Module,
    dataset: NPZRelationDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    gt_rows: list[float] = []
    pred_rows: list[float] = []
    sigma_rows: list[float] = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch in loader:
            output = model(batch["image"].to(device, non_blocking=True))
            gt_rows.extend(batch["distance_mm"].numpy().astype(float).tolist())
            pred_rows.extend(output.distance_mm.float().cpu().numpy().astype(float).tolist())
            sigma_rows.extend(output.sigma_mm.float().cpu().numpy().astype(float).tolist())
    return {
        "gt": np.asarray(gt_rows, dtype=float),
        "pred": np.asarray(pred_rows, dtype=float),
        "sigma": np.asarray(sigma_rows, dtype=float).clip(1e-4),
    }


def gaussian_nll(error: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.mean(0.5 * np.square(error / sigma) + np.log(sigma)))


def evaluate_run(calibration: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> dict[str, object]:
    cal_error = np.abs(calibration["gt"] - calibration["pred"])
    test_error = np.abs(test["gt"] - test["pred"])
    cal_sigma = calibration["sigma"]
    test_sigma = test["sigma"]
    temperature = float(np.sqrt(np.mean(np.square(cal_error / cal_sigma))))
    calibrated_sigma = test_sigma * temperature
    order = np.argsort(calibrated_sigma)
    fractions = (0.5, 0.7, 0.8, 0.9, 1.0)
    risk_coverage = {
        f"retain_{int(fraction * 100)}pct_mae_mm": float(test_error[order[: max(1, round(len(order) * fraction))]].mean())
        for fraction in fractions
    }
    cumulative_risk = np.cumsum(test_error[order]) / np.arange(1, len(order) + 1)
    intervals: dict[str, object] = {}
    for coverage in (0.90, 0.95):
        alpha = 1.0 - coverage
        q_scaled = finite_conformal_quantile(cal_error / cal_sigma, alpha)
        radius = q_scaled * test_sigma
        lower = np.maximum(0.0, test["pred"] - radius)
        upper = test["pred"] + radius
        q_constant = finite_conformal_quantile(cal_error, alpha)
        constant_lower = np.maximum(0.0, test["pred"] - q_constant)
        constant_upper = test["pred"] + q_constant
        intervals[f"coverage_{int(coverage * 100)}"] = {
            "target_coverage": coverage,
            "scaled_sigma_quantile": q_scaled,
            "empirical_coverage": float(np.mean((test["gt"] >= lower) & (test["gt"] <= upper))),
            "mean_interval_width_mm": float(np.mean(upper - lower)),
            "constant_empirical_coverage": float(
                np.mean((test["gt"] >= constant_lower) & (test["gt"] <= constant_upper))
            ),
            "constant_mean_interval_width_mm": float(np.mean(constant_upper - constant_lower)),
        }
    high_error = test_error > 2.0
    review_count = max(1, round(len(test_error) * 0.2))
    reviewed = np.argsort(-calibrated_sigma)[:review_count]
    failure_recall = float(high_error[reviewed].sum() / max(1, high_error.sum()))
    spearman = spearmanr(calibrated_sigma, test_error).statistic
    return {
        "calibration": {
            "relations": len(cal_error),
            "temperature_scale": temperature,
            "uncalibrated_nll": gaussian_nll(calibration["gt"] - calibration["pred"], cal_sigma),
            "temperature_scaled_nll": gaussian_nll(
                calibration["gt"] - calibration["pred"], cal_sigma * temperature
            ),
        },
        "test": {
            "relations": len(test_error),
            "mae_mm": float(test_error.mean()),
            "sigma_error_spearman": float(spearman),
            "auroc_error_gt_1mm": safe_auc(test_error > 1.0, calibrated_sigma),
            "auroc_error_gt_2mm": safe_auc(high_error, calibrated_sigma),
            "failure_recall_at_20pct_review_error_gt_2mm": failure_recall,
            "aurc_mm": float(cumulative_risk.mean()),
            "risk_coverage": risk_coverage,
            "conformal_intervals": intervals,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if len(args.checkpoints) != len(args.seeds):
        raise SystemExit("--checkpoints and --seeds must have equal length")
    dataset = NPZRelationDataset(args.data, crop_size=args.crop_size)
    by_id = {path.stem: index for index, path in enumerate(dataset.files)}
    split = json.loads(Path(args.split_json).read_text())
    indices = {
        name: [by_id[sample_id] for sample_id in split["samples"][name]]
        for name in ("calibration", "test")
    }
    device = torch.device("cuda")
    runs = []
    for seed, checkpoint_path in zip(args.seeds, args.checkpoints):
        model = PublicationRelMeasure3D(base=args.base).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        calibration = predict(model, dataset, indices["calibration"], device, args.batch_size, args.workers)
        test = predict(model, dataset, indices["test"], device, args.batch_size, args.workers)
        runs.append({"seed": seed, "checkpoint": checkpoint_path, **evaluate_run(calibration, test)})
        del model
        torch.cuda.empty_cache()
    aggregate = {
        "mae_mm": stats([float(run["test"]["mae_mm"]) for run in runs]),
        "sigma_error_spearman": stats([float(run["test"]["sigma_error_spearman"]) for run in runs]),
        "auroc_error_gt_1mm": stats([float(run["test"]["auroc_error_gt_1mm"]) for run in runs]),
        "auroc_error_gt_2mm": stats([float(run["test"]["auroc_error_gt_2mm"]) for run in runs]),
        "failure_recall_at_20pct_review_error_gt_2mm": stats(
            [float(run["test"]["failure_recall_at_20pct_review_error_gt_2mm"]) for run in runs]
        ),
        "aurc_mm": stats([float(run["test"]["aurc_mm"]) for run in runs]),
        "risk_coverage": {
            key: stats([float(run["test"]["risk_coverage"][key]) for run in runs])
            for key in runs[0]["test"]["risk_coverage"]
        },
        "conformal_intervals": {
            key: {
                metric: stats(
                    [float(run["test"]["conformal_intervals"][key][metric]) for run in runs]
                )
                for metric in (
                    "empirical_coverage", "mean_interval_width_mm",
                    "constant_empirical_coverage", "constant_mean_interval_width_mm",
                )
            }
            for key in runs[0]["test"]["conformal_intervals"]
        },
    }
    result = {"status": "complete", "seeds": args.seeds, "runs": runs, "aggregate": aggregate}
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
