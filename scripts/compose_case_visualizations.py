#!/usr/bin/env python3
"""Compose downloaded per-model panels into publication-ready case plates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


TEXT = "#16324F"
MUTED = "#5B6573"
ACCENT = "#0072B2"
OBJECT_A = "#E76F51"
OBJECT_B = "#1B9E9A"
GT_COLOR = "#F0E442"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def panel_paths(cohort: Path, manifest: dict) -> list[Path]:
    paths = [cohort / manifest["reference_panel"]]
    paths.extend(cohort / model["panel"] for model in manifest["models"])
    return paths


def compose_full(cohort: Path, output: Path) -> None:
    manifest = load_json(cohort / "manifest.json")
    paths = panel_paths(cohort, manifest)
    count = len(paths)
    columns = 4 if count >= 10 else (3 if count >= 6 else count)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.15 * columns, 3.48 * rows))
    axes = list(getattr(axes, "flat", [axes]))
    for ax, path in zip(axes, paths):
        ax.imshow(mpimg.imread(path))
        ax.set_axis_off()
    for ax in axes[len(paths):]:
        ax.set_axis_off()
    case = manifest["case_selection"]
    figure.suptitle(
        f"{manifest['dataset']} · {manifest['setting']}",
        color=TEXT, fontsize=17, fontweight="semibold", y=0.998,
    )
    figure.text(
        0.5, 0.975,
        f"Frozen test case {case['sample_id']} · GT {case['ground_truth_mm']:.2f} mm · seed 20260821",
        ha="center", va="top", fontsize=9.2, color=MUTED,
    )
    figure.subplots_adjust(left=0.012, right=0.988, top=0.942, bottom=0.012, wspace=0.018, hspace=0.025)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def find_model(manifest: dict, slug: str) -> str | None:
    for model in manifest["models"]:
        if model["slug"] == slug:
            return model["panel"]
    return None


def find_model_record(manifest: dict, slug: str) -> dict:
    for model in manifest["models"]:
        if model["slug"] == slug:
            return model
    raise RuntimeError(f"missing model record: {slug}")


def draw_clean_layer_panel(
    ax: plt.Axes,
    layer_path: Path,
    result_text: str | None,
    title: str | None = None,
    reference: bool = False,
) -> None:
    """Render an evidence-only panel from saved inference layers."""
    with np.load(layer_path, allow_pickle=False) as payload:
        image = payload["image"]
        gt_masks = payload["gt_masks"].astype(bool)
        z_index = int(payload["z_index"])
        pred_masks = payload["pred_masks"].astype(bool) if "pred_masks" in payload else None
        heat = payload["heat"] if "heat" in payload else None

    lo, hi = np.quantile(image, (0.01, 0.99))
    ax.imshow(image[z_index], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    if heat is not None and np.any(heat[z_index] > 0):
        overlay = np.ma.masked_less(heat[z_index], 0.06)
        ax.imshow(overlay, cmap="magma", vmin=0, vmax=1, alpha=np.clip(heat[z_index] * 0.82, 0, 0.82))
    for channel in range(2):
        if np.any(gt_masks[channel, z_index]):
            ax.contour(
                gt_masks[channel, z_index].astype(float), levels=(0.5,), colors=(GT_COLOR,),
                linewidths=0.9, linestyles="dashed",
            )
    if reference:
        pred_masks = gt_masks
    if pred_masks is not None:
        for channel, color in ((0, OBJECT_A), (1, OBJECT_B)):
            if np.any(pred_masks[channel, z_index]):
                ax.contour(
                    pred_masks[channel, z_index].astype(float), levels=(0.5,), colors=(color,), linewidths=1.2,
                )
    if title:
        ax.set_title(title, fontsize=11.2, color=TEXT, fontweight="semibold", pad=7)
    if result_text:
        ax.text(
            0.5, -0.035, result_text, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, color="#34495E",
        )
    ax.set_axis_off()


def compose_overview(root: Path, output: Path) -> None:
    cohort_specs = [
        ("tooth_ian", "ToothFairy3: tooth–IAN", "segresnet"),
        ("tooth_sinus", "ToothFairy3: tooth–sinus", "resenc_unet"),
        ("totalsegmentator", "TotalSegmentator: pancreas–aorta", "mednext"),
        ("amos22", "AMOS22: pancreas–aorta", "resenc_unet"),
        ("totalseg_to_amos", "TotalSeg→AMOS locked transfer", "mednext"),
    ]
    figure, axes = plt.subplots(len(cohort_specs), 4, figsize=(12.2, 16.6))
    case_rows = []
    for row, (cohort_name, row_label, segmentation_slug) in enumerate(cohort_specs):
        cohort = root / cohort_name
        manifest = load_json(cohort / "manifest.json")
        relative_paths = [
            manifest["reference_panel"],
            find_model(manifest, "direct"),
            find_model(manifest, "relmeasure3d"),
            find_model(manifest, segmentation_slug),
        ]
        for column, relative in enumerate(relative_paths):
            ax = axes[row, column]
            if relative is not None:
                ax.imshow(mpimg.imread(cohort / relative))
            ax.set_axis_off()
        axes[row, 0].text(
            -0.08, 0.5, row_label, transform=axes[row, 0].transAxes,
            rotation=90, ha="center", va="center", color=TEXT, fontsize=10.2, fontweight="semibold",
        )
        case = manifest["case_selection"]
        case_rows.append(
            {
                "label": row_label,
                "sample_id": case["sample_id"],
                "gt_mm": case["ground_truth_mm"],
                "selection_rule": case["rule"],
                "eligible_median_mm": case["eligible_median_mm"],
                "excluded_sample_ids": case.get("excluded_sample_ids", []),
            }
        )
    figure.suptitle(
        "Representative frozen-test cases across datasets and protocols",
        color=TEXT, fontsize=18, fontweight="semibold", y=0.998,
    )
    figure.text(
        0.5, 0.978,
        "Cases follow prespecified reference-distance rules independent of model errors; dashed yellow denotes GT contours.",
        ha="center", va="top", fontsize=9.5, color=MUTED,
    )
    figure.subplots_adjust(left=0.055, right=0.99, top=0.952, bottom=0.018, wspace=0.018, hspace=0.025)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)
    (output.parent / "overview_case_index.json").write_text(
        json.dumps(
            {
                "selection_rule": "Reference-distance-only rules recorded per row; independent of model prediction and error",
                "rows": case_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def compose_comparison(root: Path, output: Path) -> None:
    """One cross-dataset plate for the main comparison experiment."""
    cohort_specs = [
        ("tooth_ian", "ToothFairy3: tooth–IAN", "segresnet"),
        ("tooth_sinus", "ToothFairy3: tooth–sinus", "resenc_unet"),
        ("totalsegmentator", "TotalSegmentator: pancreas–aorta", "mednext"),
        ("amos22", "AMOS22: pancreas–aorta", "resenc_unet"),
        ("totalseg_to_amos", "TotalSeg→AMOS locked transfer", "mednext"),
    ]
    columns = [
        (None, "Reference"),
        ("direct", "Direct regression"),
        ("multitask", "Surface multitask"),
        ("analytic_softmin", "Analytic soft-min"),
        ("analytic_pool", "Analytic pooling"),
        ("relmeasure3d", "RelScope"),
        ("selected_segmentation", "Selected segmentation"),
    ]
    figure, axes = plt.subplots(len(cohort_specs), len(columns), figsize=(19.4, 15.5))
    index_rows = []
    for row, (cohort_name, row_label, segmentation_slug) in enumerate(cohort_specs):
        cohort = root / cohort_name
        manifest = load_json(cohort / "manifest.json")
        for column, (slug, column_label) in enumerate(columns):
            ax = axes[row, column]
            if slug is None:
                layer_path = cohort / "layers/reference.npz"
                result_text = None
                reference = True
            else:
                record = find_model_record(
                    manifest, segmentation_slug if slug == "selected_segmentation" else slug
                )
                layer_path = cohort / record["layers"]
                result_text = record["label"].replace(" candidate", "") if slug == "selected_segmentation" else None
                reference = False
            draw_clean_layer_panel(
                ax, layer_path, result_text,
                title=column_label if row == 0 else None,
                reference=reference,
            )
        axes[row, 0].text(
            -0.085, 0.5, row_label, transform=axes[row, 0].transAxes,
            rotation=90, ha="center", va="center", color=TEXT, fontsize=10.4, fontweight="semibold",
        )
        case = manifest["case_selection"]
        index_rows.append(
            {
                "cohort": cohort_name,
                "sample_id": case["sample_id"],
                "gt_mm": case["ground_truth_mm"],
                "selection_rule": case["rule"],
                "eligible_median_mm": case["eligible_median_mm"],
                "excluded_sample_ids": case.get("excluded_sample_ids", []),
            }
        )
    figure.suptitle(
        "Qualitative comparison across datasets and protocols",
        color=TEXT, fontsize=19, fontweight="semibold", y=0.995,
    )
    figure.subplots_adjust(left=0.04, right=0.995, top=0.958, bottom=0.015, wspace=0.025, hspace=0.06)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)
    (output.parent / "comparison_case_index.json").write_text(
        json.dumps(
            {
                "selection_rule": "Reference-distance-only rules recorded per row; independent of model prediction and error",
                "columns": [label for _, label in columns],
                "rows": index_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def compose_ablation(root: Path, output: Path) -> None:
    """One-dataset mechanism ablation plate using ToothFairy3 tooth–IAN."""
    cohort = root / "tooth_ian"
    manifest = load_json(cohort / "manifest.json")
    slugs = [None, "direct", "multitask", "analytic_softmin", "analytic_pool", "no_critical", "relmeasure3d"]
    records = []
    for slug in slugs:
        records.append(None if slug is None else find_model_record(manifest, slug))
    figure = plt.figure(figsize=(12.8, 7.4))
    grid = figure.add_gridspec(2, 8)
    axes = [figure.add_subplot(grid[0, start:start + 2]) for start in (0, 2, 4, 6)]
    axes.extend(figure.add_subplot(grid[1, start:start + 2]) for start in (1, 3, 5))
    for ax, record in zip(axes, records):
        if record is None:
            draw_clean_layer_panel(
                ax, cohort / "layers/reference.npz",
                f"GT {manifest['case_selection']['ground_truth_mm']:.2f} mm",
                title="Reference", reference=True,
            )
        else:
            result_text = (
                f"{record['prediction_mm']:.2f} mm | error {record['absolute_error_mm']:.2f} mm"
                if record["prediction_mm"] is not None else "failed"
            )
            display_title = "RelScope" if record["slug"] == "relmeasure3d" else record["label"]
            draw_clean_layer_panel(
                ax, cohort / record["layers"], result_text,
                title=display_title, reference=False,
            )
    figure.suptitle(
        "Mechanism ablation on ToothFairy3 tooth–IAN",
        color=TEXT, fontsize=17, fontweight="semibold", y=0.992,
    )
    figure.subplots_adjust(left=0.012, right=0.988, top=0.925, bottom=0.025, wspace=0.045, hspace=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white", bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for cohort in sorted(path.parent for path in root.glob("*/manifest.json")):
        compose_full(cohort, output / f"full_{cohort.name}")
    compose_overview(root, output / "overview_cross_dataset")
    compose_comparison(root, output / "figure_comparison_all_datasets")
    compose_ablation(root, output / "figure_ablation_tooth_ian")


if __name__ == "__main__":
    main()
