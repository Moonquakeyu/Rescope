#!/usr/bin/env python3
"""Export one deterministic test case and every available model output.

The remote GPU is used only for inference. Each model is exported as a
standalone PNG plus an NPZ layer bundle and a JSON provenance record so that
the final multi-panel figure can be assembled and revised locally.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter

from relmeasure3d.data import NPZRelationDataset
from relmeasure3d.external_baselines import (
    MonaiMedNeXtBaseline,
    MonaiSegResNetBaseline,
    NNUNetResEncBaseline,
)
from relmeasure3d.publication_model import (
    PublicationAnalyticCriticalPooling,
    PublicationAnalyticSoftmin,
    PublicationDirectRegression,
    PublicationRelMeasure3D,
    PublicationSegmentationBaseline,
    PublicationSurfaceGlobalRegression,
    _analytic_surface_attention,
)


OBJECT_A = "#E76F51"
OBJECT_B = "#1B9E9A"
GT_COLOR = "#F0E442"
TEXT = "#16324F"


MODEL_FACTORIES = {
    "publication_direct": lambda base: PublicationDirectRegression(base=base),
    "publication_multitask": lambda base: PublicationSurfaceGlobalRegression(base=base),
    "publication_analytic_softmin": lambda base: PublicationAnalyticSoftmin(base=base),
    "publication_analytic_pool": lambda base: PublicationAnalyticCriticalPooling(base=base),
    "publication_relmeasure": lambda base: PublicationRelMeasure3D(base=base),
    "publication_segmentation": lambda base: PublicationSegmentationBaseline(base=base),
    "monai_segresnet": lambda base: MonaiSegResNetBaseline(base=base),
    "monai_mednext": lambda base: MonaiMedNeXtBaseline(base=base),
    "nnunet_resenc": lambda base: NNUNetResEncBaseline(base=base),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model(spec: dict, device: torch.device) -> torch.nn.Module:
    model = MODEL_FACTORIES[spec["type"]](int(spec.get("base", 16)))
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return model.to(device).eval()


def surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return mask & ~binary_erosion(mask, border_value=0)


def predicted_critical_band(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, float]:
    surface_a, surface_b = surface(mask_a), surface(mask_b)
    if not surface_a.any() or not surface_b.any():
        return np.zeros_like(mask_a, dtype=np.float32), float("nan")
    distance_to_b = distance_transform_edt(~surface_b)
    distance_to_a = distance_transform_edt(~surface_a)
    distance = float(min(distance_to_b[surface_a].min(), distance_to_a[surface_b].min()))
    band = (surface_a & (distance_to_b <= distance + 0.5)) | (surface_b & (distance_to_a <= distance + 0.5))
    return band.astype(np.float32), distance


def normalize_heat(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    positive = volume[volume > 0]
    high = float(np.quantile(positive, 0.995)) if positive.size else 1.0
    return np.clip(volume / max(high, 1e-12), 0.0, 1.0)


def attention_heat(output: object) -> np.ndarray:
    attention_a = output.attention_a[0, 0].float().cpu().numpy()
    attention_b = output.attention_b[0, 0].float().cpu().numpy()
    return normalize_heat(attention_a + attention_b)


def select_case(
    cache: Path,
    split: dict,
    positive_only: bool,
    selection: str = "closest_to_median",
    exclude_sample_ids: set[str] | None = None,
) -> tuple[str, float, float, str]:
    excluded = exclude_sample_ids or set()
    rows: list[tuple[str, float]] = []
    for sample_id in split["samples"]["test"]:
        path = cache / f"{sample_id}.npz"
        with np.load(path, allow_pickle=False) as payload:
            distance = float(payload["distance_mm"])
        if not positive_only or distance > 1e-6:
            rows.append((sample_id, distance))
    if not rows:
        raise RuntimeError("no eligible test cases for deterministic median selection")
    median = float(np.median([distance for _, distance in rows]))
    selectable_rows = [row for row in rows if row[0] not in excluded]
    if not selectable_rows:
        raise RuntimeError("all eligible test cases were excluded from deterministic selection")
    if selection == "closest_to_median":
        candidates = selectable_rows
        rule = "closest ground-truth distance to eligible test-set median"
    elif selection == "nearest_below_median":
        candidates = [row for row in selectable_rows if row[1] < median]
        if not candidates:
            raise RuntimeError("no eligible test case below the reference-distance median")
        rule = "nearest ground-truth distance below eligible test-set median"
    elif selection == "nearest_above_median":
        candidates = [row for row in selectable_rows if row[1] > median]
        if not candidates:
            raise RuntimeError("no eligible test case above the reference-distance median")
        rule = "nearest ground-truth distance above eligible test-set median"
    else:
        raise ValueError(f"unknown case selection rule: {selection}")
    sample_id, distance = min(candidates, key=lambda row: (abs(row[1] - median), row[0]))
    return sample_id, distance, median, rule


def choose_slice(critical: np.ndarray, masks: np.ndarray) -> int:
    critical_score = critical.max(axis=0).sum(axis=(1, 2))
    if critical_score.max() > 0:
        return int(np.argmax(critical_score))
    mask_score = masks.max(axis=0).sum(axis=(1, 2))
    return int(np.argmax(mask_score))


def render_panel(
    output: Path,
    image: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray | None,
    heat: np.ndarray | None,
    z_index: int,
    title: str,
    subtitle: str,
    note: str,
) -> None:
    figure, ax = plt.subplots(figsize=(3.2, 3.55))
    lo, hi = np.quantile(image, (0.01, 0.99))
    ax.imshow(image[z_index], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    if heat is not None and np.any(heat[z_index] > 0):
        overlay = np.ma.masked_less(heat[z_index], 0.06)
        ax.imshow(overlay, cmap="magma", vmin=0, vmax=1, alpha=np.clip(heat[z_index] * 0.82, 0, 0.82))
    for channel in range(2):
        if np.any(gt_masks[channel, z_index]):
            ax.contour(
                gt_masks[channel, z_index].astype(float),
                levels=(0.5,), colors=(GT_COLOR,), linewidths=0.9, linestyles="dashed",
            )
    if pred_masks is not None:
        for channel, color in ((0, OBJECT_A), (1, OBJECT_B)):
            if np.any(pred_masks[channel, z_index]):
                ax.contour(pred_masks[channel, z_index].astype(float), levels=(0.5,), colors=(color,), linewidths=1.2)
    ax.set_title(title, fontsize=10.5, fontweight="semibold", color=TEXT, pad=6)
    ax.text(0.5, -0.035, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8.1, color="#34495E")
    ax.text(0.5, -0.105, note, transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#6B7280")
    ax.set_axis_off()
    figure.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.15)
    figure.savefig(output, dpi=360, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def clean_float(value: float) -> float | None:
    return None if not math.isfinite(value) else float(value)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    output_root = Path(args.output)
    panels_dir = output_root / "panels"
    layers_dir = output_root / "layers"
    panels_dir.mkdir(parents=True, exist_ok=True)
    layers_dir.mkdir(parents=True, exist_ok=True)

    cache = Path(config["cache"])
    split = load_json(config["split"])
    sample_id, selected_gt, eligible_median, selection_rule = select_case(
        cache,
        split,
        bool(config.get("positive_only", False)),
        str(config.get("case_selection", "closest_to_median")),
        set(config.get("exclude_sample_ids", [])),
    )
    dataset = NPZRelationDataset(cache, crop_size=int(config["crop_size"]))
    lookup = {path.stem: index for index, path in enumerate(dataset.files)}
    sample = dataset[lookup[sample_id]]
    image_np = sample["image"][0].numpy().astype(np.float32)
    gt_masks = sample["mask"].numpy().astype(bool)
    critical = sample["critical"].numpy().astype(bool)
    gt_mm = float(sample["distance_mm"].item())
    z_index = choose_slice(critical, gt_masks)

    reference_heat = normalize_heat(gaussian_filter(critical.max(axis=0).astype(np.float32), 1.25))
    np.savez_compressed(
        layers_dir / "reference.npz", image=image_np, gt_masks=gt_masks.astype(np.uint8),
        critical=critical.astype(np.uint8), heat=reference_heat, z_index=z_index, distance_mm=gt_mm,
    )
    render_panel(
        panels_dir / "00_reference.png", image_np, gt_masks, gt_masks, reference_heat, z_index,
        "Reference", f"Ground truth: {gt_mm:.2f} mm", "Dashed yellow: GT; heat: active boundary",
    )

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    image = sample["image"].unsqueeze(0).to(device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    records: list[dict] = []
    with torch.inference_mode(), context:
        for index, spec in enumerate(config["models"], start=1):
            model = load_model(spec, device)
            output = model(image)
            model_type = spec["type"]
            pred_masks: np.ndarray | None = None
            heat: np.ndarray | None = None
            sigma = float(output.sigma_mm.item()) if hasattr(output, "sigma_mm") else float("nan")
            map_kind = "none"

            if model_type == "publication_direct":
                pred_mm = float(output.distance_mm.item())
                note = "Scalar-only model; no spatial map"
            elif model_type in {"publication_segmentation", "monai_segresnet", "monai_mednext", "nnunet_resenc"}:
                probability = torch.sigmoid(output.segmentation_logits)[0].float().cpu().numpy()
                pred_masks = probability >= float(spec.get("threshold", 0.5))
                band, pred_mm = predicted_critical_band(pred_masks[0], pred_masks[1])
                heat = normalize_heat(gaussian_filter(band, 1.25))
                map_kind = "critical band derived from predicted masks"
                note = f"Predicted masks; threshold {float(spec.get('threshold', 0.5)):.1f}"
            else:
                pred_mm = float(output.distance_mm.item())
                pred_masks = output.sdf[0].float().cpu().numpy() <= 0
                if model_type == "publication_multitask":
                    note = "Predicted surfaces; no critical search"
                    map_kind = "none"
                elif model_type == "publication_analytic_softmin":
                    _, _, attention_a, attention_b = _analytic_surface_attention(
                        output.sdf, model.surface_temperature_mm, model.distance_temperature_mm
                    )
                    heat = normalize_heat(
                        attention_a[0, 0].float().cpu().numpy() + attention_b[0, 0].float().cpu().numpy()
                    )
                    note = "Analytic soft-min relevance"
                    map_kind = "analytic attention"
                else:
                    heat = attention_heat(output)
                    note = "Model spatial relevance"
                    map_kind = "predicted attention"

            error = abs(pred_mm - gt_mm) if math.isfinite(pred_mm) else float("nan")
            slug = spec["slug"]
            layer_payload = {
                "image": image_np,
                "gt_masks": gt_masks.astype(np.uint8),
                "z_index": np.asarray(z_index),
                "pred_mm": np.asarray(pred_mm),
                "gt_mm": np.asarray(gt_mm),
            }
            if pred_masks is not None:
                layer_payload["pred_masks"] = pred_masks.astype(np.uint8)
            if heat is not None:
                layer_payload["heat"] = heat.astype(np.float32)
            np.savez_compressed(layers_dir / f"{index:02d}_{slug}.npz", **layer_payload)
            subtitle = (
                f"Prediction failed | GT {gt_mm:.2f} mm"
                if not math.isfinite(pred_mm)
                else f"Pred. {pred_mm:.2f} mm | error {error:.2f} mm"
            )
            render_panel(
                panels_dir / f"{index:02d}_{slug}.png", image_np, gt_masks, pred_masks, heat,
                z_index, spec["label"], subtitle, note,
            )
            records.append(
                {
                    "slug": slug,
                    "label": spec["label"],
                    "type": model_type,
                    "checkpoint": spec["checkpoint"],
                    "threshold": spec.get("threshold"),
                    "prediction_mm": clean_float(pred_mm),
                    "absolute_error_mm": clean_float(error),
                    "sigma_mm": clean_float(sigma),
                    "spatial_map": map_kind,
                    "panel": f"panels/{index:02d}_{slug}.png",
                    "layers": f"layers/{index:02d}_{slug}.npz",
                }
            )
            del model, output
            if device.type == "cuda":
                torch.cuda.empty_cache()

    manifest = {
        "dataset": config["dataset"],
        "setting": config["setting"],
        "case_selection": {
            "split": "test",
            "rule": selection_rule,
            "positive_only": bool(config.get("positive_only", False)),
            "excluded_sample_ids": list(config.get("exclude_sample_ids", [])),
            "eligible_median_mm": eligible_median,
            "sample_id": sample_id,
            "ground_truth_mm": selected_gt,
        },
        "crop_size": int(config["crop_size"]),
        "slice_index": z_index,
        "reference_panel": "panels/00_reference.png",
        "models": records,
        "legend": {
            "dashed_yellow": "ground-truth contours",
            "coral": "predicted structure A",
            "teal": "predicted structure B",
            "magma": "normalized measurement-critical relevance",
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
