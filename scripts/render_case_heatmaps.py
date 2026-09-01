#!/usr/bin/env python3
"""Render a real single-case localization comparison from trained checkpoints.

This script does not synthesize attention. RelMeasure3D-family panels show the
model's saved spatial distributions. Segmentation panels show the
measurement-critical band derived from each model's predicted masks. All
panels use the same case, slice, intensity window, masks, and color scale.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

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
    PublicationRelMeasure3D,
)


NAVY = "#16324F"
TEAL = "#1B9E9A"
CORAL = "#E76F51"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Directory of cached relation NPZ files")
    parser.add_argument("--sample-id", required=True, help="NPZ stem from the frozen test split")
    parser.add_argument("--output", required=True, help="Output PDF or PNG path")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))

    parser.add_argument("--rel-checkpoint", required=True)
    parser.add_argument("--no-critical-checkpoint", required=True)
    parser.add_argument("--analytic-pool-checkpoint", required=True)
    parser.add_argument("--segresnet-checkpoint", required=True)
    parser.add_argument("--mednext-checkpoint", required=True)
    parser.add_argument("--resenc-checkpoint", required=True)

    parser.add_argument("--rel-base", type=int, default=24)
    parser.add_argument("--segresnet-base", type=int, default=24)
    parser.add_argument("--mednext-base", type=int, default=24)
    parser.add_argument("--resenc-base", type=int, default=32)
    parser.add_argument("--segresnet-threshold", type=float, default=0.5)
    parser.add_argument("--mednext-threshold", type=float, default=0.5)
    parser.add_argument("--resenc-threshold", type=float, default=0.5)
    return parser.parse_args()


def device_from_arg(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_weights(model: torch.nn.Module, path: str, device: torch.device) -> torch.nn.Module:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return model.to(device).eval()


def normalize_heat(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    high = float(np.quantile(volume[volume > 0], 0.995)) if np.any(volume > 0) else 1.0
    return np.clip(volume / max(high, 1e-12), 0.0, 1.0)


def reference_heat(critical: np.ndarray) -> np.ndarray:
    return normalize_heat(gaussian_filter(np.asarray(critical).max(axis=0).astype(np.float32), 1.25))


def attention_heat(output: object) -> np.ndarray:
    attention_a = getattr(output, "attention_a")[0, 0].float().cpu().numpy()
    attention_b = getattr(output, "attention_b")[0, 0].float().cpu().numpy()
    return normalize_heat(attention_a + attention_b)


def predicted_critical_band(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    def surface(mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        return mask & ~binary_erosion(mask, border_value=0)

    surface_a, surface_b = surface(mask_a), surface(mask_b)
    if not surface_a.any() or not surface_b.any():
        raise ValueError("predicted mask is empty; no spatial measurement can be derived")
    distance_to_b = distance_transform_edt(~surface_b)
    distance_to_a = distance_transform_edt(~surface_a)
    distance_mm = float(min(distance_to_b[surface_a].min(), distance_to_a[surface_b].min()))
    critical_a = surface_a & (distance_to_b <= distance_mm + 0.5)
    critical_b = surface_b & (distance_to_a <= distance_mm + 0.5)
    return critical_a.astype(np.float32), critical_b.astype(np.float32), distance_mm


def segmentation_case(
    model: torch.nn.Module,
    image: torch.Tensor,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    logits = model(image).segmentation_logits
    probability = torch.sigmoid(logits)[0].float().cpu().numpy()
    masks = probability >= threshold
    critical_a, critical_b, distance_mm = predicted_critical_band(masks[0], masks[1])
    heat = normalize_heat(gaussian_filter(np.maximum(critical_a, critical_b), 1.25))
    return masks, heat, float(distance_mm)


def panel(
    ax: plt.Axes,
    image: np.ndarray,
    masks: np.ndarray,
    heat: np.ndarray,
    z_index: int,
    title: str,
    subtitle: str,
) -> None:
    lo, hi = np.quantile(image, (0.01, 0.99))
    ax.imshow(image[z_index], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    overlay = np.ma.masked_less(heat[z_index], 0.08)
    ax.imshow(overlay, cmap="magma", vmin=0, vmax=1, alpha=np.clip(heat[z_index] * 0.82, 0, 0.82))
    for channel, color in ((0, CORAL), (1, TEAL)):
        if np.any(masks[channel, z_index]):
            ax.contour(masks[channel, z_index].astype(float), levels=(0.5,), colors=(color,), linewidths=1.15)
    ax.set_title(title, color=NAVY, fontsize=9.5, fontweight="semibold", pad=5)
    ax.text(0.5, -0.055, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#455A64")
    ax.set_axis_off()


def main() -> None:
    args = parse_args()
    dataset = NPZRelationDataset(args.data, crop_size=args.crop_size)
    index_by_id = {path.stem: i for i, path in enumerate(dataset.files)}
    if args.sample_id not in index_by_id:
        raise KeyError(f"sample {args.sample_id!r} not found under {args.data}")
    sample = dataset[index_by_id[args.sample_id]]
    image_np = np.asarray(sample["image"][0].numpy(), dtype=np.float32)
    target_masks = np.asarray(sample["mask"].numpy(), dtype=bool)
    critical = np.asarray(sample["critical"].numpy(), dtype=bool)
    gt_mm = float(sample["distance_mm"].item())
    z_index = int(np.argmax(critical.max(axis=0).sum(axis=(1, 2))))

    device = device_from_arg(args.device)
    image = sample["image"].unsqueeze(0).to(device)
    models = {
        "SegResNet": load_weights(MonaiSegResNetBaseline(base=args.segresnet_base), args.segresnet_checkpoint, device),
        "MedNeXt": load_weights(MonaiMedNeXtBaseline(base=args.mednext_base), args.mednext_checkpoint, device),
        "Residual-encoder U-Net": load_weights(NNUNetResEncBaseline(base=args.resenc_base), args.resenc_checkpoint, device),
        "Analytic pooling": load_weights(PublicationAnalyticCriticalPooling(base=args.rel_base), args.analytic_pool_checkpoint, device),
        "Without critical supervision": load_weights(PublicationRelMeasure3D(base=args.rel_base), args.no_critical_checkpoint, device),
        "RelScope": load_weights(PublicationRelMeasure3D(base=args.rel_base), args.rel_checkpoint, device),
    }
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    panels: list[tuple[str, np.ndarray, np.ndarray, float]] = [
        ("Reference active band", target_masks, reference_heat(critical), gt_mm)
    ]
    records: list[dict[str, object]] = []
    with torch.no_grad(), context:
        for name, threshold in (
            ("SegResNet", args.segresnet_threshold),
            ("MedNeXt", args.mednext_threshold),
            ("Residual-encoder U-Net", args.resenc_threshold),
        ):
            masks, heat, pred_mm = segmentation_case(models[name], image, threshold)
            panels.append((name, masks, heat, pred_mm))
            records.append({"model": name, "map": "critical band derived from predicted masks", "pred_mm": pred_mm})
        for name in ("Analytic pooling", "Without critical supervision", "RelScope"):
            output = models[name](image)
            pred_mm = float(output.distance_mm.item())
            predicted_masks = output.sdf[0].float().cpu().numpy() <= 0
            panels.append((name, predicted_masks, attention_heat(output), pred_mm))
            records.append({"model": name, "map": "predicted spatial distribution", "pred_mm": pred_mm})

    figure, axes = plt.subplots(2, 4, figsize=(11.2, 5.9), constrained_layout=False)
    axes_flat = axes.ravel()
    for ax, (name, masks, heat, pred_mm) in zip(axes_flat, panels):
        subtitle = f"GT {gt_mm:.2f} mm" if name.startswith("Reference") else f"Pred. {pred_mm:.2f} mm  |  error {abs(pred_mm - gt_mm):.2f} mm"
        panel(ax, image_np, masks, heat, z_index, name, subtitle)
    axes_flat[-1].set_axis_off()
    scalar = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, 1))
    cbar = figure.colorbar(scalar, ax=axes_flat.tolist(), orientation="horizontal", fraction=0.028, pad=0.075, aspect=45)
    cbar.set_label("Normalized measurement-critical relevance", fontsize=8.5, color=NAVY)
    cbar.ax.tick_params(labelsize=7.5)
    figure.suptitle(
        f"One frozen test relation across model families  |  {args.sample_id}  |  slice {z_index}",
        fontsize=12,
        fontweight="semibold",
        color=NAVY,
        y=0.985,
    )
    figure.subplots_adjust(left=0.025, right=0.985, top=0.91, bottom=0.12, wspace=0.045, hspace=0.16)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    provenance = {
        "sample_id": args.sample_id,
        "slice": z_index,
        "gt_mm": gt_mm,
        "checkpoints": {key: value for key, value in vars(args).items() if key.endswith("checkpoint")},
        "panels": records,
        "note": "No scalar-only model is shown as a heatmap because it has no spatial output.",
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
