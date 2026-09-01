#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from relmeasure3d.data import save_json
from relmeasure3d.geometry import critical_band_voxels, signed_distance_mm


def _world_grid(center: np.ndarray, size: int, spacing: float) -> np.ndarray:
    axis = (np.arange(size, dtype=np.float32) - (size - 1) / 2.0) * spacing
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([gx + center[0], gy + center[1], gz + center[2]], axis=0)


def _sample_volume(data: np.ndarray, inv_affine: np.ndarray, world: np.ndarray, order: int) -> np.ndarray:
    flat = world.reshape(3, -1)
    homogeneous = np.concatenate([flat, np.ones((1, flat.shape[1]), dtype=np.float32)], axis=0)
    vox = (inv_affine @ homogeneous)[:3]
    sampled = map_coordinates(data, vox, order=order, mode="constant", cval=0.0, prefilter=order > 1)
    return sampled.reshape(world.shape[1:])


def process_patient(
    patient_id: str,
    relations: list[dict[str, object]],
    output_text: str,
    size: int,
    spacing: float,
    truncate_mm: float,
    delta_mm: float,
    center_mode: str,
) -> dict[str, object]:
    output = Path(output_text)
    written = []
    errors = []
    try:
        image_path = str(relations[0]["image_path"])
        image_img = nib.load(image_path)
        image = np.asanyarray(image_img.dataobj).astype(np.float32)
        inv_affine = np.linalg.inv(image_img.affine)
        label_img = None
        label = None
        if "label_path" in relations[0]:
            label_img = nib.load(str(relations[0]["label_path"]))
            label = np.asanyarray(label_img.dataobj)
    except Exception as exc:
        return {
            "patient_id": patient_id,
            "written": [],
            "errors": [
                {
                    "sample_id": str(relation["sample_id"]),
                    "image_path": str(relation.get("image_path", "")),
                    "error": repr(exc),
                }
                for relation in relations
            ],
        }
    for relation in relations:
        sample_id = str(relation["sample_id"])
        try:
            pa = np.asarray(relation["closest_point_a_world"], dtype=np.float32)
            pb = np.asarray(relation["closest_point_b_world"], dtype=np.float32)
            if "mask_a_path" in relation and "mask_b_path" in relation:
                mask_a_img = nib.load(str(relation["mask_a_path"]))
                mask_b_img = nib.load(str(relation["mask_b_path"]))
                raw_a = np.asanyarray(mask_a_img.dataobj)
                raw_b = np.asanyarray(mask_b_img.dataobj)
                mask_a_label = relation.get("mask_a_label")
                mask_b_label = relation.get("mask_b_label")
                mask_a_full = raw_a == int(mask_a_label) if mask_a_label is not None else raw_a > 0
                mask_b_full = raw_b == int(mask_b_label) if mask_b_label is not None else raw_b > 0
                if center_mode in ("tooth_centroid", "structure_a_centroid"):
                    structure_voxels = np.argwhere(mask_a_full)
                    if not len(structure_voxels):
                        raise ValueError("queried structure A is absent from the full mask")
                    center = nib.affines.apply_affine(mask_a_img.affine, structure_voxels.mean(axis=0)).astype(np.float32)
                else:
                    center = (pa + pb) / 2.0
            else:
                if label_img is None or label is None:
                    raise ValueError("multiclass label volume is unavailable")
                if center_mode in ("tooth_centroid", "structure_a_centroid"):
                    structure_voxels = np.argwhere(label == int(relation["structure_a"]))
                    if not len(structure_voxels):
                        raise ValueError("queried structure A is absent from the full label volume")
                    center = nib.affines.apply_affine(label_img.affine, structure_voxels.mean(axis=0)).astype(np.float32)
                else:
                    center = (pa + pb) / 2.0
            world = _world_grid(center, size=size, spacing=spacing)
            roi_image = _sample_volume(image, inv_affine, world, order=1).astype(np.float32)
            if "mask_a_path" in relation and "mask_b_path" in relation:
                mask_a = _sample_volume(
                    mask_a_full.astype(np.uint8), np.linalg.inv(mask_a_img.affine), world, order=0
                ) > 0.5
                mask_b = _sample_volume(
                    mask_b_full.astype(np.uint8), np.linalg.inv(mask_b_img.affine), world, order=0
                ) > 0.5
            else:
                roi_label = _sample_volume(label, np.linalg.inv(label_img.affine), world, order=0)
                mask_a = roi_label == int(relation["structure_a"])
                mask_b = roi_label == int(relation["structure_b"])
            if not mask_a.any() or not mask_b.any():
                raise ValueError("world ROI missed one queried structure")
            finite = roi_image[np.isfinite(roi_image)]
            lo, hi = np.percentile(finite, [0.5, 99.5])
            roi_image = np.clip(np.nan_to_num(roi_image), lo, hi)
            roi_image = (roi_image - roi_image.mean()) / (roi_image.std() + 1e-6)
            sdf_a = signed_distance_mm(mask_a, (spacing,) * 3, truncate_mm=truncate_mm)
            sdf_b = signed_distance_mm(mask_b, (spacing,) * 3, truncate_mm=truncate_mm)
            crit_a, crit_b, roi_distance = critical_band_voxels(mask_a, mask_b, (spacing,) * 3, delta_mm=delta_mm)
            final_path = output / f"{sample_id}.npz"
            tmp_path = output / f".{sample_id}.tmp.npz"
            np.savez_compressed(
                tmp_path,
                image=roi_image.astype(np.float16),
                sdf=np.stack([sdf_a, sdf_b]).astype(np.float16),
                mask=np.stack([mask_a, mask_b]).astype(np.uint8),
                critical=np.stack([crit_a, crit_b]).astype(np.uint8),
                distance_mm=np.float32(relation["distance_mm"]),
                roi_distance_mm=np.float32(roi_distance),
                sample_id=np.asarray(sample_id),
                center_world=((pa + pb) / 2.0).astype(np.float32),
                crop_center_world=center.astype(np.float32),
                center_mode=np.asarray(center_mode),
                spacing_mm=np.float32(spacing),
            )
            tmp_path.replace(final_path)
            written.append(sample_id)
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": repr(exc)})
    return {"patient_id": patient_id, "written": written, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory prepended to relative image and label paths in the released relation index.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--spacing-mm", type=float, default=1.0)
    parser.add_argument("--truncate-mm", type=float, default=10.0)
    parser.add_argument("--delta-mm", type=float, default=1.0)
    parser.add_argument(
        "--center-mode",
        choices=("closest_midpoint", "tooth_centroid", "structure_a_centroid"),
        default="closest_midpoint",
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.index).read_text())
    relations = payload["relations"][: args.max_samples]
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    for relation in relations:
        for key in ("image_path", "label_path", "mask_a_path", "mask_b_path"):
            if key in relation and not Path(str(relation[key])).is_absolute():
                relation[key] = str(dataset_root / str(relation[key]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for relation in relations:
        grouped[str(relation["patient_id"])].append(relation)
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(grouped))) as pool:
        futures = [
            pool.submit(
                process_patient,
                patient_id,
                group,
                str(output),
                args.size,
                args.spacing_mm,
                args.truncate_mm,
                args.delta_mm,
                args.center_mode,
            )
            for patient_id, group in grouped.items()
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {"patient_id": "worker_failure", "written": [], "errors": [{"error": repr(exc)}]}
            results.append(result)
            print(json.dumps({"patient_id": result["patient_id"], "written": len(result["written"]), "errors": result["errors"]}), flush=True)
    written = [sample for result in results for sample in result["written"]]
    errors = [error for result in results for error in result["errors"]]
    summary = {
        "index": args.index,
        "output": str(output),
        "requested": len(relations),
        "written": len(written),
        "errors": errors,
        "size": args.size,
        "spacing_mm": args.spacing_mm,
        "center_mode": args.center_mode,
    }
    save_json(output / "preprocess_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if len(written) < min(8, len(relations)):
        raise SystemExit("too few valid preprocessed samples")


if __name__ == "__main__":
    main()
