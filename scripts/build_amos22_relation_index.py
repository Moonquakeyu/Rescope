#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from relmeasure3d.data import save_json
from relmeasure3d.geometry import mask_mesh_world, mesh_distance_from_vertices


AORTA_LABEL = 8
PANCREAS_LABEL = 10


def process_case(image_text: str, label_text: str, official_split: str, delta_mm: float, min_voxels: int) -> dict[str, object]:
    image_path = Path(image_text)
    label_path = Path(label_text)
    patient_id = label_path.name.removesuffix(".nii.gz")
    result: dict[str, object] = {"patient_id": patient_id, "relation": None, "error": None}
    try:
        label_image = nib.load(str(label_path))
        labels = np.asanyarray(label_image.dataobj)
        pancreas = labels == PANCREAS_LABEL
        aorta = labels == AORTA_LABEL
        pancreas_voxels = int(pancreas.sum())
        aorta_voxels = int(aorta.sum())
        if pancreas_voxels < min_voxels or aorta_voxels < min_voxels:
            raise ValueError(f"insufficient foreground: pancreas={pancreas_voxels}, aorta={aorta_voxels}")
        distance = mesh_distance_from_vertices(
            mask_mesh_world(pancreas, label_image.affine),
            mask_mesh_world(aorta, label_image.affine),
            delta_mm=delta_mm,
        )
        result["relation"] = {
            "sample_id": f"{patient_id}__pancreas__aorta",
            "patient_id": patient_id,
            "image_path": str(image_path),
            "mask_a_path": str(label_path),
            "mask_b_path": str(label_path),
            "mask_a_label": PANCREAS_LABEL,
            "mask_b_label": AORTA_LABEL,
            "structure_a": "pancreas",
            "structure_b": "aorta",
            "relation_type": "pancreas_to_aorta_clearance",
            "distance_mm": distance.distance_mm,
            "closest_point_a_world": list(distance.point_a_world),
            "closest_point_b_world": list(distance.point_b_world),
            "critical_count_a": distance.critical_count_a,
            "critical_count_b": distance.critical_count_b,
            "structure_a_voxels": pancreas_voxels,
            "structure_b_voxels": aorta_voxels,
            "voxel_spacing": [float(value) for value in label_image.header.get_zooms()[:3]],
            "official_split": official_split,
            "modality": "CT",
        }
    except Exception as exc:
        result["error"] = {"patient_id": patient_id, "error": repr(exc)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--delta-mm", type=float, default=1.5)
    parser.add_argument("--min-voxels", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root)
    cases: list[tuple[Path, Path, str]] = []
    for image_dir, label_dir, split in (
        (root / "imagesTr", root / "labelsTr", "train"),
        (root / "imagesVa", root / "labelsVa", "test"),
    ):
        for label_path in sorted(label_dir.glob("amos_*.nii.gz")):
            if int(label_path.name[5:9]) >= 500:
                continue
            image_path = image_dir / label_path.name
            if image_path.exists():
                cases.append((image_path, label_path, split))

    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, max(1, len(cases)))) as pool:
        futures = [
            pool.submit(process_case, str(image), str(label), split, args.delta_mm, args.min_voxels)
            for image, label, split in cases
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({
                "patient_id": result["patient_id"],
                "valid": result["relation"] is not None,
                "error": result["error"],
            }), flush=True)

    results.sort(key=lambda row: str(row["patient_id"]))
    relations = [row["relation"] for row in results if row["relation"] is not None]
    errors = [row["error"] for row in results if row["error"] is not None]
    distances = np.asarray([float(row["distance_mm"]) for row in relations], dtype=float)
    output = {
        "dataset": "AMOS22 CT",
        "dataset_root": str(root),
        "protocol": "CT only (case id < 500); official training cohort for development and official validation cohort as held-out test",
        "patient_count": len(relations),
        "relation_count": len(relations),
        "split_counts": {
            split: sum(str(row["official_split"]) == split for row in relations)
            for split in ("train", "test")
        },
        "distance_summary_mm": {
            "min": float(distances.min()) if len(distances) else None,
            "median": float(np.median(distances)) if len(distances) else None,
            "p95": float(np.quantile(distances, 0.95)) if len(distances) else None,
            "max": float(distances.max()) if len(distances) else None,
            "zero_fraction": float(np.mean(distances == 0.0)) if len(distances) else None,
        },
        "errors": errors,
        "relations": relations,
    }
    save_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key not in ("relations", "errors")}, indent=2))
    if len(relations) < 250:
        raise SystemExit(f"too few valid AMOS22 CT relations: {len(relations)}")


if __name__ == "__main__":
    main()
