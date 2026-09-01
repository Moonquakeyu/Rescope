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


def process_case(label_text: str, images_dir_text: str, delta_mm: float, min_voxels: int) -> dict[str, object]:
    label_path = Path(label_text)
    patient_id = label_path.name.removesuffix(".nii.gz")
    image_path = Path(images_dir_text) / label_path.name
    result: dict[str, object] = {"patient_id": patient_id, "relations": [], "errors": []}
    try:
        label_img = nib.load(str(label_path))
        label = np.asanyarray(label_img.dataobj)
        vessel = label == 1
        tumour = label == 2
        if int(vessel.sum()) < min_voxels or int(tumour.sum()) < min_voxels:
            return result
        vertices_tumour = mask_mesh_world(tumour, label_img.affine)
        vertices_vessel = mask_mesh_world(vessel, label_img.affine)
        distance = mesh_distance_from_vertices(vertices_tumour, vertices_vessel, delta_mm=delta_mm)
        result["relations"].append({
            "sample_id": f"{patient_id}__tumour__hepatic_vessel",
            "patient_id": patient_id,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "structure_a": 2,
            "structure_b": 1,
            "structure_a_name": "tumour",
            "structure_b_name": "hepatic_vessel",
            "relation_type": "tumour_to_hepatic_vessel_clearance",
            "distance_mm": distance.distance_mm,
            "closest_point_a_world": list(distance.point_a_world),
            "closest_point_b_world": list(distance.point_b_world),
            "critical_count_a": distance.critical_count_a,
            "critical_count_b": distance.critical_count_b,
            "structure_a_voxels": int(tumour.sum()),
            "structure_b_voxels": int(vessel.sum()),
            "voxel_spacing": [float(value) for value in label_img.header.get_zooms()[:3]],
        })
    except Exception as exc:
        result["errors"].append({"patient_id": patient_id, "path": str(label_path), "error": repr(exc)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-patients", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--delta-mm", type=float, default=1.5)
    parser.add_argument("--min-voxels", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root)
    labels_dir = root / "labelsTr"
    images_dir = root / "imagesTr"
    labels = sorted(labels_dir.glob("*.nii.gz"))[: args.max_patients]
    case_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(process_case, str(path), str(images_dir), args.delta_mm, args.min_voxels)
            for path in labels
        ]
        for future in as_completed(futures):
            result = future.result()
            case_results.append(result)
            print(json.dumps({
                "patient_id": result["patient_id"],
                "relations": len(result["relations"]),
                "errors": len(result["errors"]),
            }), flush=True)

    relations = [relation for case in case_results for relation in case["relations"]]
    errors = [error for case in case_results for error in case["errors"]]
    distances = np.asarray([float(row["distance_mm"]) for row in relations], dtype=float)
    output = {
        "dataset": "Medical Segmentation Decathlon Task08 HepaticVessel",
        "dataset_root": str(root),
        "patient_count": len(relations),
        "relation_count": len(relations),
        "relation_counts": {"tumour_to_hepatic_vessel_clearance": len(relations)},
        "distance_summary_mm": {
            "min": float(distances.min()) if len(distances) else None,
            "median": float(np.median(distances)) if len(distances) else None,
            "p95": float(np.quantile(distances, 0.95)) if len(distances) else None,
            "max": float(distances.max()) if len(distances) else None,
            "zero_fraction": float(np.mean(distances <= 1e-6)) if len(distances) else None,
        },
        "errors": errors,
        "relations": relations,
    }
    save_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items() if key not in ("relations", "errors")}, indent=2))
    if len(relations) < 30:
        raise SystemExit("too few valid tumour-vessel relations")


if __name__ == "__main__":
    main()
