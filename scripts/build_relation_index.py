#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from relmeasure3d.data import (
    LEFT_IAN,
    LEFT_MAXILLARY_SINUS,
    LEFT_TEETH,
    LEFT_UPPER_POSTERIOR_TEETH,
    RIGHT_IAN,
    RIGHT_MAXILLARY_SINUS,
    RIGHT_TEETH,
    RIGHT_UPPER_POSTERIOR_TEETH,
    image_path_from_label,
    save_json,
)
from relmeasure3d.geometry import mask_mesh_world, mesh_distance_from_vertices


def process_case(
    label_text: str,
    images_text: str,
    delta_mm: float,
    min_voxels: int,
    relation_set: str,
) -> dict[str, object]:
    label_path = Path(label_text)
    images_dir = Path(images_text)
    patient_id = label_path.name[:-7]
    image_path = image_path_from_label(label_path, images_dir)
    img = nib.load(str(label_path))
    label = np.asanyarray(img.dataobj)
    result: dict[str, object] = {"patient_id": patient_id, "relations": [], "errors": []}
    targets = []
    if relation_set in ("ian", "all"):
        targets.extend([
            (LEFT_IAN, LEFT_TEETH, "left", "tooth_ian_clearance", "ian"),
            (RIGHT_IAN, RIGHT_TEETH, "right", "tooth_ian_clearance", "ian"),
        ])
    if relation_set in ("sinus", "all"):
        targets.extend([
            (LEFT_MAXILLARY_SINUS, LEFT_UPPER_POSTERIOR_TEETH, "left", "tooth_sinus_clearance", "sinus"),
            (RIGHT_MAXILLARY_SINUS, RIGHT_UPPER_POSTERIOR_TEETH, "right", "tooth_sinus_clearance", "sinus"),
        ])
    for target_label, teeth, side, relation_type, target_name in targets:
        target = label == target_label
        if int(target.sum()) < min_voxels:
            continue
        try:
            target_vertices = mask_mesh_world(target, img.affine)
        except Exception as exc:
            result["errors"].append({"structure": f"{side}_{target_name}", "error": repr(exc)})  # type: ignore[index]
            continue
        for tooth_label in teeth:
            tooth = label == tooth_label
            voxels = int(tooth.sum())
            if voxels < min_voxels:
                continue
            try:
                tooth_vertices = mask_mesh_world(tooth, img.affine)
                dist = mesh_distance_from_vertices(tooth_vertices, target_vertices, delta_mm=delta_mm)
                relation = {
                    "sample_id": f"{patient_id}__tooth_{tooth_label}__{side}_{target_name}",
                    "patient_id": patient_id,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "structure_a": int(tooth_label),
                    "structure_b": int(target_label),
                    "relation_type": relation_type,
                    "side": side,
                    "distance_mm": dist.distance_mm,
                    "closest_point_a_world": list(dist.point_a_world),
                    "closest_point_b_world": list(dist.point_b_world),
                    "critical_count_a": dist.critical_count_a,
                    "critical_count_b": dist.critical_count_b,
                    "tooth_voxels": voxels,
                    "target_voxels": int(target.sum()),
                    "affine": img.affine.tolist(),
                    "voxel_spacing": [float(x) for x in img.header.get_zooms()[:3]],
                }
                result["relations"].append(relation)  # type: ignore[index]
            except Exception as exc:
                result["errors"].append({"structure": tooth_label, "error": repr(exc)})  # type: ignore[index]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-patients", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delta-mm", type=float, default=0.5)
    parser.add_argument("--min-voxels", type=int, default=32)
    parser.add_argument("--relation-set", choices=("ian", "sinus", "all"), default="ian")
    args = parser.parse_args()
    root = Path(args.root)
    labels = sorted((root / "labelsTr").glob("*.nii.gz"))[: args.max_patients]
    case_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                process_case,
                str(p),
                str(root / "imagesTr"),
                args.delta_mm,
                args.min_voxels,
                args.relation_set,
            )
            for p in labels
        ]
        for future in as_completed(futures):
            case = future.result()
            case_results.append(case)
            print(json.dumps({"patient_id": case["patient_id"], "relations": len(case["relations"]), "errors": len(case["errors"])}, ensure_ascii=False), flush=True)
    case_results.sort(key=lambda x: str(x["patient_id"]))
    relations = [r for case in case_results for r in case["relations"]]  # type: ignore[index]
    errors = [e for case in case_results for e in case["errors"]]  # type: ignore[index]
    distances = np.asarray([float(r["distance_mm"]) for r in relations], dtype=float)
    output = {
        "dataset_root": str(root),
        "patient_count": len(case_results),
        "relation_count": len(relations),
        "distance_summary_mm": {
            "min": float(distances.min()) if len(distances) else None,
            "median": float(np.median(distances)) if len(distances) else None,
            "p95": float(np.quantile(distances, 0.95)) if len(distances) else None,
            "max": float(distances.max()) if len(distances) else None,
        },
        "errors": errors,
        "relations": relations,
    }
    save_json(args.output, output)
    print(json.dumps({k: v for k, v in output.items() if k != "relations"}, ensure_ascii=False, indent=2))
    if not relations:
        raise SystemExit(f"no valid relations found for relation_set={args.relation_set}")


if __name__ == "__main__":
    main()
