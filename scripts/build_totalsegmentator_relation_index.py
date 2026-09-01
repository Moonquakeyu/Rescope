#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

from relmeasure3d.data import save_json
from relmeasure3d.geometry import mask_mesh_world, mesh_distance_from_vertices


def process_case(case_text: str, pairs: list[tuple[str, str]], delta_mm: float, min_voxels: int) -> dict[str, object]:
    case = Path(case_text)
    patient_id = case.name
    result: dict[str, object] = {"patient_id": patient_id, "relations": [], "errors": []}
    for structure_a, structure_b in pairs:
        path_a = case / "segmentations" / f"{structure_a}.nii.gz"
        path_b = case / "segmentations" / f"{structure_b}.nii.gz"
        if not path_a.exists() or not path_b.exists():
            continue
        try:
            image_a = nib.load(str(path_a))
            image_b = nib.load(str(path_b))
            mask_a = np.asanyarray(image_a.dataobj) > 0
            mask_b = np.asanyarray(image_b.dataobj) > 0
            voxels_a = int(mask_a.sum())
            voxels_b = int(mask_b.sum())
            if voxels_a < min_voxels or voxels_b < min_voxels:
                continue
            vertices_a = mask_mesh_world(mask_a, image_a.affine)
            vertices_b = mask_mesh_world(mask_b, image_b.affine)
            distance = mesh_distance_from_vertices(vertices_a, vertices_b, delta_mm=delta_mm)
            relation_type = f"{structure_a}_to_{structure_b}_clearance"
            result["relations"].append({
                "sample_id": f"{patient_id}__{structure_a}__{structure_b}",
                "patient_id": patient_id,
                "image_path": str(case / "ct.nii.gz"),
                "mask_a_path": str(path_a),
                "mask_b_path": str(path_b),
                "structure_a": structure_a,
                "structure_b": structure_b,
                "relation_type": relation_type,
                "distance_mm": distance.distance_mm,
                "closest_point_a_world": list(distance.point_a_world),
                "closest_point_b_world": list(distance.point_b_world),
                "critical_count_a": distance.critical_count_a,
                "critical_count_b": distance.critical_count_b,
                "structure_a_voxels": voxels_a,
                "structure_b_voxels": voxels_b,
                "voxel_spacing": [float(value) for value in image_a.header.get_zooms()[:3]],
            })
        except Exception as exc:
            result["errors"].append({"pair": [structure_a, structure_b], "error": repr(exc)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--meta-csv", default=None)
    parser.add_argument("--pairs", nargs="+", default=("pancreas:aorta",))
    parser.add_argument("--max-patients", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--delta-mm", type=float, default=1.5)
    parser.add_argument("--min-voxels", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root)
    pairs = [tuple(value.split(":", 1)) for value in args.pairs]
    cases = sorted(path for path in root.glob("s[0-9][0-9][0-9][0-9]") if (path / "ct.nii.gz").exists())[: args.max_patients]
    official_split: dict[str, str] = {}
    if args.meta_csv:
        with Path(args.meta_csv).open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle, delimiter=";"):
                official_split[str(row["image_id"])] = str(row["split"])

    case_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_case, str(case), pairs, args.delta_mm, args.min_voxels) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            for relation in result["relations"]:
                relation["official_split"] = official_split.get(str(result["patient_id"]), "unknown")
            case_results.append(result)
            print(json.dumps({"patient_id": result["patient_id"], "relations": len(result["relations"]), "errors": len(result["errors"])}), flush=True)

    case_results.sort(key=lambda row: str(row["patient_id"]))
    relations = [relation for case in case_results for relation in case["relations"]]
    errors = [error for case in case_results for error in case["errors"]]
    distances = np.asarray([float(relation["distance_mm"]) for relation in relations], dtype=float)
    counts_by_pair: dict[str, int] = {}
    for relation in relations:
        key = str(relation["relation_type"])
        counts_by_pair[key] = counts_by_pair.get(key, 0) + 1
    output = {
        "dataset": "TotalSegmentator v1",
        "dataset_root": str(root),
        "patient_count": len({str(relation["patient_id"]) for relation in relations}),
        "relation_count": len(relations),
        "relation_counts": counts_by_pair,
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
    print(json.dumps({key: value for key, value in output.items() if key not in ("relations", "errors")}, indent=2))
    if not relations:
        raise SystemExit("no valid TotalSegmentator relations found")


if __name__ == "__main__":
    main()
