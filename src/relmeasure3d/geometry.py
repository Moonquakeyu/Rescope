from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class SurfaceDistance:
    distance_mm: float
    point_a_world: tuple[float, float, float]
    point_b_world: tuple[float, float, float]
    critical_count_a: int
    critical_count_b: int


def _bbox(mask: np.ndarray, pad: int = 2) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("mask is empty")
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def mask_mesh_world(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Extract a 0.5-isosurface and return vertices in NIfTI world mm."""
    mask = np.asarray(mask, dtype=np.uint8)
    sl = _bbox(mask, pad=2)
    crop = mask[sl]
    if min(crop.shape) < 2:
        raise ValueError("mask crop is too small for marching cubes")
    vertices, _, _, _ = marching_cubes(crop, level=0.5, spacing=(1.0, 1.0, 1.0))
    origin = np.array([s.start for s in sl], dtype=np.float64)
    vertices_ijk = vertices.astype(np.float64) + origin[None]
    return nib.affines.apply_affine(affine, vertices_ijk).astype(np.float32)


def mesh_distance_from_vertices(
    vertices_a: np.ndarray,
    vertices_b: np.ndarray,
    delta_mm: float = 0.5,
) -> SurfaceDistance:
    if len(vertices_a) == 0 or len(vertices_b) == 0:
        raise ValueError("surface vertices must be non-empty")
    tree_b = cKDTree(vertices_b)
    dist_a, idx_b = tree_b.query(vertices_a, k=1, workers=1)
    ia = int(np.argmin(dist_a))
    ib = int(idx_b[ia])
    d = float(dist_a[ia])
    tree_a = cKDTree(vertices_a)
    dist_b, _ = tree_a.query(vertices_b, k=1, workers=1)
    return SurfaceDistance(
        distance_mm=d,
        point_a_world=tuple(float(x) for x in vertices_a[ia]),
        point_b_world=tuple(float(x) for x in vertices_b[ib]),
        critical_count_a=int(np.count_nonzero(dist_a <= d + delta_mm)),
        critical_count_b=int(np.count_nonzero(dist_b <= d + delta_mm)),
    )


def mesh_surface_distance(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    affine: np.ndarray,
    delta_mm: float = 0.5,
) -> SurfaceDistance:
    if np.any(mask_a & mask_b):
        overlap = np.argwhere(mask_a & mask_b)[0]
        p = nib.affines.apply_affine(affine, overlap)
        point = tuple(float(x) for x in p)
        return SurfaceDistance(0.0, point, point, 1, 1)
    return mesh_distance_from_vertices(
        mask_mesh_world(mask_a, affine),
        mask_mesh_world(mask_b, affine),
        delta_mm=delta_mm,
    )


def binary_surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~binary_erosion(mask, border_value=0)


def signed_distance_mm(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    truncate_mm: float = 10.0,
) -> np.ndarray:
    """Negative inside, positive outside, clipped in physical millimeters."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.full(mask.shape, truncate_mm, dtype=np.float32)
    outside = distance_transform_edt(~mask, sampling=spacing_mm)
    inside = distance_transform_edt(mask, sampling=spacing_mm)
    sdf = outside - inside
    return np.clip(sdf, -truncate_mm, truncate_mm).astype(np.float32)


def critical_band_voxels(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing_mm: tuple[float, float, float],
    delta_mm: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    surf_a = binary_surface(mask_a)
    surf_b = binary_surface(mask_b)
    if not surf_a.any() or not surf_b.any():
        raise ValueError("critical band requires two non-empty surfaces")
    dt_b = distance_transform_edt(~surf_b, sampling=spacing_mm)
    dt_a = distance_transform_edt(~surf_a, sampling=spacing_mm)
    d = float(min(float(dt_b[surf_a].min()), float(dt_a[surf_b].min())))
    ca = surf_a & (dt_b <= d + delta_mm)
    cb = surf_b & (dt_a <= d + delta_mm)
    return ca.astype(np.float32), cb.astype(np.float32), d
