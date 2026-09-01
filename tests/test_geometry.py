from __future__ import annotations

import numpy as np

from relmeasure3d.geometry import critical_band_voxels, mesh_surface_distance, signed_distance_mm


def test_separated_cubes_and_affine_translation() -> None:
    a = np.zeros((32, 32, 32), dtype=bool)
    b = np.zeros_like(a)
    a[5:10, 8:14, 8:14] = True
    b[14:19, 8:14, 8:14] = True
    affine = np.eye(4)
    affine[:3, 3] = [12.0, -4.0, 8.0]
    result = mesh_surface_distance(a, b, affine)
    assert abs(result.distance_mm - 4.0) < 1e-5


def test_anisotropic_spacing() -> None:
    a = np.zeros((32, 32, 32), dtype=bool)
    b = np.zeros_like(a)
    a[5:10, 8:14, 8:14] = True
    b[12:17, 8:14, 8:14] = True
    affine = np.diag([2.0, 0.5, 1.25, 1.0])
    result = mesh_surface_distance(a, b, affine)
    assert abs(result.distance_mm - 4.0) < 1e-5


def test_touching_returns_zero() -> None:
    a = np.zeros((24, 24, 24), dtype=bool)
    b = np.zeros_like(a)
    a[3:9, 3:9, 3:9] = True
    b[9:15, 3:9, 3:9] = True
    result = mesh_surface_distance(a, b, np.eye(4))
    assert result.distance_mm < 1e-5


def test_signed_distance_and_critical_band() -> None:
    a = np.zeros((32, 32, 32), dtype=bool)
    b = np.zeros_like(a)
    a[5:10, 10:15, 10:15] = True
    b[14:19, 10:15, 10:15] = True
    sdf = signed_distance_mm(a, (1.0, 1.0, 1.0), truncate_mm=10.0)
    assert sdf[7, 12, 12] < 0
    assert sdf[20, 20, 20] > 0
    ca, cb, distance = critical_band_voxels(a, b, (1.0, 1.0, 1.0), delta_mm=1.0)
    assert ca.sum() > 0 and cb.sum() > 0
    assert distance > 0
