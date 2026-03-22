"""
HIP kernel for triangle scoring on AMD GPUs

This module provides a pure GPU implementation that eliminates
torch.combinations overhead by generating triangles directly in HIP kernel.
"""

import os
import importlib
import torch
from asterisms_py.core import radecmag_to_cartesian, radecmag_to_angular

# Import triangle_hip from this package's directory
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "triangle_hip",
        os.path.join(_pkg_dir, "triangle_hip.so")
    )
    triangle_hip = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(triangle_hip)
    HIP_AVAILABLE = True
except (ImportError, OSError) as e:
    HIP_AVAILABLE = False
    triangle_hip = None
    print(f"  hip_triangle load error: {e}")


def score_triangles_hip(stars_tensor, mode='3d', search_radius_deg=None):
    """
    Score all triangles using HIP kernel.

    Args:
        stars_tensor: [n_stars, 3] float32 tensor on GPU (RA, Dec, Mag)
        mode: '3d' for Cartesian scoring, '2d' for angular-only
        search_radius_deg: scale depth axis to match angular extent (for 3d mode)

    Returns:
        scores: [n_triangles] float32 tensor of triangle scores
        points: [n_triangles, 3, 3] float32 tensor of triangle vertices (original RA/Dec/Mag)
    """
    if not HIP_AVAILABLE:
        raise RuntimeError("HIP kernel not compiled. Run: cd hip_triangle && ./build.sh")

    if not stars_tensor.is_cuda:
        raise ValueError("stars_tensor must be on GPU")

    scoring_data = radecmag_to_cartesian(stars_tensor, search_radius_deg=search_radius_deg) if mode == '3d' else radecmag_to_angular(stars_tensor)
    scores, indices = triangle_hip.triangle_score(scoring_data)

    # Gather original (RA/Dec/Mag) vertices using indices
    indices_long = indices.long()
    p1 = stars_tensor[indices_long[:, 0]]
    p2 = stars_tensor[indices_long[:, 1]]
    p3 = stars_tensor[indices_long[:, 2]]
    points = torch.stack([p1, p2, p3], dim=1)

    return scores, points


__all__ = ['score_triangles_hip', 'HIP_AVAILABLE']
