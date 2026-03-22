"""
HIP kernel for square scoring on AMD GPUs

Generates all C(N,4) combinations on-the-fly in the kernel,
tests all 3 quadrilateral cycle orderings per combo,
and returns the best squareness score.
"""

import os
import importlib
import torch
from asterisms_py.core import radecmag_to_cartesian, radecmag_to_angular

_pkg_dir = os.path.dirname(os.path.abspath(__file__))
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "square_hip",
        os.path.join(_pkg_dir, "square_hip.so")
    )
    square_hip = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(square_hip)
    HIP_SQUARE_AVAILABLE = True
except (ImportError, OSError) as e:
    HIP_SQUARE_AVAILABLE = False
    square_hip = None
    print(f"  hip_square load error: {e}")


def score_squares_hip(stars_tensor, mode='3d', max_dist=float('inf'), search_radius_deg=None):
    """
    Score all 4-star combinations for squareness using HIP kernel.

    Args:
        stars_tensor: [n_stars, 3] float32 tensor on GPU (RA, Dec, Mag)
        mode: '3d' for perceptual distance scoring, '2d' for angular-only
        max_dist: max pairwise distance threshold (in scoring coordinate space).
                  Quads with any pair exceeding this are skipped.
        search_radius_deg: scale depth axis to match angular extent (for 3d mode)

    Returns:
        scores: [n_quads] float32 tensor of squareness scores
        points: [n_quads, 4, 3] float32 tensor of quad vertices (original RA/Dec/Mag)
    """
    if not HIP_SQUARE_AVAILABLE:
        raise RuntimeError("HIP kernel not compiled. Run: cd hip_square && ./build.sh")

    if not stars_tensor.is_cuda:
        raise ValueError("stars_tensor must be on GPU")

    scoring_data = radecmag_to_cartesian(stars_tensor, search_radius_deg=search_radius_deg) if mode == '3d' else radecmag_to_angular(stars_tensor)
    scores, indices = square_hip.square_score(scoring_data, max_dist)

    indices_long = indices.long()
    p1 = stars_tensor[indices_long[:, 0]]
    p2 = stars_tensor[indices_long[:, 1]]
    p3 = stars_tensor[indices_long[:, 2]]
    p4 = stars_tensor[indices_long[:, 3]]
    points = torch.stack([p1, p2, p3, p4], dim=1)

    return scores, points


__all__ = ['score_squares_hip', 'HIP_SQUARE_AVAILABLE']
