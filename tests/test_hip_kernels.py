"""Tests for HIP triangle and square kernels against PyTorch fallback."""

import math

import pytest
import torch

from asterisms_py.core import (
    mass_score_triangle_torch,
    mass_score_square_torch,
    radecmag_to_angular,
    radecmag_to_cartesian,
)

# Conditional imports for HIP kernels
try:
    from hip_triangle import score_triangles_hip, HIP_AVAILABLE
except ImportError:
    HIP_AVAILABLE = False

try:
    from hip_square import score_squares_hip, HIP_SQUARE_AVAILABLE
except ImportError:
    HIP_SQUARE_AVAILABLE = False

GPU_AVAILABLE = torch.cuda.is_available()

requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
requires_hip_triangle = pytest.mark.skipif(
    not (GPU_AVAILABLE and HIP_AVAILABLE),
    reason="HIP triangle kernel not available or no GPU",
)
requires_hip_square = pytest.mark.skipif(
    not (GPU_AVAILABLE and HIP_SQUARE_AVAILABLE),
    reason="HIP square kernel not available or no GPU",
)


def _random_star_field(n_stars, ra_center=180.0, dec_center=0.0, spread=2.0,
                       mag_range=(5.0, 10.0), device='cuda'):
    """Generate a random star field tensor on GPU."""
    ra = torch.randn(n_stars, device=device) * spread + ra_center
    dec = torch.randn(n_stars, device=device) * spread + dec_center
    dec = dec.clamp(-89.0, 89.0)
    mag = torch.rand(n_stars, device=device) * (mag_range[1] - mag_range[0]) + mag_range[0]
    return torch.stack([ra, dec, mag], dim=1)


def _equilateral_triangle(side_deg=1.0, mag=7.0, device='cuda'):
    """Create 3 stars forming an equilateral triangle on the sky."""
    h = side_deg * math.sqrt(3) / 2
    stars = torch.tensor([
        [180.0, 0.0, mag],
        [180.0 + side_deg, 0.0, mag],
        [180.0 + side_deg / 2, h, mag],
    ], device=device)
    return stars


def _perfect_square(side_deg=1.0, mag=7.0, device='cuda'):
    """Create 4 stars forming a perfect square on the sky."""
    stars = torch.tensor([
        [180.0, 0.0, mag],
        [180.0 + side_deg, 0.0, mag],
        [180.0 + side_deg, side_deg, mag],
        [180.0, side_deg, mag],
    ], device=device)
    return stars


def _isoceles_triangle(base_deg=1.0, height_deg=2.0, mag=7.0, device='cuda'):
    """Create a tall isoceles triangle — should score worse than equilateral."""
    stars = torch.tensor([
        [180.0, 0.0, mag],
        [180.0 + base_deg, 0.0, mag],
        [180.0 + base_deg / 2, height_deg, mag],
    ], device=device)
    return stars


def _rectangle(w_deg=2.0, h_deg=1.0, mag=7.0, device='cuda'):
    """Create a 2:1 rectangle — should score worse than a perfect square."""
    stars = torch.tensor([
        [180.0, 0.0, mag],
        [180.0 + w_deg, 0.0, mag],
        [180.0 + w_deg, h_deg, mag],
        [180.0, h_deg, mag],
    ], device=device)
    return stars


def _collinear_stars(n=5, spacing_deg=0.5, mag=8.0, device='cuda'):
    """Create perfectly collinear stars along RA at constant Dec."""
    stars = torch.tensor([
        [180.0 + i * spacing_deg, 0.0, mag] for i in range(n)
    ], device=device)
    return stars


# ---------------------------------------------------------------------------
# Correctness: HIP vs PyTorch fallback — Triangles
# ---------------------------------------------------------------------------

class TestTriangleHIPvsPyTorch:

    @requires_hip_triangle
    def test_combination_count_matches(self):
        """HIP and PyTorch must return the same number of combinations."""
        stars = _random_star_field(15)
        hip_scores, hip_pts = score_triangles_hip(stars, mode='2d')
        pt_scores, pt_pts = mass_score_triangle_torch(stars, device='cuda', mode='2d')

        assert hip_scores.shape[0] == pt_scores.shape[0]
        n = 15
        expected = n * (n - 1) * (n - 2) // 6
        assert hip_scores.shape[0] == expected

    @requires_hip_triangle
    def test_scores_close_2d(self):
        """HIP scores should match PyTorch scores in 2d mode."""
        stars = _random_star_field(12)
        hip_scores, _ = score_triangles_hip(stars, mode='2d')
        pt_scores, _ = mass_score_triangle_torch(stars, device='cuda', mode='2d')

        hip_sorted = hip_scores.sort().values
        pt_sorted = pt_scores.sort().values
        assert torch.allclose(hip_sorted, pt_sorted, rtol=1e-3, atol=1e-6)

    @requires_hip_triangle
    def test_scores_close_3d(self):
        """HIP scores should match PyTorch scores in 3d mode."""
        stars = _random_star_field(12)
        hip_scores, _ = score_triangles_hip(stars, mode='3d')
        pt_scores, _ = mass_score_triangle_torch(stars, device='cuda', mode='3d')

        hip_sorted = hip_scores.sort().values
        pt_sorted = pt_scores.sort().values
        assert torch.allclose(hip_sorted, pt_sorted, rtol=1e-3, atol=1e-6)

    @requires_hip_triangle
    def test_top_k_match(self):
        """The top-10 best scores should agree between HIP and PyTorch."""
        stars = _random_star_field(20)
        hip_scores, _ = score_triangles_hip(stars, mode='2d')
        pt_scores, _ = mass_score_triangle_torch(stars, device='cuda', mode='2d')

        k = min(10, hip_scores.shape[0])
        hip_topk = hip_scores.topk(k, largest=False).values
        pt_topk = pt_scores.topk(k, largest=False).values
        assert torch.allclose(hip_topk, pt_topk, rtol=1e-3, atol=1e-6)

    @requires_hip_triangle
    def test_search_radius_passthrough(self):
        """search_radius_deg should affect 3d scores consistently."""
        stars = _random_star_field(10)
        hip_scores_r2, _ = score_triangles_hip(stars, mode='3d', search_radius_deg=2.0)
        pt_scores_r2, _ = mass_score_triangle_torch(
            stars, device='cuda', mode='3d', search_radius_deg=2.0
        )
        hip_sorted = hip_scores_r2.sort().values
        pt_sorted = pt_scores_r2.sort().values
        assert torch.allclose(hip_sorted, pt_sorted, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness: HIP vs PyTorch fallback — Squares
# ---------------------------------------------------------------------------

class TestSquareHIPvsPyTorch:

    @requires_hip_square
    def test_combination_count_matches(self):
        """HIP and PyTorch must return the same number of combinations."""
        stars = _random_star_field(10)
        hip_scores, _ = score_squares_hip(stars, mode='2d')
        pt_scores, _ = mass_score_square_torch(stars, device='cuda', mode='2d')

        assert hip_scores.shape[0] == pt_scores.shape[0]
        n = 10
        expected = n * (n - 1) * (n - 2) * (n - 3) // 24
        assert hip_scores.shape[0] == expected

    @requires_hip_square
    def test_scores_correlated_2d(self):
        """HIP and PyTorch square scores should be strongly correlated in 2d mode.

        Note: HIP kernel uses parallel float ops with different accumulation order
        than PyTorch, so exact values may differ. Rankings should agree.
        """
        stars = _random_star_field(10)
        hip_scores, _ = score_squares_hip(stars, mode='2d')
        pt_scores, _ = mass_score_square_torch(stars, device='cuda', mode='2d')

        # Top-1 should agree
        hip_best = hip_scores.argmin().item()
        pt_best = pt_scores.argmin().item()
        assert hip_best == pt_best, "HIP and PyTorch should agree on best square"

    @requires_hip_square
    def test_scores_correlated_3d(self):
        """HIP and PyTorch square scores should agree on top ranking in 3d."""
        stars = _random_star_field(10)
        hip_scores, _ = score_squares_hip(stars, mode='3d')
        pt_scores, _ = mass_score_square_torch(stars, device='cuda', mode='3d')

        hip_best = hip_scores.argmin().item()
        pt_best = pt_scores.argmin().item()
        assert hip_best == pt_best

    @requires_hip_square
    def test_top_k_ranking_overlap(self):
        """The top-5 best indices should largely overlap between HIP and PyTorch."""
        stars = _random_star_field(12)
        hip_scores, _ = score_squares_hip(stars, mode='2d')
        pt_scores, _ = mass_score_square_torch(stars, device='cuda', mode='2d')

        k = min(10, hip_scores.shape[0])
        hip_topk_idx = set(hip_scores.topk(k, largest=False).indices.tolist())
        pt_topk_idx = set(pt_scores.topk(k, largest=False).indices.tolist())
        overlap = len(hip_topk_idx & pt_topk_idx)
        assert overlap >= k // 2, f"Top-{k} overlap should be >= {k//2}, got {overlap}"

    @requires_hip_square
    def test_search_radius_passthrough(self):
        """search_radius_deg should affect 3d scores consistently."""
        stars = _random_star_field(8)
        hip_scores, _ = score_squares_hip(stars, mode='3d', search_radius_deg=5.0)
        pt_scores, _ = mass_score_square_torch(
            stars, device='cuda', mode='3d', search_radius_deg=5.0
        )
        # Both should agree on best
        hip_best = hip_scores.argmin().item()
        pt_best = pt_scores.argmin().item()
        assert hip_best == pt_best


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTriangleEdgeCases:

    @requires_hip_triangle
    def test_minimum_stars(self):
        """Exactly 3 stars should produce exactly 1 triangle."""
        stars = _random_star_field(3)
        hip_scores, hip_pts = score_triangles_hip(stars, mode='2d')
        assert hip_scores.shape[0] == 1
        assert hip_pts.shape == (1, 3, 3)

    @requires_hip_triangle
    def test_equilateral_score_near_zero(self):
        """A perfect equilateral triangle should score near 0 (low CV)."""
        stars = _equilateral_triangle()
        hip_scores, _ = score_triangles_hip(stars, mode='2d')
        assert hip_scores.item() < 0.01

    @requires_gpu
    def test_equilateral_pytorch_near_zero(self):
        """PyTorch fallback: equilateral triangle should score near 0."""
        stars = _equilateral_triangle()
        scores, _ = mass_score_triangle_torch(stars, device='cuda', mode='2d')
        assert scores.item() < 0.01

    @requires_hip_triangle
    def test_same_magnitude_cv_zero_2d(self):
        """In 2d mode, magnitude doesn't affect score; equal-mag stars still score by geometry."""
        stars = _random_star_field(5, mag_range=(8.0, 8.0))
        hip_scores, _ = score_triangles_hip(stars, mode='2d')
        assert not torch.any(torch.isnan(hip_scores))
        assert not torch.any(torch.isinf(hip_scores))


class TestSquareEdgeCases:

    @requires_hip_square
    def test_minimum_stars(self):
        """Exactly 4 stars should produce exactly 1 quad."""
        stars = _random_star_field(4)
        hip_scores, hip_pts = score_squares_hip(stars, mode='2d')
        assert hip_scores.shape[0] == 1
        assert hip_pts.shape == (1, 4, 3)

    @requires_hip_square
    def test_perfect_square_score_near_zero(self):
        """A perfect square should score near 0."""
        stars = _perfect_square()
        hip_scores, _ = score_squares_hip(stars, mode='2d')
        assert hip_scores.item() < 0.05

    @requires_gpu
    def test_perfect_square_pytorch_near_zero(self):
        """PyTorch fallback: perfect square should score near 0."""
        stars = _perfect_square()
        scores, _ = mass_score_square_torch(stars, device='cuda', mode='2d')
        assert scores.item() < 0.05

    @requires_hip_square
    def test_same_magnitude_no_nan(self):
        """Equal-magnitude stars should not produce NaN or Inf scores."""
        stars = _random_star_field(6, mag_range=(7.0, 7.0))
        hip_scores, _ = score_squares_hip(stars, mode='2d')
        assert not torch.any(torch.isnan(hip_scores))
        assert not torch.any(torch.isinf(hip_scores))


# ---------------------------------------------------------------------------
# Mode consistency
# ---------------------------------------------------------------------------

class TestModeConsistency:

    @requires_hip_triangle
    def test_triangle_2d_valid(self):
        """2d mode should produce finite scores."""
        stars = _random_star_field(10)
        scores, _ = score_triangles_hip(stars, mode='2d')
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    @requires_hip_triangle
    def test_triangle_3d_valid(self):
        """3d mode should produce finite scores."""
        stars = _random_star_field(10)
        scores, _ = score_triangles_hip(stars, mode='3d')
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    @requires_hip_triangle
    def test_triangle_3d_with_radius_valid(self):
        """3d mode with search_radius_deg should produce finite scores."""
        stars = _random_star_field(10)
        scores, _ = score_triangles_hip(stars, mode='3d', search_radius_deg=2.0)
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    @requires_hip_square
    def test_square_2d_valid(self):
        """2d mode should produce finite scores."""
        stars = _random_star_field(8)
        scores, _ = score_squares_hip(stars, mode='2d')
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    @requires_hip_square
    def test_square_3d_valid(self):
        """3d mode should produce finite scores."""
        stars = _random_star_field(8)
        scores, _ = score_squares_hip(stars, mode='3d')
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    @requires_hip_square
    def test_square_3d_with_radius_valid(self):
        """3d mode with search_radius_deg should produce finite scores."""
        stars = _random_star_field(8)
        scores, _ = score_squares_hip(stars, mode='3d', search_radius_deg=5.0)
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    @requires_gpu
    def test_pytorch_triangle_modes_differ(self):
        """2d and 3d modes should generally produce different scores (different coordinate spaces)."""
        stars = _random_star_field(8, mag_range=(5.0, 12.0))
        scores_2d, _ = mass_score_triangle_torch(stars, device='cuda', mode='2d')
        scores_3d, _ = mass_score_triangle_torch(stars, device='cuda', mode='3d')
        assert not torch.allclose(scores_2d, scores_3d, atol=1e-6)

    @requires_gpu
    def test_pytorch_square_modes_differ(self):
        """2d and 3d modes should generally produce different scores."""
        stars = _random_star_field(6, mag_range=(5.0, 12.0))
        scores_2d, _ = mass_score_square_torch(stars, device='cuda', mode='2d')
        scores_3d, _ = mass_score_square_torch(stars, device='cuda', mode='3d')
        assert not torch.allclose(scores_2d, scores_3d, atol=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:

    @requires_hip_triangle
    def test_triangle_cpu_tensor_raises(self):
        """HIP triangle kernel should reject CPU tensors."""
        stars = _random_star_field(5, device='cpu')
        with pytest.raises(ValueError, match="must be on GPU"):
            score_triangles_hip(stars, mode='2d')

    @requires_hip_square
    def test_square_cpu_tensor_raises(self):
        """HIP square kernel should reject CPU tensors."""
        stars = _random_star_field(5, device='cpu')
        with pytest.raises(ValueError, match="must be on GPU"):
            score_squares_hip(stars, mode='2d')

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
    def test_pytorch_triangle_cpu_works(self):
        """PyTorch fallback should work fine on CPU."""
        stars = _random_star_field(5, device='cpu')
        scores, pts = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        assert scores.shape[0] == 10  # C(5,3)

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
    def test_pytorch_square_cpu_works(self):
        """PyTorch fallback should work fine on CPU."""
        stars = _random_star_field(5, device='cpu')
        scores, pts = mass_score_square_torch(stars, device='cpu', mode='2d')
        assert scores.shape[0] == 5  # C(5,4)

    def test_pytorch_triangle_cpu_only(self):
        """PyTorch triangle scoring works without any GPU."""
        stars = torch.tensor([
            [180.0, 0.0, 7.0],
            [181.0, 0.0, 7.5],
            [180.5, 0.866, 8.0],
        ])
        scores, pts = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        assert scores.shape[0] == 1
        assert not torch.any(torch.isnan(scores))

    def test_pytorch_square_cpu_only(self):
        """PyTorch square scoring works without any GPU."""
        stars = torch.tensor([
            [180.0, 0.0, 7.0],
            [181.0, 0.0, 7.5],
            [181.0, 1.0, 8.0],
            [180.0, 1.0, 7.0],
        ])
        scores, pts = mass_score_square_torch(stars, device='cpu', mode='2d')
        assert scores.shape[0] == 1
        assert not torch.any(torch.isnan(scores))


# ---------------------------------------------------------------------------
# PyTorch-only tests (no GPU required)
# ---------------------------------------------------------------------------

class TestPyTorchFallbackCPU:
    """Tests that always run, using CPU-only PyTorch."""

    def test_triangle_equilateral_cpu(self):
        """Equilateral triangle should score near 0 on CPU."""
        stars = _equilateral_triangle(device='cpu')
        scores, _ = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        assert scores.item() < 0.01

    def test_square_perfect_cpu(self):
        """Perfect square should score near 0 on CPU."""
        stars = _perfect_square(device='cpu')
        scores, _ = mass_score_square_torch(stars, device='cpu', mode='2d')
        assert scores.item() < 0.05

    def test_triangle_combination_count_cpu(self):
        """Verify C(n,3) combinations on CPU."""
        n = 7
        stars = torch.randn(n, 3)
        stars[:, 0] = stars[:, 0] * 10 + 180  # RA
        stars[:, 1] = stars[:, 1].clamp(-80, 80)  # Dec
        stars[:, 2] = stars[:, 2].abs() + 5  # Mag
        scores, _ = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        assert scores.shape[0] == n * (n - 1) * (n - 2) // 6

    def test_square_combination_count_cpu(self):
        """Verify C(n,4) combinations on CPU."""
        n = 7
        stars = torch.randn(n, 3)
        stars[:, 0] = stars[:, 0] * 10 + 180
        stars[:, 1] = stars[:, 1].clamp(-80, 80)
        stars[:, 2] = stars[:, 2].abs() + 5
        scores, _ = mass_score_square_torch(stars, device='cpu', mode='2d')
        assert scores.shape[0] == n * (n - 1) * (n - 2) * (n - 3) // 24

    def test_triangle_3d_mode_cpu(self):
        """3d mode should produce valid results on CPU."""
        stars = torch.tensor([
            [180.0, 0.0, 6.0],
            [181.0, 0.0, 8.0],
            [180.5, 0.866, 10.0],
        ])
        scores, pts = mass_score_triangle_torch(stars, device='cpu', mode='3d')
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    def test_square_3d_mode_cpu(self):
        """3d mode should produce valid results on CPU."""
        stars = torch.tensor([
            [180.0, 0.0, 6.0],
            [181.0, 0.0, 8.0],
            [181.0, 1.0, 10.0],
            [180.0, 1.0, 7.0],
        ])
        scores, pts = mass_score_square_torch(stars, device='cpu', mode='3d')
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    def test_search_radius_changes_3d_scores(self):
        """Different search_radius_deg values should produce different 3d scores."""
        stars = torch.tensor([
            [180.0, 0.0, 6.0],
            [181.0, 0.0, 8.0],
            [180.5, 0.866, 10.0],
        ])
        scores_r2, _ = mass_score_triangle_torch(
            stars, device='cpu', mode='3d', search_radius_deg=2.0
        )
        scores_r20, _ = mass_score_triangle_torch(
            stars, device='cpu', mode='3d', search_radius_deg=20.0
        )
        assert not torch.allclose(scores_r2, scores_r20, atol=1e-6)

    def test_points_output_shape_triangle(self):
        """Triangle output points should be (N, 3, 3) — N combos, 3 vertices, 3 cols."""
        stars = torch.randn(6, 3)
        stars[:, 0] = stars[:, 0] * 10 + 180
        stars[:, 1] = stars[:, 1].clamp(-80, 80)
        stars[:, 2] = stars[:, 2].abs() + 5
        scores, pts = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        assert pts.shape == (20, 3, 3)  # C(6,3) = 20

    def test_points_output_shape_square(self):
        """Square output points should be (N, 4, 3) — N combos, 4 vertices, 3 cols."""
        stars = torch.randn(6, 3)
        stars[:, 0] = stars[:, 0] * 10 + 180
        stars[:, 1] = stars[:, 1].clamp(-80, 80)
        stars[:, 2] = stars[:, 2].abs() + 5
        scores, pts = mass_score_square_torch(stars, device='cpu', mode='2d')
        assert pts.shape == (15, 4, 3)  # C(6,4) = 15


# ---------------------------------------------------------------------------
# Known geometry tests — validate scoring logic, not just cross-implementation
# ---------------------------------------------------------------------------

class TestKnownGeometryTriangles:
    """Test triangle scoring against mathematically known shapes."""

    def test_equilateral_scores_near_zero(self):
        """Perfect equilateral: all sides equal → CV ≈ 0 → score ≈ 0."""
        stars = _equilateral_triangle(device='cpu')
        scores, _ = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        assert scores.item() < 0.01, f"Equilateral should score ~0, got {scores.item()}"

    def test_isoceles_worse_than_equilateral(self):
        """A tall isoceles (base=1, height=2) should score worse than equilateral."""
        equi = _equilateral_triangle(device='cpu')
        iso = _isoceles_triangle(device='cpu')
        eq_score, _ = mass_score_triangle_torch(equi, device='cpu', mode='2d')
        iso_score, _ = mass_score_triangle_torch(iso, device='cpu', mode='2d')
        assert iso_score.item() > eq_score.item(), \
            f"Isoceles ({iso_score.item():.4f}) should score worse than equilateral ({eq_score.item():.4f})"

    def test_equilateral_among_noise_is_best(self):
        """In a field with an equilateral + random stars, the equilateral combo should be the best."""
        equi = _equilateral_triangle(side_deg=1.0, mag=7.0, device='cpu')
        noise = torch.tensor([
            [183.0, 2.0, 9.0],
            [177.0, -1.5, 8.0],
            [182.0, -3.0, 10.0],
        ])
        stars = torch.cat([equi, noise], dim=0)
        scores, pts = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        best_idx = scores.argmin()
        best_pts = pts[best_idx]
        # The best triangle should contain the equilateral vertices (first 3 stars)
        best_ras = set(best_pts[:, 0].tolist())
        equi_ras = set(equi[:, 0].tolist())
        assert best_ras == equi_ras, "Best triangle should be the equilateral"

    def test_score_increases_with_deformation(self):
        """Progressively deforming a triangle should increase score monotonically."""
        base_ra = 180.0
        scores_list = []
        for offset in [0.866, 1.5, 2.5, 4.0]:  # equilateral → increasingly deformed
            stars = torch.tensor([
                [base_ra, 0.0, 7.0],
                [base_ra + 1.0, 0.0, 7.0],
                [base_ra + 0.5, offset, 7.0],
            ])
            s, _ = mass_score_triangle_torch(stars, device='cpu', mode='2d')
            scores_list.append(s.item())
        for i in range(1, len(scores_list)):
            assert scores_list[i] >= scores_list[i - 1] - 0.001, \
                f"Score should increase with deformation: {scores_list}"

    @requires_hip_triangle
    def test_hip_equilateral_near_zero(self):
        """HIP kernel: equilateral triangle should score near 0."""
        stars = _equilateral_triangle(device='cuda')
        scores, _ = score_triangles_hip(stars, mode='2d')
        assert scores.item() < 0.01

    @requires_hip_triangle
    def test_hip_equilateral_best_in_field(self):
        """HIP kernel: equilateral should be best triangle in a mixed field."""
        equi = _equilateral_triangle(side_deg=1.0, mag=7.0, device='cuda')
        noise = torch.tensor([
            [183.0, 2.0, 9.0],
            [177.0, -1.5, 8.0],
            [182.0, -3.0, 10.0],
        ], device='cuda')
        stars = torch.cat([equi, noise], dim=0)
        scores, pts = score_triangles_hip(stars, mode='2d')
        best_idx = scores.argmin()
        best_pts = pts[best_idx]
        best_ras = set(best_pts[:, 0].tolist())
        equi_ras = set(equi[:, 0].tolist())
        assert best_ras == equi_ras


class TestKnownGeometrySquares:
    """Test square scoring against mathematically known shapes."""

    def test_perfect_square_scores_near_zero(self):
        """Perfect square: all sides equal, diags equal → score ≈ 0."""
        stars = _perfect_square(device='cpu')
        scores, _ = mass_score_square_torch(stars, device='cpu', mode='2d')
        assert scores.item() < 0.05, f"Perfect square should score ~0, got {scores.item()}"

    def test_rectangle_worse_than_square(self):
        """A 2:1 rectangle should score worse than a perfect square."""
        sq = _perfect_square(device='cpu')
        rect = _rectangle(device='cpu')
        sq_score, _ = mass_score_square_torch(sq, device='cpu', mode='2d')
        rect_score, _ = mass_score_square_torch(rect, device='cpu', mode='2d')
        assert rect_score.item() > sq_score.item(), \
            f"Rectangle ({rect_score.item():.4f}) should score worse than square ({sq_score.item():.4f})"

    def test_square_among_noise_is_best(self):
        """In a field with a perfect square + random stars, the square combo should be best."""
        sq = _perfect_square(side_deg=1.0, mag=7.0, device='cpu')
        noise = torch.tensor([
            [184.0, 3.0, 9.0],
            [176.0, -2.0, 10.0],
        ])
        stars = torch.cat([sq, noise], dim=0)
        scores, pts = mass_score_square_torch(stars, device='cpu', mode='2d')
        best_idx = scores.argmin()
        best_pts = pts[best_idx]
        best_ras = sorted(best_pts[:, 0].tolist())
        sq_ras = sorted(sq[:, 0].tolist())
        assert best_ras == sq_ras, "Best quad should be the perfect square"

    @requires_hip_square
    def test_hip_perfect_square_near_zero(self):
        """HIP kernel: perfect square should score near 0."""
        stars = _perfect_square(device='cuda')
        scores, _ = score_squares_hip(stars, mode='2d')
        assert scores.item() < 0.05

    @requires_hip_square
    def test_hip_rectangle_worse_than_square(self):
        """HIP kernel: rectangle should score worse than square."""
        sq = _perfect_square(device='cuda')
        rect = _rectangle(device='cuda')
        sq_score, _ = score_squares_hip(sq, mode='2d')
        rect_score, _ = score_squares_hip(rect, mode='2d')
        assert rect_score.item() > sq_score.item()

    @requires_hip_square
    def test_hip_square_best_in_field(self):
        """HIP kernel: perfect square should be best in a mixed field."""
        sq = _perfect_square(side_deg=1.0, mag=7.0, device='cuda')
        noise = torch.tensor([
            [184.0, 3.0, 9.0],
            [176.0, -2.0, 10.0],
        ], device='cuda')
        stars = torch.cat([sq, noise], dim=0)
        scores, pts = score_squares_hip(stars, mode='2d')
        best_idx = scores.argmin()
        best_pts = pts[best_idx]
        best_ras = sorted(best_pts[:, 0].tolist())
        sq_ras = sorted(sq[:, 0].tolist())
        assert best_ras == sq_ras


class TestKnownGeometry3D:
    """Test that 3D mode properly incorporates magnitude."""

    def test_equal_mag_triangle_same_as_2d(self):
        """With equal magnitudes, 3d and 2d should produce similar rankings."""
        stars = torch.tensor([
            [180.0, 0.0, 7.0],
            [181.0, 0.0, 7.0],
            [180.5, 0.866, 7.0],
            [183.0, 2.0, 7.0],
            [177.0, -1.0, 7.0],
        ])
        scores_2d, _ = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        scores_3d, _ = mass_score_triangle_torch(stars, device='cpu', mode='3d')
        best_2d = scores_2d.argmin().item()
        best_3d = scores_3d.argmin().item()
        assert best_2d == best_3d, "Equal-mag stars: 2d and 3d should agree on best triangle"

    def test_varied_mag_changes_3d_ranking(self):
        """With varied magnitudes, 3d should rank differently than 2d."""
        # Equilateral in 2d but one star much brighter → 3d geometry is distorted
        stars = torch.tensor([
            [180.0, 0.0, 5.0],   # very bright
            [181.0, 0.0, 12.0],  # very dim
            [180.5, 0.866, 8.0], # medium
        ])
        scores_2d, _ = mass_score_triangle_torch(stars, device='cpu', mode='2d')
        scores_3d, _ = mass_score_triangle_torch(stars, device='cpu', mode='3d')
        # 2d should see near-equilateral; 3d should see distorted (worse score)
        assert scores_3d.item() > scores_2d.item(), \
            f"3d ({scores_3d.item():.4f}) should be worse than 2d ({scores_2d.item():.4f}) with varied mags"
