"""Run all 6 config/mode combinations for asterism search."""
import sys
import os

# Add project root for hip_triangle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
import torch

from asterisms_py.core import (
    SearchConfig, NAKED_EYE, BINOCULAR, TELESCOPIC,
    get_grid_points, reset_gpu_catalog,
    process_all_regions, add_to_result_and_save,
    chain_extent_deg, score_collinear_region,
)

# Import HIP kernels
try:
    from hip_triangle import score_triangles_hip, HIP_AVAILABLE
    if HIP_AVAILABLE:
        print("HIP triangle kernel loaded")
except ImportError:
    HIP_AVAILABLE = False
    print("HIP triangle kernel not available")

try:
    from hip_square import score_squares_hip, HIP_SQUARE_AVAILABLE
    if HIP_SQUARE_AVAILABLE:
        print("HIP square kernel loaded")
except ImportError:
    HIP_SQUARE_AVAILABLE = False
    print("HIP square kernel not available")

df = pl.read_parquet('support/tyc2-3.parquet')
print(f"Catalog: {len(df):,} stars")

# (config, base_name, shape)
triangle_configs = [
    (TELESCOPIC, "result_triangle2", "triangle"),
    (BINOCULAR, "result_triangle2_bino", "triangle"),
    (NAKED_EYE, "result_triangle2_naked", "triangle"),
]

TELESCOPIC_SQUARES = SearchConfig(
    name="telescopic_squares", max_mag=15, max_extent_deg=2,
    grid_step_deg=4, search_radius_deg=2, max_stars_per_region=200,
)
square_configs = [
    (TELESCOPIC_SQUARES, "result_squareness", "square"),
]

TELESCOPIC_COLLINEAR = SearchConfig(
    name="telescopic_collinear", max_mag=15, max_extent_deg=2,
    grid_step_deg=4, search_radius_deg=2, max_stars_per_region=200,
)
BINOCULAR_COLLINEAR = SearchConfig(
    name="binocular_collinear", max_mag=11, max_extent_deg=6,
    grid_step_deg=4, search_radius_deg=6, max_stars_per_region=200,
)
collinear_configs = [
    (TELESCOPIC_COLLINEAR, "result_collinear", "collinear"),
    (BINOCULAR_COLLINEAR, "result_collinear_bino", "collinear"),
]

all_configs = triangle_configs + square_configs + collinear_configs
modes = ["3d", "2d"]
hip_tri_fn = score_triangles_hip if HIP_AVAILABLE else None
hip_sq_fn = score_squares_hip if HIP_SQUARE_AVAILABLE else None

for config, base_name, shape in all_configs:
    shape_modes = ["2d"] if shape == "collinear" else modes
    for scoring_mode in shape_modes:
        suffix = "" if (scoring_mode == "3d" or shape == "collinear") else "_2d"
        result_filename = f"{base_name}{suffix}.parquet"

        grid_points = get_grid_points(config, -63, 63)
        print(f"\n{'=' * 60}")
        print(f"{config.name} / {scoring_mode} / {shape}, {len(grid_points)} grid points")

        reset_gpu_catalog()
        result = process_all_regions(
            grid_points, df, config=config, start=0, batch_size=100,
            mode=scoring_mode, hip_available=HIP_AVAILABLE or HIP_SQUARE_AVAILABLE,
            score_triangles_hip=hip_tri_fn, score_squares_hip=hip_sq_fn,
            shape=shape,
        )

        if shape == 'collinear':
            # process_all_regions returns dict of 3 DataFrames for collinear
            if isinstance(result, dict) and result:
                for score_key, df_k in result.items():
                    fn = result_filename.replace(".parquet", f"_{score_key}.parquet")
                    df_k.sort("score").write_parquet(fn)
                    print(f"Saved {fn} ({len(df_k):,} rows)")
                    print(df_k.sort("score").head(3))
            else:
                print(f"No results for {config.name}/{scoring_mode}")
        else:
            print(f"Results: {len(result)}")
            if not result.is_empty():
                add_to_result_and_save(pl.DataFrame(), result, result_filename)
                print(f"Saved {result_filename}")
                print(result.sort("score").head(3))
            else:
                print(f"No results for {config.name}/{scoring_mode}")

print("\n\nDone! All runs complete.")
