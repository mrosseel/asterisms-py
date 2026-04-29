#!/usr/bin/env python3
"""Run universal asterism pipeline across overlapping magnitude bands and FOV bins.

Produces pre-baked results for a website: one run per (FOV bin, mag band),
with results filterable by max star magnitude at serving time.

Magnitude bands (overlapping):
  Band A: mag 0-10   (sparse, trivial compute)
  Band B: mag 8-12   (current catalog depth)
  Band C: mag 10-14  (needs deeper Gaia extract)
  Band D: mag 12-15  (needs deeper Gaia extract)

FOV bins (search radius):
  narrow:  0.3° (high-power eyepieces, planetary cameras)
  medium:  0.7° (mid-range eyepieces)
  wide:    1.2° (low-power eyepieces, Seestar-class)
  xwide:   2.0° (widefield eyepieces, binoculars)

Usage:
    python run_universal.py                          # run all bands/FOVs
    python run_universal.py --band B --fov medium    # run specific combo
    python run_universal.py --list                   # show all combos and estimates
    python run_universal.py --catalog support/gaia-14.parquet --band C
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl
import torch
from dataclasses import replace

from asterisms_py.core import (
    SearchConfig, get_grid_points, reset_gpu_catalog,
    process_all_regions, dedup_results,
    score_bright_isolated_chains,
    process_circle_regions, load_catalog_to_gpu,
)


# Magnitude bands: (name, min_mag, max_mag)
MAG_BANDS = {
    "A": (0.0, 10.0),
    "B": (8.0, 12.0),
    "C": (10.0, 14.0),
    "D": (12.0, 15.0),
}

# FOV bins: (name, search_radius_deg, grid_step_deg, max_stars_per_region)
FOV_BINS = {
    "narrow":  (0.3, 1.0, 500),
    "medium":  (0.7, 1.0, 500),
    "wide":    (1.2, 2.0, 500),
    "xwide":   (2.0, 2.0, 500),
}

# Shape configs: (shape, mode, grid_step_override, max_stars_override)
SHAPES = [
    ("triangle", "2d", None, None),
    ("triangle", "3d", None, None),
    ("square", "2d", None, None),
    ("square", "3d", None, None),
    ("collinear", "2d", None, None),
    ("circle", "2d", None, None),
]


def make_config(band_name, fov_name, shape=None):
    """Build a SearchConfig for a (band, FOV) combo."""
    _, max_mag = MAG_BANDS[band_name]
    radius, grid_step, max_stars = FOV_BINS[fov_name]

    name = f"universal_{fov_name}_{band_name}"
    max_extent = min(radius * 1.5, 4.0)

    return SearchConfig(
        name=name,
        max_mag=max_mag,
        max_extent_deg=max_extent,
        grid_step_deg=grid_step,
        search_radius_deg=radius,
        max_stars_per_region=max_stars,
    )


def filter_catalog(df, min_mag, max_mag):
    """Filter catalog to a magnitude band."""
    return df.filter(
        (pl.col("Vmag") >= min_mag) & (pl.col("Vmag") <= max_mag)
    )


def estimate_time(band_name, fov_name):
    """Rough time estimate based on star density and FOV."""
    min_mag, max_mag = MAG_BANDS[band_name]
    radius, grid_step, _ = FOV_BINS[fov_name]

    # Approximate star counts per sq deg by mag
    density_at_mag = {10: 6.8, 12: 55.9, 14: 408, 15: 1100}
    d_hi = density_at_mag.get(int(max_mag), 50)
    d_lo = density_at_mag.get(int(min_mag), 0)
    density = d_hi - d_lo

    area = np.pi * radius**2
    n_stars = density * area
    n_regions = int(360 / grid_step) * int(126 / grid_step)

    return n_stars, n_regions


def run_combo(df_band, band_name, fov_name, shapes=None, output_dir="results/universal"):
    """Run all shapes for one (band, FOV) combo."""
    if shapes is None:
        shapes = SHAPES

    cfg = make_config(band_name, fov_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        from hip_triangle import score_triangles_hip, HIP_AVAILABLE
    except ImportError:
        HIP_AVAILABLE = False
        score_triangles_hip = None
    try:
        from hip_square import score_squares_hip, HIP_SQUARE_AVAILABLE
    except ImportError:
        HIP_SQUARE_AVAILABLE = False
        score_squares_hip = None

    for shape, mode, grid_override, stars_override in shapes:
        run_cfg = cfg
        if grid_override:
            run_cfg = replace(run_cfg, grid_step_deg=grid_override)
        if stars_override:
            run_cfg = replace(run_cfg, max_stars_per_region=stars_override)

        suffix = "" if (mode == "3d" or shape == "collinear") else "_2d"
        base = f"{output_dir}/result_{shape}_{cfg.name}"

        if shape == "collinear":
            smooth_file = f"{base}_smooth.parquet"
            snake_file = f"{base}_snake.parquet"
            if os.path.exists(smooth_file) and os.path.exists(snake_file):
                print(f"  SKIP {shape} {mode} (exists)")
                continue
        elif shape == "circle":
            result_file = f"{base}_2d.parquet"
            if os.path.exists(result_file):
                print(f"  SKIP {shape} {mode} (exists)")
                continue
        else:
            result_file = f"{base}{suffix}.parquet"
            if os.path.exists(result_file):
                print(f"  SKIP {shape} {mode} (exists)")
                continue

        grid_points = get_grid_points(run_cfg, -63, 63)
        print(f"\n  {shape} {mode}: {len(grid_points):,} grid pts, "
              f"radius={run_cfg.search_radius_deg}°, max_mag={run_cfg.max_mag}")

        reset_gpu_catalog()
        t0 = time.time()

        if shape == "circle":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            gpu_cat = load_catalog_to_gpu(df_band, device)
            grid_indexed = list(zip(range(len(grid_points)), grid_points))
            result = process_circle_regions(grid_indexed, gpu_cat, run_cfg, device=device)
            elapsed = time.time() - t0

            if not result.is_empty():
                result = result.sort("score")
                result.write_parquet(f"{base}_2d.parquet")
                print(f"  Saved {len(result):,} circles [{elapsed:.0f}s]")
            else:
                print(f"  No circles [{elapsed:.0f}s]")
        else:
            result = process_all_regions(
                grid_points, df_band, config=run_cfg, batch_size=100,
                mode=mode,
                hip_available=HIP_AVAILABLE or HIP_SQUARE_AVAILABLE,
                score_triangles_hip=score_triangles_hip if HIP_AVAILABLE else None,
                score_squares_hip=score_squares_hip if HIP_SQUARE_AVAILABLE else None,
                shape=shape,
            )
            elapsed = time.time() - t0

            if shape == "collinear":
                if isinstance(result, dict) and result:
                    for score_key, df_k in result.items():
                        fn = f"{base}_{score_key}.parquet"
                        df_k = dedup_results(df_k)
                        df_k.sort("score").write_parquet(fn)
                        print(f"  Saved {fn} ({len(df_k):,} rows) [{elapsed:.0f}s]")
                else:
                    print(f"  No collinear results [{elapsed:.0f}s]")
            else:
                if not result.is_empty():
                    result = dedup_results(result)
                    result.sort("score").write_parquet(f"{base}{suffix}.parquet")
                    print(f"  Saved {len(result):,} {shape}s [{elapsed:.0f}s]")
                else:
                    print(f"  No {shape} results [{elapsed:.0f}s]")



def main():
    parser = argparse.ArgumentParser(description="Universal asterism pipeline")
    parser.add_argument('--catalog', type=str, default='support/gaia-12.parquet',
                        help='Star catalog parquet file')
    parser.add_argument('--band', type=str, help='Magnitude band (A/B/C/D or "all")')
    parser.add_argument('--fov', type=str, help='FOV bin (narrow/medium/wide/xwide or "all")')
    parser.add_argument('--shape', type=str, help='Shape filter (triangle/square/collinear/circle)')
    parser.add_argument('--mode', type=str, help='Mode filter (2d/3d)')
    parser.add_argument('--list', action='store_true', help='List all combos with estimates')
    parser.add_argument('--subprocess', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--bright-lines', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', type=str, default='results/universal',
                        help='Output directory')
    args = parser.parse_args()

    if args.list:
        print("\nUniversal pipeline combos:\n")
        print(f"{'Band':>6} {'Mag range':>12} {'FOV':>8} {'Radius':>8} "
              f"{'~Stars/reg':>12} {'Regions':>10}")
        print("-" * 70)
        for bname, (bmin, bmax) in MAG_BANDS.items():
            for fname in FOV_BINS:
                n_stars, n_regions = estimate_time(bname, fname)
                print(f"  {bname:>4} {bmin:4.0f}-{bmax:4.0f}  {fname:>8} "
                      f"{FOV_BINS[fname][0]:7.1f}°  {n_stars:>10.0f}  {n_regions:>10,}")
        return

    # Load catalog
    print(f"Loading catalog: {args.catalog}")
    df_full = pl.read_parquet(args.catalog)
    print(f"  {len(df_full):,} stars, mag {df_full['Vmag'].min():.1f} to {df_full['Vmag'].max():.1f}")

    # Subprocess mode: run a single (band, FOV, shape) combo and exit
    if args.subprocess:
        bname = args.band.upper()
        fname = args.fov.lower()
        bmin, bmax = MAG_BANDS[bname]
        effective_max = min(bmax, df_full['Vmag'].max())
        df_band = filter_catalog(df_full, bmin, effective_max)
        print(f"\nBand {bname}: mag {bmin:.0f}-{effective_max:.1f}, {len(df_band):,} stars")
        print(f"--- FOV: {fname} (radius={FOV_BINS[fname][0]}°) ---")

        if args.bright_lines:
            cfg = make_config(bname, fname)
            smooth_file = f"{args.output_dir}/result_collinear_{cfg.name}_smooth.parquet"
            bright_file = f"{args.output_dir}/result_bright_line_{cfg.name}.parquet"
            if os.path.exists(smooth_file) and not os.path.exists(bright_file):
                print(f"\n  Bright lines from {smooth_file}...")
                smooth_df = pl.read_parquet(smooth_file)
                bright_df = score_bright_isolated_chains(smooth_df, df_band, cfg.max_mag)
                bright_df = bright_df.with_columns(
                    pl.col("bright_score").alias("score")
                ).drop("bright_score").sort("score")
                bright_df = dedup_results(bright_df)
                bright_df.write_parquet(bright_file)
                print(f"  Saved {bright_file} ({len(bright_df):,} rows)")
            return

        shapes = SHAPES
        if args.shape:
            shapes = [(s, m, g, st) for s, m, g, st in SHAPES if s == args.shape]
        if args.mode:
            shapes = [(s, m, g, st) for s, m, g, st in shapes if m == args.mode]
        t0 = time.time()
        run_combo(df_band, bname, fname, shapes=shapes, output_dir=args.output_dir)
        elapsed = time.time() - t0
        print(f"\n  Combo {bname}/{fname} done in {elapsed/60:.1f} min")
        return

    # Determine which combos to run
    bands = list(MAG_BANDS.keys()) if not args.band or args.band == "all" else [args.band.upper()]
    fovs = list(FOV_BINS.keys()) if not args.fov or args.fov == "all" else [args.fov.lower()]

    # Filter shapes if requested
    shapes = SHAPES
    if args.shape:
        shapes = [(s, m, g, st) for s, m, g, st in SHAPES if s == args.shape]
    if args.mode:
        shapes = [(s, m, g, st) for s, m, g, st in shapes if m == args.mode]

    # Run each (band, FOV, shape) combo as a subprocess to reclaim memory.
    # Python's allocator doesn't return freed memory to the OS, so collinear
    # on dense catalogs can leave 100GB+ of unreclaimable RSS.  Per-shape
    # isolation ensures collinear memory is fully reclaimed before the next shape.
    for bname in bands:
        if bname not in MAG_BANDS:
            print(f"Unknown band: {bname}. Available: {list(MAG_BANDS.keys())}")
            continue

        bmin, bmax = MAG_BANDS[bname]
        catalog_max = df_full['Vmag'].max()
        if bmin > catalog_max:
            print(f"\nSKIP band {bname} (mag {bmin}-{bmax}): catalog only goes to {catalog_max:.1f}")
            continue

        for fname in fovs:
            if fname not in FOV_BINS:
                print(f"Unknown FOV: {fname}. Available: {list(FOV_BINS.keys())}")
                continue

            for shape, mode, _, _ in shapes:
                shape_tag = f"{shape}_{mode}" if shape not in ("collinear", "circle") else shape
                print(f"\n{'='*60}")
                print(f"Spawning subprocess: band {bname}, FOV {fname}, {shape_tag}")
                cmd = [sys.executable, __file__,
                       '--catalog', args.catalog,
                       '--band', bname, '--fov', fname,
                       '--output-dir', args.output_dir,
                       '--shape', shape, '--mode', mode,
                       '--subprocess']
                result = subprocess.run(cmd, env=os.environ)
                if result.returncode != 0:
                    print(f"  WARNING: subprocess exited with code {result.returncode}")

            # Post-processing: bright lines (runs in its own subprocess too)
            cfg = make_config(bname, fname)
            smooth_file = f"{args.output_dir}/result_collinear_{cfg.name}_smooth.parquet"
            bright_file = f"{args.output_dir}/result_bright_line_{cfg.name}.parquet"
            if os.path.exists(smooth_file) and not os.path.exists(bright_file):
                print(f"\n  Spawning subprocess: bright lines for {bname}/{fname}")
                cmd = [sys.executable, __file__,
                       '--catalog', args.catalog,
                       '--band', bname, '--fov', fname,
                       '--output-dir', args.output_dir,
                       '--bright-lines',
                       '--subprocess']
                subprocess.run(cmd, env=os.environ)

    print("\n\nAll done!")


if __name__ == "__main__":
    main()
