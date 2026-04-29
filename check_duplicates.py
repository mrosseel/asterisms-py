#!/usr/bin/env python3
"""Report duplicate asterisms across FOV configs (eyepiece/camera).

For each shape type, compares results across configs by hashing star
coordinates. Shows how many asterisms appear in multiple configs.
"""
import glob
import os
import re
from collections import defaultdict

import polars as pl
import numpy as np


def star_hash(stars):
    """Hash a star pattern by sorted rounded coordinates."""
    pts = sorted(tuple(round(c, 3) for c in s) for s in stars)
    return str(pts)


def analyze_shape(shape_name, file_pattern, max_per_file=5000):
    """Analyze duplicates across configs for one shape type."""
    files = sorted(glob.glob(file_pattern))
    if not files:
        return

    print(f"\n{'='*60}")
    print(f"  {shape_name}")
    print(f"{'='*60}")

    # Extract config tag from filename
    def config_tag(path):
        base = os.path.basename(path)
        # Remove prefix and suffix to get config name
        base = re.sub(r'^result_(triangle2|squareness|collinear|circle|bright_line)_', '', base)
        base = re.sub(r'(_2d|_smooth|_smooth_mag|_snake|_rms|_msd)?\.parquet$', '', base)
        return base

    # Load top N from each file, hash stars
    config_hashes = {}  # config -> set of hashes
    hash_to_configs = defaultdict(set)  # hash -> set of configs
    config_counts = {}

    for f in files:
        tag = config_tag(f)
        df = pl.read_parquet(f)
        config_counts[tag] = len(df)
        top = df.sort("score").head(max_per_file)
        hashes = set()
        for row in top.iter_rows(named=True):
            h = star_hash(row['stars'])
            hashes.add(h)
            hash_to_configs[h].add(tag)
        config_hashes[tag] = hashes

    configs = sorted(config_hashes.keys())
    print(f"\nConfigs ({len(configs)}):")
    for c in configs:
        print(f"  {c}: {config_counts[c]:,} total, {len(config_hashes[c]):,} sampled")

    # Count duplicates
    n_unique = len(hash_to_configs)
    n_shared = sum(1 for h, cs in hash_to_configs.items() if len(cs) > 1)
    n_total = sum(len(h) for h in config_hashes.values())

    print(f"\nOverlap (top {max_per_file} per config):")
    print(f"  Total entries across configs: {n_total:,}")
    print(f"  Unique patterns: {n_unique:,}")
    print(f"  Shared across 2+ configs: {n_shared:,} ({100*n_shared/max(n_unique,1):.1f}%)")

    # Pairwise overlap matrix
    if len(configs) > 1:
        print(f"\nPairwise overlap (shared / min(A,B)):")
        # Header
        short = [c.split('_')[-2] + '_' + c.split('_')[-1] if '_' in c else c for c in configs]
        header = f"  {'':>20}" + "".join(f"{s:>12}" for s in short)
        print(header)
        for i, ci in enumerate(configs):
            si = short[i]
            row = f"  {si:>20}"
            for j, cj in enumerate(configs):
                overlap = len(config_hashes[ci] & config_hashes[cj])
                min_size = min(len(config_hashes[ci]), len(config_hashes[cj]))
                pct = 100 * overlap / max(min_size, 1)
                if i == j:
                    row += f"{'---':>12}"
                else:
                    row += f"{overlap:>6} ({pct:4.1f}%)"
                    # row += f"{pct:>11.1f}%"
            print(row)


if __name__ == '__main__':
    print("Duplicate analysis across FOV configs")
    print("(comparing top 5000 results per config)")

    # Triangles 3D
    analyze_shape("Triangles (3D)",
                  "results/result_triangle2_10in_f5_*.parquet",
                  max_per_file=5000)

    # Triangles 2D
    analyze_shape("Triangles (2D)",
                  "results/result_triangle2_10in_f5_*_2d.parquet",
                  max_per_file=5000)

    # Squares 3D
    analyze_shape("Squares (3D)",
                  "results/result_squareness_10in_f5_*.parquet",
                  max_per_file=5000)

    # Collinear smooth
    analyze_shape("Collinear (smooth)",
                  "results/result_collinear_10in_f5_*_smooth.parquet",
                  max_per_file=5000)

    # Collinear snake
    analyze_shape("Collinear (snake)",
                  "results/result_collinear_10in_f5_*_snake.parquet",
                  max_per_file=5000)

    # Bright lines
    analyze_shape("Bright isolated lines",
                  "results/result_bright_line_10in_f5_*.parquet",
                  max_per_file=5000)

    # Circles
    analyze_shape("Circles/arcs",
                  "results/result_circle_10in_f5_*_2d.parquet",
                  max_per_file=5000)

    print("\n\nDone!")
