#!/usr/bin/env python3
"""Re-score existing snake parquets with updated formula.

Applies turn magnitude penalty and 45° hard filter to existing results.
Note: star order in existing parquets is SVD-sorted (not path order).
For full path-order benefits, re-run the pipeline.

Usage:
    python rescore_snakes.py                    # re-score all snake parquets
    python rescore_snakes.py --dry-run          # show what would change
"""
import argparse
import glob
import math
import sys

import numpy as np
import polars as pl
import torch


def rescore_chain(stars, cutoff=1.5):
    """Re-score a single chain using the updated snake formula.

    Args:
        stars: list of [RA, Dec, Vmag] — in stored order
        cutoff: score cutoff (chains above this are dropped)

    Returns:
        new_score or None if chain should be dropped
    """
    pts = torch.tensor(stars, dtype=torch.float32)
    K = len(pts)
    if K < 5:
        return None

    # Convert to unit vectors
    ra_rad = torch.deg2rad(pts[:, 0])
    dec_rad = torch.deg2rad(pts[:, 1])
    x = torch.cos(dec_rad) * torch.cos(ra_rad)
    y = torch.cos(dec_rad) * torch.sin(ra_rad)
    z = torch.sin(dec_rad)
    uv = torch.stack([x, y, z], dim=1)

    # Direction vectors
    dirs = uv[1:] - uv[:-1]
    dir_norms = torch.linalg.norm(dirs, dim=1, keepdim=True).clamp(min=1e-12)
    dirs_n = dirs / dir_norms

    # Turn angles
    cos_turn = (dirs_n[:-1] * dirs_n[1:]).sum(dim=1).clamp(-1, 1)
    turn_deg = torch.rad2deg(torch.acos(cos_turn))

    # Spacing CV
    spacings = torch.acos((uv[:-1] * uv[1:]).sum(dim=1).clamp(-1, 1))
    mean_sp = spacings.mean()
    cv = (spacings.std() / mean_sp).item() if mean_sp > 1e-12 else 0.0

    # SVD for chain normal
    centroid = uv.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(uv - centroid, full_matrices=False)
    chain_normal = Vh[2]

    # Alternation
    crosses = torch.cross(dirs_n[:-1], dirs_n[1:], dim=1)
    turn_sign = (crosses * chain_normal.unsqueeze(0)).sum(dim=1)
    signs = torch.sign(turn_sign)
    sign_changes = (signs[:-1] * signs[1:] < 0).float()
    alternation_ratio = sign_changes.mean().item()

    # Turn amplitude regularity
    turn_abs = turn_deg.abs()
    turn_mean = turn_abs.mean().item()
    turn_angle_cv = (turn_abs.std() / turn_abs.mean()).item() if turn_mean > 1e-12 else 0.0

    # Turn magnitude penalty
    turn_mag_penalty = max((turn_mean - 15.0) / 30.0, 0.0)

    # Hard filter
    if turn_mean > 45.0:
        return None

    score = (1.0 - alternation_ratio) + 0.3 * turn_angle_cv + 0.2 * cv + 0.4 * turn_mag_penalty

    return score if score <= cutoff else None


def rescore_file(filepath, dry_run=False):
    """Re-score a single snake parquet file."""
    df = pl.read_parquet(filepath)
    n_before = len(df)

    new_scores = []
    keep_mask = []
    for row in df.iter_rows(named=True):
        new_score = rescore_chain(row['stars'])
        if new_score is not None:
            new_scores.append(new_score)
            keep_mask.append(True)
        else:
            new_scores.append(999.0)
            keep_mask.append(False)

    df = df.with_columns(pl.Series("score", new_scores))
    df = df.filter(pl.Series(keep_mask))
    df = df.sort("score")
    n_after = len(df)

    print(f"  {filepath}: {n_before:,} -> {n_after:,} chains ({n_before - n_after:,} dropped)")

    if not dry_run:
        df.write_parquet(filepath)
        print(f"    Saved.")


def main():
    parser = argparse.ArgumentParser(description="Re-score snake parquets with updated formula")
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing')
    args = parser.parse_args()

    files = sorted(glob.glob('results/result_collinear_*_snake.parquet'))
    if not files:
        print("No snake parquet files found in results/")
        sys.exit(1)

    print(f"Found {len(files)} snake parquet files")
    for f in files:
        rescore_file(f, dry_run=args.dry_run)

    print("\nDone!")


if __name__ == '__main__':
    main()
