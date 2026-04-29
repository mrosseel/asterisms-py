#!/usr/bin/env python3
"""Generate PDF reports from universal pipeline results.

Scans results/universal/ for parquet files and generates reports for each
(band, FOV, shape) combo found.

Usage:
    python generate_universal_reports.py                    # all results
    python generate_universal_reports.py --band A --fov narrow  # specific combo
    python generate_universal_reports.py --band A           # all FOVs for band A
"""
import argparse
import glob
import os
import re
import sys

from generate_reports import _process_mode, OUTPUT_DIR


# Map file patterns to (shape, scoring) for _process_mode
FILE_PATTERNS = {
    r'result_triangle_(.+)_2d\.parquet':      ('triangle', '2d'),
    r'result_triangle_(.+)(?<!_2d)\.parquet':  ('triangle', '3d'),
    r'result_square_(.+)_2d\.parquet':         ('square', '2d'),
    r'result_square_(.+)(?<!_2d)\.parquet':    ('square', '3d'),
    r'result_collinear_(.+)_smooth\.parquet':  ('collinear', 'smooth'),
    r'result_collinear_(.+)_smooth_mag\.parquet': ('collinear', 'smooth_mag'),
    r'result_collinear_(.+)_snake\.parquet':   ('collinear', 'snake'),
    r'result_bright_line_(.+)\.parquet':       ('collinear', 'bright_line'),
    r'result_circle_(.+)_2d\.parquet':         ('circle', 'circle'),
}

# Magnitude bands
MAG_BANDS = {
    "A": (0.0, 10.0),
    "B": (8.0, 12.0),
    "C": (10.0, 14.0),
    "D": (12.0, 15.0),
}

FOV_LABELS = {
    "narrow": "0.3\u00b0",
    "medium": "0.7\u00b0",
    "wide": "1.2\u00b0",
    "xwide": "2.0\u00b0",
}


def build_modes(results_dir="results/universal", band_filter=None, fov_filter=None):
    """Scan results directory and build mode list for report generation."""
    modes = []
    files = sorted(glob.glob(os.path.join(results_dir, "*.parquet")))

    for filepath in files:
        basename = os.path.basename(filepath)

        # Try each pattern
        for pattern, (shape, scoring) in FILE_PATTERNS.items():
            m = re.match(pattern, basename)
            if not m:
                continue

            config_tag = m.group(1)
            # Parse config tag: universal_{fov}_{band}
            parts = config_tag.split('_')
            if len(parts) < 3 or parts[0] != 'universal':
                break
            fov_name = parts[1]
            band_name = parts[2]

            # Apply filters
            if band_filter and band_name != band_filter:
                break
            if fov_filter and fov_name != fov_filter:
                break

            max_mag = MAG_BANDS.get(band_name, (0, 12))[1]
            fov_label = FOV_LABELS.get(fov_name, fov_name)

            display = f"Band {band_name} {fov_name} ({fov_label}) {shape} {scoring}"
            subdir = f"band_{band_name}/{fov_name}/{shape}"

            modes.append((subdir, scoring, filepath, max_mag, display, shape))
            break

    return modes


def main():
    parser = argparse.ArgumentParser(description="Generate universal pipeline reports")
    parser.add_argument('--band', type=str, help='Filter by band (A/B/C/D)')
    parser.add_argument('--fov', type=str, help='Filter by FOV (narrow/medium/wide/xwide)')
    parser.add_argument('--results-dir', default='results/universal', help='Results directory')
    parser.add_argument('--list', action='store_true', help='List modes without generating')
    args = parser.parse_args()

    band = args.band.upper() if args.band else None
    fov = args.fov.lower() if args.fov else None

    modes = build_modes(args.results_dir, band_filter=band, fov_filter=fov)

    if not modes:
        print(f"No results found in {args.results_dir}")
        if band or fov:
            print(f"  (filter: band={band}, fov={fov})")
        return

    if args.list:
        print(f"\n{len(modes)} report modes found:\n")
        for subdir, scoring, filepath, max_mag, display, shape in modes:
            print(f"  {display}")
            print(f"    {filepath}")
        return

    run_dir = os.path.join(OUTPUT_DIR, "universal")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nOutput directory: {run_dir}")
    print(f"Processing {len(modes)} report modes\n")

    for subdir, scoring, filepath, max_mag, display, shape in modes:
        _process_mode(subdir, scoring, filepath, max_mag, display, shape,
                      run_dir=run_dir)

    print(f"\n\nDone! Reports in {run_dir}")


if __name__ == '__main__':
    main()
