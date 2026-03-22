#!/usr/bin/env python3
"""Configure telescope and eyepieces, then run pipeline + report generation.

Usage:
    python configure_instrument.py                    # interactive mode
    python configure_instrument.py --list-presets     # show built-in presets
    python configure_instrument.py --preset 10in_f5   # use a preset directly
    python configure_instrument.py --reports-only     # skip pipeline, just generate reports
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import replace

from asterisms_py.core import (
    InstrumentConfig, Eyepiece, Camera, DEFAULT_INSTRUMENT,
    eyepiece_to_search_config, camera_to_search_config,
    sqm_to_nelm, telescope_limiting_mag,
)


PRESETS = {
    "10in_f5": DEFAULT_INSTRUMENT,
    "8in_f10": InstrumentConfig(
        name="8in_f10",
        aperture_mm=203,
        focal_ratio=10,
        sqm=20,
        eyepieces=[
            Eyepiece(40, 68),
            Eyepiece(25, 68),
            Eyepiece(18, 82),
            Eyepiece(12, 60),
            Eyepiece(8, 52),
        ],
    ),
    "6in_f8": InstrumentConfig(
        name="6in_f8",
        aperture_mm=152,
        focal_ratio=8,
        sqm=20,
        eyepieces=[
            Eyepiece(32, 52),
            Eyepiece(25, 68),
            Eyepiece(15, 68),
            Eyepiece(10, 60),
        ],
    ),
    "4in_f7": InstrumentConfig(
        name="4in_f7",
        aperture_mm=102,
        focal_ratio=7,
        sqm=20,
        eyepieces=[
            Eyepiece(32, 52),
            Eyepiece(20, 68),
            Eyepiece(13, 100),
            Eyepiece(8, 52),
        ],
    ),
    "seestar_s50": InstrumentConfig(
        name="seestar_s50",
        aperture_mm=50,
        focal_ratio=5,
        sqm=20,
        camera=Camera("imx462", pixel_size_um=2.9, res_x=1920, res_y=1080, limiting_mag=13.5),
    ),
}

COMMON_EYEPIECES = {
    "5mm/65":  Eyepiece(5, 65),
    "7mm/82":  Eyepiece(7, 82),
    "7mm/86":  Eyepiece(7, 86),
    "8mm/52":  Eyepiece(8, 52),
    "10mm/60": Eyepiece(10, 60),
    "12mm/60": Eyepiece(12, 60),
    "13mm/100": Eyepiece(13, 100),
    "15mm/68": Eyepiece(15, 68),
    "18mm/82": Eyepiece(18, 82),
    "20mm/68": Eyepiece(20, 68),
    "20mm/86": Eyepiece(20, 86),
    "25mm/68": Eyepiece(25, 68),
    "32mm/52": Eyepiece(32, 52),
    "38mm/70": Eyepiece(38, 70),
    "40mm/68": Eyepiece(40, 68),
}


def _show_eyepiece_table(inst):
    """Display derived search parameters for each eyepiece and/or camera."""
    scope_fl = inst.aperture_mm * inst.focal_ratio
    nelm = sqm_to_nelm(inst.sqm)
    print(f"\n  {inst.name}: {inst.aperture_mm:.0f}mm f/{inst.focal_ratio:.0f} "
          f"(FL {scope_fl:.0f}mm), SQM {inst.sqm}, NELM {nelm:.1f}")

    if inst.eyepieces:
        print(f"  {'EP':>6}  {'AFOV':>5}  {'Mag':>4}  {'TFOV':>6}  {'Exit':>5}  {'LM':>5}  {'Radius':>7}  {'Grid':>5}")
        print(f"  {'-'*6}  {'-'*5}  {'-'*4}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*5}")
        for ep in inst.eyepieces:
            mag = scope_fl / ep.focal_length_mm
            tfov = ep.afov_deg / mag
            exit_pupil = ep.focal_length_mm / inst.focal_ratio
            cfg = eyepiece_to_search_config(inst, ep)
            print(f"  {ep.focal_length_mm:5.0f}mm  {ep.afov_deg:4.0f}°  {mag:4.0f}x  {tfov:5.2f}°  "
                  f"{exit_pupil:4.1f}mm  {cfg.max_mag:5.1f}  {cfg.search_radius_deg:6.2f}°  {cfg.grid_step_deg:4.1f}°")

    if inst.camera:
        cam = inst.camera
        sensor_w = cam.res_x * cam.pixel_size_um / 1000
        sensor_h = cam.res_y * cam.pixel_size_um / 1000
        fov_w = 2 * math.degrees(math.atan(sensor_w / (2 * scope_fl)))
        fov_h = 2 * math.degrees(math.atan(sensor_h / (2 * scope_fl)))
        plate_scale = cam.pixel_size_um / scope_fl * 206.265
        cfg = camera_to_search_config(inst)
        print(f"  Camera: {cam.name} ({cam.res_x}x{cam.res_y}, {cam.pixel_size_um}µm)")
        print(f"    FOV {fov_w:.2f}° x {fov_h:.2f}°, plate scale {plate_scale:.2f}\"/px")
        print(f"    LM {cam.limiting_mag:.1f}  radius {cfg.search_radius_deg:.2f}°  grid {cfg.grid_step_deg:.1f}°")

    print()


def _input_float(prompt, default):
    val = input(f"  {prompt} [{default}]: ").strip()
    return float(val) if val else default



def _select_eyepieces():
    """Interactively select eyepieces."""
    print("\n  Common eyepieces (FL/AFOV):")
    keys = list(COMMON_EYEPIECES.keys())
    for i, k in enumerate(keys):
        ep = COMMON_EYEPIECES[k]
        print(f"    {i+1:2}. {ep.focal_length_mm:.0f}mm / {ep.afov_deg:.0f}° AFOV")
    print(f"    {len(keys)+1:2}. Custom")
    print()

    selected = []
    while True:
        choice = input("  Add eyepiece (number, or 'done'): ").strip().lower()
        if choice in ('done', 'd', ''):
            if selected:
                break
            print("    Need at least one eyepiece.")
            continue

        try:
            idx = int(choice)
        except ValueError:
            print("    Invalid input.")
            continue

        if idx == len(keys) + 1:
            fl = _input_float("Focal length (mm)", 13)
            afov = _input_float("Apparent FOV (deg)", 68)
            ep = Eyepiece(fl, afov)
        elif 1 <= idx <= len(keys):
            ep = COMMON_EYEPIECES[keys[idx - 1]]
        else:
            print("    Out of range.")
            continue

        selected.append(ep)
        print(f"    Added {ep.focal_length_mm:.0f}mm / {ep.afov_deg:.0f}° AFOV "
              f"({len(selected)} selected)")

    return selected


def _configure_interactive():
    """Full interactive configuration."""
    print("\n=== Telescope Configuration ===\n")
    print("  Presets available:")
    for name, inst in PRESETS.items():
        scope_fl = inst.aperture_mm * inst.focal_ratio
        parts = f"{inst.aperture_mm:.0f}mm f/{inst.focal_ratio:.0f} (FL {scope_fl:.0f}mm)"
        if inst.eyepieces:
            parts += f", {len(inst.eyepieces)} eyepieces"
        if inst.camera:
            parts += f", camera: {inst.camera.name}"
        print(f"    {name}: {parts}")
    print()

    choice = input("  Use preset (name) or 'custom': ").strip().lower()
    if choice in PRESETS:
        inst = PRESETS[choice]
        _show_eyepiece_table(inst)
        modify = input("  Modify eyepieces? [y/N]: ").strip().lower()
        if modify == 'y':
            eyepieces = _select_eyepieces()
            inst = InstrumentConfig(
                name=inst.name,
                aperture_mm=inst.aperture_mm,
                focal_ratio=inst.focal_ratio,
                sqm=inst.sqm,
                eyepieces=eyepieces,
            )
            _show_eyepiece_table(inst)
        return inst

    print("\n  --- Custom Telescope ---")
    aperture = _input_float("Aperture (mm)", 254)
    fratio = _input_float("Focal ratio (f/)", 5)
    sqm = _input_float("Sky quality (SQM)", 20)

    scope_fl = aperture * fratio
    name_default = f"{aperture:.0f}mm_f{fratio:.0f}"
    name = input(f"  Name [{name_default}]: ").strip() or name_default

    eyepieces = _select_eyepieces()

    inst = InstrumentConfig(
        name=name,
        aperture_mm=aperture,
        focal_ratio=fratio,
        sqm=sqm,
        eyepieces=eyepieces,
    )
    _show_eyepiece_table(inst)
    return inst


def main():
    parser = argparse.ArgumentParser(description="Configure telescope and run asterism search pipeline")
    parser.add_argument('--list-presets', action='store_true', help='Show available presets')
    parser.add_argument('--preset', type=str, help='Use a preset directly (e.g. 10in_f5)')
    parser.add_argument('--sqm', type=float, help='Override sky quality meter reading')
    parser.add_argument('--reports-only', action='store_true',
                        help='Skip pipeline, only generate reports from existing parquet files')
    parser.add_argument('--run-name', type=str, help='Output directory name (default: instrument name)')
    parser.add_argument('--yes', '-y', action='store_true', help='Overwrite existing output without asking')
    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable presets:\n")
        for name, inst in PRESETS.items():
            _show_eyepiece_table(inst)
        return

    if args.preset:
        if args.preset not in PRESETS:
            print(f"Unknown preset: {args.preset}")
            print(f"Available: {', '.join(PRESETS.keys())}")
            sys.exit(1)
        inst = PRESETS[args.preset]
    else:
        inst = _configure_interactive()

    if args.sqm:
        inst = replace(inst, sqm=args.sqm)

    _show_eyepiece_table(inst)

    confirm = input("  Proceed? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Aborted.")
        return

    run_name = args.run_name or inst.name

    if not args.reports_only:
        print("\n=== Running Pipeline ===\n")
        _run_pipeline(inst)

    print("\n=== Generating Reports ===\n")
    _run_reports(inst, run_name, overwrite=args.yes)

    print("\nDone!")


def _run_pipeline(inst):
    """Run the GPU pipeline for all search configs."""
    import time
    import polars as pl
    import torch
    from asterisms_py.core import (
        get_grid_points, reset_gpu_catalog,
        process_all_regions, add_to_result_and_save, dedup_results,
    )

    try:
        from hip_triangle import score_triangles_hip, HIP_AVAILABLE
    except ImportError:
        HIP_AVAILABLE = False
    try:
        from hip_square import score_squares_hip, HIP_SQUARE_AVAILABLE
    except ImportError:
        HIP_SQUARE_AVAILABLE = False

    hip_tri_fn = score_triangles_hip if HIP_AVAILABLE else None
    hip_sq_fn = score_squares_hip if HIP_SQUARE_AVAILABLE else None

    df = pl.read_parquet('support/gaia-12.parquet')
    print(f"Catalog: {len(df):,} stars")
    os.makedirs('results', exist_ok=True)

    search_configs = []
    for ep in inst.eyepieces:
        search_configs.append(eyepiece_to_search_config(inst, ep))
    if inst.camera:
        search_configs.append(camera_to_search_config(inst))

    runs = []
    for cfg in search_configs:
        runs.append((cfg, f"results/result_triangle2_{cfg.name}", "triangle", "3d"))
        runs.append((cfg, f"results/result_triangle2_{cfg.name}", "triangle", "2d"))

        sq_cfg = cfg
        if cfg.search_radius_deg <= 1.0:
            sq_cfg = replace(cfg, max_stars_per_region=150)
        elif cfg.search_radius_deg <= 3.0:
            sq_cfg = replace(cfg, max_stars_per_region=200)
        runs.append((sq_cfg, f"results/result_squareness_{cfg.name}", "square", "3d"))
        runs.append((sq_cfg, f"results/result_squareness_{cfg.name}", "square", "2d"))

        col_cfg = cfg
        if cfg.search_radius_deg >= 3.0:
            col_cfg = replace(cfg, grid_step_deg=2, max_stars_per_region=200)
        elif cfg.search_radius_deg <= 1.0:
            col_cfg = replace(cfg, max_stars_per_region=150)
        runs.append((col_cfg, f"results/result_collinear_{cfg.name}", "collinear", "2d"))

    total = len(runs)
    start_all = time.time()

    for i, (config, base_name, shape, scoring_mode) in enumerate(runs):
        suffix = "" if (scoring_mode == "3d" or shape == "collinear") else "_2d"
        result_filename = f"{base_name}{suffix}.parquet"

        if shape == 'collinear':
            skip_file = result_filename.replace(".parquet", "_smooth.parquet")
        else:
            skip_file = result_filename
        if os.path.exists(skip_file):
            print(f"[{i+1}/{total}] SKIP {skip_file} (exists)")
            continue

        grid_points = get_grid_points(config, -63, 63)
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{total}] {config.name} / {scoring_mode} / {shape}")
        print(f"  Grid: {len(grid_points):,} pts, radius={config.search_radius_deg}°, "
              f"max_mag={config.max_mag}, max_stars={config.max_stars_per_region}")

        reset_gpu_catalog()
        t0 = time.time()
        result = process_all_regions(
            grid_points, df, config=config, start=0, batch_size=100,
            mode=scoring_mode, hip_available=HIP_AVAILABLE or HIP_SQUARE_AVAILABLE,
            score_triangles_hip=hip_tri_fn, score_squares_hip=hip_sq_fn,
            shape=shape,
        )
        elapsed = time.time() - t0

        if shape == 'collinear':
            if isinstance(result, dict) and result:
                for score_key, df_k in result.items():
                    fn = result_filename.replace(".parquet", f"_{score_key}.parquet")
                    df_k = dedup_results(df_k)
                    df_k.sort("score").write_parquet(fn)
                    print(f"  Saved {fn} ({len(df_k):,} rows) [{elapsed:.0f}s]")
            else:
                print(f"  No results [{elapsed:.0f}s]")
        else:
            print(f"  Results: {len(result)} [{elapsed:.0f}s]")
            if not result.is_empty():
                add_to_result_and_save(pl.DataFrame(), result, result_filename)
                saved = pl.read_parquet(result_filename)
                saved = dedup_results(saved)
                saved.write_parquet(result_filename)
                print(f"  Saved {result_filename} ({len(saved):,} rows)")

    total_elapsed = time.time() - start_all
    print(f"\nPipeline done in {total_elapsed/60:.1f} minutes.")


def _instrument_to_dict(inst):
    """Serialize InstrumentConfig to a JSON-compatible dict."""
    d = {
        "name": inst.name,
        "aperture_mm": inst.aperture_mm,
        "focal_ratio": inst.focal_ratio,
        "sqm": inst.sqm,
        "eyepieces": [{"focal_length_mm": ep.focal_length_mm, "afov_deg": ep.afov_deg}
                      for ep in inst.eyepieces],
    }
    if inst.camera:
        d["camera"] = {
            "name": inst.camera.name,
            "pixel_size_um": inst.camera.pixel_size_um,
            "res_x": inst.camera.res_x,
            "res_y": inst.camera.res_y,
            "limiting_mag": inst.camera.limiting_mag,
        }
    return d


def _run_reports(inst, run_name, overwrite=False):
    """Generate reports, passing instrument config via temp JSON file."""
    inst_dict = _instrument_to_dict(inst)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(inst_dict, f)
        inst_json_path = f.name

    stdin_data = run_name + "\n"
    if overwrite:
        stdin_data += "y\n"

    try:
        result = subprocess.run(
            [sys.executable, "-u", "generate_reports.py",
             "--instrument-json", inst_json_path],
            input=stdin_data,
            text=True,
        )
    finally:
        os.unlink(inst_json_path)

    if result.returncode != 0:
        print(f"Report generation failed (exit code {result.returncode})")
        sys.exit(1)


if __name__ == '__main__':
    main()
