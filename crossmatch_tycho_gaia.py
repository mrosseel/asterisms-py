#!/usr/bin/env python3
"""Cross-match Tycho-2 and Gaia DR3 catalogs.

Propagates Tycho-2 positions (J2000.0) to the Gaia catalog epoch using
proper motions, then performs a spatial cross-match to find:
- Which Tycho-2 stars have a Gaia counterpart
- Which Tycho-2 stars are missing from Gaia
- Separation statistics and magnitude comparisons

Tycho-2: 2.5M stars, epoch J2000.0, pmRA/pmDE in mas/yr
Gaia (PiFinder tiles): positions at J2026.21, PM pre-applied, ~2.4" precision
"""

import struct
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from support.gaia_to_parquet import (
    iter_all_tiles, TILE_HEADER_FORMAT, TILE_HEADER_SIZE,
    STAR_RECORD_DTYPE, STAR_RECORD_SIZE,
)

# --- Configuration ---

TYCHO2_DAT = Path("support/tyc2.dat")
GAIA_CATALOG = Path("/home/mike/dev/amateur_astro/myPiFinder/wt-deepchart/astro_data/gaia_stars")
OUTPUT_DIR = Path("results/crossmatch")

# Cross-match radius in arcseconds
MATCH_RADIUS_ARCSEC = 5.0

# Tycho-2 mag limit (Gaia tiles go to mag 18, Tycho-2 completeness ~11.5)
TYCHO_MAX_MAG = 13.0

# Gaia extraction mag limit (generous margin above Tycho-2 range)
GAIA_MAX_MAG = 14.0


# --- Tycho-2 ingestion with proper motions ---

def read_tycho2_with_pm(filename: Path) -> pl.DataFrame:
    """Read Tycho-2 catalog keeping positions and proper motions."""
    labels = [
        "TYC123", "pflag", "RAmdeg", "DEmdeg", "pmRA", "pmDE",
        "e_RAmdeg", "e_DEmdeg", "e_pmRA", "e_pmDE", "EpRAm", "EpDEm",
        "Num", "q_RAmdeg", "q_DEmdeg", "q_pmRA", "q_pmDE", "BTmag",
        "e_BTmag", "VTmag", "e_VTmag", "prox", "TYC", "HIPCCDM",
        "RAdeg", "DEdeg", "EpRA-1990", "EpDE-1990", "e_RAdeg", "e_DEdeg",
        "posflg", "corr"
    ]

    df = pl.read_csv(
        str(filename), separator='|', has_header=False, new_columns=labels,
        dtypes={
            'RAmdeg': pl.Float64, 'DEmdeg': pl.Float64,
            'pmRA': pl.Float64, 'pmDE': pl.Float64,
            'BTmag': pl.Float32, 'VTmag': pl.Float32,
            'HIPCCDM': pl.Utf8, 'pflag': pl.Utf8,
        }
    )

    # Compute visual magnitude: V = VT - 0.090*(BT-VT)
    df = df.with_columns(
        pl.col("BTmag").fill_null(pl.col("VTmag")),
    ).with_columns(
        pl.col("VTmag").fill_null(pl.col("BTmag")),
    ).with_columns(
        (pl.col("VTmag") - 0.090 * (pl.col("BTmag") - pl.col("VTmag"))).alias("Vmag"),
    )

    # Keep only what we need
    df = df.select([
        "TYC123", "pflag", "RAmdeg", "DEmdeg", "pmRA", "pmDE", "Vmag", "BTmag", "VTmag",
    ])

    # Drop stars with no position
    df = df.filter(pl.col("RAmdeg").is_not_null() & pl.col("DEmdeg").is_not_null())

    return df


def propagate_positions(df: pl.DataFrame, target_epoch: float, source_epoch: float = 2000.0) -> pl.DataFrame:
    """Propagate Tycho-2 positions using proper motions.

    pmRA is already in mas/yr * cos(dec), pmDE in mas/yr.
    """
    dt_yr = target_epoch - source_epoch

    df = df.with_columns(
        # Stars with no PM: keep original position
        pl.col("pmRA").fill_null(0.0),
        pl.col("pmDE").fill_null(0.0),
    ).with_columns(
        # pmRA is already cos(dec)-corrected, so divide by cos(dec) to get delta_RA
        (pl.col("RAmdeg") + pl.col("pmRA") * dt_yr / 3_600_000.0
         / pl.col("DEmdeg").radians().cos()).alias("RA_prop"),
        (pl.col("DEmdeg") + pl.col("pmDE") * dt_yr / 3_600_000.0).alias("Dec_prop"),
    )

    return df


# --- Gaia extraction from PiFinder tiles ---

def extract_gaia_stars(catalog_path: Path, max_mag: float) -> pl.DataFrame:
    """Extract Gaia stars from PiFinder HEALPix tiles."""
    import healpy as hp
    import json

    with open(catalog_path / "metadata.json") as f:
        metadata = json.load(f)

    nside = metadata["nside"]
    catalog_epoch = metadata["catalog_epoch"]
    print(f"Gaia catalog: epoch={catalog_epoch}, nside={nside}")
    print(f"Extracting stars to mag {max_mag}")

    pixel_size_deg = np.sqrt(hp.nside2pixarea(nside, degrees=True))
    max_offset_arcsec = pixel_size_deg * 3600.0 * 0.75

    all_ras = []
    all_decs = []
    all_mags = []

    bands = sorted(catalog_path.glob("mag_*"))
    for band_dir in bands:
        parts = band_dir.name.split("_")
        band_min = int(parts[1])
        if band_min >= max_mag:
            continue

        tiles_file = band_dir / "tiles.bin"
        index_file = band_dir / "index.bin"
        if not tiles_file.exists() or not index_file.exists():
            continue

        tile_count = 0
        band_stars = 0

        for tile_id, data in iter_all_tiles(index_file, tiles_file):
            if len(data) < TILE_HEADER_SIZE:
                continue

            healpix_pixel, num_stars = struct.unpack(TILE_HEADER_FORMAT, data[:TILE_HEADER_SIZE])
            star_data = data[TILE_HEADER_SIZE:]
            num_stars = min(num_stars, len(star_data) // STAR_RECORD_SIZE)
            if num_stars == 0:
                continue

            records = np.frombuffer(star_data, dtype=STAR_RECORD_DTYPE, count=num_stars)

            pixel_ra, pixel_dec = hp.pix2ang(nside, healpix_pixel, lonlat=True)

            ra_off = (records["ra_offset"] / 127.5 - 1.0) * max_offset_arcsec
            dec_off = (records["dec_offset"] / 127.5 - 1.0) * max_offset_arcsec

            decs = pixel_dec + dec_off / 3600.0
            ras = pixel_ra + ra_off / 3600.0 / np.cos(np.radians(decs))
            mags = records["mag"] / 10.0

            mask = mags <= max_mag
            all_ras.append(ras[mask])
            all_decs.append(decs[mask])
            all_mags.append(mags[mask])
            tile_count += 1
            band_stars += mask.sum()

        print(f"  {band_dir.name}: {tile_count} tiles, {band_stars:,} stars")

    ras = np.concatenate(all_ras)
    decs = np.concatenate(all_decs)
    mags = np.concatenate(all_mags)

    print(f"Total Gaia stars extracted: {len(ras):,}")

    return pl.DataFrame({
        "gaia_ra": ras.astype(np.float64),
        "gaia_dec": decs.astype(np.float64),
        "gaia_mag": mags.astype(np.float32),
    }), catalog_epoch


# --- Cross-matching ---

def radec_to_cartesian(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """Convert RA/Dec (degrees) to unit vectors on the celestial sphere."""
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack([x, y, z])


def angular_separation_arcsec(chord_distance: np.ndarray) -> np.ndarray:
    """Convert chord distance between unit vectors to angular separation in arcsec."""
    # chord = 2 * sin(theta/2), theta = 2 * arcsin(chord/2)
    return np.degrees(2.0 * np.arcsin(np.clip(chord_distance / 2.0, 0, 1))) * 3600.0


def crossmatch(tycho_ra: np.ndarray, tycho_dec: np.ndarray,
               gaia_ra: np.ndarray, gaia_dec: np.ndarray,
               radius_arcsec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cross-match using KD-tree on unit sphere.

    Returns:
        matched_tycho_idx: indices into Tycho arrays that matched
        matched_gaia_idx: corresponding Gaia indices
        separations_arcsec: angular separations
    """
    print(f"Building Gaia KD-tree ({len(gaia_ra):,} stars)...")
    t0 = time.time()
    gaia_xyz = radec_to_cartesian(gaia_ra, gaia_dec)
    tree = cKDTree(gaia_xyz)
    print(f"  KD-tree built in {time.time() - t0:.1f}s")

    print(f"Querying Tycho-2 ({len(tycho_ra):,} stars) with radius {radius_arcsec}\"...")
    t0 = time.time()
    tycho_xyz = radec_to_cartesian(tycho_ra, tycho_dec)

    # Convert arcsec radius to chord distance
    radius_rad = np.radians(radius_arcsec / 3600.0)
    chord_radius = 2.0 * np.sin(radius_rad / 2.0)

    distances, indices = tree.query(tycho_xyz, k=1, distance_upper_bound=chord_radius)
    print(f"  Query completed in {time.time() - t0:.1f}s")

    # Filter matches (unmatched get distance=inf, index=tree.n)
    matched_mask = np.isfinite(distances)
    matched_tycho_idx = np.where(matched_mask)[0]
    matched_gaia_idx = indices[matched_mask]
    separations = angular_separation_arcsec(distances[matched_mask])

    return matched_tycho_idx, matched_gaia_idx, separations


# --- Analysis ---

def analyze_results(tycho_df: pl.DataFrame, gaia_df: pl.DataFrame,
                    matched_tycho_idx: np.ndarray, matched_gaia_idx: np.ndarray,
                    separations: np.ndarray):
    """Print analysis of cross-match results."""
    n_tycho = len(tycho_df)
    n_matched = len(matched_tycho_idx)
    n_unmatched = n_tycho - n_matched

    print("\n" + "=" * 60)
    print("CROSS-MATCH RESULTS")
    print("=" * 60)
    print(f"Tycho-2 stars:      {n_tycho:>10,}")
    print(f"Gaia stars:         {len(gaia_df):>10,}")
    print(f"Matched:            {n_matched:>10,}  ({100 * n_matched / n_tycho:.2f}%)")
    print(f"Unmatched (Tycho):  {n_unmatched:>10,}  ({100 * n_unmatched / n_tycho:.2f}%)")

    print(f"\nSeparation statistics (matched stars):")
    print(f"  Median: {np.median(separations):.2f}\"")
    print(f"  Mean:   {np.mean(separations):.2f}\"")
    print(f"  Std:    {np.std(separations):.2f}\"")
    print(f"  Max:    {np.max(separations):.2f}\"")
    print(f"  <1\":    {np.sum(separations < 1.0):,} ({100 * np.sum(separations < 1.0) / n_matched:.1f}%)")
    print(f"  <2\":    {np.sum(separations < 2.0):,} ({100 * np.sum(separations < 2.0) / n_matched:.1f}%)")
    print(f"  <3\":    {np.sum(separations < 3.0):,} ({100 * np.sum(separations < 3.0) / n_matched:.1f}%)")

    # Magnitude comparison
    tycho_mags = tycho_df["Vmag"].to_numpy()[matched_tycho_idx]
    gaia_mags = gaia_df["gaia_mag"].to_numpy()[matched_gaia_idx]
    mag_diff = tycho_mags - gaia_mags
    print(f"\nMagnitude differences (Tycho Vmag - Gaia Vmag):")
    print(f"  Median: {np.nanmedian(mag_diff):+.3f}")
    print(f"  Mean:   {np.nanmean(mag_diff):+.3f}")
    print(f"  Std:    {np.nanstd(mag_diff):.3f}")

    # Unmatched star analysis
    unmatched_mask = np.ones(n_tycho, dtype=bool)
    unmatched_mask[matched_tycho_idx] = False
    unmatched_mags = tycho_df["Vmag"].to_numpy()[unmatched_mask]
    unmatched_pflags = tycho_df["pflag"].to_numpy()[unmatched_mask]
    unmatched_pmra = tycho_df["pmRA"].to_numpy()[unmatched_mask]

    print(f"\nUnmatched Tycho-2 star analysis:")
    print(f"  Magnitude distribution:")
    for lo, hi in [(0, 6), (6, 8), (8, 10), (10, 11), (11, 12), (12, 13), (13, 99)]:
        count = np.sum((unmatched_mags >= lo) & (unmatched_mags < hi))
        pct = 100 * count / n_unmatched if n_unmatched > 0 else 0
        print(f"    mag {lo:>2}-{hi:>2}: {count:>6,}  ({pct:.1f}%)")

    # pflag: 'X' means no mean position (problematic stars)
    n_pflag_x = np.sum(unmatched_pflags == 'X')
    print(f"  With pflag='X' (no mean position): {n_pflag_x:,}")

    # High proper motion unmatched (might indicate PM propagation issues)
    has_pm = ~np.isnan(unmatched_pmra.astype(float))
    if has_pm.any():
        pm_total = np.sqrt(
            np.nan_to_num(unmatched_pmra.astype(float)) ** 2 +
            np.nan_to_num(tycho_df["pmDE"].to_numpy()[unmatched_mask].astype(float)) ** 2
        )
        high_pm = np.sum(pm_total > 100)  # > 100 mas/yr
        print(f"  With PM > 100 mas/yr: {high_pm:,}")

    return unmatched_mask


def save_results(tycho_df: pl.DataFrame, gaia_df: pl.DataFrame,
                 matched_tycho_idx: np.ndarray, matched_gaia_idx: np.ndarray,
                 separations: np.ndarray, unmatched_mask: np.ndarray):
    """Save cross-match results to parquet files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Matched pairs
    matched_df = pl.DataFrame({
        "tycho_ra": tycho_df["RA_prop"].to_numpy()[matched_tycho_idx],
        "tycho_dec": tycho_df["Dec_prop"].to_numpy()[matched_tycho_idx],
        "tycho_ra_j2000": tycho_df["RAmdeg"].to_numpy()[matched_tycho_idx],
        "tycho_dec_j2000": tycho_df["DEmdeg"].to_numpy()[matched_tycho_idx],
        "tycho_vmag": tycho_df["Vmag"].to_numpy()[matched_tycho_idx],
        "tycho_id": tycho_df["TYC123"].to_numpy()[matched_tycho_idx],
        "gaia_ra": gaia_df["gaia_ra"].to_numpy()[matched_gaia_idx],
        "gaia_dec": gaia_df["gaia_dec"].to_numpy()[matched_gaia_idx],
        "gaia_mag": gaia_df["gaia_mag"].to_numpy()[matched_gaia_idx],
        "separation_arcsec": separations.astype(np.float32),
    })
    matched_path = OUTPUT_DIR / "matched_pairs.parquet"
    matched_df.write_parquet(matched_path)
    print(f"\nSaved {len(matched_df):,} matched pairs to {matched_path}")

    # Unmatched Tycho-2 stars
    unmatched_df = tycho_df.filter(pl.Series(unmatched_mask))
    unmatched_path = OUTPUT_DIR / "unmatched_tycho.parquet"
    unmatched_df.write_parquet(unmatched_path)
    print(f"Saved {len(unmatched_df):,} unmatched Tycho-2 stars to {unmatched_path}")


# --- Main ---

def main():
    print("=" * 60)
    print("Tycho-2 / Gaia DR3 Cross-Match")
    print("=" * 60)

    # 1. Read Tycho-2 with proper motions
    print(f"\n[1/4] Reading Tycho-2 catalog from {TYCHO2_DAT}...")
    t0 = time.time()
    tycho = read_tycho2_with_pm(TYCHO2_DAT)
    print(f"  {len(tycho):,} stars read in {time.time() - t0:.1f}s")
    print(f"  Stars with proper motion: {tycho.filter(pl.col('pmRA').is_not_null()).height:,}")

    # Filter by magnitude
    tycho = tycho.filter(pl.col("Vmag") <= TYCHO_MAX_MAG)
    print(f"  After mag <= {TYCHO_MAX_MAG} filter: {len(tycho):,}")

    # 2. Extract Gaia
    print(f"\n[2/4] Extracting Gaia stars from PiFinder tiles...")
    t0 = time.time()
    gaia, gaia_epoch = extract_gaia_stars(GAIA_CATALOG, GAIA_MAX_MAG)
    print(f"  Extracted in {time.time() - t0:.1f}s")

    # 3. Propagate Tycho-2 positions to Gaia epoch
    # Parse epoch string like "J2026.21"
    target_epoch = float(gaia_epoch.replace("J", ""))
    print(f"\n[3/4] Propagating Tycho-2 positions: J2000.0 → J{target_epoch}...")
    tycho = propagate_positions(tycho, target_epoch)

    # Check for problematic propagated positions (near poles, wrapping)
    tycho = tycho.with_columns(
        (pl.col("RA_prop") % 360.0).alias("RA_prop"),
        pl.col("Dec_prop").clip(-90.0, 90.0).alias("Dec_prop"),
    )

    # 4. Cross-match
    print(f"\n[4/4] Cross-matching...")
    tycho_ra = tycho["RA_prop"].to_numpy()
    tycho_dec = tycho["Dec_prop"].to_numpy()
    gaia_ra = gaia["gaia_ra"].to_numpy()
    gaia_dec = gaia["gaia_dec"].to_numpy()

    matched_tycho_idx, matched_gaia_idx, separations = crossmatch(
        tycho_ra, tycho_dec, gaia_ra, gaia_dec, MATCH_RADIUS_ARCSEC
    )

    # Analysis
    unmatched_mask = analyze_results(
        tycho, gaia, matched_tycho_idx, matched_gaia_idx, separations
    )

    # Save
    save_results(tycho, gaia, matched_tycho_idx, matched_gaia_idx, separations, unmatched_mask)

    # Also try without PM propagation for comparison
    print("\n" + "-" * 60)
    print("COMPARISON: matching WITHOUT proper motion propagation")
    print("-" * 60)
    matched_nopm, _, sep_nopm = crossmatch(
        tycho["RAmdeg"].to_numpy(), tycho["DEmdeg"].to_numpy(),
        gaia_ra, gaia_dec, MATCH_RADIUS_ARCSEC
    )
    n_nopm = len(matched_nopm)
    n_pm = len(matched_tycho_idx)
    print(f"  Without PM: {n_nopm:,} matched ({100 * n_nopm / len(tycho):.2f}%)")
    print(f"  With PM:    {n_pm:,} matched ({100 * n_pm / len(tycho):.2f}%)")
    print(f"  Improvement: +{n_pm - n_nopm:,} stars")
    if len(sep_nopm) > 0:
        print(f"  Median separation without PM: {np.median(sep_nopm):.2f}\"")
        print(f"  Median separation with PM:    {np.median(separations):.2f}\"")


if __name__ == "__main__":
    main()
