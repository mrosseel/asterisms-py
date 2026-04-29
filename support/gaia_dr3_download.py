#!/usr/bin/env python3
"""Download Gaia DR3 stars via TAP service and produce a parquet catalog.

Queries the ESA Gaia archive in declination strips to avoid timeouts.
Converts Gaia G-band magnitude to approximate Johnson V using Gaia colors.

Usage:
    python support/gaia_dr3_download.py --max-mag 15.0 --output support/gaia-15.parquet
    python support/gaia_dr3_download.py --max-mag 14.0 --output support/gaia-14.parquet
"""
import argparse
import io
import time

import numpy as np
import polars as pl
import requests


TAP_URL = "https://gea.esac.esa.int/tap-server/tap/sync"

# Declination strip size — smaller = less likely to timeout
DEC_STRIP = 5  # degrees


def gaia_g_to_johnson_v(phot_g_mean_mag, bp_rp):
    """Convert Gaia G magnitude to Johnson V using the Gaia DR3 relation.

    From Riello et al. 2021, Table 5.7 in Gaia DR3 docs:
    V - G = -0.02704 + 0.01424*C - 0.2156*C² + 0.01426*C³
    where C = BP-RP color index.

    For stars without BP-RP, assume V ≈ G (color term ~0 for solar-type).
    """
    c = np.where(np.isfinite(bp_rp), bp_rp, 0.0)
    v_minus_g = -0.02704 + 0.01424 * c + 0.2156 * c**2 + 0.01426 * c**3
    return phot_g_mean_mag + v_minus_g


def query_strip(dec_min, dec_max, max_gmag):
    """Query one declination strip from Gaia TAP."""
    # Query G mag slightly deeper than needed since V conversion shifts magnitudes
    query = f"""
    SELECT ra, dec, phot_g_mean_mag, bp_rp
    FROM gaiadr3.gaia_source
    WHERE dec >= {dec_min} AND dec < {dec_max}
      AND phot_g_mean_mag <= {max_gmag + 0.5}
      AND phot_g_mean_mag IS NOT NULL
      AND ra IS NOT NULL AND dec IS NOT NULL
    """

    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "QUERY": query,
    }

    for attempt in range(3):
        try:
            resp = requests.post(TAP_URL, data=params, timeout=600)
            resp.raise_for_status()

            df = pl.read_csv(io.StringIO(resp.text))
            return df
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                raise


def main():
    parser = argparse.ArgumentParser(description="Download Gaia DR3 catalog via TAP")
    parser.add_argument("--max-mag", type=float, default=15.0,
                        help="Maximum Johnson V magnitude")
    parser.add_argument("--output", type=str, default="support/gaia-15.parquet",
                        help="Output parquet path")
    parser.add_argument("--dec-min", type=float, default=-90.0,
                        help="Minimum declination")
    parser.add_argument("--dec-max", type=float, default=90.0,
                        help="Maximum declination")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from partial parquet (appends new strips)")
    args = parser.parse_args()

    all_dfs = []
    if args.resume and pl.scan_parquet(args.resume).collect().height > 0:
        print(f"Resuming from {args.resume}")
        existing = pl.read_parquet(args.resume)
        all_dfs.append(existing)
        # Find the max dec already downloaded
        existing_dec_max = existing["DEmdeg"].max()
        start_dec = int(np.floor(existing_dec_max / DEC_STRIP) * DEC_STRIP)
        print(f"  {len(existing):,} existing stars, resuming from dec={start_dec}")
    else:
        start_dec = int(args.dec_min)

    total_stars = sum(len(d) for d in all_dfs)

    for dec_lo in range(start_dec, int(args.dec_max), DEC_STRIP):
        dec_hi = min(dec_lo + DEC_STRIP, args.dec_max)
        print(f"Querying dec [{dec_lo}, {dec_hi})...", end=" ", flush=True)
        t0 = time.time()

        df = query_strip(dec_lo, dec_hi, args.max_mag)
        elapsed = time.time() - t0

        if df.is_empty():
            print(f"0 stars [{elapsed:.0f}s]")
            continue

        # Convert G mag to Johnson V
        g_mag = df["phot_g_mean_mag"].to_numpy()
        bp_rp = df["bp_rp"].to_numpy()
        v_mag = gaia_g_to_johnson_v(g_mag, bp_rp)

        # Filter to max V mag
        result = pl.DataFrame({
            "RAmdeg": df["ra"].cast(pl.Float32),
            "DEmdeg": df["dec"].cast(pl.Float32),
            "Vmag": pl.Series(v_mag.astype(np.float32)),
        }).filter(pl.col("Vmag") <= args.max_mag)

        all_dfs.append(result)
        total_stars += len(result)
        print(f"{len(result):,} stars (total: {total_stars:,}) [{elapsed:.0f}s]")

        # Save checkpoint every 10 strips
        if (dec_lo - start_dec) % (DEC_STRIP * 10) == 0 and dec_lo > start_dec:
            checkpoint = pl.concat(all_dfs)
            checkpoint.write_parquet(args.output + ".partial")
            print(f"  Checkpoint: {len(checkpoint):,} stars saved")

    # Final output
    if all_dfs:
        final = pl.concat(all_dfs)
        # Deduplicate on coordinates (within ~0.36 arcsec)
        final = final.with_columns([
            (pl.col("RAmdeg") * 10000).round().alias("_ra_key"),
            (pl.col("DEmdeg") * 10000).round().alias("_dec_key"),
        ]).unique(subset=["_ra_key", "_dec_key"], keep="first").drop(["_ra_key", "_dec_key"])

        final = final.sort(["DEmdeg", "RAmdeg"])
        final.write_parquet(args.output)
        print(f"\nDone: {len(final):,} stars written to {args.output}")
        print(f"Mag range: {final['Vmag'].min():.1f} to {final['Vmag'].max():.1f}")
    else:
        print("No stars found!")


if __name__ == "__main__":
    main()
