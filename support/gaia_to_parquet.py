#!/usr/bin/env python3
"""Extract Gaia stars from HEALPix tile files to a parquet file.

Produces a parquet with columns (RAmdeg, DEmdeg, Vmag) matching
the Tycho-2 parquet schema used by py-asterisms.

Usage:
    python support/gaia_to_parquet.py \
        --catalog /path/to/gaia_stars \
        --output support/gaia-12.parquet \
        --max-mag 12.0
"""

import argparse
import json
import mmap
import struct
from pathlib import Path

import numpy as np
import polars as pl


TILE_HEADER_FORMAT = "<IH"
TILE_HEADER_SIZE = 6
STAR_RECORD_DTYPE = np.dtype([
    ("ra_offset", "u1"),
    ("dec_offset", "u1"),
    ("mag", "u1"),
])
STAR_RECORD_SIZE = 3


def iter_all_tiles(index_path: Path, tiles_path: Path):
    """Iterate all tiles, yielding (tile_id, tile_data) pairs."""
    with open(index_path, "rb") as f:
        index_data = f.read()

    version, num_tiles, num_runs = struct.unpack_from("<III", index_data, 0)
    if version != 3:
        raise ValueError(f"Unsupported index version: {version}")

    # Read run directory
    runs = []
    pos = 12
    for _ in range(num_runs):
        start_tile, data_offset = struct.unpack_from("<IQ", index_data, pos)
        runs.append((start_tile, data_offset))
        pos += 12

    with open(tiles_path, "rb") as f:
        tiles_mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    for start_tile, data_offset in runs:
        # data_offset is absolute offset into the index file's run data section
        # Run data: length(2) + offset_base(8) + sizes(2 * length)
        run_length, offset_base = struct.unpack_from("<HQ", index_data, data_offset)
        sizes_data = index_data[data_offset + 10: data_offset + 10 + run_length * 2]
        sizes = struct.unpack(f"<{run_length}H", sizes_data)

        tile_offset = offset_base
        for i, size in enumerate(sizes):
            if size > 0:
                tile_id = start_tile + i
                yield tile_id, tiles_mm[tile_offset:tile_offset + size]
            tile_offset += size

    tiles_mm.close()


def extract_band(band_dir: Path, nside: int, max_mag: float):
    """Extract all stars from a magnitude band directory."""
    import healpy as hp

    tiles_file = band_dir / "tiles.bin"
    index_file = band_dir / "index.bin"

    if not tiles_file.exists() or not index_file.exists():
        print(f"  Skipping {band_dir.name}: missing files")
        return np.empty((0, 3), dtype=np.float32)

    all_ras = []
    all_decs = []
    all_mags = []
    tile_count = 0

    pixel_size_deg = np.sqrt(hp.nside2pixarea(nside, degrees=True))
    max_offset_arcsec = pixel_size_deg * 3600.0 * 0.75

    for tile_id, data in iter_all_tiles(index_file, tiles_file):
        if len(data) < TILE_HEADER_SIZE:
            continue

        healpix_pixel, num_stars = struct.unpack(
            TILE_HEADER_FORMAT, data[:TILE_HEADER_SIZE]
        )
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

    print(f"  {band_dir.name}: {tile_count} tiles read")

    if not all_ras:
        return np.empty((0, 3), dtype=np.float32)

    return np.column_stack([
        np.concatenate(all_ras),
        np.concatenate(all_decs),
        np.concatenate(all_mags),
    ]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract Gaia stars to parquet")
    parser.add_argument("--catalog", required=True, help="Path to gaia_stars directory")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--max-mag", type=float, default=12.0, help="Maximum magnitude")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    with open(catalog_path / "metadata.json") as f:
        metadata = json.load(f)

    nside = metadata["nside"]
    print(f"Catalog: {metadata['source']} epoch={metadata['catalog_epoch']} nside={nside}")
    print(f"Extracting stars to mag {args.max_mag}")

    bands = sorted(catalog_path.glob("mag_*"))
    all_stars = []
    for band_dir in bands:
        parts = band_dir.name.split("_")
        band_min = int(parts[1])
        if band_min >= args.max_mag:
            continue
        stars = extract_band(band_dir, nside, args.max_mag)
        if len(stars) > 0:
            all_stars.append(stars)
            print(f"    -> {len(stars):,} stars")

    combined = np.vstack(all_stars)
    print(f"\nTotal: {len(combined):,} stars")

    df = pl.DataFrame({
        "RAmdeg": combined[:, 0],
        "DEmdeg": combined[:, 1],
        "Vmag": combined[:, 2],
    })
    df.write_parquet(args.output)
    print(f"Written to {args.output}")
    print(f"Mag range: {combined[:, 2].min():.1f} to {combined[:, 2].max():.1f}")


if __name__ == "__main__":
    main()
