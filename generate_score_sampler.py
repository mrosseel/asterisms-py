"""Generate diagnostic PDFs sampling chains at regular score intervals.

Helps determine quality cutoffs for each scoring mode.
"""
import numpy as np
import polars as pl
import torch
import pandas
import os

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from skyfield.api import Star, load, position_of_radec, load_constellation_map, load_constellation_names
from skyfield.units import Angle
from skyfield.data import hipparcos, stellarium
from skyfield.projections import build_stereographic_projection

from asterisms_py.core import stars_for_center_and_radius, chain_extent_deg

from generate_reports import (
    draw_points, draw_focus_stars, _draw_shape_edges,
    earth, t, hip_stars, edges_star1, edges_star2,
    dftycho, constellation_at, con_full_names,
)

N_SAMPLES = 25

CONFIGS = [
    ("results/result_collinear_rms.parquet", "telescopic/collinear/rms", "collinear_rms_sampler", "RMS", 15.0),
    ("results/result_collinear_msd.parquet", "telescopic/collinear/msd", "collinear_msd_sampler", "MSD", 15.0),
    ("results/result_collinear_smooth.parquet", "telescopic/collinear/smooth", "collinear_smooth_sampler", "Smooth", 15.0),
    ("results/result_collinear_smooth_mag.parquet", "telescopic/collinear/smooth_mag", "collinear_smooth_mag_sampler", "Smooth+Mag", 15.0),
]

star_positions = earth.at(t).observe(Star.from_dataframe(hip_stars))
edges = list(zip(edges_star1, edges_star2))


def generate_sampler_pdf(parquet_file, subdir, pdf_name, mode_label, max_mag):
    df = pl.read_parquet(parquet_file)
    # Filter to 8-star chains — long enough to see quality clearly
    df = df.filter(pl.col("chain_len") == 8).sort("score")
    if len(df) == 0:
        print(f"  No 8-star chains in {parquet_file}")
        return

    scores = df["score"].to_numpy()
    s_min, s_max = scores[0], scores[-1]
    step = (s_max - s_min) / N_SAMPLES
    bin_edges = np.arange(s_min, s_max, step)

    # Pick one representative chain per bin
    samples = []
    for b in bin_edges:
        candidates = df.filter(
            (pl.col("score") >= b) & (pl.col("score") < b + step)
        )
        if len(candidates) > 0:
            # Pick the median-scored one
            mid = len(candidates) // 2
            samples.append(candidates.sort("score").row(mid, named=True))

    if not samples:
        print(f"  No samples found for {mode_label}")
        return

    print(f"  {mode_label}: {len(samples)} samples from {samples[0]['score']:.4f} to {samples[-1]['score']:.4f}")

    pdf_dir = os.path.join("reports", subdir)
    pdf_path = os.path.join(pdf_dir, f"{pdf_name}.pdf")
    os.makedirs(pdf_dir, exist_ok=True)

    with PdfPages(pdf_path) as pdf:
        for idx, entry in enumerate(samples):
            try:
                focus_stars = entry['stars']
                star_pts = torch.tensor(focus_stars)
                mean_pt = star_pts.mean(dim=0)
                center_ra = mean_pt[0].item()
                center_dec = mean_pt[1].item()
                extent = chain_extent_deg(star_pts)
                fov = max(0.5, extent * 1.8)
                cos_dec = max(0.1, np.cos(np.radians(center_dec)))
                region_radius = fov * 1.5 / cos_dec
                chart_mag = min(max_mag, 10.0)

                region_stars = stars_for_center_and_radius(
                    dftycho, (center_ra, center_dec), region_radius, chart_mag
                )
                if len(region_stars) == 0:
                    continue

                ra_h = center_ra / 15.0
                looking_at = Star(ra_hours=ra_h, dec_degrees=center_dec)
                center = earth.at(t).observe(looking_at)
                projection = build_stereographic_projection(center)

                hip_copy = hip_stars.copy()
                hip_copy['x'], hip_copy['y'] = projection(star_positions)
                xy1 = hip_copy[['x', 'y']].loc[edges_star1].values
                xy2 = hip_copy[['x', 'y']].loc[edges_star2].values
                lines_xy = np.rollaxis(np.array([xy1, xy2]), 1)

                angle = np.pi - fov / 360.0 * np.pi
                limit = np.sin(angle) / (1.0 - np.cos(angle))

                # Constellation
                ra_angle = Angle(degrees=center_ra)
                pos = position_of_radec(ra_angle._hours, center_dec)
                con_abbr = constellation_at(pos)
                con_name = con_full_names.get(con_abbr, con_abbr)

                fig = plt.figure(figsize=[8.5, 6])
                fig.patch.set_facecolor('#111111')

                ax = fig.add_subplot(111)
                ax.set_facecolor('black')
                ax.add_collection(LineCollection(lines_xy, colors='#4488ff', alpha=0.6, linewidths=0.8))
                draw_points(ax, t, earth, projection, region_stars, chart_mag)
                fs_focus = draw_focus_stars(ax, t, earth, projection, focus_stars, chart_mag)
                _draw_shape_edges(ax, fs_focus, chart_mag, n_stars=len(focus_stars),
                                  shape='collinear', data_limit=limit)

                ax.set_xlim(-limit, limit)
                ax.set_ylim(-limit, limit)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_aspect(1.0)

                star_mags = [f"{s[2]:.1f}" for s in focus_stars]
                ax.set_title(
                    f"Sample {idx+1}/{len(samples)}  |  {mode_label} score: {entry['score']:.6f}  |  "
                    f"{con_name}  |  mags: {', '.join(star_mags)}",
                    fontsize=10, color='white', pad=8,
                )
                for spine in ax.spines.values():
                    spine.set_edgecolor('#333333')

                pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close(fig)
            except Exception as e:
                print(f"    Error on sample {idx}: {e}")
                plt.close('all')

    print(f"  Wrote {pdf_path}")


for parquet_file, subdir, pdf_name, mode_label, max_mag in CONFIGS:
    if not os.path.exists(parquet_file):
        print(f"Skipping {mode_label}: {parquet_file} not found")
        continue
    print(f"\nGenerating {mode_label} sampler...")
    generate_sampler_pdf(parquet_file, subdir, pdf_name, mode_label, max_mag)

print("\nDone!")
