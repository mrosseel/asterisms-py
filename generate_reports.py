"""Generate PDF reports and PiFinder observing lists from parquet results."""
import os
import json
import datetime
import numpy as np
import polars as pl
import torch
import pandas

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

from skyfield.api import Star, load, position_of_radec, load_constellation_map, load_constellation_names
from skyfield.units import Angle
from skyfield.data import hipparcos, stellarium
from skyfield.projections import build_stereographic_projection

from asterisms_py.core import (
    stars_for_center_and_radius, triangle_extent_deg, chain_extent_deg,
)

# --- Seasonal constants for 51 deg N ---
MIN_DEC = -19.0  # Dec > -19 to reach ~20 deg altitude at 51N

SEASONS = {
    'spring': (120, 240),   # RA 8h-16h (degrees)
    'summer': (210, 330),   # RA 14h-22h
    'fall':   (300, 60),    # RA 20h-4h (wraps around 0)
    'winter': (30, 150),    # RA 2h-10h
}

# Load shared astronomy data
print("Loading astronomy data...")
ts = load.timescale()
t = ts.utc(2023, 7, 27)
eph = load('de421.bsp')
earth = eph['earth']

with load.open(hipparcos.URL) as f:
    hip_stars = hipparcos.load_dataframe(f)

url = ('https://raw.githubusercontent.com/Stellarium/stellarium/master'
       '/skycultures/modern_st/constellationship.fab')
with load.open(url) as f:
    constellations = stellarium.parse_constellations(f)

edges = [edge for name, edges in constellations for edge in edges]
edges_star1 = [star1 for star1, star2 in edges]
edges_star2 = [star2 for star1, star2 in edges]

constellation_at = load_constellation_map()
con_full_names = dict(load_constellation_names())
con_full_names['CVn'] = 'Canes Venatici'

dftycho = pl.read_parquet('support/tyc2.parquet')
print(f"Catalog: {len(dftycho):,} stars ({list(dftycho.columns)})")


# --- B-V color mapping ---

def bv_to_rgb(bv):
    """Piecewise linear B-V index to RGB color."""
    # Anchor points: (B-V, R, G, B)
    anchors = [
        (-0.4, 0.6, 0.7, 1.0),   # deep blue-white
        (-0.1, 0.8, 0.85, 1.0),  # blue-white
        ( 0.0, 1.0, 1.0,  1.0),  # white
        ( 0.3, 1.0, 1.0,  0.8),  # yellow-white
        ( 0.6, 1.0, 0.95, 0.5),  # yellow
        ( 1.0, 1.0, 0.7,  0.3),  # orange
        ( 1.5, 1.0, 0.5,  0.3),  # red-orange
        ( 2.0, 0.9, 0.4,  0.3),  # red
    ]
    if bv <= anchors[0][0]:
        return (anchors[0][1], anchors[0][2], anchors[0][3])
    if bv >= anchors[-1][0]:
        return (anchors[-1][1], anchors[-1][2], anchors[-1][3])
    for i in range(len(anchors) - 1):
        if anchors[i][0] <= bv <= anchors[i+1][0]:
            frac = (bv - anchors[i][0]) / (anchors[i+1][0] - anchors[i][0])
            r = anchors[i][1] + frac * (anchors[i+1][1] - anchors[i][1])
            g = anchors[i][2] + frac * (anchors[i+1][2] - anchors[i][2])
            b = anchors[i][3] + frac * (anchors[i+1][3] - anchors[i][3])
            return (r, g, b)
    return (1.0, 1.0, 1.0)


def _compute_star_colors(bt_arr, vt_arr):
    """Compute RGB colors from BTmag and VTmag arrays."""
    bv = bt_arr - vt_arr
    return [bv_to_rgb(b) for b in bv]


# --- Enrichment functions ---

def get_mean(value):
    points = torch.tensor(value.to_list())
    return points, points.mean(dim=0)

def _get_center(value):
    _, mean = get_mean(value)
    ra = Angle(degrees=mean[0].item())
    return position_of_radec(ra._hours, mean[1])


def compute_isolation(focus_stars, center_coord, extent_deg, max_mag):
    """Count field stars within 1.5x extent that are brighter than the faintest asterism star."""
    faintest = max(s[2] for s in focus_stars)
    radius = extent_deg * 1.5
    cos_dec = max(0.1, np.cos(np.radians(center_coord[1])))
    query_radius = radius / cos_dec
    nearby = stars_for_center_and_radius(dftycho, center_coord, query_radius, faintest)

    # Exclude the 3 asterism stars by RA/Dec proximity
    focus_ras = [s[0] for s in focus_stars]
    focus_decs = [s[1] for s in focus_stars]
    tol = 0.001  # degrees
    mask = pl.lit(True)
    for fra, fdec in zip(focus_ras, focus_decs):
        is_this_star = (
            (pl.col("RAmdeg") - fra).abs() < tol
        ) & (
            (pl.col("DEmdeg") - fdec).abs() < tol
        )
        mask = mask & ~is_this_star
    nearby_filtered = nearby.filter(mask)
    return len(nearby_filtered)


def enrich_results(head, max_mag=15.0):
    head = head.with_columns(
        pl.col("stars").map_elements(lambda v: Angle(degrees=get_mean(v)[1][0].item()).degrees, return_dtype=pl.Float64).alias("Ra"),
        pl.col("stars").map_elements(lambda v: Angle(degrees=get_mean(v)[1][0].item())._hours, return_dtype=pl.Float64).alias("Rah"),
        pl.col("stars").map_elements(lambda v: Angle(degrees=get_mean(v)[1][1].item()).degrees, return_dtype=pl.Float64).alias("Dec"),
        pl.col("stars").map_elements(lambda v: Angle(degrees=get_mean(v)[1][0].item()).hstr(places=4, warn=False), return_dtype=pl.Utf8).alias("Rah_full"),
        pl.col("stars").map_elements(lambda v: Angle(degrees=get_mean(v)[1][1].item()).dstr(places=4, warn=False), return_dtype=pl.Utf8).alias("Dec_full"),
        pl.col("stars").map_elements(lambda v: constellation_at(_get_center(v)), return_dtype=pl.Utf8).alias("CON"),
    )
    head = head.with_columns(pl.col("CON").replace(con_full_names).alias("CONSTELLATION"))

    # Compute isolation per row
    isolations = []
    for row in head.iter_rows(named=True):
        focus_stars = row['stars']
        center = (row['Ra'], row['Dec'])
        star_pts = torch.tensor(focus_stars)
        extent = chain_extent_deg(star_pts) if len(focus_stars) > 3 else triangle_extent_deg(star_pts)
        iso = compute_isolation(focus_stars, center, extent, max_mag)
        isolations.append(iso)
    head = head.with_columns(pl.Series("isolation", isolations))
    return head


def _lookup_bv_for_focus_stars(focus_stars):
    """Look up B-V for focus stars (from result parquet, only RA/Dec/Vmag) by catalog match."""
    colors = []
    for s in focus_stars:
        ra, dec = s[0], s[1]
        tol = 0.005
        match = dftycho.filter(
            ((pl.col("RAmdeg") - ra).abs() < tol) &
            ((pl.col("DEmdeg") - dec).abs() < tol)
        )
        if len(match) > 0 and "BTmag" in match.columns and "VTmag" in match.columns:
            row = match.row(0, named=True)
            bt, vt = row["BTmag"], row["VTmag"]
            if bt is not None and vt is not None:
                colors.append(bv_to_rgb(bt - vt))
            else:
                colors.append((1.0, 1.0, 1.0))
        else:
            colors.append((1.0, 1.0, 1.0))
    return colors


def draw_points(ax, t, earth, projection, region_df, limiting_magnitude):
    """Draw stars from a Polars DataFrame with B-V colors."""
    if isinstance(region_df, pl.DataFrame):
        ra_arr = region_df["RAmdeg"].to_numpy()
        dec_arr = region_df["DEmdeg"].to_numpy()
        mag_arr = region_df["Vmag"].to_numpy()
        if "BTmag" in region_df.columns and "VTmag" in region_df.columns:
            bt_arr = region_df["BTmag"].fill_null(0.6).to_numpy()
            vt_arr = region_df["VTmag"].fill_null(0.0).to_numpy()
            colors = _compute_star_colors(bt_arr, vt_arr)
        else:
            colors = [(1.0, 1.0, 1.0)] * len(region_df)
    else:
        # Legacy: list of tuples (ra_deg, dec_deg, mag)
        ra_arr = np.array([s[0] for s in region_df])
        dec_arr = np.array([s[1] for s in region_df])
        mag_arr = np.array([s[2] for s in region_df])
        colors = [(1.0, 1.0, 1.0)] * len(region_df)

    fs = pandas.DataFrame({
        'ra_hours': ra_arr / 15.0,
        'dec_degrees': dec_arr,
        'magnitude': mag_arr,
        'epoch_year': 2000.0,
    })
    star_positions = earth.at(t).observe(Star.from_dataframe(fs))
    fs['x'], fs['y'] = projection(star_positions)
    marker_size = (0.5 + limiting_magnitude - fs['magnitude']) ** 2.0
    ax.scatter(fs['x'], fs['y'], s=marker_size, c=colors, edgecolors='none', zorder=5)


def draw_focus_stars(ax, t, earth, projection, focus_stars, limiting_magnitude):
    """Draw focus stars with B-V colors and return projected DataFrame for triangle edges."""
    colors = _lookup_bv_for_focus_stars(focus_stars)
    ra_arr = np.array([s[0] for s in focus_stars])
    dec_arr = np.array([s[1] for s in focus_stars])
    mag_arr = np.array([s[2] for s in focus_stars])

    fs = pandas.DataFrame({
        'ra_hours': ra_arr / 15.0,
        'dec_degrees': dec_arr,
        'magnitude': mag_arr,
        'epoch_year': 2000.0,
    })
    star_positions = earth.at(t).observe(Star.from_dataframe(fs))
    fs['x'], fs['y'] = projection(star_positions)
    marker_size = (0.5 + limiting_magnitude - fs['magnitude']) ** 2.0
    ax.scatter(fs['x'], fs['y'], s=marker_size, c=colors, edgecolors='none', zorder=10)
    return fs


def _draw_shape_edges(ax, fs_focus, chart_mag, n_stars, shape='triangle', data_limit=1.0):
    """Draw polygon edges with gaps near each vertex so stars remain visible."""
    xs = list(fs_focus['x'])
    ys = list(fs_focus['y'])
    mags = list(fs_focus['magnitude'])

    # For squares, sort vertices by angle from centroid to get proper polygon order
    if n_stars == 4 and shape == 'square':
        cx = np.mean(xs)
        cy = np.mean(ys)
        angles = [np.arctan2(y - cy, x - cx) for x, y in zip(xs, ys)]
        order = np.argsort(angles)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        mags = [mags[i] for i in order]

    # Convert marker radius from points to data coords.
    # Chart subplot is ~400 points wide spanning 2*data_limit data units.
    pts_per_data = 400.0 / (2.0 * data_limit)
    fixed_gap_data = 0.0002  # fixed gap beyond marker edge

    # Collinear chains: open path (no wrap-around); stars already sorted along the line
    n_edges = n_stars - 1 if shape == 'collinear' else n_stars

    for i in range(n_edges):
        j = (i + 1) if shape == 'collinear' else (i + 1) % n_stars
        x0, y0, m0 = xs[i], ys[i], mags[i]
        x1, y1, m1 = xs[j], ys[j], mags[j]
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        if length < 1e-10:
            continue
        # Marker radius in data coords + fixed gap
        r0 = (0.5 + chart_mag - m0) / (2.0 * pts_per_data) + fixed_gap_data
        r1 = (0.5 + chart_mag - m1) / (2.0 * pts_per_data) + fixed_gap_data
        frac0 = min(r0 / length, 0.3)
        frac1 = min(r1 / length, 0.3)
        sx = x0 + dx * frac0
        sy = y0 + dy * frac0
        ex = x1 - dx * frac1
        ey = y1 - dy * frac1
        ax.plot([sx, ex], [sy, ey], color='#ffdd44', linewidth=0.8, alpha=0.4, zorder=9,
                clip_on=True)


def _shape_edge_vertices(stars, shape):
    """Return ordered RA/Dec vertices forming the shape edges (closed or open polyline).

    For triangles: closed polygon 0→1→2→0
    For squares: closed polygon sorted by angle from centroid
    For collinear: open path 0→1→...→N-1
    """
    pts = [(s[0], s[1]) for s in stars]
    if shape == 'square' and len(pts) == 4:
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        angles = [np.arctan2(p[1] - cy, p[0] - cx) for p in pts]
        order = np.argsort(angles)
        pts = [pts[i] for i in order]
    verts = [[round(ra, 6), round(dec, 6)] for ra, dec in pts]
    if shape in ('triangle', 'square'):
        verts.append(verts[0])
    return verts


def generate_pifinder_list(enriched_results, outdir, name, shape='triangle'):
    list_dir = os.path.join(outdir, 'lists')
    os.makedirs(list_dir, exist_ok=True)
    list_path = os.path.join(list_dir, f'{name}.pifinder')

    objects = []
    for ast_idx, row in enumerate(enriched_results.iter_rows(named=True)):
        con = row['CONSTELLATION']
        score = row['score']
        ra_deg = row['Ra']
        dec_deg = row['Dec']
        star_mags = [s[2] for s in row['stars']]
        star_pts = torch.tensor(row['stars'])
        extent = chain_extent_deg(star_pts) if len(row['stars']) > 3 else triangle_extent_deg(star_pts)
        edge_verts = _shape_edge_vertices(row['stars'], shape)
        extent_arcsec = extent * 3600.0

        notes_parts = [f"score={score:.6f}"]
        if 'tilt' in row:
            notes_parts.append(f"tilt={row['tilt']:.0f}°")
        notes_parts.append(f"mags={','.join(f'{m:.1f}' for m in star_mags)}")

        obj = {
            "name": f"{con} asterism #{ast_idx+1}",
            "obj_type": "Ast",
            "ra": round(ra_deg, 6),
            "dec": round(dec_deg, 6),
            "mag": {
                "mags": [round(m, 2) for m in star_mags],
                "filter_mag": round(min(star_mags), 2),
            },
            "const": row['CON'],
            "extents": {"shape": edge_verts},
            "notes": ", ".join(notes_parts),
        }
        objects.append(obj)

    pifinder_data = {
        "version": 1,
        "name": name,
        "description": f"Asterism search results ({shape})",
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "objects": objects,
    }

    with open(list_path, 'w') as f:
        json.dump(pifinder_data, f, indent=2)

    print(f"  Wrote {len(objects)} entries to {list_path}")


def generate_pdf(enriched_results, outdir, name, title, max_mag=15.0, fov_override=None, shape='triangle'):
    fov_suffix = f'_fov{fov_override:.1f}deg' if fov_override else ''
    pdf_path = os.path.join(outdir, f'{name}{fov_suffix}.pdf')
    limiting_magnitude = max_mag

    star_positions = earth.at(t).observe(Star.from_dataframe(hip_stars))

    with PdfPages(pdf_path) as pdf:
        # --- Summary page (dark mode) ---
        fig, ax = plt.subplots(figsize=[8.5, 11])
        fig.patch.set_facecolor('#111111')
        ax.set_facecolor('#111111')
        ax.axis('off')
        title_extra = f' - {fov_override}\u00b0 FOV' if fov_override else ''
        ax.text(0.5, 0.92, f'Asterism Search Results', fontsize=20, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes, color='white')
        ax.text(0.5, 0.87, f'{title}{title_extra}',
                fontsize=12, ha='center', va='top', transform=ax.transAxes, color='#cccccc')

        has_isolation = 'isolation' in enriched_results.columns
        has_tilt = 'tilt' in enriched_results.columns
        table_data = []
        col_labels = ['Rank', 'Score', 'Constellation', 'RA', 'Dec']
        if has_tilt:
            col_labels.append('Tilt')
        if has_isolation:
            col_labels.append('Isolation')
        for idx, row in enumerate(enriched_results.iter_rows(named=True)):
            row_data = [
                str(idx + 1), f"{row['score']:.6f}", row['CONSTELLATION'],
                row['Rah_full'], row['Dec_full'],
            ]
            if has_tilt:
                row_data.append(f"{row['tilt']:.0f}°")
            if has_isolation:
                row_data.append(str(row['isolation']))
            table_data.append(row_data)

        col_widths = [0.06, 0.12, 0.18, 0.22, 0.22]
        if has_tilt:
            col_widths.append(0.08)
        if has_isolation:
            col_widths.append(0.10)
        table = ax.table(cellText=table_data,
                         colLabels=col_labels,
                         loc='center', cellLoc='center',
                         colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)
        # Style table cells for dark mode
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_facecolor('#1a1a1a')
            cell.set_text_props(color='white')
            cell.set_edgecolor('#333333')
            if row_idx == 0:
                cell.set_facecolor('#2a2a2a')
                cell.set_text_props(color='#ffdd44', fontweight='bold')

        shape_word = {'triangle': 'triangle', 'square': 'square', 'collinear': 'chain'}.get(shape, shape)
        ax.text(0.5, 0.15, f'Yellow edges = asterism {shape_word},  Stars = B-V color,  Blue = constellations',
                fontsize=11, ha='center', va='top', transform=ax.transAxes, fontfamily='monospace', color='#cccccc')
        pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)

        # --- Per-asterism pages ---
        for idx, entry in enumerate(enriched_results.iter_rows(named=True)):
            try:
                focus_stars = entry['stars']
                center_coord = (entry['Ra'], entry['Dec'])
                star_pts = torch.tensor(focus_stars)
                extent = chain_extent_deg(star_pts) if len(focus_stars) > 3 else triangle_extent_deg(star_pts)
                field_of_view_degrees = fov_override if fov_override else max(0.5, extent * 1.8)
                cos_dec = max(0.1, np.cos(np.radians(center_coord[1])))
                region_radius = field_of_view_degrees * 1.5 / cos_dec

                chart_mag = min(limiting_magnitude, 10.0)
                region_stars = stars_for_center_and_radius(dftycho, center_coord, region_radius, chart_mag)
                if len(focus_stars) == 0 or len(region_stars) == 0:
                    continue

                looking_at = Star(ra_hours=entry['Rah'], dec_degrees=entry['Dec'])
                center = earth.at(t).observe(looking_at)
                projection = build_stereographic_projection(center)

                hip_stars_copy = hip_stars.copy()
                hip_stars_copy['x'], hip_stars_copy['y'] = projection(star_positions)

                xy1 = hip_stars_copy[['x', 'y']].loc[edges_star1].values
                xy2 = hip_stars_copy[['x', 'y']].loc[edges_star2].values
                lines_xy = np.rollaxis(np.array([xy1, xy2]), 1)

                fig = plt.figure(figsize=[8.5, 11])
                fig.patch.set_facecolor('#111111')
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2.5], hspace=0.25)

                # --- Finder chart (dark mode) ---
                ax_chart = fig.add_subplot(gs[0])
                ax_chart.set_facecolor('black')
                ax_chart.add_collection(LineCollection(lines_xy, colors='#4488ff', alpha=0.6, linewidths=0.8))
                angle = np.pi - field_of_view_degrees / 360.0 * np.pi
                limit = np.sin(angle) / (1.0 - np.cos(angle))

                draw_points(ax_chart, t, earth, projection, region_stars, chart_mag)
                fs_focus = draw_focus_stars(ax_chart, t, earth, projection, focus_stars, chart_mag)

                # Shape edges with gaps near stars
                _draw_shape_edges(ax_chart, fs_focus, chart_mag, n_stars=len(focus_stars), shape=shape, data_limit=limit)
                ax_chart.set_xlim(-limit, limit)
                ax_chart.set_ylim(-limit, limit)
                ax_chart.xaxis.set_visible(False)
                ax_chart.yaxis.set_visible(False)
                ax_chart.set_aspect(1.0)
                ax_chart.set_title(f"#{idx+1}: {entry['CONSTELLATION']}", fontsize=16,
                                   fontweight='bold', color='white')
                ax_chart.tick_params(colors='white')
                for spine in ax_chart.spines.values():
                    spine.set_edgecolor('#333333')

                # Magnitude scale legend
                mag_steps = list(range(0, int(chart_mag) + 1, 2))
                if mag_steps[-1] != int(chart_mag):
                    mag_steps.append(int(chart_mag))
                for m in mag_steps:
                    s = (0.5 + chart_mag - m) ** 2.0
                    ax_chart.scatter([], [], s=s, c='white', label=f'mag {m}')
                legend = ax_chart.legend(loc='upper right', title='Magnitude', fontsize=8,
                               title_fontsize=9, framealpha=0.7, facecolor='#1a1a1a',
                               edgecolor='#333333', labelcolor='white')
                legend.get_title().set_color('white')

                # --- Info panel (dark mode) ---
                ax_info = fig.add_subplot(gs[1])
                ax_info.set_facecolor('#111111')
                ax_info.axis('off')

                star_mags = [s[2] for s in focus_stars]
                n_stars = len(focus_stars)

                ra_rad = torch.deg2rad(star_pts[:, 0])
                dec_rad = torch.deg2rad(star_pts[:, 1])
                sep_labels = []
                for i in range(n_stars):
                    for j in range(i+1, n_stars):
                        cos_sep = (torch.sin(dec_rad[i]) * torch.sin(dec_rad[j]) +
                                   torch.cos(dec_rad[i]) * torch.cos(dec_rad[j]) * torch.cos(ra_rad[i] - ra_rad[j]))
                        sep_deg = torch.rad2deg(torch.acos(torch.clamp(cos_sep, -1, 1))).item()
                        sep_labels.append(f"{sep_deg:.3f}")

                score = entry['score']
                if shape == 'collinear':
                    # Detect scoring mode from the report name
                    if 'msd' in scoring:
                        if score < 0.000025:
                            quality = "Excellent"
                        elif score < 0.0004:
                            quality = "Very good"
                        elif score < 0.0025:
                            quality = "Good"
                        else:
                            quality = "Fair"
                        score_desc = f"Score: {score:.6f}  ({quality} collinear fit)\n  MSD perp. deviation + 0.3*spacing CV (0 = perfect)."
                    elif 'smooth_mag' in scoring:
                        if score < 0.01:
                            quality = "Excellent"
                        elif score < 0.05:
                            quality = "Very good"
                        elif score < 0.15:
                            quality = "Good"
                        else:
                            quality = "Fair"
                        score_desc = f"Score: {score:.6f}  ({quality} smooth+mag fit)\n  Turn std + 0.2*CV + 0.2*mag_gradient_std."
                    elif 'smooth' in scoring:
                        if score < 0.01:
                            quality = "Excellent"
                        elif score < 0.05:
                            quality = "Very good"
                        elif score < 0.15:
                            quality = "Good"
                        else:
                            quality = "Fair"
                        score_desc = f"Score: {score:.6f}  ({quality} smooth fit)\n  Turn std + 0.2*CV + 0.3*mag_CV (0 = perfect curve/line)."
                    else:
                        if score < 0.005:
                            quality = "Excellent"
                        elif score < 0.02:
                            quality = "Very good"
                        elif score < 0.05:
                            quality = "Good"
                        else:
                            quality = "Fair"
                        score_desc = f"Score: {score:.6f}  ({quality} collinear fit)\n  RMS perp. deviation + 0.3*spacing CV (0 = perfect)."
                    # Show sequential spacings along the chain
                    seq_seps = []
                    for si in range(n_stars - 1):
                        cos_s = (torch.sin(dec_rad[si]) * torch.sin(dec_rad[si+1]) +
                                 torch.cos(dec_rad[si]) * torch.cos(dec_rad[si+1]) * torch.cos(ra_rad[si] - ra_rad[si+1]))
                        seq_seps.append(f"{torch.rad2deg(torch.acos(torch.clamp(cos_s, -1, 1))).item():.3f}")
                    # Wrap spacings across multiple lines for long chains
                    per_line = 10
                    spacing_lines = []
                    for chunk_start in range(0, len(seq_seps), per_line):
                        chunk = seq_seps[chunk_start:chunk_start + per_line]
                        spacing_lines.append(',  '.join(chunk))
                    sep_line = f"  {n_stars}-star chain,  spacings (deg):\n    " + '\n    '.join(spacing_lines)
                elif shape == 'triangle':
                    if score < 0.001:
                        quality = "Excellent"
                    elif score < 0.01:
                        quality = "Very good"
                    elif score < 0.05:
                        quality = "Good"
                    elif score < 0.1:
                        quality = "Fair"
                    else:
                        quality = "Poor"
                    score_desc = f"Score: {score:.6f}  ({quality} equilateral fit)\n  CV of three side lengths (0 = perfect)."
                    sep_line = f"  Side separations: {',  '.join(sep_labels)} deg"
                else:
                    if score < 0.01:
                        quality = "Excellent"
                    elif score < 0.05:
                        quality = "Very good"
                    elif score < 0.1:
                        quality = "Good"
                    else:
                        quality = "Fair"
                    score_desc = f"Score: {score:.6f}  ({quality} square fit)\n  Squareness deviation (0 = perfect)."
                    # Sort: 4 shortest = sides, 2 longest = diagonals
                    sep_vals = sorted([float(s) for s in sep_labels])
                    sides_str = ',  '.join(f"{v:.3f}" for v in sep_vals[:4])
                    diag_str = ',  '.join(f"{v:.3f}" for v in sep_vals[4:])
                    sep_line = f"  Sides: {sides_str} deg\n  Diagonals: {diag_str} deg"

                tilt_line = ""
                if 'tilt' in entry and shape in ('triangle', 'square'):
                    tilt_val = entry['tilt']
                    tilt_line = f"Tilt: {tilt_val:.1f}°  (0° = face-on, 90° = edge-on)\n"

                isolation_line = ""
                if has_isolation:
                    isolation_line = f"Isolation: {entry['isolation']} nearby field stars\n"

                # Star table: show for short chains, skip for long ones
                MAX_TABLE_STARS = 8
                show_table = n_stars <= MAX_TABLE_STARS

                star_table_data = []
                if show_table:
                    for si, s in enumerate(focus_stars):
                        s_ra = Angle(degrees=s[0])
                        s_dec = Angle(degrees=s[1])
                        star_table_data.append([
                            str(si + 1),
                            s_ra.hstr(places=0, warn=False),
                            s_dec.dstr(places=0, warn=False),
                            f"{s[2]:.2f}",
                        ])

                # For long chains, add mag range and endpoint coords to info text
                extra_info = ""
                if not show_table:
                    min_mag = min(star_mags)
                    max_mag_val = max(star_mags)
                    s0_ra = Angle(degrees=focus_stars[0][0])
                    s0_dec = Angle(degrees=focus_stars[0][1])
                    sN_ra = Angle(degrees=focus_stars[-1][0])
                    sN_dec = Angle(degrees=focus_stars[-1][1])
                    extra_info = (
                        f"\nMagnitude range: {min_mag:.1f} - {max_mag_val:.1f}\n"
                        f"Start: {s0_ra.hstr(places=0, warn=False)}  {s0_dec.dstr(places=0, warn=False)}  (mag {focus_stars[0][2]:.1f})\n"
                        f"End:   {sN_ra.hstr(places=0, warn=False)}  {sN_dec.dstr(places=0, warn=False)}  (mag {focus_stars[-1][2]:.1f})"
                    )

                info_text = (
                    f"Constellation: {entry['CONSTELLATION']}\n"
                    f"Center: RA {entry['Rah_full']},  Dec {entry['Dec_full']}\n"
                    f"\n"
                    f"{score_desc}\n"
                    f"\n"
                    f"Angular extent: {extent:.3f} deg\n"
                    f"{sep_line}\n"
                    f"\n"
                    f"{tilt_line}"
                    f"{isolation_line}"
                    f"Chart: {field_of_view_degrees:.1f} deg FOV,  search limiting mag {limiting_magnitude:.0f}"
                    f"{extra_info}"
                )

                ax_info.text(0.05, 0.98, info_text, transform=ax_info.transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            color='white',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a', edgecolor='#333333', alpha=0.9))

                if show_table:
                    tbl_rows = len(star_table_data)
                    tbl_height = 0.07 * (tbl_rows + 1)
                    star_tbl = ax_info.table(cellText=star_table_data,
                                             colLabels=['Star', 'RA', 'Dec', 'Mag'],
                                             cellLoc='center',
                                             bbox=[0.05, -0.05, 0.9, tbl_height],
                                             colWidths=[0.08, 0.35, 0.35, 0.12])
                    star_tbl.auto_set_font_size(False)
                    star_tbl.set_fontsize(10)
                    for (row_idx, col_idx), cell in star_tbl.get_celld().items():
                        cell.set_facecolor('#1a1a1a')
                        cell.set_text_props(color='white')
                        cell.set_edgecolor('#333333')
                        if row_idx == 0:
                            cell.set_facecolor('#2a2a2a')
                            cell.set_text_props(color='#ffdd44', fontweight='bold')

                pdf.savefig(fig, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close(fig)
            except Exception as e:
                print(f"  Error on entry {idx}: {e}")
                import traceback
                traceback.print_exc()
                plt.close('all')

    print(f"  Wrote {pdf_path}")


# --- Seasonal filtering ---

def _ra_in_range(ra_deg, ra_min, ra_max):
    """Check if RA is in range, handling wrap-around at 360."""
    if ra_min < ra_max:
        return ra_min <= ra_deg <= ra_max
    else:
        return ra_deg >= ra_min or ra_deg <= ra_max


def filter_visible_51n(enriched):
    """Filter to asterisms visible from 51N (Dec > MIN_DEC)."""
    return enriched.filter(pl.col("Dec") > MIN_DEC)


def filter_seasonal(enriched, season):
    """Filter enriched results to a specific season by RA range."""
    ra_min, ra_max = SEASONS[season]
    mask = [_ra_in_range(row['Ra'], ra_min, ra_max) for row in enriched.iter_rows(named=True)]
    return enriched.filter(pl.Series(mask))


# --- Output directory structure ---
# reports/<scope>/<shape>[/<scoring>]/
REPORT_DIR = 'reports'

# (subdir_path, scoring, parquet_file, max_mag, display_name, shape)
MODES = [
    # Telescopic
    ("telescopic/triangles", "3d", "result_triangle2.parquet",        15.0, "Telescopic triangles 3D", "triangle"),
    ("telescopic/triangles", "2d", "result_triangle2_2d.parquet",     15.0, "Telescopic triangles 2D", "triangle"),
    ("telescopic/squares",   "3d", "result_squareness.parquet",       15.0, "Telescopic squares 3D", "square"),
    ("telescopic/squares",   "2d", "result_squareness_2d.parquet",    15.0, "Telescopic squares 2D", "square"),
    ("telescopic/collinear/smooth",     "smooth",     "result_collinear_smooth.parquet",     15.0, "Telescopic collinear (smooth)", "collinear"),
    ("telescopic/collinear/smooth_mag", "smooth_mag", "result_collinear_smooth_mag.parquet", 15.0, "Telescopic collinear (smooth+mag)", "collinear"),
    # Binocular
    ("binocular/triangles", "3d", "result_triangle2_bino.parquet",    11.0, "Binocular triangles 3D", "triangle"),
    ("binocular/triangles", "2d", "result_triangle2_bino_2d.parquet", 11.0, "Binocular triangles 2D", "triangle"),
    ("binocular/collinear/smooth",     "smooth",     "result_collinear_bino_smooth.parquet",     11.0, "Binocular collinear (smooth)", "collinear"),
    ("binocular/collinear/smooth_mag", "smooth_mag", "result_collinear_bino_smooth_mag.parquet", 11.0, "Binocular collinear (smooth+mag)", "collinear"),
    # Naked eye
    ("naked_eye/triangles", "3d", "result_triangle2_naked.parquet",    6.0, "Naked eye triangles 3D", "triangle"),
    ("naked_eye/triangles", "2d", "result_triangle2_naked_2d.parquet", 6.0, "Naked eye triangles 2D", "triangle"),
]

if __name__ == '__main__':
 for subdir_path, scoring, filename, max_mag, display_name, shape in MODES:
    if not os.path.exists(filename):
        print(f"Skipping {subdir_path}/{scoring}: {filename} not found")
        continue

    outdir = os.path.join(REPORT_DIR, subdir_path)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Generating: {subdir_path}/{scoring} (limiting mag {max_mag})")

    df = pl.read_parquet(filename)
    print(f"  {len(df)} results")

    # Filter 3D triangle/square results by tilt: keep 10-60° (meaningful depth, not edge-on)
    if scoring == '3d' and shape in ('triangle', 'square') and 'tilt' in df.columns:
        before = len(df)
        df = df.filter((pl.col("tilt") >= 10) & (pl.col("tilt") <= 60))
        print(f"  Tilt filter (10°-60°): {before} → {len(df)} results")

    if shape == 'collinear' and 'chain_len' in df.columns:
        # Group chain lengths into bins of ~20 pages each
        chain_lengths = sorted(df["chain_len"].unique().to_list())
        MAX_PAGES = 20

        # First pass: count visible results per chain length
        visible_per_clen = {}
        for clen in chain_lengths:
            clen_df = df.filter(pl.col("chain_len") == clen).sort("score")
            enriched = enrich_results(clen_df.head(100), max_mag=max_mag)
            vis = filter_visible_51n(enriched)
            if len(vis) > 0:
                visible_per_clen[clen] = vis
                print(f"  {clen}-star: {len(vis)} visible")
            else:
                print(f"  {clen}-star: 0 visible")

        # Build bins greedily: accumulate chain lengths until ~MAX_PAGES visible
        bins = []
        cur_clens = []
        cur_count = 0
        for clen in sorted(visible_per_clen.keys()):
            n_vis = len(visible_per_clen[clen])
            # Single chain length already fills a bin? Give it its own.
            if n_vis >= MAX_PAGES:
                if cur_clens:
                    bins.append(cur_clens)
                bins.append([clen])
                cur_clens = []
                cur_count = 0
            else:
                cur_clens.append(clen)
                cur_count += n_vis
                if cur_count >= MAX_PAGES:
                    bins.append(cur_clens)
                    cur_clens = []
                    cur_count = 0
        if cur_clens:
            bins.append(cur_clens)

        # Generate one PDF per bin
        for bin_clens in bins:
            if len(bin_clens) == 1:
                clen = bin_clens[0]
                bin_name = f"{scoring}_{clen}star"
                bin_display = f"{display_name} ({clen}-star)"
            else:
                bin_name = f"{scoring}_{bin_clens[0]}-{bin_clens[-1]}star"
                bin_display = f"{display_name} ({bin_clens[0]}-{bin_clens[-1]}-star)"

            # Combine visible results from all chain lengths in this bin
            bin_visible = pl.concat([visible_per_clen[c] for c in bin_clens])
            bin_visible = bin_visible.sort("score").head(MAX_PAGES)

            print(f"\n  --- {bin_display}: {len(bin_visible)} pages ---")
            generate_pdf(bin_visible, outdir, f"{bin_name}_51N",
                         f"Top {len(bin_visible)} visible from 51\u00b0N - {bin_display}",
                         max_mag=max_mag, shape=shape)
            # Strip scoring prefix from pifinder list names — directory makes it clear
            list_bin_name = bin_name.removeprefix(f"{scoring}_")
            generate_pifinder_list(bin_visible, outdir, f"{list_bin_name}_51N", shape=shape)

            for season in SEASONS:
                seasonal = filter_seasonal(bin_visible, season).head(MAX_PAGES)
                if len(seasonal) > 0:
                    generate_pifinder_list(seasonal, outdir, f"{list_bin_name}_{season}_51N", shape=shape)

        # Also generate an "all lengths" combined report (top 10 overall)
        all_enriched = enrich_results(df.sort("score").head(100), max_mag=max_mag)
        visible = filter_visible_51n(all_enriched)
        if len(visible) > 0:
            top10 = visible.head(10)
            generate_pdf(top10, outdir, f"{scoring}_51N",
                         f"Top 10 visible from 51\u00b0N - {display_name}", max_mag=max_mag, shape=shape)
            generate_pifinder_list(top10, outdir, "top10_51N", shape=shape)
        continue

    # Non-collinear: existing logic
    all_enriched = enrich_results(df.sort("score").head(100), max_mag=max_mag)
    visible = filter_visible_51n(all_enriched)

    if len(visible) == 0:
        print(f"  No asterisms visible from 51N for {subdir_path}/{scoring}")
        continue

    top10 = visible.head(10)
    generate_pdf(top10, outdir, f"{scoring}_51N",
                 f"Top 10 visible from 51\u00b0N - {display_name}", max_mag=max_mag, shape=shape)
    generate_pifinder_list(top10, outdir, f"{scoring}_51N", shape=shape)
    print(f"  51N: {len(top10)} asterisms")

    for season in SEASONS:
        seasonal = filter_seasonal(visible, season).head(10)
        if len(seasonal) > 0:
            generate_pifinder_list(seasonal, outdir, f"{scoring}_{season}_51N", shape=shape)
        else:
            print(f"  No {season} asterisms for {subdir_path}/{scoring}")

    if subdir_path.startswith("telescopic") and shape == "triangle":
        for fov in [1.0, 0.5]:
            def _extent_fits_fov(v, fov=fov):
                pts = torch.tensor(v.to_list())
                ext = chain_extent_deg(pts) if len(pts) > 3 else triangle_extent_deg(pts)
                return ext <= fov
            fov_filtered = visible.filter(
                pl.col("stars").map_elements(
                    _extent_fits_fov,
                    return_dtype=pl.Boolean
                )
            ).head(10)
            if len(fov_filtered) == 0:
                print(f"  No 51N asterisms fit within {fov}\u00b0 FOV, skipping")
                continue
            generate_pdf(fov_filtered, outdir, f"{scoring}_51N", display_name,
                         max_mag=max_mag, fov_override=fov, shape=shape)
            generate_pifinder_list(fov_filtered, outdir, f"{scoring}_fov{fov:.1f}deg_51N", shape=shape)

 print("\n\nDone! All reports generated.")
