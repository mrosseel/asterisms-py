"""Generate PDF reports and PiFinder observing lists from parquet results."""
import json
import os
import sys
import datetime
import traceback
from io import BytesIO

import numpy as np
import polars as pl
import requests
import torch
import pandas
from PIL import Image, ImageOps

import matplotlib
matplotlib.use('Agg')
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
    dedup_results, camera_fov,
    DEFAULT_INSTRUMENT, instrument_search_configs,
    eyepiece_to_search_config, camera_to_search_config,
    InstrumentConfig, Eyepiece, Camera,
    score_bright_isolated_chains,
    asterism_id, assign_asterism_ids,
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

dftycho = pl.read_parquet('support/gaia-12.parquet')
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


# --- POSS sky survey image support ---

SKYVIEW_URL = (
    "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
    "?Survey=digitized+sky+survey&position={ra},{dec}"
    "&Return=JPEG&size={size}&pixels=1024"
)

POSS_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache", "poss")

_poss_session = None

def _get_poss_session():
    global _poss_session
    if _poss_session is None:
        _poss_session = requests.Session()
        _poss_session.headers["User-Agent"] = "asterisms-py/1.0"
    return _poss_session


def _poss_cache_path(ra_deg, dec_deg, fov_deg):
    """Build cache filename from coordinates and FOV, rounded to avoid near-duplicates."""
    ra_key = round(ra_deg, 3)
    dec_key = round(dec_deg, 3)
    fov_key = round(fov_deg, 2)
    return os.path.join(POSS_CACHE_DIR, f"poss_{ra_key}_{dec_key}_{fov_key}.jpg")


def fetch_poss_image(ra_deg, dec_deg, fov_deg):
    """Fetch a POSS/DSS image, using local cache when available."""
    cache_file = _poss_cache_path(ra_deg, dec_deg, fov_deg)

    if os.path.exists(cache_file):
        try:
            return Image.open(cache_file).convert('L')
        except Exception:
            pass  # corrupted cache file, re-fetch

    url = SKYVIEW_URL.format(ra=ra_deg, dec=dec_deg, size=fov_deg)
    try:
        resp = _get_poss_session().get(url, timeout=60)
        if resp.status_code != 200:
            return None
        img = Image.open(BytesIO(resp.content)).convert('L')
        img = ImageOps.autocontrast(img, cutoff=2)

        os.makedirs(POSS_CACHE_DIR, exist_ok=True)
        img.save(cache_file, "JPEG", quality=90)
        return img
    except Exception as e:
        print(f"  POSS fetch failed: {e}")
        return None


def _draw_circle_arc(ax, star_ras, star_decs, n_arc_points=100):
    """Draw a best-fit circle arc through stars as a dashed line."""
    if len(star_ras) < 3:
        return
    # Fit circle to star positions using least-squares
    x = np.array(star_ras)
    y = np.array(star_decs)
    # Algebraic circle fit: (x-cx)^2 + (y-cy)^2 = r^2
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    try:
        result = np.linalg.lstsq(A, b, rcond=None)
        params = result[0]
    except np.linalg.LinAlgError:
        return
    cx, cy = params[0], params[1]
    r_sq = params[2] + cx ** 2 + cy ** 2
    if r_sq <= 0:
        return
    r = np.sqrt(r_sq)
    # Angular positions of stars around circle
    angles = np.arctan2(y - cy, x - cx)
    angle_min, angle_max = angles.min(), angles.max()
    # Extend slightly beyond star positions
    margin = 0.05 * (angle_max - angle_min)
    theta = np.linspace(angle_min - margin, angle_max + margin, n_arc_points)
    arc_x = cx + r * np.cos(theta)
    arc_y = cy + r * np.sin(theta)
    ax.plot(arc_x, arc_y, color='#ffdd44', linewidth=1.0, alpha=0.5,
            linestyle='--', zorder=9)


def _poss_edge_with_gap(ax, ra0, dec0, ra1, dec1, gap_frac=0.12):
    """Draw a line between two points with gaps near each endpoint."""
    dx = ra1 - ra0
    dy = dec1 - dec0
    length = np.hypot(dx, dy)
    if length < 1e-10:
        return
    frac0 = min(gap_frac, 0.3)
    frac1 = min(gap_frac, 0.3)
    sx = ra0 + dx * frac0
    sy = dec0 + dy * frac0
    ex = ra1 - dx * frac1
    ey = dec1 - dy * frac1
    ax.plot([sx, ex], [sy, ey], color='#ffdd44', linewidth=1.0, alpha=0.7, zorder=9)


def _draw_poss_overlay(ax, poss_img, center_ra, center_dec, fov_deg, focus_stars, shape):
    """Draw POSS image with shape edges overlaid (no star markers, gapped lines)."""
    half = fov_deg / 2.0
    cos_dec = max(0.1, np.cos(np.radians(center_dec)))
    ra_half = half / cos_dec

    # Display image: RA increases to the left (standard sky orientation)
    ax.imshow(poss_img, cmap='gray', origin='upper',
              extent=[center_ra + ra_half, center_ra - ra_half,
                      center_dec - half, center_dec + half],
              aspect='auto')

    star_ras = [s[0] for s in focus_stars]
    star_decs = [s[1] for s in focus_stars]
    n = len(focus_stars)

    # For squares, sort vertices by angle from centroid for proper polygon order
    if shape == 'square' and n == 4:
        cx = np.mean(star_ras)
        cy = np.mean(star_decs)
        angles = [np.arctan2(d - cy, r - cx) for r, d in zip(star_ras, star_decs)]
        order = np.argsort(angles)
        star_ras = [star_ras[k] for k in order]
        star_decs = [star_decs[k] for k in order]

    if shape == 'circle':
        # Draw dashed arc through the stars
        _draw_circle_arc(ax, star_ras, star_decs)
    else:
        n_edges = n - 1 if shape == 'collinear' else n
        for i in range(n_edges):
            j = (i + 1) if shape == 'collinear' else (i + 1) % n
            _poss_edge_with_gap(ax, star_ras[i], star_decs[i], star_ras[j], star_decs[j])

    ax.set_xlim(center_ra + ra_half, center_ra - ra_half)
    ax.set_ylim(center_dec - half, center_dec + half)
    ax.set_aspect(1.0 / cos_dec)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')


# --- Enrichment functions ---

def get_mean(value):
    points = torch.tensor(value.to_list())
    return points, points.mean(dim=0)

def _get_center(value):
    _, mean = get_mean(value)
    ra = Angle(degrees=mean[0].item())
    return position_of_radec(ra._hours, mean[1])


def asterism_full_id(stars, shape, chain_len=None, con=None):
    """Constellation-prefixed asterism ID: CON-shape-hash.

    If con (3-letter abbreviation) is not provided, computes it from star centroid.
    """
    base_id = asterism_id(stars, shape, chain_len)
    if not con:
        pts = torch.tensor(stars) if not isinstance(stars, torch.Tensor) else stars
        mean = pts.mean(dim=0)
        ra = Angle(degrees=mean[0].item())
        center = position_of_radec(ra._hours, mean[1].item())
        con = constellation_at(center)
    return f"{con}-{base_id}"


def compute_solitary_score(focus_stars, center_ra, center_dec, extent_deg, max_mag, geom_score,
                           catalog=None):
    """Compute solitary score: geometric score weighted by flux dominance.

    solitary = geom_score / flux_ratio, where flux_ratio is the fraction of
    total field light contributed by the asterism stars. Lower = better
    (good geometry + dominant in field).

    If catalog is provided, use it instead of the global dftycho (for pre-filtered subsets).
    """
    cat = catalog if catalog is not None else dftycho
    radius = max(extent_deg * 1.5, 0.05)
    cos_dec = max(0.1, np.cos(np.radians(center_dec)))
    query_radius = radius / cos_dec
    field = stars_for_center_and_radius(cat, (center_ra, center_dec), query_radius, max_mag)

    asterism_mags = np.array([s[2] for s in focus_stars])
    asterism_flux = np.sum(10 ** (-0.4 * asterism_mags))

    if len(field) == 0:
        return geom_score  # no field stars, flux_ratio = 1

    field_mags = field['Vmag'].to_numpy()
    total_flux = np.sum(10 ** (-0.4 * field_mags))

    flux_ratio = asterism_flux / total_flux if total_flux > 0 else 1.0
    flux_ratio = max(flux_ratio, 1e-6)  # avoid division by zero
    return geom_score / flux_ratio


def _prefilter_catalog_for_solitary(candidates, max_mag):
    """Pre-filter the global catalog to a bounding box covering all solitary candidates."""
    all_ras = []
    all_decs = []
    max_extent = 0.0
    for row in candidates.iter_rows(named=True):
        pts = torch.tensor(row['stars'])
        ra_mean = pts[:, 0].mean().item()
        dec_mean = pts[:, 1].mean().item()
        extent = chain_extent_deg(pts) if len(row['stars']) > 3 else triangle_extent_deg(pts)
        all_ras.append(ra_mean)
        all_decs.append(dec_mean)
        max_extent = max(max_extent, extent)
    margin = max(max_extent * 2.0, 1.0)
    ra_min = min(all_ras) - margin
    ra_max = max(all_ras) + margin
    dec_min = min(all_decs) - margin
    dec_max = max(all_decs) + margin
    filtered = dftycho.filter(
        (pl.col("Vmag") <= max_mag) &
        (pl.col("DEmdeg") >= dec_min) & (pl.col("DEmdeg") <= dec_max)
    )
    if ra_min >= 0 and ra_max <= 360:
        filtered = filtered.filter(
            (pl.col("RAmdeg") >= ra_min) & (pl.col("RAmdeg") <= ra_max)
        )
    elif ra_min < 0:
        filtered = filtered.filter(
            (pl.col("RAmdeg") >= (ra_min % 360)) | (pl.col("RAmdeg") <= ra_max)
        )
    elif ra_max > 360:
        filtered = filtered.filter(
            (pl.col("RAmdeg") >= ra_min) | (pl.col("RAmdeg") <= (ra_max % 360))
        )
    return filtered


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


def add_full_ids(df, shape, scoring=None):
    """Add 'asterism_id' column with constellation-prefixed IDs to enriched results.

    For collinear shapes, uses scoring mode (smooth/snake/etc) as the ID shape
    since different scoring modes find genuinely different chain orderings.
    """
    if df.is_empty():
        return df.with_columns(pl.lit("").alias("asterism_id"))
    id_shape = scoring if shape == 'collinear' and scoring else shape
    ids = []
    for row in df.iter_rows(named=True):
        stars = row['stars']
        chain_len = row.get('chain_len') or len(stars)
        con = row.get('CON')
        ids.append(asterism_full_id(stars, id_shape, chain_len, con=con))
    return df.with_columns(pl.Series("asterism_id", ids))


_bv_cache = {}

def _lookup_bv_for_focus_stars(focus_stars):
    """Look up B-V for focus stars (from result parquet, only RA/Dec/Vmag) by catalog match."""
    colors = []
    for s in focus_stars:
        ra, dec = s[0], s[1]
        key = (round(ra, 3), round(dec, 3))
        if key in _bv_cache:
            colors.append(_bv_cache[key])
            continue
        tol = 0.005
        match = dftycho.filter(
            ((pl.col("RAmdeg") - ra).abs() < tol) &
            ((pl.col("DEmdeg") - dec).abs() < tol)
        )
        if len(match) > 0 and "BTmag" in match.columns and "VTmag" in match.columns:
            row = match.row(0, named=True)
            bt, vt = row["BTmag"], row["VTmag"]
            if bt is not None and vt is not None:
                color = bv_to_rgb(bt - vt)
            else:
                color = (1.0, 1.0, 1.0)
        else:
            color = (1.0, 1.0, 1.0)
        _bv_cache[key] = color
        colors.append(color)
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
    marker_size = np.maximum((0.5 + limiting_magnitude - fs['magnitude']) ** 2.0, 0.5)
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
    marker_size = np.maximum((0.5 + limiting_magnitude - fs['magnitude']) ** 2.0, 6.0)
    ax.scatter(fs['x'], fs['y'], s=marker_size, c='#ffdd00', edgecolors='none', zorder=10)
    return fs


def _draw_circle_arc_projected(ax, xs, ys, n_arc_points=100):
    """Draw best-fit circle arc in projected coordinates (x,y from stereographic)."""
    if len(xs) < 3:
        return
    x = np.array(xs)
    y = np.array(ys)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    try:
        result = np.linalg.lstsq(A, b, rcond=None)
        params = result[0]
    except np.linalg.LinAlgError:
        return
    cx, cy = params[0], params[1]
    r_sq = params[2] + cx ** 2 + cy ** 2
    if r_sq <= 0:
        return
    r = np.sqrt(r_sq)
    angles = np.arctan2(y - cy, x - cx)
    angle_min, angle_max = angles.min(), angles.max()
    margin = 0.05 * (angle_max - angle_min)
    theta = np.linspace(angle_min - margin, angle_max + margin, n_arc_points)
    arc_x = cx + r * np.cos(theta)
    arc_y = cy + r * np.sin(theta)
    ax.plot(arc_x, arc_y, color='#ffdd44', linewidth=0.8, alpha=0.4,
            linestyle='--', zorder=9, clip_on=True)


def _draw_shape_edges(ax, fs_focus, limiting_magnitude, n_stars, shape='triangle', data_limit=1.0):
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

    size_lm = limiting_magnitude

    # Circle/arc: draw dashed arc overlay
    if shape == 'circle':
        _draw_circle_arc_projected(ax, xs, ys)
        return

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
        # Marker radius in data coords + fixed gap (floor at fixed_gap to handle faint stars)
        r0 = max((0.5 + size_lm - m0) / (2.0 * pts_per_data), 0) + fixed_gap_data
        r1 = max((0.5 + size_lm - m1) / (2.0 * pts_per_data), 0) + fixed_gap_data
        frac0 = min(r0 / length, 0.3)
        frac1 = min(r1 / length, 0.3)
        sx = x0 + dx * frac0
        sy = y0 + dy * frac0
        ex = x1 - dx * frac1
        ey = y1 - dy * frac1
        ax.plot([sx, ex], [sy, ey], color='#ffdd44', linewidth=0.8, alpha=0.4, zorder=9,
                clip_on=True)


def _shape_edge_segments(stars, shape, gap_deg=0.02):
    """Return line segments with gaps near star vertices.

    Each segment is shortened by gap_deg (degrees) from both ends.

    For triangles: closed 0->1, 1->2, 2->0
    For squares: closed polygon sorted by angle from centroid
    For collinear: open 0->1, 1->2, ..., N-2->N-1
    For circle: approximate arc as short segments between consecutive stars (sorted by angle)
    """
    pts = [(s[0], s[1]) for s in stars]
    if shape == 'circle':
        # Sort stars by angle from centroid, then connect as open path
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        angles = [np.arctan2(p[1] - cy, p[0] - cx) for p in pts]
        order = np.argsort(angles)
        pts = [pts[i] for i in order]
    if shape == 'square' and len(pts) == 4:
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        angles = [np.arctan2(p[1] - cy, p[0] - cx) for p in pts]
        order = np.argsort(angles)
        pts = [pts[i] for i in order]

    n = len(pts)
    edges = [(i, i + 1) for i in range(n - 1)]
    if shape in ('triangle', 'square'):
        edges.append((n - 1, 0))

    segments = []
    for i, j in edges:
        ra0, dec0 = pts[i]
        ra1, dec1 = pts[j]
        cos_dec = np.cos(np.radians((dec0 + dec1) / 2))
        dra = (ra1 - ra0) * cos_dec
        ddec = dec1 - dec0
        length = np.hypot(dra, ddec)
        if length < 1e-10:
            continue
        frac = min(gap_deg / length, 0.3)
        segments.append([
            [round(ra0 + (ra1 - ra0) * frac, 6),
             round(dec0 + (dec1 - dec0) * frac, 6)],
            [round(ra1 - (ra1 - ra0) * frac, 6),
             round(dec1 - (dec1 - dec0) * frac, 6)],
        ])
    return segments


def generate_pifinder_list(enriched_results, outdir, name, shape='triangle', pifinder_outdir=None):
    if not pifinder_outdir:
        return
    os.makedirs(pifinder_outdir, exist_ok=True)
    pifinder_paths = [os.path.join(pifinder_outdir, f'{name}.pifinder')]

    objects = []
    for ast_idx, row in enumerate(enriched_results.iter_rows(named=True)):
        con = row['CONSTELLATION']
        score = row['score']
        ra_deg = row['Ra']
        dec_deg = row['Dec']
        star_mags = [s[2] for s in row['stars']]
        star_pts = torch.tensor(row['stars'])
        extent = chain_extent_deg(star_pts) if len(row['stars']) > 3 else triangle_extent_deg(star_pts)
        edge_segments = _shape_edge_segments(row['stars'], shape)
        extent_arcsec = extent * 3600.0
        aid = row.get('asterism_id', f"{row.get('CON', 'UNK')} asterism #{ast_idx+1}")

        notes_parts = [f"score={score:.6f}"]
        if 'tilt' in row:
            notes_parts.append(f"tilt={row['tilt']:.0f}\u00b0")
        notes_parts.append(f"mags={','.join(f'{m:.1f}' for m in star_mags)}")

        obj = {
            "name": aid,
            "obj_type": "Ast",
            "ra": round(ra_deg, 6),
            "dec": round(dec_deg, 6),
            "mag": {
                "mags": [round(m, 2) for m in star_mags],
                "filter_mag": round(min(star_mags), 2),
            },
            "const": row['CON'],
            "extents": {"geometry": "segments", "shape": edge_segments},
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

    for path in pifinder_paths:
        with open(path, 'w') as f:
            json.dump(pifinder_data, f, indent=2)

    print(f"  Wrote {len(objects)} entries to {len(pifinder_paths)} location(s)")


def generate_pdf(enriched_results, outdir, name, title, max_mag=15.0, fov_override=None, shape='triangle', instrument_info=None, scoring=''):
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

        # Instrument info subtitle
        if instrument_info:
            ax.text(0.5, 0.83, instrument_info,
                    fontsize=9, ha='center', va='top', transform=ax.transAxes, color='#999999',
                    fontfamily='monospace')

        has_isolation = 'isolation' in enriched_results.columns
        has_tilt = 'tilt' in enriched_results.columns
        has_solitary = 'solitary_score' in enriched_results.columns
        has_id = 'asterism_id' in enriched_results.columns
        table_data = []
        col_labels = ['#', 'ID', 'Score', 'Constellation', 'RA', 'Dec']
        if has_solitary:
            col_labels.append('Solitary')
        if has_tilt:
            col_labels.append('Tilt')
        if has_isolation:
            col_labels.append('Iso')
        for idx, row in enumerate(enriched_results.iter_rows(named=True)):
            aid = row.get('asterism_id', '') if has_id else ''
            row_data = [
                str(idx + 1), aid, f"{row['score']:.4f}", row['CONSTELLATION'],
                row['Rah_full'], row['Dec_full'],
            ]
            if has_solitary:
                row_data.append(f"{row['solitary_score']:.4f}")
            if has_tilt:
                row_data.append(f"{row['tilt']:.0f}\u00b0")
            if has_isolation:
                row_data.append(str(row['isolation']))
            table_data.append(row_data)

        col_widths = [0.04, 0.20, 0.10, 0.16, 0.18, 0.18]
        if has_solitary:
            col_widths.append(0.10)
        if has_tilt:
            col_widths.append(0.06)
        if has_isolation:
            col_widths.append(0.06)
        n_rows = len(table_data)
        row_height = 0.035
        table_height = (n_rows + 1) * row_height
        table_top = 0.78 if not instrument_info else 0.75
        table_bottom = table_top - table_height
        table_bbox = [0.02, max(table_bottom, 0.18), 0.96, min(table_height, table_top - 0.18)]
        table = ax.table(cellText=table_data,
                         colLabels=col_labels,
                         cellLoc='center',
                         bbox=table_bbox,
                         colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
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
        n_failures = 0
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

                # Fetch POSS sky survey image
                poss_fov = max(0.5, extent * 2.0)
                poss_img = fetch_poss_image(center_coord[0], center_coord[1], poss_fov)

                if poss_img:
                    fig = plt.figure(figsize=[8.5, 14])
                    fig.patch.set_facecolor('#111111')
                    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2.5, 2.5], hspace=0.20)
                else:
                    fig = plt.figure(figsize=[8.5, 11])
                    fig.patch.set_facecolor('#111111')
                    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2.5], hspace=0.25)

                # --- Finder chart (dark mode) ---
                ax_chart = fig.add_subplot(gs[0])
                ax_chart.set_facecolor('black')
                ax_chart.add_collection(LineCollection(lines_xy, colors='#4488ff', alpha=0.6, linewidths=0.8))
                angle = np.pi - field_of_view_degrees / 360.0 * np.pi
                limit = np.sin(angle) / (1.0 - np.cos(angle))

                draw_points(ax_chart, t, earth, projection, region_stars, limiting_magnitude)
                fs_focus = draw_focus_stars(ax_chart, t, earth, projection, focus_stars, limiting_magnitude)

                # Shape edges with gaps near stars
                _draw_shape_edges(ax_chart, fs_focus, limiting_magnitude, n_stars=len(focus_stars), shape=shape,
                                  data_limit=limit)
                ax_chart.set_xlim(-limit, limit)
                ax_chart.set_ylim(-limit, limit)
                ax_chart.xaxis.set_visible(False)
                ax_chart.yaxis.set_visible(False)
                ax_chart.set_aspect(1.0)
                chart_title = entry.get('asterism_id', f"#{idx+1}") if 'asterism_id' in entry else f"#{idx+1}"
                ax_chart.set_title(f"{chart_title}  {entry['CONSTELLATION']}", fontsize=13,
                                   fontweight='bold', color='white')
                ax_chart.tick_params(colors='white')
                for spine in ax_chart.spines.values():
                    spine.set_edgecolor('#333333')

                # Magnitude scale legend — bounded by actual stars in chart
                focus_mags = [s[2] for s in focus_stars]
                field_mags = region_stars["Vmag"].to_list() if len(region_stars) > 0 else []
                all_mags = focus_mags + field_mags
                legend_mag_min = int(np.floor(min(all_mags))) if all_mags else 0
                legend_mag_max = int(np.ceil(max(all_mags))) if all_mags else int(limiting_magnitude)
                mag_steps = list(range(legend_mag_min, legend_mag_max + 1, 2))
                if not mag_steps or mag_steps[-1] != legend_mag_max:
                    mag_steps.append(legend_mag_max)
                for m in mag_steps:
                    s = max((0.5 + limiting_magnitude - m) ** 2.0, 0.5)
                    ax_chart.scatter([], [], s=s, c='white', label=f'{m}')
                legend = ax_chart.legend(loc='upper right', title='Mag', fontsize=8,
                               title_fontsize=9, framealpha=0.7, facecolor='#1a1a1a',
                               edgecolor='#333333', labelcolor='white')
                legend.get_title().set_color('white')

                # Extent and mag range annotation (bottom-left)
                min_star_mag = min(s[2] for s in focus_stars)
                max_star_mag = max(s[2] for s in focus_stars)
                ann_text = f"{extent:.2f}°  mag {min_star_mag:.1f}\u2013{max_star_mag:.1f}"
                ax_chart.text(0.03, 0.03, ann_text, transform=ax_chart.transAxes,
                             fontsize=8, color='#aaaaaa', fontfamily='monospace',
                             verticalalignment='bottom')

                # --- POSS sky survey image panel ---
                if poss_img:
                    ax_poss = fig.add_subplot(gs[1])
                    ax_poss.set_facecolor('black')
                    _draw_poss_overlay(ax_poss, poss_img, center_coord[0], center_coord[1],
                                       poss_fov, focus_stars, shape)
                    ax_poss.set_title("POSS / Digitized Sky Survey", fontsize=10, color='#999999')

                # --- Info panel (dark mode) ---
                info_gs_idx = 2 if poss_img else 1
                ax_info = fig.add_subplot(gs[info_gs_idx])
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
                    elif 'snake' in scoring:
                        if score < 0.2:
                            quality = "Excellent"
                        elif score < 0.5:
                            quality = "Very good"
                        elif score < 1.0:
                            quality = "Good"
                        else:
                            quality = "Fair"
                        score_desc = f"Score: {score:.6f}  ({quality} snake fit)\n  Alternation + 0.3*turn_CV + 0.2*spacing_CV (0 = perfect S-curve)."
                    elif 'bright' in scoring:
                        if score < 0.01:
                            quality = "Excellent"
                        elif score < 0.05:
                            quality = "Very good"
                        elif score < 0.15:
                            quality = "Good"
                        else:
                            quality = "Fair"
                        score_desc = f"Score: {score:.6f}  ({quality} bright isolation)\n  Geometric score / (flux_ratio * contrast)."
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
                elif shape == 'circle':
                    if score < 0.1:
                        quality = "Excellent"
                    elif score < 0.3:
                        quality = "Very good"
                    elif score < 0.5:
                        quality = "Good"
                    else:
                        quality = "Fair"
                    arc_frac = entry.get('arc_fraction', 0)
                    radius_deg = entry.get('radius_deg', 0)
                    score_desc = (f"Score: {score:.6f}  ({quality} circle/arc fit)\n"
                                  f"  RMS radial dev + spacing CV + arc coverage.")
                    sep_line = (f"  {n_stars}-star arc,  radius: {radius_deg:.3f} deg,  "
                                f"arc: {arc_frac*100:.0f}%")
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
                    tilt_line = f"Tilt: {tilt_val:.1f}\u00b0  (0\u00b0 = face-on, 90\u00b0 = edge-on)\n"

                isolation_line = ""
                if has_isolation:
                    isolation_line = f"Isolation: {entry['isolation']} nearby field stars\n"

                solitary_line = ""
                if has_solitary:
                    solitary_line = f"Solitary: {entry['solitary_score']:.4f}  (score / flux dominance, lower = better)\n"

                instrument_line = ""
                if instrument_info:
                    instrument_line = f"{instrument_info}\n"

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

                id_line = f"ID: {entry['asterism_id']}\n" if 'asterism_id' in entry else ""
                info_text = (
                    f"{id_line}"
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
                    f"{solitary_line}"
                    f"{instrument_line}"
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
            except (ValueError, RuntimeError, KeyError) as e:
                n_failures += 1
                print(f"  Error on entry {idx}: {e}")
                traceback.print_exc()
                plt.close('all')

        if n_failures > 0:
            print(f"  WARNING: {n_failures}/{len(enriched_results)} pages failed to render")
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
# output/<run_name>/
#   <shape>/<eyepiece>/          ← PDFs
#   pifinder_lists/<shape>/<eyepiece>/  ← PiFinder files
OUTPUT_DIR = 'output'

# (subdir_path, scoring, parquet_file, max_mag, display_name, shape)
LEGACY_MODES = [
    # Telescopic
    ("telescopic/triangles", "3d", "results/result_triangle2.parquet",        15.0, "Telescopic triangles 3D", "triangle"),
    ("telescopic/triangles", "2d", "results/result_triangle2_2d.parquet",     15.0, "Telescopic triangles 2D", "triangle"),
    ("telescopic/squares",   "3d", "results/result_squareness.parquet",       15.0, "Telescopic squares 3D", "square"),
    ("telescopic/squares",   "2d", "results/result_squareness_2d.parquet",    15.0, "Telescopic squares 2D", "square"),
    ("telescopic/collinear/smooth",     "smooth",     "results/result_collinear_smooth.parquet",     15.0, "Telescopic collinear (smooth)", "collinear"),
    ("telescopic/collinear/smooth_mag", "smooth_mag", "results/result_collinear_smooth_mag.parquet", 15.0, "Telescopic collinear (smooth+mag)", "collinear"),
    # Binocular
    ("binocular/triangles", "3d", "results/result_triangle2_bino.parquet",    11.0, "Binocular triangles 3D", "triangle"),
    ("binocular/triangles", "2d", "results/result_triangle2_bino_2d.parquet", 11.0, "Binocular triangles 2D", "triangle"),
    ("binocular/collinear/smooth",     "smooth",     "results/result_collinear_bino_smooth.parquet",     11.0, "Binocular collinear (smooth)", "collinear"),
    ("binocular/collinear/smooth_mag", "smooth_mag", "results/result_collinear_bino_smooth_mag.parquet", 11.0, "Binocular collinear (smooth+mag)", "collinear"),
    # Naked eye
    ("naked_eye/triangles", "3d", "results/result_triangle2_naked.parquet",    6.0, "Naked eye triangles 3D", "triangle"),
    ("naked_eye/triangles", "2d", "results/result_triangle2_naked_2d.parquet", 6.0, "Naked eye triangles 2D", "triangle"),
]


def _build_instrument_modes(inst):
    """Build MODES list from instrument config.

    Directory structure: <shape>/<tag>/
    Shapes: triangles (3d/2d), squares (3d/2d), collinear (smooth/smooth_mag)
    Supports both eyepieces and camera configs.
    """
    modes = []

    configs = []
    for ep in inst.eyepieces:
        cfg = eyepiece_to_search_config(inst, ep)
        tag = f"{ep.focal_length_mm:.0f}mm"
        configs.append((cfg, tag, ep))
    if inst.camera:
        cfg = camera_to_search_config(inst)
        tag = inst.camera.name
        configs.append((cfg, tag, None))

    for cfg, tag, ep in configs:
        config_tag = cfg.name

        # Triangles
        for scoring in ("3d", "2d"):
            suffix = "" if scoring == "3d" else "_2d"
            modes.append((
                f"triangles/{tag}",
                scoring,
                f"results/result_triangle2_{config_tag}{suffix}.parquet",
                cfg.max_mag,
                f"{inst.name} {tag} triangles {scoring.upper()}",
                "triangle",
                cfg,
                ep,
            ))

        # Squares
        for scoring in ("3d", "2d"):
            suffix = "" if scoring == "3d" else "_2d"
            modes.append((
                f"squares/{tag}",
                scoring,
                f"results/result_squareness_{config_tag}{suffix}.parquet",
                cfg.max_mag,
                f"{inst.name} {tag} squares {scoring.upper()}",
                "square",
                cfg,
                ep,
            ))

        # Collinear
        for col_scoring in ("smooth", "smooth_mag", "snake"):
            modes.append((
                f"collinear/{col_scoring}/{tag}",
                col_scoring,
                f"results/result_collinear_{config_tag}_{col_scoring}.parquet",
                cfg.max_mag,
                f"{inst.name} {tag} collinear ({col_scoring})",
                "collinear",
                cfg,
                ep,
            ))

        # Bright isolated lines (post-processed from smooth collinear)
        modes.append((
            f"bright_line/{tag}",
            "bright_line",
            f"results/result_bright_line_{config_tag}.parquet",
            cfg.max_mag,
            f"{inst.name} {tag} bright lines",
            "collinear",
            cfg,
            ep,
        ))

        # Circles/arcs
        modes.append((
            f"circles/{tag}",
            "circle",
            f"results/result_circle_{config_tag}_2d.parquet",
            cfg.max_mag,
            f"{inst.name} {tag} circles/arcs",
            "circle",
            cfg,
            ep,
        ))

    return modes


def _instrument_info_str(inst, ep, cfg):
    """Build instrument info string for PDF display."""
    fl = inst.focal_length_mm
    if ep is not None:
        magnification = fl / ep.focal_length_mm
        tfov = ep.afov_deg / magnification
        return (
            f"{inst.aperture_mm:.0f}mm f/{inst.focal_ratio:.0f}  |  "
            f"{ep.focal_length_mm:.0f}mm ({ep.afov_deg:.0f}\u00b0 AFOV)  |  "
            f"{magnification:.0f}x  |  TFOV {tfov:.2f}\u00b0  |  "
            f"LM {cfg.max_mag}  |  SQM {inst.sqm}"
        )
    cam = inst.camera
    fov_w, fov_h = camera_fov(cam, fl)
    return (
        f"{inst.aperture_mm:.0f}mm f/{inst.focal_ratio:.0f}  |  "
        f"{cam.name} ({cam.res_x}\u00d7{cam.res_y})  |  "
        f"FOV {fov_w:.2f}\u00b0\u00d7{fov_h:.2f}\u00b0  |  "
        f"LM {cfg.max_mag}  |  SQM {inst.sqm}"
    )


def _process_mode(subdir_path, scoring, filename, max_mag, display_name, shape,
                   search_config=None, eyepiece=None, inst=None, run_dir=None):
    """Process a single report mode (legacy or instrument-derived)."""
    if not os.path.exists(filename):
        print(f"Skipping {subdir_path}/{scoring}: {filename} not found")
        return

    if run_dir is None:
        run_dir = OUTPUT_DIR
    outdir = os.path.join(run_dir, subdir_path)
    os.makedirs(outdir, exist_ok=True)
    run_name = os.path.basename(run_dir)
    pifinder_outdir = os.path.join(OUTPUT_DIR, "pifinder_lists", run_name, subdir_path)

    print(f"\n{'=' * 60}")
    print(f"Generating: {subdir_path}/{scoring} (limiting mag {max_mag})")

    df = pl.read_parquet(filename)
    df = dedup_results(df)
    print(f"  {len(df)} results (after dedup)")

    # Build instrument info string if applicable
    inst_info = None
    if inst and search_config and (eyepiece is not None or inst.camera):
        inst_info = _instrument_info_str(inst, eyepiece, search_config)

    # Filter 3D triangle/square results by tilt: keep 10-60deg
    if scoring == '3d' and shape in ('triangle', 'square') and 'tilt' in df.columns:
        before = len(df)
        df = df.filter((pl.col("tilt") >= 10) & (pl.col("tilt") <= 60))
        print(f"  Tilt filter (10\u00b0-60\u00b0): {before} \u2192 {len(df)} results")

    if shape == 'collinear' and 'chain_len' in df.columns:
        # Snake chains require at least 5 stars for meaningful alternation
        if scoring == 'snake':
            before = len(df)
            df = df.filter(pl.col("chain_len") >= 5)
            print(f"  Snake filter (chain_len >= 5): {before} -> {len(df)} results")
            if len(df) == 0:
                print(f"  No snake chains with >= 5 stars")
                return
        # Filter chains with duplicate stars (near-zero spacings)
        def _has_unique_stars(stars):
            coords = [(round(s[0], 4), round(s[1], 4)) for s in stars]
            return len(coords) == len(set(coords))
        before = len(df)
        df = df.filter(pl.col("stars").map_elements(_has_unique_stars, return_dtype=pl.Boolean))
        if len(df) < before:
            print(f"  Removed {before - len(df)} chains with duplicate stars")
        _process_collinear_mode(df, outdir, pifinder_outdir, scoring, max_mag,
                                display_name, shape, search_config, inst_info, run_dir)
        return

    # Non-collinear: enrich top 500 once and reuse for both main report and solitary
    sorted_df = df.sort("score")
    all_enriched_500 = enrich_results(sorted_df.head(500), max_mag=max_mag)
    all_enriched_500 = add_full_ids(all_enriched_500, shape)
    all_enriched = all_enriched_500.head(100)
    visible = filter_visible_51n(all_enriched)

    if len(visible) == 0:
        print(f"  No asterisms visible from 51N for {subdir_path}/{scoring}")
        return

    top10 = visible.head(10)
    generate_pdf(top10, outdir, f"{scoring}_51N",
                 f"Top 10 visible from 51\u00b0N - {display_name}", max_mag=max_mag,
                 shape=shape, instrument_info=inst_info)
    generate_pifinder_list(top10, outdir, f"{scoring}_51N", shape=shape, pifinder_outdir=pifinder_outdir)
    print(f"  51N: {len(top10)} asterisms")

    for season in SEASONS:
        seasonal = filter_seasonal(visible, season).head(10)
        if len(seasonal) > 0:
            generate_pifinder_list(seasonal, outdir, f"{scoring}_{season}_51N", shape=shape, pifinder_outdir=pifinder_outdir)
        else:
            print(f"  No {season} asterisms for {subdir_path}/{scoring}")

    # Solitary scoring pass: score / flux_ratio over top 500 candidates
    solitary_candidates = sorted_df.head(500)
    solitary_catalog = _prefilter_catalog_for_solitary(solitary_candidates, max_mag)
    print(f"  Solitary pre-filtered catalog: {len(solitary_catalog):,} stars (from {len(dftycho):,})")
    solitary_scores = []
    for row in solitary_candidates.iter_rows(named=True):
        focus_stars = row['stars']
        pts = torch.tensor(focus_stars)
        center_ra = pts[:, 0].mean().item()
        center_dec = pts[:, 1].mean().item()
        extent = chain_extent_deg(pts) if len(focus_stars) > 3 else triangle_extent_deg(pts)
        sol = compute_solitary_score(focus_stars, center_ra, center_dec, extent, max_mag, row['score'],
                                     catalog=solitary_catalog)
        solitary_scores.append(sol)
    scored = solitary_candidates.with_columns(pl.Series("solitary_score", solitary_scores))
    # Reuse already-enriched data for solitary top 20
    enriched_with_sol = all_enriched_500.join(
        scored.select("score", "stars", "solitary_score"),
        on=["score", "stars"], how="inner"
    )
    solitary_top = enriched_with_sol.sort("solitary_score").head(20)
    solitary_top = filter_visible_51n(solitary_top).head(10)

    if len(solitary_top) > 0:
        generate_pdf(solitary_top, outdir, f"{scoring}_solitary_51N",
                     f"Top 10 solitary - {display_name}", max_mag=max_mag,
                     shape=shape, instrument_info=inst_info)
        generate_pifinder_list(solitary_top, outdir, f"{scoring}_solitary_51N",
                               shape=shape, pifinder_outdir=pifinder_outdir)
        print(f"  Solitary: {len(solitary_top)} asterisms")

    # Legacy FOV-filtered reports for telescopic triangles
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
                         max_mag=max_mag, fov_override=fov, shape=shape, instrument_info=inst_info)
            generate_pifinder_list(fov_filtered, outdir, f"{scoring}_fov{fov:.1f}deg_51N",
                                   shape=shape, pifinder_outdir=pifinder_outdir)


def _diverse_top_n(df, n=10):
    """Pick top N results with at most ceil(N/num_lengths) per chain length.

    Round-robin across chain lengths sorted ascending, picking the best-scored
    from each in turn until we have N results.
    """
    chain_lengths = sorted(df["chain_len"].unique().to_list())
    per_length = {clen: df.filter(pl.col("chain_len") == clen) for clen in chain_lengths}
    cursors = {clen: 0 for clen in chain_lengths}
    selected_rows = []
    while len(selected_rows) < n:
        added_this_round = False
        for clen in chain_lengths:
            if len(selected_rows) >= n:
                break
            clen_df = per_length[clen]
            idx = cursors[clen]
            if idx < len(clen_df):
                selected_rows.append(clen_df.row(idx, named=True))
                cursors[clen] = idx + 1
                added_this_round = True
        if not added_this_round:
            break
    if not selected_rows:
        return df.head(0)
    return pl.DataFrame(selected_rows, schema=df.schema)


def _process_collinear_mode(df, outdir, pifinder_outdir, scoring, max_mag,
                             display_name, shape, search_config, inst_info, run_dir=None):
    """Process collinear results with chain-length binning."""
    chain_lengths = sorted(df["chain_len"].unique().to_list())
    MAX_PAGES = 20

    # First pass: count visible results per chain length
    visible_per_clen = {}
    for clen in chain_lengths:
        clen_df = df.filter(pl.col("chain_len") == clen).sort("score")
        enriched = enrich_results(clen_df.head(100), max_mag=max_mag)
        enriched = add_full_ids(enriched, shape, scoring)
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

        bin_visible = pl.concat([visible_per_clen[c] for c in bin_clens])
        bin_visible = bin_visible.sort("score").head(MAX_PAGES)

        print(f"\n  --- {bin_display}: {len(bin_visible)} pages ---")
        generate_pdf(bin_visible, outdir, f"{bin_name}_51N",
                     f"Top {len(bin_visible)} visible from 51\u00b0N - {bin_display}",
                     max_mag=max_mag, shape=shape, instrument_info=inst_info, scoring=scoring)
        list_bin_name = bin_name.removeprefix(f"{scoring}_")
        generate_pifinder_list(bin_visible, outdir, f"{list_bin_name}_51N",
                               shape=shape, pifinder_outdir=pifinder_outdir)

        for season in SEASONS:
            seasonal = filter_seasonal(bin_visible, season).head(MAX_PAGES)
            if len(seasonal) > 0:
                generate_pifinder_list(seasonal, outdir, f"{list_bin_name}_{season}_51N",
                                       shape=shape, pifinder_outdir=pifinder_outdir)

    # Combined report (top 10, best per chain length) - use diverse candidate pool
    sorted_df = df.sort("score")
    diverse_500 = _diverse_top_n(sorted_df, n=500)
    all_enriched_500 = enrich_results(diverse_500, max_mag=max_mag)
    all_enriched_500 = add_full_ids(all_enriched_500, shape, scoring)
    all_enriched = all_enriched_500.head(100)
    visible = filter_visible_51n(all_enriched)
    if len(visible) > 0:
        top10 = _diverse_top_n(visible, n=10)
        generate_pdf(top10, outdir, f"{scoring}_51N",
                     f"Top 10 visible from 51\u00b0N - {display_name}",
                     max_mag=max_mag, shape=shape, instrument_info=inst_info, scoring=scoring)
        generate_pifinder_list(top10, outdir, "top10_51N", shape=shape, pifinder_outdir=pifinder_outdir)

    # Solitary scoring pass for collinear
    if run_dir:
        solitary_candidates = diverse_500
        solitary_catalog = _prefilter_catalog_for_solitary(solitary_candidates, max_mag)
        print(f"  Solitary pre-filtered catalog: {len(solitary_catalog):,} stars (from {len(dftycho):,})")
        solitary_scores = []
        for row in solitary_candidates.iter_rows(named=True):
            focus_stars = row['stars']
            pts = torch.tensor(focus_stars)
            center_ra = pts[:, 0].mean().item()
            center_dec = pts[:, 1].mean().item()
            extent = chain_extent_deg(pts)
            sol = compute_solitary_score(focus_stars, center_ra, center_dec, extent, max_mag, row['score'],
                                         catalog=solitary_catalog)
            solitary_scores.append(sol)
        scored = solitary_candidates.with_columns(pl.Series("solitary_score", solitary_scores))
        enriched_with_sol = all_enriched_500.join(
            scored.select("score", "stars", "solitary_score"),
            on=["score", "stars"], how="inner"
        )
        solitary_visible = filter_visible_51n(enriched_with_sol.sort("solitary_score").head(100))
        solitary_top = _diverse_top_n(solitary_visible, n=10)

        if len(solitary_top) > 0:
            generate_pdf(solitary_top, outdir, f"{scoring}_solitary_51N",
                         f"Top 10 solitary - {display_name}", max_mag=max_mag,
                         shape=shape, instrument_info=inst_info, scoring=scoring)
            generate_pifinder_list(solitary_top, outdir, f"{scoring}_solitary_51N",
                                   shape=shape, pifinder_outdir=pifinder_outdir)
            print(f"  Solitary: {len(solitary_top)} asterisms")


def _prompt_run_name(inst):
    """Prompt for run name, check if exists, ask to override."""
    default_name = inst.name
    name = input(f"Run name [{default_name}]: ").strip() or default_name
    run_dir = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(run_dir):
        resp = input(f"  '{run_dir}' already exists. Override? [y/N]: ").strip().lower()
        if resp != 'y':
            print("Aborted.")
            return None
    return run_dir


if __name__ == '__main__':
    inst = DEFAULT_INSTRUMENT

    # Check for --instrument-json flag
    if '--instrument-json' in sys.argv:
        idx = sys.argv.index('--instrument-json')
        with open(sys.argv[idx + 1]) as f:
            inst = InstrumentConfig.from_dict(json.load(f))

    if '--legacy' in sys.argv:
        # Legacy preset modes (no run name prompt)
        run_dir = os.path.join(OUTPUT_DIR, 'legacy')
        os.makedirs(run_dir, exist_ok=True)
        for subdir_path, scoring, filename, max_mag, display_name, shape in LEGACY_MODES:
            _process_mode(subdir_path, scoring, filename, max_mag, display_name, shape,
                          run_dir=run_dir)
    else:
        run_dir = _prompt_run_name(inst)
        if run_dir is None:
            sys.exit(0)

    os.makedirs(run_dir, exist_ok=True)
    print(f"\nOutput directory: {run_dir}")

    inst_modes = _build_instrument_modes(inst)
    for subdir_path, scoring, filename, max_mag, display_name, shape, cfg, ep in inst_modes:
        _process_mode(subdir_path, scoring, filename, max_mag, display_name, shape,
                      search_config=cfg, eyepiece=ep, inst=inst, run_dir=run_dir)

    print(f"\n\nDone! All reports generated in {run_dir}")
