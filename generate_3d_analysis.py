"""Analysis of 3D scoring for asterism detection.

Compares the current linear scaling (r = 1.0 + 0.01*mag) with the correct
perceived-distance formula (d = 10^((m+5)/5) with M=0 assumption).
"""
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

plt.style.use('dark_background')

COLORS = {
    'cyan': '#00FFFF',
    'yellow': '#FFD700',
    'magenta': '#FF00FF',
    'lime': '#00FF00',
    'orange': '#FF8C00',
    'white': '#FFFFFF',
    'light_gray': '#CCCCCC',
    'dim': '#888888',
}

# The bad square example
BAD_SQUARE = [
    (289.11, 46.60, 7.88),
    (288.60, 45.63, 8.35),
    (289.80, 45.49, 9.59),
    (288.52, 45.90, 10.31),
]


def distance_from_magnitude(m, M=0):
    """Perceived distance from apparent magnitude. d = 10^((m - M + 5) / 5)."""
    return 10 ** ((m - M + 5) / 5)


def radecmag_to_xyz_linear(ra_deg, dec_deg, mag, k=0.01):
    """Current (broken) linear scaling: r = 1.0 + k * mag."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    r = 1.0 + k * mag
    return r * np.cos(dec) * np.cos(ra), r * np.cos(dec) * np.sin(ra), r * np.sin(dec)


def radecmag_to_xyz_perceptual(ra_deg, dec_deg, mag):
    """Correct perceptual distance: r = 10^((m+5)/5)."""
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    r = distance_from_magnitude(mag)
    return r * np.cos(dec) * np.cos(ra), r * np.cos(dec) * np.sin(ra), r * np.sin(dec)


def angular_sep_deg(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2])
    cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return np.degrees(np.arccos(np.clip(cos_sep, -1, 1)))


def pairwise_3d(stars, xyz_fn, **kwargs):
    """Compute all 6 pairwise 3D distances for 4 stars."""
    coords = [xyz_fn(ra, dec, mag, **kwargs) for ra, dec, mag in stars]
    coords = np.array(coords)
    pairs = list(combinations(range(4), 2))
    dists = [np.linalg.norm(coords[i] - coords[j]) for i, j in pairs]
    return pairs, np.array(dists), coords


def draw_square(ax, stars, pairs, dists, coords_2d=True):
    """Draw the 4 stars with sides/diagonals color-coded by 3D scoring."""
    sorted_idx = np.argsort(dists)
    side_set = set(sorted_idx[:4])

    ras = [s[0] for s in stars]
    decs = [s[1] for s in stars]

    for idx, (i, j) in enumerate(pairs):
        is_side = idx in side_set
        color = COLORS['cyan'] if is_side else COLORS['magenta']
        ls = '-' if is_side else '--'
        lw = 2 if is_side else 1.5
        ax.plot([ras[i], ras[j]], [decs[i], decs[j]],
                color=color, linestyle=ls, linewidth=lw, alpha=0.8)

        mid_ra = (ras[i] + ras[j]) / 2
        mid_dec = (decs[i] + decs[j]) / 2
        ang = angular_sep_deg(ras[i], decs[i], ras[j], decs[j])
        ax.text(mid_ra, mid_dec,
                f'{ang:.2f}°\n3D: {dists[idx]:.4f}',
                fontsize=6.5, color=color, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

    for i, (ra, dec, mag) in enumerate(stars):
        size = max(300 - 20 * mag, 40)
        ax.scatter(ra, dec, s=size, color=COLORS['yellow'], zorder=5,
                   edgecolors=COLORS['white'], linewidth=0.5)
        ax.text(ra, dec - 0.08, f'★{i+1} mag {mag:.2f}',
                fontsize=8, color=COLORS['white'], ha='center', va='top')


def page_title(fig, pdf):
    fig.set_size_inches(8.5, 11)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.text(0.5, 0.93, '3D Scoring Analysis for Asterism Detection',
            fontsize=22, fontweight='bold', color=COLORS['cyan'], ha='center', va='top')
    ax.text(0.5, 0.88, 'Linear scaling vs perceptual distance from magnitude',
            fontsize=14, color=COLORS['yellow'], ha='center', va='top')

    body = """THE THEORY

When observing stars, brighter stars appear "closer" and dimmer stars "further
away". This perceptual depth can be modeled using the distance modulus:

    d = 10^((m - M + 5) / 5)        where M = 0 (assumed)

This gives the perceived distance from apparent magnitude m.
A star 5 magnitudes fainter appears 10× further away.

Examples:
    mag  7  →  d =    251
    mag  8  →  d =    398       (1.58× further than mag 7)
    mag  9  →  d =    631       (2.51× further than mag 7)
    mag 10  →  d =  1,000       (3.98× further than mag 7)
    mag 15  →  d = 100,000      (398× further than mag 7)

The 3D coordinates become:
    x = d(m) × cos(Dec) × cos(RA)
    y = d(m) × cos(Dec) × sin(RA)
    z = d(m) × sin(Dec)


THE CURRENT IMPLEMENTATION (BROKEN)

    r = 1.0 + 0.01 × magnitude

This linear approximation:
  • Completely misses the exponential nature of the relationship
  • mag 7 → r=1.0788,  mag 10 → r=1.1031,  ratio = 1.02
  • Should be: ratio = 3.98

This paper compares both approaches and their effect on pattern detection."""

    ax.text(0.08, 0.82, body, fontsize=10, color=COLORS['light_gray'],
            ha='left', va='top', fontfamily='monospace', linespacing=1.4)
    pdf.savefig(fig); fig.clear()


def page_distance_comparison(fig, pdf):
    """Page 2: Perceived distance vs magnitude for both formulas."""
    fig.set_size_inches(8.5, 11)
    gs = fig.add_gridspec(2, 1, hspace=0.35, top=0.92, bottom=0.08, left=0.12, right=0.92)

    mags = np.linspace(4, 15, 200)

    # Top: absolute distances
    ax1 = fig.add_subplot(gs[0])
    d_perceptual = distance_from_magnitude(mags)
    d_linear = 1.0 + 0.01 * mags
    ax1.semilogy(mags, d_perceptual, color=COLORS['cyan'], linewidth=2.5,
                 label='Perceptual: d = 10^((m+5)/5)')
    ax1.semilogy(mags, d_linear, color=COLORS['magenta'], linewidth=2.5,
                 label='Linear: r = 1.0 + 0.01×m')
    ax1.set_xlabel('Apparent magnitude', fontsize=12)
    ax1.set_ylabel('Radial distance (log scale)', fontsize=12)
    ax1.set_title('Radial Distance vs Magnitude', fontsize=14,
                  fontweight='bold', color=COLORS['cyan'])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, which='both')

    # Bottom: distance RATIOS relative to mag 7
    ax2 = fig.add_subplot(gs[1])
    ref_mag = 7.0
    ratio_perceptual = distance_from_magnitude(mags) / distance_from_magnitude(ref_mag)
    ratio_linear = (1.0 + 0.01 * mags) / (1.0 + 0.01 * ref_mag)

    ax2.plot(mags, ratio_perceptual, color=COLORS['cyan'], linewidth=2.5,
             label='Perceptual ratio')
    ax2.plot(mags, ratio_linear, color=COLORS['magenta'], linewidth=2.5,
             label='Linear ratio')
    ax2.axhline(1, color=COLORS['dim'], linestyle=':', linewidth=0.8)

    # Annotate key points
    for m in [8, 10, 12, 15]:
        rp = distance_from_magnitude(m) / distance_from_magnitude(ref_mag)
        rl = (1.0 + 0.01 * m) / (1.0 + 0.01 * ref_mag)
        ax2.plot(m, rp, 'o', color=COLORS['cyan'], markersize=6)
        ax2.annotate(f'{rp:.1f}×', (m, rp), textcoords='offset points',
                     xytext=(8, 5), fontsize=8, color=COLORS['cyan'])
        ax2.plot(m, rl, 'o', color=COLORS['magenta'], markersize=6)
        ax2.annotate(f'{rl:.2f}×', (m, rl), textcoords='offset points',
                     xytext=(8, -12), fontsize=8, color=COLORS['magenta'])

    ax2.set_xlabel('Apparent magnitude', fontsize=12)
    ax2.set_ylabel(f'Distance ratio (relative to mag {ref_mag:.0f})', fontsize=12)
    ax2.set_title(f'Distance Ratios Relative to mag {ref_mag:.0f} Star',
                  fontsize=14, fontweight='bold', color=COLORS['cyan'])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, max(ratio_perceptual[-1] * 1.1, 10))

    pdf.savefig(fig); fig.clear()


def page_3d_distance_curves(fig, pdf):
    """Page 3: 3D distance vs angular separation for both formulas."""
    fig.set_size_inches(8.5, 11)
    gs = fig.add_gridspec(2, 1, hspace=0.35, top=0.92, bottom=0.08, left=0.12, right=0.92)

    ang_seps = np.linspace(0.01, 5, 500)
    dmags = [0, 1, 2, 4]
    color_list = [COLORS['cyan'], COLORS['lime'], COLORS['yellow'], COLORS['orange']]
    mag1 = 8.0

    # Top: linear scaling
    ax1 = fig.add_subplot(gs[0])
    for dmag, color in zip(dmags, color_list):
        mag2 = mag1 + dmag
        r1 = 1.0 + 0.01 * mag1
        r2 = 1.0 + 0.01 * mag2
        ra2 = np.radians(ang_seps)
        dx = r2 * np.cos(ra2) - r1
        dy = r2 * np.sin(ra2)
        dists = np.sqrt(dx**2 + dy**2)
        label = f'Δmag = {dmag}' if dmag > 0 else 'Same mag'
        ax1.plot(ang_seps, dists, color=color, linewidth=2, label=label)

    ax1.axvline(2, color=COLORS['dim'], linestyle=':', linewidth=1)
    ax1.text(2.1, ax1.get_ylim()[1] * 0.1, 'Telescopic\n2° radius', fontsize=8, color=COLORS['dim'])
    ax1.set_xlabel('Angular separation (degrees)', fontsize=11)
    ax1.set_ylabel('3D distance', fontsize=11)
    ax1.set_title('Linear scaling: r = 1.0 + 0.01 × mag',
                  fontsize=13, fontweight='bold', color=COLORS['magenta'])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Bottom: perceptual scaling
    ax2 = fig.add_subplot(gs[1])
    for dmag, color in zip(dmags, color_list):
        mag2 = mag1 + dmag
        r1 = distance_from_magnitude(mag1)
        r2 = distance_from_magnitude(mag2)
        ra2 = np.radians(ang_seps)
        dx = r2 * np.cos(ra2) - r1
        dy = r2 * np.sin(ra2)
        dists = np.sqrt(dx**2 + dy**2)
        label = f'Δmag = {dmag}' if dmag > 0 else 'Same mag'
        ax2.plot(ang_seps, dists, color=color, linewidth=2, label=label)

    ax2.axvline(2, color=COLORS['dim'], linestyle=':', linewidth=1)
    ax2.text(2.1, ax2.get_ylim()[1] * 0.1, 'Telescopic\n2° radius', fontsize=8, color=COLORS['dim'])
    ax2.set_xlabel('Angular separation (degrees)', fontsize=11)
    ax2.set_ylabel('3D distance', fontsize=11)
    ax2.set_title('Perceptual scaling: r = 10^((m+5)/5)',
                  fontsize=13, fontweight='bold', color=COLORS['cyan'])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    pdf.savefig(fig); fig.clear()


def page_radial_fraction(fig, pdf):
    """Page 4: What fraction of 3D distance is radial vs angular."""
    fig.set_size_inches(8.5, 11)
    gs = fig.add_gridspec(2, 1, hspace=0.35, top=0.92, bottom=0.08, left=0.12, right=0.92)

    dmags = np.linspace(0.01, 6, 500)
    ang_seps = [0.5, 1.0, 2.0, 5.0]
    color_list = [COLORS['magenta'], COLORS['cyan'], COLORS['lime'], COLORS['yellow']]
    mag1 = 8.0

    for ax_idx, (title, r_fn) in enumerate([
        ('Linear: r = 1.0 + 0.01 × mag', lambda m: 1.0 + 0.01 * m),
        ('Perceptual: r = 10^((m+5)/5)', distance_from_magnitude),
    ]):
        ax = fig.add_subplot(gs[ax_idx])

        for ang_sep, color in zip(ang_seps, color_list):
            radial_pcts = []
            for dm in dmags:
                mag2 = mag1 + dm
                r1 = r_fn(mag1)
                r2 = r_fn(mag2)
                dr = abs(r2 - r1)
                r_mean = (r1 + r2) / 2.0
                ang_comp = 2.0 * r_mean * np.sin(np.radians(ang_sep) / 2.0)
                total_sq = dr**2 + ang_comp**2
                radial_pcts.append(dr**2 / total_sq * 100 if total_sq > 0 else 0)
            ax.plot(dmags, radial_pcts, color=color, linewidth=2,
                    label=f'{ang_sep}° separation')

        ax.axhline(50, color=COLORS['dim'], linestyle='--', linewidth=0.8, alpha=0.5)
        ax.text(0.1, 53, 'radial dominates →', fontsize=8, color=COLORS['dim'])
        ax.set_xlabel('Magnitude difference', fontsize=11)
        ax.set_ylabel('Radial fraction of 3D distance (%)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold',
                     color=COLORS['magenta'] if ax_idx == 0 else COLORS['cyan'])
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2)

    pdf.savefig(fig); fig.clear()


def page_bad_square_linear(fig, pdf):
    """Page 5: The bad square — linear scoring on sky."""
    fig.set_size_inches(8.5, 11)
    gs = fig.add_gridspec(2, 1, hspace=0.4, top=0.92, bottom=0.08, left=0.12, right=0.92)

    stars = BAD_SQUARE
    pairs, dists, coords = pairwise_3d(stars, radecmag_to_xyz_linear)

    sorted_idx = np.argsort(dists)
    sides = dists[sorted_idx[:4]]
    diags = dists[sorted_idx[4:]]
    cv_s = np.std(sides) / np.mean(sides)
    cv_d = np.std(diags) / np.mean(diags)
    ratio = np.mean(diags) / np.mean(sides)

    ax1 = fig.add_subplot(gs[0])
    draw_square(ax1, stars, pairs, dists)
    ax1.set_xlabel('RA (degrees)', fontsize=11)
    ax1.set_ylabel('Dec (degrees)', fontsize=11)
    ax1.set_title(f'Linear scoring on sky (r = 1.0 + 0.01×mag)\n'
                  f'Side CV={cv_s:.4f}, Diag CV={cv_d:.4f}, Diag/Side={ratio:.3f} (√2={math.sqrt(2):.3f})\n'
                  f'Cyan=sides, Magenta=diagonals (as identified by 3D scorer)',
                  fontsize=11, fontweight='bold', color=COLORS['magenta'])
    ax1.invert_xaxis()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # 3D view
    ax2 = fig.add_subplot(gs[1], projection='3d')
    sorted_idx = np.argsort(dists)
    side_set = set(sorted_idx[:4])

    for idx, (i, j) in enumerate(pairs):
        is_side = idx in side_set
        color = COLORS['cyan'] if is_side else COLORS['magenta']
        ls = '-' if is_side else '--'
        ax2.plot([coords[i][0], coords[j][0]],
                 [coords[i][1], coords[j][1]],
                 [coords[i][2], coords[j][2]],
                 color=color, linestyle=ls, linewidth=2, alpha=0.8)

    for i, (ra, dec, mag) in enumerate(stars):
        ax2.scatter(*coords[i], s=80, color=COLORS['yellow'],
                    edgecolors=COLORS['white'], linewidth=0.5, zorder=5)
        ax2.text(coords[i][0], coords[i][1], coords[i][2],
                 f'  ★{i+1} m={mag:.1f}', fontsize=8, color=COLORS['white'])
    ax2.set_title('Same stars in 3D (linear scaling)\nLooks like a square here!',
                  fontsize=11, fontweight='bold', color=COLORS['magenta'])
    ax2.view_init(elev=25, azim=135)
    ax2.tick_params(labelsize=7, colors=COLORS['dim'])

    pdf.savefig(fig); fig.clear()


def page_bad_square_perceptual(fig, pdf):
    """Page 6: The bad square — perceptual scoring on sky."""
    fig.set_size_inches(8.5, 11)
    gs = fig.add_gridspec(2, 1, hspace=0.4, top=0.92, bottom=0.08, left=0.12, right=0.92)

    stars = BAD_SQUARE
    pairs, dists, coords = pairwise_3d(stars, radecmag_to_xyz_perceptual)

    sorted_idx = np.argsort(dists)
    sides = dists[sorted_idx[:4]]
    diags = dists[sorted_idx[4:]]
    cv_s = np.std(sides) / np.mean(sides)
    cv_d = np.std(diags) / np.mean(diags)
    ratio = np.mean(diags) / np.mean(sides)

    ax1 = fig.add_subplot(gs[0])
    draw_square(ax1, stars, pairs, dists)
    ax1.set_xlabel('RA (degrees)', fontsize=11)
    ax1.set_ylabel('Dec (degrees)', fontsize=11)
    ax1.set_title(f'Perceptual scoring on sky (r = 10^((m+5)/5))\n'
                  f'Side CV={cv_s:.4f}, Diag CV={cv_d:.4f}, Diag/Side={ratio:.3f} (√2={math.sqrt(2):.3f})\n'
                  f'Cyan=sides, Magenta=diagonals (as identified by 3D scorer)',
                  fontsize=11, fontweight='bold', color=COLORS['cyan'])
    ax1.invert_xaxis()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # 3D view
    ax2 = fig.add_subplot(gs[1], projection='3d')
    sorted_idx = np.argsort(dists)
    side_set = set(sorted_idx[:4])

    for idx, (i, j) in enumerate(pairs):
        is_side = idx in side_set
        color = COLORS['cyan'] if is_side else COLORS['magenta']
        ls = '-' if is_side else '--'
        ax2.plot([coords[i][0], coords[j][0]],
                 [coords[i][1], coords[j][1]],
                 [coords[i][2], coords[j][2]],
                 color=color, linestyle=ls, linewidth=2, alpha=0.8)

    for i, (ra, dec, mag) in enumerate(stars):
        ax2.scatter(*coords[i], s=80, color=COLORS['yellow'],
                    edgecolors=COLORS['white'], linewidth=0.5, zorder=5)
        ax2.text(coords[i][0], coords[i][1], coords[i][2],
                 f'  ★{i+1} m={mag:.1f}', fontsize=8, color=COLORS['white'])
    ax2.set_title('Same stars in 3D (perceptual scaling)\nMagnitude depth completely dominates',
                  fontsize=11, fontweight='bold', color=COLORS['cyan'])
    ax2.view_init(elev=25, azim=135)
    ax2.tick_params(labelsize=7, colors=COLORS['dim'])

    pdf.savefig(fig); fig.clear()


def page_pair_analysis(fig, pdf):
    """Page 7: Detailed pair-by-pair breakdown for the bad square."""
    fig.set_size_inches(8.5, 11)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.text(0.5, 0.95, 'Pair-by-Pair Breakdown: The Bad Square',
            fontsize=18, fontweight='bold', color=COLORS['cyan'], ha='center', va='top')

    stars = BAD_SQUARE
    pairs = list(combinations(range(4), 2))

    header = f'{"Pair":>6} {"Sky°":>7} {"Δmag":>5} {"Lin 3D":>9} {"Lin Δr%":>8} {"Perc 3D":>10} {"Perc Δr%":>9}'
    y = 0.87
    ax.text(0.08, y, header, fontsize=10, color=COLORS['yellow'],
            fontfamily='monospace', fontweight='bold')
    y -= 0.025
    ax.text(0.08, y, '─' * 70, fontsize=10, color=COLORS['dim'], fontfamily='monospace')

    for i, j in pairs:
        ra1, dec1, m1 = stars[i]
        ra2, dec2, m2 = stars[j]
        ang = angular_sep_deg(ra1, dec1, ra2, dec2)
        dmag = abs(m2 - m1)

        # Linear
        r1_l = 1.0 + 0.01 * m1
        r2_l = 1.0 + 0.01 * m2
        c1_l = radecmag_to_xyz_linear(ra1, dec1, m1)
        c2_l = radecmag_to_xyz_linear(ra2, dec2, m2)
        d_l = math.sqrt(sum((a-b)**2 for a,b in zip(c1_l, c2_l)))
        dr_l = abs(r2_l - r1_l)
        pct_l = dr_l / d_l * 100 if d_l > 0 else 0

        # Perceptual
        r1_p = distance_from_magnitude(m1)
        r2_p = distance_from_magnitude(m2)
        c1_p = radecmag_to_xyz_perceptual(ra1, dec1, m1)
        c2_p = radecmag_to_xyz_perceptual(ra2, dec2, m2)
        d_p = math.sqrt(sum((a-b)**2 for a,b in zip(c1_p, c2_p)))
        dr_p = abs(r2_p - r1_p)
        pct_p = dr_p / d_p * 100 if d_p > 0 else 0

        y -= 0.03
        line = f'{i+1}-{j+1}:   {ang:6.3f}  {dmag:5.2f}  {d_l:9.5f}  {pct_l:6.0f}%   {d_p:10.1f}  {pct_p:6.0f}%'
        color = COLORS['lime'] if pct_l < 30 and pct_p < 30 else COLORS['orange']
        ax.text(0.08, y, line, fontsize=10, color=color, fontfamily='monospace')

    y -= 0.06
    analysis = """KEY OBSERVATIONS

With LINEAR scaling (r = 1.0 + 0.01×mag):
  • Pair 2-4: only 0.28° apart on sky, but Δr is 96% of 3D distance
  • The scorer sees 4 roughly equal "sides" and 2 "diagonals" — a square!
  • But on the sky, it's clearly not a square

With PERCEPTUAL scaling (r = 10^((m+5)/5)):
  • The radial component is MUCH larger (distances in hundreds)
  • Magnitude differences dominate even more at this scale
  • The same stars would score even worse as a square

THE FUNDAMENTAL ISSUE

The perceptual distance formula is correct for modeling how we perceive
star depths. But for a SQUARE pattern to be visually meaningful, the
pattern must look like a square ON THE SKY (projected on the celestial
sphere). The depth dimension (brightness) can add a preference for
similar-magnitude stars, but the geometric shape must be evaluated
angularly.

For TRIANGLES: 3D scoring works because the CV of side lengths is
scale-free — if the 3D sides happen to be equal, the angular sides
are likely also similar (unless radial dominates).

For SQUARES: the constraint is stricter — we need 4 equal sides AND
the diagonal/side ratio = √2. Radial contamination can easily satisfy
these constraints with stars that are NOT in a square on the sky."""

    ax.text(0.08, y, analysis, fontsize=9.5, color=COLORS['light_gray'],
            ha='left', va='top', fontfamily='monospace', linespacing=1.35)

    pdf.savefig(fig); fig.clear()


def page_what_works(fig, pdf):
    """Page 8: When does 3D scoring work vs fail."""
    fig.set_size_inches(8.5, 11)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.text(0.5, 0.95, 'When Does 3D Scoring Work?',
            fontsize=20, fontweight='bold', color=COLORS['cyan'], ha='center', va='top')

    body = """
THE ROLE OF MAGNITUDE IN PATTERN DETECTION

The original insight is correct: visually, stars at similar brightness
form more striking patterns. The perceived-distance formula d=10^((m+5)/5)
correctly models this depth perception.

However, the question is: should magnitude affect the GEOMETRY or be a
separate QUALITY metric?


APPROACH A: Magnitude as depth dimension (current 3D scoring)

  ✓ Works for EQUILATERAL TRIANGLES at LARGE angular scales
    • CV of 3 sides is scale-invariant
    • At naked-eye scales, angular term dominates
    • Magnitude similarity is naturally rewarded

  ✗ Fails for SQUARES (any scale)
    • The diagonal/side ratio √2 is easily faked by radial distances
    • 4+2 distance constraint is over-determined

  ✗ Fails for TRIANGLES at SMALL angular scales (telescopic)
    • Radial term dominates → scoring geometric quality of mag differences

  ✗ Fails for COLLINEAR chains
    • Straightness is purely angular — depth is meaningless


APPROACH B: Angular geometry + magnitude quality score (recommended)

  Score = geometric_score(angular_positions) + weight × mag_penalty

  Where:
    geometric_score: evaluated on unit vectors (celestial sphere)
    mag_penalty: CV of magnitudes, or std(mag), etc.

  This cleanly separates "does it look like a square on the sky?"
  from "are the stars similar in brightness?"

  Already implemented for collinear chains:
    visual = rms_perp + 0.3*cv + 0.2*mag_cv + 0.1*mean_mag_norm


RECOMMENDATION

  • Triangles:  Keep 2D angular scoring + add mag_cv penalty (like visual)
  • Squares:    Use 2D angular scoring only (or + mag_cv)
  • Collinear:  Already uses angular + mag terms ✓

  The 3D scoring mode can be retired — its function is better served
  by the angular + magnitude quality approach, which keeps geometry
  and brightness clearly separated."""

    ax.text(0.06, 0.88, body, fontsize=10, color=COLORS['light_gray'],
            ha='left', va='top', fontfamily='monospace', linespacing=1.3)

    pdf.savefig(fig); fig.clear()


def main():
    output_path = 'reports/3d_scoring_analysis.pdf'
    fig = plt.figure()

    with PdfPages(output_path) as pdf:
        page_title(fig, pdf)
        page_distance_comparison(fig, pdf)
        page_3d_distance_curves(fig, pdf)
        page_radial_fraction(fig, pdf)
        page_bad_square_linear(fig, pdf)
        page_bad_square_perceptual(fig, pdf)
        page_pair_analysis(fig, pdf)
        page_what_works(fig, pdf)

    plt.close(fig)
    print(f'Generated {output_path} (8 pages)')


if __name__ == '__main__':
    main()
