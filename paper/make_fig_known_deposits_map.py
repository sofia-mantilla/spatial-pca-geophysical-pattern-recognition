"""Figure 2: TMI of the study area with the twelve known deposits labelled.

Rebuilt so that every deposit carries the name its table row carries. The
earlier version of this figure took its labels from the Name field of the
deposit shapefile, which holds placeholders for deposits 8 to 12, so five of
the twelve appeared as "Dep. 8" and so on while the rest were named. Names now
come from deposit_names.py, which follows Table 1.

Colours are the paper's TMI colormap and the Case-1 limits, so this figure
matches the TMI panels elsewhere.

Run from inside paper/:

    python make_fig_known_deposits_map.py --outdir ../docs/Spatial_PCA_paper_overleaf/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import case1_uni_repro as C1
import deposit_names as D
import numpy_repro_concat as R
from make_fig_multiprofiles import PAPER_CMAP

HERE = Path(__file__).resolve().parent
REF_DEP = 6                      # Paulo Afonso, the Case-1 reference
VMIN, VMAX = -200.0, 400.0       # as published
REF_COLOUR = "#7b1fa2"

# Label offsets in km from the top-left corner of each deposit box, tuned by
# hand so that no label covers another deposit or runs off the map.
OFFSET = {
    1: (-6, 7), 2: (-4, 8), 3: (-10, 9), 4: (4, 9), 5: (-24, 2), 6: (6, 7),
    7: (7, -5), 8: (-6, 9), 9: (-18, -5), 10: (-21, 6), 11: (-20, 7),
    12: (-14, 8),
}


def deposit_boxes():
    """(number, x0, y0, width, height) for each deposit, in metres."""
    out = []
    for i, rings in enumerate(C1.deposits_case1_wgs84(), start=1):
        xs = [p[0] for p in rings[0]]
        ys = [p[1] for p in rings[0]]
        out.append((i, min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / "figures_out")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--name", default="Fig2")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    extent = (ox, ox + width * R.PX, oy - height * R.PX, oy)

    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    im = ax.imshow(tmi, origin="upper", cmap=PAPER_CMAP, vmin=VMIN, vmax=VMAX,
                   extent=extent, interpolation="nearest")

    for n, x0, y0, w, h in deposit_boxes():
        is_ref = n == REF_DEP
        colour = REF_COLOUR if is_ref else "black"
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec=colour,
                               lw=1.9 if is_ref else 1.6, zorder=6))
        dx, dy = OFFSET[n]
        ax.annotate(
            D.label(n),
            xy=(x0 + 0.5 * w, y0 + h if dy > 0 else y0),
            xytext=(x0 + dx * 1000.0, y0 + h + dy * 1000.0),
            fontsize=10, fontweight="bold", color=colour, zorder=8,
            ha="left", va="center",
            bbox=dict(fc="white", ec=colour, lw=0.7, alpha=0.92,
                      boxstyle="round,pad=0.22"),
            arrowprops=dict(arrowstyle="-", color=colour, lw=0.7,
                            shrinkA=1.0, shrinkB=1.0),
        )

    # 50 km scale bar, bottom left
    bar = 50_000.0
    bx, by = extent[0] + 12_000, extent[2] + 11_000
    ax.plot([bx, bx + bar], [by, by], color="black", lw=3.0, zorder=9,
            solid_capstyle="butt")
    ax.text(bx + bar / 2, by + 3_500, "50 km", ha="center", va="bottom",
            fontsize=10, zorder=9,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # north arrow, bottom right
    nx, ny = extent[1] - 17_000, extent[2] + 10_000
    ax.annotate("", xy=(nx, ny + 15_000), xytext=(nx, ny),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6), zorder=9)
    ax.text(nx, ny - 3_000, "N", ha="center", va="top", fontsize=11,
            fontweight="bold", zorder=9,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Easting (m)", fontsize=11)
    ax.set_ylabel("Northing (m)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.ticklabel_format(style="plain")

    ax.legend(
        handles=[Line2D([], [], color=REF_COLOUR, lw=1.9,
                        label="Reference deposit (%s)" % D.label(REF_DEP)),
                 Line2D([], [], color="black", lw=1.6, label="Test deposits")],
        loc="lower center", bbox_to_anchor=(0.5, -0.145), ncol=2,
        frameon=False, fontsize=10)

    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.030)
    cb.set_label("TMI (nT)", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.078, right=0.93, top=0.965, bottom=0.145)
    stem = args.outdir / args.name
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print("wrote %s.png with all twelve deposits named" % stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
