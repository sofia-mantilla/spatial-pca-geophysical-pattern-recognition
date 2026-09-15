"""Rebuild Figure 2 (fig:deposits) from data at publication resolution.

Replaces a raster of the same v9 lineage as Figure 18 (1429x595 px, ~220 dpi at
\\textwidth, no generating script in the repo): centered south-north TMI profiles and
the TMI window for each of the twelve known deposits.

The figure is laid out at its printed width (\\textwidth = 5.14 in) with 8 pt text, so
nothing prints below the journal's 8 pt floor; the twelve windows sit on a 4x3 grid.

Conventions are shared with make_fig_multiprofiles.py (Figure 18) so the two figures
read as a pair: same colour ramp, the project's canonical per-deposit TMI limits from
configs/carajas_uni_tmi.yaml, same kilometre extents, deposits named once.

NOTE ON PROFILE DIRECTION. Distance increases NORTHWARD from the window centre, so
the northern (negative, blue) dipole lobe plots at positive x and the southern
(positive, magenta) lobe at negative x. The raster this replaces was mirrored
relative to that - and mirrored relative to old Figure 18, which used the same
convention adopted here. See --flip-profiles to reproduce the old Figure 2 direction.

Run from the worktree root:
    python make_fig_deposit_profiles.py --outdir docs/Spatial_PCA_paper_overleaf/figures
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from extract_case1_windows import collect
from make_fig_multiprofiles import PAPER_CMAP, km_extent, tmi_limits

REF_DEP = 6          # Paulo Afonso, the Case-1 reference deposit
NAMES = {1: "Tucumã", 2: "Pedra Branca", 3: "Alemão", 4: "Furnas", 5: "Salobo",
         6: "Paulo Afonso", 7: "Cabano"}
# 12 well-separated hues; the reference deposit is the heavy dark red.
CURVE = {
    1: "#1f4fd8", 2: "#e07b00", 3: "#00897b", 4: "#2e9e2e", 5: "#8e24aa",
    6: "#c62828", 7: "#5d4037", 8: "#0288d1", 9: "#c2185b", 10: "#827717",
    11: "#455a64", 12: "#ef6c00",
}
DASHED = {8, 9, 10, 11, 12}   # unnamed deposits, dashed to separate them at a glance

mpl.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---------------------------------------------------------------- layout (inches)
# The figure is drawn at the printed width, so a font size in points IS its printed
# size. Every text element is 8 pt or larger.
FIG_W, FIG_H = 5.14, 7.46
NCOL, NROW = 4, 3

MARG_L, MARG_R = 0.16, 0.16            # window-grid side margins
COL_W = (FIG_W - MARG_L - MARG_R - 0.66) / NCOL      # 0.22 in between columns
COL_GAP = 0.22

PROF_L, PROF_R = 0.62, 0.08            # profile axes margins
PROF_TOP, PROF_H = 0.10, 1.45          # top edge (from figure top) and height
PROF_XLAB = 0.36                       # ticks + x label under the profile axes
LEGEND_TOP = PROF_TOP + PROF_H + PROF_XLAB + 0.10
GRID_TOP = LEGEND_TOP + 0.68           # first window row starts here

TITLE_H = 0.17                         # deposit name above each window row
SIZE_H = 0.16                          # "w x h km" under the window
CBAR_H = 0.07
CBAR_LAB_H = 0.17                      # colourbar tick labels
ROW_GAP = 0.10


def _ax(fig, x, y_top, w, h):
    """Axes rect from inches measured from the top-left corner."""
    return fig.add_axes([x / FIG_W, (FIG_H - y_top - h) / FIG_H, w / FIG_W, h / FIG_H])


def col_x(j):
    return MARG_L + j * (COL_W + COL_GAP)


def style_window(ax, ref):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.1 if ref else 0.5)
        s.set_color("#c62828" if ref else "black")


def label_for(d):
    return f"{d} {NAMES[d]}" if d in NAMES else f"{d}"


def build(data, outdir: Path, stem: str, dpi: int, flip: bool = False):
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---------------------------------------------------------- profiles
    ax = _ax(fig, PROF_L, PROF_TOP, FIG_W - PROF_L - PROF_R, PROF_H)
    for d in sorted(data):
        v = data[d]
        ref = d == REF_DEP
        x = -v["prof_d"] if flip else v["prof_d"]
        ax.plot(x, v["prof_tmi"], color=CURVE[d],
                lw=2.0 if ref else 0.9,
                ls="--" if d in DASHED else "-",
                zorder=3 if ref else 2,
                label=label_for(d) + (" (reference)" if ref else ""))
    ax.axhline(0, color="#9e9e9e", lw=0.5, zorder=1)
    ax.axvline(0, color="#9e9e9e", lw=0.5, zorder=1)
    ax.set_xlabel("Distance along south\u2013north line, centered (km)", labelpad=2)
    ax.set_ylabel("TMI, centered (nT)", labelpad=2)
    ax.grid(alpha=0.25, lw=0.4)
    ax.margins(x=0.01)
    ax.tick_params(length=2.5, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    if not flip:
        ax.annotate("south", xy=(0.012, 0.055), xycoords="axes fraction",
                    fontsize=8, color="#616161")
        ax.annotate("north", xy=(0.988, 0.055), xycoords="axes fraction",
                    fontsize=8, color="#616161", ha="right")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, (FIG_H - LEGEND_TOP) / FIG_H),
               ncol=3, frameon=False, fontsize=8, handlelength=1.8,
               columnspacing=1.4, handletextpad=0.5, labelspacing=0.35)

    # ---------------------------------------------------------- window grid
    order = sorted(data)
    # a row is as tall as its tallest window (windows are bottom-aligned on a
    # common baseline so the size labels and colourbars line up)
    row_h = []
    for r in range(NROW):
        aspects = []
        for d in order[r * NCOL:(r + 1) * NCOL]:
            ext = km_extent(data[d]["wshape"])
            aspects.append((ext[3] - ext[2]) / (ext[1] - ext[0]))
        row_h.append(COL_W * max(aspects))

    y_top = GRID_TOP
    for r in range(NROW):
        img_bot = y_top + TITLE_H + row_h[r]
        for j, d in enumerate(order[r * NCOL:(r + 1) * NCOL]):
            v = data[d]
            ext = km_extent(v["wshape"])
            ref = d == REF_DEP
            x0 = col_x(j)
            h = COL_W * (ext[3] - ext[2]) / (ext[1] - ext[0])

            ax = _ax(fig, x0, img_bot - h, COL_W, h)
            vmin, vmax = tmi_limits(d, v["tmi"])
            im = ax.imshow(v["tmi"], origin="upper", cmap=PAPER_CMAP, vmin=vmin,
                           vmax=vmax, extent=ext, aspect="auto",
                           interpolation="nearest")
            style_window(ax, ref)

            fig.text((x0 + COL_W / 2) / FIG_W, (FIG_H - img_bot - 0.04) / FIG_H,
                     f"{ext[1] - ext[0]:.1f} \u00d7 {ext[3] - ext[2]:.1f} km",
                     ha="center", va="top", fontsize=8, color="#616161")

            cax = _ax(fig, x0 + 0.06, img_bot + SIZE_H, COL_W - 0.30, CBAR_H)
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.set_ticks([vmin, 0, vmax])
            cb.ax.tick_params(labelsize=8, length=2, pad=1.5)
            cb.outline.set_linewidth(0.5)
            fig.text((x0 + COL_W - 0.21) / FIG_W,
                     (FIG_H - img_bot - SIZE_H - CBAR_H / 2) / FIG_H, "nT",
                     ha="left", va="center", fontsize=8)

            y_name = FIG_H - img_bot + h + 0.045 + (TITLE_H if ref else 0.0)
            fig.text((x0 + COL_W / 2) / FIG_W, y_name / FIG_H, label_for(d),
                     ha="center", va="bottom", fontsize=8,
                     color="#c62828" if ref else "black")
            if ref:
                fig.text((x0 + COL_W / 2) / FIG_W,
                         (FIG_H - img_bot + h + 0.045) / FIG_H,
                         "reference deposit", ha="center", va="bottom",
                         fontsize=8, color="#c62828")
        y_top = img_bot + SIZE_H + CBAR_H + CBAR_LAB_H + ROW_GAP

    outdir.mkdir(parents=True, exist_ok=True)
    pdf, png = outdir / f"{stem}.pdf", outdir / f"{stem}.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return pdf, png


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rebuild fig:deposits from data.")
    p.add_argument("--outdir", type=Path, default=Path("figures_out"))
    p.add_argument("--stem", default="deposit_profiles_windows")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--flip-profiles", action="store_true",
                   help="mirror the profile x-axis to match the raster this replaces "
                        "(north at negative x); off by default because it contradicts "
                        "the south-north label and old Figure 18")
    a = p.parse_args()

    data = collect()
    pdf, png = build(data, a.outdir, a.stem, a.dpi, a.flip_profiles)
    for f in (pdf, png):
        print(f"wrote {f} ({f.stat().st_size / 1024:.0f} KB)")
