"""Rebuild Figure 2 (fig:deposits) from data at publication resolution.

Replaces a raster of the same v9 lineage as Figure 18 (1429x595 px, ~220 dpi at
\\textwidth, no generating script in the repo): centred south-north TMI profiles and
the TMI window for each of the twelve known deposits.

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
NAMES = {1: "Tucuma", 2: "Pedra Branca", 3: "Alemao", 4: "Furnas", 5: "Salobo",
         6: "Paulo Afonso", 7: "Cabano"}
# 12 well-separated hues; the reference deposit is the heavy dark red.
CURVE = {
    1: "#1f4fd8", 2: "#e07b00", 3: "#00897b", 4: "#2e9e2e", 5: "#8e24aa",
    6: "#c62828", 7: "#5d4037", 8: "#0288d1", 9: "#c2185b", 10: "#827717",
    11: "#455a64", 12: "#ef6c00",
}
DASHED = {8, 9, 10, 11, 12}   # unnamed deposits, dashed to separate them at a glance

mpl.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---------------------------------------------------------------- layout constants
FIG_W, FIG_H = 7.2, 5.90
L, RGT = 0.085, 0.915
NCOL = 6
COL_W = 0.1215
GAP = (RGT - L - NCOL * COL_W) / (NCOL - 1)

PROF_TOP, PROF_H = 0.945, 0.175
LEGEND_Y = 0.722
ROW_TITLE = (0.628, 0.300)     # title baseline, top and bottom window rows
ROW_BOT = (0.470, 0.142)       # image baseline
ROW_SIZE = (0.454, 0.126)      # "w x h km" annotation under each window
ROW_CBAR = (0.422, 0.094)      # colourbar strip
CBAR_H = 0.013


def col_x(j):
    return L + j * (COL_W + GAP)


def image_box(ext, x0, y_bottom):
    w_data, h_data = ext[1] - ext[0], ext[3] - ext[2]
    return [x0, y_bottom, COL_W, COL_W * (FIG_W / FIG_H) * (h_data / w_data)]


def style_window(ax, ext, ref):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.1 if ref else 0.5)
        s.set_color("#c62828" if ref else "black")


def label_for(d):
    return f"{d} {NAMES[d]}" if d in NAMES else f"{d}"


def build(data, outdir: Path, stem: str, dpi: int, flip: bool = False):
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---------------------------------------------------------- profiles
    ax = fig.add_axes([L, PROF_TOP - PROF_H, RGT - L, PROF_H])
    for d in sorted(data):
        v = data[d]
        ref = d == REF_DEP
        x = -v["prof_d"] if flip else v["prof_d"]
        ax.plot(x, v["prof_tmi"], color=CURVE[d],
                lw=2.0 if ref else 0.9,
                ls="--" if d in DASHED else "-",
                zorder=3 if ref else 2,
                label=label_for(d) + (" - reference" if ref else ""))
    ax.axhline(0, color="#9e9e9e", lw=0.5, zorder=1)
    ax.axvline(0, color="#9e9e9e", lw=0.5, zorder=1)
    ax.set_xlabel("Distance along south-north line, centred (km)", labelpad=2)
    ax.set_ylabel("TMI, centred (nT)", labelpad=2)
    ax.grid(alpha=0.25, lw=0.4)
    ax.margins(x=0.01)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    if not flip:
        ax.annotate("south", xy=(0.012, 0.055), xycoords="axes fraction",
                    fontsize=6.5, color="#616161")
        ax.annotate("north", xy=(0.965, 0.055), xycoords="axes fraction",
                    fontsize=6.5, color="#616161", ha="right")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, LEGEND_Y),
               ncol=6, frameon=False, fontsize=7, handlelength=1.7,
               columnspacing=1.4, handletextpad=0.5)

    # ---------------------------------------------------------- window grid
    for i, d in enumerate(sorted(data)):
        row, j = divmod(i, NCOL)
        v = data[d]
        ext = km_extent(v["wshape"])
        ref = d == REF_DEP
        x0 = col_x(j)

        ax = fig.add_axes(image_box(ext, x0, ROW_BOT[row]))
        vmin, vmax = tmi_limits(d, v["tmi"])
        im = ax.imshow(v["tmi"], origin="upper", cmap=PAPER_CMAP, vmin=vmin, vmax=vmax,
                       extent=ext, aspect="auto", interpolation="nearest")
        style_window(ax, ext, ref)
        fig.text(x0 + COL_W / 2, ROW_SIZE[row],
                 f"{ext[1] - ext[0]:.1f} x {ext[3] - ext[2]:.1f} km",
                 ha="center", va="top", fontsize=6, color="#616161")

        cax = fig.add_axes([x0 + 0.010, ROW_CBAR[row], COL_W - 0.020, CBAR_H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([vmin, 0, vmax])
        cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
        cb.outline.set_linewidth(0.5)
        cb.set_label("nT", fontsize=5.5, labelpad=1)

        fig.text(x0 + COL_W / 2, ROW_TITLE[row], label_for(d), ha="center",
                 va="bottom", fontsize=7.5, color="#c62828" if ref else "black")
        if ref:
            fig.text(x0 + COL_W / 2, ROW_TITLE[row] - 0.020, "reference deposit",
                     ha="center", va="bottom", fontsize=6, color="#c62828")

    fig.text(0.5, 0.990,
             "The twelve known deposits: centred south-north TMI profiles "
             "and deposit windows",
             ha="center", va="top", fontsize=9.5)

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
