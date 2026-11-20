"""Figures: high-ranking windows on the mapped geology (R2.6).

One figure per case. The background is the Para 1:1,000,000 sheet of the
Servico Geologico do Brasil (CPRM, 2008), coloured by lithostratigraphic
unit. On top are the top 250 windows of the published ranking, the twelve
known deposits, and the reference deposit of that case.

    Fig26  Case 1, univariate TMI, reference Paulo Afonso, k=17
    Fig27  Case 2, multivariate TMI + radiometric U, reference Alemao,
           k_TMI=2, k_U=8, alpha_TMI=0.503

Windows that touch a known deposit are drawn in one colour and those that do
not in another, since the second group is what the comment asks about.

Run from inside paper/:

    python make_fig_geology_targets.py --outdir ../docs/Spatial_PCA_paper_overleaf/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.geometry import box

import case1_uni_repro as C1
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
GEOLOGY = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson"
UTM22S = "EPSG:32722"
N_TOP = 250
# Age-ordered colours, following the International Chronostratigraphic Chart,
# so the map reads as geological time rather than as arbitrary unit colours.
ERA_COLOUR = {
    "Paleoarqueano":     ("#E54A96", "Palaeoarchean, 3.6-3.2 Ga"),
    "Mesoarqueano":      ("#F074AF", "Mesoarchean, 3.2-2.8 Ga"),
    "Neoarqueano":       ("#F8A9D0", "Neoarchean, 2.8-2.5 Ga"),
    "Paleoproterozóico": ("#F74370", "Palaeoproterozoic, 2.5-1.6 Ga"),
    "Mesoproterozóico":  ("#FDB462", "Mesoproterozoic, 1.6-1.0 Ga"),
    "Neoproterozóico":   ("#FEB342", "Neoproterozoic, 1.0-0.54 Ga"),
    "Paleozóico":        ("#99C08D", "Palaeozoic, 541-252 Ma"),
    "Mesozóico":         ("#67C5CA", "Mesozoic, 252-66 Ma"),
    "Cenozóico":         ("#F9F97F", "Cenozoic, 66-0 Ma"),
}
ERA_ORDER = list(ERA_COLOUR)
FIG_W = 5.14                      # \textwidth = 372 pt

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.0})


def study_area():
    tmi, _u, ox, oy = R.load_cropped()
    height, width = tmi.shape
    return ox, oy - height * R.PX, ox + width * R.PX, oy


def deposit_frames():
    """The twelve deposit outlines in the grid's own frame."""
    polys = [box(min(p[0] for p in rings[0]), min(p[1] for p in rings[0]),
                 max(p[0] for p in rings[0]), max(p[1] for p in rings[0]))
             for rings in C1.deposits_case1_wgs84()]
    return gpd.GeoDataFrame(geometry=polys, crs=UTM22S)


def ranked_windows(case):
    """Top 250 window rectangles, split by whether they touch a known deposit."""
    if case == 1:
        order, idx, wshape, origin, dep = C1.rank_spca(17)
    else:
        order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, 0.503)
    ox, oy = origin
    wh, ww = wshape
    deposits = deposit_frames().geometry
    touching, clear = [], []
    for i in [w for w in order if w != dep][:N_TOP]:
        r, c, _ = idx[i]
        x1, y2 = ox + c * R.PX, oy - r * R.PX
        rect = box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2)
        (touching if deposits.intersects(rect).any() else clear).append(rect)
    return touching, clear, wshape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / "figures_out")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    geology = gpd.read_file(GEOLOGY).to_crs(UTM22S)
    xmin, ymin, xmax, ymax = study_area()
    geology = gpd.clip(geology, box(xmin, ymin, xmax, ymax))
    geology = geology[~geology.geometry.is_empty]

    geology["colour"] = [ERA_COLOUR.get(e, ("#d9d9d9", ""))[0] for e in geology["ERA_MAX"]]
    present = [e for e in ERA_ORDER if (geology["ERA_MAX"] == e).any()]

    deposits = deposit_frames()

    for case, name, ref, title in ((1, "Fig26", 6, "Case 1, univariate TMI, $k=17$"),
                                   (2, "Fig27", 3, "Case 2, TMI + radiometric U")):
        touching, clear, wshape = ranked_windows(case)
        fig, ax = plt.subplots(figsize=(FIG_W, 4.25))
        geology.plot(ax=ax, color=geology["colour"], edgecolor="white", linewidth=0.25)

        for rect in clear:
            ax.add_patch(Rectangle((rect.bounds[0], rect.bounds[1]),
                                   rect.bounds[2] - rect.bounds[0],
                                   rect.bounds[3] - rect.bounds[1],
                                   fill=False, ec="black", lw=0.45))
        for rect in touching:
            ax.add_patch(Rectangle((rect.bounds[0], rect.bounds[1]),
                                   rect.bounds[2] - rect.bounds[0],
                                   rect.bounds[3] - rect.bounds[1],
                                   fill=False, ec="#d62728", lw=0.7))
        deposits.boundary.plot(ax=ax, color="white", linewidth=1.3)
        deposits.boundary.plot(ax=ax, color="black", linewidth=0.7)
        deposits.iloc[[ref - 1]].boundary.plot(ax=ax, color="#d62728", linewidth=1.3)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(450000, 700001, 50000))
        ax.set_yticks(np.arange(9240000, 9360001, 40000))
        ax.set_xticklabels([f"{v/1000:.0f}" for v in ax.get_xticks()])
        ax.set_yticklabels([f"{v/1000:.0f}" for v in ax.get_yticks()])
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.set_title(f"{title}; window {wshape[0]} $\\times$ {wshape[1]} px", pad=3)
        ax.tick_params(length=2, width=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

        handles = [Rectangle((0, 0), 1, 1, fc=ERA_COLOUR[e][0], ec="white", lw=0.3)
                   for e in present]
        labels = [ERA_COLOUR[e][1] for e in present]
        handles += [Line2D([], [], color="black", lw=0.6),
                    Line2D([], [], color="#d62728", lw=0.9),
                    Line2D([], [], color="black", lw=1.0)]
        labels += [f"top {N_TOP}, no known deposit", f"top {N_TOP}, on a known deposit",
                   "known deposit (red: reference)"]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=2, frameon=False, fontsize=5.6, handlelength=1.4,
                  handleheight=0.9, labelspacing=0.35, columnspacing=1.0)

        fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.30)
        stem = args.outdir / name
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
        fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
        plt.close(fig)
        print(f"wrote {stem}.pdf  ({len(clear)} clear, {len(touching)} on deposits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
