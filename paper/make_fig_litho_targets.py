"""Figures: high-ranking windows on the mapped lithology (R2.6).

One figure per case. The background is a geological map of the Servico Geologico do Brasil, held
in GeoSGB and retrieved for the study-area bounding box only, with its units
grouped into nine lithological classes by litho_classes.py.

  --source 1m    Para 1:1,000,000 (CPRM, 2008). Twelve sheets, no gaps.
                 This is the default, because at the printed width of 13 cm
                 the survey plots at about 1:2,000,000 and finer detail
                 cannot be resolved anyway.
  --source 100k  The 1:100,000 sheets. Only twelve have been published here,
                 so they cover 76.5 per cent of the survey and the rest is
                 left blank. Better for the quantitative tests than for a
                 page-size figure.

On top are the top 250 windows of the published ranking, the twelve known
deposits, and the reference deposit of that case.

    Fig28  Case 1, univariate TMI, reference Paulo Afonso, k=17
    Fig29  Case 2, multivariate TMI + radiometric U, reference Alemao,
           k_TMI=2, k_U=8, alpha_TMI=0.541

Run from inside paper/:

    python make_fig_litho_targets.py --outdir ../docs/Spatial_PCA_paper_overleaf/figures
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
import litho_classes as L
import numpy_repro_concat as R

HERE = Path(__file__).resolve().parent
GEOLOGY = {
    "1m": HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson",
    "100k": HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_100k_carajas.geojson",
}
SOURCE_NOTE = {"1m": "Para 1:1,000,000", "100k": "Para 1:100,000"}
UTM22S = "EPSG:32722"
N_TOP = 250
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
        order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, R.ALPHA_TMI)
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
    parser.add_argument("--source", choices=sorted(GEOLOGY), default="1m")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = study_area()
    area = box(xmin, ymin, xmax, ymax)
    geology = gpd.read_file(GEOLOGY[args.source]).to_crs(UTM22S)
    geology = gpd.clip(geology, area)
    geology = geology[~geology.geometry.is_empty]
    geology["cls"] = geology["LITOTIPOS"].apply(L.classify)
    geology["colour"] = [L.COLOUR.get(c, L.UNMAPPED[1]) for c in geology["cls"]]

    mapped = geology.union_all() if hasattr(geology, "union_all") else geology.unary_union
    print(f"{SOURCE_NOTE[args.source]} covers "
          f"{100 * mapped.area / area.area:.1f}% of the survey")
    present = [c for c in L.ORDER if (geology["cls"] == c).any()]
    deposits = deposit_frames()

    for case, name, ref, title in ((1, "Fig28", 6, "Case 1, univariate TMI, $k=17$"),
                                   (2, "Fig29", 3, "Case 2, TMI + radiometric U")):
        touching, clear, wshape = ranked_windows(case)
        outside = sum(1 for rect in clear if not rect.intersects(mapped))
        fig, ax = plt.subplots(figsize=(FIG_W, 4.55))
        ax.set_facecolor(L.UNMAPPED[1])
        geology.plot(ax=ax, color=geology["colour"], edgecolor="white", linewidth=0.12)

        for rect in clear:
            ax.add_patch(Rectangle((rect.bounds[0], rect.bounds[1]),
                                   rect.bounds[2] - rect.bounds[0],
                                   rect.bounds[3] - rect.bounds[1],
                                   fill=False, ec="black", lw=0.45))
        for rect in touching:
            ax.add_patch(Rectangle((rect.bounds[0], rect.bounds[1]),
                                   rect.bounds[2] - rect.bounds[0],
                                   rect.bounds[3] - rect.bounds[1],
                                   fill=False, ec="#1f3fd6", lw=0.7))
        deposits.boundary.plot(ax=ax, color="white", linewidth=1.3)
        deposits.boundary.plot(ax=ax, color="black", linewidth=0.7)
        deposits.iloc[[ref - 1]].boundary.plot(ax=ax, color="#1f3fd6", linewidth=1.3)

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

        handles = [Rectangle((0, 0), 1, 1, fc=L.COLOUR[c], ec="white", lw=0.3)
                   for c in present]
        labels = [L.LABEL[c] for c in present]
        if mapped.area < 0.999 * area.area:
            handles.append(Rectangle((0, 0), 1, 1, fc=L.UNMAPPED[1], ec="white", lw=0.3))
            labels.append(L.UNMAPPED[0])
        handles += [Line2D([], [], color="black", lw=0.6),
                    Line2D([], [], color="#1f3fd6", lw=0.9),
                    Line2D([], [], color="black", lw=1.0)]
        labels += [f"top {N_TOP}, no known deposit", f"top {N_TOP}, on a known deposit",
                   "known deposit (blue: reference)"]
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.145),
                  ncol=2, frameon=False, fontsize=5.4, handlelength=1.4,
                  handleheight=0.9, labelspacing=0.3, columnspacing=0.8)

        fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.345)
        stem = args.outdir / name
        fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
        fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
        plt.close(fig)
        print(f"wrote {stem}.pdf  ({len(clear)} clear, {len(touching)} on deposits, "
              f"{outside} clear windows on unmapped ground)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
