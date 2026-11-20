"""Figure: high-ranking windows on the mapped lithology, both cases (R2.6).

Two panels on one page-width figure. The background is the Para 1:1,000,000
sheet of the Servico Geologico do Brasil (CPRM, 2008), held in GeoSGB and
retrieved for the study-area bounding box only, with its units grouped into
nine lithological classes by litho_classes.py. That grouping is ours, not the
survey's. The 1:1,000,000 sheet is used rather than the 1:100,000 sheets
because only twelve of the latter have been published here, covering 76.5 per
cent of the survey, and because at the printed width the survey plots at about
1:2,000,000 and finer detail cannot be resolved.

    (a) Case 1, univariate TMI, reference Paulo Afonso, k=17
    (b) Case 2, multivariate TMI + radiometric U, reference Alemao,
        k_TMI=2, k_U=8, alpha_TMI=0.503

Run from inside paper/:

    python make_fig_litho_combined.py --outdir ../docs/Spatial_PCA_paper_overleaf/figures
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

import litho_classes as L
import numpy_repro_concat as R
from make_fig_geology_targets import ERA_COLOUR, ERA_ORDER
from make_fig_litho_targets import deposit_frames, ranked_windows, study_area

HERE = Path(__file__).resolve().parent
GEOLOGY = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson"
UTM22S = "EPSG:32722"
N_TOP = 250
FIG_W = 5.14                      # \textwidth = 372 pt

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.0})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / "figures_out")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--name", default="Fig26")
    parser.add_argument("--colour", choices=("lithology", "age"), default="lithology")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    xmin, ymin, xmax, ymax = study_area()
    geology = gpd.read_file(GEOLOGY).to_crs(UTM22S)
    geology = gpd.clip(geology, box(xmin, ymin, xmax, ymax))
    geology = geology[~geology.geometry.is_empty]
    if args.colour == "lithology":
        geology["cls"] = geology["LITOTIPOS"].apply(L.classify)
        colour = {c: L.COLOUR[c] for c in L.ORDER}
        label = {c: L.LABEL[c] for c in L.ORDER}
        order = L.ORDER
    else:
        geology["cls"] = geology["ERA_MAX"].astype(str)
        colour = {e: ERA_COLOUR[e][0] for e in ERA_ORDER}
        label = {e: ERA_COLOUR[e][1] for e in ERA_ORDER}
        order = ERA_ORDER
    geology["colour"] = [colour.get(c, L.UNMAPPED[1]) for c in geology["cls"]]
    present = [c for c in order if (geology["cls"] == c).any()]
    deposits = deposit_frames()

    panel_h = (FIG_W - 0.62) / ((xmax - xmin) / (ymax - ymin))
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 2 * panel_h + 1.55))

    for ax, (case, ref, title) in zip(axes, ((1, 6, "(a) Case 1, univariate TMI, $k=17$"),
                                             (2, 3, "(b) Case 2, TMI + radiometric U"))):
        touching, clear, wshape = ranked_windows(case)
        ax.set_facecolor(L.UNMAPPED[1])
        geology.plot(ax=ax, color=geology["colour"], edgecolor="white", linewidth=0.15)
        for rect in clear:
            ax.add_patch(Rectangle((rect.bounds[0], rect.bounds[1]),
                                   rect.bounds[2] - rect.bounds[0],
                                   rect.bounds[3] - rect.bounds[1],
                                   fill=False, ec="black", lw=0.42))
        for rect in touching:
            ax.add_patch(Rectangle((rect.bounds[0], rect.bounds[1]),
                                   rect.bounds[2] - rect.bounds[0],
                                   rect.bounds[3] - rect.bounds[1],
                                   fill=False, ec="#1f3fd6", lw=0.65))
        deposits.boundary.plot(ax=ax, color="white", linewidth=1.2)
        deposits.boundary.plot(ax=ax, color="black", linewidth=0.65)
        deposits.iloc[[ref - 1]].boundary.plot(ax=ax, color="#1f3fd6", linewidth=1.2)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.arange(450000, 700001, 50000))
        ax.set_yticks(np.arange(9240000, 9360001, 40000))
        ax.set_xticklabels([f"{v/1000:.0f}" for v in ax.get_xticks()])
        ax.set_yticklabels([f"{v/1000:.0f}" for v in ax.get_yticks()])
        ax.set_ylabel("Northing (km)")
        ax.set_title(f"{title}; window {wshape[0]} $\\times$ {wshape[1]} px", pad=2.5)
        ax.tick_params(length=2, width=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
    axes[1].set_xlabel("Easting (km)")
    axes[0].set_xticklabels([])

    handles = [Rectangle((0, 0), 1, 1, fc=colour[c], ec="white", lw=0.3) for c in present]
    labels = [label[c] for c in present]
    handles += [Line2D([], [], color="black", lw=0.6),
                Line2D([], [], color="#1f3fd6", lw=0.9),
                Line2D([], [], color="black", lw=1.0)]
    labels += [f"top {N_TOP}, no known deposit", f"top {N_TOP}, on a known deposit",
               "known deposit (blue: reference)"]
    axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                   ncol=2, frameon=False, fontsize=5.5, handlelength=1.4,
                   handleheight=0.9, labelspacing=0.3, columnspacing=0.8)

    fig.subplots_adjust(left=0.088, right=0.988, top=0.957, bottom=0.225, hspace=0.115)
    stem = args.outdir / args.name
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
