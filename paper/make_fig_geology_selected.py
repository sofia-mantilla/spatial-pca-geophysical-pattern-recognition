"""Mapped geology at full colour inside the selection, pale outside it.

The treatment Figure 21 gives the rasters: the background is the same map at
low saturation, and the geology inside the top-250 windows is drawn over it at
full colour, so the eye reads which units the ranking selected rather than
which units are large.

Run from inside paper/:

    python make_fig_geology_selected.py --outdir OUT --colour age
    python make_fig_geology_selected.py --outdir OUT --colour lithology --case 1
    python make_fig_geology_selected.py --outdir OUT --compare    # Case 2, both rules

Writes <name>.pdf and <name>.png.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from shapely.geometry import box
from shapely.ops import unary_union

import case1_uni_repro as C1
import litho_classes as L
import numpy_repro_concat as R
from make_fig_geology_targets import ERA_COLOUR, ERA_ORDER
from make_fig_litho_targets import deposit_frames, study_area

HERE = Path(__file__).resolve().parent
GEOLOGY = HERE.parent / "data" / "Carajas_Brazil_Geology" / "para_1m_carajas.geojson"
ARIM_LITO = (HERE.parent / "data" / "Carajas_Brazil_Structures"
             / "sig_arim_carajas_versao_preliminar_vr" / "Arim_Carajas"
             / "Litologia" / "arim_carajas_lito.shp")
N_LEGEND = 12          # ARIM units named in the legend, the largest by area
STRUCTURES = HERE.parent / "data" / "Carajas_Brazil_Structures" / "arim_estrutura.shp"
MAGNETIC = "Lineamento Magnetico"   # drawn, never measured against
SHEAR_COLOUR = "#b5003c"
LINEAMENT_COLOUR = "#0b6e6e"
UTM22S = "EPSG:32722"
N_TOP = 250
DIM = 0.22          # background saturation, as Figure 21 dims its raster to 0.35


def window_rects(case, scale=R.BLOCK_SCALE, alpha=None):
    """Top-250 rectangles and the reference-deposit index."""
    if case == 1:
        order, idx, wshape, origin, dep = C1.rank_spca(17)
    else:
        a = R.ALPHA_TMI if alpha is None else alpha
        order, idx, wshape, origin, dep = R.rank_concat(3, 2, 8, a, scale=scale)
    ox, oy = origin
    wh, ww = wshape
    out = []
    for i in [w for w in order if w != dep][:N_TOP]:
        r, c, _ = idx[i]
        x1, y2 = ox + c * R.PX, oy - r * R.PX
        out.append(box(x1, y2 - wh * R.PX, x1 + ww * R.PX, y2))
    return out


def load_arim():
    """The ARIM Carajas 1:250,000 lithology in the survey's own colours.

    No classification of ours: the polygons are the survey's and the colours
    come from the QGIS style file shipped beside the shapefile, keyed on
    SIGLA_UNID, so the base map is the survey's own rendering.
    """
    qml = ARIM_LITO.with_suffix(".qml")
    q = qml.read_text(encoding="utf-8", errors="ignore")
    cats = re.findall(r'<category value="([^"]*)"\s+symbol="(\d+)"', q)
    syms = {}
    for blk in q.split("<symbol ")[1:]:
        nm = re.search(r'name="([^"]*)"', blk)
        cl = re.search(r'<prop k="color" v="([^"]*)"', blk)
        if nm and cl:
            syms[nm.group(1)] = cl.group(1)
    col = {}
    for val, sym in cats:
        rgba = syms.get(sym)
        if rgba:
            col[val] = tuple(int(v) / 255 for v in rgba.split(",")[:3])

    xmin, ymin, xmax, ymax = study_area()
    g = gpd.clip(gpd.read_file(ARIM_LITO).to_crs(UTM22S), box(xmin, ymin, xmax, ymax))
    g = g[~g.geometry.is_empty]
    g["cls"] = g["SIGLA_UNID"].astype(str)
    g["colour"] = [col.get(s, (0.85, 0.85, 0.85)) for s in g["cls"]]
    name = dict(zip(g["cls"], g["NOME_UNIDA"].astype(str)))
    by_area = (g.assign(a=g.geometry.area).groupby("cls")["a"].sum()
               .sort_values(ascending=False))
    order = list(by_area.index)
    colour = {c: col.get(c, (0.85, 0.85, 0.85)) for c in order}
    label = {c: name.get(c, c) for c in order}
    print("ARIM: %d polygons, %d units, %d with a style colour"
          % (len(g), len(order), sum(1 for c in order if c in col)))
    return g, colour, label, order, (xmin, ymin, xmax, ymax)


def load_geology(colour_by):
    xmin, ymin, xmax, ymax = study_area()
    g = gpd.read_file(GEOLOGY).to_crs(UTM22S)
    g = gpd.clip(g, box(xmin, ymin, xmax, ymax))
    g = g[~g.geometry.is_empty]
    if colour_by == "lithology":
        g["cls"] = g["LITOTIPOS"].apply(L.classify)
        colour = {c: L.COLOUR[c] for c in L.ORDER}
        label = {c: L.LABEL[c] for c in L.ORDER}
        order = L.ORDER
    else:
        g["cls"] = g["ERA_MAX"].astype(str)
        colour = {e: ERA_COLOUR[e][0] for e in ERA_ORDER}
        label = {e: ERA_COLOUR[e][1] for e in ERA_ORDER}
        order = ERA_ORDER
    g["colour"] = [colour.get(c, L.UNMAPPED[1]) for c in g["cls"]]
    return g, colour, label, order, (xmin, ymin, xmax, ymax)


def load_structures(extent, lineaments=True):
    """Mapped shear zones, the named systems, and the magnetic lineaments.

    The lineaments are interpretations of the same aeromagnetic survey the
    ranking is built on. Drawing them is context; measuring against them would
    be circular, which is why run_structure_proximity.py excludes them.
    """
    xmin, ymin, xmax, ymax = extent
    st = gpd.read_file(STRUCTURES).to_crs(UTM22S)
    st = gpd.clip(st, box(xmin, ymin, xmax, ymax))
    st = st[~st.geometry.is_empty]
    mag = st["TIPO_ESTRU"].str.contains("Magn", case=False, na=False)
    mapped = st[~mag]
    shear = mapped[mapped["TIPO_ESTRU"].str.contains("cisalhamento", case=False, na=False)]
    named = mapped[mapped["NOME_ESTRU"].notna()]
    print("%d structures: %d shear zones, %d magnetic lineaments, %d named"
          % (len(st), len(shear), int(mag.sum()), len(named)))
    return shear, named, (st[mag] if lineaments else None)


def draw_structures(ax, struct, extent):
    shear, named, lineaments = struct
    xmin, ymin, xmax, ymax = extent
    if lineaments is not None and len(lineaments):
        lineaments.plot(ax=ax, color="white", linewidth=1.6, alpha=0.85, zorder=4.4)
        lineaments.plot(ax=ax, color=LINEAMENT_COLOUR, linewidth=0.7,
                        linestyle=(0, (3.5, 2.0)), alpha=0.9, zorder=4.5)
    if shear is not None and len(shear):
        shear.plot(ax=ax, color="white", linewidth=2.2, alpha=0.9, zorder=4.6)
        shear.plot(ax=ax, color=SHEAR_COLOUR, linewidth=1.1, zorder=4.7)
        for name, grp in named.groupby("NOME_ESTRU"):
            c = grp.geometry.union_all().centroid
            if not (xmin < c.x < xmax and ymin < c.y < ymax):
                continue
            ax.annotate(str(name).replace("Zona de Cisalhamento ", "ZC ")
                        .replace("Lineamento ", "Lin. ").replace("Falha ", "F. "),
                        (c.x, c.y), fontsize=6.5, color=SHEAR_COLOUR, ha="center",
                        va="center", fontweight="bold", zorder=5,
                        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])


def draw(ax, g, rects, present, colour, extent, title, struct=None):
    """Dim map, then the same map clipped to the windows at full colour."""
    xmin, ymin, xmax, ymax = extent
    # 1. the whole map, desaturated
    for c in present:
        sub = g[g["cls"] == c]
        sub.plot(ax=ax, color=colour.get(c, L.UNMAPPED[1]), lw=0.1,
                 edgecolor="white", alpha=DIM, zorder=1)
    # 2. the geology the ranking selected, at full colour
    union = unary_union(rects)
    bright = gpd.clip(g, union)
    bright = bright[~bright.geometry.is_empty]
    for c in present:
        sub = bright[bright["cls"] == c]
        if len(sub):
            sub.plot(ax=ax, color=colour.get(c, L.UNMAPPED[1]), lw=0, zorder=3)
    # 3. window outlines, thin, so the shapes stay readable
    for r in rects:
        x0, y0, x1, y1 = r.bounds
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="#2a2a2a", lw=0.35, zorder=4))
    if struct is not None:
        draw_structures(ax, struct, extent)
    deposit_frames().boundary.plot(ax=ax, color="white", lw=2.0, zorder=5.8)
    deposit_frames().boundary.plot(ax=ax, color="black", lw=1.0, zorder=6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(450000, 700001, 50000))
    ax.set_yticks(np.arange(9240000, 9360001, 40000))
    ax.set_xticklabels(["%d" % (t / 1000) for t in ax.get_xticks()])
    ax.set_yticklabels(["%d" % (t / 1000) for t in ax.get_yticks()])
    ax.set_ylabel("Northing (km)")
    ax.set_title(title, loc="left")
    return bright


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=HERE / "figures_out")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--colour", choices=("lithology", "age", "arim"), default="age",
                    help="'arim' is the ARIM 1:250,000 map in the survey's own colours")
    ap.add_argument("--case", type=int, choices=(1, 2), default=None,
                    help="one case only; default draws both")
    ap.add_argument("--compare", action="store_true",
                    help="Case 2 under the published rule and the scaled rule")
    ap.add_argument("--structures", action="store_true",
                    help="overlay the ARIM shear zones and magnetic lineaments")
    ap.add_argument("--no-lineaments", dest="lineaments", action="store_false",
                    help="with --structures, omit the magnetic lineaments")
    ap.add_argument("--paper", action="store_true",
                    help="size at \\textwidth with 6.5 pt type, for the manuscript")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)

    if a.colour == "arim":
        g, colour, label, order, extent = load_arim()
    else:
        g, colour, label, order, extent = load_geology(a.colour)
    present = [c for c in order if (g["cls"] == c).any()]
    legend_units = present
    if a.colour == "arim":
        # Name the units the enrichment table discusses, so a reader can move
        # between the two. Falls back to the largest by area if it has not run.
        ej = HERE / "results" / "case2_unit_enrichment.json"
        named = []
        if ej.exists():
            rec = json.loads(ej.read_text())["units"]
            named = [sg for sg, r in rec.items()
                     if r["survey_share_pct"] >= 1.0 and r["case2"].get("above_random")]
            named = [sg for sg in present if sg in named]
        legend_units = named or present[:N_LEGEND]
    struct = load_structures(extent, a.lineaments) if a.structures else None

    if a.compare:
        panels = [("(a) Case 2, published rule ($\\alpha_{\\mathrm{TMI}}=0.503$)",
                   window_rects(2, scale=None, alpha=0.503)),
                  ("(b) Case 2, scaled rule ($\\alpha_{\\mathrm{TMI}}=0.541$)",
                   window_rects(2))]
        stem = a.name or "geology_highlight_compare_%s" % a.colour
        if a.structures:
            stem += "_struct"
    elif a.case:
        panels = [("Case %d" % a.case, window_rects(a.case))]
        stem = a.name or "geology_highlight_case%d_%s" % (a.case, a.colour)
    else:
        panels = [("(a) Case 1, univariate TMI, $k=17$", window_rects(1)),
                  ("(b) Case 2, TMI + radiometric U", window_rects(2))]
        stem = a.name or "geology_highlight_%s" % a.colour
        if a.structures:
            stem += "_struct"

    if a.paper:
        plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.0,
                             "pdf.fonttype": 42, "ps.fonttype": 42})
    fig_w = 5.14 if a.paper else 10.5
    xmin, ymin, xmax, ymax = extent
    panel_h = (fig_w - 0.62) / ((xmax - xmin) / (ymax - ymin))
    nrow_legend = -(-(len(legend_units) + 4) // (2 if a.paper else 3))
    leg_h = (0.135 if a.paper else 0.28) * nrow_legend + 0.12
    h = (panel_h if a.paper else 4.6) * len(panels) + leg_h + (0.25 if a.paper else 0)
    fig, axes = plt.subplots(len(panels), 1, figsize=(fig_w, h), squeeze=False)
    for j, (ax, (title, rects)) in enumerate(zip(axes.flat, panels)):
        bright = draw(ax, g, rects, present, colour, extent, title, struct)
        if j < len(panels) - 1:
            ax.set_xticklabels([])
        shares = (bright.area.groupby(bright["cls"]).sum() / bright.area.sum() * 100)
        top = shares.sort_values(ascending=False).head(3)
        print("%s: selected area by class -> %s" % (
            title, ", ".join("%s %.1f%%" % (label.get(k, k)[:28], v) for k, v in top.items())))
    axes.flat[-1].set_xlabel("Easting (km)")

    handles = [Rectangle((0, 0), 1, 1, fc=colour.get(c, L.UNMAPPED[1])) for c in legend_units]
    labels = [label.get(c, c) for c in legend_units]
    if len(legend_units) < len(present):
        handles.append(Rectangle((0, 0), 1, 1, fc="#d9d9d9"))
        labels.append("other mapped units (%d)" % (len(present) - len(legend_units)))
    handles += [Line2D([0], [0], color="#2a2a2a", lw=0.6),
                Line2D([0], [0], color="black", lw=1.2)]
    labels += ["top 250 window", "known deposit"]
    if struct is not None:
        handles.append(Line2D([0], [0], color=SHEAR_COLOUR, lw=1.3))
        labels.append("mapped shear zone (ARIM, 1:250,000)")
        if struct[2] is not None and len(struct[2]):
            handles.append(Line2D([0], [0], color=LINEAMENT_COLOUR, lw=0.9,
                                  linestyle=(0, (3.5, 2.0))))
            labels.append("magnetic lineament (drawn, not measured against)")
    fig.legend(handles, labels, loc="lower center",
               ncol=2 if a.paper else 3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               fontsize=6.0 if a.paper else None)
    fig.tight_layout(rect=(0, leg_h / h, 1, 1),
                     h_pad=1.4 if a.paper else 1.0)
    pdf, png = a.outdir / (stem + ".pdf"), a.outdir / (stem + ".png")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=a.dpi, facecolor="white")
    plt.close(fig)
    print("wrote %s and %s" % (pdf, png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
