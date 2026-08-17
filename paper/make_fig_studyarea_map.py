"""Study-area location map (new figure, Adel's review comment, 2026-08).

Two panels:
  (a) Brazil on the NASA Blue Marble composite, with country/state borders,
      Para highlighted, and the study area as a red rectangle.
  (b) The study area itself on a satellite-image background, with the search
      polygon and the two reference deposits (Paulo Afonso, Alemao) as points.

Background for (b): Esri World Imagery fetched from the ArcGIS export REST
endpoint at the exact UTM extent (requires network; works on the Mac).
Where that host is unreachable (the cloud sandbox blocks tile servers), the
panel falls back to an upsampled Blue Marble crop and says so in the credit
line. Rerun this script with network access to swap in the full-res imagery.

Inputs (auto-located relative to this script):
  data/Carajas_Brazil_Univariate_TMI/Demo_area_polygon.shp     search polygon
  data/Carajas_Brazil_Univariate_TMI/Prospect_in Carajas_v2.shp deposit outlines
  data/naturalearth/ne_50m_admin_0_countries.geojson  (auto-downloaded if absent)
  data/naturalearth/ne_50m_admin_1_states_provinces.geojson
  NASA Blue Marble bmng.jpg from the pip package `basemap-data`

Output: docs/Spatial_PCA_paper_overleaf/figures/study_area_location_map.{png,pdf}
"""
from __future__ import annotations

import importlib.util
import io
import os
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Rectangle
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = Path(__file__).resolve().parent
for _cand in (HERE, HERE.parent):
    if (_cand / "data" / "Carajas_Brazil_Univariate_TMI").exists():
        ROOT = _cand
        break
else:  # pragma: no cover
    raise FileNotFoundError("Cannot locate the repo data/ directory.")

DATA = ROOT / "data"
FIGDIR_CANDIDATES = (
    ROOT / "docs" / "Spatial_PCA_paper_overleaf" / "figures",
    HERE.parent / "docs" / "Spatial_PCA_paper_overleaf" / "figures",
)
FIGDIR = next((p for p in FIGDIR_CANDIDATES if p.parent.exists()), FIGDIR_CANDIDATES[0])
FIGDIR.mkdir(parents=True, exist_ok=True)

NE_DIR = DATA / "naturalearth"
NE_DIR.mkdir(parents=True, exist_ok=True)
NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
NE_FILES = {
    "countries": "ne_50m_admin_0_countries.geojson",
    "states": "ne_50m_admin_1_states_provinces.geojson",
}

UTM = "EPSG:32722"  # SIRGAS-era UTM 22S used by the grids
ESRI_EXPORT = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/export?bbox={xmin},{ymin},{xmax},{ymax}&bboxSR={sr}"
    "&imageSR={sr}&size={w},{h}&format=jpg&f=image"
)

REFERENCE_COLOR = "#8b12c9"  # purple = reference deposit, as in Figure 1


def _ne_layer(key: str) -> gpd.GeoDataFrame:
    path = NE_DIR / NE_FILES[key]
    if not path.exists():
        urllib.request.urlretrieve(f"{NE_BASE}/{NE_FILES[key]}", path)
    return gpd.read_file(path)


def _blue_marble() -> np.ndarray:
    """NASA Blue Marble global composite — offline fallback only."""
    spec = importlib.util.find_spec("mpl_toolkits.basemap_data")
    if spec is None:
        raise ImportError(
            "No network route to Esri World Imagery AND basemap-data is not "
            "installed. Either run with network access, or `pip install "
            "basemap-data` for the offline Blue Marble fallback."
        )
    p = Path(spec.submodule_search_locations[0]) / "bmng.jpg"
    return np.asarray(Image.open(p))


def _crop_global(img: np.ndarray, lon0, lon1, lat0, lat1) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int((lon0 + 180.0) / 360.0 * w)
    x1 = int((lon1 + 180.0) / 360.0 * w)
    y0 = int((90.0 - lat1) / 180.0 * h)
    y1 = int((90.0 - lat0) / 180.0 * h)
    return img[max(y0, 0) : y1, max(x0, 0) : x1]


def _esri_image(xmin, ymin, xmax, ymax, w=4200, h=2300, sr=32722):
    url = ESRI_EXPORT.format(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, w=w, h=h, sr=sr)
    with urllib.request.urlopen(url, timeout=90) as r:
        return np.asarray(Image.open(io.BytesIO(r.read())))


def main() -> Path:
    demo = gpd.read_file(DATA / "Carajas_Brazil_Univariate_TMI/Demo_area_polygon.shp").to_crs(UTM)
    deps = gpd.read_file(DATA / "Carajas_Brazil_Univariate_TMI/Prospect_in Carajas_v2.shp").to_crs(UTM)
    refs = {
        "Paulo Afonso": deps[deps["Name"] == "Paulo Afonso"].geometry.union_all().centroid,
        "Alemão": deps[deps["Name"] == "Alemao"].geometry.union_all().centroid,
    }

    xmin, ymin, xmax, ymax = demo.total_bounds
    demo_ll = demo.to_crs("EPSG:4326")
    gxmin, gymin, gxmax, gymax = demo_ll.total_bounds

    countries = _ne_layer("countries")
    states = _ne_layer("states")
    brazil = countries[countries["ADMIN"] == "Brazil"]
    para = states[(states["admin"] == "Brazil") & (states["name"].isin(["Pará", "Para"]))]

    fig = plt.figure(figsize=(13.2, 5.4), dpi=200)
    ax_a = fig.add_axes([0.045, 0.10, 0.33, 0.84])
    ax_b = fig.add_axes([0.455, 0.10, 0.535, 0.84])

    # ---------------------------------------------------------- panel (a)
    LON0, LON1, LAT0, LAT1 = -76.0, -32.0, -35.0, 8.0
    try:
        img_a = _esri_image(LON0, LAT0, LON1, LAT1, w=2600, h=2540, sr=4326)
        interp_a = "none"
    except Exception:
        img_a = _crop_global(_blue_marble(), LON0, LON1, LAT0, LAT1)
        interp_a = "bilinear"
    ax_a.imshow(
        img_a,
        extent=(LON0, LON1, LAT0, LAT1),
        origin="upper",
        interpolation=interp_a,
    )
    countries.boundary.plot(ax=ax_a, color="white", linewidth=0.5, alpha=0.75)
    brazil.boundary.plot(ax=ax_a, color="white", linewidth=1.1)
    para.boundary.plot(ax=ax_a, color="#ffd21e", linewidth=1.1)
    cx = 0.5 * (gxmin + gxmax)
    cy = 0.5 * (gymin + gymax)
    ax_a.add_patch(
        Rectangle(
            (gxmin, gymin),
            gxmax - gxmin,
            gymax - gymin,
            fill=False,
            edgecolor="red",
            linewidth=1.6,
            zorder=6,
        )
    )
    ax_a.annotate(
        "Study area\n(Carajás Mineral Province)",
        xy=(cx, cy - 0.8),
        xytext=(-66.5, -15.5),
        color="white",
        fontsize=10,
        ha="left",
        va="top",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        arrowprops={"arrowstyle": "-", "color": "white", "linewidth": 1.0},
        zorder=7,
    )
    ax_a.text(
        -46.5,
        -11.5,
        "BRAZIL",
        color="white",
        fontsize=13,
        fontweight="bold",
        ha="center",
        path_effects=[pe.withStroke(linewidth=2.4, foreground="black")],
    )
    ax_a.text(
        -52.6,
        -4.6,
        "Pará",
        color="#ffd21e",
        fontsize=10,
        ha="center",
        path_effects=[pe.withStroke(linewidth=2.0, foreground="black")],
    )
    ax_a.set_xlim(LON0, LON1)
    ax_a.set_ylim(LAT0, LAT1)
    ax_a.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (LAT0 + LAT1))))
    ax_a.set_xlabel("Longitude (°)", fontsize=10)
    ax_a.set_ylabel("Latitude (°)", fontsize=10)
    ax_a.tick_params(labelsize=8)
    ax_a.set_title("(a) Location in Brazil", fontsize=12)

    # ---------------------------------------------------------- panel (b)
    pad = 12_000.0
    bxmin, bxmax = xmin - pad, xmax + pad
    bymin, bymax = ymin - pad, ymax + pad
    credit = "Basemap: Esri World Imagery"
    try:
        img_b = _esri_image(bxmin, bymin, bxmax, bymax)
        interp = "none"
    except Exception:
        b_ll = (
            gpd.GeoSeries.from_wkt(
                [
                    f"POLYGON(({bxmin} {bymin},{bxmax} {bymin},{bxmax} {bymax},"
                    f"{bxmin} {bymax},{bxmin} {bymin}))"
                ],
                crs=UTM,
            )
            .to_crs("EPSG:4326")
            .total_bounds
        )
        img_b = _crop_global(_blue_marble(), b_ll[0], b_ll[2], b_ll[1], b_ll[3])
        interp = "bilinear"
        credit = "Basemap: NASA Blue Marble (fallback — rerun with network for Esri World Imagery)"
    ax_b.imshow(
        img_b,
        extent=(bxmin, bxmax, bymin, bymax),
        origin="upper",
        interpolation=interp,
        zorder=1,
    )
    demo.boundary.plot(ax=ax_b, color="red", linewidth=2.0, zorder=5)
    for i, (name, pt) in enumerate(refs.items()):
        ax_b.plot(
            pt.x,
            pt.y,
            marker="*",
            ms=17,
            mfc=REFERENCE_COLOR,
            mec="white",
            mew=1.1,
            linestyle="none",
            zorder=7,
        )
        dy = 6_000 if i == 0 else -12_500
        ax_b.annotate(
            name,
            xy=(pt.x, pt.y),
            xytext=(pt.x + 5_000, pt.y + dy),
            color="white",
            fontsize=11,
            fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="black")],
            zorder=8,
        )
    # scale bar (50 km)
    L = 50_000.0
    sx0 = bxmin + 0.055 * (bxmax - bxmin)
    sy0 = bymin + 0.075 * (bymax - bymin)
    ax_b.plot([sx0, sx0 + L], [sy0, sy0], color="white", lw=3.5, solid_capstyle="butt", zorder=9)
    ax_b.text(
        sx0 + 0.5 * L,
        sy0 + 0.018 * (bymax - bymin),
        "50 km",
        color="white",
        ha="center",
        fontsize=10,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        zorder=9,
    )
    # north arrow
    nx = bxmin + 0.965 * (bxmax - bxmin)
    ny = bymin + 0.83 * (bymax - bymin)
    ax_b.add_patch(
        FancyArrow(
            nx,
            ny,
            0,
            0.09 * (bymax - bymin),
            width=1_300,
            head_width=5_200,
            head_length=6_500,
            color="white",
            zorder=9,
        )
    )
    ax_b.text(
        nx,
        ny - 0.045 * (bymax - bymin),
        "N",
        color="white",
        ha="center",
        fontsize=11,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=2.2, foreground="black")],
        zorder=9,
    )
    ax_b.text(
        0.995,
        0.012,
        credit,
        transform=ax_b.transAxes,
        color="white",
        fontsize=6.5,
        ha="right",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=1.6, foreground="black")],
        zorder=9,
    )
    handles = [
        plt.Line2D([], [], color="red", lw=2.0, label="Search area"),
        plt.Line2D(
            [],
            [],
            marker="*",
            ms=13,
            mfc=REFERENCE_COLOR,
            mec="white",
            linestyle="none",
            label="Reference deposits",
        ),
    ]
    ax_b.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.85)
    ax_b.set_xlim(bxmin, bxmax)
    ax_b.set_ylim(bymin, bymax)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Easting (m)", fontsize=10)
    ax_b.set_ylabel("Northing (m)", fontsize=10)
    ax_b.tick_params(labelsize=8)
    ax_b.ticklabel_format(style="plain")
    ax_b.set_title("(b) Study area and reference deposits", fontsize=12)

    out_png = FIGDIR / "study_area_location_map.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png} (+.pdf) | {credit}")
    return out_png


if __name__ == "__main__":
    main()
