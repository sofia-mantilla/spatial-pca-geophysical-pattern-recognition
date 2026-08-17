"""Rebuild Figure 18 (fig:multiprofiles) from data at publication resolution.

Content matches the figure it replaces (which was a raster lifted from v9 image18 and
had no generating script): centred south-north profiles for Deposits 1-5 and each
deposit's TMI and radiometric-U windows.

Differences from the raster it replaces, all deliberate:
  * regular grid, no dead space, legible type at printed size;
  * windows drawn on a kilometre extent so deposit footprints are physically comparable;
  * radiometric-U on the paper's shared 0-15 limits (config analysis_defaults), one
    colourbar for the row;
  * TMI uses the project's canonical per-deposit limits, read from
    configs/carajas_uni_tmi.yaml (visualization.deposit_limits_tmi) - the same limits
    the pipeline plots with, so panel colours match the rest of the paper. They are
    asymmetric by design: raw TMI carries a regional baseline, so zero is not neutral.
    A shared TMI scale is NOT used either: amplitudes span 8x across the five deposits,
    so a common scale would flatten the Deposit 3 dipole. Each panel carries its own
    colourbar.

Run from the worktree root (or anywhere numpy_repro_concat is importable):
    python make_fig_multiprofiles.py --outdir docs/Spatial_PCA_paper_overleaf/figures
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

import numpy_repro_concat as R
from extract_dep_windows import collect, DEPS

# ---------------------------------------------------------------- conventions
# Same rainbow ramp used by every other figure in the paper (paper_figures_corrected.py).
PAPER_CMAP = LinearSegmentedColormap.from_list("spatial_pca_paper", (
    (0.00, "#0101ff"), (0.167, "#2dc4ff"), (0.334, "#26ff00"), (0.501, "#ffe101"),
    (0.668, "#ff0109"), (0.835, "#ef00ff"), (1.00, "#de9eff")))
U_LIMS = (0.0, 15.0)          # config analysis_defaults vmin_var2/vmax_var2
REF_DEP = 3                   # Alemao, the reference deposit
DEP_NAMES = {1: "Tucuma", 2: "Pedra Branca", 3: "Alemao", 4: "Furnas", 5: "Salobo"}
CURVE = {1: "#1f4fd8", 2: "#e07b00", 3: "#c62828", 4: "#2e9e2e", 5: "#8e24aa"}

mpl.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,       # embed TrueType, not Type 3 - journal requirement
    "ps.fonttype": 42,
})


def _find_config():
    """Locate configs/carajas_uni_tmi.yaml from the worktree or the repo root."""
    here = Path(__file__).resolve().parent
    for base in (here, here.parent, here.parent.parent):
        c = base / "configs" / "carajas_uni_tmi.yaml"
        if c.exists():
            return c
    return None


def deposit_limits_tmi():
    """Per-deposit TMI display limits from the project config.

    These are the paper's canonical limits (visualization.deposit_limits_tmi) and are
    ASYMMETRIC - raw TMI carries a regional baseline, so zero is not a neutral value
    and a symmetric scale would recolour every panel. Read from the config rather than
    hard-coded so the figures track it.
    """
    cfg = _find_config()
    if cfg is None:
        return {}
    out, inside = {}, False
    for line in cfg.read_text().splitlines():
        if re.match(r"^\s*deposit_limits_tmi:\s*$", line):
            inside = True
            continue
        if inside:
            m = re.match(r'^\s+"(\d+)":\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]', line)
            if m:
                out[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
                continue
            if line.strip() and not line.startswith((" ", "\t")):
                break
            if line.strip() and not m:
                break
    return out


_LIMITS = deposit_limits_tmi()


def tmi_limits(dep, M):
    """(vmin, vmax) for a deposit: config limits, else a symmetric 98th-percentile fallback."""
    if dep in _LIMITS:
        return _LIMITS[dep]
    v = float(np.percentile(np.abs(M), 98))
    v = max(50.0, 50.0 * np.ceil(v / 50.0))
    return (-v, v)


def km_extent(shape):
    h, w = shape
    dy, dx = h * R.PX / 2000.0, w * R.PX / 2000.0
    return (-dx, dx, -dy, dy)



# ---------------------------------------------------------------- layout constants
FIG_W, FIG_H = 7.2, 5.45
L, RGT = 0.105, 0.895          # drawing margins (figure fraction)
COL_W = 0.136                  # width of one deposit column
GAP = (RGT - L - 5 * COL_W) / 4.0

PROF_TOP, PROF_H = 0.895, 0.195        # profile axes: top edge and height
LEGEND_Y = 0.645                       # shared legend, below both profile panels
TMI_BOT, TMI_TITLE_Y = 0.392, 0.575    # TMI window row: image baseline, title baseline
CBAR_Y, CBAR_H = 0.305, 0.016          # TMI colourbar strip
U_BOT = 0.095                          # radiometric-U window row (named once, on top row)


def col_x(j):
    return L + j * (COL_W + GAP)


def image_box(ext, x0, y_bottom):
    """Axes rect that renders `ext` at true aspect, bottom-aligned in its column."""
    w_data, h_data = ext[1] - ext[0], ext[3] - ext[2]
    h = COL_W * (FIG_W / FIG_H) * (h_data / w_data)
    return [x0, y_bottom, COL_W, h]


def build(data, outdir: Path, stem: str, dpi: int, normalize_profiles: bool = False):
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---------------------------------------------------------- profiles (top)
    pw = (RGT - L - 0.085) / 2.0
    ax_t = fig.add_axes([L, PROF_TOP - PROF_H, pw, PROF_H])
    ax_u = fig.add_axes([L + pw + 0.085, PROF_TOP - PROF_H, pw, PROF_H])
    if normalize_profiles:
        ylabs = ("TMI, centred and scaled (s.d.)",
                 "Radiometric U, centred and scaled (s.d.)")
    else:
        ylabs = ("TMI, centred (nT)", "Radiometric U, centred (ppm eU)")
    for ax, key, ylab in ((ax_t, "prof_tmi", ylabs[0]), (ax_u, "prof_u", ylabs[1])):
        for d in DEPS:
            v = data[d]
            y = v[key]
            if normalize_profiles:
                s = float(np.std(y))
                y = y / s if s > 0 else y
            ref = d == REF_DEP
            ax.plot(v["prof_d"], y, color=CURVE[d],
                    lw=1.7 if ref else 1.0, zorder=3 if ref else 2,
                    label=f"{d} {DEP_NAMES[d]}" + (" - reference" if ref else ""))
        ax.axhline(0, color="#9e9e9e", lw=0.5, zorder=1)
        ax.axvline(0, color="#9e9e9e", lw=0.5, zorder=1)
        ax.set_xlabel("Distance along south-north line, centred (km)", labelpad=2)
        ax.set_ylabel(ylab, labelpad=2)
        ax.grid(alpha=0.25, lw=0.4)
        ax.margins(x=0.02)
        for s_ in ax.spines.values():
            s_.set_linewidth(0.6)
    ax_t.set_title("Centred south-north profiles - TMI", pad=4)
    ax_u.set_title("Centred south-north profiles - radiometric U", pad=4)
    # one shared legend under both panels, so no curve is hidden by a legend box
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, LEGEND_Y),
               ncol=5, frameon=False, fontsize=7, handlelength=1.5,
               columnspacing=1.6, handletextpad=0.5)

    # ---------------------------------------------------------- window rows
    imu = None
    for j, d in enumerate(DEPS):
        v = data[d]
        ext = km_extent(v["wshape"])
        ref = d == REF_DEP
        x0 = col_x(j)

        ax = fig.add_axes(image_box(ext, x0, TMI_BOT))
        vmin, vmax = tmi_limits(d, v["tmi"])
        im = ax.imshow(v["tmi"], origin="upper", cmap=PAPER_CMAP,
                       vmin=vmin, vmax=vmax, extent=ext, aspect="auto",
                       interpolation="nearest")
        style_window(ax, j, ext, ref)

        cax = fig.add_axes([x0 + 0.012, CBAR_Y, COL_W - 0.024, CBAR_H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([vmin, 0, vmax])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cb.outline.set_linewidth(0.5)
        cb.set_label("nT", fontsize=6, labelpad=1)

        ax = fig.add_axes(image_box(ext, x0, U_BOT))
        imu = ax.imshow(v["u"], origin="upper", cmap=PAPER_CMAP,
                        vmin=U_LIMS[0], vmax=U_LIMS[1], extent=ext, aspect="auto",
                        interpolation="nearest")
        style_window(ax, j, ext, ref)

        # Each deposit is named once, on a common baseline above the top row; the two
        # rows share a column, so the name carries down to the radiometric-U panel.
        fig.text(x0 + COL_W / 2, TMI_TITLE_Y, f"{d} {DEP_NAMES[d]}", ha="center",
                 va="bottom", fontsize=8, color="#c62828" if ref else "black")
        if ref:
            fig.text(x0 + COL_W / 2, TMI_TITLE_Y - 0.024, "reference deposit",
                     ha="center", va="bottom", fontsize=6.5, color="#c62828")

    cax = fig.add_axes([RGT + 0.012, U_BOT + 0.02, 0.013, 0.125])
    cb = fig.colorbar(imu, cax=cax)
    cb.set_label("ppm eU", fontsize=6.5, labelpad=2)
    cb.set_ticks([0, 5, 10, 15])
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.outline.set_linewidth(0.5)

    fig.text(0.014, TMI_BOT + 0.076, "TMI windows", rotation=90,
             va="center", ha="center", fontsize=8.5)
    fig.text(0.014, U_BOT + 0.076, "Radiometric-U windows", rotation=90,
             va="center", ha="center", fontsize=8.5)
    fig.text(0.5, 0.985,
             "Deposits 1-5 (multivariate case): centred south-north profiles "
             "and deposit windows",
             ha="center", va="top", fontsize=9.5)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return pdf, png


def style_window(ax, col, ext, ref=False):
    ax.tick_params(length=2, pad=1, labelsize=6)
    ax.set_xticks([round(ext[0], 1), round(ext[1], 1)])
    ax.set_yticks([round(ext[2], 1), round(ext[3], 1)])
    ax.set_xlabel("km", labelpad=0, fontsize=6.5)
    if col == 0:
        ax.set_ylabel("km", labelpad=0, fontsize=6.5)
    for s in ax.spines.values():
        s.set_linewidth(1.1 if ref else 0.5)
        s.set_color("#c62828" if ref else "black")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rebuild fig:multiprofiles from data.")
    p.add_argument("--outdir", type=Path, default=Path("figures_out"))
    p.add_argument("--stem", default="multi_profiles_windows")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--normalize-profiles", action="store_true",
                   help="scale each profile by its own s.d. so low-amplitude deposits "
                        "(notably the Deposit 3 dipole) are shape-comparable")
    a = p.parse_args()

    data = collect()
    pdf, png = build(data, a.outdir, a.stem, a.dpi, a.normalize_profiles)
    for f in (pdf, png):
        print(f"wrote {f} ({f.stat().st_size / 1024:.0f} KB)")
