"""Rebuild Figure 18 (fig:multiprofiles) from data at publication resolution.

Content matches the figure it replaces (which was a raster lifted from v9 image18 and
had no generating script): centered south-north profiles for the Case-2 deposits and
each deposit's TMI and radiometric U windows.

The figure is laid out at its printed width (\\textwidth = 5.14 in) with 8 pt text, so
nothing prints below the journal's 8 pt floor.

Differences from the raster it replaces, all deliberate:
  * regular grid, no dead space, legible type at printed size;
  * windows drawn on a kilometre extent so deposit footprints are physically comparable;
  * radiometric-U on the paper's shared 0-15 limits (config analysis_defaults), one
    colourbar for the row;
  * TMI uses the project's canonical per-deposit limits, read from
    configs/carajas_uni_tmi.yaml (visualization.deposit_limits_tmi) - the same limits
    the pipeline plots with, so panel colours match the rest of the paper. They are
    asymmetric by design: raw TMI carries a regional baseline, so zero is not neutral.
    A shared TMI scale is NOT used either: amplitudes span 8x across the deposits,
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

import deposit_names as DN
import numpy_repro_concat as R
from extract_dep_windows import collect, DEPS

# ---------------------------------------------------------------- conventions
# Same rainbow ramp used by every other figure in the paper (paper_figures_corrected.py).
PAPER_CMAP = LinearSegmentedColormap.from_list("spatial_pca_paper", (
    (0.00, "#0101ff"), (0.167, "#2dc4ff"), (0.334, "#26ff00"), (0.501, "#ffe101"),
    (0.668, "#ff0109"), (0.835, "#ef00ff"), (1.00, "#de9eff")))
U_LIMS = (0.0, 15.0)          # config analysis_defaults vmin_var2/vmax_var2
REF_DEP = 3                   # Alemao, the reference deposit
# Names come from deposit_names.py, the single source the tables and figures share.
DEP_NAMES = DN.NAMES
CURVE = {1: "#1f4fd8", 2: "#e07b00", 3: "#c62828", 4: "#2e9e2e", 5: "#8e24aa",
         6: "#00838f"}

mpl.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
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



# ---------------------------------------------------------------- layout (inches)
# Drawn at the printed width, so a font size in points IS its printed size.
FIG_W, FIG_H = 5.14, 6.62

PROF_L, PROF_R = 0.62, 0.08            # profile axes margins
PROF_TOP, PROF_H, PROF_GAP = 0.10, 1.45, 0.14
PROF_XLAB = 0.38                       # ticks + x label under the lower profile axes
LEGEND_TOP = PROF_TOP + 2 * PROF_H + PROF_GAP + PROF_XLAB + 0.06
GRID_TOP = LEGEND_TOP + 0.50           # window grid starts here

MARG_L = 0.16
CBAR_U_W = 0.55                        # room on the right for the shared U colourbar
COL_GAP = 0.14
N_COL = len(DEPS)
COL_W = (FIG_W - MARG_L - CBAR_U_W - (N_COL - 1) * COL_GAP) / N_COL

ROW_LAB_H = 0.18                       # "TMI windows" / "Radiometric U windows"
NAME_H = 0.20                          # deposit name above the TMI row
SIZE_H = 0.16                          # "w x h km" under the TMI window
CBAR_H = 0.07
CBAR_LAB_H = 0.17
SECTION_GAP = 0.16


def _ax(fig, x, y_top, w, h):
    """Axes rect from inches measured from the top-left corner."""
    return fig.add_axes([x / FIG_W, (FIG_H - y_top - h) / FIG_H, w / FIG_W, h / FIG_H])


def _text(fig, x, y_top, s, **kw):
    return fig.text(x / FIG_W, (FIG_H - y_top) / FIG_H, s, **kw)


def col_x(j):
    return MARG_L + j * (COL_W + COL_GAP)


def image_h(ext):
    return COL_W * (ext[3] - ext[2]) / (ext[1] - ext[0])


def build(data, outdir: Path, stem: str, dpi: int, normalize_profiles: bool = False):
    fig = plt.figure(figsize=(FIG_W, FIG_H))

    # ---------------------------------------------------------- profiles (top)
    pw = FIG_W - PROF_L - PROF_R
    ax_t = _ax(fig, PROF_L, PROF_TOP, pw, PROF_H)
    ax_u = _ax(fig, PROF_L, PROF_TOP + PROF_H + PROF_GAP, pw, PROF_H)
    if normalize_profiles:
        ylabs = ("TMI, centered\nand scaled (s.d.)",
                 "Radiometric U, centered\nand scaled (s.d.)")
    else:
        ylabs = ("TMI, centered (nT)", "Radiometric U,\ncentered (ppm eU)")
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
                    label=f"{d} {DEP_NAMES[d]}" + (" (reference)" if ref else ""))
        ax.axhline(0, color="#9e9e9e", lw=0.5, zorder=1)
        ax.axvline(0, color="#9e9e9e", lw=0.5, zorder=1)
        ax.set_ylabel(ylab, labelpad=2)
        ax.grid(alpha=0.25, lw=0.4)
        ax.margins(x=0.02)
        ax.tick_params(length=2.5, pad=1.5)
        for s_ in ax.spines.values():
            s_.set_linewidth(0.6)
    ax_t.set_xticklabels([])
    ax_t.set_xlim(ax_u.get_xlim())
    ax_u.set_xlabel("Distance along south\u2013north line, centered (km)", labelpad=2)

    # one shared legend under both panels, so no curve is hidden by a legend box
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, (FIG_H - LEGEND_TOP) / FIG_H),
               ncol=3, frameon=False, fontsize=8, handlelength=1.6,
               columnspacing=1.6, handletextpad=0.5, labelspacing=0.35)

    # ---------------------------------------------------------- window rows
    exts = {d: km_extent(data[d]["wshape"]) for d in DEPS}
    row_h = max(image_h(exts[d]) for d in DEPS)

    tmi_lab_y = GRID_TOP
    tmi_img_bot = GRID_TOP + ROW_LAB_H + NAME_H + row_h
    u_lab_y = tmi_img_bot + SIZE_H + CBAR_H + CBAR_LAB_H + SECTION_GAP
    u_img_bot = u_lab_y + ROW_LAB_H + row_h

    _text(fig, MARG_L, tmi_lab_y + 0.13,
          "TMI windows (extent in km, colour scale in nT)", ha="left",
          va="baseline", fontsize=8)
    _text(fig, MARG_L, u_lab_y + 0.13, "Radiometric U windows", ha="left",
          va="baseline", fontsize=8)

    imu = None
    for j, d in enumerate(DEPS):
        v = data[d]
        ext = exts[d]
        ref = d == REF_DEP
        x0 = col_x(j)
        h = image_h(ext)

        ax = _ax(fig, x0, tmi_img_bot - h, COL_W, h)
        vmin, vmax = tmi_limits(d, v["tmi"])
        im = ax.imshow(v["tmi"], origin="upper", cmap=PAPER_CMAP,
                       vmin=vmin, vmax=vmax, extent=ext, aspect="auto",
                       interpolation="nearest")
        style_window(ax, ref)

        # window size, once per column: the two variables share the window
        _text(fig, x0 + COL_W / 2, tmi_img_bot + 0.04,
              f"{ext[1] - ext[0]:.1f}\u00d7{ext[3] - ext[2]:.1f}",
              ha="center", va="top", fontsize=8, color="#616161")

        # Narrow enough that the end-value labels of adjacent columns clear
        # each other at 8 pt.
        cax = _ax(fig, x0 + 0.15, tmi_img_bot + SIZE_H, COL_W - 0.30, CBAR_H)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([vmin, vmax])   # end values only: the columns are narrow
        cb.ax.tick_params(labelsize=8, length=2, pad=1.5)
        cb.outline.set_linewidth(0.5)

        ax = _ax(fig, x0, u_img_bot - h, COL_W, h)
        imu = ax.imshow(v["u"], origin="upper", cmap=PAPER_CMAP,
                        vmin=U_LIMS[0], vmax=U_LIMS[1], extent=ext, aspect="auto",
                        interpolation="nearest")
        style_window(ax, ref)

        # The columns are too narrow for a deposit name at 8 pt, so each column
        # is headed by its deposit number and the legend above carries the names.
        # Headers sit on one line, above the tallest window of the row, rather
        # than following each window's own height.
        _text(fig, x0 + COL_W / 2, tmi_img_bot - row_h - 0.05, f"{d}",
              ha="center", va="bottom", fontsize=8,
              color="#c62828" if ref else "black")

    cax = _ax(fig, col_x(N_COL - 1) + COL_W + 0.10, u_img_bot - row_h + 0.02, 0.09,
              row_h - 0.04)
    cb = fig.colorbar(imu, cax=cax)
    cb.set_label("ppm eU", fontsize=8, labelpad=2)
    cb.set_ticks([0, 5, 10, 15])
    cb.ax.tick_params(labelsize=8, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)

    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"
    fig.savefig(pdf, facecolor="white")
    fig.savefig(png, dpi=dpi, facecolor="white")
    plt.close(fig)
    return pdf, png


def style_window(ax, ref=False):
    ax.set_xticks([]); ax.set_yticks([])
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
