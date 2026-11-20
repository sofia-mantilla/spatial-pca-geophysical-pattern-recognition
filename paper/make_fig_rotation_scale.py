"""Figure: rotation and scale sensitivity of the reference window (R2.3).

Print-size version for the manuscript appendix. Four rows: the reference
picture rotated, the top 250 windows each rotation selects, the reference at
the sizes the province shows, and the top 250 windows each size selects.

Rotation leaves the window at 28 x 46 px and turns only its contents, because
turning a body does not change how much ground it covers. Scale changes the
window itself, because the method takes the window dimensions from the
reference deposit's extent: a body at 0.5 spreads the same dipole over half
the distance, so the window is 14 x 23 px and every sliding window on the
survey is re-extracted at that shape.

Run from inside paper/, after run_rotation_scale_case1.py:

    python make_fig_rotation_scale.py --outdir ../docs/Spatial_PCA_paper_overleaf/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import case1_uni_repro as C1
import numpy_repro_concat as R
import run_rotation_scale_case1 as RS
from make_fig_multiprofiles import PAPER_CMAP, tmi_limits

HERE = Path(__file__).resolve().parent
ANGLES = (0.0, -15.0, 15.0, 90.0)
SCALES = (1.0, 0.65, 0.5, 0.35)
N_TOP = 250
FIG_W, FIG_H = 5.14, 4.05          # \textwidth = 372 pt

plt.rcParams.update({
    "font.size": 6.5,
    "axes.titlesize": 6.5,
})


def top_windows(tmi, template, rebuild, shared):
    """Union of the top 250 windows for one variant, as a boolean grid."""
    if rebuild:
        X, idx, wshape = RS.window_matrix_for(tmi, template)
    else:
        X0, idx, wshape = shared
        X = np.vstack([X0[:-1], template.ravel()])
    dep = X.shape[0] - 1
    Z = R.block_scores(X, RS.K_PCS)
    dist = np.sqrt(((Z - Z[dep]) ** 2) @ R.blk_w(Z[dep]))
    order = [i for i in np.argsort(dist) if i != dep][:N_TOP]
    wh, ww = wshape
    mask = np.zeros(tmi.shape, dtype=bool)
    for i in order:
        mask[idx[i, 0]:idx[i, 0] + wh, idx[i, 1]:idx[i, 1] + ww] = True
    return mask, wshape


def draw_template(ax, template, big_h, big_w, vmin, vmax):
    """Reference picture centred at its true size against the largest one."""
    h, w = template.shape
    x0, y0 = (big_w - w) / 2.0, (big_h - h) / 2.0
    ax.imshow(template, origin="upper", cmap=PAPER_CMAP, vmin=vmin, vmax=vmax,
              extent=(x0, x0 + w, y0 + h, y0))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=0.5, ec="k"))
    ax.set_xlim(0, big_w)
    ax.set_ylim(big_h, 0)
    ax.set_axis_off()


def draw_map(ax, tmi, mask, vmin, vmax, note):
    ax.imshow(tmi, origin="upper", cmap=PAPER_CMAP, vmin=vmin, vmax=vmax)
    ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=0.25)
    ax.text(0.015, 0.03, note, transform=ax.transAxes, fontsize=5.6, ha="left",
            va="bottom", bbox=dict(fc="white", ec="none", alpha=0.82, pad=0.8))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / "figures_out")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--name", default="Fig25")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    tmi, rs, re, cs, ce = RS.reference_bbox()
    _t, _u, ox, oy = R.load_cropped()
    RS.ORIGIN = (ox, oy)
    native = tmi[rs:re, cs:ce]
    big_h, big_w = native.shape
    vmin, vmax = tmi_limits(C1.REF_DEP, native)
    shared = RS.window_matrix_for(tmi, native)

    scored = json.loads((HERE / "results" / "rotation_scale_case1.json").read_text())

    def note(key):
        row = scored[key]
        return f"{row['end']:.1f}%, {row['ratio_matched']:.1f}$\\times$"

    # Row heights follow the panel aspects, so no band of white opens up
    # between a short template row and the map row under it.
    fig, axes = plt.subplots(4, 4, figsize=(FIG_W, FIG_H),
                             gridspec_kw={"height_ratios": [0.61, 0.53, 0.61, 0.53]})

    for col, angle in enumerate(ANGLES):
        template = RS.rotated_reference(tmi, rs, re, cs, ce, angle)
        mask, _ = top_windows(tmi, template, False, shared)
        draw_template(axes[0, col], template, big_h, big_w, vmin, vmax)
        axes[0, col].set_title("reference" if angle == 0.0
                               else f"{angle:+.0f}\N{DEGREE SIGN}", pad=2.0)
        draw_map(axes[1, col], tmi, mask, vmin, vmax, note(f"rot{angle:+g}"))

    for col, scale in enumerate(SCALES):
        template = RS.scaled_reference(native, scale)
        mask, wshape = top_windows(tmi, template, True, shared)
        draw_template(axes[2, col], template, big_h, big_w, vmin, vmax)
        axes[2, col].set_title(f"{scale:g}, {wshape[0]}$\\times${wshape[1]} px", pad=2.0)
        draw_map(axes[3, col], tmi, mask, vmin, vmax, note(f"scale{scale:g}"))

    for row, letter in enumerate("abcd"):
        axes[row, 0].text(-0.06, 1.0, f"({letter})", transform=axes[row, 0].transAxes,
                          fontsize=7.0, fontweight="bold", ha="right", va="top")

    fig.subplots_adjust(left=0.035, right=0.995, top=0.945, bottom=0.008,
                        hspace=0.42, wspace=0.06)
    stem = args.outdir / args.name
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    plt.close(fig)
    print(f"wrote {stem}.pdf and {stem}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
