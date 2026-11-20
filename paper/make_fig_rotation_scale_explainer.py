"""Explainer figure for the rotation and scale test (R2.3).

Shows what the test changes and what it does not. The candidate windows sweep
the survey on a fixed grid and are never turned: a rotation changes only the
picture the ranking looks for. A scale is a different kind of change, because
the method takes the window dimensions from the reference deposit's extent, so
a smaller body means the same dipole over fewer pixels and a smaller window
everywhere.

Rows 1 and 2: the reference picture rotated, and the top 250 windows each
rotation selects. Rows 3 and 4: the same for size. The reference pictures in
row 3 are drawn at their true relative pixel sizes, so the shrinking window is
visible; the black box on each map is that variant's window.

Every panel uses the paper colormap and the Case-1 TMI limits.

Run from inside paper/, after run_rotation_scale_case1.py:

    python make_fig_rotation_scale_explainer.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import case1_uni_repro as C1
import numpy_repro_concat as R
import run_rotation_scale_case1 as RS
from make_fig_multiprofiles import PAPER_CMAP, tmi_limits

ANGLES = (0.0, -15.0, 15.0, 90.0)
SCALES = (1.0, 0.65, 0.5, 0.35)
N_TOP = 250
OUT = Path(__file__).resolve().parent / "results"


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
    """Draw a reference picture centred at its true size against the largest one.

    The axes frame is removed and the picture carries its own outline, so the
    blank area around a small window reads as page rather than as window.
    """
    h, w = template.shape
    x0, y0 = (big_w - w) / 2.0, (big_h - h) / 2.0
    ax.imshow(template, origin="upper", cmap=PAPER_CMAP, vmin=vmin, vmax=vmax,
              extent=(x0, x0 + w, y0 + h, y0))
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, lw=0.8, ec="k"))
    ax.set_xlim(0, big_w)
    ax.set_ylim(big_h, 0)
    ax.set_axis_off()


def main() -> int:
    tmi, rs, re, cs, ce = RS.reference_bbox()
    _t, _u, ox, oy = R.load_cropped()
    RS.ORIGIN = (ox, oy)
    native = tmi[rs:re, cs:ce]
    vmin, vmax = tmi_limits(C1.REF_DEP, native)
    shared = RS.window_matrix_for(tmi, native)

    scored = {}
    results = OUT / "rotation_scale_case1.json"
    if results.exists():
        scored = json.loads(results.read_text())

    def caption(key):
        row = scored.get(key)
        if not row or "end" not in row:
            return f"top {N_TOP} windows"
        return (f"top {N_TOP} windows\nc(250) = {row['end']:.1f}%, "
                f"{row['ratio_matched']:.2f}x random")

    fig, axes = plt.subplots(4, 4, figsize=(11.6, 11.4))
    big_h, big_w = native.shape

    for col, angle in enumerate(ANGLES):
        template = RS.rotated_reference(tmi, rs, re, cs, ce, angle)
        mask, wshape = top_windows(tmi, template, False, shared)

        draw_template(axes[0, col], template, big_h, big_w, vmin, vmax)
        axes[0, col].set_title("reference" if angle == 0.0
                               else f"rotated {angle:+.0f}\N{DEGREE SIGN}", fontsize=9)

        ax = axes[1, col]
        ax.imshow(tmi, origin="upper", cmap=PAPER_CMAP, vmin=vmin, vmax=vmax)
        ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        ax.add_patch(Rectangle((cs, rs), wshape[1], wshape[0],
                               fill=False, lw=1.0, ec="k"))
        ax.set_title(caption(f"rot{angle:+g}"), fontsize=8)

    for col, scale in enumerate(SCALES):
        template = RS.scaled_reference(native, scale)
        mask, wshape = top_windows(tmi, template, True, shared)

        # Centred at true relative size, with no panel frame: the blank area is
        # page, not window, and must not read as part of what is searched.
        draw_template(axes[2, col], template, big_h, big_w, vmin, vmax)
        axes[2, col].set_title(
            f"body {scale:g} of its size\nwindow {wshape[0]} x {wshape[1]} px", fontsize=9)

        ax = axes[3, col]
        ax.imshow(tmi, origin="upper", cmap=PAPER_CMAP, vmin=vmin, vmax=vmax)
        ax.contour(mask.astype(float), levels=[0.5], colors="k", linewidths=0.5)
        ax.add_patch(Rectangle((cs, rs), wshape[1], wshape[0],
                               fill=False, lw=1.0, ec="k"))
        ax.set_title(caption(f"scale{scale:g}"), fontsize=8)

    for row in (1, 3):
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Rotation turns the picture the ranking looks for. Scale changes the window itself:\n"
        "the same dipole over fewer pixels, and a smaller window everywhere on the survey.",
        fontsize=11, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(exist_ok=True)
    png = OUT / "rotation_scale_explainer.png"
    fig.savefig(png, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
