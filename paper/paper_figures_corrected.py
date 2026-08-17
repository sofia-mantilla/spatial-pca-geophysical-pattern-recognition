"""Case-2 paper figures for the CORRECTED demonstration config.

Config: unwhitened concat, TMI k=2, U k=8, alpha = alpha_uni (brownfield rule,
computed from the other four deposits' univariate coverages), reference Deposit 3.

Produces (case2_experiments/figures/paper_corrected/):
  multi_recovery_comparison.png  - uni TMI(k2), uni U(k6), raw multi, concat alpha_uni
  multi_top250_windows.png       - top-250 ranked windows spatial map (TMI + U panels)
  multi_profiles_windows.png     - Deposit-3 TMI & U windows with profiles

Needs only numpy + matplotlib (uses numpy_repro_concat, the validated pure-numpy
replication of the pipeline). Run from the worktree root:
    python case2_experiments/paper_figures_corrected.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

import numpy_repro_concat as R

import argparse

_p = argparse.ArgumentParser(description="Corrected Case-2 paper figures (concat, honest alpha_uni).")
_p.add_argument("--k1", type=int, default=2, help="TMI stage-1 PCs (default 2)")
_p.add_argument("--k2", type=int, default=8, help="Radiometric-U stage-1 PCs (default 8)")
_p.add_argument("--dep", type=int, default=3, help="reference deposit, 1-based (default 3)")
_p.add_argument("--alpha", type=float, default=None, help="override alpha (default: alpha_uni rule)")
_a = _p.parse_args()
DEP, K1, K2 = _a.dep, _a.k1, _a.k2

DEST = HERE / "figures" / ("paper_corrected" if (K1, K2) == (2, 8) else f"paper_k{K1}_{K2}")
DEST.mkdir(parents=True, exist_ok=True)

PAPER_CMAP = LinearSegmentedColormap.from_list("spatial_pca_paper", (
    (0.00, "#0101ff"), (0.167, "#2dc4ff"), (0.334, "#26ff00"), (0.501, "#ffe101"),
    (0.668, "#ff0109"), (0.835, "#ef00ff"), (1.00, "#de9eff")))
UNIT_LABELS = {"TMI": "TMI (nT)", "Radiometric_U": "eU (ppm)"}
TMI_LIMS = (-150.0, 150.0)     # config visualization.deposit_limits_tmi["3"]
U_LIMS = (0.0, 15.0)           # config analysis_defaults vmin_var2/vmax_var2

# ---------------------------------------------------------------- alpha_uni
def coverage(alpha):
    return R.run(DEP, K1, K2, alpha)["coverage_by_deposit"]

others = [t for t in (1, 2, 3, 4, 5) if t != DEP]
covT, covU = coverage(1.0), coverage(0.0)
pT = float(np.mean([covT.get(t - 1, 0.0) for t in others]))
pU = float(np.mean([covU.get(t - 1, 0.0) for t in others]))
ALPHA = _a.alpha if _a.alpha is not None else pT / (pT + pU)
print(f"alpha_uni(dep {DEP}, k{K1}/{K2}) = {pT / (pT + pU):.4f}  ->  using alpha = {ALPHA:.4f}")

# ---------------------------------------------------------------- FIG A: recovery
runs = [
    (f"wPCA Uni (TMI, k={K1})", R.run(DEP, K1, K2, 1.0), "#1f4fd8"),
    (f"wPCA Uni (Radiometric_U, k={K2})", R.run(DEP, K1, K2, 0.0), "#8e24aa"),
    ("Raw Multi (TMI+U)", R.run(DEP, K1, K2, 0.0, mode="raw"), "#e0a800"),
    (rf"wPCA Multi (TMI+U, $\alpha_{{\mathrm{{TMI}}}}$={ALPHA:.3f})",
     R.run(DEP, K1, K2, ALPHA), "#2e9e2e"),
]
fig, ax = plt.subplots(figsize=(10, 7))
for lab, d, col in runs:
    y = np.asarray(d["cum_mean_recovered_frac"], float)
    x = np.arange(1, len(y) + 1)
    ax.step(x, y, where="post", color=col, lw=2.4, label=lab)
    for r, ids in d["overlap_by_rank"].items():
        if r in d["hit_by_rank"]:
            continue
        ax.scatter([r], [y[min(r - 1, len(y) - 1)]], s=42, facecolors="#ffd21e",
                   edgecolors="k", lw=.6, zorder=5)
    for r, ids in d["hit_by_rank"].items():
        yy = y[min(r - 1, len(y) - 1)]
        ax.scatter([r], [yy], s=70, facecolors="#e03131", edgecolors="k", lw=.7, zorder=6)
        ax.annotate("+".join(str(t + 1) for t in ids), (r, yy), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10, fontweight="bold", color="#7a1010")
ax.scatter([], [], s=42, facecolors="#ffd21e", edgecolors="k", label="Overlap event (yellow)")
ax.scatter([], [], s=70, facecolors="#e03131", edgecolors="k", label="Hit event (red)")
ax.set_xlim(0, 250); ax.set_ylim(0, 0.8)
ax.set_xlabel("Prediction rank"); ax.set_ylabel("Cumulative recovered fraction")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(DEST / "multi_recovery_comparison.png", dpi=170, facecolor="white")
plt.close(fig)
print("multi_recovery_comparison.png written")

# ---------------------------------------------------------------- FIG B: top-250 map
tmi, u, ox, oy = R.load_cropped()
H, W = tmi.shape
extent = (ox, ox + W * R.PX, oy - H * R.PX, oy)   # left, right, bottom, top

# ranked windows (concat, alpha_uni)
order_run = R.rank_concat(DEP, K1, K2, ALPHA)
order, idx, wshape, origin, dep_index = order_run
wh, ww = wshape
valid = order[order != dep_index][:250]
wins = [tuple(idx[j][:2]) for j in valid]          # (row, col) top-left pixel offsets

window_mask = np.zeros((H, W), dtype=bool)
for r0, c0 in wins:
    window_mask[r0:r0 + wh, c0:c0 + ww] = True

deps_wgs = R.deposits_wgs84()


def rgba(bg, vmin, vmax, bright):
    a = np.asarray(bg, float)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    out = np.asarray(PAPER_CMAP(norm(a)), dtype=float)
    out[..., :3] *= bright
    out[..., 3] = np.where(np.isfinite(a), out[..., 3], 0.0)
    return out


def nice_len(target):
    best = 1.0
    for e in range(2, 6):
        for b in (1, 2, 5):
            v = b * 10 ** e
            if v <= target:
                best = v
    return best


fig, axes = plt.subplots(1, 2, figsize=(20, 10), squeeze=False)
for _a in axes.ravel():
    _a.tick_params(labelsize=16)
panels = [("TMI", tmi, TMI_LIMS), ("Radiometric_U", u, U_LIMS)]
for ax, (name, bg, (vmin, vmax)) in zip(axes.flat, panels):
    dark = rgba(bg, vmin, vmax, 0.35)
    bright = rgba(bg, vmin, vmax, 1.0)
    bright[..., 3] *= window_mask.astype(float)
    ax.imshow(dark, extent=extent, origin="upper")
    ax.imshow(bright, extent=extent, origin="upper")
    # top-10 predicted windows, white boundary
    for r0, c0 in wins[:10]:
        x1, y2 = ox + c0 * R.PX, oy - r0 * R.PX
        ax.plot([x1, x1 + ww * R.PX, x1 + ww * R.PX, x1, x1],
                [y2, y2, y2 - wh * R.PX, y2 - wh * R.PX, y2],
                color="white", lw=1.0, zorder=6)
    # deposits: testing black, training red (reprojected, as in validation)
    for i, rings in enumerate(deps_wgs):
        col = "red" if i == DEP - 1 else "black"
        lw = 3.0 if i == DEP - 1 else 2.5
        for ring in rings:
            xs = [p[0] for p in ring]; ys = [p[1] for p in ring]
            ax.plot(xs, ys, color=col, lw=lw, zorder=8 if col == "red" else 7)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting", fontsize=20); ax.set_ylabel("Northing", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_title(name, fontsize=23, fontweight="bold")
    # scale bar
    length = nice_len((extent[1] - extent[0]) * 0.2)
    bx0 = extent[0] + (extent[1] - extent[0]) * 0.06
    by = extent[2] + (extent[3] - extent[2]) * 0.08
    ax.plot([bx0, bx0 + length], [by, by], color="black", lw=4, solid_capstyle="butt", zorder=20)
    th = (extent[3] - extent[2]) * 0.015
    for bx in (bx0, bx0 + length):
        ax.plot([bx, bx], [by - th / 2, by + th / 2], color="black", lw=2, zorder=20)
    lab = f"{int(length / 1000)} km" if length >= 1000 else f"{int(length)} m"
    ax.text(bx0 + length / 2, by + (extent[3] - extent[2]) * 0.02, lab, ha="center",
            va="bottom", fontsize=19, color="black",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5}, zorder=21)
    sm = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=PAPER_CMAP)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(UNIT_LABELS.get(name, name), fontsize=20)

handles = [
    Line2D([0], [0], color="black", lw=2.5, label="Test deposits"),
    Line2D([0], [0], color="red", lw=3.0, label="Reference deposit"),
    Line2D([0], [0], color="white", lw=1.0, label="Top 10 Predicted Windows"),
    Line2D([0], [0], color="black", marker=r"$\uparrow$", linestyle="None", markersize=12, label="N"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.015), fontsize=20)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(DEST / "multi_top250_windows.png", dpi=400, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("multi_top250_windows.png written")

# ---------------------------------------------------------------- FIG C: profiles
X, idx2, wshape2, _ = R.window_matrix(DEP)
npix = wshape2[0] * wshape2[1]
dep_row = X[-1]
T = dep_row[:npix].reshape(wshape2)
U_ = dep_row[npix:2 * npix].reshape(wshape2)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, M, name, (vmin, vmax) in ((axes[0], T, "TMI window", TMI_LIMS), (axes[1], U_, "Radiometric-U window", U_LIMS)):
    im = ax.imshow(M, origin="upper", cmap=PAPER_CMAP, vmin=vmin, vmax=vmax)
    ax.set_title(f"Reference Deposit 3 — {name}")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle("Reference Deposit 3 (Alemão) — TMI and Radiometric-U windows", fontsize=13)
fig.tight_layout()
fig.savefig(DEST / "multi_profiles_windows.png", dpi=170, facecolor="white")
plt.close(fig)
print("multi_profiles_windows.png written")

d = R.run(DEP, K1, K2, ALPHA)
mf = d["cum_mean_recovered_frac"]
print(f"\nconcat k{K1}/{K2} alpha={ALPHA:.3f}: AUC {mf.sum()*250/len(mf):.1f} "
      f"end {mf[-1]*100:.1f}% hits {len(d['hit_by_rank'])} "
      f"{ {r: [t+1 for t in ids] for r, ids in sorted(d['hit_by_rank'].items())} }")
print("All corrected Case-2 figures in:", DEST)
