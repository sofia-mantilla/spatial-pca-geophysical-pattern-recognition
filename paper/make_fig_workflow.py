"""Workflow schematic (Figure 4): the seven steps of the method.

Replaces the hand-made raster that had no generating script. Changes against it:
  * window dimensions are W_x x W_y, matching the text
  * no hardcoded "see Figure N" strings (they break when a figure is inserted)
  * c_min is left symbolic; its value belongs to the case studies
  * text sized in true points at the on-page width, so the annotation is legible

Pure matplotlib. Run from the worktree root:
    python make_fig_workflow.py [--outdir DIR] [--dpi 400]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "figures_out"))
ap.add_argument("--dpi", type=int, default=400)
ap.add_argument("--stem", default="workflow_steps_schematic")
args = ap.parse_args()
OUT = Path(args.outdir); OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17456e"
FILL = "#eef2f9"
GRAY = "#7a7a7a"
INK = "#111111"

# figure is included at 0.82\textwidth; textwidth is ~16.0 cm on a4 with 2.5 cm
# margins, so the drawing width below equals the on-page width and point sizes
# in this script are point sizes in the printed figure.
W_IN = 0.82 * 6.30
FS_TITLE, FS_HEAD, FS_BODY, FS_NOTE = 12.5, 10.8, 8.4, 7.6

STEPS = [
    dict(head="1. EXTENT", dashed=False, lines=[
        "The drilled outline of the reference deposit",
        r"sets the window size ($W_x \times W_y$).",
    ]),
    dict(head="2. SAMPLING", dashed=False, lines=[
        r"The window slides across the study area with strides ($s_x$, $s_y$).",
        r"Each window is flattened into one row of $X \in \mathbb{R}^{n \times p}$;",
        r"the deposit window is the row $x_d^{\top}$.",
    ]),
    dict(head="3. DECOMPOSITION (wPCA)", dashed=False, loop=True, lines=[
        r"Standardize $X$ and decompose: $\widetilde{X} = ZL^{\top}$.",
        "Each component is one building block of the geometry; the",
        r"scores $Z$ measure how strongly each window expresses each.",
    ]),
    dict(head="4. DEPOSIT WEIGHTS", dashed=False, loop=True, lines=[
        r"$w_m = z_{dm}^{2} / \sum_u z_{du}^{2}$",
        "Components the deposit expresses strongly receive",
        "large weights; background ones near zero.",
    ]),
    dict(head="5. DISTANCE", dashed=False, lines=[
        r"Weighted distance of every window $i$ to the deposit:",
        r"$d_i^{2} = \sum_m w_m (Z_{im} - Z_{dm})^{2}$",
    ]),
    dict(head=r"5.1  SEVERAL VARIABLES ($r > 1$)", dashed=True, lines=[
        "The same distance, balanced across the variables:",
        r"$d_i^{2} = \sum_v \alpha_v \sum_m w_m^{(v)} (Z_{im}^{(v)} - Z_{dm}^{(v)})^{2},$"
        r"   $\sum_v \alpha_v = 1$",
        r"$\alpha_v$ from the other deposits where available; equal otherwise.",
    ]),
    dict(head="6. RANKING", dashed=False, lines=[
        r"Ranking all windows by the distance $d_i$ gives the similarity",
        "map and the top windows: geometry compared with geometry.",
    ]),
    dict(head="7. VALIDATION (other known deposits)", dashed=True, lines=[
        "The remaining known deposits are test deposits.",
        "Cumulative footprint recovery; a deposit is hit at the",
        r"first rank where its recovered fraction reaches $c_{\min}$.",
    ]),
]

# ---------------------------------------------------------------- layout
PAD_TOP, PAD_BOT = 0.17, 0.09          # inches inside a box above/below text
LINE_H = 0.132                          # inches per body line
HEAD_H = 0.18                           # inches for the heading line
GAP = 0.12                              # inches between boxes (arrow sits here)
TITLE_H = 0.42
LOOP_PAD = 0.13                         # dashed multivariate wrapper padding

def line_h(ln):
    return LINE_H * (1.60 if r"\sum" in ln else 1.0)

heights = [PAD_TOP + HEAD_H + sum(line_h(l) for l in s["lines"]) + PAD_BOT
           for s in STEPS]
loop_idx = [i for i, s in enumerate(STEPS) if s.get("loop")]
extra_loop = 2 * LOOP_PAD + 0.13        # wrapper padding + its label line
H_IN = TITLE_H + sum(heights) + GAP * (len(STEPS) - 1) + extra_loop + 0.22

fig = plt.figure(figsize=(W_IN, H_IN), dpi=args.dpi)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W_IN); ax.set_ylim(0, H_IN); ax.axis("off")

fig.text(0.5, 1 - 0.30 / H_IN, "Windowed PCA similarity workflow",
         ha="center", va="center", fontsize=FS_TITLE, fontweight="bold", color=INK)

x0, x1 = 0.30, W_IN - 0.30
if loop_idx:
    x1 -= 0.74                          # room for the "next variable" return path

y = H_IN - TITLE_H
tops, bots = [], []
for i, (s, h) in enumerate(zip(STEPS, heights)):
    if i == loop_idx[0] if loop_idx else False:
        y -= LOOP_PAD + 0.16
    top = y
    bot = y - h
    tops.append(top); bots.append(bot)
    box = FancyBboxPatch((x0, bot), x1 - x0, h,
                         boxstyle="round,pad=0,rounding_size=0.09",
                         linewidth=1.6, edgecolor=NAVY if not s["dashed"] else GRAY,
                         facecolor=FILL if not s["dashed"] else "none",
                         linestyle="-" if not s["dashed"] else (0, (5, 4)), zorder=2)
    ax.add_patch(box)
    xc = (x0 + x1) / 2
    ax.text(xc, top - PAD_TOP + 0.03, s["head"], ha="center", va="center",
            fontsize=FS_HEAD, fontweight="bold", color=NAVY, zorder=3)
    ty = top - PAD_TOP - HEAD_H + 0.02
    for ln in s["lines"]:
        h = line_h(ln)
        ax.text(xc, ty - h / 2 + LINE_H / 2, ln, ha="center", va="center",
                fontsize=FS_BODY, color=INK, zorder=3)
        ty -= h
    y = bot - GAP
    if loop_idx and i == loop_idx[-1]:
        y -= LOOP_PAD

    if i < len(STEPS) - 1:              # downward arrow head
        ax.add_patch(Polygon([[xc - 0.075, bot - 0.015], [xc + 0.075, bot - 0.015],
                              [xc, bot - GAP + 0.035]], closed=True,
                             facecolor=NAVY, edgecolor="none", zorder=4))

# dashed wrapper around the per-variable steps
if loop_idx:
    a, b = loop_idx[0], loop_idx[-1]
    wy1 = tops[a] + LOOP_PAD + 0.16
    wy0 = bots[b] - LOOP_PAD
    ax.add_patch(Rectangle((x0 - 0.16, wy0), (x1 - x0) + 0.16 + 0.66, wy1 - wy0,
                           linewidth=1.4, edgecolor=GRAY, facecolor="none",
                           linestyle=(0, (6, 5)), zorder=1))
    ax.text(x0 - 0.06, wy1 - 0.13,
            r"several variables: repeat for each variable  $v = 1 \dots r$",
            ha="left", va="center", fontsize=FS_NOTE, color=GRAY, style="italic")
    xr = x1 + 0.30
    ax.add_patch(FancyArrowPatch((xr, bots[b] + 0.10), (xr, tops[a] - 0.12),
                                 connectionstyle="arc3,rad=0", arrowstyle="-|>",
                                 mutation_scale=9, linewidth=1.2, color=GRAY,
                                 linestyle=(0, (4, 3)), zorder=3))
    ax.plot([x1, xr], [bots[b] + 0.10, bots[b] + 0.10], color=GRAY, lw=1.2,
            linestyle=(0, (4, 3)), zorder=3)
    ax.plot([xr, x1], [tops[a] - 0.12, tops[a] - 0.12], color=GRAY, lw=1.2,
            linestyle=(0, (4, 3)), zorder=3)
    ax.text(xr + 0.11, (tops[a] + bots[b]) / 2, "next variable", ha="center",
            va="center", fontsize=FS_NOTE, color=GRAY, style="italic",
            rotation=90)

for ext in ("pdf", "png"):
    fig.savefig(OUT / f"{args.stem}.{ext}", dpi=args.dpi, facecolor="white")
print(f"{args.stem}.pdf/.png written to {OUT}  ({W_IN:.2f} x {H_IN:.2f} in)")
