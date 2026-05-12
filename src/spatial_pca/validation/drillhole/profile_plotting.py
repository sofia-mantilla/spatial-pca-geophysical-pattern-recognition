#!/usr/bin/env python3
"""Borehole-profile plotting helpers for drillhole validation."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatial_pca_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd


def _pick_tick_labels(labels: list[str], max_labels: int) -> tuple[list[int], list[str]]:
    if not labels:
        return [], []
    if len(labels) <= max_labels:
        idx = list(range(len(labels)))
        return idx, labels

    step = max(1, math.ceil(len(labels) / max_labels))
    idx = list(range(0, len(labels), step))
    return idx, [labels[i] for i in idx]


def plot_binary_profile(
    *,
    intervals: pd.DataFrame,
    order: pd.DataFrame,
    output_path: Path,
    plot_cfg: dict,
    title_text: str,
    state_column: str = "mineralized_state",
) -> Path:
    intervals = intervals.copy()
    order = order.sort_values("borehole_rank").reset_index(drop=True).copy()

    intervals[state_column] = pd.array(intervals[state_column], dtype="Int8")
    hole_order = order["hole_id"].astype(str).tolist()

    valid = intervals.loc[
        intervals["hole_id"].notna()
        & intervals["From"].notna()
        & intervals["To"].notna()
        & intervals["hole_id"].isin(hole_order)
    ].copy()
    if valid.empty:
        raise ValueError("No valid intervals available for plotting.")

    max_depth = float(valid["To"].max())
    column_width = float(plot_cfg.get("column_width", 0.9))
    holes_per_panel = int(plot_cfg.get("holes_per_panel", 84))
    max_labels_per_panel = int(plot_cfg.get("max_hole_labels_per_panel", 12))
    panel_count = max(1, math.ceil(len(hole_order) / holes_per_panel))
    fig_width = float(plot_cfg.get("figure_width", 18.0))
    panel_height = float(plot_cfg.get("panel_height", 4.0))
    fig_height = max(float(plot_cfg.get("min_figure_height", 12.0)), panel_count * panel_height)
    title_fontsize = float(plot_cfg.get("title_fontsize", 26))
    axis_label_fontsize = float(plot_cfg.get("axis_label_fontsize", 18))
    tick_label_fontsize = float(plot_cfg.get("tick_label_fontsize", 15))
    legend_fontsize = float(plot_cfg.get("legend_fontsize", 15))
    panel_title_fontsize = float(plot_cfg.get("panel_title_fontsize", 16))

    fig, axes = plt.subplots(
        nrows=panel_count,
        ncols=1,
        figsize=(fig_width, fig_height),
        sharey=True,
        constrained_layout=True,
    )
    if panel_count == 1:
        axes = [axes]
    fig.patch.set_facecolor(plot_cfg.get("background_color", "#ffffff"))

    colors = {
        1: plot_cfg.get("mineralized_color", "#ff5a36"),
        0: plot_cfg.get("not_mineralized_color", "#ffe082"),
        pd.NA: plot_cfg.get("unknown_color", "#33b5a5"),
    }
    edge_color = plot_cfg.get("outline_color", "#111111")
    line_width = float(plot_cfg.get("outline_width", 0.2))

    for panel_idx, ax in enumerate(axes):
        panel_holes = hole_order[panel_idx * holes_per_panel : (panel_idx + 1) * holes_per_panel]
        panel_positions = {hole_id: idx for idx, hole_id in enumerate(panel_holes)}
        panel_valid = valid.loc[valid["hole_id"].isin(panel_holes)].copy()

        ax.set_facecolor(plot_cfg.get("background_color", "#ffffff"))

        for row in panel_valid.itertuples(index=False):
            x_center = panel_positions.get(str(row.hole_id))
            if x_center is None:
                continue
            state = getattr(row, state_column)
            if pd.isna(state):
                facecolor = colors[pd.NA]
            elif int(state) == 1:
                facecolor = colors[1]
            else:
                facecolor = colors[0]
            rect = Rectangle(
                (x_center - column_width / 2.0, float(row.From)),
                column_width,
                float(row.To) - float(row.From),
                facecolor=facecolor,
                edgecolor=edge_color,
                linewidth=line_width,
            )
            ax.add_patch(rect)

        ax.set_xlim(-0.5, len(panel_holes) - 0.5)
        ax.set_ylim(max_depth, 0.0)
        ax.set_ylabel("Depth (m)", fontsize=axis_label_fontsize)
        ax.tick_params(axis="y", labelsize=tick_label_fontsize)
        ax.grid(axis="y", color=plot_cfg.get("grid_color", "#d9cbb8"), linewidth=0.5)
        ax.set_axisbelow(True)

        tick_idx, tick_labels = _pick_tick_labels(panel_holes, max_labels_per_panel)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=tick_label_fontsize)

        start_rank = panel_idx * holes_per_panel + 1
        end_rank = start_rank + len(panel_holes) - 1
        ax.set_title(
            f"Holes {start_rank}-{end_rank} of {len(hole_order)}",
            fontsize=panel_title_fontsize,
            loc="left",
        )

    axes[-1].set_xlabel("Borehole order (NW to SE)", fontsize=axis_label_fontsize)
    fig.suptitle(title_text, fontsize=title_fontsize)

    legend_handles = [
        Patch(facecolor=plot_cfg.get("mineralized_color", "#ff5a36"), edgecolor=edge_color, label="Mineralized"),
        Patch(facecolor=plot_cfg.get("not_mineralized_color", "#ffe082"), edgecolor=edge_color, label="Not mineralized"),
        Patch(facecolor=plot_cfg.get("unknown_color", "#33b5a5"), edgecolor=edge_color, label="Missing Cu/Au assay"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        frameon=True,
        fontsize=legend_fontsize,
        bbox_to_anchor=(1.005, 0.5),
        borderaxespad=0.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(plot_cfg.get("dpi", 300)), bbox_inches="tight")
    plt.close(fig)
    return output_path
