"""Reusable helpers for the synthetic illustrative SPCA notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from spatial_pca.geodata.exports import build_top_windows_gdf
from spatial_pca.spca.windows import WindowMatrix, build_univariate_window_matrix
from spatial_pca.validation.footprint_recovery import FootprintRecoveryResult


@dataclass(frozen=True)
class IllustrativeWindowContext:
    """Sliding-window setup used by the synthetic illustrative example."""

    n_rows: int
    n_cols: int
    train_row: int
    train_col: int
    training_template: np.ndarray
    window_matrix: WindowMatrix
    X: np.ndarray
    pca_input: np.ndarray
    deposit_index: int


def window_row_col(window_index: int, n_cols: int) -> tuple[int, int]:
    """Return the zero-based sliding-window row and column for a window ID."""

    row = int(window_index) // int(n_cols)
    col = int(window_index) % int(n_cols)
    return row, col


def prepare_univariate_window_context(
    *,
    field: np.ndarray,
    window_shape: tuple[int, int],
    stride_y: int,
    stride_x: int,
    training_window_index: int,
    variable_name: str,
) -> IllustrativeWindowContext:
    """Build the univariate window matrix and related notebook context."""

    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    n_rows = (field.shape[0] - win_h) // int(stride_y) + 1
    n_cols = (field.shape[1] - win_w) // int(stride_x) + 1
    train_row, train_col = window_row_col(training_window_index, n_cols)
    training_template = field[train_row : train_row + win_h, train_col : train_col + win_w]

    window_matrix = build_univariate_window_matrix(
        raster=field,
        deposit_template=training_template,
        variable_name=variable_name,
        stride_y=stride_y,
        stride_x=stride_x,
    )

    return IllustrativeWindowContext(
        n_rows=n_rows,
        n_cols=n_cols,
        train_row=train_row,
        train_col=train_col,
        training_template=training_template,
        window_matrix=window_matrix,
        X=window_matrix.combined_sliding_windows,
        pca_input=window_matrix.data_for_pca,
        deposit_index=window_matrix.deposit_index,
    )


def plot_grid_matrix_schematic(
    *,
    context: IllustrativeWindowContext,
    field: np.ndarray,
    window_shape: tuple[int, int],
    training_window_index: int,
    output_path: str | Path,
    gp_min: float,
    gp_max: float,
    cmap: str = "magma",
    reference_window_ids: tuple[int, ...] = (0, 1, 11, 80),
) -> Path:
    """Plot the toy geophysical grid and its sliding-window matrix."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    n = int(field.shape[0])
    x_matrix = context.X

    fig, (ax_grid, ax_matrix) = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        dpi=150,
        gridspec_kw={"width_ratios": [1.3, 0.35]},
    )

    ax_grid.imshow(field, origin="lower", interpolation="nearest", cmap=cmap, vmin=gp_min, vmax=gp_max)
    ax_grid.set_title(r"Geophysical grid ($r = 1$)", fontsize=14)
    ax_grid.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_grid.grid(which="minor", linestyle="-", linewidth=0.6, alpha=0.7)
    ax_grid.tick_params(axis="both", which="both", length=0, labelsize=14)

    for window_id in (*reference_window_ids, int(training_window_index)):
        row, col = window_row_col(window_id, context.n_cols)
        ax_grid.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                win_w,
                win_h,
                fill=False,
                edgecolor="white",
                linewidth=3.0,
                linestyle="--",
            )
        )
        ax_grid.text(
            col + 0.7,
            row + 1.0,
            str(window_id),
            color="white",
            fontsize=12,
            fontweight="bold",
        )

    ax_grid.add_patch(
        Rectangle(
            (context.train_col - 0.5, context.train_row - 0.5),
            win_w,
            win_h,
            fill=False,
            edgecolor="red",
            linewidth=3.5,
        )
    )
    ax_grid.text(
        context.train_col - 0.4,
        context.train_row + 0.1,
        "Known\nDeposit",
        color="black",
        fontsize=12,
        fontweight="bold",
    )
    ax_grid.text(0.9, -0.9, r"$\rightarrow$", color="black", fontsize=30, va="center", ha="center")
    ax_grid.text(-1.1, 1.0, r"$\uparrow$", color="black", fontsize=30, va="center", ha="center")
    ax_grid.text(1.0, -1.3, r"$s_x$", fontsize=16, color="black", va="center", ha="center")
    ax_grid.text(-1.5, 1.1, r"$s_y$", fontsize=16, color="black", va="center", ha="center", rotation=90)
    ax_grid.annotate(
        "",
        xy=(context.train_col - 0.9, context.train_row + 0.5),
        xytext=(context.train_col - 1.3, context.train_row + 0.5),
        arrowprops={"arrowstyle": "-[,widthB=2.9,lengthB=0.5", "lw": 1.6, "color": "black"},
    )
    ax_grid.text(
        context.train_col - 1.5,
        context.train_row + 0.5,
        r"$w_y$",
        fontsize=14,
        color="black",
        va="center",
        ha="center",
        rotation=90,
    )
    ax_grid.annotate(
        "",
        xy=(context.train_col + 0.5, context.train_row - 0.9),
        xytext=(context.train_col + 0.5, context.train_row - 1.3),
        arrowprops={"arrowstyle": "-[,widthB=2.9,lengthB=0.5", "lw": 1.6, "color": "black"},
    )
    ax_grid.text(
        context.train_col + 0.5,
        context.train_row - 1.6,
        r"$w_x$",
        fontsize=14,
        color="black",
        va="center",
        ha="center",
    )

    ax_matrix.imshow(x_matrix, aspect="auto", cmap=cmap, vmin=gp_min, vmax=gp_max)
    ax_matrix.set_xticks(np.arange(x_matrix.shape[1]))
    ax_matrix.set_xticklabels(np.arange(1, x_matrix.shape[1] + 1), fontsize=14)
    y_tick_positions = [0, 20, 40, 60, 80]
    y_tick_labels = ["0", "20", "40", "60", "80"]
    y_ticks = [y for y in y_tick_positions if y < x_matrix.shape[0]]
    ax_matrix.set_yticks(y_ticks)
    ax_matrix.set_yticklabels(y_tick_labels[: len(y_ticks)], fontsize=14)
    ax_matrix.set_xlabel(r"Feature index $j \in [1, p]$", fontsize=14)
    ax_matrix.set_ylabel("Sliding window ID", fontsize=14)
    ax_matrix.set_title(
        rf"$X \in \mathbb{{R}}^{{n \times p}}$  ($n={x_matrix.shape[0]}$, $p={x_matrix.shape[1]}$)",
        fontsize=14,
    )
    ax_matrix.annotate(
        "Known" "\n" "Deposit" "\n" f"Window ID {training_window_index}" "\n" r"$\mathbf{x}_d^{\top} \subset X$",
        xy=(x_matrix.shape[1] - 1, int(training_window_index)),
        xytext=(x_matrix.shape[1] + 0.8, int(training_window_index)),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2.5},
        va="center",
        ha="left",
        color="black",
        fontsize=14,
        fontweight="bold",
    )

    ax_grid.annotate(
        "",
        xy=(0.50, 0.5),
        xytext=(0.46, 0.5),
        xycoords="figure fraction",
        textcoords="figure fraction",
        arrowprops={"arrowstyle": "->", "lw": 3, "color": "black"},
        annotation_clip=False,
    )

    fig.tight_layout(w_pad=2.5)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_known_deposit_windows(
    *,
    field: np.ndarray,
    n_cols: int,
    window_shape: tuple[int, int],
    training_window_index: int,
    known_deposit_indices: list[int],
    output_path: str | Path,
    variable_name: str,
    gp_min: float,
    gp_max: float,
    cmap: str = "magma",
) -> Path:
    """Plot all known deposit windows and exact test-deposit chips."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    n = int(field.shape[0])

    fig = plt.figure(figsize=(10, 4.3), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)

    ax_deposits = fig.add_subplot(gs[0, 0])
    im = ax_deposits.imshow(field, origin="lower", interpolation="nearest", cmap=cmap, vmin=gp_min, vmax=gp_max)
    ax_deposits.set_title("Known Deposit Windows")
    ax_deposits.set_xlabel("Column")
    ax_deposits.set_ylabel("Row")
    ax_deposits.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_deposits.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_deposits.grid(which="minor", color="white", linewidth=0.5, alpha=0.55)
    ax_deposits.tick_params(which="minor", length=0)

    _add_window_outline_with_label(
        ax_deposits,
        window_id=training_window_index,
        n_cols=n_cols,
        window_shape=window_shape,
        edge_color="red",
        text_color="red",
        linewidth=2.8,
    )
    for window_id in known_deposit_indices:
        _add_window_outline_with_label(
            ax_deposits,
            window_id=window_id,
            n_cols=n_cols,
            window_shape=window_shape,
            edge_color="black",
            text_color="black",
            linewidth=2.2,
        )

    ax_deposits.legend(
        handles=[
            Line2D([0], [0], color="red", linewidth=2.8, label="Reference deposit"),
            Line2D([0], [0], color="black", linewidth=2.2, label="Testing known deposits"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    fig.colorbar(im, ax=ax_deposits, fraction=0.046, pad=0.03, label=variable_name)

    chip_grid = gs[0, 1].subgridspec(1, len(known_deposit_indices), wspace=0.45)
    for chip_idx, window_id in enumerate(known_deposit_indices):
        row, col = window_row_col(window_id, n_cols)
        chip = field[row : row + win_h, col : col + win_w]

        ax_chip = fig.add_subplot(chip_grid[0, chip_idx])
        chip_im = ax_chip.imshow(chip, origin="lower", interpolation="nearest", cmap=cmap, vmin=gp_min, vmax=gp_max)
        ax_chip.add_patch(Rectangle((-0.5, -0.5), win_w, win_h, fill=False, edgecolor="black", linewidth=2.0))
        ax_chip.set_title(f"Test ID {window_id}", fontsize=9)
        ax_chip.set_xticks([])
        ax_chip.set_yticks([])
        chip_cbar = fig.colorbar(chip_im, ax=ax_chip, fraction=0.05, pad=0.03)
        chip_cbar.ax.tick_params(labelsize=7)
        chip_cbar.set_label(variable_name, fontsize=7)

    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def build_illustrative_geodataframes(
    *,
    window_matrix: WindowMatrix,
    top_indices: np.ndarray,
    top_distances: np.ndarray,
    top_n: int,
    training_window_index: int,
    known_windows: pd.DataFrame,
    transform: Any,
    crs: str,
) -> tuple[Any, Any]:
    """Build top-window and known-deposit GeoDataFrames for the toy example."""

    top_window_rows = window_matrix.window_indices_for_mapping[top_indices[:top_n]]
    top_windows_gdf = build_top_windows_gdf(
        window_indices=top_window_rows,
        window_shape=window_matrix.window_shape,
        transform=transform,
        crs=crs,
        ranks=np.arange(1, len(top_window_rows) + 1, dtype=int),
        scores=top_distances[:top_n],
        window_ids=top_window_rows[:, 2],
    )

    window_index_lookup = {
        int(row[2]): row
        for row in window_matrix.window_indices_for_mapping
    }
    known_deposit_indices = known_windows["window_index"].astype(int).tolist()
    deposit_window_ids = [int(training_window_index)] + known_deposit_indices
    deposit_window_rows = np.asarray(
        [window_index_lookup[int(idx)] for idx in deposit_window_ids],
        dtype=int,
    )
    deposits_gdf = build_top_windows_gdf(
        window_indices=deposit_window_rows,
        window_shape=window_matrix.window_shape,
        transform=transform,
        crs=crs,
        window_ids=deposit_window_rows[:, 2],
    )
    deposits_gdf["name"] = [str(training_window_index)] + known_windows["label"].astype(str).tolist()
    return top_windows_gdf, deposits_gdf


def build_recovery_hit_table(
    *,
    recovery: FootprintRecoveryResult,
    hit_label_by_deposit: dict[int, str | int],
) -> pd.DataFrame:
    """Build a rank-by-rank recovery table with overlapped deposit IDs."""

    ranks = np.arange(1, recovery.cum_recovered_frac_total.size + 1)
    overlapped_known_test_deposit_id = [
        ", ".join(str(hit_label_by_deposit[int(dep)]) for dep in recovery.overlap_by_rank.get(int(rank), []))
        for rank in ranks
    ]
    threshold_hit_known_test_deposit_id = [
        ", ".join(str(hit_label_by_deposit[int(dep)]) for dep in recovery.hit_by_rank.get(int(rank), []))
        for rank in ranks
    ]
    known_deposits_recovered = np.asarray([
        sum(first_rank <= rank for first_rank in recovery.first_hit_rank_by_deposit.values())
        for rank in ranks
    ], dtype=int)

    return pd.DataFrame({
        "rank": ranks,
        "overlapped_known_test_deposit_id": overlapped_known_test_deposit_id,
        "threshold_hit_known_test_deposit_id": threshold_hit_known_test_deposit_id,
        "known_deposits_recovered": known_deposits_recovered,
        "recovery_fraction": recovery.cum_recovered_frac_total,
    })


def plot_image_pair(
    *,
    left_path: str | Path,
    right_path: str | Path,
    output_path: str | Path,
    figsize: tuple[float, float] = (13, 5.8),
) -> Path:
    """Place two generated figure files next to each other in one summary panel."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    left_image = plt.imread(str(left_path))
    right_image = plt.imread(str(right_path))

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)
    for ax, image in zip(axes, [left_image, right_image]):
        ax.imshow(image)
        ax.axis("off")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _add_window_outline_with_label(
    ax: Any,
    *,
    window_id: int,
    n_cols: int,
    window_shape: tuple[int, int],
    edge_color: str,
    text_color: str,
    linewidth: float,
) -> None:
    row, col = window_row_col(window_id, n_cols)
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    ax.add_patch(
        Rectangle(
            (col - 0.5, row - 0.5),
            win_w,
            win_h,
            fill=False,
            edgecolor=edge_color,
            linewidth=linewidth,
        )
    )
    label = ax.text(
        col + 0.5,
        row + 0.5,
        str(window_id),
        color=text_color,
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
    )
    label.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
