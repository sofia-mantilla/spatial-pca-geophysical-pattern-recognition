"""Diagnostic plotting helpers for SPCA runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatial_pca_matplotlib_cache"))

import matplotlib.pyplot as plt
import math

import numpy as np
from matplotlib import patheffects as pe
from rasterio.transform import Affine
from shapely.geometry.base import BaseGeometry

from spatial_pca.units import variable_display_label

# Journal artwork rules (Springer NRR): no figure-level titles inside the
# artwork; captions carry the information. Panel labels stay.
SHOW_FIGURE_TITLES = False
from spatial_pca.colormaps import DEFAULT_PAPER_CMAP, resolve_colormap

DEFAULT_IMAGE_CMAP = DEFAULT_PAPER_CMAP


def plot_pc_score_map(
    *,
    scores: Any,
    window_indices: Any,
    window_shape: tuple[int, int],
    transform: Affine,
    background: Any,
    background_extent: tuple[float, float, float, float],
    deposit_index: int,
    deposit_polygon: BaseGeometry,
    variable_name: str,
    output_path: str | Path,
    pc: int | None = None,
) -> Path:
    """Plot candidate window centers colored by one PC score."""

    Z = np.asarray(scores, dtype=float)
    indices = np.asarray(window_indices, dtype=int)
    if Z.ndim != 2:
        raise ValueError(f"scores must be 2D, got shape {Z.shape}.")
    if indices.ndim != 2 or indices.shape[1] < 3:
        raise ValueError("window_indices must have columns [row, col, window_id].")
    if not (0 <= int(deposit_index) < Z.shape[0]):
        raise IndexError("deposit_index is out of bounds for scores.")

    Z_windows = np.delete(Z, int(deposit_index), axis=0)
    if Z_windows.shape[0] != indices.shape[0]:
        raise ValueError(
            f"Window score count {Z_windows.shape[0]} does not match index count {indices.shape[0]}."
        )

    if pc is None:
        pc_idx = int(np.argmax(np.abs(Z[int(deposit_index), :])))
    else:
        pc_idx = int(pc)
    if not (0 <= pc_idx < Z.shape[1]):
        raise IndexError("pc is out of bounds for scores.")

    x, y = _window_centers_from_indices(indices, window_shape, transform)
    pc_vals = Z_windows[:, pc_idx]
    deposit_score = float(Z[int(deposit_index), pc_idx])
    pc_std = float(np.std(Z[:, pc_idx]))
    vmin = deposit_score - pc_std
    vmax = deposit_score + pc_std

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        np.asarray(background),
        extent=background_extent,
        origin="upper",
        cmap="RdGy",
    )
    sc = ax.scatter(
        x,
        y,
        c=pc_vals,
        s=20,
        alpha=0.9,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    centroid = deposit_polygon.centroid
    ax.scatter(
        [centroid.x],
        [centroid.y],
        marker="*",
        s=450,
        c="none",
        edgecolors="white",
        linewidths=2,
        zorder=10,
        label="Reference deposit",
    )
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_title(f"{variable_name}: window centers colored by PC{pc_idx + 1} score")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(f"PC{pc_idx + 1} score")
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def plot_rotated_deposit(
    *,
    deposit_array: Any,
    deposit_extent: tuple[float, float, float, float],
    deposit_1based: int,
    variable_name: str,
    rotation_angle: float,
    output_path: str | Path,
    vmin: float | None = None,
    vmax: float | None = None,
    image_cmap: str | Any | None = None,
) -> Path:
    """Plot the deposit template in map coordinates using the legacy visual style."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = resolve_colormap(image_cmap or DEFAULT_IMAGE_CMAP)
    im = ax.imshow(
        np.asarray(deposit_array, dtype=float),
        extent=deposit_extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Rotated {rotation_angle:g}\N{DEGREE SIGN} - Deposit {deposit_1based} - {variable_name}", fontsize=35.1)
    ax.set_xlabel("Easting (m)", fontsize=27.0)
    ax.set_ylabel("Northing (m)", fontsize=27.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.06)
    cbar.set_label(_variable_label(variable_name), fontsize=24.3)
    fig.tight_layout()
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def plot_multivariate_rotated_deposit(
    *,
    deposit_arrays: dict[str, Any],
    deposit_1based: int,
    rotation_angle: float,
    output_path: str | Path,
    vmin_by_var: dict[str, float | None] | None = None,
    vmax_by_var: dict[str, float | None] | None = None,
    image_cmap: str | Any | None = None,
) -> Path:
    """Plot multivariate deposit templates side by side."""

    ordered_vars = list(deposit_arrays)
    cmap = resolve_colormap(image_cmap or DEFAULT_IMAGE_CMAP)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(ordered_vars), figsize=(6 * len(ordered_vars), 6), squeeze=False)
    fig.suptitle(f"Rotated {rotation_angle:g}\N{DEGREE SIGN} - Deposit {deposit_1based}", fontsize=22.0, y=1.005)
    for ax, var in zip(axes.flat, ordered_vars):
        arr = np.asarray(deposit_arrays[var], dtype=float)
        vmin = None if vmin_by_var is None else vmin_by_var.get(var)
        vmax = None if vmax_by_var is None else vmax_by_var.get(var)
        im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(var, fontsize=24.3)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(_variable_label(var), fontsize=16.2)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_top_similar_windows(
    *,
    flattened_windows: Any,
    ranked_window_rows: Any,
    ranked_distances: Any,
    window_ids: Any,
    window_shape: tuple[int, int],
    variable_name: str,
    output_path: str | Path,
    n_rows: int = 2,
    n_cols: int = 4,
    vmin: float | None = None,
    vmax: float | None = None,
    image_cmap: str | Any | None = None,
    feature_mask: np.ndarray | None = None,
) -> Path:
    """Plot the top-ranked raw sliding-window patches."""

    windows = np.asarray(flattened_windows, dtype=float)
    rows = np.asarray(ranked_window_rows, dtype=int)
    dists = np.asarray(ranked_distances, dtype=float)
    ids = np.asarray(window_ids, dtype=int)
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    n_show = min(n_rows * n_cols, rows.size)

    cmap = resolve_colormap(image_cmap or DEFAULT_IMAGE_CMAP)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7), squeeze=False)
    if SHOW_FIGURE_TITLES:
        fig.suptitle(f"Top {n_show} most similar sliding windows", fontsize=22.0, y=1.005)
    last_im = None
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis("off")
            continue
        patch = _vector_to_display_patch(
            windows[rows[i]],
            (win_h, win_w),
            feature_mask=feature_mask,
        )
        last_im = ax.imshow(patch, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Rank {i + 1} | idx={int(ids[i])} | d={float(dists[i]):.3f}", fontsize=16.2)
        ax.set_xticks([])
        ax.set_yticks([])
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
        cbar.set_label(_variable_label(variable_name), fontsize=21.6)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_multivariate_top_similar_windows(
    *,
    per_variable_windows: dict[str, np.ndarray],
    ranked_window_rows: Any,
    ranked_distances: Any,
    window_ids: Any,
    window_shape: tuple[int, int],
    output_path: str | Path,
    n_show: int,
    vmin_by_var: dict[str, float | None] | None = None,
    vmax_by_var: dict[str, float | None] | None = None,
    image_cmap: str | Any | None = None,
    feature_mask: np.ndarray | None = None,
) -> Path:
    """Plot top-ranked multivariate windows with one row per variable."""

    ordered_vars = list(per_variable_windows)
    rows = np.asarray(ranked_window_rows, dtype=int)
    dists = np.asarray(ranked_distances, dtype=float)
    ids = np.asarray(window_ids, dtype=int)
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    n_show = min(int(n_show), rows.size)
    cmap = resolve_colormap(image_cmap or DEFAULT_IMAGE_CMAP)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(ordered_vars),
        n_show,
        figsize=(2.8 * max(1, n_show), 2.8 * len(ordered_vars)),
        squeeze=False,
    )
    if SHOW_FIGURE_TITLES:
        fig.suptitle(f"Top {n_show} most similar sliding windows", fontsize=22.0, y=1.005)
    for col_idx in range(n_show):
        for row_idx, var in enumerate(ordered_vars):
            ax = axes[row_idx, col_idx]
            window = _vector_to_display_patch(
                per_variable_windows[var][rows[col_idx]],
                (win_h, win_w),
                feature_mask=feature_mask,
            )
            vmin = None if vmin_by_var is None else vmin_by_var.get(var)
            vmax = None if vmax_by_var is None else vmax_by_var.get(var)
            im = ax.imshow(window, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(f"Rank {col_idx + 1}\nidx={int(ids[col_idx])}\nd={float(dists[col_idx]):.3f}", fontsize=13.5)
            if col_idx == 0:
                ax.set_ylabel(var, fontsize=16.2)
            ax.set_xticks([])
            ax.set_yticks([])
    for row_idx, var in enumerate(ordered_vars):
        sm = plt.cm.ScalarMappable(cmap=cmap)
        vmin = None if vmin_by_var is None else vmin_by_var.get(var)
        vmax = None if vmax_by_var is None else vmax_by_var.get(var)
        if vmin is not None and vmax is not None:
            sm.set_clim(vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(sm, ax=axes[row_idx, :].tolist(), fraction=0.015, pad=0.02)
        cbar.set_label(_variable_label(var), fontsize=14.9)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_deposit_scores_and_weights(
    *,
    scores: Any,
    explained_variance_ratio: Any,
    weights: Any,
    deposit_index: int,
    k_used: int,
    output_path: str | Path,
    k_display: int | None = None,
    recompute_weights_from_scores: bool = False,
    weight_ylim: tuple[float, float] | None = None,
) -> Path:
    """Plot deposit PCA scores and ranking weights for the PCs used."""

    Z = np.asarray(scores, dtype=float)
    explained = np.asarray(explained_variance_ratio, dtype=float)
    k_weight = min(int(k_used), Z.shape[1])
    if k_weight < 1:
        raise ValueError("k_used must select at least one PCA component.")

    if recompute_weights_from_scores:
        z_dep_full = Z[int(deposit_index), :k_weight]
        raw_weights = z_dep_full**2
        weight_sum = float(raw_weights.sum())
        if weight_sum > 0:
            w_full = raw_weights / weight_sum
        else:
            w_full = np.ones(k_weight, dtype=float) / float(k_weight)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        k_weight = min(k_weight, w.size)
        w_full = w[:k_weight]

    if k_display is None:
        k_display = k_weight
    k = max(1, min(int(k_display), k_weight))
    pcs = np.arange(1, k + 1)
    z_dep = Z[int(deposit_index), :k]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 11), sharex=True)

    ax_top = axes[0]
    ax_top.set_title(r"Deposit raw PCA scores $z_{d,m}$", fontsize=21.6)
    ax_top.bar(pcs, z_dep, color="#4c8bbe")
    ax_top.set_ylabel("Raw PCA score", fontsize=18.9)

    finite_scores = z_dep[np.isfinite(z_dep)]
    if finite_scores.size:
        score_min = min(0.0, float(np.nanmin(finite_scores)))
        score_max = max(0.0, float(np.nanmax(finite_scores)))
    else:
        score_min, score_max = -0.5, 0.5
    score_span = max(score_max - score_min, 1.0)
    label_offset = 0.035 * score_span
    axis_margin = 0.28 * score_span
    ax_top.set_ylim(score_min - axis_margin, score_max + axis_margin)

    explained_display = explained[:k].copy()
    explained_sum = float(np.nansum(explained))
    if explained_sum > 1.0 + 1e-6:
        explained_display = explained_display / explained_sum

    for pc, score, var_frac in zip(pcs, z_dep, explained_display):
        label = rf"Var$_{{{pc}}}$ = {100 * float(var_frac):.1f}%"
        va = "bottom" if score >= 0 else "top"
        offset = label_offset if score >= 0 else -label_offset
        ax_top.text(
            pc,
            score + offset,
            label,
            ha="center",
            va=va,
            fontsize=15.0,
            rotation=90,
            rotation_mode="anchor",
            clip_on=False,
        )
    ax_top.axhline(0, color="black", linewidth=0.8)

    ax_bot = axes[1]
    weights_used = w_full[:k]
    ax_bot.bar(pcs, weights_used, color="#2ca02c")
    ax_bot.set_title(r"Deposit-based weights $w_m$", fontsize=21.6)
    ax_bot.set_xlabel("Principal component (m)", fontsize=18.9)
    ax_bot.set_ylabel("Weight", fontsize=18.9)
    finite_weights = weights_used[np.isfinite(weights_used)]
    max_weight = float(np.nanmax(finite_weights)) if finite_weights.size else 0.0
    if weight_ylim is not None:
        ax_bot.set_ylim(*weight_ylim)
    else:
        ax_bot.set_ylim(0, max_weight * 1.15 if max_weight > 0 else 1.0)
    ax_bot.set_xticks(pcs)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_loading_maps(
    *,
    loadings: Any,
    scores: Any,
    weights: Any,
    deposit_index: int,
    window_shape: tuple[int, int],
    output_path: str | Path,
    max_pcs: int = 4,
    image_cmap: str | Any | None = None,
    feature_mask: np.ndarray | None = None,
    origin: str = "upper",
) -> Path:
    """Plot weighted spatial PCA loading maps."""

    load = np.asarray(loadings, dtype=float)
    Z = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()

    win_h, win_w = int(window_shape[0]), int(window_shape[1])

    # If loadings are accidentally transposed, fix orientation.
    # Expected: load shape = (n_pcs, n_features)
    expected_features = win_h * win_w
    if load.shape[0] == expected_features and load.shape[1] != expected_features:
        load = load.T

    n_available = min(load.shape[0], Z.shape[1], w.size)
    n_show = min(int(max_pcs), n_available)

    if n_show < 1:
        raise ValueError("No loading maps are available to plot.")

    cmap = resolve_colormap(image_cmap or DEFAULT_IMAGE_CMAP)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    order = np.argsort(w[:n_available])[::-1][:n_show]

    fig, axes = plt.subplots(
        1,
        n_show,
        figsize=(4.6 * n_show + 0.8, 5.6),
        squeeze=False,
    )

    fig.suptitle(
        "Top-weighted windowed PCA loading maps (from ranking)",
        fontsize=22.0,
        y=1.005,
    )
    fig.subplots_adjust(top=0.80)

    selected_loadings = load[order, :]
    vmax = float(np.nanmax(np.abs(selected_loadings)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    last_im = None

    for ax, pc_idx in zip(axes.flat, order):
        loading_map = _vector_to_display_patch(
            load[pc_idx],
            (win_h, win_w),
            feature_mask=feature_mask,
        )

        last_im = ax.imshow(
            loading_map,
            origin=origin,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )

        ax.set_title(
            f"PC {pc_idx + 1}\n"
            rf"$z_{{d}}$ = {Z[int(deposit_index), pc_idx]:.2f}" "\n"
            f"w = {100 * w[pc_idx]:.1f}%",
            fontsize=23.0,
        )

        ax.set_xticks([])
        ax.set_yticks([])

    # Reserve space on the right for the colorbar
    fig.subplots_adjust(
        left=0.04,
        right=0.88,
        top=0.78,
        bottom=0.08,
        wspace=0.22,
    )

    if last_im is not None:
        cbar_ax = fig.add_axes([0.91, 0.18, 0.018, 0.58])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("PCA loading", fontsize=24.3)
        cbar.ax.tick_params(labelsize=20)

    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return path


def plot_fused_loading_maps(
    *,
    fusion_details: dict[str, Any],
    output_path: str | Path,
    max_pcs: int = 4,
) -> Path:
    """Plot fused PCA loading vectors over stage-1 PC features."""

    pca_fused = fusion_details["pca_fused"]
    K_var1 = int(fusion_details["K_var1"])
    K_var2 = int(fusion_details["K_var2"])
    components = np.asarray(pca_fused.components_, dtype=float)
    n_show = min(int(max_pcs), components.shape[0])
    if n_show < 1:
        raise ValueError("No fused loading vectors are available to plot.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.8 * n_show), squeeze=False)
    if SHOW_FIGURE_TITLES:
        fig.suptitle("Fused PCA loading vectors across stage-1 PCs", fontsize=22.0, y=1.005)
    vmax = float(np.nanmax(np.abs(components[:n_show, :])))

    for idx, ax in enumerate(axes.flat[:n_show]):
        row = components[idx : idx + 1, :]
        im = ax.imshow(row, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.axvline(K_var1 - 0.5, color="black", linewidth=1.0, linestyle="--")
        ax.set_yticks([])
        ax.set_title(f"Fused PC {idx + 1}", fontsize=18.9)
        ax.set_xlabel(f"Stage-1 PCs | first {K_var1} = var1, next {K_var2} = var2", fontsize=14.9)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Fused loading", fontsize=16.2)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_pairs(
    *,
    comparison_space: Any,
    weights: Any,
    deposit_index: int,
    output_path: str | Path,
    n_pairs: int | None = None,
    max_pairs: int | None = None,
    pc_pairs: Any | None = None,
    pc_index_base: int = 1,
    ranked_idx: Any | None = None,
    top_n_to_plot: int = 0,
    exclude_deposit_from_highlights: bool = True,
    highlight_color: str = "#1f77b4",
    highlight_size: int = 95,
    annotate_highlights: bool = True,
    star_size: int = 240,
    point_size: int = 18,
) -> Path | None:
    """Plot PCA score pairs ordered by descending deposit-weight percentage.

    By default, panels use adjacent PCs in descending weight order, e.g.
    PC1-vs-PC3, then PC3-vs-PC2, then PC2-vs-PC4. Pass ``pc_pairs`` to choose
    exact pairs manually. Manual pairs use 1-based PC numbers by default;
    set ``pc_index_base=0`` for zero-based indices.

    ``n_pairs`` controls how many automatic pair subplots are drawn.
    ``max_pairs`` is kept as a backwards-compatible alias for older callers.
    Pass ``ranked_idx`` with ``top_n_to_plot`` to highlight top-ranked windows.
    """

    Z = np.asarray(comparison_space, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    if Z.shape[1] < 2:
        return None
    k = min(Z.shape[1], w.size)
    n_pairs_to_plot = max_pairs if max_pairs is not None else n_pairs

    if pc_pairs is None:
        n_pairs_to_plot = 4 if n_pairs_to_plot is None else int(n_pairs_to_plot)
        if n_pairs_to_plot < 1:
            return None
        pc_order = np.argsort(w[:k])[::-1]
        pairs = [
            (int(pc_order[i]), int(pc_order[i + 1]))
            for i in range(min(n_pairs_to_plot, k - 1))
        ]
    else:
        pairs = [
            _normalize_pc_pair(pair, k=k, pc_index_base=pc_index_base)
            for pair in pc_pairs
        ]
        if n_pairs_to_plot is not None:
            n_pairs_to_plot = int(n_pairs_to_plot)
            if n_pairs_to_plot < 1:
                return None
            pairs = pairs[:n_pairs_to_plot]
    if not pairs:
        return None

    dep = Z[int(deposit_index), :k]
    dists = np.sqrt(np.sum((Z[:, :k] - dep) ** 2, axis=1))
    mask = np.ones(Z.shape[0], dtype=bool)
    mask[int(deposit_index)] = False
    highlight_indices = _select_score_pair_highlights(
        ranked_idx=ranked_idx,
        top_n_to_plot=top_n_to_plot,
        n_samples=Z.shape[0],
        deposit_index=int(deposit_index),
        exclude_deposit=exclude_deposit_from_highlights,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # lay the pairs out on two rows so the panel titles stop colliding
    n_pair = len(pairs)
    n_col = 2 if n_pair > 2 else n_pair
    n_row = int(math.ceil(n_pair / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(8.4 * n_col, 7.6 * n_row),
                             squeeze=False)
    for extra in axes.flat[n_pair:]:
        extra.axis("off")
    title_prefix = "Selected" if pc_pairs is not None else "Top-weighted"
    if SHOW_FIGURE_TITLES:
        fig.suptitle(f"{title_prefix} PCA score plots colored by distance",
                 fontsize=26.0, y=0.995)
    last_sc = None
    for ax, (pc_x, pc_y) in zip(axes.flat, pairs):
        ax.tick_params(axis="both", labelsize=18)
        last_sc = ax.scatter(
            Z[mask, pc_x],
            Z[mask, pc_y],
            c=dists[mask],
            cmap="magma_r",
            s=point_size,
            alpha=0.85,
            linewidths=0,
        )
        ax.scatter(
            [Z[int(deposit_index), pc_x]],
            [Z[int(deposit_index), pc_y]],
            marker="*",
            s=star_size,
            c="none",
            edgecolors="black",
            linewidths=1.5,
            label="Reference deposit",
            zorder=5,
        )
        if highlight_indices.size:
            ax.scatter(
                Z[highlight_indices, pc_x],
                Z[highlight_indices, pc_y],
                marker="o",
                s=highlight_size,
                facecolors="none",
                edgecolors=highlight_color,
                linewidths=2.0,
                label=f"Top {highlight_indices.size} ranked windows",
                zorder=6,
            )
            if annotate_highlights:
                for rank, window_idx in enumerate(highlight_indices, start=1):
                    text = ax.annotate(
                        str(rank),
                        xy=(Z[int(window_idx), pc_x], Z[int(window_idx), pc_y]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color=highlight_color,
                        fontsize=13.5,
                        fontweight="bold",
                        zorder=7,
                    )
                    text.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
        ax.set_title(
            f"PC {pc_x + 1} (w={100 * w[pc_x]:.1f}%) vs "
            f"PC {pc_y + 1} (w={100 * w[pc_y]:.1f}%)",
            fontsize=21.6,
        )
        ax.set_xlabel(f"PC {pc_x + 1} score", fontsize=18.9)
        ax.set_ylabel(f"PC {pc_y + 1} score", fontsize=18.9)
    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04)
        cbar.set_label("Distance to deposit", fontsize=21.6)
    if highlight_indices.size:
        axes.flat[0].legend(loc="best", fontsize=14.9, frameon=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def _select_score_pair_highlights(
    *,
    ranked_idx: Any | None,
    top_n_to_plot: int,
    n_samples: int,
    deposit_index: int,
    exclude_deposit: bool,
) -> np.ndarray:
    if ranked_idx is None or int(top_n_to_plot) <= 0:
        return np.asarray([], dtype=int)

    selected: list[int] = []
    for idx in np.asarray(ranked_idx, dtype=int).ravel():
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= int(n_samples):
            continue
        if exclude_deposit and idx_int == int(deposit_index):
            continue
        selected.append(idx_int)
        if len(selected) >= int(top_n_to_plot):
            break

    return np.asarray(selected, dtype=int)


def _normalize_pc_pair(pair: Any, *, k: int, pc_index_base: int) -> tuple[int, int]:
    if pc_index_base not in {0, 1}:
        raise ValueError("pc_index_base must be 0 or 1.")
    try:
        pc_x_raw, pc_y_raw = pair
    except (TypeError, ValueError) as exc:
        raise ValueError("Each pc_pairs entry must contain exactly two PC indices.") from exc

    pc_x = _normalize_pc_index(pc_x_raw, k=k, pc_index_base=pc_index_base)
    pc_y = _normalize_pc_index(pc_y_raw, k=k, pc_index_base=pc_index_base)
    if pc_x == pc_y:
        raise ValueError("Each PC pair must contain two different PCs.")
    return pc_x, pc_y


def _normalize_pc_index(value: Any, *, k: int, pc_index_base: int) -> int:
    if isinstance(value, str):
        value_clean = value.strip().upper()
        if value_clean.startswith("PC"):
            value_clean = value_clean[2:].strip()
        raw_index = int(value_clean)
    else:
        raw_index = int(value)

    pc_index = raw_index - pc_index_base
    if not 0 <= pc_index < k:
        pc_min = pc_index_base
        pc_max = k - 1 + pc_index_base
        raise ValueError(f"PC index {raw_index} is out of range [{pc_min}, {pc_max}].")
    return pc_index


def plot_reconstruction_progression(
    *,
    scores: Any,
    loadings: Any,
    mean: Any,
    std_safe: Any,
    deposit_index: int,
    deposit_1based: int,
    window_shape: tuple[int, int],
    optimal_k: int,
    output_path: str | Path,
    variable_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    max_k: int = 10,
    image_cmap: str | Any | None = None,
    feature_mask: np.ndarray | None = None,
) -> Path:
    """Plot deposit reconstructions using the first k PCs."""

    Z = np.asarray(scores, dtype=float)
    components = np.asarray(loadings, dtype=float)
    x_mean = np.asarray(mean, dtype=float).reshape(-1)
    x_std = np.asarray(std_safe, dtype=float).reshape(-1)
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    n_show = min(int(max_k), Z.shape[1], components.shape[0])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cmap = resolve_colormap(image_cmap or DEFAULT_IMAGE_CMAP)
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.4 * n_rows), squeeze=False)
    if SHOW_FIGURE_TITLES:
        fig.suptitle(f"Deposit {deposit_1based} reconstruction progression (optimal k={optimal_k})", fontsize=22.0, y=1.005)
    last_im = None
    z_dep = Z[int(deposit_index)]
    for idx, ax in enumerate(axes.flat):
        k = idx + 1
        if k > n_show:
            ax.axis("off")
            continue
        x_stdzd = z_dep[:k] @ components[:k, :]
        reconstruction = _vector_to_display_patch(
            x_stdzd * x_std + x_mean,
            (win_h, win_w),
            feature_mask=feature_mask,
        )
        last_im = ax.imshow(
            reconstruction,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Recon. using: k={k}", fontsize=20.2)
        ax.set_xticks([])
        ax.set_yticks([])
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
        cbar.set_label(_variable_label(variable_name), fontsize=21.6)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_two_stage_multivariate_reconstruction_progression(
    *,
    fusion_details: dict[str, Any],
    window_shape: tuple[int, int],
    variable_names: tuple[str, str],
    output_path: str | Path,
    max_k_fused: int,
    image_cmap: str | Any | None = None,
    vmin_by_var: dict[str, float | None] | None = None,
    vmax_by_var: dict[str, float | None] | None = None,
    title: str | None = None,
    feature_mask: np.ndarray | None = None,
) -> Path:
    """Plot reconstruction progression for the two-stage fused multivariate method."""

    h, w = int(window_shape[0]), int(window_shape[1])
    n_pix = int(np.asarray(feature_mask, dtype=bool).sum()) if feature_mask is not None else (h * w)
    K1 = int(fusion_details["K_var1"])
    K2 = int(fusion_details["K_var2"])
    M_fused = int(fusion_details["M_fused"])
    zf_dep_full = np.asarray(fusion_details["zf_dep_full"], dtype=float).ravel()
    pca1 = fusion_details["pca_var1"]
    pca2 = fusion_details["pca_var2"]
    pca_fused = fusion_details["pca_fused"]
    use_std_fused = bool(fusion_details["standardize_fused_input"])
    X1_mean = np.asarray(fusion_details["X1_mean"], dtype=float).ravel()
    X1_std_safe = np.asarray(fusion_details["X1_std_safe"], dtype=float).ravel()
    X2_mean = np.asarray(fusion_details["X2_mean"], dtype=float).ravel()
    X2_std_safe = np.asarray(fusion_details["X2_std_safe"], dtype=float).ravel()
    F_mu = fusion_details.get("F_mu", None)
    F_std_safe = fusion_details.get("F_std_safe", None)
    if use_std_fused:
        F_mu = np.asarray(F_mu, dtype=float).ravel()
        F_std_safe = np.asarray(F_std_safe, dtype=float).ravel()

    M_show = max(1, min(int(max_k_fused), M_fused))
    k_list = list(range(1, M_show + 1))
    cmap = resolve_colormap(image_cmap or DEFAULT_IMAGE_CMAP)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(k_list),
        2,
        figsize=(9, 2.6 * len(k_list)),
        dpi=400,
        squeeze=False,
    )

    im1_last = None
    im2_last = None
    for i, k_fused in enumerate(k_list):
        zf_k = np.zeros_like(zf_dep_full)
        zf_k[:k_fused] = zf_dep_full[:k_fused]

        F_in_hat = pca_fused.inverse_transform(zf_k[None, :]).ravel()
        F_hat = F_in_hat * F_std_safe + F_mu if use_std_fused else F_in_hat
        z1_hat_kept = F_hat[:K1]
        z2_hat_kept = F_hat[K1 : K1 + K2]

        z1_full = np.zeros(int(pca1.components_.shape[0]), dtype=float)
        z1_full[:K1] = z1_hat_kept
        x1_std_hat = pca1.inverse_transform(z1_full[None, :]).ravel()
        x1_hat = x1_std_hat * X1_std_safe + X1_mean

        z2_full = np.zeros(int(pca2.components_.shape[0]), dtype=float)
        z2_full[:K2] = z2_hat_kept
        x2_std_hat = pca2.inverse_transform(z2_full[None, :]).ravel()
        x2_hat = x2_std_hat * X2_std_safe + X2_mean

        m1 = _vector_to_display_patch(x1_hat[:n_pix], (h, w), feature_mask=feature_mask)
        m2 = _vector_to_display_patch(x2_hat[:n_pix], (h, w), feature_mask=feature_mask)
        v1min = None if vmin_by_var is None else vmin_by_var.get(variable_names[0])
        v1max = None if vmax_by_var is None else vmax_by_var.get(variable_names[0])
        v2min = None if vmin_by_var is None else vmin_by_var.get(variable_names[1])
        v2max = None if vmax_by_var is None else vmax_by_var.get(variable_names[1])

        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        im1_last = ax1.imshow(m1, cmap=cmap, vmin=v1min, vmax=v1max)
        im2_last = ax2.imshow(m2, cmap=cmap, vmin=v2min, vmax=v2max)
        ax1.set_title(f"{variable_names[0]} | fused k={k_fused}", fontsize=14.9)
        ax2.set_title(f"{variable_names[1]} | fused k={k_fused}", fontsize=14.9)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

    if title is None:
        title = "Two-stage fused PCA reconstruction progression"
    fig.suptitle(title, y=0.995, fontsize=21.6)
    fig.subplots_adjust(right=0.90, top=0.96, hspace=0.35, wspace=0.15)
    if im1_last is not None:
        cax1 = fig.add_axes([0.92, 0.55, 0.015, 0.35])
        cb1 = fig.colorbar(im1_last, cax=cax1)
        cb1.set_label(_variable_label(variable_names[0]), fontsize=13.5)
    if im2_last is not None:
        cax2 = fig.add_axes([0.92, 0.12, 0.015, 0.35])
        cb2 = fig.colorbar(im2_last, cax=cax2)
        cb2.set_label(_variable_label(variable_names[1]), fontsize=13.5)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def _window_centers_from_indices(
    window_indices: np.ndarray,
    window_shape: tuple[int, int],
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray]:
    if window_indices.shape[1] >= 5:
        center_rows = window_indices[:, 3].astype(float) + 0.5
        center_cols = window_indices[:, 4].astype(float) + 0.5
    else:
        win_h = int(window_shape[0])
        win_w = int(window_shape[1])
        center_rows = window_indices[:, 0].astype(float) + 0.5 * win_h
        center_cols = window_indices[:, 1].astype(float) + 0.5 * win_w
    x = transform.c + center_cols * transform.a + center_rows * transform.b
    y = transform.f + center_cols * transform.d + center_rows * transform.e
    return x, y


def _variable_label(variable_name: str) -> str:
    return variable_display_label(variable_name)


def _vector_to_display_patch(
    vector: Any,
    window_shape: tuple[int, int],
    *,
    feature_mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(vector, dtype=float).ravel()
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    if feature_mask is None:
        return arr.reshape((win_h, win_w))

    mask = np.asarray(feature_mask, dtype=bool)
    if mask.shape != (win_h, win_w):
        raise ValueError(
            f"feature_mask shape {mask.shape} does not match window_shape {(win_h, win_w)}."
        )
    if arr.size != int(mask.sum()):
        raise ValueError(
            f"Vector length {arr.size} does not match number of True cells in feature_mask {int(mask.sum())}."
        )
    out = np.full(mask.shape, np.nan, dtype=float)
    out[mask] = arr
    return out
