"""3D surface plotting helpers for extracted deposit templates."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "spatial_pca_matplotlib_cache"),
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from spatial_pca.colormaps import DEFAULT_PAPER_CMAP, resolve_colormap
from spatial_pca.units import variable_display_label


DEFAULT_SURFACE_CMAP = DEFAULT_PAPER_CMAP
TITLE_FONTSIZE = 28
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 18
COLORBAR_LABEL_FONTSIZE = 22
# x position (axes fraction) of the rotated variable name beside the z axis.
# Negative keeps it clear of the z tick numbers; bbox_inches="tight" trims the margin.
VARIABLE_LABEL_X = -0.20
# Journal artwork rules (Springer NRR): no figure-level titles inside the
# artwork; the information lives in the LaTeX caption. Panel labels stay.
SHOW_FIGURE_TITLES = False


def plot_deposit_surface(
    *,
    deposit_array: Any,
    deposit_extent: tuple[float, float, float, float],
    deposit_1based: int,
    variable_name: str,
    output_path: str | Path,
    vmin: float | None = None,
    vmax: float | None = None,
    image_cmap: str | Any | None = None,
    max_grid_points: int = 40_000,
    surface_alpha: float = 0.42,
    view_elev: float = 26.0,
    view_azim: float = 28.0,
    mesh_linewidth: float = 0.3,
    mesh_alpha: float = 0.8,
    vertical_exaggeration: float = 0.6,
) -> Path:
    """Plot one deposit template as a 3D x/y/z surface.

    The x and y axes use the template map coordinates. The z axis is the
    selected geophysical variable value from the deposit raster patch.
    """

    return plot_deposit_surfaces(
        deposit_arrays={variable_name: deposit_array},
        deposit_extents={variable_name: deposit_extent},
        deposit_1based=deposit_1based,
        output_path=output_path,
        vmin_by_var={variable_name: vmin},
        vmax_by_var={variable_name: vmax},
        image_cmap=image_cmap,
        max_grid_points=max_grid_points,
        surface_alpha=surface_alpha,
        view_elev=view_elev,
        view_azim=view_azim,
        mesh_linewidth=mesh_linewidth,
        mesh_alpha=mesh_alpha,
        vertical_exaggeration=vertical_exaggeration,
    )


def plot_deposit_surfaces(
    *,
    deposit_arrays: Mapping[str, Any],
    deposit_extents: Mapping[str, tuple[float, float, float, float]],
    deposit_1based: int,
    output_path: str | Path,
    vmin_by_var: Mapping[str, float | None] | None = None,
    vmax_by_var: Mapping[str, float | None] | None = None,
    image_cmap: str | Any | None = None,
    max_grid_points: int = 40_000,
    surface_alpha: float = 0.42,
    view_elev: float = 26.0,
    view_azim: float = 28.0,
    mesh_linewidth: float = 0.2,
    mesh_alpha: float = 0.3,
    vertical_exaggeration: float = 1.25,
) -> Path:
    """Plot one or more deposit raster templates as 3D surfaces."""

    if not deposit_arrays:
        raise ValueError("deposit_arrays must contain at least one variable.")
    if not 0.0 <= float(surface_alpha) <= 1.0:
        raise ValueError("surface_alpha must be between 0 and 1.")
    if not 0.0 <= float(mesh_alpha) <= 1.0:
        raise ValueError("mesh_alpha must be between 0 and 1.")
    if float(mesh_linewidth) < 0.0:
        raise ValueError("mesh_linewidth must be non-negative.")
    if float(vertical_exaggeration) <= 0.0:
        raise ValueError("vertical_exaggeration must be greater than 0.")

    ordered_vars = list(deposit_arrays)
    cmap = resolve_colormap(image_cmap or DEFAULT_SURFACE_CMAP)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.8 * len(ordered_vars), 7.6), dpi=150)
    if SHOW_FIGURE_TITLES:
        fig.suptitle(
            f"Geometry of Reference Deposit {deposit_1based}",
            fontsize=TITLE_FONTSIZE,
        )

    for idx, variable_name in enumerate(ordered_vars, start=1):
        if variable_name not in deposit_extents:
            raise ValueError(f"Missing deposit extent for variable '{variable_name}'.")

        z_values = np.asarray(deposit_arrays[variable_name], dtype=float)
        x_grid, y_grid, z_grid = _surface_grids(
            z_values,
            deposit_extents[variable_name],
            max_grid_points=max_grid_points,
        )
        finite = z_grid[np.isfinite(z_grid)]
        if finite.size == 0:
            raise ValueError(
                f"Deposit surface for '{variable_name}' has no finite z values."
            )

        vmin = None if vmin_by_var is None else vmin_by_var.get(variable_name)
        vmax = None if vmax_by_var is None else vmax_by_var.get(variable_name)

        ax = fig.add_subplot(1, len(ordered_vars), idx, projection="3d")
        surface = ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.2,
            edgecolor=(0.2, 0.2, 0.2, 0.28),
            alpha=float(surface_alpha),
            antialiased=True,
            shade=True,
        )
        if float(mesh_linewidth) > 0.0 and float(mesh_alpha) > 0.0:
            ax.plot_wireframe(
                x_grid,
                y_grid,
                z_grid,
                color=(0.05, 0.05, 0.05, float(mesh_alpha)),
                linewidth=float(mesh_linewidth),
                rstride=1,
                cstride=1,
        )
        ax.view_init(elev=float(view_elev), azim=float(view_azim))
        ax.set_xlabel("Easting", fontsize=LABEL_FONTSIZE, labelpad=16)
        ax.set_ylabel("Northing", fontsize=LABEL_FONTSIZE, labelpad=16)
        ax.set_zlabel("")
        ax.text2D(
            VARIABLE_LABEL_X,
            0.42,
            variable_display_label(variable_name),
            transform=ax.transAxes,
            rotation=90,
            fontsize=LABEL_FONTSIZE,
            ha="center",
            va="center",
        )
        _set_min_max_axis_ticks(ax, x_grid, y_grid, z_grid)
        ax.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE, pad=6)
        ax.zaxis.set_tick_params(labelsize=TICK_FONTSIZE, pad=8)
        _set_surface_box_aspect(
            ax,
            x_grid,
            y_grid,
            z_grid,
            vertical_exaggeration=float(vertical_exaggeration),
        )
        cbar = fig.colorbar(surface, ax=ax, fraction=0.035, pad=0.08, shrink=0.72)
        _set_min_max_colorbar_ticks(cbar, finite, vmin=vmin, vmax=vmax)
        cbar.set_label(variable_display_label(variable_name), fontsize=COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(left=0.04, right=0.92, top=0.86, bottom=0.08, wspace=0.24)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_deposit_and_loading_surfaces(
    *,
    deposit_array: Any,
    deposit_extent: tuple[float, float, float, float],
    loadings: Any,
    scores: Any,
    weights: Any,
    deposit_index: int,
    deposit_1based: int,
    window_shape: tuple[int, int],
    output_path: str | Path,
    variable_name: str,
    max_pcs: int = 4,
    deposit_vmin: float | None = None,
    deposit_vmax: float | None = None,
    image_cmap: str | Any | None = None,
    loading_cmap: str | Any | None = None,
    feature_mask: np.ndarray | None = None,
    surface_alpha: float = 0.42,
    view_elev: float = 26.0,
    view_azim: float = 28.0,
    mesh_linewidth: float = 0.2,
    mesh_alpha: float = 0.3,
    vertical_exaggeration: float = 0.6,
    tick_fontsize: int = 13,
    figsize: tuple[float, float] | None = None,
    fontsize: float | None = None,
    dpi: int = 200,
    line_scale: float = 1.0,
    xy_tick_formatter: Any | None = None,
    z_tick_formatter: Any | None = None,
    pc_panel_xy_ticks: bool = True,
    pc_panel_z_ticks: bool = True,
    top_panel_xy_labels: bool = True,
    top_panel_xy_ticks: bool = True,
    show_top_variable_label: bool = True,
    variable_label_x: float = VARIABLE_LABEL_X,
    xy_labelpad: float = 16,
    tick_pad: float = 6,
    z_tick_pad: float = 8,
    xlabel: str = "Easting",
    ylabel: str = "Northing",
) -> Path:
    """Plot the training deposit above top-weighted loading maps as 3D surfaces.

    ``tick_fontsize`` sets the x/y/z coordinate tick labels of the 3D panels;
    it defaults below the module TICK_FONTSIZE because this figure's long
    coordinate numbers crowd each other at the standard size.
    """

    load = np.asarray(loadings, dtype=float)
    Z = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    win_h, win_w = int(window_shape[0]), int(window_shape[1])

    expected_features = win_h * win_w
    if load.shape[0] == expected_features and load.shape[1] != expected_features:
        load = load.T

    n_available = min(load.shape[0], Z.shape[1], w.size)
    n_show = min(int(max_pcs), n_available)
    if n_show < 1:
        raise ValueError("No loading maps are available to plot as 3D surfaces.")

    order = np.argsort(w[:n_available])[::-1][:n_show]
    loading_maps = [
        _vector_to_display_patch(
            load[pc_idx],
            (win_h, win_w),
            feature_mask=feature_mask,
        )
        for pc_idx in order
    ]
    loading_vmax = float(np.nanmax(np.abs(np.asarray(loading_maps, dtype=float))))
    if not np.isfinite(loading_vmax) or loading_vmax == 0.0:
        loading_vmax = 1.0

    deposit_cmap = resolve_colormap(image_cmap or DEFAULT_SURFACE_CMAP)
    load_cmap = resolve_colormap(loading_cmap or image_cmap or DEFAULT_SURFACE_CMAP)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs_panel = 22 if fontsize is None else float(fontsize)
    fs_label = LABEL_FONTSIZE if fontsize is None else float(fontsize)
    fs_cbar_label = COLORBAR_LABEL_FONTSIZE if fontsize is None else float(fontsize)
    fs_cbar_tick = TICK_FONTSIZE if fontsize is None else max(8.0, float(fontsize) - 1.0)
    panel_pad = 8 if fontsize is None else 3
    n_grid_cols = max(6, 2 * n_show)
    top_margin_cols = max(1, n_grid_cols // 4)
    fig = plt.figure(figsize=figsize if figsize is not None else (5.6 * n_show, 9.6), dpi=150)
    grid = fig.add_gridspec(
        2,
        n_grid_cols,
        height_ratios=[1.05, 1.0],
        hspace=0.32,
        wspace=0.20,
    )
    if SHOW_FIGURE_TITLES:
        fig.suptitle(
            f"Reference Deposit {deposit_1based} and PCA Loading Maps",
            fontsize=TITLE_FONTSIZE,
            y=0.98,
        )

    ax_top = fig.add_subplot(
        grid[0, top_margin_cols : n_grid_cols - top_margin_cols],
        projection="3d",
    )
    top_surface = _plot_surface_on_axis(
        ax_top,
        deposit_array,
        deposit_extent,
        cmap=deposit_cmap,
        vmin=deposit_vmin,
        vmax=deposit_vmax,
        label=variable_display_label(variable_name),
        view_elev=view_elev,
        view_azim=view_azim,
        surface_alpha=surface_alpha,
        mesh_linewidth=mesh_linewidth,
        mesh_alpha=mesh_alpha,
        vertical_exaggeration=vertical_exaggeration,
        show_xy_labels=top_panel_xy_labels,
        show_xy_ticks=top_panel_xy_ticks,
        show_variable_label=show_top_variable_label,
        tick_fontsize=tick_fontsize,
        xlabel=xlabel,
        ylabel=ylabel,
        xy_label_fontsize=fs_label,
        xy_labelpad=xy_labelpad,
        variable_label_fontsize=fs_label,
        variable_label_x=variable_label_x,
        line_scale=line_scale,
        tick_formatter=xy_tick_formatter,
        z_tick_formatter=z_tick_formatter,
        tick_pad=tick_pad,
        z_tick_pad=z_tick_pad,
    )
    ax_top.set_position([0.255, 0.47, 0.49, 0.45])
    cbar_top_ax = fig.add_axes([0.775, 0.55, 0.013, 0.29])
    cbar_top = fig.colorbar(top_surface, cax=cbar_top_ax)
    _set_min_max_colorbar_ticks(
        cbar_top,
        np.asarray(deposit_array, dtype=float),
        vmin=deposit_vmin,
        vmax=deposit_vmax,
    )
    cbar_top.set_label(variable_display_label(variable_name), fontsize=fs_cbar_label)
    cbar_top.ax.tick_params(labelsize=fs_cbar_tick)

    bottom_axes = []
    for col, pc_idx in enumerate(order):
        col_start = int(round(col * n_grid_cols / n_show))
        col_stop = int(round((col + 1) * n_grid_cols / n_show))
        ax = fig.add_subplot(grid[1, col_start:col_stop], projection="3d")
        _plot_surface_on_axis(
            ax,
            loading_maps[col],
            deposit_extent,
            cmap=load_cmap,
            vmin=-loading_vmax,
            vmax=loading_vmax,
            label="Loading",
            view_elev=view_elev,
            view_azim=view_azim,
            surface_alpha=surface_alpha,
            mesh_linewidth=mesh_linewidth,
            mesh_alpha=mesh_alpha,
            vertical_exaggeration=vertical_exaggeration,
            show_xy_labels=False,
            show_variable_label=False,
            tick_fontsize=tick_fontsize,
            line_scale=line_scale,
            tick_formatter=xy_tick_formatter,
            z_tick_formatter=z_tick_formatter,
            show_xy_ticks=pc_panel_xy_ticks,
            show_z_ticks=pc_panel_z_ticks,
            tick_pad=tick_pad,
            z_tick_pad=z_tick_pad,
        )
        if fontsize is None:
            panel_title = (
                f"PC {pc_idx + 1}\n"
                rf"$z_{{d}}$={Z[int(deposit_index), pc_idx]:.2f} | "
                f"w={100 * w[pc_idx]:.1f}%"
            )
        else:
            panel_title = (
                f"PC {pc_idx + 1}\n"
                rf"$z_{{d}}$ = {Z[int(deposit_index), pc_idx]:.2f}" "\n"
                f"w = {100 * w[pc_idx]:.1f}%"
            )
        ax.set_title(panel_title, fontsize=fs_panel, pad=panel_pad)
        bottom_axes.append(ax)

    norm = colors.Normalize(vmin=-loading_vmax, vmax=loading_vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=load_cmap)
    sm.set_array([])
    cbar_bottom_ax = fig.add_axes([0.945, 0.09, 0.013, 0.29])
    cbar_bottom = fig.colorbar(sm, cax=cbar_bottom_ax)
    _set_min_max_colorbar_ticks(
        cbar_bottom,
        np.asarray(loading_maps, dtype=float),
        vmin=-loading_vmax,
        vmax=loading_vmax,
    )
    cbar_bottom.set_label("PCA loading", fontsize=fs_cbar_label)
    cbar_bottom.ax.tick_params(labelsize=fs_cbar_tick)

    fig.subplots_adjust(left=0.03, right=0.90, top=0.92, bottom=0.02)
    ax_top.set_position([0.255, 0.47, 0.49, 0.45])
    cbar_top_ax.set_position([0.775, 0.55, 0.013, 0.29])
    cbar_bottom_ax.set_position([0.945, 0.09, 0.013, 0.29])
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_top_similar_windows_surfaces(
    *,
    flattened_windows: Any,
    ranked_window_rows: Any,
    ranked_distances: Any,
    window_ids: Any,
    window_shape: tuple[int, int],
    output_path: str | Path,
    variable_name: str,
    n_rows: int = 2,
    n_cols: int = 4,
    vmin: float | None = None,
    vmax: float | None = None,
    image_cmap: str | Any | None = None,
    feature_mask: np.ndarray | None = None,
    surface_alpha: float = 0.42,
    view_elev: float = 26.0,
    view_azim: float = 28.0,
    mesh_linewidth: float = 0.2,
    mesh_alpha: float = 0.3,
    vertical_exaggeration: float = 0.35,
    figsize: tuple[float, float] | None = None,
    fontsize: float | None = None,
    tick_fontsize: float | None = None,
    dpi: int = 200,
    line_scale: float = 1.0,
    plain_panel_titles: bool = False,
    show_variable_label: bool = True,
    show_xy_labels: bool = True,
    show_xy_ticks: bool = True,
    xlabel: str = "Easting (pixels)",
    ylabel: str = "Northing (pixels)",
    z_tick_formatter: Any | None = None,
    variable_label_x: float = VARIABLE_LABEL_X,
    xy_labelpad: float = 8,
    tick_pad: float = 6,
    z_tick_pad: float = 8,
    hspace: float = 0.34,
    wspace: float = 0.18,
) -> Path:
    """Plot top-ranked similar windows as 3D surfaces.

    Journal mode: give ``figsize`` at the printed size and a base ``fontsize``;
    ``plain_panel_titles`` writes "Rank 1, d = 3.61" instead of the code-style
    "Rank 1 | idx=1189"; ``z_tick_formatter`` avoids scientific tick text and
    ``show_xy_ticks``/``show_xy_labels`` drop horizontal coordinate text that
    does not fit on small panels.
    """

    windows = np.asarray(flattened_windows, dtype=float)
    rows = np.asarray(ranked_window_rows, dtype=int)
    dists = np.asarray(ranked_distances, dtype=float)
    ids = np.asarray(window_ids, dtype=int)
    n_show = min(int(n_rows) * int(n_cols), rows.size)
    if n_show < 1:
        raise ValueError("No top similar windows are available to plot as 3D surfaces.")

    cmap = resolve_colormap(image_cmap or DEFAULT_SURFACE_CMAP)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    patches = [
        _vector_to_display_patch(
            windows[rows[i]],
            window_shape,
            feature_mask=feature_mask,
        )
        for i in range(n_show)
    ]
    fs_panel = 16 if fontsize is None else float(fontsize)
    fs_label = LABEL_FONTSIZE if fontsize is None else float(fontsize)
    fs_xy_label = 13 if fontsize is None else float(fontsize)
    if tick_fontsize is not None:
        fs_tick = float(tick_fontsize)
    elif fontsize is None:
        fs_tick = TICK_FONTSIZE
    else:
        fs_tick = max(8.0, float(fontsize) - 1.0)
    fs_cbar_label = COLORBAR_LABEL_FONTSIZE if fontsize is None else float(fontsize)
    fig = plt.figure(
        figsize=figsize if figsize is not None else (5.3 * int(n_cols), 4.6 * int(n_rows) + 1.7),
        dpi=150,
    )
    if SHOW_FIGURE_TITLES:
        fig.suptitle(
            f"Top {n_show} Most Similar Windows - 3D",
            fontsize=TITLE_FONTSIZE,
            y=0.995,
        )
    grid = fig.add_gridspec(int(n_rows), int(n_cols), hspace=hspace, wspace=wspace,
                            top=0.86)
    axes = []
    for i, patch in enumerate(patches):
        ax = fig.add_subplot(grid[i // int(n_cols), i % int(n_cols)], projection="3d")
        surface = _plot_surface_on_axis(
            ax,
            patch,
            _unit_extent(patch),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label=variable_display_label(variable_name),
            view_elev=view_elev,
            view_azim=view_azim,
            surface_alpha=surface_alpha,
            mesh_linewidth=mesh_linewidth,
            mesh_alpha=mesh_alpha,
            vertical_exaggeration=vertical_exaggeration,
            show_xy_labels=show_xy_labels,
            xlabel=xlabel,
            ylabel=ylabel,
            xy_label_fontsize=fs_xy_label,
            xy_labelpad=xy_labelpad,
            show_variable_label=show_variable_label and i % int(n_cols) == 0,
            variable_label_fontsize=fs_label,
            variable_label_x=variable_label_x,
            tick_fontsize=fs_tick,
            line_scale=line_scale,
            z_tick_formatter=z_tick_formatter,
            show_xy_ticks=show_xy_ticks,
            tick_pad=tick_pad,
            z_tick_pad=z_tick_pad,
        )
        if plain_panel_titles:
            panel_title = f"Rank {i + 1}\nd = {float(dists[i]):.2f}"
        else:
            panel_title = f"Rank {i + 1} | idx={int(ids[i])}\nd={float(dists[i]):.3f}"
        ax.set_title(panel_title, fontsize=fs_panel, pad=8 if fontsize is None else 2)
        axes.append(ax)

    fig.subplots_adjust(left=0.03, right=0.88, top=0.90, bottom=0.05)
    cbar_ax = fig.add_axes([0.915, 0.16, 0.013, 0.58])
    cbar = fig.colorbar(surface, cax=cbar_ax)
    _set_min_max_colorbar_ticks(
        cbar,
        np.asarray(patches, dtype=float),
        vmin=vmin,
        vmax=vmax,
    )
    cbar.set_label(variable_display_label(variable_name), fontsize=fs_cbar_label)
    cbar.ax.tick_params(labelsize=fs_tick)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_reconstruction_surface_animation(
    *,
    scores: Any,
    loadings: Any,
    mean: Any,
    std_safe: Any,
    deposit_index: int,
    deposit_1based: int,
    deposit_extent: tuple[float, float, float, float],
    deposit_reference_array: Any,
    window_shape: tuple[int, int],
    optimal_k: int,
    output_path: str | Path,
    variable_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
    max_k: int = 10,
    image_cmap: str | Any | None = None,
    feature_mask: np.ndarray | None = None,
    surface_alpha: float = 0.42,
    view_elev: float = 26.0,
    view_azim: float = 28.0,
    mesh_linewidth: float = 0.2,
    mesh_alpha: float = 0.3,
    vertical_exaggeration: float = 0.6,
    duration_ms: int = 650,
) -> Path:
    """Animate training-deposit reconstruction as PCs are added."""

    from PIL import Image

    Z = np.asarray(scores, dtype=float)
    components = np.asarray(loadings, dtype=float)
    x_mean = np.asarray(mean, dtype=float).reshape(-1)
    x_std = np.asarray(std_safe, dtype=float).reshape(-1)
    win_h, win_w = int(window_shape[0]), int(window_shape[1])
    n_show = min(int(max_k), Z.shape[1], components.shape[0])
    if n_show < 1:
        raise ValueError("No reconstruction frames are available.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cmap = resolve_colormap(image_cmap or DEFAULT_SURFACE_CMAP)
    z_dep = Z[int(deposit_index)]
    reconstructions: list[np.ndarray] = []
    for k in range(1, n_show + 1):
        x_stdzd = z_dep[:k] @ components[:k, :]
        reconstructions.append(
            _vector_to_display_patch(
                x_stdzd * x_std + x_mean,
                (win_h, win_w),
                feature_mask=feature_mask,
            )
    )
    deposit_min, deposit_max = _finite_min_max(np.asarray(deposit_reference_array, dtype=float))
    z_min = deposit_min
    z_max = deposit_max
    if z_min == z_max:
        z_max = z_min + 1.0
    zlim = (z_min, z_max)
    color_vmin = deposit_min if vmin is None else float(vmin)
    color_vmax = deposit_max if vmax is None else float(vmax)

    frames: list[Image.Image] = []
    frame_paths: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="spatial_pca_recon3d_") as tmpdir:
        tmp_path = Path(tmpdir)
        for k, reconstruction in enumerate(reconstructions, start=1):
            fig = plt.figure(figsize=(8.8, 7.6), dpi=150)
            fig.suptitle(
                f"Deposit {deposit_1based} geometry reconstruction",
                fontsize=TITLE_FONTSIZE,
            )
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            surface = _plot_surface_on_axis(
                ax,
                reconstruction,
                deposit_extent,
                cmap=cmap,
                vmin=color_vmin,
                vmax=color_vmax,
                label=variable_display_label(variable_name),
                view_elev=view_elev,
                view_azim=view_azim,
                surface_alpha=surface_alpha,
                mesh_linewidth=mesh_linewidth,
                mesh_alpha=mesh_alpha,
                vertical_exaggeration=vertical_exaggeration,
                show_xy_labels=False,
                zlim=zlim,
            )
            ax.set_title(
                f"Using first {k} PC{'s' if k != 1 else ''}"
                + (f" | optimal k={optimal_k}" if int(optimal_k) == k else ""),
                fontsize=LABEL_FONTSIZE,
                pad=14,
            )
            cbar_ax = fig.add_axes([0.88, 0.22, 0.018, 0.54])
            cbar = fig.colorbar(surface, cax=cbar_ax)
            _set_min_max_colorbar_ticks(
                cbar,
                reconstruction,
                vmin=color_vmin,
                vmax=color_vmax,
            )
            cbar.set_label(variable_display_label(variable_name), fontsize=COLORBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
            fig.subplots_adjust(left=0.04, right=0.84, top=0.86, bottom=0.08)

            frame_path = tmp_path / f"frame_{k:03d}.png"
            fig.savefig(frame_path, dpi=160, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            frame_paths.append(frame_path)

        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(image.convert("P", palette=Image.ADAPTIVE).copy())

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return path


def _plot_surface_on_axis(
    ax: Any,
    z_values: Any,
    extent: tuple[float, float, float, float],
    *,
    cmap: Any,
    vmin: float | None,
    vmax: float | None,
    label: str,
    view_elev: float,
    view_azim: float,
    surface_alpha: float,
    mesh_linewidth: float,
    mesh_alpha: float,
    vertical_exaggeration: float,
    show_xy_labels: bool,
    show_variable_label: bool = True,
    zlim: tuple[float, float] | None = None,
    tick_fontsize: int = TICK_FONTSIZE,
    xlabel: str = "Easting",
    ylabel: str = "Northing",
    xy_label_fontsize: float = LABEL_FONTSIZE,
    xy_labelpad: float = 16,
    variable_label_fontsize: float = LABEL_FONTSIZE,
    variable_label_x: float = VARIABLE_LABEL_X,
    line_scale: float = 1.0,
    tick_formatter: Any | None = None,
    z_tick_formatter: Any | None = None,
    show_xy_ticks: bool = True,
    show_z_ticks: bool = True,
    tick_pad: float = 6,
    z_tick_pad: float = 8,
) -> Any:
    x_grid, y_grid, z_grid = _surface_grids(
        np.asarray(z_values, dtype=float),
        extent,
        max_grid_points=40_000,
    )
    surface = ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.2 * float(line_scale),
        edgecolor=(0.2, 0.2, 0.2, 0.28),
        alpha=float(surface_alpha),
        antialiased=True,
        shade=True,
    )
    if float(mesh_linewidth) > 0.0 and float(mesh_alpha) > 0.0:
        ax.plot_wireframe(
            x_grid,
            y_grid,
            z_grid,
            color=(0.05, 0.05, 0.05, float(mesh_alpha)),
            linewidth=float(mesh_linewidth) * float(line_scale),
            rstride=1,
            cstride=1,
        )
    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    if show_xy_labels:
        ax.set_xlabel(xlabel, fontsize=xy_label_fontsize, labelpad=xy_labelpad)
        ax.set_ylabel(ylabel, fontsize=xy_label_fontsize, labelpad=xy_labelpad)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.set_zlabel("")
    if show_variable_label:
        ax.text2D(
            variable_label_x,
            0.42,
            label,
            transform=ax.transAxes,
            rotation=90,
            fontsize=variable_label_fontsize,
            ha="center",
            va="center",
        )
    _set_min_max_axis_ticks(
        ax,
        x_grid,
        y_grid,
        z_grid,
        tick_formatter=tick_formatter,
        z_tick_formatter=z_tick_formatter,
        show_xy_ticks=show_xy_ticks,
        show_z_ticks=show_z_ticks,
    )
    if zlim is not None:
        ax.set_zlim(float(zlim[0]), float(zlim[1]))
        ax.set_zticks([float(zlim[0]), float(zlim[1])])
        ax.set_zticklabels([_format_tick(float(zlim[0])), _format_tick(float(zlim[1]))])
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, pad=tick_pad)
    ax.zaxis.set_tick_params(labelsize=tick_fontsize, pad=z_tick_pad)
    _set_surface_box_aspect(
        ax,
        x_grid,
        y_grid,
        z_grid,
        vertical_exaggeration=float(vertical_exaggeration),
    )
    return surface



def _unit_extent(array: Any) -> tuple[float, float, float, float]:
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {arr.shape}.")
    rows, cols = arr.shape
    return 0.0, float(cols), 0.0, float(rows)


def _surface_grids(
    z_values: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    max_grid_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if z_values.ndim != 2:
        raise ValueError(f"deposit_array must be 2D, got shape {z_values.shape}.")
    if max_grid_points < 1:
        raise ValueError("max_grid_points must be positive.")

    rows, cols = z_values.shape
    if rows < 1 or cols < 1:
        raise ValueError("deposit_array must be non-empty.")

    left, right, bottom, top = (float(value) for value in extent)
    dx = (right - left) / float(cols)
    dy = (top - bottom) / float(rows)
    x = np.linspace(left + 0.5 * dx, right - 0.5 * dx, cols)
    y = np.linspace(top - 0.5 * dy, bottom + 0.5 * dy, rows)  # row 0 = north, matching imshow(origin='upper') in the 2D plots

    step = int(np.ceil(np.sqrt(z_values.size / float(max_grid_points))))
    step = max(1, step)
    z_grid = z_values[::step, ::step]
    x_grid, y_grid = np.meshgrid(x[::step], y[::step])
    return x_grid, y_grid, z_grid


def _set_surface_box_aspect(
    ax: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    vertical_exaggeration: float = 0.5,
) -> None:
    finite_z = z_grid[np.isfinite(z_grid)]
    if finite_z.size == 0:
        return

    x_span = _finite_span(x_grid)
    y_span = _finite_span(y_grid)
    z_span = _finite_span(finite_z)
    xy_span = max(x_span, y_span, 1.0)
    z_visual = max(0.35 * xy_span, min(z_span, 0.75 * xy_span))
    z_visual *= float(vertical_exaggeration)

    try:
        ax.set_box_aspect((max(x_span, 1.0), max(y_span, 1.0), z_visual))
    except AttributeError:
        return


def _set_min_max_axis_ticks(
    ax: Any,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    *,
    tick_formatter: Any | None = None,
    z_tick_formatter: Any | None = None,
    show_xy_ticks: bool = True,
    show_z_ticks: bool = True,
) -> None:
    """Label each 3D axis with its minimum and maximum only.

    ``tick_formatter`` replaces the default number formatting (used to print
    map coordinates in km and to avoid scientific "e3" tick text);
    ``show_xy_ticks=False`` drops the horizontal coordinate ticks, which do
    not fit on small multi-panel figures.
    """

    fmt = tick_formatter or _format_tick
    z_fmt = z_tick_formatter or _format_tick
    x_min, x_max = _finite_min_max(x_grid)
    y_min, y_max = _finite_min_max(y_grid)
    z_min, z_max = _finite_min_max(z_grid)

    if show_xy_ticks:
        ax.set_xticks([x_min, x_max])
        ax.set_yticks([y_min, y_max])
        ax.set_xticklabels([fmt(x_min), fmt(x_max)])
        ax.set_yticklabels([fmt(y_min), fmt(y_max)])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if show_z_ticks:
        ax.set_zticks([z_min, z_max])
        ax.set_zticklabels([z_fmt(z_min), z_fmt(z_max)])
    else:
        ax.set_zticks([])


def _set_min_max_colorbar_ticks(
    cbar: Any,
    values: np.ndarray,
    *,
    vmin: float | None,
    vmax: float | None,
) -> None:
    data_min, data_max = _finite_min_max(values)
    tick_min = data_min if vmin is None else float(vmin)
    tick_max = data_max if vmax is None else float(vmax)
    cbar.set_ticks([tick_min, tick_max])
    cbar.set_ticklabels([_format_tick(tick_min), _format_tick(tick_max)])


def _finite_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    min_value = float(np.nanmin(finite))
    max_value = float(np.nanmax(finite))
    if min_value == max_value:
        return min_value, max_value + 1.0
    return min_value, max_value


def _format_tick(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 1000 or (0 < abs_value < 0.01):
        return _format_scientific_tick(value)
    if abs_value >= 10:
        return f"{value:.0f}"
    if abs_value >= 1:
        return f"{value:.1f}"
    return f"{value:.2g}"


def _format_scientific_tick(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_value = int(exponent)
    return f"{mantissa}e{exponent_value}"


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


def _finite_span(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    span = float(np.nanmax(finite) - np.nanmin(finite))
    return span if span > 0.0 else 1.0
