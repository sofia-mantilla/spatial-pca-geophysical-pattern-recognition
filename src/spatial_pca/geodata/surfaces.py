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

from spatial_pca.colormaps import DEFAULT_PAPER_CMAP, resolve_colormap


DEFAULT_SURFACE_CMAP = DEFAULT_PAPER_CMAP
TITLE_FONTSIZE = 28
LABEL_FONTSIZE = 22
TICK_FONTSIZE = 20
COLORBAR_LABEL_FONTSIZE = 22


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
    view_elev: float = 25.0,
    view_azim: float = 28.0,
    mesh_linewidth: float = 0.3,
    mesh_alpha: float = 0.8,
    vertical_exaggeration: float = 1.2,
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
    vertical_exaggeration: float = 2.5,
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
    fig.suptitle(
        f"Geometry of Training Deposit ID={deposit_1based}",
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
            0.02,
            0.37,
            variable_name,
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
        cbar.set_label(variable_name, fontsize=COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    fig.subplots_adjust(left=0.04, right=0.92, top=0.86, bottom=0.08, wspace=0.24)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


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
    y = np.linspace(top - 0.5 * dy, bottom + 0.5 * dy, rows)

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
    vertical_exaggeration: float = 1.0,
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
) -> None:
    x_min, x_max = _finite_min_max(x_grid)
    y_min, y_max = _finite_min_max(y_grid)
    z_min, z_max = _finite_min_max(z_grid)

    ax.set_xticks([x_min, x_max])
    ax.set_yticks([y_min, y_max])
    ax.set_zticks([z_min, z_max])
    ax.set_xticklabels([_format_tick(x_min), _format_tick(x_max)])
    ax.set_yticklabels([_format_tick(y_min), _format_tick(y_max)])
    ax.set_zticklabels([_format_tick(z_min), _format_tick(z_max)])


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


def _finite_span(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    span = float(np.nanmax(finite) - np.nanmin(finite))
    return span if span > 0.0 else 1.0
