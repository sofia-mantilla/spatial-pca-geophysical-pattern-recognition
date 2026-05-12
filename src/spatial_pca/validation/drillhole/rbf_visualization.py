"""Diagnostic plots for RBF mineralization model outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_grid_training_html(
    *,
    grid: pd.DataFrame,
    assays: pd.DataFrame,
    output_path: Path,
    max_grid_points: int = 30000,
    random_seed: int = 42,
) -> Path:
    """Write a rotatable HTML showing prediction grid support and training points."""

    grid_sample = _sample_frame(grid, max_rows=max_grid_points, random_seed=random_seed)
    assay_positive = assays["mineralized_state"].eq(1)
    data = [
        _scatter3d_trace(
            grid_sample,
            x_col="X",
            y_col="Y",
            z_col="depth_m",
            name=f"Prediction grid sample ({len(grid_sample):,} of {len(grid):,})",
            color="#c9c9c9",
            size=2,
            opacity=0.25,
        ),
        _scatter3d_trace(
            assays.loc[~assay_positive],
            x_col="X",
            y_col="Y",
            z_col="depth_mid_m",
            name=f"10 m averaged assays below threshold ({int((~assay_positive).sum()):,})",
            color="#4c4c4c",
            size=3,
            opacity=0.65,
        ),
        _scatter3d_trace(
            assays.loc[assay_positive],
            x_col="X",
            y_col="Y",
            z_col="depth_mid_m",
            name=f"10 m averaged mineralized assays ({int(assay_positive.sum()):,})",
            color="#f46d43",
            size=3,
            opacity=0.8,
        ),
    ]
    layout = _plotly_layout("Step 3: 3D RBF prediction grid and 10 m averaged assay classes")
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def write_predicted_grade_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    value_col: str,
    title: str,
    colorbar_title: str,
    max_points: int = 60000,
    random_seed: int = 42,
) -> Path:
    """Write a rotatable HTML for continuous 3D predicted grade values."""

    sample = _sample_frame(grid, max_rows=max_points, random_seed=random_seed)
    values = pd.to_numeric(sample[value_col], errors="coerce")
    data = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": _json_float_list(sample["X"]),
            "y": _json_float_list(sample["Y"]),
            "z": _json_float_list(sample["depth_m"]),
            "marker": {
                "size": 2,
                "opacity": 0.65,
                "color": _json_float_list(values),
                "colorscale": "Viridis",
                "colorbar": {"title": colorbar_title},
            },
            "name": value_col,
        }
    ]
    return _write_plotly_html(data=data, layout=_plotly_layout(title), output_path=output_path)


def write_predicted_mineralized_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    assays: pd.DataFrame | None = None,
    show_unsupported: bool = True,
    max_points: int = 60000,
    random_seed: int = 42,
) -> Path:
    """Write a rotatable HTML for thresholded modeled mineralization."""

    sample = _sample_frame(grid, max_rows=max_points, random_seed=random_seed)
    supported = sample["drill_supported"].astype(bool) if "drill_supported" in sample.columns else pd.Series(True, index=sample.index)
    positive = sample["modeled_mineralized"].astype(bool)
    data = []
    if show_unsupported:
        data.append(
            _scatter3d_trace(
                sample.loc[~supported],
                x_col="X",
                y_col="Y",
                z_col="depth_m",
                name="Unsupported by nearby drilling",
                color="#d7d7d7",
                size=2,
                opacity=0.14,
                include_support_hover=True,
            )
        )
    data.extend(
        [
            _scatter3d_trace(
            sample.loc[supported & ~positive],
            x_col="X",
            y_col="Y",
            z_col="depth_m",
            name="Supported, below Cu/Au threshold",
            color="#4c4c4c",
            size=2,
            opacity=0.22,
            include_support_hover=True,
        ),
        _scatter3d_trace(
            sample.loc[positive],
            x_col="X",
            y_col="Y",
            z_col="depth_m",
            name="Modeled mineralized",
            color="#f46d43",
            size=3,
            opacity=0.78,
            include_support_hover=True,
        ),
        ]
    )
    if assays is not None:
        assay_positive = assays["mineralized_state"].eq(1)
        data.extend(
            [
                _scatter3d_trace(
                    assays.loc[~assay_positive],
                    x_col="X",
                    y_col="Y",
                    z_col="depth_mid_m",
                    name="Training assay below threshold",
                    color="#111111",
                    size=3,
                    opacity=0.35,
                ),
                _scatter3d_trace(
                    assays.loc[assay_positive],
                    x_col="X",
                    y_col="Y",
                    z_col="depth_mid_m",
                    name="Training assay mineralized",
                    color="#00a878",
                    size=5,
                    opacity=0.95,
                ),
            ]
        )
    layout = _plotly_layout("Step 5: 3D RBF model after Cu/Au mineralization thresholds")
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def plot_depth_slice_summary(
    slice_layers: pd.DataFrame,
    *,
    output_path: Path,
    title: str = "Step 6: RBF-modeled mineralized thickness by depth slice",
) -> Path:
    """Plot three 2D depth-slice maps of modeled mineralized thickness."""

    required = ["X", "Y", "slice_id", "mineralized_thickness_m", "modeled_footprint"]
    missing = [col for col in required if col not in slice_layers.columns]
    if missing:
        raise ValueError(f"Missing required depth-slice plot columns: {missing}")

    slices = list(dict.fromkeys(slice_layers["slice_id"].astype(str).tolist()))
    if not slices:
        raise ValueError("No depth slices available to plot.")

    fig, axes = plt.subplots(1, len(slices), figsize=(6.2 * len(slices), 6.0), constrained_layout=True)
    if len(slices) == 1:
        axes = [axes]
    vmax = float(slice_layers["mineralized_thickness_m"].quantile(0.98))
    if vmax <= 0:
        vmax = float(slice_layers["mineralized_thickness_m"].max() or 1.0)

    last_artist = None
    for ax, slice_id in zip(axes, slices):
        frame = slice_layers.loc[slice_layers["slice_id"].astype(str).eq(slice_id)].copy()
        last_artist = ax.scatter(
            frame["X"],
            frame["Y"],
            c=frame["mineralized_thickness_m"],
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            s=8,
            linewidths=0,
        )
        footprint = frame.loc[frame["modeled_footprint"].astype(bool)]
        if not footprint.empty:
            ax.scatter(
                footprint["X"],
                footprint["Y"],
                facecolors="none",
                edgecolors="#33b5e5",
                s=16,
                linewidths=0.35,
                alpha=0.8,
            )
        ax.set_title(slice_id)
        ax.set_xlabel("Easting X (m)")
        ax.set_ylabel("Northing Y (m)")
        ax.set_aspect("equal", adjustable="box")

    if last_artist is not None:
        colorbar = fig.colorbar(last_artist, ax=axes, shrink=0.82)
        colorbar.set_label("Modeled mineralized thickness (m)")
    fig.suptitle(title, fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_rbf_volume_model_html(
    *,
    grid: pd.DataFrame,
    assays: pd.DataFrame,
    output_path: Path,
    mineralization_rule: dict,
    title: str = "3D RBF mineralization model",
    include_below_threshold_assays: bool = True,
) -> Path:
    """Write a rotatable isosurface-style RBF mineralization volume."""

    supported = grid.loc[grid["drill_supported"].astype(bool)].copy() if "drill_supported" in grid.columns else grid.copy()
    if supported.empty:
        raise ValueError("No drill-supported grid cells available for volume plotting.")

    score = _mineralization_score(supported, mineralization_rule=mineralization_rule)
    positive_assays = assays.loc[assays["mineralized_state"].eq(1)].copy()
    below_assays = assays.loc[~assays["mineralized_state"].eq(1)].copy()

    data: list[dict] = [
        {
            "type": "isosurface",
            "x": _json_float_list(supported["X"]),
            "y": _json_float_list(supported["Y"]),
            "z": _json_float_list(supported["depth_m"]),
            "value": _json_float_list(score),
            "isomin": 0.85,
            "isomax": max(1.6, float(np.nanpercentile(score, 99))),
            "surface": {"count": 5, "fill": 0.65},
            "caps": {"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
            "opacity": 0.34,
            "colorscale": [
                [0.0, "rgba(255,237,160,0.20)"],
                [0.35, "rgba(254,178,76,0.35)"],
                [0.7, "rgba(240,59,32,0.50)"],
                [1.0, "rgba(189,0,38,0.62)"],
            ],
            "colorbar": {"title": "Cu/Au threshold score"},
            "name": "RBF Cu/Au threshold score",
        },
        _scatter3d_trace(
            positive_assays,
            x_col="X",
            y_col="Y",
            z_col="depth_mid_m",
            name=f"Mineralized 10 m assays ({len(positive_assays):,})",
            color="#111111",
            size=3,
            opacity=0.75,
        ),
    ]
    if include_below_threshold_assays:
        data.append(
            _scatter3d_trace(
                below_assays,
                x_col="X",
                y_col="Y",
                z_col="depth_mid_m",
                name=f"Below-threshold 10 m assays ({len(below_assays):,})",
                color="#6b6b6b",
                size=2,
                opacity=0.18,
            )
        )

    layout = _plotly_layout(title)
    layout["scene"]["aspectmode"] = "data"
    layout["scene"]["zaxis"]["title"] = "Depth (m)"
    layout["legend"] = {"x": 0.01, "y": 0.98}
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def write_rbf_depth_slices_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    value_col: str = "Cu_pct_pred",
    depth_interval_m: float = 50.0,
    filter_drill_supported: bool = True,
    title: str | None = None,
    colorbar_title: str | None = None,
) -> Path:
    """Write 3D visualization of horizontal depth slices as 2D heatmaps.

    Creates stacked horizontal slices through the RBF model, visualizing
    each slice as a colored 2D surface at its depth.

    Parameters:
    -----------
    grid : pd.DataFrame
        RBF prediction grid with coordinates (X, Y, depth_m) and prediction columns.
    output_path : Path
        Output HTML file path.
    value_col : str
        Column name to color by. Default: Cu_pct_pred.
    depth_interval_m : float
        Depth spacing for slices in meters. Default: 50.0.
    filter_drill_supported : bool
        If True, only use drill-supported cells. Default: True.
    title : str, optional
        Plot title. If None, generates default.
    colorbar_title : str, optional
        Colorbar label. If None, uses value_col name.

    Returns:
    --------
    Path
        Output path.
    """

    if value_col not in grid.columns:
        raise ValueError(f"Column '{value_col}' not found in grid. Available: {list(grid.columns)}")

    frame = grid.copy()
    if filter_drill_supported and "drill_supported" in frame.columns:
        frame = frame.loc[frame["drill_supported"].astype(bool)].copy()

    if frame.empty:
        raise ValueError("No grid cells available for depth slice plotting.")

    # Get depth boundaries
    min_depth = float(frame["depth_m"].min())
    max_depth = float(frame["depth_m"].max())
    depths = np.arange(min_depth, max_depth + depth_interval_m, depth_interval_m)

    # Create slices
    data: list[dict] = []

    for i, depth in enumerate(depths[:-1]):
        depth_next = depths[i + 1]
        # Get cells in this depth slice
        slice_data = frame.loc[
            (frame["depth_m"] >= depth) & (frame["depth_m"] < depth_next)
        ].copy()

        if slice_data.empty:
            continue

        values = pd.to_numeric(slice_data[value_col], errors="coerce")

        # Create a scatter surface effect by plotting points
        data.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": _json_float_list(slice_data["X"]),
                "y": _json_float_list(slice_data["Y"]),
                "z": [depth_next] * len(slice_data),  # Fixed depth for this slice
                "marker": {
                    "size": 5,
                    "opacity": 0.85,
                    "color": _json_float_list(values),
                    "colorscale": "Viridis",
                    "colorbar": {"title": colorbar_title or value_col} if i == 0 else None,
                },
                "name": f"Depth {depth:.0f}-{depth_next:.0f} m",
            }
        )

    if not data:
        raise ValueError("No depth slices generated from grid data.")

    if title is None:
        title = f"3D RBF horizontal depth slices ({value_col})"
        if filter_drill_supported:
            title += " (drill-supported)"

    layout = _plotly_layout(title)
    layout["scene"]["aspectmode"] = "data"
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def write_rbf_depth_slice_maps_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    value_col: str = "Cu_pct_pred",
    depth_list: list[float] | None = None,
    depth_interval_m: float | None = None,
    depth_window_m: float = 25.0,
    filter_drill_supported: bool = True,
    colorbar_title: str | None = None,
) -> Path:
    """Write 2D grid maps for specific depth slices in an interactive 3D plot.

    Creates a 3D visualization where each depth slice is rendered as a colored
    surface/mesh layer.

    Parameters:
    -----------
    grid : pd.DataFrame
        RBF prediction grid with coordinates (X, Y, depth_m) and prediction columns.
    output_path : Path
        Output HTML file path.
    value_col : str
        Column name to color by. Default: Cu_pct_pred.
    depth_list : list[float], optional
        Specific depths to slice at. If None, uses depth_interval_m.
    depth_interval_m : float, optional
        Depth spacing for slices. If None, auto-computed from data range.
    depth_window_m : float
        Half-width around target depth to average. Default: 25.0 m.
    filter_drill_supported : bool
        If True, only use drill-supported cells. Default: True.
    colorbar_title : str, optional
        Colorbar label. If None, uses value_col name.

    Returns:
    --------
    Path
        Output path.
    """

    if value_col not in grid.columns:
        raise ValueError(f"Column '{value_col}' not found in grid. Available: {list(grid.columns)}")

    frame = grid.copy()
    if filter_drill_supported and "drill_supported" in frame.columns:
        frame = frame.loc[frame["drill_supported"].astype(bool)].copy()

    if frame.empty:
        raise ValueError("No grid cells available for depth slice plotting.")

    # Determine depths to slice
    if depth_list is None:
        if depth_interval_m is None:
            min_depth = float(frame["depth_m"].min())
            max_depth = float(frame["depth_m"].max())
            depth_interval_m = (max_depth - min_depth) / 5  # 5 slices by default
        else:
            depth_interval_m = float(depth_interval_m)

        min_depth = float(frame["depth_m"].min())
        max_depth = float(frame["depth_m"].max())
        depth_list = list(np.arange(min_depth, max_depth + depth_interval_m, depth_interval_m))

    # Create slices
    data: list[dict] = []
    values_global_min = float('inf')
    values_global_max = float('-inf')

    # First pass: get global min/max for consistent colorscale
    for depth in depth_list:
        slice_data = frame.loc[
            (frame["depth_m"] >= depth - depth_window_m)
            & (frame["depth_m"] <= depth + depth_window_m)
        ].copy()
        if not slice_data.empty:
            values = pd.to_numeric(slice_data[value_col], errors="coerce")
            fin_vals = values[np.isfinite(values)]
            if len(fin_vals) > 0:
                values_global_min = min(values_global_min, float(fin_vals.min()))
                values_global_max = max(values_global_max, float(fin_vals.max()))

    if values_global_min == float('inf'):
        raise ValueError("No valid values found in depth slices.")

    # Second pass: create traces
    for i, depth in enumerate(depth_list):
        slice_data = frame.loc[
            (frame["depth_m"] >= depth - depth_window_m)
            & (frame["depth_m"] <= depth + depth_window_m)
        ].copy()

        if slice_data.empty:
            continue

        values = pd.to_numeric(slice_data[value_col], errors="coerce")

        data.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": _json_float_list(slice_data["X"]),
                "y": _json_float_list(slice_data["Y"]),
                "z": [depth] * len(slice_data),
                "marker": {
                    "size": 6,
                    "opacity": 0.80,
                    "color": _json_float_list(values),
                    "colorscale": "Viridis",
                    "cmin": values_global_min,
                    "cmax": values_global_max,
                    "colorbar": {
                        "title": colorbar_title or value_col,
                    } if i == 0 else None,
                },
                "name": f"{depth:.0f} m depth",
            }
        )

    if not data:
        raise ValueError("No depth slices generated from grid data.")

    title = f"3D RBF depth slices ({value_col})"
    layout = _plotly_layout(title)
    layout["scene"]["aspectmode"] = "data"
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)
def write_rbf_isosurface_volume_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    value_col: str = "Cu_pct_pred",
    threshold: float | None = None,
    filter_drill_supported: bool = True,
    title: str | None = None,
    colorbar_title: str | None = None,
    isomin: float | None = None,
    isomax: float | None = None,
    surface_count: int = 5,
    surface_fill: float = 0.65,
    opacity: float = 0.45,
) -> Path:
    """Write a 3D isosurface volume visualization without wells.

    Creates a smooth, translucent 3D volume like a mineralized body blob.
    Perfect for visualizing continuous grade distributions in 3D.

    Parameters:
    -----------
    grid : pd.DataFrame
        RBF prediction grid with coordinates (X, Y, depth_m) and prediction columns.
    output_path : Path
        Output HTML file path.
    value_col : str
        Column name to render as isosurface. Default: Cu_pct_pred.
    threshold : float, optional
        Optional minimum threshold value. Values below are masked out.
    filter_drill_supported : bool
        If True, only use drill-supported cells. Default: True.
    title : str, optional
        Plot title. If None, generates default.
    colorbar_title : str, optional
        Colorbar label. If None, uses value_col name.
    isomin : float, optional
        Minimum isosurface value. If None, computed from data percentiles.
    isomax : float, optional
        Maximum isosurface value. If None, computed from data percentiles.
    surface_count : int
        Number of isosurfaces to render. Default: 5.
    surface_fill : float
        Isosurface fill factor (0-1). Default: 0.65.
    opacity : float
        Overall volume opacity (0-1). Default: 0.45.

    Returns:
    --------
    Path
        Output path.
    """

    if value_col not in grid.columns:
        raise ValueError(f"Column '{value_col}' not found in grid. Available: {list(grid.columns)}")

    frame = grid.copy()
    if filter_drill_supported and "drill_supported" in frame.columns:
        frame = frame.loc[frame["drill_supported"].astype(bool)].copy()

    if frame.empty:
        raise ValueError("No grid cells available for volume plotting.")

    # Apply threshold if provided
    if threshold is not None:
        values_numeric = pd.to_numeric(frame[value_col], errors="coerce")
        frame = frame.loc[values_numeric >= threshold].copy()
        if frame.empty:
            raise ValueError(f"No grid cells with {value_col} >= {threshold} available.")

    values = pd.to_numeric(frame[value_col], errors="coerce")

    # Auto-compute iso limits if not provided
    if isomin is None or isomax is None:
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size:
            # Use non-zero values for better percentile estimation
            nonzero_vals = finite_vals[finite_vals > 1e-6]
            if len(nonzero_vals) > 10:
                # Use percentiles from non-zero data
                p10 = float(np.percentile(nonzero_vals, 10))
                p90 = float(np.percentile(nonzero_vals, 90))
                if isomin is None:
                    isomin = p10 * 0.5  # Start below the 10th percentile
                if isomax is None:
                    isomax = p90 * 1.2  # Extend above the 90th percentile
            else:
                # Fallback if not enough non-zero values
                if isomin is None:
                    isomin = float(np.nanmin(finite_vals)) * 0.9
                if isomax is None:
                    isomax = float(np.nanmax(finite_vals)) * 1.1
        else:
            isomin = isomin or 0.0
            isomax = isomax or 1.0

    # Set titles
    if title is None:
        title = f"3D RBF {value_col} volume"
        if threshold is not None:
            title += f" ({value_col} ≥ {threshold:g})"
        if filter_drill_supported:
            title += " (drill-supported)"
    if colorbar_title is None:
        colorbar_title = value_col

    # Create isosurface visualization
    data: list[dict] = [
        {
            "type": "isosurface",
            "x": _json_float_list(frame["X"]),
            "y": _json_float_list(frame["Y"]),
            "z": _json_float_list(frame["depth_m"]),
            "value": _json_float_list(values),
            "isomin": isomin,
            "isomax": isomax,
            "surface": {"count": surface_count, "fill": surface_fill},
            "caps": {"x": {"show": False}, "y": {"show": False}, "z": {"show": False}},
            "opacity": opacity,
            "colorscale": [
                [0.0, "rgba(255,237,160,0.15)"],
                [0.2, "rgba(254,208,62,0.30)"],
                [0.5, "rgba(254,178,76,0.45)"],
                [0.75, "rgba(240,59,32,0.60)"],
                [1.0, "rgba(189,0,38,0.75)"],
            ],
            "colorbar": {"title": colorbar_title},
            "name": value_col,
        }
    ]

    layout = _plotly_layout(title)
    layout["scene"]["aspectmode"] = "data"
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def write_rbf_voxel_volume_html(
    *,
    grid: pd.DataFrame,
    assays: pd.DataFrame,
    output_path: Path,
    mineralization_rule: dict,
    title: str = "3D RBF mineralization model - supported mineralized volume",
    include_assays: bool = True,
    voxel_size: int = 12,
    voxel_opacity: float = 0.62,
) -> Path:
    """Write a blob-like 3D volume using translucent supported mineralized voxels."""

    supported = grid.loc[grid["drill_supported"].astype(bool)].copy() if "drill_supported" in grid.columns else grid.copy()
    mineralized = supported.loc[supported["modeled_mineralized"].astype(bool)].copy()
    if mineralized.empty:
        raise ValueError("No supported modeled-mineralized grid cells available for voxel plotting.")

    score = _mineralization_score(mineralized, mineralization_rule=mineralization_rule)
    data: list[dict] = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": _json_float_list(mineralized["X"]),
            "y": _json_float_list(mineralized["Y"]),
            "z": _json_float_list(mineralized["depth_m"]),
            "marker": {
                "symbol": "square",
                "size": voxel_size,
                "opacity": voxel_opacity,
                "color": _json_float_list(score),
                "colorscale": [
                    [0.0, "rgba(255,237,160,0.30)"],
                    [0.35, "rgba(254,178,76,0.48)"],
                    [0.7, "rgba(240,59,32,0.66)"],
                    [1.0, "rgba(189,0,38,0.78)"],
                ],
                "colorbar": {"title": "Cu/Au threshold score"},
            },
            "name": f"Supported modeled mineralized cells ({len(mineralized):,})",
        }
    ]
    if include_assays:
        positive_assays = assays.loc[assays["mineralized_state"].eq(1)].copy()
        data.append(
            _scatter3d_trace(
                positive_assays,
                x_col="X",
                y_col="Y",
                z_col="depth_mid_m",
                name=f"Mineralized 10 m assays ({len(positive_assays):,})",
                color="#111111",
                size=3,
                opacity=0.8,
            )
        )

    layout = _plotly_layout(title)
    layout["scene"]["aspectmode"] = "data"
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def write_rbf_heat_map_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    value_col: str = "Cu_pct_pred",
    filter_drill_supported: bool = True,
    title: str | None = None,
    colorbar_title: str | None = None,
    max_points: int = 60000,
    random_seed: int = 42,
    marker_size: int = 4,
    marker_opacity: float = 0.75,
) -> Path:
    """Write a 3D heat map of RBF predictions without assay well locations.

    Creates a Plotly 3D scatter plot colored by a continuous RBF prediction value.
    Useful for visualizing the modeled mineralization grade distribution in 3D space.

    Parameters:
    -----------
    grid : pd.DataFrame
        RBF prediction grid with coordinates (X, Y, depth_m) and prediction columns.
    output_path : Path
        Output HTML file path.
    value_col : str
        Column name to color by. Options: Cu_pct_pred, Au_ppm_pred, log1p_Cu_pct_pred,
        log1p_Au_ppm_pred, or custom score column. Default: Cu_pct_pred.
    filter_drill_supported : bool
        If True, only plot drill-supported cells. Default: True.
    title : str, optional
        Plot title. If None, generates default based on value_col.
    colorbar_title : str, optional
        Colorbar label. If None, uses value_col name.
    max_points : int
        Maximum grid points to plot for performance. 0 = plot all. Default: 60000.
    random_seed : int
        Random seed for sampling if max_points > 0. Default: 42.
    marker_size : int
        Plotly marker size. Default: 4.
    marker_opacity : float
        Marker opacity (0-1). Default: 0.75.

    Returns:
    --------
    Path
        Output path.
    """

    if value_col not in grid.columns:
        raise ValueError(f"Column '{value_col}' not found in grid. Available: {list(grid.columns)}")

    frame = grid.copy()
    if filter_drill_supported and "drill_supported" in frame.columns:
        frame = frame.loc[frame["drill_supported"].astype(bool)].copy()
        if frame.empty:
            raise ValueError("No drill-supported grid cells available for heat map plotting.")

    frame = _sample_frame(frame, max_rows=max_points, random_seed=random_seed)
    values = pd.to_numeric(frame[value_col], errors="coerce")

    # Set titles if not provided
    if title is None:
        title = f"3D RBF {value_col} heat map"
        if filter_drill_supported:
            title += " (drill-supported)"
    if colorbar_title is None:
        colorbar_title = value_col

    # Create 3D scatter plot colored by values
    data = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": _json_float_list(frame["X"]),
            "y": _json_float_list(frame["Y"]),
            "z": _json_float_list(frame["depth_m"]),
            "marker": {
                "size": marker_size,
                "opacity": marker_opacity,
                "color": _json_float_list(values),
                "colorscale": "Viridis",
                "colorbar": {"title": colorbar_title},
            },
            "name": value_col,
        }
    ]

    layout = _plotly_layout(title)
    layout["scene"]["aspectmode"] = "data"
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def write_rbf_masked_heat_map_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    value_col: str = "Cu_pct_pred",
    threshold: float = 0.3,
    filter_drill_supported: bool = True,
    title: str | None = None,
    colorbar_title: str | None = None,
    max_points: int = 60000,
    random_seed: int = 42,
    marker_size: int = 5,
    marker_opacity: float = 0.8,
) -> Path:
    """Write a 3D heat map with threshold masking applied.

    Only grid cells meeting the threshold are plotted. Useful for visualizing
    zones of significant predicted mineralization.

    Parameters:
    -----------
    grid : pd.DataFrame
        RBF prediction grid with coordinates (X, Y, depth_m) and prediction columns.
    output_path : Path
        Output HTML file path.
    value_col : str
        Column name to threshold and color by. Default: Cu_pct_pred.
    threshold : float
        Minimum value threshold. Only cells >= threshold are plotted. Default: 0.3.
    filter_drill_supported : bool
        If True, only plot drill-supported cells. Default: True.
    title : str, optional
        Plot title. If None, generates default based on threshold and value_col.
    colorbar_title : str, optional
        Colorbar label. If None, uses value_col name.
    max_points : int
        Maximum grid points to plot for performance. 0 = plot all. Default: 60000.
    random_seed : int
        Random seed for sampling if max_points > 0. Default: 42.
    marker_size : int
        Plotly marker size. Default: 5.
    marker_opacity : float
        Marker opacity (0-1). Default: 0.8.

    Returns:
    --------
    Path
        Output path.
    """

    if value_col not in grid.columns:
        raise ValueError(f"Column '{value_col}' not found in grid. Available: {list(grid.columns)}")

    frame = grid.copy()
    if filter_drill_supported and "drill_supported" in frame.columns:
        frame = frame.loc[frame["drill_supported"].astype(bool)].copy()

    # Apply threshold mask
    values_numeric = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.loc[values_numeric >= threshold].copy()

    if frame.empty:
        raise ValueError(f"No grid cells with {value_col} >= {threshold} available for plotting.")

    frame = _sample_frame(frame, max_rows=max_points, random_seed=random_seed)
    values = pd.to_numeric(frame[value_col], errors="coerce")

    # Set titles if not provided
    if title is None:
        title = f"3D RBF {value_col} heat map ({value_col} ≥ {threshold:g})"
        if filter_drill_supported:
            title += " (drill-supported)"
    if colorbar_title is None:
        colorbar_title = value_col

    # Create 3D scatter plot colored by values
    data = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": _json_float_list(frame["X"]),
            "y": _json_float_list(frame["Y"]),
            "z": _json_float_list(frame["depth_m"]),
            "marker": {
                "size": marker_size,
                "opacity": marker_opacity,
                "color": _json_float_list(values),
                "colorscale": "Viridis",
                "colorbar": {"title": colorbar_title},
            },
            "name": f"{value_col} ≥ {threshold:g}",
        }
    ]

    layout = _plotly_layout(title)
    layout["scene"]["aspectmode"] = "data"
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def write_rbf_solid_blob_html(
    *,
    grid: pd.DataFrame,
    output_path: Path,
    title: str = "3D RBF mineralized body",
    marker_size: int = 16,
    color: str = "#e4572e",
) -> Path:
    """Write a local, solid-color view of modeled mineralized cells only."""

    mineralized = grid.loc[grid["modeled_mineralized"].astype(bool)].copy()
    if mineralized.empty:
        raise ValueError("No modeled-mineralized grid cells available for solid blob plotting.")

    x0 = float(mineralized["X"].mean())
    y0 = float(mineralized["Y"].mean())
    d0 = float(mineralized["depth_m"].mean())
    local = pd.DataFrame(
        {
            "x_offset_m": mineralized["X"].astype(float) - x0,
            "y_offset_m": mineralized["Y"].astype(float) - y0,
            "depth_offset_m": mineralized["depth_m"].astype(float) - d0,
        }
    )
    data = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": _json_float_list(local["x_offset_m"]),
            "y": _json_float_list(local["y_offset_m"]),
            "z": _json_float_list(local["depth_offset_m"]),
            "marker": {
                "symbol": "circle",
                "size": marker_size,
                "opacity": 1.0,
                "color": color,
            },
            "name": f"Modeled mineralized cells ({len(mineralized):,})",
        }
    ]
    layout = {
        "title": title,
        "scene": {
            "xaxis": {"title": f"X offset from {x0:.0f} m"},
            "yaxis": {"title": f"Y offset from {y0:.0f} m"},
            "zaxis": {"title": f"Depth offset from {d0:.0f} m", "autorange": "reversed"},
            "aspectmode": "cube",
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 60},
        "showlegend": True,
    }
    return _write_plotly_html(data=data, layout=layout, output_path=output_path)


def _mineralization_score(grid: pd.DataFrame, *, mineralization_rule: dict) -> pd.Series:
    cu_threshold = float(mineralization_rule.get("cu_threshold_pct", 0.1))
    au_threshold = float(mineralization_rule.get("au_threshold_ppm", 0.1))
    joint_cu = float(mineralization_rule.get("joint_cu_threshold_pct", 0.08))
    joint_au = float(mineralization_rule.get("joint_au_threshold_ppm", 0.08))
    cu_score = pd.to_numeric(grid["Cu_pct_pred"], errors="coerce") / cu_threshold
    au_score = pd.to_numeric(grid["Au_ppm_pred"], errors="coerce") / au_threshold
    joint_score = np.minimum(
        pd.to_numeric(grid["Cu_pct_pred"], errors="coerce") / joint_cu,
        pd.to_numeric(grid["Au_ppm_pred"], errors="coerce") / joint_au,
    )
    return pd.concat([cu_score, au_score, pd.Series(joint_score, index=grid.index)], axis=1).max(axis=1)


def _sample_frame(frame: pd.DataFrame, *, max_rows: int, random_seed: int) -> pd.DataFrame:
    if max_rows > 0 and len(frame) > max_rows:
        return frame.sample(n=max_rows, random_state=random_seed).copy()
    return frame.copy()


def _scatter3d_trace(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    name: str,
    color: str,
    size: int,
    opacity: float,
    include_support_hover: bool = False,
) -> dict:
    trace = {
        "type": "scatter3d",
        "mode": "markers",
        "x": _json_float_list(frame[x_col]),
        "y": _json_float_list(frame[y_col]),
        "z": _json_float_list(frame[z_col]),
        "marker": {"size": size, "opacity": opacity, "color": color},
        "name": name,
    }
    if include_support_hover and {"nearest_assay_xy_distance_m", "nearest_assay_3d_distance_m"}.issubset(frame.columns):
        trace["customdata"] = [
            [xy_dist, xyz_dist]
            for xy_dist, xyz_dist in zip(
                _json_float_list(frame["nearest_assay_xy_distance_m"]),
                _json_float_list(frame["nearest_assay_3d_distance_m"]),
            )
        ]
        trace["hovertemplate"] = (
            "x: %{x:.0f}<br>"
            "y: %{y:.0f}<br>"
            "depth: %{z:.0f} m<br>"
            "nearest assay XY: %{customdata[0]:.0f} m<br>"
            "nearest assay 3D: %{customdata[1]:.0f} m"
            "<extra>%{fullData.name}</extra>"
        )
    return trace


def _plotly_layout(title: str) -> dict:
    return {
        "title": title,
        "scene": {
            "xaxis": {"title": "Easting X (m)"},
            "yaxis": {"title": "Northing Y (m)"},
            "zaxis": {"title": "Depth (m)", "autorange": "reversed"},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 60},
        "legend": {"x": 0.01, "y": 0.98},
    }


def _json_float_list(values: pd.Series) -> list[float | None]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return [None if not np.isfinite(value) else float(value) for value in array]


def _write_plotly_html(*, data: list[dict], layout: dict, output_path: Path) -> Path:
    data_json = json.dumps(data)
    layout_json = json.dumps(layout)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spatial PCA RBF validation diagnostic</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    html, body, #plot {{
      width: 100%;
      height: 100%;
      margin: 0;
    }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <script>
    const data = {data_json};
    const layout = {layout_json};
    Plotly.newPlot("plot", data, layout, {{responsive: true}});
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
