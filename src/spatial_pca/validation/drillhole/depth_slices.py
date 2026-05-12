"""Depth-slice summaries for 3D RBF mineralization grids."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_depth_slice_layers(
    grid: pd.DataFrame,
    *,
    slice_edges_m: list[float],
    depth_spacing_m: float,
) -> pd.DataFrame:
    """Summarize a 3D RBF grid into 2D x/y validation layers by depth slice."""

    if len(slice_edges_m) < 2:
        raise ValueError("slice_edges_m must contain at least two values.")
    if sorted(slice_edges_m) != list(slice_edges_m):
        raise ValueError("slice_edges_m must be sorted in ascending order.")
    if depth_spacing_m <= 0:
        raise ValueError("depth_spacing_m must be positive.")

    required = ["X", "Y", "depth_m", "Cu_pct_pred", "Au_ppm_pred", "modeled_mineralized"]
    missing = [col for col in required if col not in grid.columns]
    if missing:
        raise ValueError(f"Missing required RBF grid columns: {missing}")

    layers: list[pd.DataFrame] = []
    for zmin, zmax in zip(slice_edges_m[:-1], slice_edges_m[1:]):
        slice_id = format_depth_slice_id(float(zmin), float(zmax))
        subset = grid.loc[(grid["depth_m"] >= zmin) & (grid["depth_m"] < zmax)].copy()
        if subset.empty:
            continue

        subset["mineralized_thickness_component_m"] = np.where(
            subset["modeled_mineralized"].astype(bool),
            float(depth_spacing_m),
            0.0,
        )
        grouped = (
            subset.groupby(["X", "Y"], as_index=False)
            .agg(
                modeled_footprint=("modeled_mineralized", "max"),
                supported_cell_count=("drill_supported", "sum") if "drill_supported" in subset.columns else ("depth_m", "size"),
                mineralized_thickness_m=("mineralized_thickness_component_m", "sum"),
                max_Cu_pct=("Cu_pct_pred", "max"),
                max_Au_ppm=("Au_ppm_pred", "max"),
                cell_count=("depth_m", "size"),
            )
        )
        grouped["slice_id"] = slice_id
        grouped["slice_from_m"] = float(zmin)
        grouped["slice_to_m"] = float(zmax)
        grouped["modeled_footprint"] = grouped["modeled_footprint"].astype(bool)
        layers.append(grouped)

    if not layers:
        return pd.DataFrame()
    return pd.concat(layers, ignore_index=True)


def format_depth_slice_id(slice_from_m: float, slice_to_m: float) -> str:
    """Return stable depth-slice IDs such as ``0_200m``."""

    left = _format_depth_value(slice_from_m)
    right = _format_depth_value(slice_to_m)
    return f"{left}_{right}m"


def _format_depth_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")
