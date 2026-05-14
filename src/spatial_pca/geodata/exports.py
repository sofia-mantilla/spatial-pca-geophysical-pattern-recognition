"""Geospatial export helpers for top-ranked windows and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Point, Polygon


def build_window_polygons(
    window_indices: np.ndarray,
    window_shape: tuple[int, int],
    transform: Affine,
) -> list[Polygon]:
    """Build window polygons from top-left pixel indices and a raster transform."""

    rows, cols = _extract_row_col_indices(window_indices)
    win_h, win_w = _validate_window_shape(window_shape)

    polygons: list[Polygon] = []
    for row, col in zip(rows, cols):
        top_left = transform * (col, row)
        top_right = transform * (col + win_w, row)
        bottom_right = transform * (col + win_w, row + win_h)
        bottom_left = transform * (col, row + win_h)
        polygons.append(Polygon([top_left, top_right, bottom_right, bottom_left, top_left]))

    return polygons


def build_top_windows_gdf(
    window_indices: np.ndarray,
    window_shape: tuple[int, int],
    transform: Affine,
    crs: Any,
    ranks: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    window_ids: np.ndarray | None = None,
    extra_columns: dict[str, Any] | None = None,
    layer_name: str = "top_windows",
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame for top-ranked windows with polygon geometries."""

    window_indices_arr = np.asarray(window_indices, dtype=int)
    if window_indices_arr.ndim != 2 or window_indices_arr.shape[1] < 2:
        raise ValueError(
            "window_indices must be a 2D array with at least two columns: row and col."
        )

    rows, cols = _extract_row_col_indices(window_indices_arr)
    polygons = build_window_polygons(window_indices_arr, window_shape, transform)

    data: dict[str, Any] = {
        "row": rows,
        "col": cols,
        "geometry": polygons,
    }

    if window_ids is not None:
        data["window_id"] = np.asarray(window_ids, dtype=int)
    elif window_indices_arr.shape[1] >= 3:
        data["window_id"] = window_indices_arr[:, 2].astype(int)

    if ranks is not None:
        data["rank"] = np.asarray(ranks, dtype=int)
    if scores is not None:
        data["score"] = np.asarray(scores, dtype=float)

    if extra_columns is not None:
        for name, values in extra_columns.items():
            data[name] = np.asarray(values)

    gdf = gpd.GeoDataFrame(data, geometry="geometry", crs=crs)
    gdf = gdf.reset_index(drop=True)
    if layer_name != "top_windows":
        gdf.attrs["layer_name"] = layer_name

    return gdf


def save_geopackage(
    gdf: gpd.GeoDataFrame,
    output_path: Path | str,
    layer_name: str = "top_windows",
    engine: str | None = "pyogrio",
) -> Path:
    """Save a GeoDataFrame to a GeoPackage."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if gdf.empty:
        raise ValueError(f"Cannot write empty GeoPackage layer: {layer_name}")

    kwargs: dict[str, Any] = {"layer": layer_name, "driver": "GPKG"}
    if engine is not None:
        kwargs["engine"] = engine

    gdf.to_file(output_path, **kwargs)
    return output_path


def save_shapefile(
    gdf: gpd.GeoDataFrame,
    output_path: Path | str,
    engine: str | None = "pyogrio",
) -> Path:
    """Save a GeoDataFrame to a shapefile.

    Note
    ----
    Shapefile field names are limited to 10 characters and some data types may be
    truncated or converted. Prefer GeoPackage for richer exports.
    """

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".shp":
        raise ValueError("output_path must end with .shp for shapefile export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.empty:
        raise ValueError(f"Cannot write empty shapefile: {output_path}")

    kwargs: dict[str, Any] = {"driver": "ESRI Shapefile"}
    if engine is not None:
        kwargs["engine"] = engine

    gdf.to_file(output_path, **kwargs)
    return output_path


def build_circle_top_windows_gdf(
    *,
    window_indices: np.ndarray,
    window_shape: tuple[int, int],
    transform: Affine,
    radius_m: float,
    crs: Any,
    ranks: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    window_ids: np.ndarray | None = None,
    extra_columns: dict[str, Any] | None = None,
    layer_name: str = "top_windows",
) -> gpd.GeoDataFrame:
    """Create a polygon GeoDataFrame for circle patch predictions."""

    window_indices_arr = np.asarray(window_indices, dtype=int)
    rows, cols = _extract_row_col_indices(window_indices_arr)
    center_rows, center_cols = _extract_center_row_col_indices(window_indices_arr, window_shape)
    center_x, center_y = _pixel_centers_to_xy(center_rows, center_cols, transform)
    polygons = [Point(float(x), float(y)).buffer(float(radius_m)) for x, y in zip(center_x, center_y)]

    data: dict[str, Any] = {
        "row": rows,
        "col": cols,
        "center_row": center_rows,
        "center_col": center_cols,
        "center_x": center_x,
        "center_y": center_y,
        "radius_m": np.full(rows.shape[0], float(radius_m)),
        "patch_shape": np.full(rows.shape[0], "circle"),
    }
    if window_ids is not None:
        data["window_id"] = np.asarray(window_ids, dtype=int)
    elif window_indices_arr.shape[1] >= 3:
        data["window_id"] = window_indices_arr[:, 2].astype(int)
    if ranks is not None:
        data["rank"] = np.asarray(ranks, dtype=int)
    if scores is not None:
        data["score"] = np.asarray(scores, dtype=float)
    if extra_columns is not None:
        for name, values in extra_columns.items():
            data[name] = np.asarray(values)
    data["geometry"] = polygons

    gdf = gpd.GeoDataFrame(data, geometry="geometry", crs=crs).reset_index(drop=True)
    if layer_name != "top_windows":
        gdf.attrs["layer_name"] = layer_name
    return gdf


def build_circle_top_window_centers_gdf(
    polygon_gdf: gpd.GeoDataFrame,
    *,
    layer_name: str = "top_window_centers",
) -> gpd.GeoDataFrame:
    """Create a point GeoDataFrame from a circle-polygon export GeoDataFrame."""

    centers = polygon_gdf.drop(columns="geometry").copy()
    centers["geometry"] = [Point(float(x), float(y)) for x, y in zip(centers["center_x"], centers["center_y"])]
    centers_gdf = gpd.GeoDataFrame(centers, geometry="geometry", crs=polygon_gdf.crs).reset_index(drop=True)
    if layer_name != "top_window_centers":
        centers_gdf.attrs["layer_name"] = layer_name
    return centers_gdf


def _extract_row_col_indices(window_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if window_indices.ndim != 2 or window_indices.shape[1] < 2:
        raise ValueError(
            "window_indices must be a 2D array with at least two columns: row and col."
        )
    return window_indices[:, 0].astype(int), window_indices[:, 1].astype(int)


def _extract_center_row_col_indices(
    window_indices: np.ndarray,
    window_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if window_indices.shape[1] >= 5:
        return window_indices[:, 3].astype(int), window_indices[:, 4].astype(int)
    win_h, win_w = _validate_window_shape(window_shape)
    return (
        window_indices[:, 0].astype(int) + (win_h // 2),
        window_indices[:, 1].astype(int) + (win_w // 2),
    )


def _pixel_centers_to_xy(
    center_rows: np.ndarray,
    center_cols: np.ndarray,
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray]:
    col_vals = center_cols.astype(float) + 0.5
    row_vals = center_rows.astype(float) + 0.5
    x = transform.c + col_vals * transform.a + row_vals * transform.b
    y = transform.f + col_vals * transform.d + row_vals * transform.e
    return x.astype(float), y.astype(float)


def _validate_window_shape(window_shape: tuple[int, int]) -> tuple[int, int]:
    if len(window_shape) != 2:
        raise ValueError("window_shape must be a two-item tuple: (height, width).")
    win_h = int(window_shape[0])
    win_w = int(window_shape[1])
    if win_h <= 0 or win_w <= 0:
        raise ValueError("window_shape dimensions must be positive integers.")
    return win_h, win_w
