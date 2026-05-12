"""Deposit-vector loading and template extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from .rasters import RasterData


@dataclass(frozen=True)
class TemplateData:
    """Container for an extracted deposit template."""

    array: np.ndarray
    extent: tuple[float, float, float, float]
    polygon: BaseGeometry
    deposit_id: int | None = None
    feature_mask: np.ndarray | None = None
    center_x: float | None = None
    center_y: float | None = None
    radius_m: float | None = None


def load_deposits(path: str | Path, *, target_crs: str | CRS | None = None) -> gpd.GeoDataFrame:
    """Load deposit polygons and optionally reproject to a target CRS."""

    deposit_path = Path(path).expanduser()
    deposits = gpd.read_file(deposit_path)

    if deposits.crs is None:
        raise ValueError(
            f"Deposit file '{deposit_path}' has no CRS. Repair the vector CRS metadata before loading."
        )

    if target_crs is not None:
        deposits = deposits.to_crs(target_crs)

    return deposits


def extract_raster_template(array: Any, transform: Affine, polygon: BaseGeometry) -> TemplateData:
    """Extract a deposit template from a single-band raster and a polygon."""

    raster = np.asarray(array)
    if raster.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {raster.shape}.")

    if polygon is None:
        raise ValueError("polygon geometry must be provided.")
    if polygon.is_empty:
        raise ValueError("polygon geometry is empty.")
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            raise ValueError("Deposit polygon is invalid and could not be repaired.")

    minx, miny, maxx, maxy = polygon.bounds
    col_start, row_start = _geometry_to_pixel_indices(transform, minx, maxy, np.floor)
    col_stop, row_stop = _geometry_to_pixel_indices(transform, maxx, miny, np.ceil)

    col_start, col_stop = _clamp_indices(col_start, col_stop, raster.shape[1])
    row_start, row_stop = _clamp_indices(row_start, row_stop, raster.shape[0])

    if row_start >= row_stop or col_start >= col_stop:
        raise ValueError("Deposit bounds do not intersect the raster grid.")

    template_array = raster[row_start:row_stop, col_start:col_stop]
    if template_array.size == 0:
        raise ValueError("Extracted deposit template is empty.")

    extent = _template_extent(transform, row_start, row_stop, col_start, col_stop)
    return TemplateData(array=template_array, extent=extent, polygon=polygon)


def get_deposit_template(
    deposits_gdf: gpd.GeoDataFrame,
    deposit_index: int,
    raster_data: RasterData,
) -> TemplateData:
    """Select one deposit polygon, align CRS, and extract its raster template."""

    if deposits_gdf.crs is None:
        raise ValueError("Deposit GeoDataFrame has no CRS.")

    raster_crs = raster_data.crs
    if raster_crs is None:
        raise ValueError("Raster data must include a valid CRS.")

    if deposits_gdf.crs != raster_crs:
        deposits_gdf = deposits_gdf.to_crs(raster_crs)

    try:
        deposit_row = deposits_gdf.iloc[deposit_index]
    except IndexError as exc:
        raise IndexError(
            f"Deposit index {deposit_index} is out of range for {len(deposits_gdf)} deposits."
        ) from exc

    template = extract_raster_template(
        raster_data.array,
        raster_data.transform,
        deposit_row.geometry,
    )
    return TemplateData(
        array=template.array,
        extent=template.extent,
        polygon=template.polygon,
        deposit_id=deposit_index,
        feature_mask=template.feature_mask,
        center_x=template.center_x,
        center_y=template.center_y,
        radius_m=template.radius_m,
    )


def get_circle_patch_template(
    *,
    patch_config: dict[str, Any],
    deposits_gdf: gpd.GeoDataFrame,
    deposit_index: int,
    raster_data: RasterData,
) -> TemplateData:
    """Build a circle patch template from config and raster data."""

    if patch_config["source"] == "manual":
        manual = patch_config["manual"]
        center_x = float(manual["center_x"])
        center_y = float(manual["center_y"])
        radius_m = float(manual["radius_m"])
    elif patch_config["source"] == "deposit_bounds":
        deposits_local = deposits_gdf if deposits_gdf.crs == raster_data.crs else deposits_gdf.to_crs(raster_data.crs)
        try:
            deposit_row = deposits_local.iloc[deposit_index]
        except IndexError as exc:
            raise IndexError(
                f"Deposit index {deposit_index} is out of range for {len(deposits_local)} deposits."
            ) from exc
        center_x, center_y, radius_m = _circle_from_bounds(
            deposit_row.geometry,
            radius_rule=str(patch_config["deposit_bounds"]["radius_rule"]),
        )
    else:
        raise ValueError(f"Unsupported patch source '{patch_config['source']}'.")

    return extract_circle_template(
        array=raster_data.array,
        transform=raster_data.transform,
        center_x=center_x,
        center_y=center_y,
        radius_m=radius_m,
        deposit_id=deposit_index,
    )


def extract_circle_template(
    *,
    array: Any,
    transform: Affine,
    center_x: float,
    center_y: float,
    radius_m: float,
    deposit_id: int | None = None,
) -> TemplateData:
    """Extract a circle-defined patch from a raster using a bbox-plus-mask strategy."""

    raster = np.asarray(array, dtype=float)
    if raster.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {raster.shape}.")
    if radius_m <= 0:
        raise ValueError("radius_m must be greater than 0.")

    pixel_size_x = abs(float(transform.a))
    pixel_size_y = abs(float(transform.e))
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("Raster transform must have non-zero pixel size.")

    center_col_float, center_row_float = (~transform) * (center_x, center_y)
    center_col = int(np.round(center_col_float - 0.5))
    center_row = int(np.round(center_row_float - 0.5))
    radius_cols = int(np.ceil(radius_m / pixel_size_x))
    radius_rows = int(np.ceil(radius_m / pixel_size_y))

    row_start = center_row - radius_rows
    row_stop = center_row + radius_rows + 1
    col_start = center_col - radius_cols
    col_stop = center_col + radius_cols + 1

    bbox = _extract_bbox_with_nan_padding(
        raster,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
    )
    mask = _build_circle_feature_mask(
        transform=transform,
        row_start=row_start,
        row_stop=row_stop,
        col_start=col_start,
        col_stop=col_stop,
        center_x=center_x,
        center_y=center_y,
        radius_m=radius_m,
    )
    if not np.any(mask):
        raise ValueError("Circle patch mask selected no pixels.")

    template_array = np.where(mask, bbox, np.nan)
    selected = template_array[mask]
    if not np.isfinite(selected).all():
        raise ValueError("Circle patch contains non-finite in-circle raster values.")

    extent = _template_extent(transform, row_start, row_stop, col_start, col_stop)
    polygon = Point(float(center_x), float(center_y)).buffer(float(radius_m))
    return TemplateData(
        array=template_array,
        extent=extent,
        polygon=polygon,
        deposit_id=deposit_id,
        feature_mask=mask,
        center_x=float(center_x),
        center_y=float(center_y),
        radius_m=float(radius_m),
    )


def _geometry_to_pixel_indices(transform: Affine, x: float, y: float, op: Any) -> tuple[int, int]:
    col, row = (~transform) * (x, y)
    return int(op(col)), int(op(row))


def _clamp_indices(start: int, stop: int, max_size: int) -> tuple[int, int]:
    return max(0, start), min(max_size, stop)


def _template_extent(
    transform: Affine,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> tuple[float, float, float, float]:
    top_left = transform * (col_start, row_start)
    bottom_right = transform * (col_stop, row_stop)
    left = min(top_left[0], bottom_right[0])
    right = max(top_left[0], bottom_right[0])
    bottom = min(top_left[1], bottom_right[1])
    top = max(top_left[1], bottom_right[1])
    return left, right, bottom, top


def _circle_from_bounds(polygon: BaseGeometry, *, radius_rule: str) -> tuple[float, float, float]:
    if polygon is None or polygon.is_empty:
        raise ValueError("Deposit geometry must be non-empty for deposit_bounds circle patches.")
    minx, miny, maxx, maxy = polygon.bounds
    center_x = float((minx + maxx) / 2.0)
    center_y = float((miny + maxy) / 2.0)
    width_m = float(maxx - minx)
    height_m = float(maxy - miny)
    if radius_rule != "half_max_extent":
        raise ValueError(f"Unsupported circle radius_rule '{radius_rule}'.")
    radius_m = 0.5 * max(width_m, height_m)
    return center_x, center_y, float(radius_m)


def _extract_bbox_with_nan_padding(
    raster: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> np.ndarray:
    out = np.full((row_stop - row_start, col_stop - col_start), np.nan, dtype=float)
    src_row_start = max(0, row_start)
    src_row_stop = min(raster.shape[0], row_stop)
    src_col_start = max(0, col_start)
    src_col_stop = min(raster.shape[1], col_stop)
    if src_row_start >= src_row_stop or src_col_start >= src_col_stop:
        return out

    dst_row_start = src_row_start - row_start
    dst_row_stop = dst_row_start + (src_row_stop - src_row_start)
    dst_col_start = src_col_start - col_start
    dst_col_stop = dst_col_start + (src_col_stop - src_col_start)
    out[dst_row_start:dst_row_stop, dst_col_start:dst_col_stop] = raster[
        src_row_start:src_row_stop,
        src_col_start:src_col_stop,
    ]
    return out


def _build_circle_feature_mask(
    *,
    transform: Affine,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
    center_x: float,
    center_y: float,
    radius_m: float,
) -> np.ndarray:
    rows = np.arange(row_start, row_stop, dtype=float)
    cols = np.arange(col_start, col_stop, dtype=float)
    col_grid, row_grid = np.meshgrid(cols + 0.5, rows + 0.5)
    x_grid = transform.c + col_grid * transform.a + row_grid * transform.b
    y_grid = transform.f + col_grid * transform.d + row_grid * transform.e
    dist2 = (x_grid - float(center_x)) ** 2 + (y_grid - float(center_y)) ** 2
    return dist2 <= float(radius_m) ** 2
