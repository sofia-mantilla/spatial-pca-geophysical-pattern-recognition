"""Raster loading, validation, and alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.transform import array_bounds


@dataclass(frozen=True)
class RasterData:
    """Container for a loaded single-band raster."""

    path: Path
    array: np.ndarray
    transform: Any
    crs: CRS
    meta: dict[str, Any]
    extent: tuple[float, float, float, float]


def load_raster(
    path: str | Path,
    *,
    force_crs: str | CRS | None = None,
    nodata_to_nan: float | None = None,
    band: int = 1,
    crop_polygon_path: str | Path | None = None,
) -> RasterData:
    """Load one raster band and return array, transform, CRS, metadata, and extent.

    ``force_crs`` repairs CRS metadata only; it does not reproject raster cells.
    Use it only when the configured CRS is known to be correct for the source
    raster. ``crop_polygon_path`` crops the raster before analysis using
    polygon coordinates in the raster CRS; if the polygon file has no CRS,
    coordinates are assumed to already match the raster grid.
    """

    raster_path = Path(path).expanduser()
    with rasterio.open(raster_path) as src:
        if band < 1 or band > src.count:
            raise ValueError(f"band must be between 1 and {src.count}, got {band}.")
        meta = src.meta.copy()
        crs = _resolve_crs(src.crs, force_crs=force_crs)
        if crop_polygon_path is None:
            array = src.read(band)
            transform = src.transform
        else:
            crop_gdf = _load_crop_polygons(crop_polygon_path, raster_crs=src.crs or crs)
            cropped, transform = mask(
                dataset=src,
                shapes=list(crop_gdf.geometry),
                crop=True,
                nodata=src.nodata,
                indexes=band,
            )
            array = cropped

    array = np.asarray(array)
    if nodata_to_nan is not None:
        array = array.astype(np.float32, copy=False)
        array[array == nodata_to_nan] = np.nan

    meta["crs"] = crs
    meta["transform"] = transform
    meta["height"] = array.shape[0]
    meta["width"] = array.shape[1]

    return RasterData(
        path=raster_path,
        array=array,
        transform=transform,
        crs=crs,
        meta=meta,
        extent=raster_extent(array, transform),
    )


def raster_extent(array: Any, transform: Any) -> tuple[float, float, float, float]:
    """Return extent formatted for plotting as left, right, bottom, top."""

    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"array must be 2D, got shape {arr.shape}.")
    height, width = arr.shape
    left, bottom, right, top = array_bounds(height, width, transform)
    return float(left), float(right), float(bottom), float(top)


def _resolve_crs(source_crs: Any, *, force_crs: str | CRS | None = None) -> CRS:
    if force_crs is not None:
        return CRS.from_user_input(force_crs)
    if source_crs is None:
        raise ValueError("Raster CRS is missing. Pass force_crs only if the CRS is known.")
    return CRS.from_user_input(source_crs)


def _load_crop_polygons(path: str | Path, *, raster_crs: Any) -> gpd.GeoDataFrame:
    polygon_path = Path(path).expanduser()
    polygons = gpd.read_file(polygon_path)
    polygons = polygons[polygons.geometry.notnull()].copy()
    polygons = polygons[~polygons.geometry.is_empty].copy()
    if polygons.empty:
        raise ValueError(f"Crop polygon file '{polygon_path}' contains no usable geometries.")

    if polygons.crs is not None and raster_crs is not None and polygons.crs != raster_crs:
        polygons = polygons.to_crs(raster_crs)

    return polygons
