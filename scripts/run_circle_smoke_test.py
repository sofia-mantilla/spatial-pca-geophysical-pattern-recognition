"""Run a lightweight synthetic smoke test for experimental circle patches."""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))


def main() -> int:
    """Exercise circle template, window, ranking split, and export contracts."""

    from spatial_pca.geodata.deposits import get_circle_patch_template
    from spatial_pca.geodata.exports import (
        build_circle_top_window_centers_gdf,
        build_circle_top_windows_gdf,
    )
    from spatial_pca.geodata.rasters import RasterData, raster_extent
    from spatial_pca.spca.ranking import rank_multi_two_stage_pca_fusion
    from spatial_pca.spca.windows import (
        build_multivariate_circle_window_matrix,
        build_univariate_circle_window_matrix,
    )

    crs = CRS.from_epsg(3857)
    transform = from_origin(0.0, 12.0, 1.0, 1.0)
    raster = np.arange(144, dtype=float).reshape(12, 12)
    raster_data = RasterData(
        path=REPO_ROOT / "synthetic_circle.tif",
        array=raster,
        transform=transform,
        crs=crs,
        meta={},
        extent=raster_extent(raster, transform),
    )
    deposits = gpd.GeoDataFrame(
        {"name": ["synthetic"]},
        geometry=[box(4.0, 4.0, 8.0, 8.0)],
        crs=crs,
    )
    patch_config = {
        "geometry": "circle",
        "source": "deposit_bounds",
        "deposit_bounds": {"radius_rule": "half_max_extent"},
        "export_geometry": "both",
    }

    template = get_circle_patch_template(
        patch_config=patch_config,
        deposits_gdf=deposits,
        deposit_index=0,
        raster_data=raster_data,
    )
    feature_mask = template.feature_mask
    _check(feature_mask is not None, "Template did not include a feature mask.")
    _check(template.array.shape == feature_mask.shape, "Template and mask shapes differ.")
    _check(np.isnan(template.array[~feature_mask]).any(), "Outside-circle cells should stay out of PCA.")

    uni = build_univariate_circle_window_matrix(
        raster=raster,
        circle_template=template.array,
        variable_name="SYN",
        stride_y=1,
        stride_x=1,
        radius_m=float(template.radius_m),
        feature_mask=feature_mask,
    )
    n_circle_features = int(feature_mask.sum())
    _check(uni.patch_geometry_type == "circle", "Univariate matrix did not retain circle metadata.")
    _check(np.isfinite(uni.data_for_pca).all(), "Univariate circle PCA matrix contains non-finite values.")
    _check(
        uni.data_for_pca.shape[1] == n_circle_features,
        "Univariate feature count does not match feature_mask.sum().",
    )
    _check(
        uni.display_sliding_windows is not None and np.isnan(uni.display_sliding_windows).any(),
        "Display windows should preserve NaNs outside the circle.",
    )

    template_2 = get_circle_patch_template(
        patch_config=patch_config,
        deposits_gdf=deposits,
        deposit_index=0,
        raster_data=RasterData(
            path=REPO_ROOT / "synthetic_circle_2.tif",
            array=raster + 1000.0,
            transform=transform,
            crs=crs,
            meta={},
            extent=raster_extent(raster + 1000.0, transform),
        ),
    )
    multi = build_multivariate_circle_window_matrix(
        rasters={"SYN1": raster, "SYN2": raster + 1000.0},
        circle_templates={"SYN1": template.array, "SYN2": template_2.array},
        variable_names=("SYN1", "SYN2"),
        stride_y=1,
        stride_x=1,
        radius_m=float(template.radius_m),
        feature_mask=feature_mask,
    )
    _check(np.isfinite(multi.data_for_pca).all(), "Multivariate circle PCA matrix contains non-finite values.")
    _check(
        multi.data_for_pca.shape[1] == 2 * n_circle_features,
        "Multivariate feature count should be feature_mask.sum() per variable.",
    )
    ranking = rank_multi_two_stage_pca_fusion(
        X_multi=multi.data_for_pca,
        deposit_index=multi.deposit_index,
        window_shape=multi.window_shape,
        k_pcs_var1=2,
        k_pcs_var2=2,
        k_pcs_fused=2,
        features_per_variable=n_circle_features,
    )
    _check(ranking.ranking_mode == "two_stage_pca_fusion", "Circle multivariate ranking did not run.")

    top_indices = uni.window_indices_for_mapping[:3]
    top_windows = build_circle_top_windows_gdf(
        window_indices=top_indices,
        window_shape=uni.window_shape,
        transform=transform,
        radius_m=float(uni.patch_radius_m),
        crs=crs,
        ranks=np.arange(1, 4),
        scores=np.linspace(0.1, 0.3, 3),
        window_ids=top_indices[:, 2],
        extra_columns={"deposit_1based": np.ones(3, dtype=int)},
    )
    _check((top_windows["patch_shape"] == "circle").all(), "Circle export lost patch_shape metadata.")
    _check(
        set(top_windows.geometry.geom_type) <= {"Polygon"},
        "Circle top-window geometries should be polygons.",
    )
    centers = build_circle_top_window_centers_gdf(top_windows)
    _check(set(centers.geometry.geom_type) <= {"Point"}, "Circle center export should be points.")

    print("circle_smoke_test=PASS")
    print(f"circle_template_shape={template.array.shape}")
    print(f"circle_feature_count={n_circle_features}")
    print(f"univariate_matrix_shape={tuple(uni.data_for_pca.shape)}")
    print(f"multivariate_matrix_shape={tuple(multi.data_for_pca.shape)}")
    print(f"top_window_geometry={top_windows.geometry.geom_type.iloc[0]}")
    return 0


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    raise SystemExit(main())
