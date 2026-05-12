"""Footprint-recovery validation against known deposit polygons."""

from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatial_pca_matplotlib_cache"))

import geopandas as gpd
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from rasterio.features import rasterize
from shapely.geometry import GeometryCollection

from spatial_pca.colormaps import DEFAULT_PAPER_CMAP, resolve_colormap


@dataclass(frozen=True)
class FootprintRecoveryResult:
    """Cumulative footprint-recovery validation result."""

    ranked_pred_rows: np.ndarray
    cum_recovered_area: np.ndarray
    cum_recovered_frac_total: np.ndarray
    cum_mean_recovered_frac: np.ndarray
    coverage_by_deposit: dict[int, float]
    covered_area_by_deposit: dict[int, float]
    dep_area_by_deposit: dict[int, float]
    overlap_by_rank: dict[int, list[int]]
    hit_by_rank: dict[int, list[int]]
    first_hit_rank_by_deposit: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a pickle-friendly dictionary."""

        return {
            "ranked_pred_rows": self.ranked_pred_rows,
            "cum_recovered_area": self.cum_recovered_area,
            "cum_recovered_frac_total": self.cum_recovered_frac_total,
            "cum_mean_recovered_frac": self.cum_mean_recovered_frac,
            "coverage_by_deposit": self.coverage_by_deposit,
            "covered_area_by_deposit": self.covered_area_by_deposit,
            "dep_area_by_deposit": self.dep_area_by_deposit,
            "overlap_by_rank": self.overlap_by_rank,
            "hit_by_rank": self.hit_by_rank,
            "first_hit_rank_by_deposit": self.first_hit_rank_by_deposit,
        }


def validate_footprint_recovery(
    *,
    top_windows_gdf: gpd.GeoDataFrame,
    deposits_gdf: gpd.GeoDataFrame,
    reference_deposit_index: int,
    min_cover: float,
) -> FootprintRecoveryResult:
    """Validate ranked prediction windows against non-reference deposits."""

    if top_windows_gdf.empty:
        raise ValueError("top_windows_gdf is empty.")
    if top_windows_gdf.crs is None:
        raise ValueError("top_windows_gdf must have a CRS.")
    if deposits_gdf.crs is None:
        raise ValueError("deposits_gdf must have a CRS.")
    if not 0.0 < float(min_cover) <= 1.0:
        raise ValueError("min_cover must be in the interval (0, 1].")

    if deposits_gdf.crs != top_windows_gdf.crs:
        deposits_gdf = deposits_gdf.to_crs(top_windows_gdf.crs)

    if not (0 <= int(reference_deposit_index) < len(deposits_gdf)):
        raise IndexError("reference_deposit_index is out of bounds for deposits_gdf.")

    other_deposits = deposits_gdf.drop(index=deposits_gdf.index[int(reference_deposit_index)]).copy()
    other_deposits = other_deposits[other_deposits.geometry.notnull()].copy()
    other_deposits["geometry"] = other_deposits.geometry.buffer(0)
    other_deposits = other_deposits[~other_deposits.geometry.is_empty].copy()

    dep_area_by_deposit = {
        int(idx): float(geom.area)
        for idx, geom in other_deposits.geometry.items()
        if float(geom.area) > 0
    }
    deposit_geometries = {
        int(idx): geom
        for idx, geom in other_deposits.geometry.items()
        if int(idx) in dep_area_by_deposit
    }

    total_test_area = float(sum(dep_area_by_deposit.values()))
    n_test_deposits = len(dep_area_by_deposit)

    covered_geom_by_deposit: dict[int, Any] = {}
    covered_area_by_deposit: dict[int, float] = {}
    coverage_by_deposit: dict[int, float] = {}
    overlap_by_rank: dict[int, list[int]] = {}
    hit_by_rank: dict[int, list[int]] = {}
    first_hit_rank_by_deposit: dict[int, int] = {}
    hit_deposits: set[int] = set()

    cum_recovered_area: list[float] = []
    cum_recovered_frac_total: list[float] = []
    cum_mean_recovered_frac: list[float] = []

    ranked = top_windows_gdf.sort_values("rank").reset_index(drop=True)

    for rank, pred_row in enumerate(ranked.itertuples(), start=1):
        pred_geom = pred_row.geometry
        overlapped_this_rank: list[int] = []
        newly_hit: list[int] = []

        for dep_id, dep_geom in deposit_geometries.items():
            inter = pred_geom.intersection(dep_geom)
            if inter.is_empty or inter.area <= 0:
                continue

            overlapped_this_rank.append(dep_id)
            previous = covered_geom_by_deposit.get(dep_id, GeometryCollection())
            updated = previous.union(inter)
            covered_geom_by_deposit[dep_id] = updated

            covered_area = float(updated.area)
            covered_area_by_deposit[dep_id] = covered_area
            coverage = covered_area / dep_area_by_deposit[dep_id]
            coverage_by_deposit[dep_id] = coverage

            if coverage >= min_cover and dep_id not in hit_deposits:
                hit_deposits.add(dep_id)
                first_hit_rank_by_deposit[dep_id] = rank
                newly_hit.append(dep_id)

        if overlapped_this_rank:
            overlap_by_rank[rank] = sorted(overlapped_this_rank)
        if newly_hit:
            hit_by_rank[rank] = sorted(newly_hit)

        total_covered = float(sum(covered_area_by_deposit.values()))
        cum_recovered_area.append(total_covered)
        if total_test_area > 0:
            cum_recovered_frac_total.append(total_covered / total_test_area)
        else:
            cum_recovered_frac_total.append(0.0)

        if n_test_deposits > 0:
            mean_cov = float(
                np.mean([coverage_by_deposit.get(dep_id, 0.0) for dep_id in dep_area_by_deposit])
            )
        else:
            mean_cov = 0.0
        cum_mean_recovered_frac.append(mean_cov)

    return FootprintRecoveryResult(
        ranked_pred_rows=ranked.index.to_numpy(dtype=int),
        cum_recovered_area=np.asarray(cum_recovered_area, dtype=float),
        cum_recovered_frac_total=np.asarray(cum_recovered_frac_total, dtype=float),
        cum_mean_recovered_frac=np.asarray(cum_mean_recovered_frac, dtype=float),
        coverage_by_deposit=coverage_by_deposit,
        covered_area_by_deposit=covered_area_by_deposit,
        dep_area_by_deposit=dep_area_by_deposit,
        overlap_by_rank=overlap_by_rank,
        hit_by_rank=hit_by_rank,
        first_hit_rank_by_deposit=first_hit_rank_by_deposit,
    )


def build_validation_payload(
    *,
    recovery: FootprintRecoveryResult,
    method_name: str,
    analysis_type: str,
    deposit_id: int,
    variables: list[str],
    k_pcs: int,
    min_cover: float,
    top_windows_path: str | Path,
    run_config_path: str | Path,
    provenance_path: str | Path,
    spca_diagnostics: dict[str, Any] | None = None,
    deposit_metrics: dict[str, Any] | None = None,
    ranking_mode: str = "shared_weighted_l2",
    k_pcs_var1: float | int = np.nan,
    k_pcs_var2: float | int = np.nan,
    k_pcs_fused: float | int = np.nan,
) -> dict[str, Any]:
    """Build the required validation payload for a run output folder."""

    payload = recovery.to_dict()
    payload.update(
        {
            "method_name": method_name,
            "analysis_type": analysis_type,
            "deposit_id": int(deposit_id),
            "variables": list(variables),
            "k_pcs": int(k_pcs),
            "min_cover": float(min_cover),
            "top_windows_path": str(top_windows_path),
            "run_config_path": str(run_config_path),
            "provenance_path": str(provenance_path),
            "spca_diagnostics": spca_diagnostics or {},
            "deposit_metrics": deposit_metrics or {},
            "ranking_mode": ranking_mode,
            "k_pcs_var1": k_pcs_var1,
            "k_pcs_var2": k_pcs_var2,
            "k_pcs_fused": k_pcs_fused,
        }
    )
    return payload


def write_validation_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    """Write validation payload as a pickle file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as outfile:
        pickle.dump(payload, outfile)
    return path


def plot_cumulative_recovery(
    recovery: FootprintRecoveryResult,
    output_path: str | Path,
    *,
    deposit_1based: int | None = None,
    min_cover: float = 0.5,
    title: str | None = None,
) -> Path:
    """Plot cumulative footprint recovery using the legacy paper style."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(recovery.cum_recovered_frac_total, dtype=float).ravel()
    ranks = np.arange(1, y.size + 1)

    if title is None:
        title = (
            "Cumulative footprint recovery\n"
            f"Orange = any overlap event, Red = first threshold event (>={int(100 * min_cover)}% coverage)"
        )

    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=150)
    ax.plot(
        ranks,
        y,
        color="black",
        linewidth=2.0,
        label="Cumulative recovered fraction (unique recovered area)",
        zorder=2,
    )
    ax.set_xlim(1, max(1, y.size))
    ax.set_ylim(0, 1.02)
    if deposit_1based is None:
        ax.set_xlabel("Prediction rank (1 = most similar)")
    else:
        ax.set_xlabel(f"Prediction rank (1 = most similar to training deposit #{deposit_1based})")
    ax.set_ylabel("Cumulative recovered fraction of all test-deposit area")
    ax.set_title(title)
    if recovery.overlap_by_rank:
        overlap_ranks = np.asarray(sorted(recovery.overlap_by_rank), dtype=int)
        ax.scatter(
            overlap_ranks,
            y[overlap_ranks - 1],
            s=28,
            color="orange",
            linewidths=0,
            label="Rank with any test-deposit overlap",
            zorder=3,
        )
    if recovery.hit_by_rank:
        hit_ranks = np.asarray(sorted(recovery.hit_by_rank), dtype=int)
        ax.scatter(
            hit_ranks,
            y[hit_ranks - 1],
            s=42,
            color="red",
            linewidths=0,
            label=f"Rank where a deposit first reaches the threshold (>={int(100 * min_cover)}%)",
            zorder=4,
        )

        for rank in hit_ranks:
            dep_rows = recovery.hit_by_rank.get(int(rank), [])
            for offset_idx, dep_row in enumerate(dep_rows):
                ax.annotate(
                    text=str(int(dep_row) + 1),
                    xy=(int(rank), float(y[int(rank) - 1])),
                    xytext=(0, 6 + offset_idx * 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="red",
                )

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_top_windows_overlay(
    *,
    top_windows_gdf: gpd.GeoDataFrame,
    deposits_gdf: gpd.GeoDataFrame,
    reference_deposit_index: int,
    background_layers: dict[str, dict[str, Any]],
    transform: Any,
    output_path: str | Path,
    title: str | None = None,
    image_cmap: str | Any | None = None,
) -> Path:
    """Plot top-ranked SPCA windows with known deposit footprints."""

    if top_windows_gdf.empty:
        raise ValueError("top_windows_gdf is empty.")
    if top_windows_gdf.crs is None:
        raise ValueError("top_windows_gdf must have a CRS.")
    if deposits_gdf.crs is None:
        raise ValueError("deposits_gdf must have a CRS.")
    if not (0 <= int(reference_deposit_index) < len(deposits_gdf)):
        raise IndexError("reference_deposit_index is out of bounds for deposits_gdf.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    deposits_plot = deposits_gdf.to_crs(top_windows_gdf.crs)
    deposits_plot = deposits_plot[deposits_plot.geometry.notnull()].copy()
    deposits_plot["geometry"] = deposits_plot.geometry.buffer(0)
    deposits_plot = deposits_plot[~deposits_plot.geometry.is_empty].copy()

    reference_label = deposits_gdf.index[int(reference_deposit_index)]
    reference = deposits_plot.loc[[reference_label]]
    other_deposits = deposits_plot.drop(index=reference_label)
    ranked_windows = top_windows_gdf.sort_values("rank").reset_index(drop=True)
    top_subset = ranked_windows.head(min(10, len(ranked_windows)))

    ordered_vars = list(background_layers)
    if not ordered_vars:
        raise ValueError("background_layers must contain at least one variable panel.")

    first_layer = background_layers[ordered_vars[0]]
    first_shape = np.asarray(first_layer["array"]).shape
    if len(first_shape) != 2:
        raise ValueError("Background raster arrays must be 2D.")
    window_mask = _rasterize_highlight_mask(
        ranked_windows.geometry,
        shape=(int(first_shape[0]), int(first_shape[1])),
        transform=transform,
    )

    fig, axes = plt.subplots(
        1,
        len(ordered_vars),
        figsize=(10 * len(ordered_vars), 10),
        squeeze=False,
    )
    cmap = resolve_colormap(image_cmap or DEFAULT_PAPER_CMAP)

    for ax, variable_name in zip(axes.flat, ordered_vars):
        layer = background_layers[variable_name]
        background = np.asarray(layer["array"], dtype=float)
        if background.shape != first_shape:
            raise ValueError("All background raster arrays must share the same shape for overlay plotting.")
        extent = tuple(layer["extent"])
        vmin = layer.get("vmin")
        vmax = layer.get("vmax")

        dark_rgba = _array_to_rgba(background, cmap=cmap, vmin=vmin, vmax=vmax, brightness_scale=0.35)
        bright_rgba = _array_to_rgba(background, cmap=cmap, vmin=vmin, vmax=vmax, brightness_scale=1.0)
        bright_rgba[..., 3] *= window_mask.astype(float)

        ax.imshow(dark_rgba, extent=extent, origin="upper")
        ax.imshow(bright_rgba, extent=extent, origin="upper")

        if not top_subset.empty:
            top_subset.boundary.plot(
                ax=ax,
                color="white",
                linewidth=1.0,
                alpha=1.0,
                label="Top 10 Predicted Windows",
                zorder=6,
            )
        if not other_deposits.empty:
            other_deposits.boundary.plot(
                ax=ax,
                color="black",
                linewidth=2.5,
                label="Testing Known Deposits",
                zorder=7,
            )
        reference.boundary.plot(
            ax=ax,
            color="red",
            linewidth=3.0,
            label="Training Known Deposit",
            zorder=8,
        )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.grid(False)
        _add_scale_bar(ax, extent)
        plot_vmin, plot_vmax = _resolve_plot_limits(background, vmin, vmax)
        scalar_mappable = plt.cm.ScalarMappable(
            norm=Normalize(vmin=plot_vmin, vmax=plot_vmax, clip=True),
            cmap=cmap,
        )
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label(variable_name)
        if len(ordered_vars) > 1:
            ax.set_title(variable_name)

    legend_handles = [
        Line2D([0], [0], color="black", linewidth=2.5, label="Testing Known Deposits"),
        Line2D([0], [0], color="red", linewidth=3.0, label="Training Known Deposit"),
        Line2D([0], [0], color="white", linewidth=1.0, label="Top 10 Predicted Windows"),
        Line2D([0], [0], color="black", marker=r"$\uparrow$", linestyle="None", markersize=12, label="N"),
    ]
    axes.flat[0].legend(
        handles=legend_handles,
        loc="best",
        facecolor="#d4d4d8",
        framealpha=0.95,
        edgecolor="#52525b",
    )
    title = title or f"Top {len(ranked_windows)} Prediction Windows"
    if len(ordered_vars) == 1:
        axes.flat[0].set_title(title)
        fig.tight_layout()
    else:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _combined_total_bounds(*gdfs: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    bounds = np.asarray([gdf.total_bounds for gdf in gdfs if not gdf.empty], dtype=float)
    if bounds.size == 0:
        raise ValueError("At least one non-empty GeoDataFrame is required.")
    return (
        float(np.min(bounds[:, 0])),
        float(np.min(bounds[:, 1])),
        float(np.max(bounds[:, 2])),
        float(np.max(bounds[:, 3])),
    )


def _rasterize_highlight_mask(
    geometries: Any,
    *,
    shape: tuple[int, int],
    transform: Any,
) -> np.ndarray:
    clean_geometries = [geom.buffer(0) for geom in geometries if geom is not None and not geom.is_empty]
    if not clean_geometries:
        return np.zeros(shape, dtype=bool)
    mask = rasterize(
        [(geom, 1) for geom in clean_geometries],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype="uint8",
    )
    return np.asarray(mask, dtype=bool)


def _array_to_rgba(
    array: np.ndarray,
    *,
    cmap: Any,
    vmin: float | None,
    vmax: float | None,
    brightness_scale: float,
) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        rgba = np.zeros(arr.shape + (4,), dtype=float)
        return rgba

    vmin, vmax = _resolve_plot_limits(arr, vmin, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = np.asarray(cmap(norm(arr)), dtype=float)
    rgba[..., :3] *= float(brightness_scale)
    rgba[..., 3] = np.where(finite, rgba[..., 3], 0.0)
    return rgba


def _resolve_plot_limits(
    array: np.ndarray,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    arr = np.asarray(array, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return 0.0, 1.0
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Raster plotting limits must be finite when derived from the data.")
    if np.isclose(vmin, vmax):
        vmax = float(vmin) + 1.0
    return float(vmin), float(vmax)


def _add_scale_bar(ax: Any, extent: tuple[float, float, float, float]) -> None:
    x0, x1, y0, y1 = extent
    width = float(x1 - x0)
    height = float(y1 - y0)
    length = _nice_scale_length(width * 0.2)
    bar_x0 = x0 + width * 0.06
    bar_y = y0 + height * 0.08
    bar_x1 = bar_x0 + length

    ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="black", linewidth=4, solid_capstyle="butt", zorder=20)
    tick_height = height * 0.015
    ax.plot([bar_x0, bar_x0], [bar_y - tick_height / 2, bar_y + tick_height / 2], color="black", linewidth=2, zorder=20)
    ax.plot([bar_x1, bar_x1], [bar_y - tick_height / 2, bar_y + tick_height / 2], color="black", linewidth=2, zorder=20)
    label = f"{int(length / 1000)} km" if length >= 1000 else f"{int(length)} m"
    ax.text(
        (bar_x0 + bar_x1) / 2,
        bar_y + height * 0.02,
        label,
        ha="center",
        va="bottom",
        fontsize=11,
        color="black",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
        zorder=21,
    )


def _nice_scale_length(target_length: float) -> float:
    if target_length <= 0:
        return 1000.0
    exponent = np.floor(np.log10(target_length))
    base = target_length / (10**exponent)
    if base <= 1:
        nice_base = 1
    elif base <= 2:
        nice_base = 2
    elif base <= 5:
        nice_base = 5
    else:
        nice_base = 10
    return float(nice_base * (10**exponent))
