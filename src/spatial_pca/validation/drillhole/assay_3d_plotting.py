"""3D drillhole assay QC plots for Cu/Au mineralization validation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatial_pca_matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatial_pca.validation.drillhole.mineralization import apply_mineralization_columns


PLOT_MODES = {
    "mineralized",
    "cu",
    "au",
    "cu_threshold",
    "au_threshold",
    "joint_threshold",
}


@dataclass(frozen=True)
class Assay3DPlotConfig:
    """Display settings for 3D assay QC plots."""

    max_points: int = 20000
    random_seed: int = 42
    marker_size: float = 5.0
    marker_alpha: float = 0.65
    figure_width: float = 11.0
    figure_height: float = 9.0
    dpi: int = 300
    view_elev: float = 24.0
    view_azim: float = -58.0


def aggregate_assays_by_depth_bin(
    assays: pd.DataFrame,
    *,
    depth_bin_m: float,
    mineralization_rule: dict | None = None,
) -> pd.DataFrame:
    """Average interval assays by hole and fixed depth bin for visualization."""

    if depth_bin_m <= 0:
        raise ValueError("depth_bin_m must be positive.")
    required = ["hole_id", "X", "Y", "depth_mid_m", "interval_length", "Cu_pct", "Au_ppm"]
    missing = [col for col in required if col not in assays.columns]
    if missing:
        raise ValueError(f"Missing required depth-bin aggregation columns: {missing}")

    frame = assays.loc[
        assays["hole_id"].notna()
        & assays["X"].notna()
        & assays["Y"].notna()
        & assays["depth_mid_m"].notna()
    ].copy()
    if frame.empty:
        raise ValueError("No valid assay rows available for depth-bin aggregation.")

    frame["depth_bin_from_m"] = np.floor(frame["depth_mid_m"].astype(float) / depth_bin_m) * depth_bin_m
    frame["depth_bin_to_m"] = frame["depth_bin_from_m"] + depth_bin_m
    frame["weight_m"] = pd.to_numeric(frame["interval_length"], errors="coerce").clip(lower=0)
    frame["weight_m"] = frame["weight_m"].where(frame["weight_m"] > 0, 1.0)
    frame["cu_weighted"] = frame["Cu_pct"] * frame["weight_m"]
    frame["au_weighted"] = frame["Au_ppm"] * frame["weight_m"]

    grouped = (
        frame.groupby(["hole_id", "depth_bin_from_m", "depth_bin_to_m"], as_index=False)
        .agg(
            X=("X", "mean"),
            Y=("Y", "mean"),
            depth_mid_m=("depth_mid_m", "mean"),
            interval_count=("hole_id", "size"),
            total_weight_m=("weight_m", "sum"),
            cu_weighted_sum=("cu_weighted", "sum"),
            au_weighted_sum=("au_weighted", "sum"),
        )
    )
    grouped["Cu_pct"] = grouped["cu_weighted_sum"] / grouped["total_weight_m"]
    grouped["Au_ppm"] = grouped["au_weighted_sum"] / grouped["total_weight_m"]
    grouped["interval_length"] = grouped["total_weight_m"]
    grouped["log1p_Cu_pct"] = np.log1p(grouped["Cu_pct"])
    grouped["log1p_Au_ppm"] = np.log1p(grouped["Au_ppm"])
    grouped = apply_mineralization_columns(grouped, **(mineralization_rule or {}))
    return grouped.drop(columns=["cu_weighted_sum", "au_weighted_sum"])


def plot_assays_3d(
    assays: pd.DataFrame,
    *,
    output_path: Path,
    mode: str,
    plot_config: Assay3DPlotConfig | None = None,
    mineralization_rule: dict | None = None,
    title: str | None = None,
) -> Path:
    """Plot desurveyed interval midpoint assays in 3D.

    The vertical coordinate is plotted as depth below surface/collar proxy
    ``depth_mid_m`` and inverted so deeper intervals appear lower.
    """

    mode = mode.lower()
    if mode not in PLOT_MODES:
        raise ValueError(f"mode must be one of {sorted(PLOT_MODES)}")

    cfg = plot_config or Assay3DPlotConfig()
    frame = _sample_assays_for_plot(assays, max_points=cfg.max_points, random_seed=cfg.random_seed)
    values, colorbar_label, cmap, is_binary = _plot_values(frame, mode=mode, mineralization_rule=mineralization_rule or {})

    fig = plt.figure(figsize=(cfg.figure_width, cfg.figure_height))
    ax = fig.add_subplot(111, projection="3d")
    scatter_kwargs = {
        "xs": frame["X"].to_numpy(dtype=float),
        "ys": frame["Y"].to_numpy(dtype=float),
        "zs": frame["depth_mid_m"].to_numpy(dtype=float),
        "s": cfg.marker_size,
        "alpha": cfg.marker_alpha,
        "linewidths": 0,
        "depthshade": False,
    }

    if is_binary:
        colors = np.where(values.astype(bool), "#f46d43", "#4c4c4c")
        ax.scatter(**scatter_kwargs, c=colors)
        _add_binary_legend(ax, positive_label=colorbar_label)
    else:
        finite = values[np.isfinite(values)]
        if finite.size:
            vmax = float(np.nanpercentile(finite, 98))
            vmin = float(np.nanpercentile(finite, 2))
            if vmax <= vmin:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0
        artist = ax.scatter(**scatter_kwargs, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        colorbar = fig.colorbar(artist, ax=ax, pad=0.08, shrink=0.72)
        colorbar.set_label(colorbar_label)

    ax.set_xlabel("Easting X (m)")
    ax.set_ylabel("Northing Y (m)")
    ax.set_zlabel("Depth midpoint (m)")
    ax.invert_zaxis()
    ax.view_init(elev=cfg.view_elev, azim=cfg.view_azim)
    ax.set_title(title or _default_title(mode, len(frame), len(assays)))
    ax.grid(True, linewidth=0.4, alpha=0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_assays_3d_html(
    assays: pd.DataFrame,
    *,
    output_path: Path,
    mode: str,
    plot_config: Assay3DPlotConfig | None = None,
    mineralization_rule: dict | None = None,
    title: str | None = None,
) -> Path:
    """Write a browser-rotatable 3D assay plot as HTML using Plotly CDN."""

    mode = mode.lower()
    if mode not in PLOT_MODES:
        raise ValueError(f"mode must be one of {sorted(PLOT_MODES)}")

    cfg = plot_config or Assay3DPlotConfig()
    frame = _sample_assays_for_plot(assays, max_points=cfg.max_points, random_seed=cfg.random_seed)
    values, colorbar_label, cmap, is_binary = _plot_values(frame, mode=mode, mineralization_rule=mineralization_rule or {})
    title_text = title or _default_title(mode, len(frame), len(assays))

    if is_binary:
        positive = values.astype(bool)
        data = [
            _plotly_scatter3d_trace(frame.loc[~positive], name="Below / not classified positive", color="#4c4c4c", marker_size=3),
            _plotly_scatter3d_trace(frame.loc[positive], name=colorbar_label, color="#f46d43", marker_size=4),
        ]
    else:
        data = [
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": _json_float_list(frame["X"]),
                "y": _json_float_list(frame["Y"]),
                "z": _json_float_list(frame["depth_mid_m"]),
                "marker": {
                    "size": 3,
                    "opacity": 0.7,
                    "color": _json_float_list(pd.Series(values, index=frame.index)),
                    "colorscale": "Viridis" if cmap == "viridis" else "Plasma",
                    "colorbar": {"title": colorbar_label},
                },
                "name": colorbar_label,
            }
        ]

    layout = {
        "title": title_text,
        "scene": {
            "xaxis": {"title": "Easting X (m)"},
            "yaxis": {"title": "Northing Y (m)"},
            "zaxis": {"title": "Depth midpoint (m)", "autorange": "reversed"},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 60},
        "legend": {"x": 0.01, "y": 0.98},
    }
    html = _plotly_html(data=data, layout=layout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _sample_assays_for_plot(assays: pd.DataFrame, *, max_points: int, random_seed: int) -> pd.DataFrame:
    required = ["X", "Y", "depth_mid_m", "Cu_pct", "Au_ppm", "mineralized_state"]
    missing = [col for col in required if col not in assays.columns]
    if missing:
        raise ValueError(f"Missing required 3D assay plot columns: {missing}")

    frame = assays.loc[
        assays["X"].notna() & assays["Y"].notna() & assays["depth_mid_m"].notna()
    ].copy()
    if frame.empty:
        raise ValueError("No valid assay rows available for 3D plotting.")
    if max_points > 0 and len(frame) > max_points:
        frame = frame.sample(n=max_points, random_state=random_seed).copy()
    return frame


def _plot_values(
    frame: pd.DataFrame,
    *,
    mode: str,
    mineralization_rule: dict,
) -> tuple[np.ndarray, str, str, bool]:
    cu_threshold = float(mineralization_rule.get("cu_threshold_pct", 0.1))
    au_threshold = float(mineralization_rule.get("au_threshold_ppm", 0.1))
    joint_cu = float(mineralization_rule.get("joint_cu_threshold_pct", 0.08))
    joint_au = float(mineralization_rule.get("joint_au_threshold_ppm", 0.08))

    if mode == "cu":
        return np.log1p(frame["Cu_pct"].to_numpy(dtype=float)), "log1p(Cu %)", "viridis", False
    if mode == "au":
        return np.log1p(frame["Au_ppm"].to_numpy(dtype=float)), "log1p(Au ppm)", "plasma", False
    if mode == "cu_threshold":
        values = frame["Cu_pct"].to_numpy(dtype=float) >= cu_threshold
        return values, f"Cu >= {cu_threshold:g}%", "none", True
    if mode == "au_threshold":
        values = frame["Au_ppm"].to_numpy(dtype=float) >= au_threshold
        return values, f"Au >= {au_threshold:g} ppm", "none", True
    if mode == "joint_threshold":
        values = (
            (frame["Cu_pct"].to_numpy(dtype=float) >= joint_cu)
            & (frame["Au_ppm"].to_numpy(dtype=float) >= joint_au)
        )
        return values, f"Cu >= {joint_cu:g}% and Au >= {joint_au:g} ppm", "none", True

    values = frame["mineralized_state"].fillna(0).astype(bool).to_numpy()
    return values, "Mineralized by Cu/Au rule", "none", True


def _add_binary_legend(ax, *, positive_label: str) -> None:
    positive = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f46d43", markersize=7, label=positive_label)
    negative = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c4c4c", markersize=7, label="Below / not classified positive")
    ax.legend(handles=[positive, negative], loc="upper left", frameon=True)


def _default_title(mode: str, plotted_count: int, total_count: int) -> str:
    mode_titles = {
        "mineralized": "3D drillhole intervals: combined Cu/Au mineralization rule",
        "cu": "3D drillhole intervals colored by Cu grade",
        "au": "3D drillhole intervals colored by Au grade",
        "cu_threshold": "3D drillhole intervals: Cu threshold exceedance",
        "au_threshold": "3D drillhole intervals: Au threshold exceedance",
        "joint_threshold": "3D drillhole intervals: joint Cu/Au threshold exceedance",
    }
    suffix = f" ({plotted_count:,} plotted"
    if plotted_count != total_count:
        suffix += f" of {total_count:,}"
    suffix += ")"
    return mode_titles[mode] + suffix


def _plotly_scatter3d_trace(frame: pd.DataFrame, *, name: str, color: str, marker_size: int) -> dict:
    return {
        "type": "scatter3d",
        "mode": "markers",
        "x": _json_float_list(frame["X"]),
        "y": _json_float_list(frame["Y"]),
        "z": _json_float_list(frame["depth_mid_m"]),
        "marker": {"size": marker_size, "opacity": 0.72, "color": color},
        "name": name,
    }


def _json_float_list(values: pd.Series) -> list[float | None]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return [None if not np.isfinite(value) else float(value) for value in array]


def _plotly_html(*, data: list[dict], layout: dict) -> str:
    data_json = json.dumps(data)
    layout_json = json.dumps(layout)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spatial PCA 3D assay plot</title>
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
