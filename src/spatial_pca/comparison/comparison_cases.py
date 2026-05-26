# comparison_cases.py
# ------------------------------------------------------------
# Configurable comparison script for validation curves.
# Edit the USER COMPARISON SETTINGS block to switch inputs/groups without
# modifying plotting logic.
# ------------------------------------------------------------

import glob
import json
import os
import pickle
import re
import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from spatial_pca.colormaps import resolve_colormap

try:
    import yaml
except ImportError:  # pragma: no cover - only exercised when PyYAML is absent.
    yaml = None


BASE_OUTPUT_DIR = (
    "/Users/sofiamantillasalas/Library/CloudStorage/OneDrive-Stanford/"
    "Research_Stanford/Research_files/MineralX_research/"
    "EroCopper_project/Carajas_maps_and_data/outputs_v3"
)

DEFAULT_MINCOVER = 0.5
COMPARE_OUTDIR = os.path.join(BASE_OUTPUT_DIR, "_validation_comparisons")
BEST_KPCS_CSV = os.path.join(BASE_OUTPUT_DIR, "kpcs_best_multicriteria_by_deposit_multi.csv")

DEPOSITS_1BASED = list(range(1, 15))
FORCE_K = 250
DEFAULT_YMAX = 0.95
DEPOSIT_CMAP_FALLBACK = "RdBu_r"
DEPOSIT_LIMITS_TMI = {
    1: (-250, 300),
    2: (-200, 400),
    3: (-150, 150),
    4: (-600, 400),
    5: (-200, 400),
    6: (-200, 400),
    7: (-150, 300),
    8: (-200, 300),
    9: (-100, 100),
    10: (-200, 300),
    11: (-300, 500),
    12: (-200, 400),
}
DEPOSIT_LIMITS_U = {dep: (0, 15) for dep in range(1, 100)}


def setup_analysis_config(**_: Any) -> Dict[str, Any]:
    """Compatibility shim for the original paper comparison script."""

    return {"cmap": resolve_colormap("spatial_pca_paper")}

# -----------------------------
# USER COMPARISON SETTINGS
# -----------------------------
# Group filter keys accepted by collect_recovery_for_group:
#   method_name, analysis_type, selected_variable, min_cover
COMPARISON = {
    "name": "spca_uni_tmi_vs_spca_multi_tmi_u",
    "group_a": dict(
        method_name="Spatial_PCA",
        analysis_type="Uni",
        selected_variable="TMI",
        min_cover=DEFAULT_MINCOVER,
        label="Spatial_PCA Uni (TMI)",
    ),
    "group_b": dict(
        method_name="Spatial_PCA",
        analysis_type="Multi",
        min_cover=DEFAULT_MINCOVER,
        label="Spatial_PCA Multi (TMI+U)",
    ),
    "aggregate_title": "All deposits: footprint recovery - sPCA Uni (TMI) vs sPCA Multi (TMI+U)",
    "curves_title": "All cumulative footprint recovery curves: sPCA Uni (TMI) vs sPCA Multi (TMI+U)",
    "color_a": "blue",
    "color_b": "green",
}

# Apply per-deposit best-k filtering to Spatial_PCA groups if CSV is available.
USE_BEST_KPCS_FILTER = True

# Extra diagnostics (top gains, optimal-k scatter, strip, S-N, 2D windows) are
# only meaningful for Raw vs Spatial_PCA Uni(TMI). Set False for generic runs.
RUN_RAW_SPCA_DIAGNOSTICS = False
RUN_TOP_GAIN_SUBPLOTS = True
TOP_GAIN_N = 4

DEFAULT_CONFIG = {
    "inputs": {
        "output_roots": [BASE_OUTPUT_DIR],
        "deposits_1based": DEPOSITS_1BASED,
        "best_kpcs_csv": BEST_KPCS_CSV,
        "compare_outdir": COMPARE_OUTDIR,
    },
    "filters": {
        "default_min_cover": DEFAULT_MINCOVER,
        "use_best_kpcs_filter": USE_BEST_KPCS_FILTER,
    },
    "plot": {
        "force_k": FORCE_K,
        "default_ymax": DEFAULT_YMAX,
        "deposit_limits_tmi": DEPOSIT_LIMITS_TMI,
        "deposit_limits_u": DEPOSIT_LIMITS_U,
    },
    "comparison": COMPARISON,
    "run": {
        "run_raw_spca_diagnostics": RUN_RAW_SPCA_DIAGNOSTICS,
        "run_top_gain_subplots": RUN_TOP_GAIN_SUBPLOTS,
        "top_gain_n": TOP_GAIN_N,
        "top_gain_base_group_index": None,
        "top_gain_target_group_index": None,
        "selected_deposits_for_subplots": None,
        "sn_smoothing": {
            "enabled": False,
            "method": "moving_average",
            "window": 5
        },
    },
}


@dataclass
class CaseInfo:
    deposit_1based: int
    method_name: str
    analysis_type: str
    selected_variable: Optional[str] = None
    variable_1: Optional[str] = None
    variable_2: Optional[str] = None
    rotation_deg: Optional[float] = None
    min_cover: Optional[float] = None
    k_pcs: Optional[int] = None
    case_dir: Optional[str] = None
    pkl_path: Optional[str] = None


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _norm_varname(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip()
    if x.upper() == "MTI":
        return "TMI"
    return x


def parse_case_dir_name(case_dirname: str) -> Optional[CaseInfo]:
    name = os.path.basename(case_dirname)

    m_dep = re.search(r"Deposit_(\d+)_", name)
    if not m_dep:
        return None
    dep = int(m_dep.group(1))

    m_method = re.search(rf"Deposit_{dep}_(.+?)_(Uni|Multi)_", name)
    if not m_method:
        return None
    method = m_method.group(1)
    analysis_type = m_method.group(2)

    m_rot = re.search(r"_([-\d\.]+)_deg_", name)
    rotation = _safe_float(m_rot.group(1)) if m_rot else None

    m_cov = re.search(r"_([0-9]*\.?[0-9]+)_minCov(?:_kpcs_(\d+))?$", name)
    mincov = _safe_float(m_cov.group(1)) if m_cov else None
    k_pcs = int(m_cov.group(2)) if (m_cov and m_cov.group(2) is not None) else None

    selected_variable = None
    var1 = None
    var2 = None

    if analysis_type == "Uni":
        m_vars = re.search(
            rf"Deposit_{dep}_{re.escape(method)}_Uni_(.+?)_([-\d\.]+)_deg_",
            name,
        )
        if m_vars:
            selected_variable = m_vars.group(1)
    else:
        m_vars = re.search(
            rf"Deposit_{dep}_{re.escape(method)}_Multi_(.+?)_([-\d\.]+)_deg_",
            name,
        )
        if m_vars:
            var_chunk = m_vars.group(1)
            toks = var_chunk.split("_")
            if len(toks) >= 2:
                var1 = toks[0]
                var2 = "_".join(toks[1:])
            else:
                var1 = var_chunk

    return CaseInfo(
        deposit_1based=dep,
        method_name=method,
        analysis_type=analysis_type,
        selected_variable=selected_variable,
        variable_1=var1,
        variable_2=var2,
        rotation_deg=rotation,
        min_cover=mincov,
        k_pcs=k_pcs,
        case_dir=case_dirname,
    )


def load_best_kpcs_map(csv_path: str, k_column: str = "k_pcs") -> Dict[int, int]:
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    required = {"deposit_1based", k_column}
    if not required.issubset(df.columns):
        return {}
    out: Dict[int, int] = {}
    for _, r in df.iterrows():
        try:
            dep = int(r["deposit_1based"])
            k = int(r[k_column])
        except Exception:
            continue
        out[dep] = k
    return out


def find_all_validation_pkls(
    base_outputs_dirs: List[str],
    deposits_1based: Optional[List[int]] = None,
) -> List[CaseInfo]:
    pkls: List[str] = []
    for root in base_outputs_dirs:
        pkls.extend(
            glob.glob(
                os.path.join(root, "**", "validation_topk_results.pkl"),
                recursive=True,
            )
        )

    cases: List[CaseInfo] = []
    for pkl_path in pkls:
        case_dir = os.path.dirname(pkl_path)
        info = parse_case_dir_name(case_dir)
        if info is None:
            continue
        info.pkl_path = pkl_path
        cases.append(info)

    if deposits_1based is not None:
        cases = [c for c in cases if c.deposit_1based in deposits_1based]

    return cases


def _coerce_int_key_limits(d: Dict[Any, Any]) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    for k, v in (d or {}).items():
        try:
            kk = int(k)
            out[kk] = (float(v[0]), float(v[1]))
        except Exception:
            continue
    return out


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_runtime_config(config_path: Optional[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return cfg
    config_file = Path(config_path).expanduser()
    with config_file.open("r", encoding="utf-8") as f:
        if config_file.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise ValueError("YAML configs require PyYAML. Install dependencies from requirements.txt.")
            user_cfg = yaml.safe_load(f)
        else:
            user_cfg = json.load(f)
    if not isinstance(user_cfg, dict):
        raise ValueError("Comparison config must be an object at top level.")
    cfg = _deep_update(cfg, user_cfg)
    return cfg


def load_validation_dict(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def auc_of_curve_float(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        return float(y.sum())
    x = np.arange(1, y.size + 1, dtype=float)
    return float(np.trapezoid(y, x))


def align_curve_float(y: np.ndarray, force_k: Optional[int] = None) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if force_k is None:
        return y
    return y[: min(force_k, y.size)]


def build_summary_row(case: CaseInfo, out_val: Dict[str, Any]) -> Dict[str, Any]:
    y = align_curve_float(
        out_val.get("cum_recovered_frac_total", np.array([], dtype=float)),
        FORCE_K,
    )

    row = dict(
        deposit_1based=case.deposit_1based,
        method_name=case.method_name,
        analysis_type=case.analysis_type,
        selected_variable=_norm_varname(case.selected_variable),
        variable_1=case.variable_1,
        variable_2=case.variable_2,
        rotation_deg=case.rotation_deg,
        min_cover=case.min_cover,
        k_pcs=case.k_pcs if case.k_pcs is not None else out_val.get("k_pcs", np.nan),
        pkl_path=case.pkl_path,
        case_dir=case.case_dir,
        K_eval=int(len(y)),
        recovered_frac_end=float(y[-1]) if len(y) else 0.0,
        auc_recovery=auc_of_curve_float(y),
    )

    for kk in [25, 50, 100, 250]:
        row[f"recovery_at_{kk}"] = float(y[kk - 1]) if len(y) >= kk else np.nan

    return row


def collect_recovery_for_group(
    cases: List[CaseInfo],
    *,
    method_name: Optional[str] = None,
    analysis_type: Optional[str] = None,
    selected_variable: Optional[str] = None,
    min_cover: Optional[float] = DEFAULT_MINCOVER,
    best_kpcs_by_dep: Optional[Dict[int, int]] = None,
    best_kpcs_column: str = "k_pcs",
    case_k_value: Optional[int] = None,
    case_k_column: str = "k_pcs",
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]]]:
    curves_by_dep: Dict[int, np.ndarray] = {}
    outvals_by_dep: Dict[int, Dict[str, Any]] = {}

    for c in cases:
        if method_name is not None and c.method_name != method_name:
            continue
        if analysis_type is not None and c.analysis_type != analysis_type:
            continue
        if min_cover is not None and c.min_cover is not None:
            if abs(c.min_cover - min_cover) > 1e-9:
                continue
        if c.analysis_type == "Uni" and selected_variable is not None:
            if _norm_varname(c.selected_variable) != _norm_varname(selected_variable):
                continue

        out_val = load_validation_dict(c.pkl_path)
        if best_kpcs_by_dep is not None and c.method_name == "Spatial_PCA":
            target_k = best_kpcs_by_dep.get(c.deposit_1based, None)
            if best_kpcs_column in out_val:
                case_k = out_val.get(best_kpcs_column, None)
            elif best_kpcs_column == "k_pcs":
                case_k = c.k_pcs if c.k_pcs is not None else out_val.get("k_pcs", None)
            else:
                case_k = out_val.get("k_pcs", None)
            if target_k is not None and case_k is not None and int(case_k) != int(target_k):
                continue
        if case_k_value is not None and c.method_name == "Spatial_PCA":
            if case_k_column in out_val:
                case_k = out_val.get(case_k_column, None)
            elif case_k_column == "k_pcs":
                case_k = c.k_pcs if c.k_pcs is not None else out_val.get("k_pcs", None)
            else:
                case_k = out_val.get(case_k_column, None)
            if case_k is None or int(case_k) != int(case_k_value):
                continue
        y = align_curve_float(out_val.get("cum_recovered_frac_total", []), FORCE_K)
        if c.deposit_1based not in curves_by_dep and y.size > 0:
            curves_by_dep[c.deposit_1based] = y
            outvals_by_dep[c.deposit_1based] = out_val

    return curves_by_dep, outvals_by_dep


def stack_curves_float(curves_by_dep: Dict[int, np.ndarray]) -> np.ndarray:
    if not curves_by_dep:
        return np.empty((0, 0), dtype=float)

    curves = list(curves_by_dep.values())
    k_common = min(len(y) for y in curves if len(y) > 0)
    if k_common <= 0:
        return np.empty((len(curves), 0), dtype=float)

    return np.vstack([y[:k_common] for y in curves])


def aggregate_rank_event_counts(
    outvals_by_dep: Dict[int, Dict[str, Any]],
    k_common: int,
) -> Tuple[np.ndarray, np.ndarray]:
    overlap_count = np.zeros(k_common, dtype=int)
    hit_count = np.zeros(k_common, dtype=int)

    for out_val in outvals_by_dep.values():
        overlap_by_rank = out_val.get("overlap_by_rank", {}) or {}
        hit_by_rank = out_val.get("hit_by_rank", {}) or {}

        for r in overlap_by_rank.keys():
            rr = int(r)
            if 1 <= rr <= k_common:
                overlap_count[rr - 1] += 1

        for r in hit_by_rank.keys():
            rr = int(r)
            if 1 <= rr <= k_common:
                hit_count[rr - 1] += 1

    return overlap_count, hit_count


def _plot_rank_events(
    ax: plt.Axes,
    x: np.ndarray,
    y_curve: np.ndarray,
    overlap_count: np.ndarray,
    hit_count: np.ndarray,
) -> None:
    overlap_idx = np.where(overlap_count > 0)[0]
    if overlap_idx.size:
        y_overlap = y_curve[overlap_idx]
        ax.scatter(
            x[overlap_idx],
            y_overlap,
            s=42,
            color="orange",
            edgecolor="k",
            linewidth=0.35,
            zorder=4,
        )
        for idx, yy in zip(overlap_idx, y_overlap):
            ax.annotate(
                str(int(overlap_count[idx])),
                (x[idx], yy),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color="darkorange",
            )

    hit_idx = np.where(hit_count > 0)[0]
    if hit_idx.size:
        y_hit = y_curve[hit_idx]
        ax.scatter(
            x[hit_idx],
            y_hit,
            s=38,
            color="red",
            edgecolor="k",
            linewidth=0.35,
            zorder=5,
        )
        for idx, yy in zip(hit_idx, y_hit):
            ax.annotate(
                str(int(hit_count[idx])),
                (x[idx], yy),
                textcoords="offset points",
                xytext=(0, -11),
                ha="center",
                fontsize=7,
                color="darkred",
            )


def plot_aggregate_recovery_with_event_markers(
    *,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    overlap_a: np.ndarray,
    hit_a: np.ndarray,
    overlap_b: np.ndarray,
    hit_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    outpath: str,
    band: str = "minmax",
    ymax: Optional[float] = None,
    color_a: str = "C0",
    color_b: str = "C1",
) -> None:
    plt.figure(figsize=(8.6, 5.6), dpi=150)
    ax = plt.gca()

    def _plot_group(
        arr: np.ndarray,
        overlap_count: np.ndarray,
        hit_count: np.ndarray,
        label: str,
        color: str,
    ) -> None:
        if arr.size == 0:
            return

        k_common = arr.shape[1]
        x = np.arange(1, k_common + 1)
        mean = arr.mean(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.5, label=f"{label} (mean)")

        if band == "minmax":
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, label=f"{label} (min-max)")
        elif band == "p10p90":
            lo = np.percentile(arr, 10, axis=0)
            hi = np.percentile(arr, 90, axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12, label=f"{label} (p10-p90)")
        elif band != "none":
            raise ValueError(f"Unknown band: {band}")

        _plot_rank_events(ax, x, mean, overlap_count, hit_count)

    _plot_group(arr_a, overlap_a, hit_a, label_a, color_a)
    _plot_group(arr_b, overlap_b, hit_b, label_b, color_b)

    ax.set_title(title)
    ax.set_xlabel("Prediction rank (1 = most similar)")
    ax.set_ylabel("Cumulative recovered fraction of all test-deposit area")
    ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_auc_strip_by_deposit(
    df: pd.DataFrame,
    *,
    methods_order: Tuple[str, str] = ("Raw_comparison", "Spatial_PCA"),
    method_labels: Tuple[str, str] = ("Raw TMI", "Spatial PCA TMI"),
    figsize: Tuple[float, float] = (6.4, 4.8),
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> None:
    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    deposits = sorted(df["deposit_1based"].unique())
    cmap = plt.get_cmap("tab20", max(len(deposits), 1))
    dep_to_color = {dep: cmap(i) for i, dep in enumerate(deposits)}
    rng = np.random.default_rng(42)

    for i, (method, label) in enumerate(zip(methods_order, method_labels), start=1):
        df_method = df[df["method_name"] == method]

        for dep in deposits:
            y = df_method.loc[df_method["deposit_1based"] == dep, "auc_recovery"].values
            if len(y) == 0:
                continue
            x = np.full(len(y), i, dtype=float)
            jitter = rng.uniform(-0.08, 0.08, size=len(y))
            ax.scatter(
                x + jitter,
                y,
                s=55,
                color=dep_to_color[dep],
                edgecolor="k",
                linewidth=0.4,
                alpha=0.9,
                zorder=3,
            )

        y_all = df_method["auc_recovery"].dropna().values
        if len(y_all) > 0:
            med = np.median(y_all)
            ax.hlines(med, i - 0.25, i + 0.25, colors="black", linewidth=2.8, zorder=5)

    left_method, right_method = methods_order
    for dep in deposits:
        y_left = df.loc[
            (df["deposit_1based"] == dep) & (df["method_name"] == left_method),
            "auc_recovery",
        ].values
        y_right = df.loc[
            (df["deposit_1based"] == dep) & (df["method_name"] == right_method),
            "auc_recovery",
        ].values
        if len(y_left) == 0 or len(y_right) == 0:
            continue

        ax.annotate(
            "",
            xy=(2 - 0.08, float(y_right[0])),
            xytext=(1 + 0.08, float(y_left[0])),
            arrowprops=dict(
                arrowstyle="->",
                color=dep_to_color[dep],
                lw=1.8,
                alpha=0.75,
            ),
            zorder=2,
        )

    ax.set_xticks(range(1, len(method_labels) + 1))
    ax.set_xticklabels(method_labels)
    ax.set_ylabel("Area under cumulative recovered-deposit-footprint curve")
    ax.set_title("Integrated footprint recovery by deposit (Uni, TMI)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(0.5, len(method_labels) + 0.5)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"Deposit {dep}",
            markerfacecolor=dep_to_color[dep],
            markeredgecolor="k",
            markersize=7,
        )
        for dep in deposits
    ]
    ax.legend(
        handles=legend_elements,
        title="Deposit ID",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_recovery_curves_by_group(
    *,
    curves_a: Dict[int, np.ndarray],
    curves_b: Dict[int, np.ndarray],
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
    title: str,
    outpath: str,
    ymax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(8.6, 5.6), dpi=150)
    ax = plt.gca()

    def _label_anchor(y: np.ndarray) -> Tuple[float, float]:
        if y.size == 0:
            return 1.0, 0.0
        diff_idx = np.flatnonzero(np.abs(np.diff(y)) > 1e-12)
        if diff_idx.size == 0:
            idx = y.size - 1
        else:
            idx = int(diff_idx[-1] + 1)
        return float(idx + 1), float(y[idx])

    def _plot_curves(
        curves: Dict[int, np.ndarray],
        color: str,
        label: str,
        text_dx: int,
    ) -> None:
        first = True
        for dep in sorted(curves):
            y = np.asarray(curves[dep], dtype=float)
            if y.size == 0:
                continue
            x = np.arange(1, y.size + 1)
            ax.plot(
                x,
                y,
                color=color,
                alpha=0.45,
                linewidth=1.8,
                label=label if first else None,
            )
            x_anchor, y_anchor = _label_anchor(y)
            y_offset = 7 if dep % 2 else -7
            ax.annotate(
                str(dep),
                (x_anchor, y_anchor),
                xytext=(text_dx, y_offset),
                textcoords="offset points",
                va="center",
                ha="left" if text_dx > 0 else "right",
                fontsize=9,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8),
            )
            first = False

    _plot_curves(curves_a, color_a, label_a, text_dx=-8)
    _plot_curves(curves_b, color_b, label_b, text_dx=8)

    ax.set_title(title)
    ax.set_xlabel("Prediction rank (1 = most similar)")
    ax.set_ylabel("Cumulative recovered fraction of all test-deposit area")
    ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
    ax.set_xlim(1, FORCE_K)
    ax.grid(alpha=0.25)
    group_handles = [
        Line2D([0], [0], color=color_a, lw=2.2, label=label_a),
        Line2D([0], [0], color=color_b, lw=2.2, label=label_b),
    ]
    ax.legend(handles=group_handles, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def _as_dep_label_list(dep_rows: Any) -> str:
    if dep_rows is None:
        return ""
    vals = []
    for d in dep_rows:
        try:
            vals.append(int(d) + 1)  # stored as 0-based rows in validate_topk_hits
        except Exception:
            continue
    vals = sorted(set(vals))
    return ",".join(str(v) for v in vals)


def _plot_rank_events_single_curve(
    ax: plt.Axes,
    y_curve: np.ndarray,
    out_val: Dict[str, Any],
) -> None:
    if y_curve.size == 0:
        return

    k_common = y_curve.size
    x = np.arange(1, k_common + 1)
    overlap_by_rank = out_val.get("overlap_by_rank", {}) or {}
    hit_by_rank = out_val.get("hit_by_rank", {}) or {}

    overlap_ranks = sorted({int(r) for r in overlap_by_rank.keys() if 1 <= int(r) <= k_common})
    if overlap_ranks:
        idx = np.array(overlap_ranks, dtype=int) - 1
        ax.scatter(
            x[idx],
            y_curve[idx],
            s=24,
            color="gold",
            edgecolor="k",
            linewidth=0.3,
            zorder=4,
        )

    hit_ranks = sorted({int(r) for r in hit_by_rank.keys() if 1 <= int(r) <= k_common})
    if hit_ranks:
        idx = np.array(hit_ranks, dtype=int) - 1
        ax.scatter(
            x[idx],
            y_curve[idx],
            s=24,
            color="red",
            edgecolor="k",
            linewidth=0.3,
            zorder=5,
        )
        for r in hit_ranks:
            dep_rows = hit_by_rank.get(r, []) or []
            dep_txt = _as_dep_label_list(dep_rows)
            if dep_txt:
                ax.annotate(
                    dep_txt,
                    (r, y_curve[r - 1]),
                    textcoords="offset points",
                    xytext=(-10, 5),
                    fontsize=9,
                    fontweight="bold",
                    color="darkred",
                )


def _label_with_best_k(
    base_label: str,
    out_val: Dict[str, Any],
    *,
    include_k_for_raw: bool = False,
    forced_k_value: Optional[int] = None,
    forced_k_name: Optional[str] = None,
) -> str:
    method = str(out_val.get("method_name", "") or "")
    if method != "Spatial_PCA" and not include_k_for_raw:
        return base_label

    if method == "Spatial_PCA" and forced_k_value is not None:
        k_name = str(forced_k_name or "k").strip()
        if k_name == "k_pcs_fused":
            k_name = "k_fused"
        elif k_name == "k_pcs":
            k_name = "k"
        return f"{base_label} (best {k_name}={int(forced_k_value)})"

    ranking_mode = str(out_val.get("ranking_mode", "") or "")
    k_main = out_val.get("k_pcs", None)
    k1 = out_val.get("k_pcs_var1", None)
    k2 = out_val.get("k_pcs_var2", None)
    kf = out_val.get("k_pcs_fused", None)

    if method == "Raw_comparison":
        return f"{base_label} (no PCA k)"

    if ranking_mode == "two_stage_pca_fusion":
        if kf is not None:
            return f"{base_label} (best k_fused={int(kf)})"
        return base_label

    if ranking_mode == "separate_pca_fusion":
        parts = []
        if k1 is not None:
            parts.append(f"k1={int(k1)}")
        if k2 is not None:
            parts.append(f"k2={int(k2)}")
        return f"{base_label} (best {'; '.join(parts)})" if parts else base_label

    if k_main is not None:
        try:
            return f"{base_label} (best k={int(k_main)})"
        except Exception:
            return f"{base_label} (best k={k_main})"
    return base_label


def plot_top_gain_subplots_two_groups(
    *,
    curves_a: Dict[int, np.ndarray],
    curves_b: Dict[int, np.ndarray],
    outvals_a: Dict[int, Dict[str, Any]],
    outvals_b: Dict[int, Dict[str, Any]],
    label_a: str,
    label_b: str,
    color_a: str = "orange",
    color_b: str = "blue",
    outpath: str,
    top_n: int = 4,
    ymax: Optional[float] = DEFAULT_YMAX,
    selected_deposits: Optional[List[int]] = None,
    suptitle: Optional[str] = None,
) -> None:
    shared = sorted(set(curves_a) & set(curves_b))
    if not shared:
        return

    rows = []
    for dep in shared:
        auc_a = auc_of_curve_float(curves_a[dep])
        auc_b = auc_of_curve_float(curves_b[dep])
        rows.append((dep, auc_b - auc_a, auc_a, auc_b))

    rows.sort(key=lambda t: t[1], reverse=True)
    if selected_deposits:
        selected_set = {int(d) for d in selected_deposits}
        picked = [r for r in rows if int(r[0]) in selected_set]
    else:
        picked = rows[:top_n]
    if len(picked) == 0:
        return

    n_panels = len(picked)
    if n_panels == 1:
        ncols = 1
        nrows = 1
        figsize = (6.8, 5.2)
    else:
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        figsize = (12, 5.2 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, squeeze=False)
    flat_axes = axes.ravel()

    for ax, (dep, delta_auc, auc_a, auc_b) in zip(flat_axes, picked):
        y_a = np.asarray(curves_a[dep], dtype=float)
        y_b = np.asarray(curves_b[dep], dtype=float)
        k_common = min(y_a.size, y_b.size)
        if k_common <= 0:
            continue
        y_a = y_a[:k_common]
        y_b = y_b[:k_common]
        x = np.arange(1, k_common + 1)
        lbl_a = _label_with_best_k(label_a, outvals_a.get(dep, {}))
        lbl_b = _label_with_best_k(label_b, outvals_b.get(dep, {}))
        ax.plot(x, y_a, color=color_a, linewidth=2.1, label=lbl_a)
        ax.plot(x, y_b, color=color_b, linewidth=2.1, label=lbl_b)
        _plot_rank_events_single_curve(ax, y_a, outvals_a.get(dep, {}))
        _plot_rank_events_single_curve(ax, y_b, outvals_b.get(dep, {}))

        ax.set_title(f"Training Deposit #{dep}", fontsize=10)
        ax.set_xlabel("Prediction rank")
        ax.set_ylabel("Cumulative recovered fraction")
        ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
        ax.grid(alpha=0.25)
        event_handles = [
            Line2D([0], [0], marker="o", color="w", label="Overlap event (yellow)",
                   markerfacecolor="gold", markeredgecolor="k", markersize=6),
            Line2D([0], [0], marker="o", color="w", label="Hit event (red)",
                   markerfacecolor="red", markeredgecolor="k", markersize=6),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + event_handles, labels + [h.get_label() for h in event_handles],
                  fontsize=8, frameon=True, loc="upper left")

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    if suptitle is None:
        suptitle = (
            f"Top {top_n} deposits with largest gain: {label_b} over {label_a}"
            if not selected_deposits
            else f"Selected deposits: {label_b} vs {label_a}"
        )
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_top_gain_subplots_three_groups(
    *,
    curves_a: Dict[int, np.ndarray],
    curves_b: Dict[int, np.ndarray],
    curves_c: Dict[int, np.ndarray],
    outvals_a: Dict[int, Dict[str, Any]],
    outvals_b: Dict[int, Dict[str, Any]],
    outvals_c: Dict[int, Dict[str, Any]],
    label_a: str,
    label_b: str,
    label_c: str,
    color_a: str = "purple",
    color_b: str = "blue",
    color_c: str = "green",
    outpath: str,
    top_n: int = 4,
    ymax: Optional[float] = DEFAULT_YMAX,
    selected_deposits: Optional[List[int]] = None,
    suptitle: Optional[str] = None,
) -> None:
    # Rank deposits by gain between SPCA groups: group_c - group_b.
    shared = sorted(set(curves_a) & set(curves_b) & set(curves_c))
    if not shared:
        return

    rows = []
    for dep in shared:
        auc_a = auc_of_curve_float(curves_a[dep])
        auc_b = auc_of_curve_float(curves_b[dep])
        auc_c = auc_of_curve_float(curves_c[dep])
        rows.append((dep, auc_c - auc_b, auc_a, auc_b, auc_c))

    rows.sort(key=lambda t: t[1], reverse=True)
    if selected_deposits:
        selected_set = {int(d) for d in selected_deposits}
        picked = [r for r in rows if int(r[0]) in selected_set]
    else:
        picked = rows[:top_n]
    if len(picked) == 0:
        return

    n_panels = len(picked)
    if n_panels == 1:
        ncols = 1
        nrows = 1
        figsize = (6.8, 5.2)
    else:
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        figsize = (12, 5.2 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, squeeze=False)
    flat_axes = axes.ravel()

    for ax, (dep, delta_bc, auc_a, auc_b, auc_c) in zip(flat_axes, picked):
        y_a = np.asarray(curves_a[dep], dtype=float)
        y_b = np.asarray(curves_b[dep], dtype=float)
        y_c = np.asarray(curves_c[dep], dtype=float)
        k_common = min(y_a.size, y_b.size, y_c.size)
        if k_common <= 0:
            continue
        y_a = y_a[:k_common]
        y_b = y_b[:k_common]
        y_c = y_c[:k_common]
        x = np.arange(1, k_common + 1)
        lbl_a = _label_with_best_k(label_a, outvals_a.get(dep, {}))
        lbl_b = _label_with_best_k(label_b, outvals_b.get(dep, {}))
        lbl_c = _label_with_best_k(label_c, outvals_c.get(dep, {}))
        ax.plot(x, y_a, color=color_a, linewidth=2.0, label=lbl_a)
        ax.plot(x, y_b, color=color_b, linewidth=2.0, label=lbl_b)
        ax.plot(x, y_c, color=color_c, linewidth=2.0, label=lbl_c)

        _plot_rank_events_single_curve(ax, y_a, outvals_a.get(dep, {}))
        _plot_rank_events_single_curve(ax, y_b, outvals_b.get(dep, {}))
        _plot_rank_events_single_curve(ax, y_c, outvals_c.get(dep, {}))

        ax.set_title(f"Training Deposit #{dep}", fontsize=10)
        ax.set_xlabel("Prediction rank")
        ax.set_ylabel("Cumulative recovered fraction")
        ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
        ax.grid(alpha=0.25)

        event_handles = [
            Line2D([0], [0], marker="o", color="w", label="Overlap event (yellow)",
                   markerfacecolor="gold", markeredgecolor="k", markersize=6),
            Line2D([0], [0], marker="o", color="w", label="Hit event (red)",
                   markerfacecolor="red", markeredgecolor="k", markersize=6),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + event_handles, labels + [h.get_label() for h in event_handles],
                  fontsize=8, frameon=True, loc="upper left")

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    if suptitle is None:
        suptitle = (
            f"Top {top_n} deposits by largest gain: {label_c} over {label_b} (with {label_a})"
            if not selected_deposits
            else f"Selected deposits: {label_a}, {label_b}, {label_c}"
        )
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_aggregate_recovery_with_event_markers_n_groups(
    *,
    arrs: List[np.ndarray],
    overlaps: List[np.ndarray],
    hits: List[np.ndarray],
    labels: List[str],
    colors: List[str],
    title: str,
    outpath: str,
    band: str = "minmax",
    ymax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(9.4, 5.9), dpi=150)
    ax = plt.gca()

    def _plot_group(
        arr: np.ndarray,
        overlap_count: np.ndarray,
        hit_count: np.ndarray,
        label: str,
        color: str,
    ) -> None:
        if arr.size == 0:
            return
        k_common = arr.shape[1]
        x = np.arange(1, k_common + 1)
        mean = arr.mean(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.3, label=f"{label} (mean)")
        if band == "minmax":
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=color, alpha=0.10, label=f"{label} (min-max)")
        elif band == "p10p90":
            ax.fill_between(
                x,
                np.percentile(arr, 10, axis=0),
                np.percentile(arr, 90, axis=0),
                color=color,
                alpha=0.10,
                label=f"{label} (p10-p90)",
            )
        elif band != "none":
            raise ValueError(f"Unknown band: {band}")
        _plot_rank_events(ax, x, mean, overlap_count, hit_count)

    for arr, ov, hh, label, color in zip(arrs, overlaps, hits, labels, colors):
        _plot_group(arr, ov, hh, label, color)

    ax.set_title(title)
    ax.set_xlabel("Prediction rank (1 = most similar)")
    ax.set_ylabel("Cumulative recovered fraction of all test-deposit area")
    ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_all_recovery_curves_n_groups(
    *,
    curves_list: List[Dict[int, np.ndarray]],
    labels: List[str],
    colors: List[str],
    title: str,
    outpath: str,
    ymax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(9.4, 5.9), dpi=150)
    ax = plt.gca()

    def _label_anchor(y: np.ndarray) -> Tuple[float, float]:
        if y.size == 0:
            return 1.0, 0.0
        diff_idx = np.flatnonzero(np.abs(np.diff(y)) > 1e-12)
        idx = int(diff_idx[-1] + 1) if diff_idx.size else y.size - 1
        return float(idx + 1), float(y[idx])

    n_groups = len(curves_list)
    text_offsets = np.linspace(-14, 14, num=max(n_groups, 2)).astype(int)
    for gi, (curves, label, color) in enumerate(zip(curves_list, labels, colors)):
        first = True
        for dep in sorted(curves):
            y = np.asarray(curves[dep], dtype=float)
            if y.size == 0:
                continue
            x = np.arange(1, y.size + 1)
            ax.plot(x, y, color=color, alpha=0.45, linewidth=1.5, label=label if first else None)
            x_anchor, y_anchor = _label_anchor(y)
            y_offset = 7 if dep % 2 else -7
            ax.annotate(
                str(dep),
                (x_anchor, y_anchor),
                xytext=(int(text_offsets[gi]), y_offset),
                textcoords="offset points",
                va="center",
                ha="left" if text_offsets[gi] > 0 else "right",
                fontsize=8.2,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.8),
            )
            first = False

    ax.set_title(title)
    ax.set_xlabel("Prediction rank (1 = most similar)")
    ax.set_ylabel("Cumulative recovered fraction of all test-deposit area")
    ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
    ax.set_xlim(1, FORCE_K)
    ax.grid(alpha=0.25)
    handles = [Line2D([0], [0], color=c, lw=2.2, label=l) for l, c in zip(labels, colors)]
    ax.legend(handles=handles, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_top_gain_subplots_n_groups(
    *,
    curves_list: List[Dict[int, np.ndarray]],
    outvals_list: List[Dict[int, Dict[str, Any]]],
    labels: List[str],
    colors: List[str],
    gain_base_idx: int,
    gain_target_idx: int,
    outpath: str,
    top_n: int = 4,
    ymax: Optional[float] = DEFAULT_YMAX,
    selected_deposits: Optional[List[int]] = None,
    suptitle: Optional[str] = None,
    best_k_maps: Optional[List[Optional[Dict[int, int]]]] = None,
    best_k_columns: Optional[List[str]] = None,
) -> None:
    if len(curves_list) < 2:
        return
    n_groups = len(curves_list)
    if not (0 <= gain_base_idx < n_groups and 0 <= gain_target_idx < n_groups):
        raise ValueError("Invalid gain group indices.")
    if gain_base_idx == gain_target_idx:
        raise ValueError("gain_base_idx and gain_target_idx must be different.")

    shared = set(curves_list[gain_base_idx].keys()) & set(curves_list[gain_target_idx].keys())
    if not shared:
        return

    rows = []
    for dep in sorted(shared):
        auc_base = auc_of_curve_float(curves_list[gain_base_idx][dep])
        auc_tgt = auc_of_curve_float(curves_list[gain_target_idx][dep])
        rows.append((dep, auc_tgt - auc_base, auc_base, auc_tgt))

    rows.sort(key=lambda t: t[1], reverse=True)
    if selected_deposits:
        selected_set = {int(d) for d in selected_deposits}
        picked = [r for r in rows if int(r[0]) in selected_set]
    else:
        picked = rows[:top_n]
    if len(picked) == 0:
        return

    n_panels = len(picked)
    if n_panels == 1:
        ncols = 1
        nrows = 1
        figsize = (6.8, 5.2)
    else:
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        figsize = (12, 5.2 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=150, squeeze=False)
    flat_axes = axes.ravel()

    for ax, (dep, delta_auc, _, _) in zip(flat_axes, picked):
        k_sizes = []
        y_by_group = []
        for curves in curves_list:
            y = np.asarray(curves.get(dep, np.array([], dtype=float)), dtype=float)
            y_by_group.append(y)
            if y.size > 0:
                k_sizes.append(y.size)
        if not k_sizes:
            continue
        k_common = min(k_sizes)
        x = np.arange(1, k_common + 1)
        missing_labels = []

        for gi in range(n_groups):
            if y_by_group[gi].size == 0:
                missing_labels.append(labels[gi])
                continue
            y = y_by_group[gi][:k_common]
            outv = outvals_list[gi].get(dep, {})
            forced_k = None
            forced_k_name = None
            if best_k_maps is not None and gi < len(best_k_maps):
                gmap = best_k_maps[gi]
                if isinstance(gmap, dict) and int(dep) in gmap:
                    forced_k = int(gmap[int(dep)])
                    if best_k_columns is not None and gi < len(best_k_columns):
                        forced_k_name = best_k_columns[gi]
            lbl = _label_with_best_k(
                labels[gi],
                outv,
                forced_k_value=forced_k,
                forced_k_name=forced_k_name,
            )
            ax.plot(x, y, color=colors[gi], linewidth=2.0, label=lbl)
            _plot_rank_events_single_curve(ax, y, outv)

        if missing_labels:
            ax.text(
                0.99,
                0.02,
                "No curve: " + ", ".join(missing_labels),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="dimgray",
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )

        ax.set_title(f"Training Deposit #{dep}", fontsize=10)
        ax.set_xlabel("Prediction rank")
        ax.set_ylabel("Cumulative recovered fraction")
        ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
        ax.grid(alpha=0.25)
        event_handles = [
            Line2D([0], [0], marker="o", color="w", label="Overlap event (yellow)",
                   markerfacecolor="gold", markeredgecolor="k", markersize=6),
            Line2D([0], [0], marker="o", color="w", label="Hit event (red)",
                   markerfacecolor="red", markeredgecolor="k", markersize=6),
        ]
        handles, lgd_labels = ax.get_legend_handles_labels()
        ax.legend(handles + event_handles, lgd_labels + [h.get_label() for h in event_handles],
                  fontsize=8, frameon=True, loc="upper left")

    for ax in flat_axes[n_panels:]:
        ax.axis("off")

    if suptitle is None:
        suptitle = (
            f"Top {top_n} deposits by largest gain: {labels[gain_target_idx]} over {labels[gain_base_idx]}"
            if not selected_deposits
            else f"Selected deposits: gain {labels[gain_target_idx]} over {labels[gain_base_idx]}"
        )
    has_suptitle = bool(str(suptitle).strip())
    if has_suptitle:
        fig.suptitle(suptitle, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def build_top_gain_summary_n_groups(
    *,
    curves_list: List[Dict[int, np.ndarray]],
    outvals_list: List[Dict[int, Dict[str, Any]]],
    labels: List[str],
    gain_base_idx: int,
    gain_target_idx: int,
    top_n: int,
    selected_deposits: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Build a per-deposit/group summary table for the top-gain figure logic.
    Output is LONG/VERTICAL format (one row per deposit x group), including
    gain-pair metadata and delta for each deposit.
    """
    n_groups = len(curves_list)
    if n_groups < 2:
        return pd.DataFrame()
    if not (0 <= gain_base_idx < n_groups and 0 <= gain_target_idx < n_groups):
        return pd.DataFrame()

    shared = sorted(set(curves_list[gain_base_idx].keys()) & set(curves_list[gain_target_idx].keys()))
    if not shared:
        return pd.DataFrame()

    dep_rows: List[Dict[str, Any]] = []
    for dep in shared:
        group_metrics: List[Dict[str, Any]] = []
        auc_by_group: List[float] = []
        for gi in range(n_groups):
            y = np.asarray(curves_list[gi].get(dep, np.array([], dtype=float)), dtype=float)
            auc_val = auc_of_curve_float(y) if y.size else np.nan
            end_val = float(y[-1]) if y.size else np.nan
            hits = outvals_list[gi].get(dep, {}).get("hit_by_rank", {}) or {}
            hit_ranks = sorted(int(r) for r in hits.keys())
            group_metrics.append(
                {
                    "group_index": int(gi + 1),
                    "group_label": labels[gi],
                    "auc_recovery": float(auc_val) if np.isfinite(auc_val) else np.nan,
                    "recovery_end": end_val,
                    "red_points_count": int(len(hit_ranks)),
                    "first_red_rank": float(hit_ranks[0]) if hit_ranks else np.nan,
                }
            )
            auc_by_group.append(float(auc_val) if np.isfinite(auc_val) else np.nan)

        auc_base = auc_by_group[gain_base_idx]
        auc_target = auc_by_group[gain_target_idx]
        delta_auc = (
            float(auc_target - auc_base)
            if np.isfinite(auc_target) and np.isfinite(auc_base)
            else np.nan
        )
        dep_rows.append(
            {
                "deposit_1based": int(dep),
                "gain_base_group_index": int(gain_base_idx + 1),
                "gain_target_group_index": int(gain_target_idx + 1),
                "gain_base_label": labels[gain_base_idx],
                "gain_target_label": labels[gain_target_idx],
                "delta_auc_target_minus_base": delta_auc,
                "group_metrics": group_metrics,
            }
        )

    dep_df = pd.DataFrame(dep_rows).sort_values("delta_auc_target_minus_base", ascending=False).reset_index(drop=True)
    if dep_df.empty:
        return dep_df

    if selected_deposits:
        selected_set = {int(d) for d in selected_deposits}
        plotted_set = set(int(d) for d in dep_df["deposit_1based"].tolist() if int(d) in selected_set)
    else:
        plotted_set = set(int(d) for d in dep_df["deposit_1based"].head(int(top_n)).tolist())

    long_rows: List[Dict[str, Any]] = []
    for _, drow in dep_df.iterrows():
        dep = int(drow["deposit_1based"])
        for gm in drow["group_metrics"]:
            long_rows.append(
                {
                    "deposit_1based": dep,
                    "group_index": int(gm["group_index"]),
                    "group_label": gm["group_label"],
                    "auc_recovery": gm["auc_recovery"],
                    "recovery_end": gm["recovery_end"],
                    "red_points_count": gm["red_points_count"],
                    "first_red_rank": gm["first_red_rank"],
                    "gain_base_group_index": int(drow["gain_base_group_index"]),
                    "gain_target_group_index": int(drow["gain_target_group_index"]),
                    "gain_base_label": drow["gain_base_label"],
                    "gain_target_label": drow["gain_target_label"],
                    "delta_auc_target_minus_base": drow["delta_auc_target_minus_base"],
                    "plotted_in_top_gain_figure": dep in plotted_set,
                }
            )

    out_df = pd.DataFrame(long_rows)
    if out_df.empty:
        return out_df
    out_df = out_df.sort_values(
        ["delta_auc_target_minus_base", "deposit_1based", "group_index"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return out_df


def plot_optimal_k_vs_delta_auc(
    *,
    curves_raw: Dict[int, np.ndarray],
    curves_spca: Dict[int, np.ndarray],
    best_kpcs_by_dep: Dict[int, int],
    outpath: str,
) -> None:
    shared = sorted(set(curves_raw) & set(curves_spca))
    if not shared:
        return

    rows = []
    for dep in shared:
        k_opt = best_kpcs_by_dep.get(dep, None)
        if k_opt is None:
            continue
        auc_raw = auc_of_curve_float(curves_raw[dep])
        auc_spca = auc_of_curve_float(curves_spca[dep])
        rows.append((dep, int(k_opt), float(auc_spca - auc_raw)))

    if not rows:
        return

    plt.figure(figsize=(7.0, 5.0), dpi=150)
    ax = plt.gca()

    for dep, k_opt, delta_auc in rows:
        ax.scatter(
            k_opt,
            delta_auc,
            s=85,
            color="steelblue",
            edgecolor="k",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )
        ax.annotate(
            str(dep),
            (k_opt, delta_auc),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35")
    ax.set_xlabel("Optimal k (multicriteria)")
    ax.set_ylabel("ΔAUC = AUC(sPCA best k) - AUC(Raw)")
    ax.set_title("Optimal k vs ΔAUC across deposits")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def collect_deposit_windows_and_shapes_from_pkls(
    cases: List[CaseInfo],
    *,
    preferred_method_name: str = "Raw_comparison",
    preferred_analysis_type: Optional[str] = None,
    preferred_selected_variable: Optional[str] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, int]]]:
    preferred_var = _norm_varname(preferred_selected_variable)

    preferred_cases = [
        c
        for c in cases
        if c.method_name == preferred_method_name
        and (preferred_analysis_type is None or c.analysis_type == preferred_analysis_type)
        and (preferred_var is None or _norm_varname(c.selected_variable) == preferred_var)
    ]
    fallback_method_cases = [
        c
        for c in cases
        if c.method_name == preferred_method_name
        and (preferred_analysis_type is None or c.analysis_type == preferred_analysis_type)
    ]
    fallback_analysis_cases = [
        c for c in cases if preferred_analysis_type is None or c.analysis_type == preferred_analysis_type
    ]
    fallback_cases = [c for c in cases if c.method_name == preferred_method_name]
    source_groups = [
        ("preferred", preferred_cases),
        ("fallback_method", fallback_method_cases),
        ("fallback_analysis", fallback_analysis_cases),
        ("fallback_raw_method", fallback_cases),
        ("all_cases", list(cases)),
    ]

    last_group_name = "none"
    for group_name, source_cases in source_groups:
        if not source_cases:
            continue

        deposit_windows: Dict[int, np.ndarray] = {}
        deposit_shapes: Dict[int, Tuple[int, int]] = {}

        for c in source_cases:
            dep = int(c.deposit_1based)
            if dep in deposit_windows:
                continue

            out_val = load_validation_dict(c.pkl_path)
            raw_diag = out_val.get("raw_diagnostics", {}) or {}

            # For Multi, prioritize concatenated feature vectors (var1+var2).
            if c.analysis_type == "Multi":
                deposit_vector = out_val.get("x_dep_vec", None)
                if deposit_vector is None:
                    deposit_vector = out_val.get("X_dep_vec", None)
                if deposit_vector is None:
                    deposit_vector = out_val.get("deposit_vector", None)
                if deposit_vector is None:
                    deposit_vector = raw_diag.get("deposit_vector", None)
            else:
                # For Uni, keep legacy behavior first.
                deposit_vector = raw_diag.get("deposit_vector", None)
                if deposit_vector is None:
                    deposit_vector = out_val.get("deposit_vector", None)
                if deposit_vector is None:
                    deposit_vector = out_val.get("x_dep_vec", None)
                if deposit_vector is None:
                    deposit_vector = out_val.get("X_dep_vec", None)

            if deposit_vector is None:
                continue

            window_shape = out_val.get("window_shape", None)
            if window_shape is None:
                contrib_map = raw_diag.get("contrib_map", None)
                if contrib_map is not None:
                    window_shape = np.asarray(contrib_map).shape

            if window_shape is None:
                continue

            deposit_windows[dep] = np.asarray(deposit_vector)
            deposit_shapes[dep] = tuple(window_shape)

        if deposit_windows:
            return deposit_windows, deposit_shapes

        last_group_name = group_name

    raise ValueError(
        "No deposit vectors found in validation pickles. "
        f"Last attempted source group: {last_group_name}"
    )


def make_dep_to_color(deposits: List[int]) -> Dict[int, Any]:
    # Use a fixed global palette keyed by deposit ID so colors stay consistent
    # across Uni and Multi runs (even when only a subset of deposits is plotted).
    req = sorted(int(d) for d in deposits)
    if not req:
        return {}

    # Prefer a fixed canonical order (independent of per-run config) so
    # colors are stable across Uni and Multi runs.
    base_order = sorted(int(d) for d in DEPOSIT_LIMITS_TMI.keys())
    # Keep any extra IDs (if present) appended in sorted order.
    extra = [d for d in req if d not in set(base_order)]
    full_order = base_order + extra

    cmap = plt.get_cmap("tab20", max(len(full_order), 1))
    full_map = {dep: cmap(i % 20) for i, dep in enumerate(full_order)}
    return {dep: full_map[dep] for dep in req}


def _center_sn_profile(window2d: np.ndarray, band_halfwidth: int = 2) -> np.ndarray:
    h, w = window2d.shape
    col0 = w // 2
    c0 = max(col0 - band_halfwidth, 0)
    c1 = min(col0 + band_halfwidth + 1, w)
    return window2d[:, c0:c1].mean(axis=1)


def _smooth_profile(y: np.ndarray, *, method: Optional[str], window: int) -> np.ndarray:
    if method is None:
        return y
    m = str(method).strip().lower()
    if m in {"none", "off", ""}:
        return y
    if m != "moving_average":
        raise ValueError(f"Unsupported smoothing method: {method}")
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w <= 1 or y.size < 3:
        return y
    if w > y.size:
        w = y.size if y.size % 2 == 1 else max(1, y.size - 1)
    if w <= 1:
        return y
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(y, kernel, mode="same")


def plot_all_sn_profiles_centered(
    deposit_windows: Dict[int, np.ndarray],
    deposit_shapes: Dict[int, Tuple[int, int]],
    *,
    dep_to_color: Optional[Dict[int, Any]] = None,
    resolution_m: float = 200,
    band_halfwidth: int = 2,
    x_center_mode: str = "midpoint",
    y_center: bool = True,
    y_scale: Optional[str] = None,
    smoothing_method: Optional[str] = None,
    smoothing_window: int = 5,
    variable_index: int = 0,
    variable_label: str = "TMI",
    title: str = "Centered S-N TMI Profiles (variable window sizes)",
    save_path: Optional[str] = None,
) -> None:
    deposits = sorted(deposit_windows.keys())
    if dep_to_color is None:
        dep_to_color = make_dep_to_color(deposits)

    plt.figure(figsize=(7.2, 5.2), dpi=160)

    for dep_id in deposits:
        if dep_id not in deposit_shapes:
            raise KeyError(f"Missing deposit_shapes for deposit {dep_id}")

        h, w = deposit_shapes[dep_id]
        x_dep = np.asarray(deposit_windows[dep_id], dtype=float).ravel()
        n_pix = h * w
        if x_dep.size == n_pix:
            if variable_index != 0:
                continue
            window = x_dep.reshape(h, w)
        elif x_dep.size >= (variable_index + 1) * n_pix:
            i0 = variable_index * n_pix
            i1 = i0 + n_pix
            window = x_dep[i0:i1].reshape(h, w)
        else:
            raise ValueError(
                f"Deposit {dep_id}: deposit_vector size={x_dep.size} is incompatible with "
                f"shape={(h, w)} and variable_index={variable_index}"
            )
        profile = _center_sn_profile(window, band_halfwidth=band_halfwidth)

        x_km = np.arange(h) * (resolution_m / 1000.0)
        if x_center_mode == "midpoint":
            x0 = x_km[h // 2]
        elif x_center_mode == "peak":
            x0 = x_km[int(np.argmax(np.abs(profile)))]
        else:
            raise ValueError("x_center_mode must be 'midpoint' or 'peak'")

        y = profile.astype(float)
        if y_center:
            y = y - np.mean(y)

        if y_scale is not None:
            if y_scale == "std":
                scale = np.std(y)
            elif y_scale == "peak2peak":
                scale = np.ptp(y)
            else:
                raise ValueError("y_scale must be None, 'std', or 'peak2peak'")
            if scale > 0:
                y = y / scale

        y = _smooth_profile(y, method=smoothing_method, window=smoothing_window)

        plt.plot(
            x_km - x0,
            y,
            lw=2,
            color=dep_to_color.get(dep_id, "k"),
            label=str(dep_id),
            alpha=0.95,
        )

    plt.axvline(0, lw=1, alpha=0.5)
    plt.axhline(0, lw=1, alpha=0.5)
    plt.xlabel("Distance along S-N line (km), centered at window center")
    plt.ylabel(f"{variable_label} (centered)" + (f", scaled={y_scale}" if y_scale else ""))
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(title="Deposit", fontsize=8, ncols=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_sn_profiles_centered_multi_subplots(
    deposit_windows: Dict[int, np.ndarray],
    deposit_shapes: Dict[int, Tuple[int, int]],
    *,
    dep_to_color: Optional[Dict[int, Any]] = None,
    resolution_m: float = 200,
    band_halfwidth: int = 2,
    x_center_mode: str = "midpoint",
    y_center: bool = True,
    y_scale: Optional[str] = None,
    smoothing_method: Optional[str] = None,
    smoothing_window: int = 5,
    var1_label: str = "TMI",
    var2_label: str = "Radiometric_U",
    title: str = "Centered S-N Profiles (variable window sizes)",
    save_path: Optional[str] = None,
) -> None:
    deposits = sorted(deposit_windows.keys())
    if not deposits:
        return
    if dep_to_color is None:
        dep_to_color = make_dep_to_color(deposits)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), dpi=160, sharex=False, sharey=False)

    for var_idx, (ax, var_label) in enumerate(zip(axes, (var1_label, var2_label))):
        for dep_id in deposits:
            if dep_id not in deposit_shapes:
                continue

            h, w = deposit_shapes[dep_id]
            x_dep = np.asarray(deposit_windows[dep_id], dtype=float).ravel()
            n_pix = h * w

            # Need multi vector for side-by-side plot.
            if x_dep.size < (var_idx + 1) * n_pix:
                continue

            i0 = var_idx * n_pix
            i1 = i0 + n_pix
            window = x_dep[i0:i1].reshape(h, w)
            profile = _center_sn_profile(window, band_halfwidth=band_halfwidth)

            x_km = np.arange(h) * (resolution_m / 1000.0)
            if x_center_mode == "midpoint":
                x0 = x_km[h // 2]
            elif x_center_mode == "peak":
                x0 = x_km[int(np.argmax(np.abs(profile)))]
            else:
                raise ValueError("x_center_mode must be 'midpoint' or 'peak'")

            y = profile.astype(float)
            if y_center:
                y = y - np.mean(y)

            if y_scale is not None:
                if y_scale == "std":
                    scale = np.std(y)
                elif y_scale == "peak2peak":
                    scale = np.ptp(y)
                else:
                    raise ValueError("y_scale must be None, 'std', or 'peak2peak'")
                if scale > 0:
                    y = y / scale

            y = _smooth_profile(y, method=smoothing_method, window=smoothing_window)

            ax.plot(
                x_km - x0,
                y,
                lw=2,
                color=dep_to_color.get(dep_id, "k"),
                label=str(dep_id),
                alpha=0.95,
            )

        ax.axvline(0, lw=1, alpha=0.5)
        ax.axhline(0, lw=1, alpha=0.5)
        ax.set_title(var_label)
        ax.set_xlabel("Distance along S-N line (km), centered at window center")
        ax.set_ylabel(var_label + " (centered)" + (f", scaled={y_scale}" if y_scale else ""))
        ax.grid(alpha=0.25)
        ax.legend(title="Deposit", fontsize=8, ncols=2)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_deposit_windows_2d_subplots(
    deposit_windows: Dict[int, np.ndarray],
    deposit_shapes: Dict[int, Tuple[int, int]],
    *,
    deposit_limits: Dict[int, Tuple[float, float]],
    deposit_limits_var2: Optional[Dict[int, Tuple[float, float]]] = None,
    cmap: Any = DEPOSIT_CMAP_FALLBACK,
    colorbar_label: str = "TMI",
    variable_labels: Tuple[str, str] = ("TMI", "Radiometric_U"),
    save_path: Optional[str] = None,
) -> None:
    dep_ids = sorted(deposit_windows.keys())
    if not dep_ids:
        return

    if deposit_limits_var2 is None:
        deposit_limits_var2 = {}

    # Detect whether we can plot two-variable side-by-side panels.
    dep_nvars: Dict[int, int] = {}
    for dep in dep_ids:
        if dep not in deposit_shapes:
            continue
        h, w = deposit_shapes[dep]
        n_pix = int(h * w)
        n_feat = int(np.asarray(deposit_windows[dep]).size)
        if n_feat == 2 * n_pix:
            dep_nvars[dep] = 2
        elif n_feat == n_pix:
            dep_nvars[dep] = 1
        else:
            dep_nvars[dep] = 0

    multi_mode = any(v == 2 for v in dep_nvars.values())
    dep_per_row = 3 if multi_mode else 4
    nrows = int(np.ceil(len(dep_ids) / dep_per_row))
    ncols = dep_per_row * (2 if multi_mode else 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), dpi=150, squeeze=False)

    for idx, dep in enumerate(dep_ids):
        row = idx // dep_per_row
        col_block = idx % dep_per_row
        start_col = col_block * (2 if multi_mode else 1)

        if dep not in deposit_shapes:
            continue
        h, w = deposit_shapes[dep]
        x_dep = np.asarray(deposit_windows[dep], dtype=float).ravel()
        n_pix = int(h * w)

        if dep_nvars.get(dep, 0) == 0:
            continue

        if dep_nvars.get(dep) == 1:
            ax = axes[row, start_col]
            arr = x_dep[:n_pix].reshape(h, w)
            if dep in deposit_limits:
                vmin, vmax = deposit_limits[dep]
            else:
                vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(f"Deposit {dep}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
            cbar.ax.tick_params(labelsize=7)
            cbar.set_label(colorbar_label, fontsize=8)
            if multi_mode:
                axes[row, start_col + 1].axis("off")
        else:
            ax1 = axes[row, start_col]
            ax2 = axes[row, start_col + 1]
            arr1 = x_dep[:n_pix].reshape(h, w)
            arr2 = x_dep[n_pix : 2 * n_pix].reshape(h, w)

            if dep in deposit_limits:
                vmin1, vmax1 = deposit_limits[dep]
            else:
                vmin1, vmax1 = float(np.nanmin(arr1)), float(np.nanmax(arr1))
            if dep in deposit_limits_var2:
                vmin2, vmax2 = deposit_limits_var2[dep]
            else:
                vmin2, vmax2 = float(np.nanmin(arr2)), float(np.nanmax(arr2))

            im1 = ax1.imshow(arr1, cmap=cmap, vmin=vmin1, vmax=vmax1, origin="upper")
            im2 = ax2.imshow(arr2, cmap=cmap, vmin=vmin2, vmax=vmax2, origin="upper")
            ax1.set_title(f"Deposit {dep} - {variable_labels[0]}", fontsize=10)
            ax2.set_title(f"Deposit {dep} - {variable_labels[1]}", fontsize=10)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.03)
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
            cbar1.ax.tick_params(labelsize=7)
            cbar2.ax.tick_params(labelsize=7)
            cbar1.set_label(variable_labels[0], fontsize=8)
            cbar2.set_label(variable_labels[1], fontsize=8)

    # Hide any unused axes.
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")

    fig.suptitle(
        "All deposits (2D) with deposit-specific limits"
        + (" (side-by-side variables)" if multi_mode else ""),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sn_profiles_and_deposits_2d_combined(
    deposit_windows: Dict[int, np.ndarray],
    deposit_shapes: Dict[int, Tuple[int, int]],
    *,
    dep_to_color: Optional[Dict[int, Any]] = None,
    resolution_m: float = 200,
    band_halfwidth: int = 2,
    x_center_mode: str = "midpoint",
    y_center: bool = True,
    y_scale: Optional[str] = None,
    smoothing_method: Optional[str] = None,
    smoothing_window: int = 5,
    deposit_limits: Dict[int, Tuple[float, float]],
    deposit_limits_var2: Optional[Dict[int, Tuple[float, float]]] = None,
    cmap: Any = DEPOSIT_CMAP_FALLBACK,
    var1_label: str = "TMI",
    var2_label: str = "Radiometric_U",
    title: str = "S-N profiles + 2D deposit windows",
    save_path: Optional[str] = None,
) -> None:
    dep_ids = sorted(deposit_windows.keys())
    if not dep_ids:
        return

    if dep_to_color is None:
        dep_to_color = make_dep_to_color(dep_ids)
    if deposit_limits_var2 is None:
        deposit_limits_var2 = {}

    dep_nvars: Dict[int, int] = {}
    for dep in dep_ids:
        if dep not in deposit_shapes:
            continue
        h, w = deposit_shapes[dep]
        n_pix = int(h * w)
        n_feat = int(np.asarray(deposit_windows[dep]).size)
        if n_feat == 2 * n_pix:
            dep_nvars[dep] = 2
        elif n_feat == n_pix:
            dep_nvars[dep] = 1
        else:
            dep_nvars[dep] = 0

    has_multi_vectors = any(v == 2 for v in dep_nvars.values())

    # Multi: keep new 2-column layout.
    # Uni: use previous denser layout.
    deposit_cols = 2 if has_multi_vectors else 4
    nrows_2d = int(np.ceil(len(dep_ids) / deposit_cols))
    if has_multi_vectors:
        fig_w = 30
        fig_h = max(7.5, 2.9 * nrows_2d)
        width_ratios = [1.0, 2.4]
        outer_wspace = 0.06
    else:
        fig_w = 22
        fig_h = max(8.0, 2.7 * nrows_2d)
        width_ratios = [1.0, 1.65]
        outer_wspace = 0.08
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    outer = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=outer_wspace)

    # Left panel: S-N profiles.
    if has_multi_vectors:
        gs_sn = outer[0, 0].subgridspec(1, 2, wspace=0.26)
        sn_axes = [fig.add_subplot(gs_sn[0, 0]), fig.add_subplot(gs_sn[0, 1])]
        sn_cfg = [(0, var1_label), (1, var2_label)]
    else:
        sn_axes = [fig.add_subplot(outer[0, 0])]
        sn_cfg = [(0, var1_label)]

    for ax, (var_idx, label) in zip(sn_axes, sn_cfg):
        for dep_id in dep_ids:
            if dep_id not in deposit_shapes:
                continue
            h, w = deposit_shapes[dep_id]
            x_dep = np.asarray(deposit_windows[dep_id], dtype=float).ravel()
            n_pix = h * w

            if dep_nvars.get(dep_id, 0) == 0:
                continue
            if dep_nvars.get(dep_id, 0) == 1 and var_idx != 0:
                continue
            if x_dep.size < (var_idx + 1) * n_pix:
                continue

            i0 = var_idx * n_pix
            i1 = i0 + n_pix
            window = x_dep[i0:i1].reshape(h, w)
            profile = _center_sn_profile(window, band_halfwidth=band_halfwidth)

            x_km = np.arange(h) * (resolution_m / 1000.0)
            if x_center_mode == "midpoint":
                x0 = x_km[h // 2]
            elif x_center_mode == "peak":
                x0 = x_km[int(np.argmax(np.abs(profile)))]
            else:
                raise ValueError("x_center_mode must be 'midpoint' or 'peak'")

            y = profile.astype(float)
            if y_center:
                y = y - np.mean(y)

            if y_scale is not None:
                if y_scale == "std":
                    scale = np.std(y)
                elif y_scale == "peak2peak":
                    scale = np.ptp(y)
                else:
                    raise ValueError("y_scale must be None, 'std', or 'peak2peak'")
                if scale > 0:
                    y = y / scale

            y = _smooth_profile(y, method=smoothing_method, window=smoothing_window)

            ax.plot(
                x_km - x0,
                y,
                lw=1.9,
                color=dep_to_color.get(dep_id, "k"),
                label=str(dep_id),
                alpha=0.95,
            )

        ax.axvline(0, lw=1, alpha=0.5)
        ax.axhline(0, lw=1, alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel("Distance along S-N line (km), centered")
        ax.set_ylabel(label + " (centered)" + (f", scaled={y_scale}" if y_scale else ""))
        ax.grid(alpha=0.25)
        ax.legend(title="Deposit", fontsize=7, ncols=2, loc="best")

    # Right panel: 2D deposit windows.
    dep_per_row = deposit_cols
    nrows = int(np.ceil(len(dep_ids) / dep_per_row))
    ncols = dep_per_row * (2 if has_multi_vectors else 1)
    gs_2d = outer[0, 1].subgridspec(
        nrows,
        ncols,
        wspace=0.18 if has_multi_vectors else 0.25,
        hspace=0.25 if has_multi_vectors else 0.30,
    )

    for idx, dep in enumerate(dep_ids):
        row = idx // dep_per_row
        col_block = idx % dep_per_row
        start_col = col_block * (2 if has_multi_vectors else 1)

        if dep not in deposit_shapes:
            continue
        h, w = deposit_shapes[dep]
        x_dep = np.asarray(deposit_windows[dep], dtype=float).ravel()
        n_pix = int(h * w)
        nvars = dep_nvars.get(dep, 0)
        if nvars == 0:
            continue

        if nvars == 1:
            ax = fig.add_subplot(gs_2d[row, start_col])
            arr = x_dep[:n_pix].reshape(h, w)
            if dep in deposit_limits:
                vmin, vmax = deposit_limits[dep]
            else:
                vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(f"Dep {dep}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.02)
            cbar.ax.tick_params(labelsize=6.5)
            cbar.set_label(var1_label, fontsize=7)
            if has_multi_vectors:
                ax_blank = fig.add_subplot(gs_2d[row, start_col + 1])
                ax_blank.axis("off")
        else:
            ax1 = fig.add_subplot(gs_2d[row, start_col])
            ax2 = fig.add_subplot(gs_2d[row, start_col + 1])
            arr1 = x_dep[:n_pix].reshape(h, w)
            arr2 = x_dep[n_pix : 2 * n_pix].reshape(h, w)

            if dep in deposit_limits:
                vmin1, vmax1 = deposit_limits[dep]
            else:
                vmin1, vmax1 = float(np.nanmin(arr1)), float(np.nanmax(arr1))
            if dep in deposit_limits_var2:
                vmin2, vmax2 = deposit_limits_var2[dep]
            else:
                vmin2, vmax2 = float(np.nanmin(arr2)), float(np.nanmax(arr2))

            im1 = ax1.imshow(arr1, cmap=cmap, vmin=vmin1, vmax=vmax1, origin="upper")
            im2 = ax2.imshow(arr2, cmap=cmap, vmin=vmin2, vmax=vmax2, origin="upper")
            ax1.set_title(f"Dep {dep} - {var1_label}", fontsize=9)
            ax2.set_title(f"Dep {dep} - {var2_label}", fontsize=9)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])

            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.043, pad=0.02)
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.043, pad=0.02)
            cbar1.ax.tick_params(labelsize=6.5)
            cbar2.ax.tick_params(labelsize=6.5)
            cbar1.set_label(var1_label, fontsize=7)
            cbar2.set_label(var2_label, fontsize=7)

    # Keep the 2D block left-justified when the last row is incomplete.
    # (Unused cells remain at the right side of the last row.)
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_two_group_plot(
    cases: List[CaseInfo],
    *,
    outdir: str,
    group_a: Dict[str, Any],
    group_b: Dict[str, Any],
    best_kpcs_by_dep: Optional[Dict[int, int]],
    title: str,
    outname: str,
    band: str = "minmax",
    ymax: Optional[float] = DEFAULT_YMAX,
    color_a: str = "C0",
    color_b: str = "C1",
) -> None:
    label_a = group_a.get("label", "Group A")
    label_b = group_b.get("label", "Group B")
    collect_a = {k: v for k, v in group_a.items() if k != "label"}
    collect_b = {k: v for k, v in group_b.items() if k != "label"}

    curves_a, outvals_a = collect_recovery_for_group(
        cases,
        best_kpcs_by_dep=best_kpcs_by_dep,
        **collect_a,
    )
    curves_b, outvals_b = collect_recovery_for_group(
        cases,
        best_kpcs_by_dep=best_kpcs_by_dep,
        **collect_b,
    )

    arr_a = stack_curves_float(curves_a)
    arr_b = stack_curves_float(curves_b)
    if arr_a.size == 0 or arr_b.size == 0:
        raise RuntimeError(f"Missing data for plot {outname}")

    k_common = min(arr_a.shape[1], arr_b.shape[1])
    arr_a = arr_a[:, :k_common]
    arr_b = arr_b[:, :k_common]
    overlap_a, hit_a = aggregate_rank_event_counts(outvals_a, k_common)
    overlap_b, hit_b = aggregate_rank_event_counts(outvals_b, k_common)

    plot_aggregate_recovery_with_event_markers(
        arr_a=arr_a,
        arr_b=arr_b,
        overlap_a=overlap_a,
        hit_a=hit_a,
        overlap_b=overlap_b,
        hit_b=hit_b,
        label_a=label_a,
        label_b=label_b,
        title=title,
        outpath=os.path.join(outdir, outname),
        band=band,
        ymax=ymax,
        color_a=color_a,
        color_b=color_b,
    )


def plot_aggregate_recovery_with_event_markers_three_groups(
    *,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    arr_c: np.ndarray,
    overlap_a: np.ndarray,
    hit_a: np.ndarray,
    overlap_b: np.ndarray,
    hit_b: np.ndarray,
    overlap_c: np.ndarray,
    hit_c: np.ndarray,
    label_a: str,
    label_b: str,
    label_c: str,
    title: str,
    outpath: str,
    band: str = "minmax",
    ymax: Optional[float] = None,
    color_a: str = "C0",
    color_b: str = "C1",
    color_c: str = "C2",
) -> None:
    plt.figure(figsize=(9.2, 5.8), dpi=150)
    ax = plt.gca()

    def _plot_group(
        arr: np.ndarray,
        overlap_count: np.ndarray,
        hit_count: np.ndarray,
        label: str,
        color: str,
    ) -> None:
        if arr.size == 0:
            return
        k_common = arr.shape[1]
        x = np.arange(1, k_common + 1)
        mean = arr.mean(axis=0)
        ax.plot(x, mean, color=color, linewidth=2.3, label=f"{label} (mean)")
        if band == "minmax":
            ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=color, alpha=0.10, label=f"{label} (min-max)")
        elif band == "p10p90":
            ax.fill_between(
                x,
                np.percentile(arr, 10, axis=0),
                np.percentile(arr, 90, axis=0),
                color=color,
                alpha=0.10,
                label=f"{label} (p10-p90)",
            )
        elif band != "none":
            raise ValueError(f"Unknown band: {band}")
        _plot_rank_events(ax, x, mean, overlap_count, hit_count)

    _plot_group(arr_a, overlap_a, hit_a, label_a, color_a)
    _plot_group(arr_b, overlap_b, hit_b, label_b, color_b)
    _plot_group(arr_c, overlap_c, hit_c, label_c, color_c)

    ax.set_title(title)
    ax.set_xlabel("Prediction rank (1 = most similar)")
    ax.set_ylabel("Cumulative recovered fraction of all test-deposit area")
    ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_all_recovery_curves_three_groups(
    *,
    curves_a: Dict[int, np.ndarray],
    curves_b: Dict[int, np.ndarray],
    curves_c: Dict[int, np.ndarray],
    label_a: str,
    label_b: str,
    label_c: str,
    color_a: str,
    color_b: str,
    color_c: str,
    title: str,
    outpath: str,
    ymax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(9.2, 5.8), dpi=150)
    ax = plt.gca()

    def _label_anchor(y: np.ndarray) -> Tuple[float, float]:
        if y.size == 0:
            return 1.0, 0.0
        diff_idx = np.flatnonzero(np.abs(np.diff(y)) > 1e-12)
        idx = int(diff_idx[-1] + 1) if diff_idx.size else y.size - 1
        return float(idx + 1), float(y[idx])

    def _plot_curves(curves: Dict[int, np.ndarray], color: str, label: str, text_dx: int) -> None:
        first = True
        for dep in sorted(curves):
            y = np.asarray(curves[dep], dtype=float)
            if y.size == 0:
                continue
            x = np.arange(1, y.size + 1)
            ax.plot(x, y, color=color, alpha=0.45, linewidth=1.6, label=label if first else None)
            x_anchor, y_anchor = _label_anchor(y)
            y_offset = 7 if dep % 2 else -7
            ax.annotate(
                str(dep),
                (x_anchor, y_anchor),
                xytext=(text_dx, y_offset),
                textcoords="offset points",
                va="center",
                ha="left" if text_dx > 0 else "right",
                fontsize=8.5,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.8),
            )
            first = False

    _plot_curves(curves_a, color_a, label_a, text_dx=-9)
    _plot_curves(curves_b, color_b, label_b, text_dx=9)
    _plot_curves(curves_c, color_c, label_c, text_dx=14)

    ax.set_title(title)
    ax.set_xlabel("Prediction rank (1 = most similar)")
    ax.set_ylabel("Cumulative recovered fraction of all test-deposit area")
    ax.set_ylim(0, float(ymax) if ymax is not None else 1.02)
    ax.set_xlim(1, FORCE_K)
    ax.grid(alpha=0.25)
    handles = [
        Line2D([0], [0], color=color_a, lw=2.2, label=label_a),
        Line2D([0], [0], color=color_b, lw=2.2, label=label_b),
        Line2D([0], [0], color=color_c, lw=2.2, label=label_c),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def _make_three_group_plot(
    cases: List[CaseInfo],
    *,
    outdir: str,
    group_a: Dict[str, Any],
    group_b: Dict[str, Any],
    group_c: Dict[str, Any],
    best_kpcs_by_dep: Optional[Dict[int, int]],
    title: str,
    outname: str,
    band: str = "minmax",
    ymax: Optional[float] = DEFAULT_YMAX,
    color_a: str = "C0",
    color_b: str = "C1",
    color_c: str = "C2",
) -> None:
    label_a = group_a.get("label", "Group A")
    label_b = group_b.get("label", "Group B")
    label_c = group_c.get("label", "Group C")
    collect_a = {k: v for k, v in group_a.items() if k != "label"}
    collect_b = {k: v for k, v in group_b.items() if k != "label"}
    collect_c = {k: v for k, v in group_c.items() if k != "label"}

    curves_a, outvals_a = collect_recovery_for_group(cases, best_kpcs_by_dep=best_kpcs_by_dep, **collect_a)
    curves_b, outvals_b = collect_recovery_for_group(cases, best_kpcs_by_dep=best_kpcs_by_dep, **collect_b)
    curves_c, outvals_c = collect_recovery_for_group(cases, best_kpcs_by_dep=best_kpcs_by_dep, **collect_c)

    arr_a = stack_curves_float(curves_a)
    arr_b = stack_curves_float(curves_b)
    arr_c = stack_curves_float(curves_c)
    if arr_a.size == 0 or arr_b.size == 0 or arr_c.size == 0:
        raise RuntimeError(f"Missing data for plot {outname}")

    k_common = min(arr_a.shape[1], arr_b.shape[1], arr_c.shape[1])
    arr_a = arr_a[:, :k_common]
    arr_b = arr_b[:, :k_common]
    arr_c = arr_c[:, :k_common]
    overlap_a, hit_a = aggregate_rank_event_counts(outvals_a, k_common)
    overlap_b, hit_b = aggregate_rank_event_counts(outvals_b, k_common)
    overlap_c, hit_c = aggregate_rank_event_counts(outvals_c, k_common)

    plot_aggregate_recovery_with_event_markers_three_groups(
        arr_a=arr_a,
        arr_b=arr_b,
        arr_c=arr_c,
        overlap_a=overlap_a,
        hit_a=hit_a,
        overlap_b=overlap_b,
        hit_b=hit_b,
        overlap_c=overlap_c,
        hit_c=hit_c,
        label_a=label_a,
        label_b=label_b,
        label_c=label_c,
        title=title,
        outpath=os.path.join(outdir, outname),
        band=band,
        ymax=ymax,
        color_a=color_a,
        color_b=color_b,
        color_c=color_c,
    )


def main(argv: Optional[List[str]] = None) -> None:
    global FORCE_K, DEFAULT_YMAX

    parser = argparse.ArgumentParser(description="Compare validation outputs across methods/cases.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON or YAML config. If omitted, built-in defaults are used.",
    )
    args = parser.parse_args(argv)

    runtime = load_runtime_config(args.config)
    inputs_cfg = runtime["inputs"]
    filters_cfg = runtime["filters"]
    plot_cfg = runtime["plot"]
    comparison_cfg = runtime["comparison"]
    run_cfg = runtime["run"]

    base_output_roots = inputs_cfg.get("output_roots", [BASE_OUTPUT_DIR])
    if isinstance(base_output_roots, str):
        base_output_roots = [base_output_roots]
    deposits_filter = inputs_cfg.get("deposits_1based", DEPOSITS_1BASED)
    best_kpcs_csv = inputs_cfg.get("best_kpcs_csv", BEST_KPCS_CSV)
    compare_outdir = inputs_cfg.get("compare_outdir", COMPARE_OUTDIR)
    os.makedirs(compare_outdir, exist_ok=True)

    default_min_cover = float(filters_cfg.get("default_min_cover", DEFAULT_MINCOVER))
    use_best_kpcs_filter = bool(filters_cfg.get("use_best_kpcs_filter", USE_BEST_KPCS_FILTER))

    force_k = int(plot_cfg.get("force_k", FORCE_K))
    default_ymax = float(plot_cfg.get("default_ymax", DEFAULT_YMAX))
    deposit_limits_tmi = _coerce_int_key_limits(plot_cfg.get("deposit_limits_tmi", DEPOSIT_LIMITS_TMI))
    deposit_limits_u = _coerce_int_key_limits(plot_cfg.get("deposit_limits_u", DEPOSIT_LIMITS_U))

    run_raw_spca_diagnostics = bool(
        run_cfg.get("run_raw_spca_diagnostics", RUN_RAW_SPCA_DIAGNOSTICS)
    )
    run_top_gain_subplots = bool(run_cfg.get("run_top_gain_subplots", RUN_TOP_GAIN_SUBPLOTS))
    top_gain_n = int(run_cfg.get("top_gain_n", TOP_GAIN_N))
    selected_deposits_for_subplots = run_cfg.get("selected_deposits_for_subplots", None)
    if selected_deposits_for_subplots is not None:
        selected_deposits_for_subplots = [int(d) for d in selected_deposits_for_subplots]
    sn_smoothing_cfg = run_cfg.get("sn_smoothing", {}) or {}
    sn_smoothing_enabled = bool(sn_smoothing_cfg.get("enabled", False))
    sn_smoothing_method = sn_smoothing_cfg.get("method", "moving_average") if sn_smoothing_enabled else None
    sn_smoothing_window = int(sn_smoothing_cfg.get("window", 5))

    FORCE_K = force_k
    DEFAULT_YMAX = default_ymax

    best_kpcs_by_dep: Dict[int, int] = {}
    if use_best_kpcs_filter:
        best_kpcs_by_dep = load_best_kpcs_map(best_kpcs_csv, k_column="k_pcs")
        if best_kpcs_by_dep:
            print(f"[INFO] Loaded best-k map for {len(best_kpcs_by_dep)} deposits: {best_kpcs_csv}")
        else:
            print(f"[WARN] No best-k map loaded from: {best_kpcs_csv}")

    cases = find_all_validation_pkls(base_output_roots, deposits_filter)
    if not cases:
        raise RuntimeError(
            "No validation_topk_results.pkl found. "
            f"Check output roots: {base_output_roots}"
        )

    rows = []
    for case in cases:
        out_val = load_validation_dict(case.pkl_path)
        rows.append(build_summary_row(case, out_val))
    df = pd.DataFrame(rows)

    groups_cfg = comparison_cfg.get("groups", None)
    if isinstance(groups_cfg, list) and len(groups_cfg) >= 2:
        groups = groups_cfg
    else:
        # Backward-compatible path
        group_a = comparison_cfg["group_a"]
        group_b = comparison_cfg["group_b"]
        group_c = comparison_cfg.get("group_c", None)
        groups = [group_a, group_b] + ([group_c] if isinstance(group_c, dict) else [])

    if len(groups) < 2:
        raise RuntimeError("Comparison config requires at least 2 groups.")

    # Backward-compatible aliases for downstream logic that still references A/B.
    group_a = groups[0]
    group_b = groups[1]

    labels = [g.get("label", f"Group {i+1}") for i, g in enumerate(groups)]
    fallback_palette = ["orange", "blue", "green", "purple", "brown", "teal", "magenta"]
    legacy_colors = [comparison_cfg.get("color_a"), comparison_cfg.get("color_b"), comparison_cfg.get("color_c")]
    colors = []
    for i, g in enumerate(groups):
        c = g.get("color", None)
        if c is None and i < len(legacy_colors):
            c = legacy_colors[i]
        if c is None:
            c = fallback_palette[i % len(fallback_palette)]
        colors.append(c)

    out_stem = comparison_cfg["name"]

    # Base dataframe subset for strip/diagnostics (same filters as groups where possible).
    methods_keep = []
    groups_for_filters = groups
    for g in groups_for_filters:
        if g.get("method_name"):
            methods_keep.append(g["method_name"])
    df_sub = df[df["method_name"].isin(methods_keep)].copy()
    # Keep only configured min_cover if provided in either group.
    min_cover_vals = [g.get("min_cover", None) for g in groups_for_filters if g.get("min_cover", None) is not None]
    if min_cover_vals:
        target_cover = float(min_cover_vals[0])
        df_sub = df_sub[df_sub["min_cover"].fillna(target_cover) == target_cover].copy()
    if best_kpcs_by_dep:
        keep_spca = (
            (df_sub["method_name"] == "Spatial_PCA")
            & df_sub["deposit_1based"].map(best_kpcs_by_dep).notna()
            & (df_sub["k_pcs"].astype("Int64") == df_sub["deposit_1based"].map(best_kpcs_by_dep).astype("Int64"))
        )
        keep_other = df_sub["method_name"] != "Spatial_PCA"
        df_sub = df_sub[keep_spca | keep_other].copy()

    curves_list: List[Dict[int, np.ndarray]] = []
    outvals_list: List[Dict[int, Dict[str, Any]]] = []
    best_k_columns_list: List[str] = []
    best_k_maps_list: List[Optional[Dict[int, int]]] = []
    arrs: List[np.ndarray] = []
    overlaps: List[np.ndarray] = []
    hits: List[np.ndarray] = []

    best_k_cache: Dict[str, Dict[int, int]] = {}

    for g in groups:
        collect_kwargs = {
            k: v
            for k, v in g.items()
            if k not in {"label", "color", "best_kpcs_csv", "best_kpcs_column"}
        }
        group_method = str(g.get("method_name", "") or "")
        group_best_k_map: Optional[Dict[int, int]] = None
        group_best_col = str(g.get("best_kpcs_column", "k_pcs"))
        if use_best_kpcs_filter and group_method == "Spatial_PCA":
            group_best_csv = g.get("best_kpcs_csv", None)
            if group_best_csv:
                group_best_csv = str(group_best_csv)
                cache_key = f"{group_best_csv}::{group_best_col}"
                if cache_key not in best_k_cache:
                    best_k_cache[cache_key] = load_best_kpcs_map(group_best_csv, k_column=group_best_col)
                group_best_k_map = best_k_cache[cache_key]
                if group_best_k_map:
                    print(f"[INFO] Loaded group best-k map ({g.get('label', 'group')}): {group_best_csv} [{group_best_col}]")
                else:
                    raise RuntimeError(
                        f"Group best-k CSV missing/invalid for {g.get('label', 'group')}: "
                        f"{group_best_csv} (column: {group_best_col})"
                    )
            else:
                group_best_k_map = best_kpcs_by_dep

        curves_g, outvals_g = collect_recovery_for_group(
            cases,
            best_kpcs_by_dep=group_best_k_map,
            best_kpcs_column=group_best_col,
            **collect_kwargs,
        )
        curves_list.append(curves_g)
        outvals_list.append(outvals_g)
        best_k_columns_list.append(group_best_col)
        best_k_maps_list.append(group_best_k_map)
        arrs.append(stack_curves_float(curves_g))
        if group_method == "Spatial_PCA" and isinstance(group_best_k_map, dict) and group_best_k_map:
            deps_dbg = sorted(set(deposits_filter or []) & set(group_best_k_map.keys()))
            deps_dbg = deps_dbg[:6]
            dbg_pairs = ", ".join(f"{d}:{group_best_k_map[d]}" for d in deps_dbg)
            print(f"[DEBUG] best-k values used for '{g.get('label', 'group')}' [{group_best_col}]: {dbg_pairs}")

    if any(a.size == 0 for a in arrs):
        missing = [labels[i] for i, a in enumerate(arrs) if a.size == 0]
        raise RuntimeError(
            "Missing data for one or more comparison groups: " + ", ".join(missing)
        )

    # Backward-compatible aliases for existing 2-group diagnostics section.
    curves_a, outvals_a = curves_list[0], outvals_list[0]
    curves_b, outvals_b = curves_list[1], outvals_list[1]

    k_common = min(a.shape[1] for a in arrs)
    arrs = [a[:, :k_common] for a in arrs]
    for outvals_g in outvals_list:
        ov, hh = aggregate_rank_event_counts(outvals_g, k_common)
        overlaps.append(ov)
        hits.append(hh)

    plot_aggregate_recovery_with_event_markers_n_groups(
        arrs=arrs,
        overlaps=overlaps,
        hits=hits,
        labels=labels,
        colors=colors,
        title=comparison_cfg["aggregate_title"],
        outpath=os.path.join(compare_outdir, f"ALL_recovery_{out_stem}.png"),
        band="minmax",
        ymax=default_ymax,
    )

    plot_all_recovery_curves_n_groups(
        curves_list=curves_list,
        labels=labels,
        colors=colors,
        title=comparison_cfg["curves_title"],
        outpath=os.path.join(compare_outdir, f"ALL_recovery_curves_{out_stem}.png"),
        ymax=default_ymax,
    )

    if run_top_gain_subplots:
        # Defaults preserve previous behavior:
        # 2 groups -> gain group1 over group0
        # 3+ groups -> gain group2 over group1
        default_base = 1 if len(groups) >= 3 else 0
        default_target = 2 if len(groups) >= 3 else 1
        gain_base_idx = run_cfg.get("top_gain_base_group_index", default_base)
        gain_target_idx = run_cfg.get("top_gain_target_group_index", default_target)
        if gain_base_idx is None:
            gain_base_idx = default_base
        if gain_target_idx is None:
            gain_target_idx = default_target
        gain_base_idx = int(gain_base_idx)
        gain_target_idx = int(gain_target_idx)

        plot_top_gain_subplots_n_groups(
            curves_list=curves_list,
            outvals_list=outvals_list,
            labels=labels,
            colors=colors,
            gain_base_idx=gain_base_idx,
            gain_target_idx=gain_target_idx,
            outpath=os.path.join(compare_outdir, f"TOP{top_gain_n}_gain_{out_stem}_subplots.png"),
            top_n=top_gain_n,
            ymax=default_ymax,
            best_k_maps=best_k_maps_list,
            best_k_columns=best_k_columns_list,
        )
        df_gain_summary = build_top_gain_summary_n_groups(
            curves_list=curves_list,
            outvals_list=outvals_list,
            labels=labels,
            gain_base_idx=gain_base_idx,
            gain_target_idx=gain_target_idx,
            top_n=top_gain_n,
            selected_deposits=None,
        )
        if not df_gain_summary.empty:
            gain_csv = os.path.join(compare_outdir, f"TOPGAIN_summary_{out_stem}_all_deposits.csv")
            df_gain_summary.to_csv(gain_csv, index=False)
            print(f"[OK] Wrote top-gain summary CSV (all deposits): {gain_csv}")
        if selected_deposits_for_subplots:
            selected_suptitle = (
                ""
                if len(selected_deposits_for_subplots) == 1
                else f"Selected deposits: gain {labels[gain_target_idx]} over {labels[gain_base_idx]}"
            )
            plot_top_gain_subplots_n_groups(
                curves_list=curves_list,
                outvals_list=outvals_list,
                labels=labels,
                colors=colors,
                gain_base_idx=gain_base_idx,
                gain_target_idx=gain_target_idx,
                outpath=os.path.join(compare_outdir, f"SELECTED_gain_{out_stem}_subplots.png"),
                top_n=top_gain_n,
                ymax=default_ymax,
                selected_deposits=selected_deposits_for_subplots,
                suptitle=selected_suptitle,
                best_k_maps=best_k_maps_list,
                best_k_columns=best_k_columns_list,
            )
            df_selected_summary = build_top_gain_summary_n_groups(
                curves_list=curves_list,
                outvals_list=outvals_list,
                labels=labels,
                gain_base_idx=gain_base_idx,
                gain_target_idx=gain_target_idx,
                top_n=top_gain_n,
                selected_deposits=selected_deposits_for_subplots,
            )
            if not df_selected_summary.empty:
                selected_csv = os.path.join(compare_outdir, f"SELECTED_gain_summary_{out_stem}_all_deposits.csv")
                df_selected_summary.to_csv(selected_csv, index=False)
                print(f"[OK] Wrote selected-gain summary CSV (all deposits): {selected_csv}")

    groups_for_filters = groups
    # Prefer Multi as source for deposit vectors so var2 is available for 2D/S-N plots.
    multi_groups = [g for g in groups_for_filters if g.get("analysis_type") == "Multi"]
    source_group_2d = multi_groups[0] if multi_groups else groups_for_filters[0]
    preferred_analysis_for_2d = source_group_2d.get("analysis_type")
    preferred_var_for_2d = source_group_2d.get("selected_variable")
    preferred_method_for_2d = source_group_2d.get("method_name") or "Spatial_PCA"
    try:
        deposit_windows, deposit_shapes = collect_deposit_windows_and_shapes_from_pkls(
            cases,
            preferred_method_name=preferred_method_for_2d,
            preferred_analysis_type=preferred_analysis_for_2d,
            preferred_selected_variable=preferred_var_for_2d,
        )
        try:
            cfg_for_cmap = setup_analysis_config(
                method_name="Spatial_PCA",
                analysis_type=preferred_analysis_for_2d or "Uni",
                selected_variable=preferred_var_for_2d,
                output_dir=None,
            )
            deposit_cmap = cfg_for_cmap.get("cmap", DEPOSIT_CMAP_FALLBACK)
        except Exception:
            deposit_cmap = DEPOSIT_CMAP_FALLBACK

        dep_to_color = make_dep_to_color(sorted(deposit_windows.keys()))
        has_multi_vectors = any(
            np.asarray(deposit_windows[d]).size == 2 * int(deposit_shapes[d][0] * deposit_shapes[d][1])
            for d in deposit_windows
            if d in deposit_shapes
        )
        plot_sn_profiles_and_deposits_2d_combined(
            deposit_windows,
            deposit_shapes,
            dep_to_color=dep_to_color,
            resolution_m=200,
            band_halfwidth=2,
            x_center_mode="midpoint",
            y_center=True,
            y_scale=None,
            smoothing_method=sn_smoothing_method,
            smoothing_window=sn_smoothing_window,
            deposit_limits=deposit_limits_tmi,
            deposit_limits_var2=deposit_limits_u,
            cmap=deposit_cmap,
            var1_label="TMI",
            var2_label="Radiometric_U",
            title="S-N profiles and 2D deposit windows",
            save_path=os.path.join(compare_outdir, f"ALL_SN_and_2D_subplots_{out_stem}.png"),
        )
    except Exception as exc:
        print(f"[WARN] Skipping 2D deposit subplot figure: {exc}")

    is_raw_spca_uni_tmi = (
        run_raw_spca_diagnostics
        and {group_a.get("method_name"), group_b.get("method_name")} == {"Raw_comparison", "Spatial_PCA"}
        and group_a.get("analysis_type") == "Uni"
        and group_b.get("analysis_type") == "Uni"
        and _norm_varname(group_a.get("selected_variable")) == "TMI"
        and _norm_varname(group_b.get("selected_variable")) == "TMI"
    )

    if is_raw_spca_uni_tmi:
        # Keep the richer diagnostics only for raw-vs-spca Uni(TMI).
        if group_a.get("method_name") == "Raw_comparison":
            curves_raw_sub, outvals_raw_sub = curves_a, collect_recovery_for_group(
                cases,
                method_name=group_a.get("method_name"),
                analysis_type=group_a.get("analysis_type"),
                selected_variable=group_a.get("selected_variable"),
                min_cover=group_a.get("min_cover", default_min_cover),
                best_kpcs_by_dep=best_kpcs_by_dep,
            )[1]
            curves_spca_sub, outvals_spca_sub = curves_b, collect_recovery_for_group(
                cases,
                method_name=group_b.get("method_name"),
                analysis_type=group_b.get("analysis_type"),
                selected_variable=group_b.get("selected_variable"),
                min_cover=group_b.get("min_cover", default_min_cover),
                best_kpcs_by_dep=best_kpcs_by_dep,
            )[1]
        else:
            curves_raw_sub, outvals_raw_sub = curves_b, collect_recovery_for_group(
                cases,
                method_name=group_b.get("method_name"),
                analysis_type=group_b.get("analysis_type"),
                selected_variable=group_b.get("selected_variable"),
                min_cover=group_b.get("min_cover", default_min_cover),
                best_kpcs_by_dep=best_kpcs_by_dep,
            )[1]
            curves_spca_sub, outvals_spca_sub = curves_a, collect_recovery_for_group(
                cases,
                method_name=group_a.get("method_name"),
                analysis_type=group_a.get("analysis_type"),
                selected_variable=group_a.get("selected_variable"),
                min_cover=group_a.get("min_cover", default_min_cover),
                best_kpcs_by_dep=best_kpcs_by_dep,
            )[1]

        plot_top_gain_subplots_two_groups(
            curves_a=curves_raw_sub,
            curves_b=curves_spca_sub,
            outvals_a=outvals_raw_sub,
            outvals_b=outvals_spca_sub,
            label_a="Raw TMI",
            label_b="Spatial_PCA TMI (best k)",
            outpath=os.path.join(compare_outdir, "TOP4_gain_raw_vs_spca_subplots.png"),
            top_n=4,
            ymax=default_ymax,
        )
        # Removed by request:
        # - SCATTER_optimalK_vs_deltaAUC.png
        # - STRIP_auc_recovery_Uni_TMI_by_deposit.png

        # S-N and 2D deposit plots are already produced above with long, config-based
        # filenames (e.g., "..._{out_stem}.png"). Skip legacy duplicate outputs here.


if __name__ == "__main__":
    main()
