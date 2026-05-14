"""Search tied fused-PCA rotations for the Deposit 3 paper-era ranking.

The Deposit 3 two-stage fused PCA has nearly tied eigenvalues from PC3 onward.
This script tests whether rotating that tied PC3-PC36 subspace can move the
current ranking toward the frozen paper-era ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

DEFAULT_CONFIG = "configs/carajas_multi_tmi_u.yaml"
DEFAULT_TARGET_PICKLE = (
    "docs/repro_debug_deposit3_multivariate/paper_era_ground_truth/"
    "validation_topk_results.pkl"
)
DEFAULT_OUTPUT_DIR = "docs/repro_debug_deposit3_multivariate/rotation_recovery"


@dataclass(frozen=True)
class RotationCandidate:
    name: str
    basis: np.ndarray
    details: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover/test rotations of the tied fused PCA subspace for Deposit 3."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Run config.")
    parser.add_argument("--target-pickle", default=DEFAULT_TARGET_PICKLE, help="Paper-era pickle.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for outputs.")
    parser.add_argument("--deposit", type=int, default=3, help="1-based training deposit.")
    parser.add_argument("--kpcs", type=int, default=17, help="Fused PCs used for ranking.")
    parser.add_argument(
        "--fixed-pcs",
        type=int,
        default=2,
        help="Leading fused PCs to keep fixed before rotating the tied subspace.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--random-trials", type=int, default=400, help="Random subspace trials.")
    parser.add_argument("--local-trials", type=int, default=400, help="Local perturbation trials.")
    parser.add_argument(
        "--max-background",
        type=int,
        default=6000,
        help="Max background rows used for target-informed generalized bases.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fused = rebuild_fused_scores(args)
    target = load_target_ranking(_resolve_repo_path(args.target_pickle))
    recovery = run_rotation_search(args, fused=fused, target_rows=target["ranked_pred_rows"])

    write_outputs(output_dir, args=args, fused=fused, target=target, recovery=recovery)
    print(json.dumps(_console_summary(recovery), indent=2))
    print(f"Wrote rotation recovery outputs: {output_dir}")
    return 0


def rebuild_fused_scores(args: argparse.Namespace) -> dict[str, Any]:
    from spatial_pca.config import load_run_config
    from spatial_pca.geodata.deposits import get_deposit_template
    from spatial_pca.pipeline import (
        _get_deposits_path,
        _get_summary_variables,
        _load_case_deposits,
        _resolve_multivariate_best_kpcs,
        _validate_multivariate_rasters,
        load_multivariate_rasters,
    )
    from spatial_pca.spca.ranking import rank_multi_two_stage_pca_fusion
    from spatial_pca.spca.windows import build_multivariate_window_matrix

    config_path = _resolve_repo_path(args.config)
    config = load_run_config(config_path)
    variable_names = _get_summary_variables(config)
    raster_data = load_multivariate_rasters(config, variable_names)
    _validate_multivariate_rasters(raster_data, reference_variable=variable_names[0])
    deposits_path = _get_deposits_path(config)
    deposits_gdf = _load_case_deposits(config, deposits_path, raster_data[variable_names[0]].crs)

    templates = {
        var: get_deposit_template(deposits_gdf, int(args.deposit) - 1, raster_data[var])
        for var in variable_names
    }
    window_matrix = build_multivariate_window_matrix(
        rasters={var: raster_data[var].array for var in variable_names},
        deposit_templates={var: templates[var].array for var in variable_names},
        variable_names=variable_names,
        stride_y=int(config["analysis_defaults"]["stride_y"]),
        stride_x=int(config["analysis_defaults"]["stride_x"]),
    )
    k_var1, k_var2 = _resolve_multivariate_best_kpcs(config, int(args.deposit))
    ranking_result = rank_multi_two_stage_pca_fusion(
        X_multi=window_matrix.data_for_pca,
        deposit_index=window_matrix.deposit_index,
        window_shape=window_matrix.window_shape,
        k_pcs_var1=k_var1,
        k_pcs_var2=k_var2,
        k_pcs_fused=int(args.kpcs),
        use_whitening=False,
        use_weights=True,
        standardize_fused_input=True,
        weight_mode="square",
        normalize_weights_over="selected_pcs",
    )
    fusion = ranking_result.fusion_details or {}
    n_windows = int(window_matrix.window_indices_for_mapping.shape[0])
    return {
        "config_path": str(config_path),
        "variable_names": list(variable_names),
        "Zf": np.asarray(fusion["Zf_scores"], dtype=float),
        "eigvals_fused": np.asarray(fusion["eigvals_fused"], dtype=float),
        "baseline_ranked_idx": np.asarray(ranking_result.ranked_idx, dtype=int),
        "baseline_ranked_dists": np.asarray(ranking_result.ranked_dists, dtype=float),
        "baseline_weights": np.asarray(ranking_result.weights, dtype=float),
        "deposit_index": int(window_matrix.deposit_index),
        "n_windows": n_windows,
        "window_shape": tuple(int(v) for v in window_matrix.window_shape),
        "data_for_pca_shape": tuple(int(v) for v in window_matrix.data_for_pca.shape),
        "k_pcs_var1": int(k_var1),
        "k_pcs_var2": int(k_var2),
        "k_pcs_fused": int(args.kpcs),
    }


def load_target_ranking(target_pickle: Path) -> dict[str, Any]:
    with target_pickle.open("rb") as infile:
        payload = pickle.load(infile)
    target_rows = np.asarray(payload["ranked_pred_rows"], dtype=int)
    return {
        "target_pickle": str(target_pickle),
        "ranking_mode": payload.get("ranking_mode"),
        "k_pcs_var1": payload.get("k_pcs_var1"),
        "k_pcs_var2": payload.get("k_pcs_var2"),
        "k_pcs_fused": payload.get("k_pcs_fused"),
        "ranked_pred_rows": target_rows,
        "first_hit_rank_by_deposit": payload.get("first_hit_rank_by_deposit"),
    }


def run_rotation_search(
    args: argparse.Namespace,
    *,
    fused: dict[str, Any],
    target_rows: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(args.seed))
    Zf = np.asarray(fused["Zf"], dtype=float)
    fixed_pcs = int(args.fixed_pcs)
    kpcs = int(args.kpcs)
    if not (0 <= fixed_pcs < kpcs <= Zf.shape[1]):
        raise ValueError("Require 0 <= fixed_pcs < kpcs <= Zf.shape[1].")
    tied_dim = Zf.shape[1] - fixed_pcs
    tied_select = kpcs - fixed_pcs

    target_rows = np.asarray(target_rows, dtype=int)
    baseline_basis = np.eye(tied_dim, tied_select)
    baseline = evaluate_basis(
        baseline_basis,
        name="current_identity_basis",
        fused=fused,
        target_rows=target_rows,
        fixed_pcs=fixed_pcs,
        kpcs=kpcs,
    )

    best = baseline
    evaluated: list[dict[str, Any]] = [baseline]

    for candidate in deterministic_candidates(
        fused=fused,
        target_rows=target_rows,
        fixed_pcs=fixed_pcs,
        kpcs=kpcs,
        rng=rng,
        max_background=int(args.max_background),
    ):
        result = evaluate_basis(
            candidate.basis,
            name=candidate.name,
            fused=fused,
            target_rows=target_rows,
            fixed_pcs=fixed_pcs,
            kpcs=kpcs,
            details=candidate.details,
        )
        evaluated.append(result)
        if result["objective"] > best["objective"]:
            best = result

    for trial in range(max(0, int(args.random_trials))):
        basis = random_orthonormal_basis(rng, tied_dim, tied_select)
        result = evaluate_basis(
            basis,
            name=f"random_subspace_{trial:04d}",
            fused=fused,
            target_rows=target_rows,
            fixed_pcs=fixed_pcs,
            kpcs=kpcs,
            details={"trial": trial},
        )
        evaluated.append(result)
        if result["objective"] > best["objective"]:
            best = result

    local_start = complete_orthonormal_basis(best["basis"], rng)
    for trial in range(max(0, int(args.local_trials))):
        angle_scale = 0.45 * (0.995**trial)
        full_basis = perturb_full_basis(
            local_start,
            rng,
            steps=max(4, tied_dim // 2),
            angle_scale=angle_scale,
        )
        basis = full_basis[:, :tied_select]
        result = evaluate_basis(
            basis,
            name=f"local_perturb_best_{trial:04d}",
            fused=fused,
            target_rows=target_rows,
            fixed_pcs=fixed_pcs,
            kpcs=kpcs,
            details={"trial": trial, "angle_scale": angle_scale},
        )
        evaluated.append(result)
        if result["objective"] > best["objective"]:
            best = result
            local_start = complete_orthonormal_basis(best["basis"], rng)

    evaluated_sorted = sorted(evaluated, key=lambda item: item["objective"], reverse=True)
    return {
        "baseline": strip_arrays(baseline),
        "best": strip_arrays(best),
        "best_basis": best["basis"],
        "best_ranked_rows": best["ranked_rows"],
        "best_ranked_dists": best["ranked_dists"],
        "best_weights": best["weights"],
        "evaluated_count": len(evaluated),
        "top_candidates": [strip_arrays(item) for item in evaluated_sorted[:25]],
    }


def deterministic_candidates(
    *,
    fused: dict[str, Any],
    target_rows: np.ndarray,
    fixed_pcs: int,
    kpcs: int,
    rng: np.random.Generator,
    max_background: int,
) -> Iterable[RotationCandidate]:
    Zf = np.asarray(fused["Zf"], dtype=float)
    tied = Zf[:, fixed_pcs:]
    tied_dim = tied.shape[1]
    tied_select = kpcs - fixed_pcs
    target_rows = np.asarray(target_rows, dtype=int)
    n_windows = int(fused["n_windows"])
    deposit_index = int(fused["deposit_index"])
    diff = tied[:n_windows] - tied[deposit_index]

    for n_target in (20, 50, 100, 250):
        target_subset = target_rows[: min(n_target, target_rows.size)]
        if target_subset.size < 2:
            continue
        background_idx = sample_background_indices(
            n_windows=n_windows,
            excluded=np.concatenate([target_subset, np.array([deposit_index])]),
            max_background=max_background,
            rng=rng,
        )
        for ridge_scale in (1e-8, 1e-5, 1e-3, 1e-1):
            basis = generalized_similarity_basis(
                target_diff=diff[target_subset],
                background_diff=diff[background_idx],
                n_select=tied_select,
                ridge_scale=ridge_scale,
            )
            yield RotationCandidate(
                name=f"generalized_bg_over_target_top{n_target}_ridge{ridge_scale:g}",
                basis=basis,
                details={"target_n": int(target_subset.size), "ridge_scale": ridge_scale},
            )

        null_basis = target_null_basis(
            target_diff=diff[target_subset],
            deposit_tied=tied[deposit_index],
            n_select=tied_select,
        )
        yield RotationCandidate(
            name=f"target_null_space_top{n_target}",
            basis=null_basis,
            details={"target_n": int(target_subset.size)},
        )

    dep = tied[deposit_index]
    if np.linalg.norm(dep) > 0:
        dep_first = complete_basis_from_columns(dep[:, None], tied_dim, tied_select, rng)
        yield RotationCandidate(
            name="deposit_vector_first",
            basis=dep_first,
            details={},
        )


def evaluate_basis(
    basis: np.ndarray,
    *,
    name: str,
    fused: dict[str, Any],
    target_rows: np.ndarray,
    fixed_pcs: int,
    kpcs: int,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    basis = orthonormalize_columns(np.asarray(basis, dtype=float))
    tied_select = int(kpcs) - int(fixed_pcs)
    if basis.shape != (np.asarray(fused["Zf"]).shape[1] - fixed_pcs, tied_select):
        raise ValueError(f"Unexpected basis shape: {basis.shape}.")

    ranked_rows, ranked_dists, weights = rank_with_rotated_basis(
        basis,
        fused=fused,
        fixed_pcs=fixed_pcs,
        kpcs=kpcs,
    )
    metrics = ranking_metrics(ranked_rows, target_rows, n_windows=int(fused["n_windows"]))
    objective = (
        10000.0 * metrics["top20_overlap"]
        + 1000.0 * metrics["top50_overlap"]
        + 10.0 * metrics["top250_overlap"]
        + 2500.0 * metrics["old_top20_order_score"]
        + 500.0 * metrics["old_top20_mrr"]
        - 0.02 * metrics["old_top20_mean_rank"]
    )
    return {
        "name": name,
        "details": details or {},
        "objective": float(objective),
        "metrics": metrics,
        "basis": basis,
        "weights": weights,
        "ranked_rows": ranked_rows,
        "ranked_dists": ranked_dists,
    }


def rank_with_rotated_basis(
    basis: np.ndarray,
    *,
    fused: dict[str, Any],
    fixed_pcs: int,
    kpcs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Zf = np.asarray(fused["Zf"], dtype=float)
    deposit_index = int(fused["deposit_index"])
    n_windows = int(fused["n_windows"])
    fixed = Zf[:, :fixed_pcs]
    tied_rot = Zf[:, fixed_pcs:] @ basis
    coords = np.hstack([fixed, tied_rot])[:, :kpcs]
    dep = coords[deposit_index]
    raw = dep**2
    weights = raw / raw.sum() if raw.sum() > 0 else np.ones(kpcs, dtype=float) / float(kpcs)
    diff = coords - dep
    dists = np.sqrt((diff**2) @ weights)
    order = np.argsort(dists)
    keep = (order != deposit_index) & (order < n_windows)
    ranked_rows = order[keep]
    ranked_dists = dists[ranked_rows]
    return ranked_rows, ranked_dists, weights


def ranking_metrics(ranked_rows: np.ndarray, target_rows: np.ndarray, *, n_windows: int) -> dict[str, Any]:
    target_rows = np.asarray(target_rows, dtype=int)
    ranked_rows = np.asarray(ranked_rows, dtype=int)
    pos = np.full(n_windows, fill_value=np.iinfo(np.int32).max, dtype=np.int64)
    pos[ranked_rows[: min(ranked_rows.size, n_windows)]] = np.arange(
        min(ranked_rows.size, n_windows),
        dtype=np.int64,
    )
    top20_target = target_rows[:20]
    old_top20_ranks = pos[top20_target] + 1
    old_top250_ranks = pos[target_rows[:250]] + 1
    expected_top20_ranks = np.arange(1, top20_target.size + 1, dtype=float)
    old_top20_order_abs_error = np.abs(old_top20_ranks.astype(float) - expected_top20_ranks)
    old_top20_order_score = float(np.mean(1.0 / (1.0 + old_top20_order_abs_error)))
    prefix = 0
    for old, new in zip(target_rows, ranked_rows):
        if int(old) != int(new):
            break
        prefix += 1
    return {
        "prefix_equal_count": int(prefix),
        "top20_overlap": int(len(set(ranked_rows[:20]) & set(target_rows[:20]))),
        "top50_overlap": int(len(set(ranked_rows[:50]) & set(target_rows[:50]))),
        "top100_overlap": int(len(set(ranked_rows[:100]) & set(target_rows[:100]))),
        "top250_overlap": int(len(set(ranked_rows[:250]) & set(target_rows[:250]))),
        "old_top20_mean_rank": float(np.mean(old_top20_ranks)),
        "old_top20_median_rank": float(np.median(old_top20_ranks)),
        "old_top20_max_rank": int(np.max(old_top20_ranks)),
        "old_top20_mrr": float(np.mean(1.0 / old_top20_ranks)),
        "old_top20_order_mae": float(np.mean(old_top20_order_abs_error)),
        "old_top20_order_score": old_top20_order_score,
        "old_top250_mean_rank": float(np.mean(old_top250_ranks)),
        "first20_ranked_rows": [int(v) for v in ranked_rows[:20]],
        "old_top20_rank_positions": [int(v) for v in old_top20_ranks],
    }


def generalized_similarity_basis(
    *,
    target_diff: np.ndarray,
    background_diff: np.ndarray,
    n_select: int,
    ridge_scale: float,
) -> np.ndarray:
    target_diff = np.asarray(target_diff, dtype=float)
    background_diff = np.asarray(background_diff, dtype=float)
    dim = target_diff.shape[1]
    Ct = (target_diff.T @ target_diff) / max(1, target_diff.shape[0])
    Cb = (background_diff.T @ background_diff) / max(1, background_diff.shape[0])
    ridge = float(ridge_scale) * max(float(np.trace(Ct)) / float(dim), 1e-12)
    Ct = Ct + ridge * np.eye(dim)
    try:
        L = np.linalg.cholesky(Ct)
        temp = np.linalg.solve(L, Cb)
        A = np.linalg.solve(L, temp.T).T
        A = (A + A.T) / 2.0
        _, eigvecs = np.linalg.eigh(A)
        raw = np.linalg.solve(L.T, eigvecs[:, ::-1])
    except np.linalg.LinAlgError:
        _, _, vh = np.linalg.svd(Ct, full_matrices=True)
        raw = vh[::-1].T
    return orthonormalize_columns(raw[:, :n_select])


def target_null_basis(
    *,
    target_diff: np.ndarray,
    deposit_tied: np.ndarray,
    n_select: int,
) -> np.ndarray:
    _, _, vh = np.linalg.svd(np.asarray(target_diff, dtype=float), full_matrices=True)
    null_like = vh[::-1].T
    columns = []
    dep = np.asarray(deposit_tied, dtype=float)
    if np.linalg.norm(dep) > 0:
        columns.append(dep / np.linalg.norm(dep))
    columns.append(null_like)
    raw = np.column_stack([col if col.ndim == 1 else col for col in columns])
    return orthonormalize_columns(raw[:, :n_select])


def random_orthonormal_basis(
    rng: np.random.Generator,
    n_dim: int,
    n_select: int,
) -> np.ndarray:
    raw = rng.normal(size=(n_dim, n_select))
    return orthonormalize_columns(raw)


def perturb_full_basis(
    full_basis: np.ndarray,
    rng: np.random.Generator,
    *,
    steps: int,
    angle_scale: float,
) -> np.ndarray:
    q = np.array(full_basis, dtype=float, copy=True)
    n_dim = q.shape[0]
    for _ in range(int(steps)):
        i, j = rng.choice(n_dim, size=2, replace=False)
        theta = float(rng.normal(scale=angle_scale))
        c = np.cos(theta)
        s = np.sin(theta)
        qi = q[:, i].copy()
        qj = q[:, j].copy()
        q[:, i] = c * qi + s * qj
        q[:, j] = -s * qi + c * qj
    return q


def complete_orthonormal_basis(
    basis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    basis = orthonormalize_columns(basis)
    n_dim, n_select = basis.shape
    raw = np.column_stack([basis, rng.normal(size=(n_dim, n_dim - n_select))])
    q, _ = np.linalg.qr(raw, mode="complete")
    signs = np.sign(np.diag(np.eye(n_dim) @ q))
    signs[signs == 0] = 1
    return q * signs


def complete_basis_from_columns(
    columns: np.ndarray,
    n_dim: int,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    raw = np.column_stack([columns, rng.normal(size=(n_dim, n_select))])
    return orthonormalize_columns(raw[:, :n_select])


def orthonormalize_columns(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(np.asarray(matrix, dtype=float))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def sample_background_indices(
    *,
    n_windows: int,
    excluded: np.ndarray,
    max_background: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.ones(int(n_windows), dtype=bool)
    excluded = np.asarray(excluded, dtype=int)
    excluded = excluded[(excluded >= 0) & (excluded < n_windows)]
    mask[excluded] = False
    available = np.nonzero(mask)[0]
    if available.size <= max_background:
        return available
    return np.sort(rng.choice(available, size=int(max_background), replace=False))


def write_outputs(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    fused: dict[str, Any],
    target: dict[str, Any],
    recovery: dict[str, Any],
) -> None:
    np.savez_compressed(
        output_dir / "best_tied_subspace_basis.npz",
        basis=recovery["best_basis"],
        weights=recovery["best_weights"],
        ranked_rows=recovery["best_ranked_rows"],
        ranked_dists=recovery["best_ranked_dists"],
        eigvals_fused=fused["eigvals_fused"],
    )
    with (output_dir / "best_ranked_rows.csv").open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["rank", "window_row", "distance", "is_paper_top20", "is_paper_top250"])
        target20 = set(int(v) for v in target["ranked_pred_rows"][:20])
        target250 = set(int(v) for v in target["ranked_pred_rows"][:250])
        for rank, (row, dist) in enumerate(
            zip(recovery["best_ranked_rows"][:250], recovery["best_ranked_dists"][:250]),
            start=1,
        ):
            writer.writerow([rank, int(row), float(dist), int(row) in target20, int(row) in target250])

    plot_weights(output_dir / "best_recovered_weights.png", recovery["best_weights"])
    summary = {
        "args": vars(args),
        "fused": {
            key: _jsonable(value)
            for key, value in fused.items()
            if key not in {"Zf", "baseline_ranked_idx", "baseline_ranked_dists", "baseline_weights"}
        },
        "target": _jsonable({key: value for key, value in target.items() if key != "ranked_pred_rows"}),
        "target_first20_ranked_rows": _jsonable(target["ranked_pred_rows"][:20]),
        "recovery": {
            "evaluated_count": recovery["evaluated_count"],
            "baseline": recovery["baseline"],
            "best": recovery["best"],
            "top_candidates": recovery["top_candidates"],
        },
    }
    (output_dir / "rotation_recovery_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n",
        encoding="utf-8",
    )


def plot_weights(output_path: Path, weights: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    weights = np.asarray(weights, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(np.arange(1, weights.size + 1), weights, color="steelblue")
    ax.set_xlabel("Fused PC after candidate rotation")
    ax.set_ylabel("Deposit weight")
    ax.set_title("Best recovered rotated-basis weights")
    ax.set_xticks(np.arange(1, weights.size + 1))
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def strip_arrays(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _jsonable(value)
        for key, value in result.items()
        if key not in {"basis", "weights", "ranked_rows", "ranked_dists"}
    }


def _console_summary(recovery: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluated_count": recovery["evaluated_count"],
        "baseline": recovery["baseline"],
        "best": recovery["best"],
    }


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
