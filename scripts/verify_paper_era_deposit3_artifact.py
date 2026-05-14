"""Verify the frozen Deposit 3 paper-era SPCA artifact.

This script is intentionally artifact-backed. The fitted fused PCA scores and
loadings used for the paper-era run were not saved, so the exact old SPCA basis
cannot be recomputed from the restored output folder alone. What this verifier
does guarantee is that the restored paper-era validation pickle, best-k CSV, and
component-weight figure are the expected immutable provenance target for the
paper result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))

DEFAULT_GROUND_TRUTH_DIR = (
    "docs/repro_debug_deposit3_multivariate/paper_era_ground_truth"
)
DEFAULT_OUTPUT_DIR = (
    "docs/repro_debug_deposit3_multivariate/paper_era_verification"
)
DEFAULT_CURRENT_DIAGNOSTICS = (
    "docs/repro_debug_deposit3_multivariate/deposit3_multivariate_repro_diagnostics.json"
)

EXPECTED_SHA256 = {
    "Fused_dep_3_rot_0_deg_weights_usedK.png": (
        "fc571b8364b4ae96a5c91b41c6da46e4522e1800bcbb7cd8841381fd79aed85c"
    ),
    "Spatial_PCA_config.txt": (
        "3f2b6c67bbe969224071ad6f52a7dd26591b42c372bf7eb43e80086ba103383f"
    ),
    "kpcs_best_multicriteria_by_deposit_spca_multi_two_stage_pca_fusion.csv": (
        "5c0c3af6a5946138811b07343badd84299dad453319f9d308c5b5decf673c778"
    ),
    "validation_topk_results.pkl": (
        "b9f81f64342f2f23e3091cefe5639ed3fa3b1b962817a9602bba526fbce5bba6"
    ),
}

EXPECTED_FIRST20_ROWS = [
    2774,
    7912,
    539,
    1219,
    2615,
    3466,
    3038,
    1380,
    8885,
    2667,
    1539,
    1696,
    7419,
    5636,
    1858,
    4264,
    3093,
    2878,
    6806,
    54,
]
EXPECTED_RANKING_MODE = "two_stage_pca_fusion"
EXPECTED_K_VAR1 = 2
EXPECTED_K_VAR2 = 34
EXPECTED_K_FUSED = 17
EXPECTED_RECOVERY_END = 0.7770279819273643
EXPECTED_AUC = 128.88557758754246
EXPECTED_FIRST_HIT = {3: 11, 1: 29, 0: 48, 4: 208}


@dataclass(frozen=True)
class VerificationResult:
    summary: dict[str, Any]
    strict_failures: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and summarize the frozen Deposit 3 paper-era SPCA artifact."
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=DEFAULT_GROUND_TRUTH_DIR,
        help="Directory containing the frozen paper-era files.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where verification outputs are written.",
    )
    parser.add_argument(
        "--current-diagnostics",
        default=DEFAULT_CURRENT_DIAGNOSTICS,
        help="Optional current-rerun diagnostics JSON to compare against.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write outputs even if expected hashes or metrics do not match.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ground_truth_dir = _resolve_repo_path(args.ground_truth_dir)
    output_dir = _resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = verify_artifact(
        ground_truth_dir=ground_truth_dir,
        output_dir=output_dir,
        current_diagnostics_path=_resolve_repo_path(args.current_diagnostics),
    )
    summary_path = output_dir / "paper_era_verification_summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(result.summary), indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(_jsonable(_console_summary(result.summary, result.strict_failures)), indent=2))
    print(f"Wrote verification summary: {summary_path}")

    if result.strict_failures and not args.no_strict:
        print("Strict verification failed:")
        for failure in result.strict_failures:
            print(f"- {failure}")
        return 1
    return 0


def verify_artifact(
    *,
    ground_truth_dir: Path,
    output_dir: Path,
    current_diagnostics_path: Path | None = None,
) -> VerificationResult:
    files = {
        name: ground_truth_dir / name
        for name in EXPECTED_SHA256
    }
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing paper-era ground-truth file(s): " + ", ".join(sorted(missing))
        )

    hashes = {name: _sha256(path) for name, path in files.items()}
    payload = _load_pickle(files["validation_topk_results.pkl"])
    best_k_row = _load_best_k_row(
        files["kpcs_best_multicriteria_by_deposit_spca_multi_two_stage_pca_fusion.csv"],
        deposit_1based=3,
    )

    y = np.asarray(payload["cum_recovered_frac_total"], dtype=float)
    ranked_rows = np.asarray(payload["ranked_pred_rows"], dtype=int)
    auc = float(np.trapezoid(y, dx=1.0))
    recovery_end = float(y[-1])
    first_hit = {int(k): int(v) for k, v in payload["first_hit_rank_by_deposit"].items()}
    spca_diagnostics = dict(payload.get("spca_diagnostics", {}))

    strict_failures = _check_expected(
        hashes=hashes,
        payload=payload,
        best_k_row=best_k_row,
        ranked_rows=ranked_rows,
        auc=auc,
        recovery_end=recovery_end,
        first_hit=first_hit,
        spca_diagnostics=spca_diagnostics,
    )

    ranked_rows_csv = _write_ranked_rows_csv(
        output_dir / "paper_era_ranked_rows_top250.csv",
        ranked_rows=ranked_rows,
        overlap_by_rank=payload.get("overlap_by_rank", {}),
        hit_by_rank=payload.get("hit_by_rank", {}),
    )
    recovery_plot = _write_recovery_plot(
        output_dir / "paper_era_cumulative_recovery.png",
        ranks=np.arange(1, len(y) + 1, dtype=int),
        recovered_fraction=y,
        hit_by_rank=payload.get("hit_by_rank", {}),
    )

    current_comparison = _load_current_comparison(current_diagnostics_path, ranked_rows)
    summary = {
        "purpose": (
            "Artifact-backed verification of the Deposit 3 paper-era SPCA result. "
            "This verifies the restored output used by the paper; it does not claim "
            "that the current PCA fit can exactly regenerate the unsaved old fused basis."
        ),
        "ground_truth_dir": str(ground_truth_dir),
        "verified_files_sha256": hashes,
        "expected_files_sha256": dict(EXPECTED_SHA256),
        "hashes_match": hashes == EXPECTED_SHA256,
        "validation_pickle": {
            "ranking_mode": payload.get("ranking_mode"),
            "spca_diagnostics": spca_diagnostics,
            "k_pcs": int(payload.get("k_pcs")),
            "k_pcs_var1": int(payload.get("k_pcs_var1")),
            "k_pcs_var2": int(payload.get("k_pcs_var2")),
            "k_pcs_fused": int(payload.get("k_pcs_fused")),
            "window_shape": [int(v) for v in payload.get("window_shape", ())],
            "ranked_rows_count": int(ranked_rows.size),
            "first20_ranked_pred_rows": ranked_rows[:20].astype(int).tolist(),
            "recovery_end_250": recovery_end,
            "auc_250_trapezoid": auc,
            "first_hit_rank_by_deposit": first_hit,
            "hit_by_rank": {
                str(int(k)): [int(x) for x in v]
                for k, v in payload.get("hit_by_rank", {}).items()
            },
        },
        "best_k_csv_deposit3": best_k_row,
        "generated_outputs": {
            "ranked_rows_csv": str(ranked_rows_csv),
            "cumulative_recovery_plot": str(recovery_plot),
        },
        "current_comparison": current_comparison,
        "strict_failures": strict_failures,
        "strict_verification_passed": not strict_failures,
    }
    _write_readme(output_dir / "README.md", summary)
    return VerificationResult(summary=summary, strict_failures=strict_failures)


def _check_expected(
    *,
    hashes: dict[str, str],
    payload: dict[str, Any],
    best_k_row: dict[str, Any],
    ranked_rows: np.ndarray,
    auc: float,
    recovery_end: float,
    first_hit: dict[int, int],
    spca_diagnostics: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    for name, expected_hash in EXPECTED_SHA256.items():
        observed_hash = hashes.get(name)
        if observed_hash != expected_hash:
            failures.append(f"{name} SHA-256 changed: {observed_hash} != {expected_hash}")

    if payload.get("ranking_mode") != EXPECTED_RANKING_MODE:
        failures.append(f"ranking_mode changed: {payload.get('ranking_mode')}")
    if int(payload.get("k_pcs_var1")) != EXPECTED_K_VAR1:
        failures.append(f"k_pcs_var1 changed: {payload.get('k_pcs_var1')}")
    if int(payload.get("k_pcs_var2")) != EXPECTED_K_VAR2:
        failures.append(f"k_pcs_var2 changed: {payload.get('k_pcs_var2')}")
    if int(payload.get("k_pcs_fused")) != EXPECTED_K_FUSED:
        failures.append(f"k_pcs_fused changed: {payload.get('k_pcs_fused')}")
    if int(spca_diagnostics.get("K_var1")) != EXPECTED_K_VAR1:
        failures.append(f"spca_diagnostics.K_var1 changed: {spca_diagnostics.get('K_var1')}")
    if int(spca_diagnostics.get("K_var2")) != EXPECTED_K_VAR2:
        failures.append(f"spca_diagnostics.K_var2 changed: {spca_diagnostics.get('K_var2')}")
    if int(spca_diagnostics.get("K_fused")) != EXPECTED_K_FUSED:
        failures.append(f"spca_diagnostics.K_fused changed: {spca_diagnostics.get('K_fused')}")
    if list(ranked_rows[:20].astype(int)) != EXPECTED_FIRST20_ROWS:
        failures.append("first 20 ranked rows changed")
    if not np.isclose(recovery_end, EXPECTED_RECOVERY_END, rtol=0.0, atol=1e-12):
        failures.append(f"recovery_end_250 changed: {recovery_end}")
    if not np.isclose(auc, EXPECTED_AUC, rtol=0.0, atol=1e-12):
        failures.append(f"auc_250_trapezoid changed: {auc}")
    if first_hit != EXPECTED_FIRST_HIT:
        failures.append(f"first_hit_rank_by_deposit changed: {first_hit}")

    csv_k_fused = int(float(best_k_row.get("k_pcs_fused", "nan")))
    csv_auc = float(best_k_row.get("auc_recovery", "nan"))
    csv_recovery_end = float(best_k_row.get("recovery_end", "nan"))
    if csv_k_fused != EXPECTED_K_FUSED:
        failures.append(f"best-k CSV k_pcs_fused changed: {csv_k_fused}")
    if not np.isclose(csv_auc, EXPECTED_AUC, rtol=0.0, atol=1e-12):
        failures.append(f"best-k CSV auc_recovery changed: {csv_auc}")
    if not np.isclose(csv_recovery_end, EXPECTED_RECOVERY_END, rtol=0.0, atol=1e-12):
        failures.append(f"best-k CSV recovery_end changed: {csv_recovery_end}")
    return failures


def _write_ranked_rows_csv(
    output_path: Path,
    *,
    ranked_rows: np.ndarray,
    overlap_by_rank: dict[Any, Any],
    hit_by_rank: dict[Any, Any],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlap = {int(k): [int(x) for x in v] for k, v in overlap_by_rank.items()}
    hits = {int(k): [int(x) for x in v] for k, v in hit_by_rank.items()}
    with output_path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["rank", "ranked_pred_row", "overlap_deposit_ids", "hit_deposit_ids"],
            lineterminator="\n",
        )
        writer.writeheader()
        for rank, row in enumerate(np.asarray(ranked_rows, dtype=int), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "ranked_pred_row": int(row),
                    "overlap_deposit_ids": " ".join(str(x) for x in overlap.get(rank, [])),
                    "hit_deposit_ids": " ".join(str(x) for x in hits.get(rank, [])),
                }
            )
    return output_path


def _write_recovery_plot(
    output_path: Path,
    *,
    ranks: np.ndarray,
    recovered_fraction: np.ndarray,
    hit_by_rank: dict[Any, Any],
) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(recovered_fraction, dtype=float)
    hit_ranks = np.asarray(sorted(int(k) for k in hit_by_rank), dtype=int)

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=160)
    ax.plot(ranks, y, color="#111827", linewidth=2.0)
    if hit_ranks.size:
        ax.scatter(hit_ranks, y[hit_ranks - 1], color="#c1121f", s=28, zorder=3)
        for rank in hit_ranks:
            for dep_id in hit_by_rank.get(int(rank), []):
                ax.annotate(
                    str(int(dep_id) + 1),
                    xy=(int(rank), float(y[int(rank) - 1])),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#c1121f",
                )
    ax.set_title("Deposit 3 paper-era SPCA recovery")
    ax.set_xlabel("Prediction rank")
    ax.set_ylabel("Cumulative recovered fraction")
    ax.set_xlim(1, int(ranks[-1]))
    ax.set_ylim(0, 1.0)
    ax.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    return output_path


def _load_current_comparison(path: Path | None, paper_ranked_rows: np.ndarray) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False, "path": str(path) if path is not None else None}
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    current = diagnostics.get("current_reproduction", {})
    current_rows = [int(x) for x in current.get("first20_ranked_window_rows", [])]
    paper_first20 = [int(x) for x in paper_ranked_rows[:20]]
    return {
        "available": True,
        "path": str(path),
        "current_first20_ranked_window_rows": current_rows,
        "paper_first20_ranked_pred_rows": paper_first20,
        "top20_overlap_count": len(set(current_rows).intersection(paper_first20)),
        "data_for_pca_shape": current.get("data_for_pca_shape"),
        "deposit_index": current.get("deposit_index"),
        "window_shape": current.get("window_shape"),
        "number_valid_windows": current.get("number_valid_windows"),
        "weight_mode": current.get("weight_mode"),
        "normalize_weights_over": current.get("normalize_weights_over"),
        "use_whitening": current.get("use_whitening"),
        "use_weights": current.get("use_weights"),
    }


def _load_best_k_row(csv_path: Path, *, deposit_1based: int) -> dict[str, Any]:
    with csv_path.open("r", newline="", encoding="utf-8-sig") as infile:
        rows = list(csv.DictReader(infile))
    matches: list[dict[str, Any]] = []
    for row in rows:
        dep_value = row.get("deposit_1based", row.get("deposit", row.get("deposit_id")))
        if dep_value is None:
            continue
        try:
            if int(float(dep_value)) == int(deposit_1based):
                matches.append(row)
        except ValueError:
            continue
    if len(matches) != 1:
        raise ValueError(f"Expected one Deposit {deposit_1based} row in {csv_path}, found {len(matches)}.")
    return matches[0]


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as infile:
        payload = pickle.load(infile)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload in {path}, got {type(payload).__name__}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    v = summary["validation_pickle"]
    lines = [
        "# Paper-Era Deposit 3 Verification",
        "",
        "This folder is generated by `scripts/verify_paper_era_deposit3_artifact.py`.",
        "It verifies the frozen artifact used as the paper-era provenance target.",
        "",
        "## Result",
        "",
        f"- strict verification passed: `{summary['strict_verification_passed']}`",
        f"- ranking mode: `{v['ranking_mode']}`",
        f"- K values: var1=`{v['k_pcs_var1']}`, var2=`{v['k_pcs_var2']}`, fused=`{v['k_pcs_fused']}`",
        f"- recovery end at 250 ranks: `{v['recovery_end_250']}`",
        f"- AUC at 250 ranks: `{v['auc_250_trapezoid']}`",
        f"- first 20 ranked rows: `{v['first20_ranked_pred_rows']}`",
        "",
        "The exact fitted fused PCA basis was not stored in the paper-era pickle,",
        "so this verifier backs the paper from immutable restored artifacts rather",
        "than pretending the current code can recreate an unsaved basis.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _console_summary(summary: dict[str, Any], strict_failures: list[str]) -> dict[str, Any]:
    v = summary["validation_pickle"]
    return {
        "strict_verification_passed": not strict_failures,
        "strict_failures": strict_failures,
        "ranking_mode": v["ranking_mode"],
        "k_pcs_var1": v["k_pcs_var1"],
        "k_pcs_var2": v["k_pcs_var2"],
        "k_pcs_fused": v["k_pcs_fused"],
        "recovery_end_250": v["recovery_end_250"],
        "auc_250_trapezoid": v["auc_250_trapezoid"],
        "first20_ranked_pred_rows": v["first20_ranked_pred_rows"],
        "current_top20_overlap_count": summary["current_comparison"].get("top20_overlap_count"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
