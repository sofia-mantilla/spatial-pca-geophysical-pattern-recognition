"""Run a Spatial PCA workflow from a config file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Spatial PCA from a project config.")
    parser.add_argument("--config", required=True, help="Path to a YAML or JSON run config.")
    parser.add_argument("--deposit", type=int, help="Optional 1-based training deposit ID.")
    parser.add_argument("--kpcs", type=int, help="Optional retained PC count.")
    parser.add_argument("--output-dir", help="Optional output directory override.")
    parser.add_argument(
        "--top-k",
        type=int,
        help="Optional number of top-ranked windows to export.",
    )
    return parser


def resolve_config_path(config: str | Path) -> Path:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return config_path


def run_config(args: argparse.Namespace) -> None:
    from spatial_pca.pipeline import run_spca_from_config

    results = run_spca_from_config(
        resolve_config_path(args.config),
        deposit_1based=args.deposit,
        k_pcs=args.kpcs,
        output_dir_override=args.output_dir,
        top_k=args.top_k,
    )
    for result in results:
        print(f"Wrote top windows: {result.top_windows_path}")
        print(f"Wrote config: {result.resolved_config_path}")
        print(f"Wrote provenance: {result.provenance_path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_config(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
