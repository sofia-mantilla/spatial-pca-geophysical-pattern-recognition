"""Run provenance helpers for reproducible Spatial PCA analyses."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spatial_pca.config import load_run_config


def build_provenance(config: dict[str, Any], command: list[str] | None = None) -> dict[str, Any]:
    """Build a serializable provenance record for a Spatial PCA run."""

    resolved = config.get("resolved", {})
    project_root = Path(str(resolved.get("project_root", Path.cwd()))).expanduser()
    git_info = get_git_info(project_root)
    run = config["run"]
    sweep = config["sweep"]
    analysis = config["analysis_defaults"]

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": command or sys.argv,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "git": git_info,
        "config": {
            "config_path": resolved.get("config_path"),
            "project_root": resolved.get("project_root"),
            "output_dir": resolved.get("output_dir"),
            "absolute_paths": resolved.get("absolute_paths", []),
        },
        "run": {
            "method_name": run["method_name"],
            "run_mode": run["run_mode"],
            "analysis_type": run["analysis_type"],
            "outputs_subdir": run.get("outputs_subdir"),
            "resolved_output_dir_name": Path(str(resolved.get("output_dir", ""))).name
            if resolved.get("output_dir")
            else None,
            "uni_selected_variable": run["uni_selected_variable"],
            "multi_ranking_mode": run["multi_ranking_mode"],
        },
        "analysis_defaults": {
            "variable_1": analysis["variable_1"],
            "variable_2": analysis["variable_2"],
            "rotation_angle": analysis["rotation_angle"],
            "stride_x": analysis["stride_x"],
            "stride_y": analysis["stride_y"],
            "n_top_windows": analysis["n_top_windows"],
            "force_crs": analysis["force_crs"],
            "min_cover": analysis["min_cover"],
        },
        "sweep": {
            "deposits_1based": sweep["deposits_1based"],
            "kpcs": sweep["kpcs"],
            "targets_shp_mode": sweep["targets_shp_mode"],
        },
    }


def write_provenance(provenance: dict[str, Any], output_path: str | Path) -> Path:
    """Write a provenance record as pretty JSON and return the path."""

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(provenance, outfile, indent=2, sort_keys=True)
        outfile.write("\n")
    return path


def get_git_info(project_root: str | Path) -> dict[str, Any]:
    """Collect git commit and dirty-state information for a project root."""

    root = Path(project_root).expanduser()
    commit = _run_git(root, "rev-parse", "HEAD")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git(root, "status", "--short")
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _run_git(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip()
    return output or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Spatial PCA provenance JSON file.")
    parser.add_argument("config_path", help="Path to a JSON or YAML run config.")
    parser.add_argument("--output", help="Optional path for the provenance JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_run_config(args.config_path)
    provenance = build_provenance(config)
    if args.output:
        output_path = write_provenance(provenance, args.output)
        print(f"Wrote provenance: {output_path}")
    else:
        print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
