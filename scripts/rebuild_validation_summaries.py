"""Rebuild sweep summary CSVs from saved validation pickles."""

from __future__ import annotations

import os

from _bootstrap import add_src_to_path


REPO_ROOT = add_src_to_path()
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))


def main(argv: list[str] | None = None) -> int:
    from spatial_pca.comparison.rebuild_summaries import main as rebuild_main

    return rebuild_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
