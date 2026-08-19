# `scripts/dev/` — developer and reproducibility-debugging scripts

These scripts are **not part of the documented workflow** and are not needed to
run the method or reproduce the paper. They are kept for provenance: they were
used during development to replay paper-era code, debug specific reproductions,
run experimental window shapes, or rebuild internal sweep summaries. Several
depend on local, gitignored artifacts (saved pickles, `local_*` configs, the
analysis worktree) and will not run from a clean public clone.

For the supported entry points, use:

- `scripts/run_project_from_config.py` — the config-driven wPCA pipeline
- `scripts/run_synthetic_smoke_test.py` — quick environment check
- the `paper/` folder — the authoritative scripts behind the manuscript's
  numbers, tables, and figures
