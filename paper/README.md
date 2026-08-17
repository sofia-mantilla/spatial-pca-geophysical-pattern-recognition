# `paper/` — replication and figure scripts for the manuscript

Scripts behind *Geometry-based targeting from a single known deposit using
windowed principal component analysis: an IOCG case study in Carajás, Brazil*
(Mantilla Salas et al., submitted to Natural Resources Research).

These scripts were developed and run in a pinned analysis environment during
manuscript preparation and are published here verbatim as the scientific
record. They are self-contained around `numpy_repro_concat.py`, a compact
NumPy implementation of the full workflow (ERS raster reading, cropping,
shapefile outlines, window extraction, PCA, ranking, footprint recovery) that
was validated against the main pipeline's cached outputs (identical hit ranks
and overlap events; AUC within ±0.02 across 8 cached configs).

## How to run

Download both Carajás data folders (see the repository README) into `data/`,
then run from inside this folder so sibling imports resolve:

```bash
cd paper
python case1_uni_repro.py       # Case-1 gates
python run_corrections.py       # Case-2 final numbers
```

`numpy_repro_concat.py` looks for `data/` beside itself first and falls back
to the repository's `data/` folder automatically.

## Script map

**Core replication**

| Script | Reproduces |
|---|---|
| `numpy_repro_concat.py` | Shared primitives used by everything below |
| `case1_uni_repro.py` | Case 1 (univariate TMI, reference Paulo Afonso, k=17): recovery, AUC, hits; assertion gates in `__main__` |
| `run_corrections.py` | Case 2 final numbers (concat, TMI k=2 / U k=8, balance weight from the other deposits' univariate coverages, reference Alemão) |
| `null_expectation.py` | Exact random-selection expectation and area budget a(t) (Methods §3.5); 5–95% band |

**Appendix ablations and checks**

| Script | Reproduces |
|---|---|
| `run_ablation_checks.py` | Weight-variant ablation (both cases), joint concat-PCA disclosure, classical baselines (demeaned match, correlation match, raw) |
| `run_refswap_case1.py` | Reference-swap: every deposit as reference (Case 1, k=17) |
| `alpha_from_univariate.py` | Balance weight α set from univariate performance |
| `concat_alpha_lodo_select.py` | α selection by leave-one-deposit-out CV with a permutation-null gate |
| `disentangle_significance.py` | Held-out significance / bootstrap of the k-choice advantage |

**Figures** (file names as invoked by `main.tex`)

| Script | Figure file(s) |
|---|---|
| `make_fig_studyarea_map.py` | `study_area_location_map` (location map; uses `data/naturalearth/`) |
| `extract_case1_windows.py` + `make_fig_deposit_profiles.py` | `deposit_profiles_windows` |
| `extract_dep_windows.py` + `make_fig_multiprofiles.py` | `multi_profiles_windows` |
| `make_fig_workflow.py` | `workflow_steps_schematic` |
| `make_fig_recovery_null.py` | `case1_recovery_comparison`, `multi_recovery_comparison` (with the exact random expectation and band) |
| `make_fig_recovery_area.py` | `recovery_vs_area` |
| `plot_reconstruction_progression.py` | `case2_reconstruction_progression_tmi`, `case2_reconstruction_progression_u` |
| `paper_figures_corrected.py` | Case-2 figures for the final configuration |
| `make_multi_top_windows_3d.py` | Case-2 combined-ranking top-windows 3D figure (companion figure, not in the current manuscript text) |

The remaining paper figures (deposit surfaces, component weights, score pairs,
loading maps, top-window maps, reconstruction progression for Case 1, toy
example) are produced directly by the main pipeline
(`scripts/run_project_from_config.py`) and the synthetic demo notebook.

## Environment caveats

- Results in the paper were produced on macOS with the repository environment
  (`requirements.txt` / `environment.yml`).
- A few configurations retain a PCA basis that cuts a tied or degenerate
  eigenvalue block; those are sensitive to BLAS threading and summation order
  across machines. To pin results, set
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before running.
  All headline configurations sit at real spectral gaps and are stable.
- Four scripts (`alpha_from_univariate.py`, `concat_alpha_lodo_select.py`,
  `disentangle_significance.py`, `plot_reconstruction_progression.py`) also
  import `spatial_pca` from `src/`; run them from the repository root with
  `PYTHONPATH=src` or rely on the scripts' own path handling.
