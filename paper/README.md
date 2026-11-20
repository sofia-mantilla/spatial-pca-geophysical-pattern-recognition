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
python case1_uni_repro.py       # Case-1 gates (end 47.0, AUC 57.9, 5 hits)
python run_ablation_checks.py   # Case-2 headline (end 63.7, AUC 106.1, 3 hits)
                                # + ablations and classical baselines, both cases
```

`numpy_repro_concat.py` looks for `data/` beside itself first and falls back
to the repository's `data/` folder automatically.

## Script map

**Core replication**

| Script | Reproduces |
|---|---|
| `numpy_repro_concat.py` | Shared primitives used by everything below |
| `case1_uni_repro.py` | Case 1 (univariate TMI, reference Paulo Afonso, k=17): recovery, AUC, hits; assertion gates in `__main__` |
| `run_corrections.py` | Case-2 α/k exploration record (alpha sweep, nested LODO, bootstrap significance, greenfield fixed-α); writes `results/*.csv` |
| `null_expectation.py` | Exact random-selection expectation and area budget a(t) (Methods §3.5); 5–95% band |

**Appendix ablations and checks**

| Script | Reproduces |
|---|---|
| `run_ablation_checks.py` | **Case-2 headline numbers** (concat, TMI k=2 / U k=8, α=0.503: end 63.7, AUC 106.1, 3 hits) + weight-variant ablation (both cases), joint concat-PCA disclosure, Paulo Afonso check, classical baselines (demeaned match, correlation match, raw) |
| `run_weight_ablation_area.py` | Appendix C weight-ablation table with the area budget: four weightings (deposit-based, standardized scores, uniform, whitened 1/λ) × both cases, each with c̄(250), AUC, hits, a(250) and the ratio to a random selection of equal area; also the variance-ledness of the published weights and the distinct-group counts of the top 250 (44 and 49) |
| `run_refswap_case1.py` | Reference-swap: every deposit as reference (Case 1, k=17) |
| `alpha_from_univariate.py` | Balance weight α set from univariate performance |
| `concat_alpha_lodo_select.py` | α selection by leave-one-deposit-out CV with a permutation-null gate |
| `disentangle_significance.py` | Held-out significance / bootstrap of the k-choice advantage |

**Major revision, November 2026** — the checks added for the reviewers' comments

| Script | Reproduces |
|---|---|
| `run_rotation_scale_case1.py` | **R2.3**: Case-1 ranking with the reference window rotated (-55° to +90°) and the reference body rescaled (0.35-1.25), each scored with c̄(250), hits, a(250) and the ratio to a random selection of equal area (Table 6) |
| `reference_geometry_ranges.py` | **R2.3**: the rotations and scales the Carajás deposits themselves justify, which set the ranges tested above |
| `run_all_twelve_case2.py` | **R2.4**: the published Case-2 ranking rescored, unchanged and not re-tuned, against all twelve deposits instead of the five of the multivariate set |
| `run_weight_ablation_area.py` | Appendix C weight ablation with the area budget (see above) |
| `run_magnetite_units.py` | **R2.6**: whether the ranking finds mapped magnetitite and iron formation rather than mineralisation (Table 8) |
| `run_era_shares.py` | **R2.6**: the age of the ground the ranking selects, against the survey as a whole (Table 8) |
| `run_palaeo_units.py` | **R2.6**: which Palaeoproterozoic units carry the Case-2 age enrichment, unit by unit |
| `run_contact_proximity.py` | **R2.6**: distance from the high-ranking windows to mapped geological contacts |
| `run_structure_proximity.py` | **R2.6**: distance from the high-ranking windows to mapped shear zones, which tests a co-author's structural reading and does not confirm it. Magnetic lineaments are excluded as circular |
| `run_occurrence_proximity.py` | **R2.6**: distance to mapped Cu-Au mineral occurrences, against an Fe-Mn control; the register is too sparse to discriminate, and that is the result |
| `make_arim_footprint.py` | Reduces the ARIM lithology layer to the coverage outline the two proximity scripts above restrict their populations to |
| `run_geology_vs_baselines.py` | **R2.6**: every published geological quantity for the wPCA selection, the two classical template baselines and 2,000 random draws of 250 windows, so the maps are read on the same footing as the validation |
| `run_restricted_area.py` | **R2.6**: whether restricting the candidate pool to basement removes the advantage over a random selection |
| `run_highrank_geology.py` | **R2.6**: geology of the high-ranking areas that hold no known deposit |
| `fetch_geology.py` | Retrieves the Serviço Geológico do Brasil maps Appendix C uses, for the study-area bounding box alone |
| — | The ARIM Carajás structures and occurrences have no fetch script: `data/Carajas_Brazil_Structures/arim_estrutura.*` and `arim_recmin.*` are committed, copied from the SGB 2019 preliminary integration map (http://rigeo.cprm.gov.br/jspui/handle/doc/19048) |
| `litho_classes.py` | The lithological grouping applied to the 1:100,000 units in Figure 26; the grouping is ours, not the survey's |
| `deposit_names.py` | The names of the twelve deposits, the single source Table 1 and the figures share so they cannot disagree |

**Figures** (file names as invoked by `main.tex`)

| Script | Figure file(s) |
|---|---|
| `make_fig_studyarea_map.py` | `study_area_location_map` (location map; uses `data/naturalearth/`) |
| `extract_case1_windows.py` + `make_fig_deposit_profiles.py` | `deposit_profiles_windows` |
| `extract_dep_windows.py` + `make_fig_multiprofiles.py` | `multi_profiles_windows` |
| `make_fig_workflow.py` | `workflow_steps_schematic` |
| `make_fig_recovery_null.py` | `case1_recovery_comparison`, `multi_recovery_comparison` (with the exact random expectation and band) |
| `make_fig_recovery_area.py` | `recovery_vs_area` |
| `make_fig_known_deposits_map.py` | `Fig2` (TMI with the twelve deposits labelled; names from `deposit_names.py`) |
| `make_fig_rotation_scale.py` | `Fig25` (rotation and scale sensitivity) |
| `make_fig_litho_combined.py` | `Fig26` (high-ranking windows on the mapped lithology, both cases) |
| — | `make_fig_litho_combined.py --structures` draws the ARIM shear zones over both panels and labels the named systems; that is how Fig26 is now generated |
| `make_fig_rotation_scale_explainer.py` | Explainer for the rotation and scale test; not a manuscript figure |
| `make_fig_geology_targets.py`, `make_fig_litho_targets.py` | Single-case renderings of the high-ranking windows on the mapped geology, superseded in the paper by `make_fig_litho_combined.py` |
| `plot_reconstruction_progression.py` | `case2_reconstruction_progression_tmi`, `case2_reconstruction_progression_u` |
| `paper_figures_corrected.py` | Case-2 figures for the final configuration |
| `make_multi_top_windows_3d.py` | Case-2 combined-ranking top-windows 3D figure (companion figure, not in the current manuscript text) |
| `make_fig_appendixB_k_sensitivity.py` | `appendixB_k_sensitivity` (Appendix B k-sensitivity); reads the committed tables in [`appendix_b/`](appendix_b/), regenerate panel (a) with `run_case1_k_sweep.py` |

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
