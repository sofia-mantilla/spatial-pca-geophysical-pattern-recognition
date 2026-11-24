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

With several variables, `rank_concat` divides each block's distance by the
median of that block's own distances before taking the weighted average
(`BLOCK_SCALE = "median"`, `ALPHA_TMI = 0.541`). The two blocks' distances are
not commensurable: in Case 2 the uranium distances average 27.6 against 5.4
for the magnetic ones. `scale=None` gives the unscaled weighted sum of squared
block distances, and `scale="iqr"` and `scale="prank"` the two other robust
treatments, so the four can be compared. The Case-2 test set is the five
deposits of `case1_uni_repro.deposits_case2_wgs84()`.

## How to run

Download both Carajás data folders (see the repository README) into `data/`,
then run from inside this folder so sibling imports resolve:

```bash
cd paper
python case1_uni_repro.py       # Case-1 gates (end 47.0, AUC 57.9, 5 hits)
python run_ablation_checks.py   # Case-2 headline (end 58.8, AUC 122.9, 3 hits)
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
| `run_ablation_checks.py` | **Case-2 headline numbers** (concat, TMI k=2 / U k=8, α=0.541, median block scaling: end 58.8, AUC 122.9, 3 hits) + weight-variant ablation (both cases), joint concat-PCA disclosure, classical baselines (demeaned match, correlation match, raw) |
| `run_case2_references.py` | Every area, random and grouping reference quoted for Case 2 in Section 5.2, plus the two blocks' distance distributions; writes `results/case2_references.json` |
| `run_case2_appendixC.py` | Every Case-2 row of the Appendix C weight-ablation and classical-baseline tables and of the joint-decomposition paragraph, each with c(250), AUC, hits, a(250) and the ratio to a random selection of equal area; writes `results/case2_appendixC.json` |
| `run_case2_scalings.py` | The six scale estimators the block scaling of step 5.1 was chosen from, each scored on the five Case-2 test deposits, with how many of the top 250 the ranking shares with each single variable; writes `results/case2_scalings.json`. The median is adopted as the standard robust scale and is fourth of the six by recovery |
| `run_case2_heldout.py` | Appendix A: the held-out protocol for the Case-2 choices (30 reference and test pairs), and with `--grid` the 15 × 15 component-count grid behind Appendix B's panel (b); writes `results/case2_heldout.json`, `results/case2_heldout_grid.json` and `appendix_b/lodo_grid_by_k.csv` |
| `run_weight_ablation_area.py` | Appendix C weight-ablation table with the area budget: four weightings (deposit-based, standardized scores, uniform, whitened 1/λ) × both cases, each with c̄(250), AUC, hits, a(250) and the ratio to a random selection of equal area; also the variance-ledness of the published weights and the distinct-group counts of the top 250 (44 and 49) |
| `run_refswap_case1.py` | Reference-swap: every deposit as reference (Case 1, k=17) |
| `alpha_from_univariate.py` | Balance weight α set from univariate performance |
| `concat_alpha_lodo_select.py` | α selection by leave-one-deposit-out CV with a permutation-null gate |
| `disentangle_significance.py` | Held-out significance / bootstrap of the k-choice advantage |

**Major revision, November 2026** — the checks added for the reviewers' comments

> The geological-map comparisons below (`run_magnetite_units`, `run_era_shares`,
> `run_palaeo_units`, `run_contact_proximity`, `run_restricted_area`,
> `run_highrank_geology`, `run_geology_vs_baselines`, `run_map_agreement`,
> `run_occurrence_proximity`) were run for R2.6 and then **taken out of the
> paper**, which now answers that comment with a co-author's geological reading
> instead. `run_structure_proximity.py` went the same way on 20 November, with
> the structural claim it tested, and `run_geology_arim.py` repeats the test on
> the ARIM 1:250,000 map. All are kept here because they were done and they
> reproduce; none is behind a number in the current text.

| Script | Reproduces |
|---|---|
| `run_rotation_scale_case1.py` | **R2.3**: Case-1 ranking with the reference window rotated (-55° to +90°) and the reference body rescaled (0.35-1.25), each scored with c̄(250), hits, a(250) and the ratio to a random selection of equal area (Table 6) |
| `reference_geometry_ranges.py` | **R2.3**: the rotations and scales the Carajás deposits themselves justify, which set the ranges tested above |
| `run_all_twelve_case2.py` | The Case-2 ranking scored against all twelve deposits instead of the five that pass the uranium threshold, with the single-variable rankings scored the same way |
| `run_weight_ablation_area.py` | Appendix C weight ablation with the area budget (see above) |
| `run_magnetite_units.py` | **R2.6**: whether the ranking finds mapped magnetitite and iron formation rather than mineralisation (Table 8) |
| `run_era_shares.py` | **R2.6**: the age of the ground the ranking selects, against the survey as a whole (Table 8) |
| `run_palaeo_units.py` | **R2.6**: which Palaeoproterozoic units carry the Case-2 age enrichment, unit by unit |
| `run_contact_proximity.py` | **R2.6**: distance from the high-ranking windows to mapped geological contacts |
| `run_structure_proximity.py` | **R2.6**: distance from the high-ranking windows to mapped shear zones, which tests a co-author's structural reading and does not confirm it. Magnetic lineaments are excluded as circular |
| `run_occurrence_proximity.py` | **R2.6**: distance to mapped Cu-Au mineral occurrences, against an Fe-Mn control; the register is too sparse to discriminate, and that is the result |
| `make_arim_footprint.py` | Reduces the ARIM lithology layer to the coverage outline the two proximity scripts above restrict their populations to |
| `run_geology_vs_baselines.py` | **R2.6**: every published geological quantity for the wPCA selection, the two classical template baselines and 2,000 random draws of 250 windows, so the maps are read on the same footing as the validation |
| `run_map_agreement.py` | **R2.6**: whether the three maps used across Appendix C agree — magnetitite outlines, era, and the Table 10 headline re-read off the ARIM 1:250,000 map instead of the 1:100,000 sheets |
| `run_arim_diversity.py` | **R2.6**: how far each selection spreads over the ARIM map's own unit codes, against the twelve deposits and 2,000 random draws. Measures a co-author's claim that a prediction confined to one host lithology would be the suspect outcome; no keyword rule or grouping of ours enters it |
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
