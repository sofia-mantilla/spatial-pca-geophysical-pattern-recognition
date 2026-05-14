# Reproducibility debug: Carajas Deposit 3 multivariate TMI + Radiometric_U

Date: 2026-05-14

Case:

- Deposit: 3 in paper labels; internal zero-based `deposit_ID=2`
- Workflow: multivariate Spatial PCA, TMI + Radiometric_U
- Rotation: 0 degrees
- `min_cover`: 0.5
- Fused retained PCs: `kpcs=17`

Compared figures:

- Paper-era figure: `/Users/sofiamantillasalas/Library/CloudStorage/OneDrive-Stanford/Research_Stanford/Research_files/MineralX_research/EroCopper_project/Carajas_maps_and_data/outputs_v5/Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17/Fused_dep_3_rot_0_deg_weights_usedK.png`
- Current figure: `outputs/Output_Carajas_Brazil_Multivariate_TMI_U_Square_TMI2_U34/Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17/component_weights.png`

## Short diagnosis

The mismatch is not caused by changed raster inputs, deposit shapefile, crop polygon, deposit indexing, masking, sliding-window extraction, feature order, or plotting alone.

The first point where the current rerun diverges from the paper artifact is the two-stage fused PCA basis. In the fused PCA, eigenvalues from PC3 onward are effectively tied:

```text
1.0993106746, 1.0925209313, 1.0000757748, 1.0000757748, ...
```

That means the PC axes inside the PC3+ subspace are not uniquely defined. The workflow then keeps only the first 17 fused PCs and computes deposit weights per PC axis using `z_dep**2`, normalized over the selected PCs. Because those PC axes can rotate under tied eigenvalues, the deposit score vector, component weights, and weighted distances can change even when the input matrix is identical.

Classification against the requested options:

- A changed input data: no
- B changed preprocessing/window extraction: no
- C changed PCA standardization: no clear evidence; the relevant two-stage standardization matches the recovered code path
- D changed weight formula: not the main cause; current and recovered two-stage code both use square-score weights by default
- E changed K/PC selection: the selected `K=17` is correct, but truncating inside a tied fused PCA subspace makes the result basis-dependent
- F plotting only: no
- G true bug/reproducibility weakness: yes, in the two-stage fused PCA ranking definition when eigenvalues are degenerate/tied

## Debug artifacts added

- Script: `scripts/debug_deposit3_multivariate_repro.py`
- Legacy replay harness: `scripts/replay_old_paper_code_dep3.py`
- Fresh diagnostics JSON: `docs/repro_debug_deposit3_multivariate/deposit3_multivariate_repro_diagnostics.json`
- Fresh component-weight plot: `docs/repro_debug_deposit3_multivariate/component_weights_debug.png`

Run the diagnostic reproduction:

```bash
.venv/bin/python scripts/debug_deposit3_multivariate_repro.py
```

Compatibility options are now explicit in `src/spatial_pca/spca/ranking.py` and can be supplied through an optional config section:

```yaml
ranking:
  weight_mode: "square"                  # or "abs"
  normalize_weights_over: "selected_pcs" # or "all_pcs"
  use_whitening: false
  use_weights: true
  stage1_pca_svd_solver: "auto"
  fused_pca_svd_solver: "auto"
```

The defaults preserve the current/paper-intended behavior: square-score deposit weights, normalized over selected PCs, no whitening for two-stage ranking, weighted L2 distance.

The Codex history documents in `docs/First_chat_with_codex.pdf` and `docs/Second_chat_with_codex.pdf` explain the workflow evolution from shared/combined multivariate PCA to separate PCA fusion and then two-stage PCA fusion. The saved paper-era pickle already reports `ranking_mode = two_stage_pca_fusion`, so the current mismatch is within that final two-stage method, not merely because the current repo switched away from the original combined PCA method.

The restored `outputs_v5` case generated on 2026-03-11 14:44 is byte-identical to the previously checked `outputs_v5_single_cases` copy for the key artifacts:

| Artifact | SHA-256 |
|---|---|
| `outputs_v5/.../Fused_dep_3_rot_0_deg_weights_usedK.png` | `aad82eac6bd99fd6667c64c44c3af7405d33730153ebb2b6c30de488f5aeba70` |
| `outputs_v5/.../validation_topk_results.pkl` | `b9f81f64342f2f23e3091cefe5639ed3fa3b1b962817a9602bba526fbce5bba6` |

The committed paper-code snapshot closest to that generation time is `a34fe87` from 2026-03-11 12:59. Its `rank_multi_two_stage_pca_fusion` implementation matches the inspected current paper-code implementation for the relevant choices: `PCA(n_components=M)`, per-variable `ddof=0`, fused-column `ddof=0`, `standardize_fused_input=True`, `use_whitening=False`, `use_weights=True`, and square-score weights. That means the restored artifacts give the old target ranking, but not a saved implementation difference that directly explains the old fused PCA basis.

Local recovery checks:

- The code project is under `/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project`, not OneDrive.
- `git reflog` shows the closest committed state before the restored run is `a34fe87` at 2026-03-11 12:59:52.
- `git stash list` is empty.
- `tmutil listlocalsnapshots /` reported no local snapshots.
- A local search did not find saved fused PCA state files such as `Zf`, `weights_fused`, `.npz`, `.joblib`, or similar artifacts outside the existing figures/pickles.

Legacy replay results using `/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/my_env/bin/python` and the restored v5 config:

| Replay candidate | Prefix match with old ranked rows | Top-20 overlap | Result |
|---|---:|---:|---|
| Legacy code, sklearn default/auto PCA | 1 | 10/20 | Reproduces current ranking, not old v5. |
| Forced `svd_solver="covariance_eigh"` | 1 | 10/20 | Same as current/auto. |
| Forced `svd_solver="full"` | 1 | 1/20 | Different, but not old v5. |
| Forced `svd_solver="randomized"`, seed 0 | 1 | 2/20 | Different, but not old v5. |

The replay harness wrote:

- `docs/repro_debug_deposit3_multivariate/old_paper_code_replay_diagnostics.json`
- `docs/repro_debug_deposit3_multivariate/old_paper_code_replay_covariance_eigh.json`
- `docs/repro_debug_deposit3_multivariate/old_paper_code_replay_full.json`
- `docs/repro_debug_deposit3_multivariate/old_paper_code_replay_randomized0.json`

## Old vs current settings

| Check | Paper-era artifact / old workflow | Current repo / rerun | Conclusion |
|---|---|---|---|
| TMI raster | `1097_1125_1129_TMI_merged(.ers)` in OneDrive Carajas data | `data/Carajas_Brazil_Univariate_TMI/1097_1125_1129_TMI_merged(.ers)` | Same SHA-256 for binary and ERS. |
| Radiometric_U raster | `1097_1125_1129_RAD_eU_merged(.ers)` in OneDrive Carajas data | `data/Carajas_Brazil_Multivariate_TMI_U/1097_1125_1129_RAD_eU_merged(.ers)` | Same SHA-256 for binary and ERS. |
| Crop polygon | `Carajas_maps_and_data/Multiscale_demo/Demo_area_polygon.shp` | `data/Carajas_Brazil_Univariate_TMI/Demo_area_polygon.shp` | Same SHA-256 for `.shp`; the selected crop is the same. |
| Deposit targets | `Prospect_in Carajas_multi.shp` | `data/Carajas_Brazil_Multivariate_TMI_U/Prospect_in Carajas_multi.shp` | Same SHA-256 for `.shp`. |
| Deposit indexing | `Spatial_PCA_config.txt` has `deposit_ID: 2` | Config uses `deposit_1based: 3`, code passes index `2` to extraction | Same deposit polygon. |
| CRS | `force_crs: EPSG:32722` | `force_crs: EPSG:32722`; target policy `assume_raster` for case extraction | Same effective case CRS. |
| Nodata | Old config says `nodata_to_nan: None` | Current config maps `-99999.0` to NaN | No difference in the rebuilt final matrix for this case. |
| Raster alignment | Rebuilt old-like/current arrays compare equal | Current validates CRS, shape, and transform across variables | No mismatch found. |
| Rotation | 0 degrees, `reshape: True` | 0 degrees, same resulting template shape | Same effective 21 x 27 template. |
| Sliding window | `stride_x=8`, `stride_y=8` | `stride_x=8`, `stride_y=8` | Same 13,197 valid windows plus appended deposit row. |
| Feature order | TMI then Radiometric_U | TMI then Radiometric_U | Same order; 567 features per variable. |
| Data matrix | Rebuilt old-like matrix equals current matrix | `(13198, 1134)` | Data/preprocessing are not the cause. |
| Two-stage K | Old pickle: `K_var1=2`, `K_var2=34`, `K_fused=17` | Current debug: `K_var1=2`, `K_var2=34`, `K_fused=17` | Same K settings. |
| Fused standardization | Old pickle: `standardize_fused_input=True` | Current debug: `standardize_fused_input=True` | Same. |
| Weight mode | Recovered code path uses `z_dep**2` | Current default `weight_mode="square"` | Same formula by default. |
| Weight normalization | Recovered code path normalizes over selected fused PCs | Current default `normalize_weights_over="selected_pcs"` | Same by default. |
| Ranking whitening | Old two-stage path recovered as `use_whitening=False` | Current default `use_whitening=False` | Same by default. |
| First 20 ranked rows | `[2774, 7912, 539, 1219, 2615, 3466, 3038, 1380, 8885, 2667, 1539, 1696, 7419, 5636, 1858, 4264, 3093, 2878, 6806, 54]` | `[2774, 1219, 3466, 4413, 7419, 539, 5636, 7415, 1710, 2615, 7912, 3093, 9204, 3459, 4113, 1538, 2457, 6806, 4260, 7435]` | Ranking differs after fused PCA basis/weights. |

SHA-256 values checked for current repo inputs:

| File | SHA-256 |
|---|---|
| TMI binary | `ac78058e08e6861953d2fb7004c0403e69f09799ad3e69f0d1427cb23d247d1e` |
| TMI ERS | `987de41097ecaea367d457dde48a883694156fa79416e6e8f0c83fe854c12fd3` |
| Radiometric_U binary | `783d7befd61bfd7d95804010fe64ed92512bed13c62d05ba7cb832c67c636125` |
| Radiometric_U ERS | `d343e894e77aa1c637776992759c1d3b835138164e2135be857e5eda1c86317e` |
| Demo crop `.shp` | `0bd307baa370cff5920229a35251abff65bc88035cd7d1ff4073515eb329b2d8` |
| Deposit target `.shp` | `1e458dc3a54a1763a39d108478e1f359c8d3c223154f9641f45e39134a333c06` |

## Required diagnostics from the current reproduction

These values were saved by `scripts/debug_deposit3_multivariate_repro.py`.

| Diagnostic | Value |
|---|---|
| `data_for_pca` shape | `(13198, 1134)` |
| `deposit_index` | `13197` |
| `window_shape` | `(21, 27)` |
| Number of valid windows | `13197` |
| Feature order | TMI block first, Radiometric_U block second |
| Features per variable | `567` |
| Weight basis | `square`, from raw unwhitened fused PCA scores |
| Weight normalization | `selected_pcs` |
| Distance ranking whitening | `False` |
| Distance ranking weights | `True` |

First 20 raw fused deposit PCA scores:

```text
[0.1001742597, 1.2943782195, -8.6701475526, -1.6416629175,
 -9.9254542570, 4.0418828685, -5.9090812624, -6.3612666212,
 5.5229680815, -3.8892993145, -0.5852567570, -6.0546720094,
 3.7233578946, 1.8100133987, -6.2788854560, 2.5327506104,
 -2.7082649688, -2.7820743700, 6.4090870720, -9.7864563278]
```

First 20 fused eigenvalues:

```text
[1.0993106746, 1.0925209313, 1.0000757748, 1.0000757748,
 1.0000757748, 1.0000757748, 1.0000757748, 1.0000757748,
 1.0000757748, 1.0000757748, 1.0000757748, 1.0000757748,
 1.0000757748, 1.0000757748, 1.0000757748, 1.0000757748,
 1.0000757748, 1.0000757748, 1.0000757748, 1.0000757748]
```

First 17 weights for `K=17`:

```text
[0.0000237382, 0.0039633099, 0.1778232813, 0.0063753439,
 0.2330431957, 0.0386458701, 0.0825991480, 0.0957244412,
 0.0721573723, 0.0357831342, 0.0008102677, 0.0867195242,
 0.0327948168, 0.0077499557, 0.0932611494, 0.0151747135,
 0.0173507378]
```

Sum of weights: `1.0`

Top weighted PCs, 1-based:

```text
[5, 3, 8, 15, 12, 7, 9, 6, 10, 13, 17, 16, 14, 4, 2, 11, 1]
```

First 20 ranked window rows:

```text
[2774, 1219, 3466, 4413, 7419, 539, 5636, 7415, 1710, 2615,
 7912, 3093, 9204, 3459, 4113, 1538, 2457, 6806, 4260, 7435]
```

First 20 ranked distances:

```text
[2.0095024897, 5.1579262954, 5.4505900997, 5.4960900489,
 5.5291421283, 5.5509640858, 5.5512281645, 5.5950286918,
 5.6025624180, 5.6119448667, 5.6210620142, 5.6518682406,
 5.6697916587, 5.6758066678, 5.7015745306, 5.7019602683,
 5.7297442586, 5.7648844767, 5.7956382610, 5.8109213717]
```

## What is recoverable from the old artifact

The restored old `validation_topk_results.pkl` contains:

```text
ranking_mode = two_stage_pca_fusion
K_var1 = 2
K_var2 = 34
K_fused = 17
standardize_fused_input = True
first20 ranked_pred_rows =
[2774, 7912, 539, 1219, 2615, 3466, 3038, 1380, 8885, 2667,
 1539, 1696, 7419, 5636, 1858, 4264, 3093, 2878, 6806, 54]
```

The old pickle also stores the 1134-feature appended `deposit_vector`, validation curves, deposit overlap metrics, and the top-250 `ranked_pred_rows`. The top-window shapefile stores only an `id` field plus geometry, with no distances or fused PC scores.

The old pickle does not store the raw fused score matrix, fused PCA components, fused eigenvalues, component weights, or ranked distances, so the exact old PCA basis cannot be reconstructed from saved arrays alone. The old PNG visually confirms a different deposit score vector from the current rerun.

## Interpretation

The two-stage fused input has 36 columns: 2 TMI PCA scores and 34 Radiometric_U PCA scores. After fused input standardization, most fused components have the same variance. PCA is allowed to choose any orthonormal basis inside that tied subspace. The total subspace may be equivalent, but the named axes PC3, PC4, ..., PC36 are arbitrary.

The paper workflow then uses axis-specific quantities:

```text
weights_m = z_dep,m^2 / sum_{m=1..K} z_dep,m^2
distance_i = sqrt(sum_{m=1..K} weights_m * (z_i,m - z_dep,m)^2)
```

With tied eigenvalues, this makes the ranking depend on the arbitrary PCA basis, not just on the fused subspace. This explains why the matrix can be identical while `component_weights.png`, top weighted PCs, and ranked windows differ.

Sign flips alone are not enough to explain the mismatch. If a PCA component only changes sign, `z_dep**2` weights are unchanged and pairwise score differences flip consistently, leaving weighted distances unchanged. The observed difference requires a rotation or reordering inside a tied/near-tied subspace.

## Minimal reproduction cell

Use this command from the repository root:

```bash
.venv/bin/python scripts/debug_deposit3_multivariate_repro.py \
  --config configs/carajas_multi_tmi_u_square_tmi2_u34.yaml \
  --deposit 3 \
  --kpcs 17 \
  --weight-mode square \
  --normalize-weights-over selected_pcs
```

To test the compatibility modes:

```bash
.venv/bin/python scripts/debug_deposit3_multivariate_repro.py --weight-mode abs
.venv/bin/python scripts/debug_deposit3_multivariate_repro.py --normalize-weights-over all_pcs
.venv/bin/python scripts/debug_deposit3_multivariate_repro.py --use-whitening
.venv/bin/python scripts/debug_deposit3_multivariate_repro.py --fused-pca-svd-solver full
```

These modes are for explicit comparison and auditability. They should be reported clearly if used, because they change the scientific definition of component weights or comparison space.

## Recommended next step

For paper reproducibility, keep the old `outputs_v5_single_cases` artifacts as historical outputs and cite the exact saved ranking/figure if needed. For future reruns, either:

1. freeze the PCA solver/library environment and save the fitted fused PCA scores/loadings with each run, or
2. revise the two-stage ranking to be invariant to rotations inside tied eigenspaces, then label it as a methodological update rather than a silent reproduction of the paper figure.

## Isolated old-workflow replay test

An isolated replay was run from the old `paper_figures_and_code` workflow using the one-case Deposit 3 config as the template:

```text
/Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/Code/paper_figures_and_code/uni_multi_rotation_spatial_PCA_config_dep3_spca_multi_k17_v5.json
```

To avoid overwriting the restored paper-era `outputs_v5` folder, the copied config was written to:

```text
docs/repro_debug_deposit3_multivariate/dep3_spca_multi_k17_v5_replay_config.json
```

Only the output base/subdirectory were changed. Required best-k CSVs were copied into the isolated replay output directory:

```text
docs/repro_debug_deposit3_multivariate/old_workflow_replay/outputs_v5_replay_test/
```

The replay command was:

```bash
PYTHONFAULTHANDLER=1 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/spca_mpl_cache \
  /Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/my_env/bin/python -u \
  /Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/Code/paper_figures_and_code/uni_multi_rotation_spatial_PCA_copy.py \
  /Users/sofiamantillasalas/Documents/GitHub_Sofia/EroCopper_Xcalibur_Project/Code/Spatial_PCA_Multivariate_Repo/docs/repro_debug_deposit3_multivariate/dep3_spca_multi_k17_v5_replay_config.json
```

The run completed and reported:

```text
method=Spatial_PCA
mode=sweep_kpcs
deposit=3
k_var1=2
k_var2=34
k_fused=17
data_for_pca shape=(13198, 1134)
deposit_index=13197
window_shape=(21, 27)
```

The replay generated:

```text
docs/repro_debug_deposit3_multivariate/old_workflow_replay/outputs_v5_replay_test/Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17/
```

Comparison against the restored paper-era `outputs_v5` artifact:

| Check | Restored `outputs_v5` | Isolated old-workflow replay | Result |
|---|---:|---:|---|
| `ranking_mode` | `two_stage_pca_fusion` | `two_stage_pca_fusion` | Same |
| `K_var1` | 2 | 2 | Same |
| `K_var2` | 34 | 34 | Same |
| `K_fused` | 17 | 17 | Same |
| `standardize_fused_input` | `True` | `True` | Same |
| `validation_topk_results.pkl` SHA-256 | `b9f81f64342f2f23e3091cefe5639ed3fa3b1b962817a9602bba526fbce5bba6` | `29462a998cd4bd9852903918200002dba2b4c84a2d9f0781bac61874ba5b79e2` | Different |
| `Fused_dep_3_rot_0_deg_weights_usedK.png` SHA-256 | `aad82eac6bd99fd6667c64c44c3af7405d33730153ebb2b6c30de488f5aeba70` | `b719d4868695269e48004c9c47dfb64d1f62f10c875d925d0a3e1b6e9e234d68` | Different |
| First equal ranked rows | 1 | 1 | Diverges after row 1 |
| Top-20 overlap | 10/20 | 10/20 | Partial |
| Top-250 overlap | 118/250 | 118/250 | Partial |

Restored first 20 ranked rows:

```text
[2774, 7912, 539, 1219, 2615, 3466, 3038, 1380, 8885, 2667,
 1539, 1696, 7419, 5636, 1858, 4264, 3093, 2878, 6806, 54]
```

Replay first 20 ranked rows:

```text
[2774, 1219, 3466, 4413, 7419, 539, 5636, 7415, 1710, 2615,
 7912, 3093, 9204, 3459, 4113, 1538, 2457, 6806, 4260, 7435]
```

This replay is important because it uses the old script, old environment, old inputs, old one-case v5 config template, and the same restored best-k CSV values, yet it reproduces the current ranking pattern rather than the restored paper-era ranking. That strengthens the conclusion that the restored `outputs_v5` case likely came from an intermediate state or a numerically different fused PCA basis that was not saved in the pickle.

## March 9 Git snapshot replay

After checking OneDrive version history, no historical copies were available for:

```text
spca_paper_functions.py
uni_multi_rotation_spatial_PCA_copy.py
uni_multi_rotation_spatial_PCA_config.json
```

The next-closest recoverable source was Git commit `9e298ef`, created on 2026-03-09 at 23:03, shortly after the `outputs_v4` Deposit 3 case was generated at 2026-03-09 20:56.

A temporary Git worktree was created at:

```text
/private/tmp/erocopper_9e298ef
```

Only the experiment constants in `uni_multi_rotation_spatial_PCA_copy.py` were changed:

```text
METHOD_NAME = "Spatial_PCA"
ANALYSIS_TYPE = "Multi"
MULTI_RANKING_MODE = "two_stage_pca_fusion"
SWEEP_DEPOSITS_1BASED = [3]
SWEEP_KPCS = [17]
OUTPUTS_SUBDIR = "outputs_v4_replay_9e298ef"
BASE_OUTPUT_DIR = docs/repro_debug_deposit3_multivariate/git_9e298ef_replay
```

The run completed successfully with the expected settings:

```text
k_var1=2
k_var2=34
k_fused=17
data_for_pca shape=(13198, 1134)
deposit_index=13197
window_shape=(21, 27)
```

However, it also reproduced the current/replay ranking, not the restored paper-era ranking.

| Artifact | SHA-256 |
|---|---|
| Restored `outputs_v4`/`outputs_v5` `validation_topk_results.pkl` | `b9f81f64342f2f23e3091cefe5639ed3fa3b1b962817a9602bba526fbce5bba6` |
| Current old-workflow replay `validation_topk_results.pkl` | `29462a998cd4bd9852903918200002dba2b4c84a2d9f0781bac61874ba5b79e2` |
| Git `9e298ef` replay `validation_topk_results.pkl` | `29462a998cd4bd9852903918200002dba2b4c84a2d9f0781bac61874ba5b79e2` |

Restored first 20 ranked rows:

```text
[2774, 7912, 539, 1219, 2615, 3466, 3038, 1380, 8885, 2667,
 1539, 1696, 7419, 5636, 1858, 4264, 3093, 2878, 6806, 54]
```

Git `9e298ef` replay first 20 ranked rows:

```text
[2774, 1219, 3466, 4413, 7419, 539, 5636, 7415, 1710, 2615,
 7912, 3093, 9204, 3459, 4113, 1538, 2457, 6806, 4260, 7435]
```

This rules out a simple recovery path of "run the closest March 9 commit." The restored `outputs_v4` and `outputs_v5` validation pickles are byte-identical, but the reproducible March 9 and March 11 code states both produce the later/current ranking. The missing state is therefore narrower: it is likely an uncommitted working-tree variant before `9e298ef`, or a numerically different fused PCA basis not captured in Git or the saved pickle.

## March 9 best-k CSV from Downloads

The user provided:

```text
/Users/sofiamantillasalas/Downloads/kpcs_best_multicriteria_by_deposit_spca_multi_two_stage_pca_fusion.csv
```

This file is byte-identical to the local `outputs_v4` best-k CSV:

```text
SHA-256 = 5c0c3af6a5946138811b07343badd84299dad453319f9d308c5b5decf673c778
```

It is not identical to the current `outputs_v5` best-k CSV:

```text
outputs_v5 CSV SHA-256 = 0919e053a227f8b39dbc4f354d5387599a0ba3ac2bdc99cd360e024cbbcc0230
```

The March 9 CSV has five rows, one for each Deposit 1-5. Its Deposit 3 row is:

```text
deposit_1based=3
k_pcs=17
k_pcs_var1=2
k_pcs_var2=34
k_pcs_fused=17
auc_recovery=128.88557758754246
recovery_end=0.7770279819273643
red_points_count=4
first_red_rank=11.0
mean_red_rank=74.0
```

Those values are consistent with the restored old `outputs_v4` / `outputs_v5` validation pickle:

```text
cum_recovered_frac_total[249] = 0.7770279819273643
trapz(cum_recovered_frac_total[:250]) = 128.88557758754246
first_hit_rank_by_deposit = {"3": 11, "1": 29, "0": 48, "4": 208}
```

This means the March 9 CSV is a valid summary of the restored/paper-era ranking. The current `outputs_v5` CSV is inconsistent with the restored `outputs_v5` pickle: it contains only Deposit 3 and reports the later/current replay metrics:

```text
auc_recovery=87.78492496892379
recovery_end=0.5019596258741191
red_points_count=3
first_red_rank=41.0
```

Therefore, for provenance of the paper result, the March 9 CSV from Downloads / `outputs_v4` should be treated as the trustworthy best-k summary. The current `outputs_v5` CSV appears to have been regenerated or overwritten after the original paper-era case and should not be used as evidence for the restored `outputs_v5` Deposit 3 pickle/figure.

The paper-era artifacts were copied into a protected ground-truth bundle in this repository:

```text
docs/repro_debug_deposit3_multivariate/paper_era_ground_truth/
```

The bundle contains the March 9 best-k CSV, restored validation pickle, restored fused weights figure, original config text, and a README with hashes and key validation values.
