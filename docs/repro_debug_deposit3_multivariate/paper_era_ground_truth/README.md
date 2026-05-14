# Paper-Era Ground Truth: Deposit 3 Multivariate SPCA

This folder freezes the recovered paper-era artifacts for the Carajas Deposit 3
multivariate Spatial PCA case:

- Variables: TMI + Radiometric_U
- Ranking mode: `two_stage_pca_fusion`
- Rotation: 0 degrees
- `min_cover`: 0.5
- `k_pcs_var1`: 2
- `k_pcs_var2`: 34
- `k_pcs_fused`: 17

These artifacts should be treated as the provenance target for the paper-era
result. The current `outputs_v5` summary CSV was later regenerated or
overwritten and does not match this restored pickle/figure.

## Source Artifacts

| File | Source |
|---|---|
| `kpcs_best_multicriteria_by_deposit_spca_multi_two_stage_pca_fusion.csv` | `/Users/sofiamantillasalas/Downloads/kpcs_best_multicriteria_by_deposit_spca_multi_two_stage_pca_fusion.csv`; byte-identical to `outputs_v4/.../kpcs_best_multicriteria_by_deposit_spca_multi_two_stage_pca_fusion.csv` |
| `validation_topk_results.pkl` | `outputs_v4/Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17/validation_topk_results.pkl` |
| `Fused_dep_3_rot_0_deg_weights_usedK.png` | `outputs_v4/Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17/Fused_dep_3_rot_0_deg_weights_usedK.png` |
| `Spatial_PCA_config.txt` | `outputs_v4/Deposit_3_Spatial_PCA_Multi_TMI_Radiometric_U_0_deg_0.5_minCov_kpcs_17/Spatial_PCA_config.txt` |

## SHA-256

```text
fc571b8364b4ae96a5c91b41c6da46e4522e1800bcbb7cd8841381fd79aed85c  Fused_dep_3_rot_0_deg_weights_usedK.png
3f2b6c67bbe969224071ad6f52a7dd26591b42c372bf7eb43e80086ba103383f  Spatial_PCA_config.txt
5c0c3af6a5946138811b07343badd84299dad453319f9d308c5b5decf673c778  kpcs_best_multicriteria_by_deposit_spca_multi_two_stage_pca_fusion.csv
b9f81f64342f2f23e3091cefe5639ed3fa3b1b962817a9602bba526fbce5bba6  validation_topk_results.pkl
```

## Key Validation Values

```text
ranking_mode = two_stage_pca_fusion
k_pcs_var1 = 2
k_pcs_var2 = 34
k_pcs_fused = 17
recovery_end_250 = 0.7770279819273643
auc_250 = 128.88557758754246
first_hit_rank_by_deposit = {"3": 11, "1": 29, "0": 48, "4": 208}
```

First 20 `ranked_pred_rows`:

```text
[2774, 7912, 539, 1219, 2615, 3466, 3038, 1380, 8885, 2667,
 1539, 1696, 7419, 5636, 1858, 4264, 3093, 2878, 6806, 54]
```
