# Rotation Recovery Evidence: Deposit 3 Multivariate SPCA

This folder contains the ordered tied-subspace rotation recovery experiment for
the Carajas Deposit 3 multivariate TMI + Radiometric_U SPCA case.

The experiment tests whether the restored paper-era ranking can be approached
by keeping fused PC1-PC2 fixed and rotating only the tied fused PC3-PC36
subspace. It uses the same ranking definition as the paper-era workflow:

```text
weights_m = z_dep,m^2 / sum_selected(z_dep^2)
distance_i = sqrt(sum_m weights_m * (z_i,m - z_dep,m)^2)
```

Run command:

```bash
.venv/bin/python scripts/recover_deposit3_fused_basis_rotation.py \
  --random-trials 800 \
  --local-trials 1600 \
  --seed 11 \
  --output-dir docs/repro_debug_deposit3_multivariate/rotation_recovery_ordered
```

## Key Result

| Metric | Current basis | Best rotated tied basis |
|---|---:|---:|
| Paper top-20 rows in top 20 | 10/20 | 20/20 |
| Paper top-50 rows in top 50 | 21/50 | 36/50 |
| Paper top-100 rows in top 100 | 46/100 | 59/100 |
| Paper top-250 rows in top 250 | 118/250 | 142/250 |
| Mean rank of paper top-20 rows | 264.75 | 10.5 |
| Max rank of paper top-20 rows | 2361 | 20 |
| Paper top-20 order MAE | 256.85 | 3.4 |

This does not exactly reproduce the paper-era order, but it shows that rotating
only the tied fused PCA subspace is sufficient to recover all paper-era top-20
windows inside the top 20. This supports the diagnosis that the mismatch is due
to non-unique PCA axes in a degenerate/tied fused eigenspace.

## Files

| File | Description |
|---|---|
| `rotation_recovery_summary.json` | Full metrics for baseline, best candidate, and top candidates. |
| `best_tied_subspace_basis.npz` | Best recovered tied-subspace basis, weights, ranked rows, distances, and fused eigenvalues. |
| `best_ranked_rows.csv` | First 250 ranked rows from the best rotated basis, annotated with paper top-20/top-250 membership. |
| `best_recovered_weights.png` | Component-weight plot for the best recovered rotated basis. |
