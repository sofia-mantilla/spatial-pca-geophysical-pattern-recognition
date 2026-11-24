# Appendix B — sensitivity of the rankings to the number of retained components

Source data and scripts for the paper's Appendix B figure (`appendixB_k_sensitivity`).
Both input tables are committed here, so the figure reproduces from a clean clone with
no external data:

    python paper/make_fig_appendixB_k_sensitivity.py

which writes `appendixB_k_sensitivity.pdf` (TrueType embedded, no Type 3) and a 600-dpi
PNG into this folder. It needs only pandas + matplotlib (about 2 s).

## Panel (a) — Case 1, Paulo Afonso: `k_sweep_case1.csv`

The full univariate TMI pipeline rerun for each k in {2..12, 14, 17, 20, 25, 30, 40, 60},
every other setting unchanged. Columns are read straight from each run's
`validation_topk_results.pkl`:

| column | meaning |
|---|---|
| `k_pcs` | retained components |
| `mean_recovered_frac_250` | `cum_mean_recovered_frac[-1]` — the per-deposit metric at rank 250 |
| `auc_recovery` | `sum(cum_mean_recovered_frac)`, the AUC on the 0–250 scale used in the paper |
| `n_hits` | test deposits reaching the 50 % recovered fraction |
| `first_hit_rank` | earliest rank at which any test deposit becomes a hit |
| `deposits_hit_1based` | which test deposits, renumbered from the pickle's 0-based keys |

Regenerate this table (needs the Carajás TMI data, about 20 min):

    python paper/run_case1_k_sweep.py

## Panel (b) — Case 2, Alemão: `lodo_grid_by_k.csv`

The 15 × 15 nested leave-one-deposit-out grid at each (k_TMI, k_U). The panel shows the
section at k_TMI = 2, using columns `a_uni` (combined ranking under the α_v rule) and `U`
(radiometric U alone); the TMI-alone reference line is 0.2498 (the `TMI` column at
k_TMI = 2). Regenerate this table with

    python paper/run_case2_heldout.py --grid

which writes it here and the full per-cell record, intervals included, to
`paper/results/case2_heldout_grid.json` (about 12 min).

## Result

Case 1 recovery rises over the first components, steps up at k = 12, and then does not
change again: k = 12 to k = 60 all give 47.0 % with the same five test deposits hit. Below
the plateau the ranking is worse (38.3 % / four hits at k = 8; 24.2 % / two at k = 2). The
retained k = 17 sits inside the plateau. Locating the plateau's lower edge used all eleven
test deposits; from a single known deposit its position could not have been identified —
this is stated in the appendix.

For Case 2 the combined ranking exceeds both single variables at every k_U from 1 to 15.
Held-out recovery over the thirty reference and test pairs is highest at 0.483, at k_TMI of
12 to 15 with k_U = 9, and reaches 0.449 at the retained (2, 8); raising k_TMI to 3 costs
0.008 at k_U = 8. The grid does not single out the retained counts, which are read from the
reference deposit's reconstructions.
