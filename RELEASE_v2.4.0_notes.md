# Release v2.4.0 — prepared 23 November 2026, to be published by Sofia

**Do not tag and push expecting an archive.** Zenodo archives on release
*publication*, not on tag push. Create the release in the browser at
https://github.com/sofia-mantilla/spatial-pca-geophysical-pattern-recognition/releases/new
with tag `v2.4.0` on `main`, paste the body below, and publish.

Before publishing, check:

- `CITATION.cff` says `version: "v2.4.0"` and `date-released: "2026-11-25"`.
  If you publish on a different day, change that date first and commit.
- `main.tex` code-availability reads "release v2.4.0" and "version used here:
  v2.4.0" (already updated).
- The concept DOI in the paper, 10.5281/zenodo.20164768, needs no change: it
  always resolves to the latest release.
- `main` is pushed. The release is cut from whatever `main` points at.

After publishing, confirm the new version DOI appears on the Zenodo record and
that the concept DOI resolves to it. The paper cites the concept DOI only, so
nothing in the manuscript changes after the fact.

---

## Release title

Block scaling in the multivariate combination (step 5.1)

## Release body

Each variable's distance is now brought to a common scale before the variables
are combined.

**The change.** With several variables, step 5.1 computes one weighted distance
per variable and averages them with the balance weights. Those distances were
averaged as they stood, and they are not commensurable. In the Carajás Case-2
configuration the radiometric-uranium distances average 27.6 over the 13,197
windows against 5.4 for the magnetic ones, and the magnetic distribution is the
more skewed of the two. The average was therefore governed by the uranium block
rather than by the balance weight: of the top 250 windows, 212 were also in the
top 250 of uranium alone and 8 were in the top 250 of magnetics alone.

`rank_concat` now divides each block's distance by the median of that block's
own distances before averaging (`BLOCK_SCALE = "median"`). Under the new rule
the top 250 shares 116 windows with magnetics alone and 36 with uranium alone.
`scale=None` gives the previous construction, the weighted sum of squared block
distances, and `scale="iqr"` and `scale="prank"` two other robust treatments,
so the four can be compared. The median is adopted as the standard robust scale
for a skewed distribution, not as the highest scorer; `run_case2_scalings.py`
reports six estimators and the median is fourth of the six.

**Case 2 of the paper.** `ALPHA_TMI` is re-derived from the univariate
recoveries of the five test deposits, 52.5 / (52.5 + 44.6) = 0.541, and Paulo
Afonso enters the Case-2 test set, which is now deposits 1, 2, 4, 5 and 6 of
the twelve-deposit shapefile (`case1_uni_repro.deposits_case2_wgs84`). The
combined ranking recovers 58.8% of the test-deposit footprints at rank 250,
AUC 122.9, with three hits at ranks 11, 17 and 74, over a(250) = 8.5% of the
study area, 6.70 times what a random selection of equal area recovers. Against
all twelve deposits it recovers 29.5%, below magnetics alone at 38.7%.

**New scripts**, each writing a JSON so that no number in the paper's Case-2
text is held only in prose:

- `run_case2_references.py` — the area, random and grouping references of
  Section 5.2, and the two blocks' distance distributions.
- `run_case2_appendixC.py` — every Case-2 row of the Appendix C weight-ablation
  and classical-baseline tables, and the joint-decomposition paragraph.
- `run_case2_heldout.py` — the Appendix A held-out protocol over 30 reference
  and test pairs, and with `--grid` the 15 × 15 component-count grid behind
  Appendix B's panel (b), which also regenerates
  `paper/appendix_b/lodo_grid_by_k.csv`.
- `run_case2_scalings.py` — the six scale estimators, with the top-250 overlap
  against each single variable.

**Case 1 is unaffected.** Its recovery (47.0%, five hits, AUC 57.9), the
rotation and scale tests, the reference-swap table and the contact comparison
all reproduce unchanged. Scaling a single block does not reorder it, so every
univariate ranking in the repository is identical to v2.3.5.

**Compatibility.** `rank_concat(dep, k1, k2, alpha)` returns a different
ordering than in v2.3.5 for any `0 < alpha < 1`. Pass `scale=None` for the old
behaviour.

**The webapp follows the same rule.** `rankWindows` in `webapp/wpca_core.js`
and its inline copy in `webapp/index.html` now divide each variable's distance
by its own median before the weighted average. Before this release they
pooled the variables into one distance, so a multivariate run in the app did
not give the paper's Case-2 ranking. `webapp/validate_gates.cjs` now asserts
the Case-2 result above (58.81%, AUC 122.9, hits at ranks 11, 17 and 74, with
α_TMI = 0.541) alongside the unchanged Case-1 gate. A single-variable run in
the app is unchanged.

**The library offers the published rule.** `spatial_pca.spca.ranking` gains
`rank_multi_scaled_mean`, ranking mode `late_fusion_scaled_mean`, which gives
the same ordering as `rank_concat` on the Case-2 windows. The existing
`late_fusion_sum` mode is unchanged. It divides the squared distances by their
medians and combines them in quadrature, which orders windows differently from
the paper's rule.
