# Stress-test report — Spatial PCA paper v11

Date: 2026-08-10. Scope: every quantitative claim in `main.tex` (v11), determinism of every ranking, figure–text consistency, vocabulary, style-guide compliance, LaTeX health. Method: independent recomputation in a clean environment from the repo's own code and data, plus traceback to the versioned result tables on the `case2-concat-experiments` branch.

## Verdict

The paper survives. Every number checked traces to a reproducible computation. One factual error was found and fixed during the test (study-area pixel count). Four items remain open; none block the Jef review, one should be closed before submission.

## 1. Case 1 (Paulo Afonso, univariate TMI) — VERIFIED

| Claim in paper | Check | Result |
|---|---|---|
| sPCA k=17: 47.0% end, AUC 57.9, 5/11 hits, first at rank 5 (dep 11) | Recomputed from a fresh pipeline run in a clean environment | Match. Hits: 11@5, 5@103, 8@126, 1@166, 2@220 |
| Raw baseline: 20.7%, AUC 33.5, 1 hit (dep 1, rank 30) | Fresh `Raw_comparison` pipeline run | Match |
| Deposit scores: PC2 z=12.31, PC3 z=12.30, PC1 z=8.82 w=12.5%, PC5 −8.27, PC12 −7.00, PC2/PC3 weights ≈24%, PC7/PC10 near zero, PC1 variance 24.7% | Read off the regenerated `component_weights.png` from the fresh run | Match on all values |
| n=13,031 windows, p=1,288 (46×28) | Pipeline run; arithmetic | Match |
| **Cross-machine reproducibility** | Fresh cloud run vs. the May run on Sofia's machine: full ranking, recovery curves, and hit dictionary compared array-by-array | **Bit-identical.** Different hardware, same result |

Note: the July worklog's raw-benchmark discrepancy (independent reimplementation gave 24.1%) is resolved. The pipeline's own raw benchmark reproduces the paper-era 10.2% (area metric) exactly; the reimplementation was the outlier. The paper's raw numbers stand on the pipeline.

## 2. Case 2 (Alemão, multivariate TMI+U) — VERIFIED

| Claim in paper | Source of truth | Result |
|---|---|---|
| Concat α_uni=0.544: 59.0%, AUC 95.3, hits 1@40, 2@221 | Recomputed in clean environment AND `results/corrected_dep3_table.csv` | Match (α recomputes to 0.5436) |
| TMI k=2: 56.1%, AUC 98.5, hits 1@14, 2@95 | Same | Match |
| U k=6: 47.1%, AUC 93.8, hit 1@56 | Same | Match |
| Raw multi: 0.9%, AUC 0.6, 0 hits | Same | Match |
| LODO coverage: α_uni 0.483 > U 0.430 > fixed-0.5 0.399 > TMI 0.323 | `results/lodo_nested_by_k.csv` (0.4832 / 0.4303 / 0.3987 / 0.3226) | Match |
| Advantage +0.053, CI [−0.011, +0.112], n=5, not significant | `results/significance.csv` (0.0529, [−0.0106, +0.1117], p=0.25) | Match |
| "(2,6) recovers the most held-out area" | `lodo_nested_by_k.csv`: 0.4832 is the max of all 8 tested (k1,k2) configs | Match |
| "Gain disappears as more U components are added" | +0.053 at U=6 → +0.012 at 12 → +0.001 at 16 and 34 | Match, monotone decay |
| "TMI k=3 inverts the gain" (Discussion basis) | (3,6): −0.027; (3,16): −0.055 | Match |
| **Determinism** | Identical config run twice, full-ranking MD5 | **Identical hashes** (512bedc616bb) |

## 3. Synthetic example — VERIFIED

Smoke test (clean environment): 10×10 grid, X is 82×4, deposit = window 40, k=4, top-5 = {40, 12, 52, 31, 48}. Figure `toy_top_windows_and_recovery.png` confirms the paper's narrative point-by-point: endpoint 83%, hits 12/51/48 at ranks 2/3/5, rank-1 an overlap below threshold, window 31 overlapping nothing. Toy weight claims (PC3 w=23.2% vs PC2 7.1%; z 0.94 vs 0.52; variance 9.7% vs 12.8%) match the toy figures in the paper.

## 4. Errors found and fixed during the test

1. **Study-area pixel count (fixed in tex, recommitted).** Paper said 1,296×691 pixels; the measured crop from the demo polygon is **1,297×692** (rows×cols 692×1,297). The km figure (259×138) was already consistent with the corrected count. This was a known nit in the July notes that had not made it into the tex.

## 5. Consistency and style scans — CLEAN

- No "training"/"testing" vocabulary anywhere in prose or captions; regenerated figures carry "Reference deposit"/"Test deposits".
- No "six steps" leftovers; text and workflow figure both say seven steps with 5.1.
- No stray k=34 or two-stage fusion description; "fused/fusion" appears only in the Discussion design-lesson paragraph, deliberately.
- Style guide: zero banned words in prose; no chained transitions; no em dashes in prose (remaining `---` are in LaTeX comments and red margin notes only); flagged not-X-but-Y constructions rewritten.
- All 20 referenced figure files present; LaTeX compiles with 0 errors, 0 undefined references/citations, 25 pages, 2 minor overfull hboxes.

## 6. Open items (not blockers for the Jef review)

1. **Case 1 significance vs. random null** — the historical p=0.037 was computed on the area metric. The paper currently makes no significance claim for Case 1, so nothing is wrong, but if a null-band or p-value is wanted, it must be recomputed on the per-deposit metric. *Close before submission if the claim is added.*
2. **Window 6858 deposit name** — still a red `\todo` in Case 1.
3. **Toy figures not regenerated** — `toy_top_windows_and_recovery.png` legend still reads "Training Known Deposit". The code fix is committed; one rerun of the illustrative example refreshes it.
4. **`top_similar_windows_3d.png`** — pending common z-range, reference panel, grayed reference ranks (existing margin note).

## 7. What was NOT independently reverified

- The LODO and bootstrap tables were traced to the versioned CSVs produced by `run_corrections.py` on 2026-07-16, and spot-checked for internal consistency; the full LODO sweep was not rerun from scratch.
- Geological statements (uraninite at Alemão, survey provenance, deposit names from the shapefile record order) were not re-audited this session; deposit names were verified against `Prospect_in Carajas_v2.dbf` in the July session.
