# Changes: Spatial_PCA_paper v9 → v10

Concrete edit list. Each item says what changes, where, and why (Jeff comment # where applicable).

## A. Text replacements

1. **Introduction, final 4 paragraphs** ("This paper proposes… known copper deposit") → replaced by the 3-paragraph extent → geometry → geometric patterns version in `Revision_plan_v10.md` §1. Removes "We aim to show"/"We intend to show" (3×) and "eigenfrequencies."
2. **Terminology sweep (global, incl. captions):** replace with **"geometric pattern(s)"**:
   - "spatial pattern(s)" — 10 occurrences
   - "spatial component(s)" — 8 occurrences (keep "principal component/PC" for the math object; use "geometric pattern" when describing what it shows)
   - "spatial structure(s)" — 3 occurrences
   - "dominant pattern(s)" — 3 occurrences
   - "spatial filter" — 1 occurrence
3. **Window-size paragraph (Methodology)** — collapse 5 lines to: "The window dimensions (w_y, w_x) are set by the extent of the known ore body; at Paulo Afonso this gives 46 × 28 pixels (9.2 × 5.6 km)." (c.23, c.25, c.26, c.27 — direction is extent → window size.)
4. **Style pass (whole paper):** remove sentence-initial "To…", "Because…", "In order to…" (c.24).

## B. Cuts and moves

5. **Introduction cut by ~1/3** (c.7): compress the PCA/factor-analysis and t-SNE/UMAP/autoencoder survey to 2–3 sentences; keep only criticism the method answers (few labels; deposits used as locations, not geometries).
6. **Real Case Introduction** (c.13, c.14): delete the sliding-window mechanics paragraph (window size, strides, 13,031 windows) — moves to Methodology. Replace with one sentence: a window is placed over each known deposit; these windows are the data.
7. **Synthetic tutorial example** (c.19, c.20, c.21): replace with the real Deposit 6 case; move the synthetic 2×2 example to an appendix or delete. Fix Fig 3/4 ordering.

## C. Figures

8. **NEW workflow figure** → `docs/figures/workflow_method_figure.png` (replaces current Fig 3; real case, large fonts — c.21). Six steps: extent → geometry → geometric patterns → weights → search/rank → validation.
9. **Fig 1 (study area):** enlarge (c.10).
10. **NEW known-deposits figure** (c.53): map of training + test deposits, labeled, before validation section.
11. **Fig 12 (loading maps):** replace flat maps with `loading_maps_3d.png`; caption in geometric-pattern language.
12. **Fig 15 (top windows):** replace with `top_similar_windows_3d.png` after fixing: common z-range (or standardized windows), deposit surface as reference panel, gray-label ranks 1/2/6 as the training deposit itself.
13. **All figures:** captions under figures (c.30); no text wrapped beside figures (c.37); circles get edges (c.40).
14. Renumber all figures and in-text references after swaps.

## D. Open items to resolve (Sofia's comments)

15. c.15 — confirm deposit names (Tucumã, Pedra Branca, Salobo) and fill "[deposit name]" placeholder for window 6858.
16. c.17 — add one sentence justifying U over K/Th (uraninite at Alemão).
17. c.36 — m → k notation fix.
18. c.48 — add validation steps to Methodology overview.
19. c.11/c.12 — fold "Due to the Equator" and "Magnetite" notes into the dipole description, then delete comments.
