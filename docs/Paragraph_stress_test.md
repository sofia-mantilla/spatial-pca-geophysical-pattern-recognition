# Paragraph stress test — v10

Every paragraph tested against three questions: (1) does each sentence follow from the previous one, (2) can a first-time reader parse it in one pass, (3) does it obey Jeff's rules (direct, no filler, no "To/Because/In order to" openers, no dangling modifiers). ✅ = passes, ⚠ = fixed in tex, ❗ = needs Sofia.

## Introduction

| ¶ | Verdict | Problem | Fix applied |
|---|---|---|---|
| 1 Targeting stakes | ⚠ | 34-word opening sentence with embedded clause; reader meets three ideas before the verb. | Split: "Mineral exploration targeting identifies and prioritizes locations most likely to contain economically viable deposits. It is a high-impact decision under high uncertainty." |
| 2 Expert workflow → data-driven | ⚠ | "augment this expert interpretation" — antecedent is a *workflow*, not an interpretation. | "augment this expert-led process." |
| 3 Methods survey | ✅ | Compressed already; links forward to the gaps. | — |
| 4 Two gaps | ✅ | Clean parallel structure (First… Second…). | — |
| 5 Extent→geometry→patterns | ⚠ | Jumps to "Boreholes define…" with no signal that the proposal starts here; sPCA acronym and IOCG appear unexpanded. | Added lead-in "This paper proposes a method that closes both gaps." + expanded "spatial principal component analysis (sPCA)" at first use; IOCG expanded at first use in ¶7. |
| 6 Search template | ⚠ | "window" used before it is introduced; "three steps" mislabels what ¶5 presented as concepts. | "These three concepts…"; window introduced explicitly: "The extent sets the size of a search window that slides across the study area." |
| 7 Case study | ✅ | One sentence, concrete. | — |

## Real case introduction

| ¶ | Verdict | Problem | Fix applied |
|---|---|---|---|
| 1 Study area | ✅ | Fine; "We aim to detect…" is a tolerable echo of the intro. | — |
| 2 Paulo Afonso | ⚠ | "By adopting it as the reference geometry, the analysis is restricted…" — dangling modifier (the analysis didn't adopt anything). | "Adopting this dipole as the reference geometry restricts the search to areas expressing a similar one." |
| 3 Windows are the data | ⚠ | "A window is placed… these windows" — number mismatch. | "A window is placed over each known deposit; these deposit windows are the data used throughout the paper." |
| 4 Other deposits | ⚠ | Name spelling inconsistent (Tucumá / Tucuma / Tucumã). | Standardized to **Tucumã** in text. ❗ Map label says "Tucuma" (plain ASCII) — tell me if you want the maps regenerated with accents. |
| 5 Alemão / U vs K,Th | ⚠ | "…which is why U is used rather than K or Th (c.17)" reads like an answer to a reviewer, not prose. | "Alemão contains minor uraninite, so elevated radiometric U — rather than K or Th — provides a geologically meaningful complementary signal." |

## Methodology (rebuilt around the toy example — every step: concept → equation → toy illustration)

| ¶ | Verdict | Problem | Fix applied |
|---|---|---|---|
| Overview 6 steps | ⚠ | "which is slid over the study area" (awkward passive); no signal of how the section will teach. | "which slides across the study area" + explicit roadmap: "the rest of this section explains each step on a synthetic example small enough to display every window, every score, and every weight; Section 4 applies the method to the real cases." |
| Window size | ❗⚠ | **Dimension-order inconsistency:** methodology said "$w_y \times w_x$ … gives $46\times28$", Results says $w_y=28$, $w_x=46$. | Standardized everywhere to $w_x=46$, $w_y=28$ ("46 × 28 pixels, E–W × N–S"). Double-check against the code convention. |
| Sampling + toy | ✅ | Equations → toy walkthrough: 10×10 grid, 2×2 windows ($p=4$), stride 1, $n=81$, deposit = window 40, shown as one row of $X$. The reader sees the entire data matrix. | — |
| Multivariate extension | ✅ | Kept after the univariate toy so the simple case anchors the general one. | — |
| sPCA + toy patterns | ✅ | Each of the four toy geometric patterns is named in words (overall level, bright NE corner, NW–SE contrast, checkerboard) — the reader can verify each claim against Fig. 5 by eye. Deliberate order: show the patterns, then the math that produces them. | — |
| Weights + toy | ✅⚠ | Variance-vs-relevance was stated abstractly. | Now concrete with toy numbers: PC2 explains more variance (12.8% vs 9.7%) yet PC3 gets 3× the weight (23.2% vs 7.1%) because $z_{d,3}=0.94 > z_{d,2}=0.52$. Punchline: "the weights follow the deposit, not the regional variance." |
| Distance ¶ | ✅ | "geometry compared with geometry" closes the ranking. | — |
| Validation + toy | ✅⚠ | Overlap-vs-hit distinction was defined only in symbols. | Toy walkthrough teaches it: rank-1 window overlaps deposit 48 below 50% → orange, not red; hits at ranks 2, 3, 5; 83% recovered; window 31 overlaps nothing → "on a real grid it would be a new exploration target." Same marker convention as all later figures. |

**Coherence checks on the toy thread:** toy window IDs (0–80, deposit 40) consistent across text, Fig. 4, and Fig. 7 ✅; toy numbers in text match figure annotations (65.3/23.2/7.1/4.5%, 83%, ranks 2/3/5) ✅; the toy teaches with $k{=}K{=}4$ (no truncation), and $k$-selection is explained on the real case where it matters ✅; the real-case 3D loading maps are no longer referenced from Methodology (they belong to Results) ✅.

## Results

| ¶ | Verdict | Problem | Fix applied |
|---|---|---|---|
| Case 1 opening | ✅ | Sequence mirrors the workflow figure. | — |
| Window/extent sentence | ⚠ | "Windows … approximate the extent of the Paulo Afonso dipole" — extent belongs to the ore body, the dipole is its signal; blurs the ladder. | "Windows of 46 × 28 pixels cover the TMI dipole expressed over the deposit extent." |
| k-selection ¶ | ✅ | Reconstruction argument is easy to follow. | — |
| Scores/weights ¶ | ✅ | PC2/PC3 vs PC1 contrast is the key insight and reads clearly. | — |
| Top windows ¶ | ❗ | "[deposit name]" for window 6858 still open. | Left as red TODO. |
| Validation numbers ¶ | ✅ | Baseline defined before the comparison. | — |
| Case 2 fusion ¶ | ✅ | Two-stage description is compact. | — |
| Case 2 numbers ¶ | ❗ | 70.4% (verified) vs 77.7% (v9 insertion) unresolved. | Red TODO stays until you confirm. |

## Discussion

| ¶ | Verdict | Problem | Fix applied |
|---|---|---|---|
| 1 Aim + interpretability | ⚠ | Opens with "This study aimed to…" (aim restated instead of result); 8 sentences, three ideas. | Opens with the result: "Both case studies show that similarity to a single known deposit, measured in geometric-pattern space, yields a quantitative, reproducible, and interpretable targeting workflow." Interpretability split into its own shorter run. |
| 2 Case 1 gains | ⚠ | Repeats Results numbers verbatim (54.1%, 5 hits, AUC) — the reader saw them one page ago. | Numbers kept once, sentence trimmed; emphasis shifted to *why* (well-defined geometry → decomposition suppresses amplitude/noise). |
| 3 Case 2 progression | ✅ | Verified numbers; two-conclusions structure is clear. | — |
| 4 Complementarity + limitation | ⚠ | One paragraph carries four moves (complementarity, geology, no 1:1 correspondence, limitation). | Split into two: complementarity/geology; then the practical limitation. |

## Global checks

- "To/Because/In order to" sentence openers: **0 remaining** (c.24). ✅
- "We aim/intend to show": 0 remaining. ✅
- "geometric pattern" used consistently; "spatial pattern/structure/component/filter": 0 remaining. ✅
- Acronyms expanded at first use: TMI, IOCG, sPCA. ✅
- Figure references in order of appearance. ✅
- ❗ For Sofia: 6858 deposit name; 70.4 vs 77.7; map-label accents; $w_x/w_y$ convention vs code.
