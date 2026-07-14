# Revision plan for Spatial_PCA_paper_v10

Goal: simpler, more direct text (Jeff), and a sharper "geometry vs geometry" argument built on the 3D PC figures from `Deposit_6_Spatial_PCA_Uni_TMI_0_deg_0.5_minCov_kpcs_17`.

---

## 1. Rewritten key paragraphs (Introduction, final part)

Built on three concepts only — **extent → geometry → geometric patterns** — with "geometric pattern" as the single term used everywhere:

> Boreholes define the **extent** of a known deposit. The geophysical signal measured over that extent has a **geometry**: at Paulo Afonso, a magnetic dipole whose shape and size follow the ore body. Spatial PCA decomposes this geometry into its main **geometric patterns** and measures how strongly the deposit expresses each one.
>
> These three steps turn a single known deposit into a search template. The extent sets the window size. The geometry inside the window is the reference. The geometric patterns, weighted by the deposit, define a quantitative similarity metric: other windows are ranked by how closely their geometric patterns reproduce the deposit's. The comparison is geometry with geometry.
>
> We demonstrate the method in a brownfield case study in the Carajás Mineral Province (Brazil), applying it to magnetic and radiometric data to find IOCG geometries consistent with a known copper deposit.

Why this works:

- Opens with the borehole → extent → geometry → geometric-patterns chain: the training area is defined by what we already know, stated in the first two sentences.
- Kills the repetition of "We aim to show / We intend to show" (3×) — Jeff's directness complaint. State what the method does; the case study is the evidence.
- "Geometry with geometry" appears once, as a punchline.
- "Eigenfrequencies" dropped — new concept, unexplained, invites reviewer objections.

**Terminology rule (apply globally):** always say **"geometric pattern(s)"**. Replace every occurrence of "spatial pattern," "spatial structure," "spatial component," "pattern of pixel-to-pixel variability," "dominant pattern," and "spatial filter" with "geometric pattern" (or "main geometric patterns"). PC loading map = the picture of a geometric pattern; PC score = how strongly a window expresses that geometric pattern; weights = which geometric patterns matter for the deposit. One concept, one name, everywhere — text, captions, and figure titles.

## 2. Use the 3D figures to carry the argument

The 3D surfaces make "geometry" literal — the signal becomes a shape you can see. Suggested figure roles:

1. **`deposit_surface.png`** — introduce at the start of the Methodology (or end of Real Case Intro): "this surface is the geometry we learn." Overlay or inset the borehole-defined footprint outline on the surface so the window-size choice is visibly tied to the drilled extent. This answers Jeff's comment 27 (geometry chooses the window, not the reverse).
2. **`loading_maps_3d.png`** — replace the flat loading maps (current Fig 12). Each PC surface is one geometric pattern; PC2 (N–S ramp) and PC3 (SW–NE fold) are visibly the two halves of the dipole. Caption: "the deposit geometry is assembled from these geometric patterns; z_dep and w show how much of each it uses."
3. **`reconstruction_progression.png`** (or the 3D gif frames for a supplement) — keep for the k=17 choice; it shows the geometry being rebuilt component by component.
4. **`top_similar_windows_3d.png`** — the payoff figure. Other locations reproduce the same surface shape. **Improve before inserting:**
   - Add the training-deposit surface as a reference panel (top-left, labeled), so the reader compares shapes directly.
   - Use one common z-range and view angle for all panels. Right now z spans differ (712 vs 1.69e3 vs 1.25e3), which visually exaggerates or flattens shapes and weakens the geometry-with-geometry point. Either fix z-limits or plot standardized (z-scored) windows and say so.
   - Ranks 1, 2, 6 are the deposit itself (stride overlap) — either drop them from the panel or gray-label them "training deposit," and show the first independent matches (idx 4306/4307 = the deposit at ~rank 3, etc.). Name the known deposits they hit.
5. **`TMI_Top_250_Predicted_Windows.png` + `cumulative_footprint_recovery_fraction.png`** — updated versions of Figs 16–17.

One-sentence bridge to place next to Figure (top_similar_windows_3d): "Each ranked window is a surface built from the same geometric patterns; ranking by d_i orders them from most to least similar to the deposit geometry — geometry compared with geometry."

## 3. Structural fixes from Jeff's comments

- **Intro (c.7):** cut by ~1/3. Keep: targeting is high-stakes → data-driven methods exist → two gaps (few labels; location-only use of known deposits) → proposal. Cut or compress the survey of PCA/t-SNE/UMAP/autoencoders to 2–3 sentences; criticize only what the method fixes.
- **Real Case Intro (c.13, c.14):** move all sliding-window mechanics (window size, stride, 13,031 windows) to Methodology. Here say only: a window is placed on each known deposit; these windows are the data. (c.10: enlarge Fig 1; c.15: confirm deposit names Tucumã, Pedra Branca, Salobo.)
- **Methodology (c.19–21):** replace the synthetic tutorial figure with the real Deposit 6 case — the 3D figures above do exactly this. Fix figure order (Fig 3 vs 4) and mini-fonts.
- **Window-size paragraph (c.23–27):** collapse 5 lines to one: "The window dimensions (w_y, w_x) are set by the ore-body geometry known from drilling; the deposit's TMI dipole at Paulo Afonso gives 46 × 28 pixels (9.2 × 5.6 km)." Direction is deposit geometry → window size.
- **Style pass (c.24):** remove sentence-initial "To…", "Because…", "In order to…" throughout; also purge "We aim/intend to show."
- **Figures (c.30, c.37, c.53):** captions under figures, no text wrapping beside figures, and add a dedicated known-deposits map figure (training + test deposits, labeled) before validation.
- **Sofia's open items:** c.17 (why U and not K/Th — add one justification sentence: uraninite in Alemão), c.36 (m→k), c.40 (edge on circles), c.48 (validation steps in method overview), c.15 (deposit names), fill "[deposit name]" placeholder for window 6858.

## 4. Suggested work order

1. Style + structure pass on Intro and Real Case Intro (biggest rejection risk per Jeff).
2. Swap in 3D figures with the fixes in §2; renumber all figures and captions.
3. Rewrite Methodology overview around the real case (drop synthetic example or move to appendix).
4. Resolve open comments/placeholders; final read for "To/Because/In order to."
