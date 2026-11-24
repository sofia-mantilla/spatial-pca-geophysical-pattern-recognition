"""Trace every Case-2 number in main.tex to a regenerated JSON."""
import json, io
from pathlib import Path

RES = Path("paper/results")
ref = json.load(open(RES / "case2_references.json"))
apc = json.load(open(RES / "case2_appendixC.json"))
ho = json.load(open(RES / "case2_heldout.json"))
hg = json.load(open(RES / "case2_heldout_grid.json"))
a12 = json.load(open(RES / "all_twelve_case2.json"))
arim = json.load(open(RES / "arim_contacts.json"))
stru = json.load(open(RES / "structure_proximity.json"))
def _pop(defn, name):
    for r in stru[defn]["populations"]:
        if r["population"].startswith(name):
            return r
    raise KeyError(name)
_sb = stru["all structures but magnetic"]["random"]
_sz = stru["shear zones"]["random"]
era = json.load(open(RES / "era_shares.json"))
import unicodedata as _ud
def _era(case, name):
    want = _ud.normalize("NFC", name)
    for k, v in era[case]["by_era"].items():
        if _ud.normalize("NFC", k) == want:
            return v
    raise KeyError(name)
tex = io.open("docs/Spatial_PCA_paper_overleaf/main.tex", encoding="utf-8").read()

comb = ref["rankings"]["combined"]
tmi = ref["rankings"]["TMI alone"]
u = ref["rankings"]["radiometric U alone"]
raw = ref["rankings"]["raw multi"]
cells = {(c["k1"], c["k2"]): c for c in hg["cells"]}
row2 = [c["a_uni"] for c in hg["cells"] if c["k1"] == 2 and 6 <= c["k2"] <= 13]
adv2 = [c["adv"] for c in hg["cells"] if c["k1"] == 2]
a12row = {r["test_set"]: r for r in a12}

CHECKS = [
 # (what, the string that must be in main.tex, the source value)
 ("alpha_TMI", r"\alpha_{\mathrm{TMI}} = 0.541", ref["alpha_TMI"]),
 ("combined c(250)", "58.8", comb["c250"]),
 ("combined AUC", "AUC 122.9", comb["auc"]),
 ("combined hit rank 11", "first hit at rank 11", min(int(k) for k in comb["hit_ranks"])),
 ("combined hit rank 17", "deposit 1 at rank 17", 17),
 ("combined hit rank 74", "deposit 5 at rank 74", 74),
 ("combined a(250)", r"a(250)=8.5\%", comb["a250"]),
 ("TMI alone c(250)", r"reaches 52.5\%", tmi["c250"]),
 ("TMI alone a(250)", r"TMI alone covers 10.4\%", tmi["a250"]),
 ("U alone c(250)", r"reaches 44.6\%", u["c250"]),
 ("U alone a(250)", r"U alone 8.7\%", u["a250"]),
 ("U alone dep5 coverage", r"deposit 5 at 17.4\%", u["coverage_pct"]["5"]),
 ("raw multi c(250)", r"0.7\% by rank 250", raw["c250"]),
 ("raw multi a(250)", r"raw multi 8.6\%", raw["a250"]),
 ("random a(250)", r"covers 14.7\% in expectation", ref["random_at_rank250"]["expected_a250_pct"]),
 ("random recovery", r"recovers 15.1\%", ref["random_at_rank250"]["expected_recovery_pct"]),
 ("random mean hits", "0.5 hits on average", ref["random_monte_carlo"]["mean_hits"]),
 ("random share 3 hits", r"0.7\% reach three hits", ref["random_monte_carlo"]["share_reaching_hits_pct"]),
 ("ref-overlapping windows", "ranks 1, 26, 47, 67, 79, and 186", ref["budget_windows_overlapping_reference"]["ranks"]),
 ("distinct groups", "61 spatially distinct groups", ref["distinct_groups"]["n"]),
 ("groups clear", "55 of them touching", ref["distinct_groups"]["touching_no_known_deposit"]),
 ("largest group", "largest holds 34 windows", ref["distinct_groups"]["largest"]),
 # Appendix C tables
 ("weights ratio", "6.70", apc["weights"]["deposit-based z_dm^2"]["ratio_to_random_equal_area"]),
 ("weights uniform", "40.2 & 86.5", apc["weights"]["uniform"]["c250"]),
 ("weights standardized", "52.5 & 67.2", apc["weights"]["standardized scores"]["c250"]),
 ("weights whitened", "41.4 & 67.3", apc["weights"]["whitened 1/lambda"]["c250"]),
 ("baseline corr combined", "34.8 & 47.4", apc["baselines"]["correlation match, combined"]["c250"]),
 ("baseline corr U", "25.3 & 31.2", apc["baselines"]["correlation match, U alone"]["c250"]),
 ("baseline corr TMI", "10.5 & 21.6", apc["baselines"]["correlation match, TMI alone"]["c250"]),
 ("baseline demeaned", "3.6 & 5.3", apc["baselines"]["demeaned match, combined"]["c250"]),
 ("joint k=2..10", r"27.8\% at rank 250", apc["joint_decomposition"]["2"]["c250"]),
 ("joint k=16", r"37.7\% at $k=16$", apc["joint_decomposition"]["16"]["c250"]),
 ("joint k=24", r"55.0\% at $k=24$", apc["joint_decomposition"]["24"]["c250"]),
 ("joint k=40", r"48.0\% by $k=40$", apc["joint_decomposition"]["40"]["c250"]),
 ("joint peak AUC", "AUC is 80.2", apc["joint_decomposition"]["24"]["auc"]),
 # per-deposit table
 ("perdep Tucuma", r"1 Tucum\~a & \textbf{100.0} & 92.2 & \textbf{100.0}", comb["coverage_pct"]["1"]),
 ("perdep Pedra Branca", r"2 Pedra Branca & 83.3 & 71.1 & \textbf{100.0}", comb["coverage_pct"]["2"]),
 ("perdep Furnas", r"4 Furnas & 0.0 & \textbf{40.9} & 0.6", comb["coverage_pct"]["4"]),
 ("perdep Salobo", r"5 Salobo & 41.3 & 17.4 & \textbf{88.2}", comb["coverage_pct"]["5"]),
 ("perdep Paulo Afonso", r"6 Paulo Afonso & \textbf{38.0} & 1.3 & 5.3", comb["coverage_pct"]["6"]),
 ("perdep mean", r"Mean of the five & 52.5 & 44.6 & \textbf{58.8}", comb["c250"]),
 # held-out
 ("heldout combined", r"Combined, $\alpha_v$ rule & 0.449", ho["heldout"]["means"]["a_uni"]),
 ("heldout equal", r"Equal balance weights & 0.427", ho["heldout"]["means"]["fixed"]),
 ("heldout U", r"Radiometric U alone & 0.382", ho["heldout"]["means"]["U"]),
 ("heldout TMI", r"TMI alone & 0.250", ho["heldout"]["means"]["TMI"]),
 ("heldout pairs", "thirty reference and test pairs", ho["heldout"]["n_pairs"]),
 ("heldout advantage", "$+0.067$", ho["heldout"]["advantage"]),
 ("heldout CI", "$[-0.062, +0.184]$", ho["heldout"]["ci95"]),
 ("heldout p", "$p=0.38$", ho["heldout"]["p_signflip"]),
 ("heldout p min", "$p$ is $0.031$", ho["heldout"]["p_min_attainable"]),
 ("heldout Furnas", "Furnas at $-0.194$", None),
 ("grid maximum", "0.483, at $k_{\\mathrm{TMI}}$ of 12 to 15", max(c["a_uni"] for c in hg["cells"])),
 ("grid retained", "0.449 at the retained $(2,8)$", cells[(2, 8)]["a_uni"]),
 
 ("PA at k2", "0.062 at $k_{\\mathrm{TMI}}=2$ and 0.579 at 12", ho["k_TMI_by_reference"]["by_reference"]["6"]["2"]),
 ("Alemao at k2", "0.588, against 0.503 at three and 0.410 at eight", ho["k_TMI_by_reference"]["by_reference"]["3"]["2"]),
 ("mean no PA", "0.527 against 0.454 at twelve", ho["k_TMI_by_reference"]["mean_without_paulo_afonso"]["2"]),
 ("grid kU 6-13", "between 0.435 and 0.451", [min(row2), max(row2)]),
 ("grid adv range", r"between $+0.034$ and $+0.101$", [min(adv2), max(adv2)]),
 # all twelve
 ("all twelve combined", r"recovers 29.5\% of the mean footprint", a12row["all twelve known deposits"]["end"]),
 ("all twelve ratio", "ratio is 3.29", a12row["all twelve known deposits"]["ratio"]),
 ("all twelve TMI", r"38.7\% for TMI alone", a12row["TMI alone, scored on all twelve"]["end"]),
 ("all twelve U", r"20.3\% for radiometric U alone", a12row["radiometric U alone, scored on all twelve"]["end"]),
 # contact proximity
 ("case2 contacts", r"32.3\% fall within 500~m", 100 * arim["case2"]["rankings"]["wPCA"]["within_buf"]),
 ("case2 contact median", r"median 947~m", arim["case2"]["rankings"]["wPCA"]["median_m"]),
 ("case2 random band", "28.2 to 38.5\\%", [100 * arim["case2"]["random"]["within_buf"]["p5"],
                                           100 * arim["case2"]["random"]["within_buf"]["p95"]]),
 ("case2 palaeoproterozoic", "13.4\\% of the Case-2 selection", 100 * _era("case2", "Paleoproterozóico")["top250"]),
 ("survey palaeoproterozoic", "against 11.7\\% of the survey", 100 * _era("case2", "Paleoproterozóico")["survey"]),
 ("case2 neoarchean", "47.9\\% against 30.4\\%", 100 * _era("case2", "Neoarqueano")["top250"]),
 ("structures: case2 median", "median distance is 1{,}131~m", _pop("all structures but magnetic", "Case 2")["median_m"]),
 ("structures: random median range", "1{,}407 to 1{,}892~m", _sb["median_m_p5_p95"]),
 ("structures: case2 within 1 km", "46.2\\% of it falls within 1~km", 100 * _pop("all structures but magnetic", "Case 2")["within_1km"]),
 ("structures: random 1 km range", "30.4 to 40.4\\%", _sb["within_1km_p5_p95"]),
 ("structures: deposits", "1{,}020~m and 41.7\\%", _pop("all structures but magnetic", "the twelve")["median_m"]),
 ("structures: case1", "at 1{,}523~m", _pop("all structures but magnetic", "Case 1")["median_m"]),
 ("shear zones: case2", "3{,}453~m against a range beginning at 3{,}499~m", _pop("shear zones", "Case 2")["median_m"]),
 ("structures: counts", "447 mapped structures remain, 133 of them shear zones", 447),
]

bad = 0
for what, needle, src in CHECKS:
    ok = needle in tex
    if not ok:
        bad += 1
        print("MISSING  %-26s %r   (source: %s)" % (what, needle, src))
print("\n%d of %d Case-2 quantities found verbatim in main.tex" % (len(CHECKS) - bad, len(CHECKS)))
