"""Check the letter against the rebuilt paper and the regenerated JSONs."""
import io, json, re, zipfile
from pathlib import Path

D = Path("docs/Spatial_PCA_paper_overleaf")
RES = Path("paper/results")
xml = zipfile.ZipFile(D / "NRR_response_to_reviewers.docx").read("word/document.xml").decode("utf-8")
letter = " ".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, re.S))
letter = re.sub(r"\s+", " ", letter)
tex = io.open(D / "main.tex", encoding="utf-8").read()

sc = json.load(open(RES / "case2_scalings.json"))
ref = json.load(open(RES / "case2_references.json"))
ho = json.load(open(RES / "case2_heldout.json"))
hg = json.load(open(RES / "case2_heldout_grid.json"))
a12 = {r["test_set"]: r for r in json.load(open(RES / "all_twelve_case2.json"))}
comb = ref["rankings"]["combined"]

# (label, string the letter must contain, string the paper must also contain or None)
SHARED = [
 ("combined c(250)", "58.8%", "58.8"),
 ("TMI alone", "52.5%", "52.5"),
 ("U alone", "44.6%", "44.6"),
 ("alpha", "0.541", "0.541"),
 ("hit ranks", "ranks 11, 17 and 74", "ranks 11, 17 and 74"),
 ("all twelve combined", "29.5%", r"29.5\%"),
 ("all twelve TMI", "38.7%", r"38.7\%"),
 ("Paulo Afonso combined", "5.3%", "5.3"),
 ("Paulo Afonso TMI", "38.0%", "38.0"),
 ("Furnas U alone", "40.9%", "40.9"),
 ("Furnas combined", "0.6%", "0.6"),
 ("heldout combined", "0.449", "0.449"),
 ("heldout U", "0.382", "0.382"),
 ("heldout equal", "0.427", "0.427"),
 ("heldout TMI", "0.250", "0.250"),
 ("heldout advantage", "+0.067", "$+0.067$"),
 ("heldout CI", "[-0.062, +0.184]", "$[-0.062, +0.184]$"),
 ("pairs", "thirty", "thirty"),
 ("grid cells", "225", "225"),
 ("IQR variant", "58.5%", "58.5"),
 ("percentile rank variant", "59.6%", "59.6"),
 ("n windows", "13,197", "13{,}197"),
 ("distance means", "27.6", "27.6"),
 ("distance means TMI", "5.4", "5.4"),
 ("uranium ppm", "7.1 ppm", "7.1~ppm"),
 ("units: Rio Maria", "4.4", "4.4"),
 ("units: Plaque", "20.0", "20.0"),
 ("units: Seringa", "1.5", "1.5"),
 ("units: eight above", "eight units carry", "Eight units carry"),
 ("units: nine below", "Nine units carry less", "Nine units carry less"),
 ("structures: median", "1,131 m", "1{,}131~m"),
 ("structures: band", "1,407 to 1,892 m", "1{,}407 to 1{,}892~m"),
 ("structures: within 1 km", "46.2", "46.2"),
 ("structures: band 1 km", "30.4 to 40.4", "30.4 to 40.4"),
 ("structures: deposits", "1,020 m", "1{,}020~m"),
 ("structures: counts", "447 structures", "447 structures"),
 ("structures: shear count", "133 mapped shear zones", "133 of them shear zones"),
 ("grid maximum", "0.483", "0.483"),
 ("grid retained", "0.449", "0.449"),
]

# letter-only figures, checked against the JSONs
LETTER_ONLY = [
 ("mean scaling", "60.9%", sc["scalings"]["mean"]["c250"], 60.9),
 ("std scaling", "50.5%", sc["scalings"]["standard deviation"]["c250"], 50.5),
 ("overlap published w/ U", "212", sc["unscaled"]["weighted sum of the squared distances"]["shared_with_U_alone"], 212),
 ("overlap published w/ TMI", "8 are in the top", sc["unscaled"]["weighted sum of the squared distances"]["shared_with_TMI_alone"], 8),
 ("overlap new w/ TMI", "116", sc["scalings"]["median"]["shared_with_TMI_alone"], 116),
 ("overlap new w/ U", "36 with uranium", sc["scalings"]["median"]["shared_with_U_alone"], 36),
 ("unscaled with PA", "48.1%", sc["unscaled"]["weighted sum of the squared distances"]["c250"], 48.1),
 ("heldout spread high", "+0.232", None, None),
 ("heldout spread low", "-0.194", None, None),
 ("old submitted values", "63.7%, 56.1% and 55.4%", None, None),
 ("old hit ranks", "10, 142 and 246", None, None),
 ("old all-twelve", "23.3%", None, None),
 ("old alpha", "0.503", None, None),
]

bad = 0
print("=== figures the letter and the paper share ===")
for lab, inl, intex in SHARED:
    a, b = inl in letter, (intex is None or intex in tex)
    if not (a and b):
        bad += 1
        print("  MISMATCH %-24s letter=%s paper=%s  (%r / %r)" % (lab, a, b, inl, intex))
print("=== letter-only figures, against the JSONs ===")
for lab, inl, src, want in LETTER_ONLY:
    if inl not in letter:
        bad += 1
        print("  MISSING IN LETTER %-24s %r" % (lab, inl))
    elif src is not None and abs(src - want) > 1e-9:
        bad += 1
        print("  JSON DISAGREES   %-24s letter says %s, JSON says %s" % (lab, want, src))

# the per-cluster held-out spread the letter quotes
pairs = ho["pairs"]
import collections, statistics
cl = collections.defaultdict(list)
for p_ in pairs:
    cl[p_["t"]].append(p_["a_uni"] - p_["U"])
means = {t: statistics.mean(v) for t, v in cl.items()}
hi_t, hi = max(means.items(), key=lambda kv: kv[1])
lo_t, lo = min(means.items(), key=lambda kv: kv[1])
print("  held-out spread: highest deposit %d at %+.3f, lowest deposit %d at %+.3f"
      % (hi_t, hi, lo_t, lo))
if abs(hi - 0.232) > 5e-4 or abs(lo + 0.194) > 5e-4:
    bad += 1
    print("  JSON DISAGREES with the +0.290 / -0.243 the letter quotes")
if hi_t != 2 or lo_t != 4:
    bad += 1
    print("  the letter names Pedra Branca (2) and Furnas (4); JSON says %d and %d" % (hi_t, lo_t))

print("\n%s" % ("ALL CHECKS PASS" if bad == 0 else "%d PROBLEMS" % bad))
