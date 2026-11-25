"""One pass over everything: is the submission internally coherent?

Run from the repository root:

    .venv/bin/python paper/checks/check_submission.py

Checks the build log, that Case 1 is untouched, that the Case-2 deposit set is
stated the same way everywhere, that every Case-2 number in the text traces to
a regenerated JSON (check_paper_numbers.py), that the response letter agrees
with the paper (check_letter_agrees.py), that no value or phrasing from an
earlier version survives, that the marked copy regenerates, that the
response letter is a .docx Word can open, and that both repositories are
committed. Exits non-zero on any failure.
"""
import io, json, re, subprocess, sys, unicodedata as ud, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(".").resolve()
D = ROOT / "docs/Spatial_PCA_paper_overleaf"
RES = ROOT / "paper/results"
tex = io.open(D / "main.tex", encoding="utf-8").read()
flat = re.sub(r"\\s+", " ", tex)   # for checks that must ignore line wrapping
log = io.open(D / "main.log", encoding="utf-8", errors="replace").read()
fails = []


def check(name, ok, detail=""):
    print("  %-52s %s%s" % (name, "ok" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        fails.append(name)


print("=== build ===")
check("0 undefined references", "undefined" not in log.lower())
check("0 overfull hboxes", log.count("Overfull \\hbox") == 0)
check("0 LaTeX warnings", log.count("LaTeX Warning") == 0)
EXPECTED_PAGES = "50"      # change here when the agreed length changes
m = re.search(r"\((\d+) pages", log)
check("page count is %s" % EXPECTED_PAGES, m and m.group(1) == EXPECTED_PAGES,
      "(%s)" % (m.group(1) if m else "?"))

print("\n=== Case 1 is untouched ===")
C1 = {"47.0": "recovery", "57.9": "AUC", "20.7": "raw", "14.4": "a(250)",
      "15.2": "random at equal area", "31.6": "random at equal rank",
      "3.09": "ratio", "47.5": "contact proximity", "66.3": "standardized variant",
      "44.7": "demeaned match", "36.4": "correlation match", "3.3": "median refswap factor"}
for v, what in C1.items():
    check("Case-1 %s (%s) still present" % (v, what), v in tex)

print("\n=== the Case-2 deposit set is stated consistently ===")
for lab, s in [
    ("tab:cases lists 1, 2, 4--6", r"& 3 & 1, 2, 4--6\\"),
    ("six pass the threshold", "Six of the twelve deposits pass the threshold"),
    ("five test the ranking", "so five deposits test the ranking"),
    ("Fig19 caption says 1--6", "for Deposits 1--6"),
    ("results: three of the five", "hits three of the five test deposits"),
    ("abstract: three of five", "hits three of five test deposits"),
    ("highlight: three of the five", "and reached three of them"),
    ("per-deposit table carries Paulo Afonso", "6 Paulo Afonso &"),
    ("per-deposit mean of the five", "Mean of the five &"),
    ("held-out: thirty pairs", "thirty reference and test pairs"),
    ("held-out: six references", "Each of the six deposits that pass"),
]:
    check(lab, s in tex or re.sub(r"\s+", " ", s) in flat)

print("\n=== every Case-2 number traces to a regenerated JSON ===")
r = subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "check_paper_numbers.py")],
                   capture_output=True, text=True, cwd=ROOT)
tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
check("number audit", "of" in tail and tail.split()[0] == tail.split()[2], tail)

print("\n=== the letter agrees with the paper ===")
r = subprocess.run([sys.executable, "-I", str(Path(__file__).parent / "check_letter_agrees.py")],
                   capture_output=True, text=True, cwd=ROOT)
check("letter audit", "ALL CHECKS PASS" in r.stdout,
      "" if "ALL CHECKS PASS" in r.stdout else r.stdout.strip().splitlines()[-1])

print("\n=== values from earlier versions are gone ===")
GONE = ["four test deposits", "the four of the multivariate", "of the four test",
        "63.7", "106.1", "56.1", "55.4", "0.497", "0.491", "0.455", "0.396",
        "23.3", "7.17", "68.4", "50.8", "51.2", "twenty-five", "twenty reference",
        "0.419", "7.2$\\times$", "+0.090", "0.610", "0.479", "+0.136", "+0.161"]
for v in GONE:
    check("no %r" % v, v not in tex)

print("\n=== the marked copy regenerates ===")
r = subprocess.run([sys.executable, "make_marked.py"], capture_output=True, text=True, cwd=D)
check("make_marked.py runs", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-120:])

print("\n=== the response letter is a readable .docx ===")
# Word rejects the whole file on one malformed part, and the failure is silent
# until someone tries to open it. Editing word/document.xml by hand produced
# exactly that once: a <w:t> left open around an inserted <w:p>.
DOCX = D / "NRR_response_to_reviewers.docx"
try:
    zl = zipfile.ZipFile(DOCX)
    check("zip is intact", zl.testzip() is None)
    names = set(zl.namelist())
    for req in ("[Content_Types].xml", "_rels/.rels", "word/document.xml"):
        check("%s present" % req, req in names)
    bad = []
    for n in sorted(names):
        if n.endswith((".xml", ".rels")):
            try:
                ET.fromstring(zl.read(n))
            except ET.ParseError as e:
                bad.append("%s: %s" % (n, e))
    check("every XML part parses", not bad, bad[0] if bad else "")
    doc = zl.read("word/document.xml").decode("utf-8")
    check("no block element inside a <w:t>",
          not re.search(r"<w:t[^>]*><w:(p|tbl|r)[ >]", doc))
    check("<w:t> tags balanced", doc.count("<w:t") == doc.count("</w:t>"),
          "%d open, %d close" % (doc.count("<w:t"), doc.count("</w:t>")))
except zipfile.BadZipFile as e:
    check("zip is intact", False, str(e))

print("\n=== reviewer-facing register ===")
BAD = ["this check", "now measured", "rescores the published", "reported for that reason",
       "is a judgement", "we report it because", "the previous version", "withdrawn the claim"]
for b in BAD:
    check("no %r" % b, b.lower() not in tex.lower())

print("\n=== results JSONs are present and current ===")
for f in ["case2_references", "case2_appendixC", "case2_heldout", "case2_heldout_grid",
          "case2_scalings", "all_twelve_case2", "arim_contacts", "era_shares"]:
    check("%s.json" % f, (RES / (f + ".json")).exists())

print("\n=== git ===")
for repo, name in ((ROOT, "code"), (D, "manuscript")):
    out = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                         text=True, cwd=repo).stdout
    dirty = [l for l in out.splitlines()
             if l[3:] not in (".DS_Store",) and not l.startswith("??")]
    check("%s repo has no uncommitted tracked changes" % name, not dirty,
          "; ".join(dirty[:3]))

print("\n%s" % ("ALL COHERENT" if not fails else "%d FAILURES: %s" % (len(fails), fails)))
sys.exit(1 if fails else 0)
