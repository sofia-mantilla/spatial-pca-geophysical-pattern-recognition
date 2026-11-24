# `paper/checks/` — submission coherence tests

Run from the repository root after any change to the manuscript or the results:

    .venv/bin/python paper/checks/check_submission.py

It calls the other two and adds the build, Case-1, deposit-set, stale-value,
marked-copy and git checks. `check_paper_numbers.py` traces every Case-2 figure
in `main.tex` to a JSON under `paper/results/`; `check_letter_agrees.py` checks
that the response letter and the manuscript give the same value for everything
they both quote, and that the letter's own figures match the JSONs.

A failure naming a value usually means a script was rerun and the text was not,
or the reverse. Regenerate with the `run_case2_*.py` scripts, then rerun this.
