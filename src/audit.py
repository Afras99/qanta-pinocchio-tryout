"""Reproduce the grader-audit samples (disjoint draws from graded.csv) and verify they match the saved files.
Draw 1: 20 (seed 0); draw 2: 80 (seed 1); draw 3: 100 (seed 2); draw 4 = results/audit_clean.csv: 100 (seed 3), each from rows not yet drawn.
Only draw 4 was made after the last grader change and is the one reported in WRITEUP.md; its manual_error column
(FP/FN) was filled in by the author (AI-assisted reading of the full answer lines), not by an independent human."""
import pandas as pd
from common import *

def main():
    g = pd.read_csv("results/graded.csv").fillna("")
    seen = set()
    for n, seed in [(20, SEED), (80, 1), (100, 2), (100, 3)]:
        d = g[~g.id.isin(seen)].sample(n, random_state=seed); seen |= set(d.id); last = d
    saved = pd.read_csv("results/audit_clean.csv")
    assert list(last.id) == list(saved.id), "audit_clean.csv does not match the seeded draw"
    print("audit_clean.csv reproduced; n =", len(saved), "errors:", saved.manual_error.value_counts().to_dict())
    print("accuracy (grader labels) in audit sample:", saved.correct.mean(), "vs overall", g.correct.mean())

if __name__ == "__main__":
    main()
