"""Grade prefix answers with the same grader and gold forms as grade.py."""
import pandas as pd
from common import *
from grade import gold_forms, is_correct, is_correct_strict, alt_forms
import re

def main():
    g = pd.read_csv("results/graded.csv").fillna("").set_index("id")
    pa = pd.read_csv("results/prefix_answers.csv").fillna("")
    pa["correct"] = [is_correct(a, gold_forms(g.loc[i, "gold"], g.loc[i, "answer"])) for i, a in zip(pa.id, pa.model_answer)]
    pa.to_csv("results/prefix_graded.csv", index=False)
    full = pa[pa.level == "full"].set_index("id").correct
    assert (full == g.correct.loc[full.index]).all(), "full-level grading must equal the original grading"
    print(pa.groupby("level").correct.mean().round(3))

if __name__ == "__main__":
    main()
