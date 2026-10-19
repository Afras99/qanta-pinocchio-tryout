"""Score prefix (question, answer) pairs with Pinocchio (source_model given, as in the primary run). Identical pairs are scored once;
the full-question pairs reuse results/scores_source_model.csv.
Incremental: a (question, answer) pair already scored in results/prefix_scores.csv keeps its score, so only new or changed pairs are rescored."""
import os, time, pandas as pd
from pinocchio import Pinocchio
from common import *

def main():
    pa = pd.read_csv("results/prefix_graded.csv").fillna("")
    cache = {}
    if os.path.exists("results/prefix_scores.csv"):
        for r in pd.read_csv("results/prefix_scores.csv").fillna("").itertuples(): cache[(r.question, r.model_answer)] = float(r.p_correct)
    prim = pd.read_csv(PRIMARY_SCORES).set_index("id").p_correct
    for r in pa[pa.level == "full"].itertuples(): cache[(r.question, r.model_answer)] = float(prim.loc[r.id])
    todo = pa[[(r.question, r.model_answer) not in cache for r in pa.itertuples()]].drop_duplicates(["question", "model_answer"])
    print("to score:", len(todo), "of", len(pa), flush=True)
    judge = Pinocchio(); t = time.time(); B = 10
    pairs = [{"question": r.question, "answer": r.model_answer, "source_model": TARGET_MODEL} for r in todo.itertuples()]
    for i in range(0, len(pairs), B):
        for r, s in zip(todo.iloc[i:i + B].itertuples(), judge.score_batch(pairs[i:i + B])): cache[(r.question, r.model_answer)] = float(s)
        print(min(i + B, len(pairs)), round(time.time() - t), "s", flush=True)
    pa["p_correct"] = [cache[(r.question, r.model_answer)] for r in pa.itertuples()]
    pa.to_csv("results/prefix_scores.csv", index=False)
    print("done")

if __name__ == "__main__":
    main()
