"""Control: score each question with a model answer taken from a DIFFERENT question (seeded derangement).
If Pinocchio reads the answer, mean score should drop sharply and AUROC vs the shuffled labels should be ~0.5 or lower."""
import numpy as np, pandas as pd
from pinocchio import Pinocchio
from common import *

def main(n=200):
    df = pd.read_csv("results/graded.csv").fillna("").iloc[:n]
    rng = np.random.default_rng(SEED)
    while True:
        perm = rng.permutation(len(df))
        if (perm != np.arange(len(df))).all(): break
    judge = Pinocchio()
    pairs = [{"question": q, "answer": df.model_answer.iloc[j]} for q, j in zip(df.question, perm)]
    sc = []
    for i in range(0, len(pairs), 10): sc += list(judge.score_batch(pairs[i:i+10])); print(i + 10, flush=True)
    out = pd.DataFrame({"id": df.id, "shuffled_from_id": df.id.iloc[perm].values, "p_shuffled": sc})
    out.to_csv("results/scores_shuffled_control.csv", index=False)

if __name__ == "__main__":
    main()
