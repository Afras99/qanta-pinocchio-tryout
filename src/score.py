"""Score (question, model_answer) pairs with Pinocchio."""
import sys, time, pandas as pd
from pinocchio import Pinocchio
from common import *

def main(with_source=False):
    df = pd.read_csv("results/graded.csv").fillna("")
    judge = Pinocchio()
    pairs = [{"question": q, "answer": a, **({"source_model": TARGET_MODEL} if with_source else {})} for q, a in zip(df.question, df.model_answer)]
    scores, t = [], time.time()
    B = 10
    for i in range(0, len(pairs), B):
        scores += list(judge.score_batch(pairs[i:i+B]))
        print(i + B, round(time.time() - t), "s", flush=True)
    df["p_correct"] = scores
    df[["id", "p_correct"]].to_csv("results/scores_source_model.csv" if with_source else "results/scores.csv", index=False)
    print("total s", round(time.time() - t))

if __name__ == "__main__":
    main(with_source="--source-model" in sys.argv)
