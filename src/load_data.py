"""Load QANTA dev set, drop empty records, sample N with fixed seed."""
import json, pandas as pd
from common import *

def main():
    recs = json.load(open(DATA))["questions"]
    print("total records:", len(recs))
    df = pd.DataFrame([{"id": r["qanta_id"], "question": r["text"], "page": r["page"], "answer": r["answer"]} for r in recs])
    for c in ["question", "page", "answer"]:
        df = df[df[c].astype(str).str.strip() != ""]
    df = df.dropna()
    print("after dropping empty/missing:", len(df))
    s = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    s.to_csv("results/sample.csv", index=False)
    print("sampled", len(s), "-> results/sample.csv")

if __name__ == "__main__":
    main()
