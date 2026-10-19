"""Build prefix versions of the 500 sampled questions using QANTA's own sentence boundaries (`tokenizations`).
Levels: s1 = first sentence, s2 = first two sentences, half = first ceil(n/2) sentences, full = whole question."""
import json, math, pandas as pd
from common import *

END_CHARS = ".?!"
CLOSERS = "\"'\u201d)]"

def snap(text, e, window=15):
    """QANTA's offsets drift by a few characters on questions with garbled (mojibake) characters, which would cut words in half.
    Keep an offset that already sits at a sentence end; otherwise move it forward to the next sentence end within `window` characters."""
    def is_end(p):
        t = text[:p].rstrip()
        j = len(t)
        while j > 0 and t[j - 1] in CLOSERS: j -= 1
        return j > 0 and t[j - 1] in END_CHARS and (p >= len(text) or text[p].isspace())
    if is_end(e): return e
    for p in range(e + 1, min(len(text), e + window) + 1):
        if is_end(p): return p
    return e   # no sentence end nearby: keep the original offset (reported in the summary below)

def main():
    recs = {r["qanta_id"]: r for r in json.load(open(DATA))["questions"]}
    sample = pd.read_csv("results/sample.csv")
    rows = []
    for qid, full in zip(sample.id, sample.question):
        r = recs[qid]
        assert r["text"] == full
        spans = json.loads(r["tokenizations"]) if isinstance(r["tokenizations"], str) else r["tokenizations"]
        n = len(spans)
        for level, k in (("s1", 1), ("s2", 2), ("half", math.ceil(n / 2)), ("full", n)):
            k = min(k, n)
            end = snap(full, spans[k - 1][1])
            text = full[:end].strip()
            rows.append({"id": qid, "level": level, "n_sentences": n, "k_sentences": k, "question": text,
                         "frac_chars": len(text) / len(full), "offset_shift": end - spans[k - 1][1]})
    df = pd.DataFrame(rows)
    df.to_csv("results/prefix_questions.csv", index=False)
    g = df.groupby("level").agg(k=("k_sentences", "mean"), frac=("frac_chars", "mean"))
    print(g.round(2)); print("sentences per question:", df.drop_duplicates("id").n_sentences.describe().round(1).to_dict())
    print("boundaries moved by the snap:", int((df.offset_shift != 0).sum()), "of", len(df), "| texts not ending at sentence punctuation:",
          int((~df.question.str.rstrip().str[-1].isin(list(END_CHARS + CLOSERS))).sum()))
    print("level texts identical to full:", {l: int((d.question.values == df[(df.level == 'full')].set_index('id').loc[d.id].question.values).sum()) for l, d in df.groupby("level")})

if __name__ == "__main__":
    main()
