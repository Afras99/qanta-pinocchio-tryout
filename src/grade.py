"""Grade answers.
Gold forms = Wikipedia page title + primary answer line + 'or' alternatives in [..] (conditional clauses such as
'until X is read' are cut). Predictions and gold are normalized (lowercase, ASCII-fold, underscores->spaces, drop
parentheticals/punctuation/articles, singular/plural stemming). Correct if, for any gold form:
  1. equal, or equal after removing spaces (waste water / wastewater), or
  2. character similarity (difflib ratio) >= 0.9 with both strings >= 8 chars (spelling variants), or
  3. one is a whole-token span of the other and the shorter has >= 5 characters.
Also writes strict (rule 1 only, no stemming of tokens beyond normalize) labels as `correct_strict` for sensitivity analysis."""
import re, difflib, pandas as pd
from common import *

def stem(t):
    out = []
    for w in t.split():
        if len(w) > 4 and w.endswith("ies"): w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"): w = w[:-1]
        out.append(w)
    return " ".join(out)

def alt_forms(answer):
    """'or' alternatives inside [..] (bracket may be unclosed); conditional clauses are cut."""
    out = []
    for m in re.findall(r"\[([^\]]*)(?:\]|$)", answer):
        for piece in m.split(";"):
            piece = piece.strip()
            if not re.match(r"or\b", piece, re.I): continue
            for alt in re.split(r"\bor\b", piece[2:]):
                out.append(re.split(r"\b(?:until|before|after|when|if)\b", alt)[0])
    return out

def gold_forms(page, answer, with_page=True):
    forms = ([page] if with_page else []) + [re.split(r"[\[(]", answer)[0]] + alt_forms(answer)
    return {n for n in (stem(normalize(f)) for f in forms) if n}

def span(short, long):
    return f" {short} " in f" {long} "

def is_correct(pred, forms):
    p = stem(normalize(pred))
    if not p: return 0
    for g in forms:
        if p == g or p.replace(" ", "") == g.replace(" ", ""): return 1
        if len(p) >= 8 and len(g) >= 8 and difflib.SequenceMatcher(None, p, g).ratio() >= 0.9: return 1
        sh, lg = (p, g) if len(p) <= len(g) else (g, p)
        if len(sh) >= 5 and span(sh, lg): return 1
    return 0

def is_correct_strict(pred, forms):
    p = normalize(pred)
    return int(bool(p) and any(p == normalize(f) for f in forms))

def main():
    df = pd.read_csv("results/answers.csv").fillna("")
    df["correct"] = [is_correct(p, gold_forms(g, a)) for p, g, a in zip(df.model_answer, df.gold, df.answer)]
    raw = [{g, re.split(r"[\[(]", a)[0]} | set(alt_forms(a)) for g, a in zip(df.gold, df.answer)]
    df["correct_strict"] = [is_correct_strict(p, r) for p, r in zip(df.model_answer, raw)]
    df["page_in_answer_line"] = [int(normalize(g) in {normalize(x) for x in gold_forms("", a, with_page=False) | set(alt_forms(a)) | {re.split(r"[\[(]", a)[0]}}) for g, a in zip(df.gold, df.answer)]
    df.to_csv("results/graded.csv", index=False)
    print("accuracy:", df.correct.mean(), "strict:", df.correct_strict.mean(), "n =", len(df))
    print("page title not among answer-line forms:", int((1 - df.page_in_answer_line).sum()), "of", len(df))

if __name__ == "__main__":
    main()
