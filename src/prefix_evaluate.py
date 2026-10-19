"""Effect of how much of the question is seen: accuracy, AUROC, ECE and abstention per prefix level, answer transitions
(first sentence -> full question), a figure, and seeded-random qualitative examples (no hand picking)."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from common import *
from evaluate import ece, acc_at_coverage

LEVELS = ["s1", "s2", "half", "full"]
LABEL = {"s1": "1st sentence", "s2": "first 2", "half": "first half", "full": "full question"}

def boot_auc(y, p, rng, n=2000):
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): bs.append(roc_auc_score(y[i], p[i]))
    return [float(v) for v in np.percentile(bs, [2.5, 97.5])]

def main():
    d = pd.read_csv("results/prefix_scores.csv").fillna("")
    rng = np.random.default_rng(SEED)
    out = {}
    for lv in LEVELS:
        x = d[d.level == lv]; y, p = x.correct.values.astype(int), x.p_correct.values
        e, _ = ece(p, y)
        out[lv] = {"n": len(x), "mean_frac_chars": float(x.frac_chars.mean()), "mean_sentences": float(x.k_sentences.mean()),
                   "accuracy": float(y.mean()), "mean_p": float(p.mean()), "mean_p_right": float(p[y == 1].mean()), "mean_p_wrong": float(p[y == 0].mean()),
                   "auroc": float(roc_auc_score(y, p)), "auroc_ci95": boot_auc(y, p, rng), "ece": float(e),
                   "acc_at_40pct": float(acc_at_coverage(p, y, 0.4)), "acc_at_60pct": float(acc_at_coverage(p, y, 0.6)),
                   "share_unchanged_answer_vs_full": float((x.set_index("id").model_answer == d[d.level == "full"].set_index("id").model_answer.loc[x.id]).mean())}
    # transitions first sentence -> full question
    w = d.pivot(index="id", columns="level", values=["correct", "p_correct"])
    c1, cf = w["correct"]["s1"].astype(int), w["correct"]["full"].astype(int)
    p1, pf = w["p_correct"]["s1"], w["p_correct"]["full"]
    tr = {}
    for name, m in {"right_both": (c1 == 1) & (cf == 1), "wrong_then_right": (c1 == 0) & (cf == 1), "right_then_wrong": (c1 == 1) & (cf == 0), "wrong_both": (c1 == 0) & (cf == 0)}.items():
        tr[name] = {"n": int(m.sum()), "mean_p_s1": float(p1[m].mean()), "mean_p_full": float(pf[m].mean())}
    out["transitions_s1_to_full"] = tr
    out["share_questions_confidence_higher_at_full_than_s1"] = float((pf > p1).mean())
    json.dump(out, open("results/prefix_metrics.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items()}, indent=1))

    # figure: accuracy, mean confidence and AUROC vs share of question text seen
    xs = [out[l]["mean_frac_chars"] for l in LEVELS]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(xs, [out[l]["accuracy"] for l in LEVELS], "o-", label="Chatbot accuracy")
    ax.plot(xs, [out[l]["mean_p"] for l in LEVELS], "s--", label="Pinocchio's average confidence")
    ax.plot(xs, [out[l]["auroc"] for l in LEVELS], "^-", label="Pinocchio's right-vs-wrong score (AUROC)")
    ax.plot(xs, [out[l]["acc_at_40pct"] for l in LEVELS], "d-", label="Accuracy on the 40% Pinocchio trusts most")
    for l, xv in zip(LEVELS, xs): ax.annotate(LABEL[l], (xv, 0.02), ha=("right" if l == "full" else "center"), fontsize=8, annotation_clip=False)
    ax.set(xlabel="Share of the question text the chatbot and Pinocchio see", ylabel="Value", ylim=(0, 1.0), xlim=(0, 1.05))
    ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig("figures/prefix_effect.png", dpi=150); plt.close(fig)

    # qualitative examples: one seeded-random question per cell (full question, threshold 0.5), plus one wrong->right with its prefix path
    g = pd.read_csv("results/graded.csv").fillna("").set_index("id")
    f = d[d.level == "full"].set_index("id")
    ex = []
    for name, m in {"trusted_and_right": (f.p_correct >= .5) & (f.correct == 1), "trusted_but_wrong": (f.p_correct >= .5) & (f.correct == 0),
                    "doubted_but_right": (f.p_correct < .5) & (f.correct == 1), "doubted_and_wrong": (f.p_correct < .5) & (f.correct == 0)}.items():
        pool = sorted(f.index[m]); qid = pool[np.random.default_rng(SEED).integers(len(pool))]
        ex.append({"cell": name, "cell_size": len(pool), "id": int(qid), "gold": g.loc[qid, "gold"], "answer_line": g.loc[qid, "answer"][:160],
                   "model_answer": f.loc[qid, "model_answer"], "p_correct": float(f.loc[qid, "p_correct"]), "last_sentence": f.loc[qid, "question"].split(". ")[-1][:300]})
    pool = sorted(w.index[(c1 == 0) & (cf == 1)]); qid = pool[np.random.default_rng(SEED).integers(len(pool))]
    path = {lv: d[(d.id == qid) & (d.level == lv)].iloc[0] for lv in LEVELS}
    ex.append({"cell": "wrong_then_right_path", "cell_size": len(pool), "id": int(qid), "gold": g.loc[qid, "gold"], "answer_line": g.loc[qid, "answer"][:160],
               "first_sentence": path["s1"].question[:300], "steps": {lv: {"answer": path[lv].model_answer, "correct": int(path[lv].correct), "p": float(path[lv].p_correct)} for lv in LEVELS}})
    pool = sorted(w.index[(c1 == 1) & (cf == 0)]); 
    if pool:
        qid = pool[np.random.default_rng(SEED).integers(len(pool))]; path = {lv: d[(d.id == qid) & (d.level == lv)].iloc[0] for lv in LEVELS}
        ex.append({"cell": "right_then_wrong_path", "cell_size": len(pool), "id": int(qid), "gold": g.loc[qid, "gold"], "answer_line": g.loc[qid, "answer"][:160],
                   "first_sentence": path["s1"].question[:300], "steps": {lv: {"answer": path[lv].model_answer, "correct": int(path[lv].correct), "p": float(path[lv].p_correct)} for lv in LEVELS}})
    json.dump(ex, open("results/prefix_examples.json", "w"), indent=2)
    print(json.dumps(ex, indent=1))

if __name__ == "__main__":
    main()
