"""AUROC (+bootstrap CI), reliability diagram, ECE, abstention curve and table."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from common import *

def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
            rows.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return e, rows

def reliability_ax(ax, rows, e, title):
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly honest")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="Pinocchio (number = answers in group)")
    for c, a, n in rows: ax.annotate(str(n), (c, a), fontsize=9, xytext=(4, -11), textcoords="offset points")
    ax.set(xlabel="What Pinocchio said (average score of the group)", ylabel="How often those answers were actually right",
           title=f"{title}\nAverage gap from honest (ECE) = {e:.2f}", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper left")

def reliability_plot(rows, e, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4.5)); reliability_ax(ax, rows, e, title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def acc_at_coverage(p, y, cov):
    k = max(1, int(round(cov * len(p))))
    return y[np.argsort(-p, kind="stable")[:k]].mean()

def main():
    df = load_scored()
    p, y = df.p_correct.values, df.correct.values.astype(int)
    rng = np.random.default_rng(SEED)
    auc = roc_auc_score(y, p)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): bs.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    e, rows = ece(p, y)
    reliability_plot(rows, e, "figures/reliability.png", "Is Pinocchio's confidence honest? (all 500 answers)")

    # abstention curve
    order = np.argsort(-p, kind="stable")
    ks = np.arange(1, len(p) + 1)
    cov = ks / len(p); acc = np.cumsum(y[order]) / ks
    rand = np.mean([[y[rng.permutation(len(y))[:k]].mean() for k in range(1, len(y) + 1)] for _ in range(200)], axis=0)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    keep = cov >= 0.05   # below 5% coverage fewer than 25 answers remain: too noisy to plot
    ax.plot(cov[keep], acc[keep], label="Skip the answers Pinocchio trusts least"); ax.plot(cov[keep], rand[keep], "--", label="Skip answers at random")
    ax.axhline(y.mean(), color="gray", lw=0.8, label=f"Answer everything ({y.mean():.1%} right)")
    ax.set(xlabel="Share of the 500 questions we answer (shown from 5%)", ylabel="Share of those answers that are right", title="Answering fewer questions gets more of them right"); ax.legend()
    fig.tight_layout(); fig.savefig("figures/abstention_curve.png", dpi=150); plt.close(fig)

    table = []
    for c in [1.0, 0.8, 0.6, 0.4]:
        k = int(round(c * len(p)))
        r = float(np.mean([y[rng.permutation(len(y))[:k]].mean() for _ in range(1000)]))
        table.append({"coverage": c, "pinocchio_acc": float(acc_at_coverage(p, y, c)), "random_acc": r})
    m = {"n": len(y), "accuracy": float(y.mean()), "auroc": auc, "auroc_ci95": [lo, hi], "ece": e,
         "mean_p_correct": float(p.mean()), "abstention_table": table}
    json.dump(m, open("results/metrics.json", "w"), indent=2)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
