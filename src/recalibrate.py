"""Platt scaling: fit on 100 calibration examples, evaluate on the held-out remainder."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluate import ece, reliability_ax

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def main():
    df = load_scored()
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(df)); cal, test = df.iloc[perm[:100]], df.iloc[perm[100:]]
    lr = LogisticRegression(C=1e6).fit(logit(cal.p_correct.values)[:, None], cal.correct.values)
    yt, pt = test.correct.values.astype(int), test.p_correct.values
    pp = lr.predict_proba(logit(pt)[:, None])[:, 1]
    e0, r0 = ece(pt, yt); e1, r1 = ece(pp, yt)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
    reliability_ax(axs[0], r0, e0, "BEFORE the fix (400 unseen answers)"); reliability_ax(axs[1], r1, e1, "AFTER the fix (same 400 answers)")
    fig.tight_layout(); fig.savefig("figures/reliability_platt.png", dpi=150); plt.close(fig)
    xs = np.linspace(0.001, 0.999, 500); a, b = lr.coef_[0, 0], lr.intercept_[0]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="no change")
    ax.plot(xs, 1 / (1 + np.exp(-(a * logit(xs) + b))), lw=2.5, label="the fix (learned from 100 answers)")
    ax.scatter([q[0] for q in r0], [q[1] for q in r0], c="C3", zorder=3, label="what really happened (400 unseen answers)")
    ax.set(xlabel="Pinocchio's original score", ylabel="Score after the fix", xlim=(0, 1), ylim=(0, 1), title="The fix: how each score gets adjusted")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig("figures/platt_fix_curve.png", dpi=150); plt.close(fig)
    m = {"n_cal": len(cal), "n_heldout": len(test), "platt_coef": float(lr.coef_[0, 0]), "platt_intercept": float(lr.intercept_[0]),
         "ece_before": e0, "ece_after": e1, "auroc_before": roc_auc_score(yt, pt), "auroc_after": roc_auc_score(yt, pp),
         "mean_p_before": float(pt.mean()), "mean_p_after": float(pp.mean()), "heldout_accuracy": float(yt.mean())}
    json.dump(m, open("results/recalibration.json", "w"), indent=2)
    print(json.dumps(m, indent=2))

if __name__ == "__main__":
    main()
