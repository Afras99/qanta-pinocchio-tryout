"""Compare Pinocchio scores with vs without source_model (same 500 pairs, same labels)."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from common import *
from evaluate import ece, acc_at_coverage
from recalibrate import logit
from analysis_extra import boot

def main():
    d = load_scored(NO_SOURCE_SCORES).rename(columns={"p_correct": "p0"})
    d = d.merge(pd.read_csv("results/scores_source_model.csv").rename(columns={"p_correct": "p1"}), on="id", validate="1:1")
    y, p0, p1 = d.correct.values.astype(int), d.p0.values, d.p1.values
    R = {"n": len(y), "accuracy": float(y.mean())}
    for name, p in [("no_source_model", p0), ("with_source_model", p1)]:
        e, _ = ece(p, y)
        R[name] = {"auroc": roc_auc_score(y, p), "auroc_ci95": boot(lambda i: roc_auc_score(y[i], p[i]), len(y)), "ece": e,
                   "ece_ci95": boot(lambda i: ece(p[i], y[i])[0], len(y)), "mean_p": float(p.mean()),
                   "acc_at_coverage": {str(c): float(acc_at_coverage(p, y, c)) for c in [1.0, 0.8, 0.6, 0.4]}}
        out = []
        for s in range(200):
            perm = np.random.default_rng(s).permutation(len(y)); c, t = perm[:100], perm[100:]
            lr = LogisticRegression(C=1e6).fit(logit(p[c])[:, None], y[c]); pp = lr.predict_proba(logit(p[t])[:, None])[:, 1]
            out.append((ece(p[t], y[t])[0], ece(pp, y[t])[0]))
        o = np.array(out); R[name]["platt_200_splits"] = {"ece_before_mean": o[:, 0].mean(), "ece_after_mean": o[:, 1].mean(), "frac_improved": float((o[:, 1] < o[:, 0]).mean())}
    f = lambda i: roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i])
    R["paired_auroc_diff_with_minus_without"] = {"value": f(np.arange(len(y))), "ci95": boot(f, len(y))}
    R["spearman_between_variants"] = float(pd.Series(p0).corr(pd.Series(p1), method="spearman"))
    json.dump(R, open("results/source_model_comparison.json", "w"), indent=2, default=float); print(json.dumps(R, indent=2, default=float))

if __name__ == "__main__":
    main()
