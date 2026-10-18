"""Robustness checks: ECE CI, abstention CIs, strict-grader sensitivity, trivial baselines, target-model logprob baseline,
shuffled-answer control, and Platt scaling repeated over many random calibration/held-out splits."""
import json, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from common import *
from evaluate import ece, acc_at_coverage
from recalibrate import logit

def boot(fn, n, B=2000, seed=SEED):
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        try: v.append(fn(i))
        except ValueError: pass
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]

def main():
    d = load_scored(); g = pd.read_csv("results/graded.csv").fillna("")
    d = d.merge(g[["id", "question", "model_answer"]], on="id")
    p, y, ys = d.p_correct.values, d.correct.values.astype(int), d.correct_strict.values.astype(int)
    R = {}
    R["ece"] = {"value": ece(p, y)[0], "ci95": boot(lambda i: ece(p[i], y[i])[0], len(y))}
    # abstention: paired bootstrap of (Pinocchio - mean-of-random) i.e. minus overall accuracy of the resample
    R["abstention_gain_over_baseline"] = {}
    for c in [0.8, 0.6, 0.4]:
        f = lambda i, c=c: acc_at_coverage(p[i], y[i], c) - y[i].mean()
        R["abstention_gain_over_baseline"][str(c)] = {"value": f(np.arange(len(y))), "ci95": boot(f, len(y))}
    # label-noise sensitivity
    R["strict_labels"] = {"accuracy": float(ys.mean()), "auroc": roc_auc_score(ys, p), "auroc_ci95": boot(lambda i: roc_auc_score(ys[i], p[i]), len(ys)),
                          "ece": ece(p, ys)[0]}
    # trivial baselines (AUROC of simple features vs correctness)
    feats = {"answer_char_len (neg)": -d.model_answer.str.len(), "answer_word_len (neg)": -d.model_answer.str.split().str.len(),
             "question_char_len": d.question.str.len(), "answer_in_question": d.apply(lambda r: float(str(r.model_answer).lower() in r.question.lower()), axis=1)}
    R["trivial_baselines_auroc"] = {k: roc_auc_score(y, v) for k, v in feats.items()}
    R["pinocchio_auroc"] = roc_auc_score(y, p)
    R["pinocchio_vs_answer_len_corr"] = float(np.corrcoef(p, d.model_answer.str.len())[0, 1])
    if os.path.exists("results/baseline_logprob.csv"):
        b = pd.read_csv("results/baseline_logprob.csv"); m = d.merge(b, on="id")
        assert (m.answer_check.str.strip() == m.model_answer.str.strip()).mean() > 0.95, "logprob run answers differ from graded answers"
        yy = m.correct.values.astype(int)
        R["answer_match_rate_logprob_run"] = float((m.answer_check.str.strip() == m.model_answer.str.strip()).mean())
        R["logprob_baselines"] = {k: {"auroc": roc_auc_score(yy, m[k]), "ci95": boot(lambda i, k=k: roc_auc_score(yy[i], m[k].values[i]), len(yy))} for k in ["mean_logprob", "sum_logprob", "first_logprob"]}
        pk = m.p_correct.values
        R["logprob_baselines"]["abstention_acc_at_40pct_coverage"] = {"pinocchio": float(acc_at_coverage(pk, yy, 0.4)), "mean_logprob": float(acc_at_coverage(m.mean_logprob.values, yy, 0.4))}
        pm, lm = m.p_correct.values, m.mean_logprob.values
        dfn = lambda i: roc_auc_score(yy[i], lm[i]) - roc_auc_score(yy[i], pm[i])
        R["paired_auroc_diff_logprob_minus_pinocchio"] = {"value": dfn(np.arange(len(yy))), "ci95": boot(dfn, len(yy))}
        R["logprob_vs_pinocchio_spearman"] = float(pd.Series(pk).corr(m.mean_logprob, method="spearman"))
    if os.path.exists("results/scores_shuffled_control.csv"):
        # the control was scored WITHOUT source_model, so compare against the no-source scores of the real pairs
        s = pd.read_csv("results/scores_shuffled_control.csv").merge(load_scored(NO_SOURCE_SCORES)[["id", "p_correct"]], on="id").merge(d[["id", "correct", "model_answer"]], on="id")
        ans = d.set_index("id").model_answer; gold_true = g.set_index("id")
        # label of the shuffled pair: re-grade the borrowed answer against this question's gold
        from grade import gold_forms, is_correct
        ysh = np.array([is_correct(ans[j], gold_forms(gold_true.gold[i], gold_true.answer[i])) for i, j in zip(s.id, s.shuffled_from_id)])
        R["shuffled_control"] = {"n": len(s), "mean_p_real": float(s.p_correct.mean()), "mean_p_shuffled": float(s.p_shuffled.mean()),
                                 "frac_shuffled_pairs_actually_correct": float(ysh.mean()),
                                 "auroc_real_vs_shuffled_pairs": roc_auc_score(np.r_[np.ones(len(s)), np.zeros(len(s))], np.r_[s.p_correct, s.p_shuffled]),
                                 "auroc_on_shuffled_pairs": float(roc_auc_score(ysh, s.p_shuffled)) if 0 < ysh.sum() < len(ysh) else None}
    # Platt over many splits
    out = []
    for seed in range(200):
        rng = np.random.default_rng(seed); perm = rng.permutation(len(d)); c, t = perm[:100], perm[100:]
        lr = LogisticRegression(C=1e6).fit(logit(p[c])[:, None], y[c]); pp = lr.predict_proba(logit(p[t])[:, None])[:, 1]
        out.append((ece(p[t], y[t])[0], ece(pp, y[t])[0], roc_auc_score(y[t], p[t]), roc_auc_score(y[t], pp), lr.coef_[0, 0]))
    o = np.array(out)
    R["platt_200_splits"] = {"ece_before_mean": o[:, 0].mean(), "ece_after_mean": o[:, 1].mean(), "ece_after_ci95": [float(x) for x in np.percentile(o[:, 1], [2.5, 97.5])],
                             "frac_splits_ece_improved": float((o[:, 1] < o[:, 0]).mean()), "max_abs_auroc_change": float(np.abs(o[:, 2] - o[:, 3]).max()),
                             "coef_range": [float(o[:, 4].min()), float(o[:, 4].max())]}
    json.dump(R, open("results/robustness.json", "w"), indent=2, default=float)
    print(json.dumps(R, indent=2, default=float))

if __name__ == "__main__":
    main()
