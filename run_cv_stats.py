"""Stratified 5-fold cross-validation with the tuned hyperparameters, then
Friedman + Nemenyi post-hoc test across models and McNemar tests on the held-out test sets."""
import json, itertools
import os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from data_prep import LOADERS
from models import make_model, MODEL_ORDER, scores

OUT = os.path.join(ROOT, "results")
CV_CAP = 30000  # at most 30,000 training rows per dataset, to keep run time reasonable
rng = np.random.default_rng(42)

# Studentised range q_alpha/sqrt(2) for alpha=0.05 (Demsar, 2006)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def best_threshold(y, s):
    """Score cut-off with the best F1 (only needed for Isolation Forest)."""
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    f = [f1_score(y, (s >= q).astype(int), zero_division=0) for q in qs]
    return qs[int(np.argmax(f))]


# Step 1: 5-fold cross-validation of every model with its tuned settings from run_main.py.
# This gives 3 datasets x 5 folds = 15 F1 scores per model.
cv = {}
for ds, loader in LOADERS.items():
    d = loader()
    main = json.load(open(f"{OUT}/{ds}_main.json"))
    X, y = d["X_train"], d["y_train"]
    if len(y) > CV_CAP:
        idx = rng.choice(len(y), CV_CAP, replace=False)
        X, y = X.iloc[idx], y[idx]
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    cv[ds] = {m: [] for m in MODEL_ORDER}
    for k, (a, b) in enumerate(skf.split(X, y)):
        for name in MODEL_ORDER:
            p = {kk: (tuple(v) if isinstance(v, list) else v) for kk, v in main["models"][name]["best_params"].items()}
            m = make_model(name, p)
            if name == "IsolationForest":  # Isolation Forest learns from benign rows only
                m.fit(X.iloc[a][y[a] == 0])
                s_tr = scores(m, name, X.iloc[a]); thr = best_threshold(y[a], s_tr)
            else:
                m.fit(X.iloc[a], y[a]); thr = 0.5
            s = scores(m, name, X.iloc[b])
            cv[ds][name].append(float(f1_score(y[b], (s >= thr).astype(int))))
        print(ds, "fold", k, {n: round(v[-1], 4) for n, v in cv[ds].items()}, flush=True)

# Friedman test over 15 blocks (3 datasets x 5 folds)
blocks = np.array([[cv[ds][m][k] for m in MODEL_ORDER] for ds in cv for k in range(5)])
chi2, p_f = stats.friedmanchisquare(*blocks.T)
# Rank the models in each block (1 = best F1), then average the ranks.
ranks = np.array([stats.rankdata(-row) for row in blocks])
avg_rank = ranks.mean(0)
N, kmod = blocks.shape
# Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1) / 6N)
cd = Q05[kmod] * np.sqrt(kmod * (kmod + 1) / (6 * N))
# Two models differ significantly when their average ranks are further apart than CD.
nemenyi = {f"{MODEL_ORDER[i]} vs {MODEL_ORDER[j]}": bool(abs(avg_rank[i] - avg_rank[j]) > cd)
           for i, j in itertools.combinations(range(kmod), 2)}

# McNemar (exact binomial on discordant pairs) between top-2 models by test F1 on each dataset
mcn = {}
for ds in cv:
    main = json.load(open(f"{OUT}/{ds}_main.json"))
    tp = json.load(open(f"{OUT}/{ds}_testpreds.json"))
    y = np.array(tp["y"])
    order = sorted(MODEL_ORDER, key=lambda m: -main["models"][m]["f1"])
    res = {}
    for a_, b_ in [(order[0], order[1]), (order[0], order[2])]:
        # Count the test flows where exactly one of the two models is right (discordant pairs).
        pa, pb = np.array(tp["preds"][a_]) == y, np.array(tp["preds"][b_]) == y
        n01, n10 = int((pa & ~pb).sum()), int((~pa & pb).sum())
        pval = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        res[f"{a_} vs {b_}"] = dict(a_right_b_wrong=n01, a_wrong_b_right=n10, p=float(pval))
    mcn[ds] = res

# Save every number behind the statistics tables to results/cv_stats.json.
out = dict(cv_f1=cv, cv_mean={ds: {m: float(np.mean(v)) for m, v in cv[ds].items()} for ds in cv},
           cv_std={ds: {m: float(np.std(v)) for m, v in cv[ds].items()} for ds in cv},
           friedman=dict(chi2=float(chi2), p=float(p_f), N=int(N), k=int(kmod)),
           avg_rank=dict(zip(MODEL_ORDER, map(float, avg_rank))), nemenyi_cd=float(cd),
           nemenyi_significant=nemenyi, mcnemar=mcn)
json.dump(out, open(f"{OUT}/cv_stats.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ["friedman", "avg_rank", "nemenyi_cd", "mcnemar"]}, indent=1))
