"""Feature-selection analysis: correlation redundancy and importance-ranked subsets.

For each dataset: (1) count highly correlated feature pairs (|Pearson r| >= 0.95) on the training split;
(2) rank features by Random Forest impurity importance; (3) retrain the tuned XGBoost on the top-k
features and report validation F1 and training time, to decide whether a reduced set is justified.
Selection uses training/validation data only; the test set is never touched.
"""
import json, time
import os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np, pandas as pd, joblib
from sklearn.metrics import f1_score
from data_prep import LOADERS
from models import make_model

RES = os.path.join(ROOT, "results")
out = {}
for ds, loader in LOADERS.items():
    d = loader()
    Xtr, ytr, Xva, yva = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    corr = Xtr.corr().abs().to_numpy()
    iu = np.triu_indices_from(corr, 1)
    vals = corr[iu]
    pairs = [(Xtr.columns[i], Xtr.columns[j], float(corr[i, j])) for i, j, v in zip(*iu, vals) if v >= 0.95]
    # greedy redundancy removal: drop the second feature of each highly correlated pair
    drop = set()
    for a, b, v in pairs:
        if a not in drop and b not in drop:
            drop.add(b)
    rf = joblib.load(f"{RES}/models/{ds}_RandomForest.joblib")["model"]
    imp = pd.Series(rf.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    params = json.load(open(f"{RES}/{ds}_main.json"))["models"]["XGBoost"]["best_params"]
    rows = []
    ks = [k for k in [5, 10, 20, 30, 40] if k < len(imp)] + [len(imp)]
    subsets = [(f"top-{k}" if k < len(imp) else "all", list(imp.index[:k])) for k in ks]
    subsets.append(("all minus correlated", [c for c in Xtr.columns if c not in drop]))
    for name, cols in subsets:
        m = make_model("XGBoost", params); t = time.perf_counter(); m.fit(Xtr[cols], ytr); ft = time.perf_counter() - t
        f = f1_score(yva, m.predict(Xva[cols]))
        rows.append(dict(subset=name, n_features=len(cols), val_f1=float(f), fit_seconds=float(ft)))
        print(ds, name, len(cols), round(f, 4), round(ft, 1), flush=True)
    out[ds] = dict(n_features=len(imp), n_corr_pairs=len(pairs), n_redundant=len(drop),
                   top10_rf_importance=imp.head(10).round(4).to_dict(), subsets=rows,
                   example_pairs=[(a, b, round(v, 3)) for a, b, v in sorted(pairs, key=lambda x: -x[2])[:6]])
json.dump(out, open(f"{RES}/feature_selection.json", "w"), indent=1)
