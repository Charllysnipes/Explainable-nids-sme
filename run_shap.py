"""Phase 4: TreeSHAP explanations for the gradient-boosted models.

SHAP values are computed with the exact TreeSHAP algorithm built into XGBoost
(pred_contribs=True) and LightGBM (pred_contrib=True). Stability is checked by
(a) Spearman rank correlation of global importance across 5 bootstrap resamples
and (b) agreement of top-10 features between XGBoost and LightGBM.
"""
import json
import os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from scipy.stats import spearmanr
from data_prep import LOADERS

OUT = os.path.join(ROOT, "results")
N = 3000
rng = np.random.default_rng(42)
res = {}
for ds, loader in LOADERS.items():
    d = loader()
    Xte, yte = d["X_test"], d["y_test"]
    idx = rng.choice(len(yte), min(N, len(yte)), replace=False)
    X, y = Xte.iloc[idx], yte[idx]
    bx = joblib.load(f"{OUT}/models/{ds}_XGBoost.joblib")
    bl = joblib.load(f"{OUT}/models/{ds}_LightGBM.joblib")
    sx = bx["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    sl = bl["model"].booster_.predict(X, pred_contrib=True)[:, :-1]
    cols = list(X.columns)
    gx = pd.Series(np.abs(sx).mean(0), index=cols).sort_values(ascending=False)
    gl = pd.Series(np.abs(sl).mean(0), index=cols).sort_values(ascending=False)
    # bootstrap stability of the global ranking
    boots = []
    base = np.abs(sx).mean(0)
    for b in range(5):
        j = rng.choice(len(y), len(y), replace=True)
        boots.append(spearmanr(base, np.abs(sx[j]).mean(0)).statistic)
    top10x, top10l = list(gx.index[:10]), list(gl.index[:10])
    # local explanation: highest-confidence correctly detected attack
    p = bx["model"].predict_proba(X)[:, 1]
    tp = np.where((y == 1) & (p >= 0.5))[0]
    k = tp[np.argmax(p[tp])] if len(tp) else 0
    local = pd.Series(sx[k], index=cols)
    local = local.reindex(local.abs().sort_values(ascending=False).index)[:8]
    # attack-family explanation: mean |SHAP| top-3 per family
    fam = pd.Series(d["cat_test"][idx])
    per_family = {}
    for c in fam.unique():
        m = (fam == c).values
        if m.sum() >= 10:
            per_family[c] = list(pd.Series(np.abs(sx[m]).mean(0), index=cols).sort_values(ascending=False).index[:3])
    res[ds] = dict(global_xgb=gx[:15].round(5).to_dict(), global_lgbm=gl[:15].round(5).to_dict(),
                   bootstrap_spearman=[float(v) for v in boots],
                   top10_overlap_xgb_lgbm=len(set(top10x) & set(top10l)),
                   rank_corr_xgb_lgbm=float(spearmanr(np.abs(sx).mean(0), np.abs(sl).mean(0)).statistic),
                   local_example=dict(index=int(idx[k]), family=str(fam.iloc[k]), prob=float(p[k]),
                                      contributions=local.round(4).to_dict()),
                   per_family_top3=per_family,
                   additivity_max_err=float(np.max(np.abs(
                       bx["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True).sum(1)
                       - bx["model"].get_booster().predict(xgb.DMatrix(X), output_margin=True)))))
    np.save(f"{OUT}/{ds}_shap_xgb.npy", sx)
    np.save(f"{OUT}/{ds}_shap_idx.npy", idx)
    print(ds, "top5", top10x[:5], "boot", np.round(boots, 3), "overlap", res[ds]["top10_overlap_xgb_lgbm"], flush=True)
json.dump(res, open(f"{OUT}/shap.json", "w"), indent=1)
