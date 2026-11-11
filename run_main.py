"""Phases 2-3: tuning, training, held-out evaluation, latency, robustness, unseen attacks.

Usage: python3 run_main.py <dataset name>
Writes results/<dataset>_main.json and saves fitted models to results/models/.
"""
import json, sys, time, os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix)
from data_prep import LOADERS
from models import make_model, SPACES, MODEL_ORDER, sample_params, scores

SEED = 42
OUT = os.path.join(ROOT, "results")
os.makedirs(f"{OUT}/models", exist_ok=True)
N_ITER = 6        # hyperparameter settings tried per model in the random search
TUNE_CAP = 30000  # rows used for the hyperparameter search (speed); final fit uses all training rows


def best_threshold(y, s):
    """Pick the score cut-off that gives the best F1 on the validation set (used for Isolation Forest)."""
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))  # 197 candidate cut-offs across the scores
    f1s = [f1_score(y, (s >= q).astype(int), zero_division=0) for q in qs]
    return float(qs[int(np.argmax(f1s))])


def metrics(y, pred, s):
    """All test metrics reported in the paper, from the true labels, the predictions and the scores."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=float((tp + tn) / len(y)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                fpr=float(fp / (fp + tn)), roc_auc=float(roc_auc_score(y, s)),  # FPR = false alarm rate
                pr_auc=float(average_precision_score(y, s)), tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def fit(name, params, X, y):
    """Train one model and return it with the training time in seconds."""
    m = make_model(name, params)
    t = time.perf_counter()
    if name == "IsolationForest":
        m.fit(X[y == 0])                      # trained on benign traffic only
    else:
        m.fit(X, y)
    return m, time.perf_counter() - t


def predict(name, m, X, thr):
    """Return 0/1 predictions (score >= threshold means attack) and the raw scores."""
    s = scores(m, name, X)
    return (s >= thr).astype(int), s


def main(ds):
    """Tune, train and evaluate all six models on one dataset, saving models and results."""
    rng = np.random.default_rng(SEED)
    d = LOADERS[ds]()  # leakage-free splits from data_prep.py
    # training-time scaler and column list, used by the API to score raw CICFlowMeter rows
    joblib.dump(dict(scaler=d["scaler"], columns=d["raw_columns"]), f"{OUT}/models/{ds}_preprocessor.joblib")
    Xtr, ytr, Xva, yva, Xte, yte = d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]
    # For speed, the search uses at most 30,000 random training rows.
    if len(ytr) > TUNE_CAP:
        idx = rng.choice(len(ytr), TUNE_CAP, replace=False)
        Xtu, ytu = Xtr.iloc[idx], ytr[idx]
    else:
        Xtu, ytu = Xtr, ytr
    res = dict(dataset=ds, n_features=Xtr.shape[1], n_train=len(ytr), n_val=len(yva), n_test=len(yte),
               attack_rate_train=float(ytr.mean()), attack_rate_test=float(yte.mean()),
               stats=d["stats"], test_family_counts=pd.Series(d["cat_test"]).value_counts().to_dict(),
               models={})
    # Keep results of models already run, so one model can be re-run on its own (MODELS=XGBoost ...).
    prev = f"{OUT}/{ds}_main.json"
    if os.path.exists(prev):
        res["models"] = json.load(open(prev))["models"]
    todo = os.environ.get("MODELS", ",".join(MODEL_ORDER)).split(",")
    for name in [m for m in MODEL_ORDER if m in todo]:
        rng = np.random.default_rng(SEED + MODEL_ORDER.index(name))  # own seed per model, same every run
        print(f"[{ds}] tuning {name}", flush=True)
        # 1) Random search: try N_ITER settings, score each on the VALIDATION set (never the test set).
        trials = []
        for p in sample_params(SPACES[name], N_ITER, rng):
            m, _ = fit(name, p, Xtu, ytu)
            s = scores(m, name, Xva)
            thr = best_threshold(yva, s) if name == "IsolationForest" else 0.5
            f = f1_score(yva, (s >= thr).astype(int))
            trials.append((f, p))
        best_f, best_p = max(trials, key=lambda t: t[0])
        # 2) Refit the best setting on the full training split.
        m, fit_time = fit(name, best_p, Xtr, ytr)
        s_va = scores(m, name, Xva)
        thr = best_threshold(yva, s_va) if name == "IsolationForest" else 0.5
        # 3) Evaluate once on the held-out TEST split.
        pred, s = predict(name, m, Xte, thr)
        r = metrics(yte, pred, s)
        r.update(best_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in best_p.items()},
                 val_f1=float(best_f), threshold=thr, fit_seconds=fit_time)
        # per-family detection rate (recall on each attack family; specificity for benign)
        fam = pd.DataFrame(dict(c=d["cat_test"], y=yte, p=pred))
        r["family_detection"] = {c: float((g.p == g.y).mean()) for c, g in fam.groupby("c")}
        # unseen-attack generalisation (NSL-KDD has 17 attack types absent from training)
        if "novel_test" in d:
            nov = d["novel_test"]
            seen_att = (~nov) & (yte == 1)
            r["recall_seen_attacks"] = float(pred[seen_att].mean())   # attacks of types seen in training
            r["recall_novel_attacks"] = float(pred[nov].mean())       # attacks of types never seen
        # latency: single-flow and batch throughput
        one = Xte.iloc[[0]]
        for _ in range(5):
            scores(m, name, one)  # warm-up calls, so start-up costs are not timed
        ts = []
        for i in range(100):
            row = Xte.iloc[[i]]
            t = time.perf_counter(); scores(m, name, row); ts.append(time.perf_counter() - t)
        t = time.perf_counter(); scores(m, name, Xte); bt = time.perf_counter() - t
        r["latency_ms_single"] = float(np.median(ts) * 1000)          # median time for one flow, in ms
        r["throughput_flows_per_s"] = float(len(yte) / bt)           # flows per second in one batch
        # robustness: additive Gaussian noise on the scaled test features
        rob = {}
        noise_rng = np.random.default_rng(7)
        for sigma in [0.0, 0.01, 0.05, 0.1, 0.2]:  # noise strength; 0.0 = clean data
            Xn = Xte + noise_rng.normal(0, sigma, Xte.shape).astype(np.float32) if sigma else Xte
            pn, sn = predict(name, m, Xn, thr)
            rob[str(sigma)] = dict(f1=float(f1_score(yte, pn)), roc_auc=float(roc_auc_score(yte, sn)))
        r["robustness"] = rob
        res["models"][name] = r
        # Save the trained model with its threshold and column list (the API loads this file).
        joblib.dump(dict(model=m, threshold=thr, columns=list(Xtr.columns)),
                    f"{OUT}/models/{ds}_{name}.joblib")
        print(f"[{ds}] {name}: F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} FPR={r['fpr']:.4f} "
              f"lat={r['latency_ms_single']:.2f}ms params={best_p}", flush=True)
        json.dump(res, open(f"{OUT}/{ds}_main.json", "w"), indent=1)  # save after each model
    if set(res["models"]) != set(MODEL_ORDER):
        print("partial run", ds); return
    # Save every model's test predictions; run_cv_stats.py uses them for McNemar's test.
    preds = {}
    for name in MODEL_ORDER:
        b = joblib.load(f"{OUT}/models/{ds}_{name}.joblib")
        preds[name] = predict(name, b["model"], Xte, b["threshold"])[0].tolist()
    json.dump(dict(y=yte.tolist(), preds=preds), open(f"{OUT}/{ds}_testpreds.json", "w"))
    print("done", ds)


if __name__ == "__main__":
    main(sys.argv[1])  # e.g. python3 run_main.py CIC-IDS2017
