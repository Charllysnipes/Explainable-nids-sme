"""Inference cost measured under identical conditions for every model: one CPU thread,
single-flow latency (median of 300 calls after warm-up) and batch throughput on the full test set."""
import json, time
import os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np
import joblib
from data_prep import LOADERS
from models import MODEL_ORDER, scores

OUT = os.path.join(ROOT, "results")
res = {}
for ds, loader in LOADERS.items():
    d = loader()
    X = d["X_test"]
    res[ds] = {}
    for name in MODEL_ORDER:
        m = joblib.load(f"{OUT}/models/{ds}_{name}.joblib")["model"]
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1
        if name == "XGBoost":
            m.set_params(n_jobs=1)
        if name == "LightGBM":
            m.set_params(n_jobs=1)
        Xn = X.to_numpy()
        rows = [X.iloc[[i]] for i in range(300)]
        for r in rows[:10]:
            scores(m, name, r)
        ts = []
        for r in rows:
            t = time.perf_counter(); scores(m, name, r); ts.append(time.perf_counter() - t)
        t = time.perf_counter(); scores(m, name, X); bt = time.perf_counter() - t
        res[ds][name] = dict(latency_ms_median=float(np.median(ts) * 1e3), latency_ms_p95=float(np.percentile(ts, 95) * 1e3),
                             throughput_flows_per_s=float(len(X) / bt))
        print(ds, name, {k: round(v, 3) for k, v in res[ds][name].items()}, flush=True)
json.dump(res, open(f"{OUT}/latency.json", "w"), indent=1)
