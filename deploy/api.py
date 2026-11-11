"""Core engine: Flask REST API that pre-processes, scores and explains network flows.

Endpoints
  GET  /health        service status and model version
  GET  /model_info    test-set metrics of the deployed model (from the results files)
  POST /predict       {"flows": [{feature: scaled value, ...}]}   -> already pre-processed flows
  POST /predict_raw   {"flows": [{CICFlowMeter column: raw value, ...}]} -> raw flows; the saved
                      training-time scaler is applied, exactly as in the offline pipeline
Each result: attack_probability, alert (bool), top_features [[name, SHAP contribution], x3]
"""
import os, sys, time, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, so the code runs from any folder
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from flask import Flask, request, jsonify

DS = os.environ.get("NIDS_DATASET", "CIC-IDS2017")
MODELS = os.path.join(ROOT, "results/models")
bundle = joblib.load(f"{MODELS}/{DS}_XGBoost.joblib")
model, thr, cols = bundle["model"], bundle["threshold"], bundle["columns"]
booster = model.get_booster()
pre = joblib.load(f"{MODELS}/{DS}_preprocessor.joblib") if os.path.exists(f"{MODELS}/{DS}_preprocessor.joblib") else None
MAX_FLOWS = 5000
app = Flask(__name__)


def explain(X):
    dm = xgb.DMatrix(X)
    p = booster.predict(dm)
    contrib = booster.predict(dm, pred_contribs=True)[:, :-1]
    out = []
    for i in range(len(X)):
        top = np.argsort(-np.abs(contrib[i]))[:3]
        out.append(dict(attack_probability=float(p[i]), alert=bool(p[i] >= thr),
                        top_features=[[cols[j], round(float(contrib[i, j]), 4)] for j in top]))
    return out


def error(msg, code=400):
    return jsonify(error=msg), code


@app.get("/health")
def health():
    return jsonify(status="ok", dataset=DS, model="XGBoost", n_features=len(cols), threshold=thr,
                   raw_input_supported=pre is not None)


@app.get("/model_info")
def model_info():
    m = json.load(open(f"{ROOT}/results/{DS}_main.json"))["models"]["XGBoost"]
    return jsonify({k: m[k] for k in ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "best_params"]})


@app.post("/predict")
def predict():
    t0 = time.perf_counter()
    body = request.get_json(silent=True)
    if not body or "flows" not in body:
        return error("Request body must be JSON with a 'flows' list.")
    X = pd.DataFrame(body["flows"]).reindex(columns=cols, fill_value=0).astype(np.float32)
    return jsonify(results=explain(X), server_ms=(time.perf_counter() - t0) * 1000)


@app.post("/predict_raw")
def predict_raw():
    t0 = time.perf_counter()
    if pre is None:
        return error("Raw input is not available for this model.", 501)
    body = request.get_json(silent=True)
    if not body or not isinstance(body.get("flows"), list) or not body["flows"]:
        return error("Request body must be JSON with a non-empty 'flows' list.")
    if len(body["flows"]) > MAX_FLOWS:
        return error(f"Too many flows: {len(body['flows'])}. The limit is {MAX_FLOWS} per request.", 413)
    df = pd.DataFrame(body["flows"])
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in pre["columns"] if c not in df.columns]
    if missing:
        return error(f"{len(missing)} required CICFlowMeter column(s) missing, e.g. {', '.join(missing[:5])}.")
    raw = df[pre["columns"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = raw.isna().any(axis=1)
    X = pd.DataFrame(np.clip(pre["scaler"].transform(raw.fillna(0)), -1, 2), columns=pre["columns"]).astype(np.float32)
    res = explain(X)
    for i, r in enumerate(res):
        r["row"] = i + 1
        r["invalid_values"] = bool(bad.iloc[i])     # non-numeric / infinite values were replaced by 0
    return jsonify(results=res, n_invalid_rows=int(bad.sum()), server_ms=(time.perf_counter() - t0) * 1000)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 5005, threaded=False)
