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

# Load the trained XGBoost model once, when the service starts (created by run_main.py).
DS = os.environ.get("NIDS_DATASET", "CIC-IDS2017")
MODELS = os.path.join(ROOT, "results/models")
bundle = joblib.load(f"{MODELS}/{DS}_XGBoost.joblib")
model, thr, cols = bundle["model"], bundle["threshold"], bundle["columns"]  # model, alert cut-off, feature order
booster = model.get_booster()  # the raw XGBoost trees, needed for exact TreeSHAP values
# Training-time scaler and column list, so raw CICFlowMeter rows are transformed exactly as in training.
pre = joblib.load(f"{MODELS}/{DS}_preprocessor.joblib") if os.path.exists(f"{MODELS}/{DS}_preprocessor.joblib") else None
MAX_FLOWS = 5000  # largest request accepted, protects the service from oversized uploads
app = Flask(__name__)


def explain(X):
    """Score each flow and return its attack probability, alert flag and three main SHAP reasons."""
    dm = xgb.DMatrix(X)
    p = booster.predict(dm)  # probability that each flow is an attack
    # pred_contribs=True gives the exact TreeSHAP contribution of every feature (last column = base value).
    contrib = booster.predict(dm, pred_contribs=True)[:, :-1]
    out = []
    for i in range(len(X)):
        top = np.argsort(-np.abs(contrib[i]))[:3]  # the three features with the largest effect, either way
        out.append(dict(attack_probability=float(p[i]), alert=bool(p[i] >= thr),
                        top_features=[[cols[j], round(float(contrib[i, j]), 4)] for j in top]))
    return out


def error(msg, code=400):
    """Return a plain-language error message as JSON with an HTTP error code."""
    return jsonify(error=msg), code


@app.get("/health")
def health():
    """Quick check that the service is running and which model it serves."""
    return jsonify(status="ok", dataset=DS, model="XGBoost", n_features=len(cols), threshold=thr,
                   raw_input_supported=pre is not None)


@app.get("/model_info")
def model_info():
    """Return the deployed model's test results, read from the results file."""
    m = json.load(open(f"{ROOT}/results/{DS}_main.json"))["models"]["XGBoost"]
    return jsonify({k: m[k] for k in ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "best_params"]})


@app.post("/predict")
def predict():
    """Score flows that are already scaled (used by the replay client for the live monitor)."""
    t0 = time.perf_counter()
    body = request.get_json(silent=True)
    if not body or "flows" not in body:
        return error("Request body must be JSON with a 'flows' list.")
    # Put the columns in training order; any missing feature is filled with 0.
    X = pd.DataFrame(body["flows"]).reindex(columns=cols, fill_value=0).astype(np.float32)
    return jsonify(results=explain(X), server_ms=(time.perf_counter() - t0) * 1000)


@app.post("/predict_raw")
def predict_raw():
    """Score raw CICFlowMeter rows uploaded through the dashboard: validate, scale, score, explain."""
    t0 = time.perf_counter()
    if pre is None:
        return error("Raw input is not available for this model.", 501)
    body = request.get_json(silent=True)
    # Check 1: the request must contain a non-empty list of flows.
    if not body or not isinstance(body.get("flows"), list) or not body["flows"]:
        return error("Request body must be JSON with a non-empty 'flows' list.")
    # Check 2: reject requests that are too large (HTTP 413).
    if len(body["flows"]) > MAX_FLOWS:
        return error(f"Too many flows: {len(body['flows'])}. The limit is {MAX_FLOWS} per request.", 413)
    df = pd.DataFrame(body["flows"])
    df.columns = [str(c).strip() for c in df.columns]
    # Check 3: every column the model needs must be present; the message names the missing ones.
    missing = [c for c in pre["columns"] if c not in df.columns]
    if missing:
        return error(f"{len(missing)} required CICFlowMeter column(s) missing, e.g. {', '.join(missing[:5])}.")
    # Check 4: turn text and infinite values into NaN, and remember which rows had them.
    raw = df[pre["columns"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = raw.isna().any(axis=1)
    # Same transformation as in training: saved Min-Max scaler, then clip to [-1, 2].
    X = pd.DataFrame(np.clip(pre["scaler"].transform(raw.fillna(0)), -1, 2), columns=pre["columns"]).astype(np.float32)
    res = explain(X)
    for i, r in enumerate(res):
        r["row"] = i + 1
        r["invalid_values"] = bool(bad.iloc[i])     # non-numeric / infinite values were replaced by 0
    return jsonify(results=res, n_invalid_rows=int(bad.sum()), server_ms=(time.perf_counter() - t0) * 1000)


if __name__ == "__main__":
    # Start the service: python3 deploy/api.py 5005 (only reachable from this computer).
    app.run(host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 5005, threaded=False)
