"""Training evidence: learning curves, confusion matrices, ROC curves, training log and model inventory.

Re-fits XGBoost and LightGBM with their tuned hyperparameters while logging the loss on the
training and validation sets at every boosting round (same seed and data, so the models are
identical to the saved ones), and reads the MLP loss curve stored inside the saved model.
"""
import json, os, time, datetime
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, f1_score
import xgboost as xgb, lightgbm as lgb
from data_prep import LOADERS
from models import make_model, MODEL_ORDER, scores

RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "figs")
DS = ["CIC-IDS2017", "UNSW-NB15", "NSL-KDD"]
LBL = {"NaiveBayes": "NB", "RandomForest": "RF", "XGBoost": "XGBoost", "LightGBM": "LightGBM", "MLP": "MLP", "IsolationForest": "IF"}
COL = dict(zip(MODEL_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "savefig.dpi": 250})
log = open(f"{RES}/training_log.txt", "w")
def L(msg):
    """Print a time-stamped line and append it to results/training_log.txt."""
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True); log.write(line + "\n"); log.flush()

curves, rocs, cms = {}, {}, {}
for ds in DS:
    d = LOADERS[ds]()
    main = json.load(open(f"{RES}/{ds}_main.json"))["models"]
    Xtr, ytr, Xva, yva, Xte, yte = d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"]
    L(f"[{ds}] loaded: train={Xtr.shape}, val={Xva.shape}, test={Xte.shape}")
    curves[ds] = {}
    # XGBoost with per-round logloss
    p = main["XGBoost"]["best_params"]
    m = make_model("XGBoost", p); t = time.perf_counter()
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
    ev = m.evals_result()
    curves[ds]["XGBoost"] = dict(train=ev["validation_0"]["logloss"], val=ev["validation_1"]["logloss"])
    L(f"[{ds}] XGBoost {p} fitted in {time.perf_counter()-t:.1f}s; final logloss train={curves[ds]['XGBoost']['train'][-1]:.4f} val={curves[ds]['XGBoost']['val'][-1]:.4f}")
    # LightGBM with per-round logloss
    p = main["LightGBM"]["best_params"]
    m = make_model("LightGBM", p); t = time.perf_counter()
    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], eval_names=["train", "val"], eval_metric="binary_logloss")
    ev = m.evals_result_
    curves[ds]["LightGBM"] = dict(train=ev["train"]["binary_logloss"], val=ev["val"]["binary_logloss"])
    L(f"[{ds}] LightGBM {p} fitted in {time.perf_counter()-t:.1f}s; final logloss train={curves[ds]['LightGBM']['train'][-1]:.4f} val={curves[ds]['LightGBM']['val'][-1]:.4f}")
    # MLP: loss curve and early-stopping validation accuracy stored in the saved model
    mlp = joblib.load(f"{RES}/models/{ds}_MLP.joblib")["model"]
    curves[ds]["MLP"] = dict(train=list(map(float, mlp.loss_curve_)), val_acc=list(map(float, mlp.validation_scores_)))
    L(f"[{ds}] MLP {main['MLP']['best_params']} stopped after {mlp.n_iter_} epochs; final train loss={mlp.loss_curve_[-1]:.4f}, best val acc={max(mlp.validation_scores_):.4f}")
    # Random Forest: validation F1 as trees are added
    rf = joblib.load(f"{RES}/models/{ds}_RandomForest.joblib")["model"]
    probs = np.zeros(len(yva)); f1s = []
    Xv = Xva.to_numpy()
    for k, est in enumerate(rf.estimators_, 1):
        probs += est.predict_proba(Xv)[:, 1]
        if k in (1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300) or k == len(rf.estimators_):
            f1s.append((k, float(f1_score(yva, (probs / k) >= 0.5))))
    curves[ds]["RandomForest"] = f1s
    L(f"[{ds}] RandomForest {main['RandomForest']['best_params']}: validation F1 with 1 tree={f1s[0][1]:.4f}, all trees={f1s[-1][1]:.4f}")
    # Isolation Forest threshold selection on validation
    b = joblib.load(f"{RES}/models/{ds}_IsolationForest.joblib")
    s = scores(b["model"], "IsolationForest", Xva)
    qs = np.quantile(s, np.linspace(0.01, 0.99, 99))
    curves[ds]["IsolationForest"] = [(float(q), float(f1_score(yva, (s >= q).astype(int)))) for q in qs]
    L(f"[{ds}] IsolationForest threshold chosen on validation = {b['threshold']:.4f}")
    # ROC + confusion matrices on the test set
    rocs[ds], cms[ds] = {}, {}
    for name in MODEL_ORDER:
        bb = joblib.load(f"{RES}/models/{ds}_{name}.joblib")
        sc = scores(bb["model"], name, Xte)
        fpr, tpr, _ = roc_curve(yte, sc)
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
        rocs[ds][name] = (fpr[idx].tolist(), tpr[idx].tolist())
        cms[ds][name] = confusion_matrix(yte, (sc >= bb["threshold"]).astype(int)).tolist()
    L(f"[{ds}] ROC curves and confusion matrices computed for 6 models")
json.dump(dict(curves=curves, cms=cms), open(f"{RES}/training_evidence.json", "w"))

# ---------------- figures
# Figure 15: training and validation loss per round (XGBoost, LightGBM) and the MLP loss curve.
fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.0))
for i, ds in enumerate(DS):
    for j, (name, key) in enumerate([("XGBoost", "logloss"), ("LightGBM", "logloss"), ("MLP", "loss")]):
        ax = axs[i, j]; c = curves[ds][name]
        ax.plot(c["train"], color="#2a78d6", lw=1.2, label="training loss")
        if "val" in c:
            ax.plot(c["val"], color="#eb6834", lw=1.2, label="validation loss")
            ax.set_xlabel("boosting round", fontsize=7)
        else:
            ax2 = ax.twinx(); ax2.plot(c["val_acc"], color="#1baf7a", lw=1.2, label="validation accuracy")
            ax2.set_ylabel("val. accuracy", fontsize=6.5, color="#1b7a55"); ax2.tick_params(labelsize=6); ax2.grid(False)
            ax.set_xlabel("epoch", fontsize=7)
        ax.set_title(f"{ds} – {name}", fontsize=7.5); ax.tick_params(labelsize=6); ax.set_ylabel("log-loss", fontsize=6.5)
        if i == 0 and j < 2: ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/training_curves.png", facecolor="white"); plt.close(fig)

# Figure 18: ROC curves of all six models on each test set.
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3))
for ax, ds in zip(axs, DS):
    for name in MODEL_ORDER:
        f, t = rocs[ds][name]; ax.plot(f, t, color=COL[name], lw=1.1, label=LBL[name])
    ax.plot([0, 1], [0, 1], ls="--", color="#999", lw=0.7)
    ax.set_title(ds, fontsize=8); ax.set_xlabel("False positive rate", fontsize=7); ax.tick_params(labelsize=6)
axs[0].set_ylabel("True positive rate", fontsize=7); axs[2].legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(f"{FIG}/roc_curves.png", facecolor="white"); plt.close(fig)

# Figure 17: confusion matrices (B = benign, A = attack), shaded by row percentage.
fig, axs = plt.subplots(3, 6, figsize=(7.4, 4.0))
for i, ds in enumerate(DS):
    for j, name in enumerate(MODEL_ORDER):
        ax = axs[i, j]; M = np.array(cms[ds][name]); Mn = M / M.sum(1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
        for a in range(2):
            for b_ in range(2):
                ax.text(b_, a, f"{M[a, b_]:,}", ha="center", va="center", fontsize=5.3, color="white" if Mn[a, b_] > 0.6 else "#111")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["B", "A"], fontsize=6); ax.set_yticklabels(["B", "A"], fontsize=6)
        if i == 0: ax.set_title(LBL[name], fontsize=7)
        if j == 0: ax.set_ylabel(f"{ds}\ntrue", fontsize=6.5)
        if i == 2: ax.set_xlabel("predicted", fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{FIG}/confusion_matrices.png", facecolor="white"); plt.close(fig)

# Figure 16: Random Forest F1 as trees are added, and the Isolation Forest threshold search.
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2))
for ds in DS:
    k, f = zip(*curves[ds]["RandomForest"]); axs[0].plot(k, f, marker="o", ms=2.5, lw=1.1, label=ds)
    q, f = zip(*curves[ds]["IsolationForest"]); axs[1].plot(np.linspace(1, 99, 99), f, lw=1.1, label=ds)
axs[0].set_xscale("log"); axs[0].set_xlabel("number of trees", fontsize=7); axs[0].set_ylabel("validation F1", fontsize=7)
axs[0].set_title("Random Forest: validation F1 vs ensemble size", fontsize=8)
axs[1].set_xlabel("score percentile used as threshold", fontsize=7); axs[1].set_ylabel("validation F1", fontsize=7)
axs[1].set_title("Isolation Forest: threshold selection", fontsize=8)
for ax in axs: ax.tick_params(labelsize=6); ax.legend(fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(f"{FIG}/rf_if_tuning.png", facecolor="white"); plt.close(fig)

# Table 15: list of saved model files and their sizes.
inv = []
for f in sorted(os.listdir(f"{RES}/models")):
    inv.append((f, os.path.getsize(f"{RES}/models/{f}") / 1e6))
json.dump(inv, open(f"{RES}/model_inventory.json", "w"), indent=1)
L(f"saved model inventory: {len(inv)} files, {sum(s for _, s in inv):.1f} MB")
log.close()
