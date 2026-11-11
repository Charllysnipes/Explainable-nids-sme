"""Cross-dataset generalisation: CIC-IDS2017 <-> UNSW-NB15 on seven harmonised flow features.

Within-dataset scores on the same seven features are reported as the reference, so the
drop measures distribution shift between networks rather than the reduced feature set.
"""
import json
import os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix
from data_prep import _unsw_frames, DATA, _clean
from models import make_model, MODEL_ORDER, scores

OUT = os.path.join(ROOT, "results")
# The seven features that mean the same thing in both datasets: duration, plus packet count,
# byte count and mean packet length in each direction.
FEATS = ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "fwd_pkt_len_mean", "bwd_pkt_len_mean"]
CAP = 40000  # largest training sample, to keep run time reasonable
rng = np.random.default_rng(42)


def cic():
    """Build the seven harmonised features from CIC-IDS2017 column names."""
    df = pd.read_csv(f"{DATA}/probe/Intrusion-Detection-System-Using-Machine-Learning/data/CICIDS2017_sample.csv")
    df = _clean(df)
    X = pd.DataFrame({
        "duration_s": df["Flow Duration"].clip(lower=0) / 1e6,  # CIC-IDS2017 measures duration in microseconds
        "fwd_pkts": df["Total Fwd Packets"], "bwd_pkts": df["Total Backward Packets"],
        "fwd_bytes": df["Total Length of Fwd Packets"], "bwd_bytes": df["Total Length of Bwd Packets"],
        "fwd_pkt_len_mean": df["Fwd Packet Length Mean"], "bwd_pkt_len_mean": df["Bwd Packet Length Mean"]})
    y = (df["Label"] != "BENIGN").astype(int).values
    return X, y


def unsw():
    """Build the same seven features from UNSW-NB15 (s = source/forward, d = destination/backward)."""
    tr, te = _unsw_frames()
    df = pd.concat([tr, te])
    X = pd.DataFrame({"duration_s": df["dur"], "fwd_pkts": df["spkts"], "bwd_pkts": df["dpkts"],
                      "fwd_bytes": df["sbytes"], "bwd_bytes": df["dbytes"],
                      "fwd_pkt_len_mean": df["smean"], "bwd_pkt_len_mean": df["dmean"]})
    X["_y"] = df["label"].values
    X = X.drop_duplicates()  # remove duplicates of the seven features plus label
    y = X.pop("_y").values
    return X, y


def best_threshold(y, s):
    """Score cut-off with the best F1 (only needed for Isolation Forest)."""
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    f = [f1_score(y, (s >= q).astype(int), zero_division=0) for q in qs]
    return qs[int(np.argmax(f))]


def cap(X, y, n=CAP):
    """Take a random sample of at most n rows."""
    if len(y) > n:
        i = rng.choice(len(y), n, replace=False)
        return X.iloc[i], y[i]
    return X, y


sets = {"CIC-IDS2017": cic(), "UNSW-NB15": unsw()}
for k, (X, y) in sets.items():
    print(k, X.shape, y.mean())
# Reuse each model's tuned settings from run_main.py.
tuned = {ds: json.load(open(f"{OUT}/{ds}_main.json"))["models"] for ds in sets}
res = {}
# Train on one network (src), then test on the same network and on the other one (tgt).
for src in sets:
    Xs, ys = sets[src]
    Xs = np.log1p(Xs.clip(lower=0))  # log(1 + x) reduces the very long tails of byte and packet counts
    Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=42)
    Xa, ya = cap(Xa, ya)
    Xv, Xb2, yv, yb2 = train_test_split(Xb, yb, test_size=0.5, stratify=yb, random_state=42)
    sc = MinMaxScaler().fit(Xa)  # scaler fitted on the source training split only
    T = lambda X: pd.DataFrame(sc.transform(X), columns=FEATS)
    for name in MODEL_ORDER:
        p = {kk: (tuple(v) if isinstance(v, list) else v) for kk, v in tuned[src][name]["best_params"].items()}
        # With only seven features, every tree may use all of them.
        if name in ("XGBoost", "LightGBM", "RandomForest") and "colsample_bytree" in p:
            p["colsample_bytree"] = 1.0
        if name == "RandomForest":
            p["max_features"] = "sqrt"
        if name == "IsolationForest":
            p["max_features"] = 1.0
        m = make_model(name, p)
        if name == "IsolationForest":
            m.fit(T(Xa)[ya == 0]); thr = best_threshold(yv, scores(m, name, T(Xv)))
        else:
            m.fit(T(Xa), ya); thr = 0.5
        for tgt in sets:
            # Same network: held-out 15% test split. Other network: a sample of its records.
            if tgt == src:
                Xt, yt = Xb2, yb2
            else:
                Xt, yt = sets[tgt]
                Xt = np.log1p(Xt.clip(lower=0))
                Xt, yt = cap(Xt, yt, 60000)
            s = scores(m, name, T(Xt)); pr = (s >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
            r = dict(f1=float(f1_score(yt, pr)), recall=float(recall_score(yt, pr)),
                     fpr=float(fp / (fp + tn)), roc_auc=float(roc_auc_score(yt, s)))
            res.setdefault(src, {}).setdefault(tgt, {})[name] = r
            print(f"{src} -> {tgt} {name}: F1={r['f1']:.3f} AUC={r['roc_auc']:.3f} FPR={r['fpr']:.3f}", flush=True)
# Save all within- and cross-network results to results/cross_dataset.json.
json.dump(dict(features=FEATS, results=res), open(f"{OUT}/cross_dataset.json", "w"), indent=1)
