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
FEATS = ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "fwd_pkt_len_mean", "bwd_pkt_len_mean"]
CAP = 40000
rng = np.random.default_rng(42)


def cic():
    df = pd.read_csv(f"{DATA}/probe/Intrusion-Detection-System-Using-Machine-Learning/data/CICIDS2017_sample.csv")
    df = _clean(df)
    X = pd.DataFrame({
        "duration_s": df["Flow Duration"].clip(lower=0) / 1e6,
        "fwd_pkts": df["Total Fwd Packets"], "bwd_pkts": df["Total Backward Packets"],
        "fwd_bytes": df["Total Length of Fwd Packets"], "bwd_bytes": df["Total Length of Bwd Packets"],
        "fwd_pkt_len_mean": df["Fwd Packet Length Mean"], "bwd_pkt_len_mean": df["Bwd Packet Length Mean"]})
    y = (df["Label"] != "BENIGN").astype(int).values
    return X, y


def unsw():
    tr, te = _unsw_frames()
    df = pd.concat([tr, te])
    X = pd.DataFrame({"duration_s": df["dur"], "fwd_pkts": df["spkts"], "bwd_pkts": df["dpkts"],
                      "fwd_bytes": df["sbytes"], "bwd_bytes": df["dbytes"],
                      "fwd_pkt_len_mean": df["smean"], "bwd_pkt_len_mean": df["dmean"]})
    X["_y"] = df["label"].values
    X = X.drop_duplicates()
    y = X.pop("_y").values
    return X, y


def best_threshold(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    f = [f1_score(y, (s >= q).astype(int), zero_division=0) for q in qs]
    return qs[int(np.argmax(f))]


def cap(X, y, n=CAP):
    if len(y) > n:
        i = rng.choice(len(y), n, replace=False)
        return X.iloc[i], y[i]
    return X, y


sets = {"CIC-IDS2017": cic(), "UNSW-NB15": unsw()}
for k, (X, y) in sets.items():
    print(k, X.shape, y.mean())
tuned = {ds: json.load(open(f"{OUT}/{ds}_main.json"))["models"] for ds in sets}
res = {}
for src in sets:
    Xs, ys = sets[src]
    Xs = np.log1p(Xs.clip(lower=0))
    Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=42)
    Xa, ya = cap(Xa, ya)
    Xv, Xb2, yv, yb2 = train_test_split(Xb, yb, test_size=0.5, stratify=yb, random_state=42)
    sc = MinMaxScaler().fit(Xa)
    T = lambda X: pd.DataFrame(sc.transform(X), columns=FEATS)
    for name in MODEL_ORDER:
        p = {kk: (tuple(v) if isinstance(v, list) else v) for kk, v in tuned[src][name]["best_params"].items()}
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
json.dump(dict(features=FEATS, results=res), open(f"{OUT}/cross_dataset.json", "w"), indent=1)
