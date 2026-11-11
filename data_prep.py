"""Data acquisition and preparation for the three benchmark datasets.

Each loader returns a dict with train / val / test splits:
    X_* : pandas DataFrame of numeric, scaled features
    y_* : numpy array, 0 = benign, 1 = attack
    cat_* : attack family label (string) for per-class analysis
Identifier fields are removed, duplicates and missing/infinite values are
handled, categorical fields are one-hot encoded and Min-Max scaling is fitted
on the training split only (no leakage).
"""
import numpy as np
import os
ROOT = os.path.dirname(os.path.abspath(__file__))  # repository root, so the code runs from any folder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

SEED = 42  # fixed random seed so every split and model can be reproduced exactly
# Folder holding the downloaded datasets (scripts/get_data.sh fills ./data by default).
DATA = os.environ.get("NIDS_DATA", os.path.join(ROOT, "data"))

# NSL-KDD files have no header row, so the 41 feature names plus the label
# ("attack") and the difficulty score are listed here in file order.
NSL_COLS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack", "difficulty",
]

# Maps each NSL-KDD attack name to its family, so results can be reported per family
# (DoS = denial of service, Probe = scanning, R2L = remote to local, U2R = user to root).
NSL_FAMILY = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "named": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "worm": "R2L", "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
    "normal": "Normal",
}


def _clean(df):
    """Remove rows with infinite or missing values, then remove exact duplicate rows."""
    # CICFlowMeter writes 'Infinity' when a flow lasts zero seconds; such rows cannot be scaled.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Duplicates are removed BEFORE splitting, so the same row cannot sit in both train and test.
    return df.drop_duplicates()


def _drop_constant(train, *others):
    """Drop columns that have only one value in the training split (they carry no information)."""
    keep = [c for c in train.columns if train[c].nunique() > 1]  # decided on training data only
    return (train[keep],) + tuple(o[keep] for o in others)


def _finish(Xtr, Xva, Xte, ytr, yva, yte, ctr, cva, cte, name, extra=None):
    """Last shared step for every dataset: drop constant columns, scale, and package the splits."""
    Xtr, Xva, Xte = _drop_constant(Xtr, Xva, Xte)
    # Min-Max scaling to [0, 1], fitted on the TRAINING split only, so no test statistics leak in.
    scaler = MinMaxScaler().fit(Xtr)
    # Apply the same scaler to every split; values outside the training range are clipped to [-1, 2].
    f = lambda X: pd.DataFrame(np.clip(scaler.transform(X), -1, 2), columns=Xtr.columns,
                               index=X.index).astype(np.float32)
    out = dict(name=name, X_train=f(Xtr), X_val=f(Xva), X_test=f(Xte),
               y_train=np.asarray(ytr), y_val=np.asarray(yva), y_test=np.asarray(yte),
               cat_train=np.asarray(ctr), cat_val=np.asarray(cva), cat_test=np.asarray(cte),
               scaler=scaler, raw_columns=list(Xtr.columns))  # scaler and columns are reused by the API
    if extra:
        out.update(extra)  # dataset-specific extras, e.g. cleaning statistics or the unseen-attack mask
    return out


def load_cicids2017():
    """CIC-IDS2017: published sampled subset covering all attack families (Yang et al., MTH-IDS)."""
    df = pd.read_csv(f"{DATA}/probe/Intrusion-Detection-System-Using-Machine-Learning/data/CICIDS2017_sample.csv")
    df.columns = [c.strip() for c in df.columns]  # some column names start with a space
    n0 = len(df)
    df = _clean(df)
    stats = dict(raw_rows=n0, clean_rows=len(df))  # how many rows cleaning removed (reported in the paper)
    cat = df.pop("Label").astype(str)              # attack family name, e.g. DoS, PortScan, BENIGN
    cat = cat.replace({"BENIGN": "Benign"})
    y = (cat != "Benign").astype(int).values       # binary target: 1 = attack, 0 = benign
    X = df.astype(np.float64)
    # 70/15/15 stratified split on the attack family
    Xtr, Xtmp, ytr, ytmp, ctr, ctmp = train_test_split(X, y, cat, test_size=0.30,
                                                        stratify=cat, random_state=SEED)
    # The held-back 30% is halved into validation (for tuning) and test (used once at the end).
    Xva, Xte, yva, yte, cva, cte = train_test_split(Xtmp, ytmp, ctmp, test_size=0.50,
                                                    stratify=ctmp, random_state=SEED)
    return _finish(Xtr, Xva, Xte, ytr, yva, yte, ctr, cva, cte, "CIC-IDS2017", dict(stats=stats))


def _unsw_frames():
    """Read the official UNSW-NB15 training and testing partitions and merge the two public copies."""
    base = f"{DATA}/unsw"
    alt = f"{DATA}/probe/datasets/unsw-nb15"
    # InitRoot copy holds all 42 features (categoricals label-coded); comsyssec copy holds the
    # original string categoricals and attack family. Rows are identical and in the same order
    # (verified: dur, sbytes and label match on 100% of rows).
    tr = pd.read_csv(f"{base}/UNSW_NB15_testing-set.csv")   # file names are swapped in that copy
    te = pd.read_csv(f"{base}/UNSW_NB15_training-set.csv")
    tr_s = pd.read_csv(f"{alt}/training-flow.csv")
    te_s = pd.read_csv(f"{alt}/test-flow.csv")
    for full, s in ((tr, tr_s), (te, te_s)):
        assert (full["sbytes"].values == s["sbytes"].values).all()  # stop if the two copies ever disagree
        full["proto"] = s["protocol"].values      # protocol as text, e.g. tcp, udp
        full["service"] = s["service"].values     # service as text, e.g. http, dns
        full["attack_cat"] = s["attack_name"].values  # attack family, used for per-family results
    # recover the state string from the label code is not needed; keep code as categorical
    for full in (tr, te):
        full["state"] = full["xState"].astype(str)
        full.drop(columns=["id", "xProt", "xServ", "xState"], inplace=True)  # row id and coded duplicates
    return tr, te


def load_unsw():
    """UNSW-NB15 official partition: 175,341 training / 82,332 testing records."""
    tr, te = _unsw_frames()
    stats = dict(raw_train=len(tr), raw_test=len(te))
    # deduplicate on features + binary label: identical flows that differ only in attack_cat are duplicates
    tr = tr.replace([np.inf, -np.inf], np.nan).dropna()
    tr = tr.drop_duplicates(subset=[c for c in tr.columns if c != "attack_cat"])
    te = te.replace([np.inf, -np.inf], np.nan).dropna()  # test kept intact apart from NaN
    stats.update(clean_train=len(tr), clean_test=len(te))
    cats = ["proto", "service", "state"]  # text columns that must be one-hot encoded
    # keep the 10 most frequent protocol values, rest -> 'other'
    top_proto = tr["proto"].value_counts().index[:10]
    for d in (tr, te):
        d["proto"] = np.where(d["proto"].isin(top_proto), d["proto"], "other")
    # Separate the labels from the features.
    ctr_all = tr.pop("attack_cat").astype(str).values
    cte = te.pop("attack_cat").astype(str).values
    ytr_all = tr.pop("label").values
    yte = te.pop("label").values
    # One-hot encode train and test together so both get exactly the same columns.
    allX = pd.get_dummies(pd.concat([tr, te], keys=["tr", "te"]), columns=cats, dtype=float)
    Xtr_all, Xte = allX.loc["tr"], allX.loc["te"]
    # 15% of the training partition becomes the validation set; the official test set is untouched.
    Xtr, Xva, ytr, yva, ctr, cva = train_test_split(Xtr_all, ytr_all, ctr_all, test_size=0.15,
                                                    stratify=ctr_all, random_state=SEED)
    return _finish(Xtr, Xva, Xte, ytr, yva, yte, ctr, cva, cte, "UNSW-NB15", dict(stats=stats))


def load_nslkdd():
    """NSL-KDD: KDDTrain+ for training/validation, KDDTest+ (17 unseen attack types) for testing."""
    tr = pd.read_csv(f"{DATA}/NSL_KDD/KDDTrain+.txt", header=None, names=NSL_COLS)
    te = pd.read_csv(f"{DATA}/NSL_KDD/KDDTest+.txt", header=None, names=NSL_COLS)
    stats = dict(raw_train=len(tr), raw_test=len(te))
    seen = set(tr["attack"])  # attack types that appear in training
    tr = _clean(tr)
    stats.update(clean_train=len(tr), clean_test=len(te))
    for d in (tr, te):
        # difficulty is not a traffic feature; num_outbound_cmds is always zero
        d.drop(columns=["difficulty", "num_outbound_cmds"], inplace=True)
    # Test records whose attack type never occurs in training: used for the unseen-attack test.
    novel_te = (~te["attack"].isin(seen)).values
    ctr_all = tr["attack"].map(NSL_FAMILY).values
    cte = te["attack"].map(NSL_FAMILY).values
    ytr_all = (tr.pop("attack") != "normal").astype(int).values  # 1 = attack, 0 = normal
    yte = (te.pop("attack") != "normal").astype(int).values
    # One-hot encode the three text columns, train and test together for identical columns.
    allX = pd.get_dummies(pd.concat([tr, te], keys=["tr", "te"]),
                          columns=["protocol_type", "service", "flag"], dtype=float)
    Xtr_all, Xte = allX.loc["tr"], allX.loc["te"]
    Xtr, Xva, ytr, yva, ctr, cva = train_test_split(Xtr_all, ytr_all, ctr_all, test_size=0.15,
                                                    stratify=ctr_all, random_state=SEED)
    return _finish(Xtr, Xva, Xte, ytr, yva, yte, ctr, cva, cte, "NSL-KDD",
                   dict(stats=stats, novel_test=novel_te))


# Name -> loader function, used by every experiment script and by the tests.
LOADERS = {"CIC-IDS2017": load_cicids2017, "UNSW-NB15": load_unsw, "NSL-KDD": load_nslkdd}


if __name__ == "__main__":
    # Running this file on its own prints a summary of each prepared dataset.
    for name, fn in LOADERS.items():
        d = fn()
        print(f"== {name}: features={d['X_train'].shape[1]}  train={len(d['y_train'])} "
              f"val={len(d['y_val'])} test={len(d['y_test'])}  attack%train={d['y_train'].mean():.3f} "
              f"attack%test={d['y_test'].mean():.3f}")
        print("   stats:", d["stats"])
        print("   test families:", pd.Series(d["cat_test"]).value_counts().to_dict())
        if "novel_test" in d:
            print("   novel attack records in test:", int(d["novel_test"].sum()))
