"""Model definitions and hyperparameter search spaces."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

SEED = 42
N_JOBS = 2


def make_model(name, params):
    if name == "NaiveBayes":
        return GaussianNB(**params)
    if name == "RandomForest":
        return RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, class_weight="balanced", **params)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
                                 eval_metric="logloss", **params)
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_jobs=N_JOBS, random_state=SEED, verbose=-1,
                                  class_weight="balanced", **params)
    if name == "MLP":
        return MLPClassifier(random_state=SEED, early_stopping=True, max_iter=200,
                             n_iter_no_change=8, **params)
    if name == "IsolationForest":
        return IsolationForest(random_state=SEED, n_jobs=N_JOBS, **params)
    raise ValueError(name)


SPACES = {
    "NaiveBayes": {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1]},
    "RandomForest": {"n_estimators": [100, 200, 300], "max_depth": [None, 20, 30],
                     "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.3]},
    "XGBoost": {"n_estimators": [200, 400], "max_depth": [6, 8, 10], "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0]},
    "LightGBM": {"n_estimators": [200, 400], "num_leaves": [31, 63, 127], "learning_rate": [0.05, 0.1],
                 "subsample": [0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.6, 0.8, 1.0]},
    "MLP": {"hidden_layer_sizes": [(128, 64), (256, 128), (64, 32)], "alpha": [1e-5, 1e-4, 1e-3],
            "learning_rate_init": [1e-3, 3e-3], "batch_size": [256]},
    "IsolationForest": {"n_estimators": [100, 200], "max_samples": [256, 1024, 4096],
                        "max_features": [0.5, 1.0]},
}

MODEL_ORDER = ["NaiveBayes", "RandomForest", "XGBoost", "LightGBM", "MLP", "IsolationForest"]


def sample_params(space, n, rng):
    keys = sorted(space)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < 200:
        tries += 1
        p = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        key = repr(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def scores(model, name, X):
    """Return attack score (higher = more likely attack)."""
    if name == "IsolationForest":
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]
