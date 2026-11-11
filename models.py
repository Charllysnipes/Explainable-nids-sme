"""Model definitions and hyperparameter search spaces.

Six models from five families are compared under the same procedure:
Naive Bayes (probabilistic baseline), Random Forest (bagging), XGBoost and LightGBM
(boosting), MLP (neural network) and Isolation Forest (unsupervised, benign data only).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

SEED = 42   # same seed everywhere so results can be repeated
N_JOBS = 2  # the experiments ran on a 2-core CPU, no GPU


def make_model(name, params):
    """Build one untrained model by name, with the hyperparameters chosen by the search."""
    if name == "NaiveBayes":
        # Simple baseline: assumes every feature is independent and normally distributed.
        return GaussianNB(**params)
    if name == "RandomForest":
        # Many decision trees on random samples, averaged. class_weight balances attack and benign.
        return RandomForestClassifier(n_jobs=N_JOBS, random_state=SEED, class_weight="balanced", **params)
    if name == "XGBoost":
        # Boosted trees: each new tree corrects the errors of the trees before it. "hist" = fast binned splits.
        import xgboost as xgb
        return xgb.XGBClassifier(tree_method="hist", n_jobs=N_JOBS, random_state=SEED,
                                 eval_metric="logloss", **params)
    if name == "LightGBM":
        # Boosted trees that grow the most useful leaf first: very fast at prediction time.
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_jobs=N_JOBS, random_state=SEED, verbose=-1,
                                  class_weight="balanced", **params)
    if name == "MLP":
        # Small neural network. early_stopping ends training when the hold-out score stops improving.
        return MLPClassifier(random_state=SEED, early_stopping=True, max_iter=200,
                             n_iter_no_change=8, **params)
    if name == "IsolationForest":
        # Unsupervised: learns normal traffic only and flags flows that are easy to isolate (unusual).
        return IsolationForest(random_state=SEED, n_jobs=N_JOBS, **params)
    raise ValueError(name)


# Values the random search may pick from, for each model (Section III-E of the paper).
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

# Order in which models are trained and reported in every table.
MODEL_ORDER = ["NaiveBayes", "RandomForest", "XGBoost", "LightGBM", "MLP", "IsolationForest"]


def sample_params(space, n, rng):
    """Randomised search: draw n different hyperparameter settings from a search space."""
    keys = sorted(space)
    seen, out = set(), []
    tries = 0
    while len(out) < n and tries < 200:  # stop after 200 draws in case the space is very small
        tries += 1
        p = {k: space[k][rng.integers(len(space[k]))] for k in keys}  # pick one value per setting
        key = repr(sorted(p.items()))
        if key not in seen:  # never test the same combination twice
            seen.add(key)
            out.append(p)
    return out


def scores(model, name, X):
    """Return attack score (higher = more likely attack)."""
    if name == "IsolationForest":
        # score_samples is high for normal points, so it is negated to make "high = attack".
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]  # probability of class 1 (attack)
