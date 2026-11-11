"""Unit and integration tests for the data pipeline, models and REST API (run: python3 -m unittest -v).

Each test carries its ID from Table 5 of the supporting material (DP = data pipeline,
ML = models, API = core engine). The browser tests UI-1 to UI-7 are in test_ui.py.
"""
import os, sys, unittest, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, so the code runs from any folder
import numpy as np
sys.path.insert(0, ROOT)
from data_prep import LOADERS
from models import make_model, sample_params, SPACES, MODEL_ORDER

# Identifier columns (IP addresses, ports, flow IDs, timestamps) that must never reach a model.
ID_COLS = {"Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp", "id", "srcip", "dstip",
           "sport", "dsport", "stime", "ltime"}
_cache = {}


def ds(name):
    """Load a prepared dataset once and reuse it in every test (loading is the slow part)."""
    if name not in _cache:
        _cache[name] = LOADERS[name]()
    return _cache[name]


class TestDataPipeline(unittest.TestCase):
    """Validity tests: the data must be clean and must not leak between training and test."""

    def test_no_identifier_columns(self):
        """DP-1: IPs, ports, IDs and timestamps are removed, so models learn behaviour, not hosts."""
        for n in LOADERS:
            self.assertFalse(ID_COLS & set(ds(n)["X_train"].columns), n)

    def test_no_missing_or_infinite_values(self):
        """DP-2: no NaN or infinite value is left in any split."""
        for n in LOADERS:
            for part in ("X_train", "X_val", "X_test"):
                self.assertTrue(np.isfinite(ds(n)[part].to_numpy()).all(), f"{n} {part}")

    def test_scaler_fitted_on_training_only(self):
        """DP-3: Min-Max scaling was fitted on training data, so training values lie exactly in [0, 1]."""
        for n in LOADERS:
            X = ds(n)["X_train"].to_numpy()
            self.assertAlmostEqual(float(X.min()), 0.0, places=5)
            self.assertLessEqual(float(X.max()), 1.0 + 1e-5)

    def test_same_feature_columns_in_all_splits(self):
        """DP-4: training, validation and test splits share one schema, in the same column order."""
        for n in LOADERS:
            d = ds(n)
            self.assertEqual(list(d["X_train"].columns), list(d["X_test"].columns))
            self.assertEqual(list(d["X_train"].columns), list(d["X_val"].columns))

    def test_no_constant_features(self):
        """DP-5: columns with a single value were removed."""
        for n in LOADERS:
            self.assertTrue((ds(n)["X_train"].nunique() > 1).all(), n)

    def test_stratified_split_preserves_attack_rate(self):
        """DP-6: stratified splitting keeps the share of attacks the same in training and test (within 1%)."""
        d = ds("CIC-IDS2017")
        self.assertAlmostEqual(d["y_train"].mean(), d["y_test"].mean(), delta=0.01)

    def test_no_duplicate_rows_in_training(self):
        """DP-7: duplicates were removed before splitting.

        Duplicates are removed on features plus label, so a few rows may still share the same
        features with a different label or family; at most 2% such rows are allowed.
        """
        for n in LOADERS:
            self.assertEqual(int(ds(n)["X_train"].duplicated().sum()) <= len(ds(n)["y_train"]) * 0.02, True, n)

    def test_nslkdd_unseen_attack_records(self):
        """ML-1: the NSL-KDD test set holds 3,750 records of attack types never seen in training."""
        self.assertEqual(int(ds("NSL-KDD")["novel_test"].sum()), 3750)

    def test_unsw_official_test_partition_size(self):
        """ML-2: the official UNSW-NB15 test partition (82,332 records) is left intact."""
        self.assertEqual(len(ds("UNSW-NB15")["y_test"]), 82332)


class TestModels(unittest.TestCase):
    """Reliability tests for the model code and the saved results."""

    def test_all_models_build(self):
        """ML-3: all six model factories construct a model."""
        for m in MODEL_ORDER:
            self.assertIsNotNone(make_model(m, {}))

    def test_search_space_samples_are_unique(self):
        """ML-4: the random search never tries the same hyperparameter setting twice."""
        rng = np.random.default_rng(0)
        for m in MODEL_ORDER:
            ps = sample_params(SPACES[m], 6, rng)
            self.assertEqual(len({repr(sorted(p.items())) for p in ps}), len(ps))

    def test_saved_models_meet_minimum_quality(self):
        """ML-5: regression check, XGBoost F1 on CIC-IDS2017 must stay above 0.95."""
        res = json.load(open(os.path.join(ROOT, "results/CIC-IDS2017_main.json")))
        self.assertGreater(res["models"]["XGBoost"]["f1"], 0.95)


class TestApi(unittest.TestCase):
    """Functionality and integration tests of the Flask core engine, called through its test client."""

    @classmethod
    def setUpClass(cls):
        # Load the API module once (this loads the saved CIC-IDS2017 XGBoost model).
        os.environ["NIDS_DATASET"] = "CIC-IDS2017"
        sys.path.insert(0, os.path.join(ROOT, "deploy"))
        import api
        cls.api = api
        cls.client = api.app.test_client()  # sends requests without starting a real server
        cls.d = ds("CIC-IDS2017")

    def _flow(self, i):
        """Test-set flow number i as a JSON-ready dict of scaled features."""
        return {k: float(v) for k, v in self.d["X_test"].iloc[i].items()}

    def test_health(self):
        """API-1: the health endpoint responds with status ok."""
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_predict_schema(self):
        """API-2: each result has a probability in [0, 1], a true/false alert and exactly 3 reasons."""
        r = self.client.post("/predict", json={"flows": [self._flow(0), self._flow(1)]}).get_json()
        self.assertEqual(len(r["results"]), 2)
        for x in r["results"]:
            self.assertTrue(0.0 <= x["attack_probability"] <= 1.0)
            self.assertIsInstance(x["alert"], bool)
            self.assertEqual(len(x["top_features"]), 3)

    def test_missing_features_are_filled(self):
        """API-3: a flow with missing fields is still scored instead of crashing the service."""
        r = self.client.post("/predict", json={"flows": [{}]}).get_json()
        self.assertEqual(len(r["results"]), 1)

    def test_api_matches_offline_model(self):
        """API-4: the deployed API gives the same probabilities as the offline model (50 flows)."""
        b = self.api.booster
        import xgboost as xgb, pandas as pd
        X = self.d["X_test"].iloc[:50]
        offline = b.predict(xgb.DMatrix(X))
        online = [self.client.post("/predict", json={"flows": [self._flow(i)]}).get_json()["results"][0]["attack_probability"]
                  for i in range(50)]
        self.assertTrue(np.allclose(offline, online, atol=1e-5))

    def test_raw_input_matches_offline_pipeline(self):
        """API-5: raw CSV rows through the API equal offline pre-processing plus the model."""
        import pandas as pd
        raw = pd.read_csv(os.path.join(ROOT, "deploy/dashboard/sample_flows.csv")).head(40)
        out = self.client.post("/predict_raw", json={"flows": raw.to_dict(orient="records")}).get_json()
        # Offline path: same saved scaler, same clipping, same model.
        pre = self.api.pre
        X = np.clip(pre["scaler"].transform(raw[pre["columns"]]), -1, 2)
        import xgboost as xgb
        offline = self.api.booster.predict(xgb.DMatrix(pd.DataFrame(X, columns=pre["columns"]).astype(np.float32)))
        self.assertTrue(np.allclose(offline, [r["attack_probability"] for r in out["results"]], atol=1e-5))

    def test_raw_input_missing_columns_rejected(self):
        """API-6: missing columns give a clear HTTP 400 error that says columns are missing."""
        r = self.client.post("/predict_raw", json={"flows": [{"Flow Duration": 1}]})
        self.assertEqual(r.status_code, 400)
        self.assertIn("missing", r.get_json()["error"])

    def test_raw_input_too_many_flows_rejected(self):
        """API-7: a request above the 5,000-flow limit is rejected with HTTP 413."""
        r = self.client.post("/predict_raw", json={"flows": [{}] * 5001})
        self.assertEqual(r.status_code, 413)

    def test_raw_input_invalid_values_flagged(self):
        """API-8: a non-numeric value is flagged on its own row instead of crashing the service."""
        import pandas as pd
        raw = pd.read_csv(os.path.join(ROOT, "deploy/dashboard/sample_flows.csv")).head(2)
        rec = raw.to_dict(orient="records"); rec[0]["Flow Duration"] = "abc"  # break one value on purpose
        out = self.client.post("/predict_raw", json={"flows": rec}).get_json()
        self.assertEqual(out["n_invalid_rows"], 1)
        self.assertTrue(out["results"][0]["invalid_values"])

    def test_shap_values_are_additive(self):
        """ML-6: SHAP contributions add up to the model output (log-odds) for 200 flows."""
        import xgboost as xgb
        X = self.d["X_test"].iloc[:200]
        dm = xgb.DMatrix(X)
        c = self.api.booster.predict(dm, pred_contribs=True)
        m = self.api.booster.predict(dm, output_margin=True)
        self.assertTrue(np.allclose(c.sum(1), m, atol=1e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
