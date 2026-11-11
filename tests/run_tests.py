"""Runs the unit/integration test suite and records results for the supporting material."""
import json, sys, unittest, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, so the code runs from any folder
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.environ.setdefault("NIDS_DATASET", "CIC-IDS2017")
import test_pipeline as T
import test_ui as U

PURPOSE = {
    "test_no_identifier_columns": "IPs, ports, IDs and timestamps removed",
    "test_no_missing_or_infinite_values": "No NaN/Inf in any split",
    "test_scaler_fitted_on_training_only": "Min-Max fitted on training data only",
    "test_same_feature_columns_in_all_splits": "Train/val/test share one schema",
    "test_no_constant_features": "Constant columns removed",
    "test_stratified_split_preserves_attack_rate": "Stratification keeps class balance",
    "test_no_duplicate_rows_in_training": "Duplicates removed before splitting",
    "test_nslkdd_unseen_attack_records": "3,750 unseen-attack test records identified",
    "test_unsw_official_test_partition_size": "Official UNSW-NB15 test partition intact",
    "test_all_models_build": "All six model factories construct",
    "test_search_space_samples_are_unique": "Randomised search has no repeats",
    "test_saved_models_meet_minimum_quality": "Regression check: XGBoost F1 > 0.95 (CIC)",
    "test_health": "API health endpoint responds",
    "test_predict_schema": "API returns probability, alert flag, 3 reasons",
    "test_missing_features_are_filled": "API tolerates missing fields",
    "test_api_matches_offline_model": "Online and offline predictions agree",
    "test_shap_values_are_additive": "SHAP values sum to model output",
    "test_raw_input_matches_offline_pipeline": "Raw CSV path equals offline pre-processing + model",
    "test_raw_input_missing_columns_rejected": "Missing columns give a clear 400 error",
    "test_raw_input_too_many_flows_rejected": "Oversized requests are rejected (413)",
    "test_raw_input_invalid_values_flagged": "Non-numeric values are flagged per row",
    "test_ui_navigation_present_on_every_page": "All five navigation links on every page",
    "test_ui_sample_upload_shows_results_and_saves_run": "Upload -> results table -> run stored; < 10 s",
    "test_ui_no_file_error": "Clear message when no file is chosen",
    "test_ui_wrong_file_type_error": "Clear message for a non-CSV file",
    "test_ui_missing_columns_error_names_columns": "Missing columns named; nothing stored",
    "test_ui_history_lists_stored_runs": "History page lists stored runs",
    "test_ui_output_is_html_escaped": "HTML in input shown as text (XSS check)",
}


class Rec(unittest.TextTestResult):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); self.rows = []
    def addSuccess(self, t): super().addSuccess(t); self.rows.append((t, "Pass"))
    def addFailure(self, t, e): super().addFailure(t, e); self.rows.append((t, "Fail"))
    def addError(self, t, e): super().addError(t, e); self.rows.append((t, "Error"))
    def addSkip(self, t, r): super().addSkip(t, r); self.rows.append((t, "Skipped"))


suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(T),
                            unittest.defaultTestLoader.loadTestsFromModule(U)])
runner = unittest.TextTestRunner(verbosity=2, resultclass=Rec)
res = runner.run(suite)
cases = [dict(name=getattr(t, "_testMethodName", str(t)), purpose=PURPOSE.get(getattr(t, "_testMethodName", ""), ""), result=r) for t, r in res.rows]
out = dict(total=res.testsRun, passed=sum(c["result"] == "Pass" for c in cases), cases=cases)
json.dump(out, open(os.path.join(ROOT, "results/unit_tests.json"), "w"), indent=1)
print("TESTS", out["passed"], "/", out["total"])
