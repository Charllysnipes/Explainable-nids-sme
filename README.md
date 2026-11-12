# Explainable-nids-sme

Explainable machine learning for real-time network anomaly detection in small and medium-sized enterprises (SMEs).

Code, results and prototype for the COM748 Masters Research Project *Explainable Machine Learning for Real-Time Network Anomaly Detection in Small and Medium-Sized Enterprises*, by Charles Chinemerem Ezeh (MSc Computer Science and Technology with Business Development, Ulster University, QAHE London). Supervisor: Anwar Haq.

**Demonstration video (3 minutes):** https://youtu.be/bPj0MSedLaw

## What the project does

Six models (Naive Bayes, Random Forest, XGBoost, LightGBM, MLP and Isolation Forest) are compared on CIC-IDS2017, UNSW-NB15 and NSL-KDD under one leakage-free pipeline. The study adds significance tests, unseen-attack, cross-network and noise tests, and TreeSHAP explanations with measured stability. The best model (XGBoost) is deployed behind a Flask REST API and a PHP dashboard that shows each alert with its probability and three SHAP reasons.

| Result (held-out test data) | Value |
|---|---|
| XGBoost F1 / false positive rate, CIC-IDS2017 | 0.998 / 0.18% |
| Friedman test over 15 CV blocks | χ² = 70.3, p = 8.8e-14 |
| Unseen NSL-KDD attack recall, tree ensembles | 0.34–0.41 (Isolation Forest 0.64) |
| Cross-network ROC-AUC | 0.42–0.50 |
| SHAP ranking stability (bootstrap Spearman ρ) | ≥ 0.9996 |
| API latency per flow (median, one CPU core) | 9.4 ms |
| Automated tests | 28 / 28 pass |

## Repository layout

| Path | Purpose |
|---|---|
| `data_prep.py` | Leakage-free loading and pre-processing of the three datasets |
| `models.py` | Model factory and hyperparameter search spaces |
| `run_main.py` | Tuning, training, test metrics, per-family detection, unseen-attack recall, noise robustness. Saves models and the API pre-processor |
| `run_cv_stats.py` | 5-fold CV, Friedman and Nemenyi, McNemar |
| `run_cross.py` | Cross-network test CIC-IDS2017 ↔ UNSW-NB15 on 7 harmonised features |
| `run_shap.py` | TreeSHAP global and local explanations and their stability |
| `run_latency.py` | Single-thread latency and batch throughput |
| `feature_selection.py`, `training_evidence.py`, `make_figures.py` | Feature selection study, training curves and all figures |
| `deploy/api.py` | Flask REST API (`/predict_raw`, `/predict`, `/health`, `/model_info`) |
| `deploy/replay.py` | Replays test flows to the API to feed the live monitor |
| `deploy/dashboard/` | PHP front end: live monitor, analyse (CSV upload), history, performance, help. `schema.sql` for MySQL |
| `tests/test_pipeline.py` | 21 data, model and API tests |
| `tests/test_ui.py` | 7 browser tests (Playwright, Chromium) |
| `tests/run_tests.py` | Runs all 28 tests and writes `results/unit_tests.json` |
| `results/` | Result files (JSON) behind every table and figure in the paper |
| `figs/` | Figures, UML diagrams and wireframes |
| `scripts/get_data.sh` | Downloads the datasets into `data/` |

## Datasets

The datasets are public and are not stored in this repository. `bash scripts/get_data.sh` downloads them from these sources:

* CIC-IDS2017 sampled subset (56,661 flows) published by Yang et al. (MTH-IDS): https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning. Original: https://www.unb.ca/cic/datasets/ids-2017.html
* UNSW-NB15 official training and testing partitions: https://research.unsw.edu.au/projects/unsw-nb15-dataset (copies used: https://github.com/InitRoot/UNSW_NB15 and https://github.com/comsyssec/datasets)
* NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html (copy used: https://github.com/defcom17/NSL_KDD)

To keep the data somewhere else, set the `NIDS_DATA` environment variable to that folder.

## How to run

Requirements: Python 3.10 or later, PHP 8 with `pdo_sqlite` and `curl`, and Git. Everything runs on a normal CPU without a GPU.

```bash
pip install -r requirements.txt
python3 -m playwright install chromium        # browser for the UI tests
bash scripts/get_data.sh                      # download the datasets

# experiments (each dataset takes a few minutes on two CPU cores)
for d in CIC-IDS2017 UNSW-NB15 NSL-KDD; do python3 run_main.py $d; done
python3 run_shap.py && python3 run_latency.py && python3 run_cross.py && python3 run_cv_stats.py
python3 make_figures.py

# all 28 automated tests (needs the CIC-IDS2017 models from run_main.py)
python3 tests/run_tests.py

# prototype
python3 deploy/api.py 5005                    # terminal 1: core engine
php -S 127.0.0.1:8080 -t deploy/dashboard     # terminal 2: open http://127.0.0.1:8080/analyse.php
python3 deploy/replay.py                      # optional: stream test flows to the live monitor
```

`deploy/run_demo.sh` runs the API, replay, dashboard screenshot and tests in one go.

## Reproducibility

The random seed is 42 throughout. The original experiments ran on 2 vCPUs with 7 GB RAM and no GPU. Library versions affect tree building slightly, so a rerun can differ from the paper in the fourth decimal place. For example, a rerun of XGBoost on CIC-IDS2017 with XGBoost 3.4.1 gave F1 0.9978 and 5 false alarms, against 0.9976 and 6 in the paper. The files in `results/` are the ones reported in the paper.

## Licence and data use

The code is released for academic assessment and research. The datasets belong to their creators and are used under their research terms. Please cite the original dataset papers when using them.
