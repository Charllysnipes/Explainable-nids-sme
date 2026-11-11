"""Replays held-out test flows to the REST API as a simulated live stream, stores results in
SQLite for the PHP dashboard, and measures end-to-end latency and throughput."""
import json, sqlite3, time, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root, so the code runs from any folder
import numpy as np
import requests
sys.path.insert(0, ROOT)
from data_prep import LOADERS

DS = os.environ.get("NIDS_DATASET", "CIC-IDS2017")
URL = "http://127.0.0.1:5005/predict"                 # the running core engine (deploy/api.py)
DB = os.path.join(ROOT, "deploy/dashboard/alerts.db")  # same database the dashboard reads
N_STREAM = 1500                                        # number of flows sent one by one
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

# Pick 1,500 random flows from the held-out test set (fixed seed, so the same flows every time).
d = LOADERS[DS]()
X, y, cat = d["X_test"], d["y_test"], d["cat_test"]
rng = np.random.default_rng(3)
idx = rng.choice(len(y), N_STREAM, replace=False)

# Fresh "flows" table for the Live monitor page.
con = sqlite3.connect(DB)
con.execute("DROP TABLE IF EXISTS flows")
con.execute("""CREATE TABLE flows (id INTEGER PRIMARY KEY, ts REAL, true_label TEXT, prob REAL,
               alert INTEGER, f1 TEXT, v1 REAL, f2 TEXT, v2 REAL, f3 TEXT, v3 REAL, latency_ms REAL)""")
s = requests.Session()
s.trust_env = False  # call the local API directly, never through a proxy
lat = []
t_start = time.time()
# Send each flow on its own, as live traffic would arrive, and time the full round trip.
for n, i in enumerate(idx):
    flow = {k: float(v) for k, v in X.iloc[i].items()}
    t = time.perf_counter()
    r = s.post(URL, json={"flows": [flow]}).json()["results"][0]
    ms = (time.perf_counter() - t) * 1000
    lat.append(ms)
    (f1, v1), (f2, v2), (f3, v3) = r["top_features"]  # the three SHAP reasons
    con.execute("INSERT INTO flows VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (n, t_start + n, str(cat[i]), r["attack_probability"], int(r["alert"]), f1, v1, f2, v2, f3, v3, ms))
con.commit()
# batch throughput through the API (1000 flows per request)
B = [{k: float(v) for k, v in X.iloc[i].items()} for i in rng.choice(len(y), 1000, replace=False)]
ts = []
for _ in range(5):
    t = time.perf_counter(); s.post(URL, json={"flows": B}); ts.append(time.perf_counter() - t)
# How many alerts were raised, and how many of them were real attacks.
alerts = con.execute("SELECT COUNT(*), SUM(true_label!='Benign') FROM flows WHERE alert=1").fetchone()
res = dict(dataset=DS, n_stream=N_STREAM, api_latency_ms_median=float(np.median(lat)),
           api_latency_ms_p95=float(np.percentile(lat, 95)), api_batch1000_seconds=float(np.median(ts)),
           api_batch_throughput_flows_per_s=float(1000 / np.median(ts)),
           alerts_raised=int(alerts[0]), alerts_true=int(alerts[1] or 0))
# Note: this overwrites results/api_latency.json with the timings measured on this computer.
json.dump(res, open(os.path.join(ROOT, "results/api_latency.json"), "w"), indent=1)
print(res)
