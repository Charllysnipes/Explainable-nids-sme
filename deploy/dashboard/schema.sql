-- Back-end schema (MySQL 8). The prototype creates the same tables in SQLite automatically.
CREATE DATABASE IF NOT EXISTS nids CHARACTER SET utf8mb4;
USE nids;
CREATE TABLE IF NOT EXISTS analysis_run (
  id INT AUTO_INCREMENT PRIMARY KEY,
  created_at DATETIME NOT NULL,
  filename VARCHAR(255) NOT NULL,
  n_flows INT NOT NULL,
  n_alerts INT NOT NULL,
  n_invalid INT NOT NULL,
  model VARCHAR(64) NOT NULL,
  server_ms DOUBLE
);
CREATE TABLE IF NOT EXISTS prediction (
  id INT AUTO_INCREMENT PRIMARY KEY,
  run_id INT NOT NULL,
  row_no INT NOT NULL,
  probability DOUBLE NOT NULL,
  alert TINYINT(1) NOT NULL,
  reason1 VARCHAR(64), shap1 DOUBLE,
  reason2 VARCHAR(64), shap2 DOUBLE,
  reason3 VARCHAR(64), shap3 DOUBLE,
  true_label VARCHAR(32),
  FOREIGN KEY (run_id) REFERENCES analysis_run(id) ON DELETE CASCADE,
  INDEX idx_run_alert (run_id, alert)
);
CREATE TABLE IF NOT EXISTS flows (            -- live stream written by the replay client
  id INT PRIMARY KEY, ts DOUBLE, true_label VARCHAR(32), prob DOUBLE, alert TINYINT(1),
  f1 VARCHAR(64), v1 DOUBLE, f2 VARCHAR(64), v2 DOUBLE, f3 VARCHAR(64), v3 DOUBLE, latency_ms DOUBLE
);
