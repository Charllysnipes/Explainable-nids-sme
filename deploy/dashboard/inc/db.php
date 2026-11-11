<?php
// Back end: database access through PDO. The prototype uses SQLite; for MySQL set
//   NIDS_DSN="mysql:host=localhost;dbname=nids;charset=utf8mb4" NIDS_DB_USER=... NIDS_DB_PASS=...
// and load schema.sql once. No other code changes are needed.
date_default_timezone_set('Europe/London');
function db(): PDO {
    static $pdo = null;
    if ($pdo) return $pdo;
    $dsn = getenv('NIDS_DSN') ?: 'sqlite:' . __DIR__ . '/../alerts.db';
    $pdo = new PDO($dsn, getenv('NIDS_DB_USER') ?: null, getenv('NIDS_DB_PASS') ?: null);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
    if (str_starts_with($dsn, 'sqlite:')) {
        $pdo->exec("CREATE TABLE IF NOT EXISTS analysis_run (id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL, filename TEXT NOT NULL, n_flows INTEGER NOT NULL, n_alerts INTEGER NOT NULL,
            n_invalid INTEGER NOT NULL, model TEXT NOT NULL, server_ms REAL)");
        $pdo->exec("CREATE TABLE IF NOT EXISTS prediction (id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES analysis_run(id) ON DELETE CASCADE, row_no INTEGER NOT NULL,
            probability REAL NOT NULL, alert INTEGER NOT NULL, reason1 TEXT, shap1 REAL, reason2 TEXT, shap2 REAL,
            reason3 TEXT, shap3 REAL, true_label TEXT)");
    }
    return $pdo;
}
function h($s) { return htmlspecialchars((string)$s, ENT_QUOTES); }
const API = 'http://127.0.0.1:5005';
function api_post(string $path, array $payload, ?string &$err): ?array {
    $ch = curl_init(API . $path);
    curl_setopt_array($ch, [CURLOPT_POST => true, CURLOPT_RETURNTRANSFER => true, CURLOPT_TIMEOUT => 60,
        CURLOPT_HTTPHEADER => ['Content-Type: application/json'], CURLOPT_POSTFIELDS => json_encode($payload)]);
    $body = curl_exec($ch);
    if ($body === false) { $err = 'The analysis engine is not reachable. Please check that the API service is running.'; return null; }
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $data = json_decode($body, true);
    if ($code !== 200) { $err = $data['error'] ?? "The analysis engine returned an error (HTTP $code)."; return null; }
    return $data;
}
