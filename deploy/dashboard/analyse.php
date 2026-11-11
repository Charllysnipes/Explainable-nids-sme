<?php
// Front end: upload a CSV of raw CICFlowMeter flows; the core engine pre-processes, scores and explains them;
// the run and its predictions are stored in the back-end database.
require 'inc/db.php'; require 'inc/layout.php';
const MAX_BYTES = 5 * 1024 * 1024; const MAX_ROWS = 5000;
$error = null; $result = null; $runId = null; $fname = null; $labels = [];

function read_csv_rows(string $path, ?string &$err): ?array {
    $fh = fopen($path, 'r');
    $head = fgetcsv($fh, 0, ',', '"', '');
    if (!$head || count($head) < 2) { $err = 'The file does not look like a CSV with a header row.'; return null; }
    $head = array_map('trim', $head); $rows = []; $n = 0;
    while (($r = fgetcsv($fh, 0, ',', '"', '')) !== false) {
        if ($r === [null]) continue;
        if (++$n > MAX_ROWS) { $err = 'The file has more than ' . MAX_ROWS . ' flows. Please split it into smaller files.'; return null; }
        if (count($r) !== count($head)) { $err = "Row $n has " . count($r) . ' values but the header has ' . count($head) . ' columns.'; return null; }
        $rows[] = array_combine($head, $r);
    }
    if (!$rows) { $err = 'The file has a header but no flows.'; return null; }
    return $rows;
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['use_sample'])) { $path = __DIR__ . '/sample_flows.csv'; $fname = 'sample_flows.csv'; }
    elseif (!isset($_FILES['flows']) || $_FILES['flows']['error'] === UPLOAD_ERR_NO_FILE) { $error = 'Please choose a CSV file to upload, or use the sample file.'; }
    elseif ($_FILES['flows']['error'] !== UPLOAD_ERR_OK) { $error = 'The upload failed. Please try again.'; }
    elseif ($_FILES['flows']['size'] > MAX_BYTES) { $error = 'The file is larger than 5 MB.'; }
    elseif (strtolower(pathinfo($_FILES['flows']['name'], PATHINFO_EXTENSION)) !== 'csv') { $error = 'Only .csv files exported from CICFlowMeter are accepted.'; }
    else { $path = $_FILES['flows']['tmp_name']; $fname = basename($_FILES['flows']['name']); }
    if (!$error) {
        $rows = read_csv_rows($path, $error);
        if ($rows) {
            foreach ($rows as $r) { $labels[] = $r['Label'] ?? null; }
            $result = api_post('/predict_raw', ['flows' => $rows], $error);
        }
    }
    if ($result) {
        $db = db(); $res = $result['results']; $nAlert = count(array_filter($res, fn($r) => $r['alert']));
        $db->prepare("INSERT INTO analysis_run (created_at, filename, n_flows, n_alerts, n_invalid, model, server_ms) VALUES (?,?,?,?,?,?,?)")
           ->execute([date('Y-m-d H:i:s'), $fname, count($res), $nAlert, $result['n_invalid_rows'], 'XGBoost/CIC-IDS2017', $result['server_ms']]);
        $runId = $db->lastInsertId();
        $st = $db->prepare("INSERT INTO prediction (run_id, row_no, probability, alert, reason1, shap1, reason2, shap2, reason3, shap3, true_label) VALUES (?,?,?,?,?,?,?,?,?,?,?)");
        $db->beginTransaction();
        foreach ($res as $i => $r) { $t = $r['top_features'];
            $st->execute([$runId, $r['row'], $r['attack_probability'], (int)$r['alert'], $t[0][0], $t[0][1], $t[1][0], $t[1][1], $t[2][0], $t[2][1], $labels[$i] ?? null]); }
        $db->commit();
    }
}
page_start('Analyse traffic', 'analyse.php'); ?>
<div class="card full"><h1>Analyse a traffic file</h1>
 <p class="hint">Upload flows exported by CICFlowMeter (.csv, up to 5 MB and <?= MAX_ROWS ?> flows). Each flow is scored by the XGBoost model and the three features that most influenced the decision are shown.</p>
 <form method="post" enctype="multipart/form-data" style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-top:8px">
  <label for="flows">CSV file:</label><input id="flows" type="file" name="flows" accept=".csv">
  <button type="submit">Analyse file</button>
  <button type="submit" name="use_sample" value="1" class="btn secondary">Use sample file (300 flows)</button>
 </form></div>
<?php if ($error): ?><div class="card full msg error" role="alert"><strong>We could not analyse this file.</strong> <?= h($error) ?></div><?php endif; ?>
<?php if ($result): $res = $result['results']; $nA = count(array_filter($res, fn($r) => $r['alert'])); ?>
 <div class="card full msg success" role="status">Analysis complete for <strong><?= h($fname) ?></strong> (run #<?= $runId ?>, saved to the database).</div>
 <div class="card kpi"><div class="v"><?= count($res) ?></div><div class="l">Flows analysed</div></div>
 <div class="card kpi"><div class="v"><?= $nA ?></div><div class="l">Suspicious flows (alerts)</div></div>
 <div class="card kpi"><div class="v"><?= $result['n_invalid_rows'] ?></div><div class="l">Rows with invalid values</div></div>
 <div class="card kpi"><div class="v"><?= number_format($result['server_ms'] / max(1, count($res)), 2) ?> ms</div><div class="l">Engine time per flow</div></div>
 <div class="card full"><h3>Alerts first (showing up to 25 flows)</h3><table>
  <tr><th>Row</th><th>Verdict</th><th>Probability</th><?php if (array_filter($labels)) echo '<th>Label in file</th>'; ?><th>Why (top SHAP contributions)</th></tr>
  <?php $order = array_keys($res); usort($order, fn($a, $b) => $res[$b]['attack_probability'] <=> $res[$a]['attack_probability']);
  foreach (array_slice($order, 0, 25) as $i): $r = $res[$i]; ?>
  <tr><td><?= $r['row'] ?></td><td><span class="pill <?= $r['alert'] ? 'alert' : 'ok' ?>"><?= $r['alert'] ? 'Suspicious' : 'Benign' ?></span></td>
   <td><?= number_format($r['attack_probability'], 3) ?></td><?php if (array_filter($labels)) echo '<td>' . h($labels[$i]) . '</td>'; ?>
   <td><?= implode(', ', array_map(fn($t) => h($t[0]) . ' (' . sprintf('%+.2f', $t[1]) . ')', $r['top_features'])) ?></td></tr>
  <?php endforeach; ?></table>
  <p class="hint">Positive contributions push a flow towards "suspicious"; negative ones towards "benign".</p></div>
<?php endif; page_end();
