<?php
// Live monitor: reads the scored flows written by the replay client (simulated live stream).
require 'inc/db.php'; require 'inc/layout.php';
$db = db();
try { $db->query("SELECT 1 FROM flows LIMIT 1"); $hasFlows = true; } catch (PDOException $e) { $hasFlows = false; }
page_start('Live monitor', 'index.php');
if (!$hasFlows) { echo '<div class="card full msg error">No live data yet. Start the replay client (deploy/replay.py) to stream flows to the engine.</div>'; page_end(); exit; }
$k = $db->query("SELECT COUNT(*) n, SUM(alert) a, SUM(alert AND true_label!='Benign') tp, AVG(latency_ms) lat FROM flows")->fetch(PDO::FETCH_ASSOC);
$recent = $db->query("SELECT * FROM flows WHERE alert=1 ORDER BY id DESC LIMIT 5")->fetchAll(PDO::FETCH_ASSOC);
$series = array_reverse($db->query("SELECT id, prob, alert FROM flows ORDER BY id DESC LIMIT 300")->fetchAll(PDO::FETCH_ASSOC));
$top = $db->query("SELECT f1 f, COUNT(*) c FROM flows WHERE alert=1 GROUP BY f1 ORDER BY c DESC LIMIT 5")->fetchAll(PDO::FETCH_ASSOC);
$prec = $k['a'] ? 100 * $k['tp'] / $k['a'] : 0;
?>
<div class="card kpi"><div class="v"><?= number_format($k['n']) ?></div><div class="l">Flows analysed</div></div>
<div class="card kpi"><div class="v"><?= number_format($k['a']) ?></div><div class="l">Alerts raised</div></div>
<div class="card kpi"><div class="v"><?= number_format($prec, 1) ?>%</div><div class="l">Alerts confirmed malicious</div></div>
<div class="card kpi"><div class="v"><?= number_format($k['lat'], 1) ?> ms</div><div class="l">Mean API latency per flow</div></div>
<div class="card wide"><h3>Attack probability (last 300 flows) <span class="hint">red = alert, blue = benign</span></h3>
 <svg viewBox="0 0 600 140" width="100%" height="140" role="img" aria-label="Attack probability of recent flows">
  <line x1="0" y1="70" x2="600" y2="70" stroke="#c9d1db" stroke-dasharray="4 3"/>
  <?php foreach ($series as $i => $r): $hgt = 130 * $r['prob']; ?>
   <rect x="<?= $i * 2 ?>" y="<?= 135 - $hgt ?>" width="1.6" height="<?= $hgt ?>" fill="<?= $r['alert'] ? '#c0392b' : '#2e86c1' ?>"/>
  <?php endforeach; ?>
 </svg></div>
<div class="card"><h3>Top alert drivers (SHAP)</h3><table>
 <?php foreach ($top as $t): ?><tr><td><?= h($t['f']) ?></td><td><?= $t['c'] ?></td></tr><?php endforeach; ?>
</table></div>
<div class="card full"><h3>Recent alerts with explanations</h3><table>
 <tr><th>Flow</th><th>Probability</th><th>Ground truth</th><th>Main reasons (SHAP contribution)</th></tr>
 <?php foreach ($recent as $r): ?>
 <tr><td>#<?= $r['id'] ?></td><td><span class="pill alert"><?= number_format($r['prob'], 3) ?></span></td><td><?= h($r['true_label']) ?></td>
  <td><?= h($r['f1']) ?> (<?= sprintf('%+.2f', $r['v1']) ?>), <?= h($r['f2']) ?> (<?= sprintf('%+.2f', $r['v2']) ?>), <?= h($r['f3']) ?> (<?= sprintf('%+.2f', $r['v3']) ?>)</td></tr>
 <?php endforeach; ?>
</table></div>
<?php page_end();
