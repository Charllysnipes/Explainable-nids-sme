<?php
require 'inc/db.php'; require 'inc/layout.php';
$runs = db()->query("SELECT * FROM analysis_run ORDER BY id DESC LIMIT 20")->fetchAll(PDO::FETCH_ASSOC);
page_start('Past analyses', 'history.php'); ?>
<div class="card full"><h1>Past analyses</h1>
<?php if (!$runs): ?><p class="hint">No files have been analysed yet. Go to <a href="analyse.php">Analyse traffic</a> to start.</p>
<?php else: ?><table><tr><th>Run</th><th>Date</th><th>File</th><th>Flows</th><th>Alerts</th><th>Invalid rows</th><th>Model</th></tr>
<?php foreach ($runs as $r): ?><tr><td>#<?= $r['id'] ?></td><td><?= h($r['created_at']) ?></td><td><?= h($r['filename']) ?></td><td><?= $r['n_flows'] ?></td>
<td><span class="pill <?= $r['n_alerts'] ? 'alert' : 'ok' ?>"><?= $r['n_alerts'] ?></span></td><td><?= $r['n_invalid'] ?></td><td><?= h($r['model']) ?></td></tr><?php endforeach; ?>
</table><?php endif; ?></div>
<?php page_end();
