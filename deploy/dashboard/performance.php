<?php
// Model performance: shows the held-out test results of all six models on each dataset,
// read from the same result files used in the paper (deploy/dashboard/data/*.json).
require 'inc/db.php'; require 'inc/layout.php';
// Friendly display names for the models.
$names = ['NaiveBayes' => 'Naive Bayes', 'RandomForest' => 'Random Forest', 'XGBoost' => 'XGBoost (deployed)', 'LightGBM' => 'LightGBM', 'MLP' => 'MLP', 'IsolationForest' => 'Isolation Forest'];
page_start('Model performance', 'performance.php'); ?>
<div class="card full"><h1>How well do the models detect attacks?</h1><p class="hint">Held-out test results. F1 balances missed attacks against false alarms; the false-alarm rate is the share of benign traffic wrongly flagged.</p></div>
<?php foreach (['CIC-IDS2017', 'UNSW-NB15', 'NSL-KDD'] as $ds): $m = json_decode(file_get_contents(__DIR__ . "/data/{$ds}_main.json"), true)['models']; ?>
<div class="card full"><h3><?= $ds ?></h3><table><tr><th>Model</th><th>F1</th><th>Precision</th><th>Recall</th><th>False-alarm rate</th><th>ROC-AUC</th></tr>
<?php foreach ($names as $k => $label): $v = $m[$k]; ?><tr><td><?= $label ?></td><td><?= number_format($v['f1'], 3) ?></td><td><?= number_format($v['precision'], 3) ?></td>
<td><?= number_format($v['recall'], 3) ?></td><td><?= number_format(100 * $v['fpr'], 2) ?>%</td><td><?= number_format($v['roc_auc'], 3) ?></td></tr><?php endforeach; ?></table></div>
<?php endforeach; page_end();
