<?php
function page_start(string $title, string $active) {
    $nav = ['index.php' => 'Live monitor', 'analyse.php' => 'Analyse traffic', 'history.php' => 'Past analyses',
            'performance.php' => 'Model performance', 'help.php' => 'Help'];
    echo '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">';
    echo '<title>' . h($title) . ' – NIDS</title><link rel="stylesheet" href="style.css"></head><body>';
    echo '<header><div class="brand">Explainable Network Anomaly Detection <span>SME edition</span></div><nav>';
    foreach ($nav as $href => $label) {
        echo '<a href="' . $href . '"' . ($href === $active ? ' class="active" aria-current="page"' : '') . '>' . $label . '</a>';
    }
    echo '</nav></header><main>';
}
function page_end() { echo '</main><footer>COM748 MSc Project · Charles Chinemerem Ezeh · Model: XGBoost (CIC-IDS2017) with TreeSHAP explanations</footer></body></html>'; }
