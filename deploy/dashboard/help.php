<?php // Help page: plain-language guide to the tool, its numbers and its limits.
require 'inc/db.php'; require 'inc/layout.php'; page_start('Help', 'help.php'); ?>
<div class="card full"><h1>Help</h1>
<h3>What does this tool do?</h3><p>It checks network flows for signs of attack and explains every decision, so staff without security training can decide what to do.</p>
<h3>How do I analyse traffic?</h3><ol><li>Export flows with CICFlowMeter as a .csv file.</li><li>Open <a href="analyse.php">Analyse traffic</a>, choose the file and press <em>Analyse file</em> (or try the sample file).</li><li>Review suspicious flows first; the "Why" column lists the features that drove each decision.</li></ol>
<h3>What do the numbers mean?</h3><p><strong>Probability</strong> is the model's confidence that a flow is an attack; flows at or above 0.5 are flagged. <strong>SHAP contributions</strong> show how much each feature pushed the decision towards (+) or away from (−) an attack.</p>
<h3>Limitations</h3><p>The model was trained on public research data (CIC-IDS2017). On a new network it should be calibrated with a sample of local benign traffic before alerts are trusted, and a person should always make the final decision.</p></div>
<?php page_end();
