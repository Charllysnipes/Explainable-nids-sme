#!/bin/bash
# start API, replay flows, serve dashboard, take screenshot, run tests
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/deploy"
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
NIDS_DATASET=CIC-IDS2017 python3 api.py 5005 > api.log 2>&1 &
API=$!
until curl -s --noproxy '*' http://127.0.0.1:5005/health >/dev/null; do sleep 1; done
python3 replay.py
php -S 127.0.0.1:8080 -t dashboard > php.log 2>&1 &
PHP=$!
sleep 2
ROOT="$ROOT" python3 - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1180, "height": 760}, device_scale_factor=2)
    pg.goto("http://127.0.0.1:8080/index.php")
    import os
    pg.screenshot(path=os.path.join(os.environ["ROOT"], "figs/dashboard.png"), full_page=True)
    b.close()
print("screenshot ok")
PY
kill $PHP
cd "$ROOT" && python3 tests/run_tests.py
kill $API
echo DEMO_DONE
