"""System tests UI-1 to UI-7: drive the real PHP dashboard in a Chromium browser with Playwright.

The tests start the Flask core engine (port 5005, if it is not already running) and a PHP
development server (port 8099) that writes to a temporary SQLite database, so the project's own
alerts.db is never changed. Each test follows arrange, act, assert.
Requirements: PHP 8 with pdo_sqlite and curl, and `pip install playwright && playwright install chromium`.
"""
import os, sys, time, socket, sqlite3, tempfile, subprocess, unittest, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root
DASH = os.path.join(ROOT, "deploy", "dashboard")
API_PORT, WEB_PORT = 5005, 8099
BASE = f"http://127.0.0.1:{WEB_PORT}"
PAGES = ["index.php", "analyse.php", "history.php", "performance.php", "help.php"]
NAV_LABELS = ["Live monitor", "Analyse traffic", "Past analyses", "Model performance", "Help"]

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # reported as an error, not silently skipped
    sync_playwright = None


def _port_open(port):
    """True if something is already listening on this local port."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait(port, seconds=90):
    """Wait until a server starts listening on the port (False after the time limit)."""
    end = time.time() + seconds
    while time.time() < end:
        if _port_open(port):
            return True
        time.sleep(0.5)
    return False


class TestDashboardUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the core engine and a PHP server with a temporary database, then open Chromium."""
        if sync_playwright is None:
            raise RuntimeError("Playwright is not installed: pip install playwright && playwright install chromium")
        env = dict(os.environ, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                   NIDS_DATASET="CIC-IDS2017")
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(k, None)  # the PHP engine call must go straight to 127.0.0.1
        cls.procs = []
        if not _port_open(API_PORT):
            cls.procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "deploy", "api.py"), str(API_PORT)],
                                              env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            if not _wait(API_PORT):
                raise RuntimeError("The core engine (deploy/api.py) did not start. Train the models first.")
        cls.tmp = tempfile.mkdtemp(prefix="nids_ui_")
        cls.db_path = os.path.join(cls.tmp, "ui_test.db")
        env["NIDS_DSN"] = "sqlite:" + cls.db_path
        cls.procs.append(subprocess.Popen(["php", "-S", f"127.0.0.1:{WEB_PORT}", "-t", DASH], env=env,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if not _wait(WEB_PORT, 20):
            raise RuntimeError("The PHP development server did not start (is PHP 8 installed?).")
        urllib.request.urlopen(BASE + "/history.php", timeout=30).read()  # creates the database tables
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        """Close the browser and stop the servers started for the tests."""
        for name in ("browser", "pw"):
            obj = getattr(cls, name, None)
            if obj is not None:
                obj.close() if name == "browser" else obj.stop()
        for p in getattr(cls, "procs", []):
            p.terminate()
            p.wait(timeout=10)

    def setUp(self):
        """A fresh browser tab for every test."""
        self.page = self.browser.new_page()

    def tearDown(self):
        self.page.close()

    # helpers
    def runs(self):
        """Number of analysis runs stored in the back-end database."""
        with sqlite3.connect(self.db_path) as con:
            return con.execute("SELECT COUNT(*) FROM analysis_run").fetchone()[0]

    def upload(self, files):
        """Open the Analyse page, attach a file (or nothing) and press Analyse file."""
        self.page.goto(BASE + "/analyse.php")
        if files is not None:
            self.page.set_input_files("#flows", files)
        self.page.click("button:has-text('Analyse file')")
        self.page.wait_for_load_state("load")

    def alert_text(self):
        """Text of the error message box (role=alert) on the page."""
        return self.page.locator("[role=alert]").inner_text()

    # UI-1
    def test_ui_navigation_present_on_every_page(self):
        """UI-1: all five navigation links appear, in order, on every page."""
        for name in PAGES:
            self.page.goto(f"{BASE}/{name}")
            links = self.page.locator("header nav a")
            self.assertEqual(links.count(), 5, name)
            self.assertEqual([links.nth(i).inner_text() for i in range(5)], NAV_LABELS, name)

    # UI-2
    def test_ui_sample_upload_shows_results_and_saves_run(self):
        """UI-2: uploading the sample file shows a results table, stores the run and takes under 10 s."""
        before = self.runs()
        t0 = time.perf_counter()
        self.upload(os.path.join(DASH, "sample_flows.csv"))
        elapsed = time.perf_counter() - t0
        self.assertIn("Analysis complete", self.page.locator("[role=status]").inner_text())
        self.assertGreater(self.page.locator("table tr").count(), 1)       # header plus result rows
        self.assertEqual(self.runs(), before + 1)                           # run stored
        self.assertLess(elapsed, 10.0)                                      # usable response time

    # UI-3
    def test_ui_no_file_error(self):
        """UI-3: pressing Analyse with no file gives a clear message and stores nothing."""
        before = self.runs()
        self.upload(None)
        self.assertIn("Please choose a CSV file", self.alert_text())
        self.assertEqual(self.runs(), before)

    # UI-4
    def test_ui_wrong_file_type_error(self):
        """UI-4: a non-CSV file gives a clear message and stores nothing."""
        before = self.runs()
        self.upload({"name": "notes.txt", "mimeType": "text/plain", "buffer": b"not a flow file"})
        self.assertIn("Only .csv files", self.alert_text())
        self.assertEqual(self.runs(), before)

    # UI-5
    def test_ui_missing_columns_error_names_columns(self):
        """UI-5: a file with missing columns gets a message that names them, and nothing is stored."""
        before = self.runs()
        self.upload(os.path.join(DASH, "sample_invalid_missing_columns.csv"))
        self.assertIn("Flow Duration", self.alert_text())
        self.assertEqual(self.runs(), before)                               # nothing stored

    # UI-6
    def test_ui_history_lists_stored_runs(self):
        """UI-6: the Past analyses page lists a run that was just stored."""
        self.upload(os.path.join(DASH, "sample_flows.csv"))
        with sqlite3.connect(self.db_path) as con:
            run_id, fname = con.execute("SELECT id, filename FROM analysis_run ORDER BY id DESC LIMIT 1").fetchone()
        self.page.goto(BASE + "/history.php")
        table = self.page.locator("table").inner_text()
        self.assertIn(f"#{run_id}", table)
        self.assertIn(fname, table)

    # UI-7
    def test_ui_output_is_html_escaped(self):
        """UI-7: HTML in a file name is shown as text and never run (script-injection check)."""
        with open(os.path.join(DASH, "sample_flows.csv"), "rb") as f:
            data = f.read()
        evil = "<img id=xss src=x>.csv"  # no "/" because PHP basename() would cut the name there
        self.upload({"name": evil, "mimeType": "text/csv", "buffer": data})
        self.assertIn(evil, self.page.locator("[role=status]").inner_text())  # shown as text
        self.assertEqual(self.page.locator("#xss").count(), 0)               # not rendered as HTML


if __name__ == "__main__":
    unittest.main(verbosity=2)
