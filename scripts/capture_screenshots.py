"""Capture README screenshots of the running dashboard with headless Edge or Chrome.

Start the app first (``streamlit run app/Home.py``), then::

    python scripts/capture_screenshots.py --base-url http://localhost:8501

Images are written to ``assets/screenshots`` (override with ``--out``). Use
``--width 420`` to check the phone layout.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "screenshots"

# (file name, URL path, window height)
SHOTS: list[tuple[str, str, int]] = [
    ("01_overview", "/?page=overview", 1500),
    ("02_data_sources", "/data_sources", 1500),
    ("03_cleaning_report", "/cleaning_report", 1700),
    ("04_eda_univariate", "/eda_explorer?eda_tab=Distributions", 1700),
    ("05_eda_bivariate", "/eda_explorer?eda_tab=Relationships", 1900),
    ("06_eda_outliers", "/eda_explorer?eda_tab=Outliers", 1800),
    ("07_eda_time_series", "/eda_explorer?eda_tab=Trends", 1900),
    ("08_eda_clustering", "/eda_explorer?eda_tab=Team+groups", 1900),
    ("09_correlations", "/correlations", 1700),
    ("10_reliability_risk", "/reliability_risk", 1800),
    ("11_ranked_insights", "/ranked_insights", 1700),
    ("12_methodology", "/methodology", 1500),
]
MIN_BYTES = 60_000          # anything smaller is the loading skeleton, not the page
CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str:
    """Locate a Chromium-based browser that supports ``--headless --screenshot``."""
    for path in CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("msedge", "google-chrome", "chromium", "chromium-browser", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("No Edge or Chrome installation found.")


def close_browser(profile: str) -> None:
    """Stop every browser process that belongs to ``profile``.

    Headless Edge and Chrome can leave helper processes behind after writing a
    screenshot; without this they pile up and later captures hang.
    """
    try:
        import psutil
    except ImportError:          # psutil ships with ipykernel; without it rely on the browser exiting
        return
    for proc in psutil.process_iter(["cmdline"]):
        try:
            if any(profile in part for part in proc.info["cmdline"] or []):
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def capture(browser: str, url: str, target: Path, width: int, height: int, wait_ms: int) -> bool:
    """Take one screenshot in a throwaway browser profile. True if a file was written."""
    target.unlink(missing_ok=True)
    profile = tempfile.mkdtemp(prefix="f1sb_shot_")
    try:
        subprocess.Popen([
            browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
            f"--user-data-dir={profile}", f"--window-size={width},{height}", f"--virtual-time-budget={wait_ms}",
            f"--screenshot={target}", url,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + wait_ms / 1000 + 30
        while not target.exists() and time.monotonic() < deadline:
            time.sleep(0.5)
        time.sleep(1.5)          # let the file finish writing
    finally:
        close_browser(profile)
        time.sleep(0.5)
        shutil.rmtree(profile, ignore_errors=True)
    return target.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8501")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--wait-ms", type=int, default=30000, help="render time allowed per page")
    parser.add_argument("--out", type=Path, default=OUT, help="folder to write the images to")
    parser.add_argument("--attempts", type=int, default=4, help="retries per page while it is still loading")
    parser.add_argument("--only", nargs="*", help="capture only shots whose name contains one of these strings")
    args = parser.parse_args()

    browser = find_browser()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path, height in SHOTS:
        if args.only and not any(token in name for token in args.only):
            continue
        target = out_dir / f"{name}.png"
        # A cold page (nothing cached yet) can be captured while still loading: the
        # result is a small skeleton image. Retry until a full render comes back.
        for attempt in range(1, args.attempts + 1):
            written = capture(browser, args.base_url.rstrip("/") + path, target, args.width, height, args.wait_ms)
            if written and target.stat().st_size > MIN_BYTES:
                break
            time.sleep(3.0)
        size = target.stat().st_size // 1024 if target.exists() else 0
        status = "ok  " if size * 1024 > MIN_BYTES else "FAIL"
        print(f"{status} {target.name} ({size} KB, attempt {attempt})", flush=True)


if __name__ == "__main__":
    main()
