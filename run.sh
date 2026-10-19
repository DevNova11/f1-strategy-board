#!/usr/bin/env bash
# F1 Strategy Board - set up and launch on macOS / Linux.
#   ./run.sh            create the venv if needed, install, launch the dashboard
#   ./run.sh --fetch    also download the real data first (needs network)
#   ./run.sh --test     run the test suite instead of launching
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip --quiet
  .venv/bin/python -m pip install -r requirements.txt
fi

case "${1:-}" in
  --fetch) .venv/bin/python -m src.fetch ;;
  --test)  exec .venv/bin/python -m pytest ;;
esac
exec .venv/bin/python -m streamlit run app/Home.py --server.port "${PORT:-8501}"
