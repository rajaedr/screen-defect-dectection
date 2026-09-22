#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-command setup for the AI Visual Screen Inspection project.
#
# What this does:
#   1. Checks you have a usable Python 3 (3.10-3.12 recommended).
#   2. Creates a local virtual environment in ./.venv (isolated from your
#      system Python — nothing global gets modified).
#   3. Installs everything from requirements.txt into it.
#
# Usage:
#   chmod +x setup.sh run.sh
#   ./setup.sh
#   ./run.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================================="
echo " AI Visual Screen Inspection — setup"
echo "=============================================================="

# --- 1. Find a Python interpreter -------------------------------------------
PYTHON_BIN=""
for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No python3 found on this system."
    echo "Install Python 3.11 from https://www.python.org/downloads/ (or "
    echo "'brew install python@3.11' if you use Homebrew), then re-run ./setup.sh"
    exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
echo "Using $PYTHON_BIN (Python $PY_VERSION)"

case "$PY_VERSION" in
    3.9|3.13|3.14)
        echo "WARNING: Python $PY_VERSION is outside the tested range (3.10-3.12)."
        echo "Setup will continue, but if you hit dependency errors, install"
        echo "Python 3.11 and re-run this script."
        ;;
esac

# --- 2. Create virtual environment ------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in ./.venv ..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "Virtual environment ./.venv already exists, reusing it."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# --- 3. Install dependencies -------------------------------------------------
echo "Upgrading pip..."
pip install --upgrade pip --quiet

echo "Installing dependencies from requirements.txt (this can take a few "
echo "minutes the first time, especially for torch)..."
pip install -r requirements.txt

# --- 4. Create expected local directories (git-ignored data/output areas) --
mkdir -p datasets/raw datasets/processed models results/reports

echo ""
echo "=============================================================="
echo " Setup complete."
echo " Next: ./run.sh"
echo "=============================================================="
