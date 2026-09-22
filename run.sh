#!/usr/bin/env bash
# =============================================================================
# run.sh — launches the AI Visual Screen Inspection Streamlit app.
# Run ./setup.sh once first. Then just: ./run.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
    echo "ERROR: ./.venv not found. Run ./setup.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import streamlit" >/dev/null 2>&1; then
    echo "ERROR: streamlit is not installed in .venv. Run ./setup.sh again."
    exit 1
fi

echo "Starting AI Visual Screen Inspection..."
echo "Your browser should open automatically. If not, open the URL shown below."
echo ""

streamlit run app/app.py
