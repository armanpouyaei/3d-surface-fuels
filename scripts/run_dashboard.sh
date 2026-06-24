#!/usr/bin/env bash
# Launch the 3D Surface Fuels dashboard. Run from the project root.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d .venv ]; then source .venv/bin/activate; fi
exec streamlit run dashboard/app.py "$@"
