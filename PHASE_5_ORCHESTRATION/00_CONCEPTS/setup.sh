#!/usr/bin/env bash
# Build this section's own Python environment (.venv) and register it as the
# Jupyter kernel "Phase 5 · Concepts". Nothing outside this folder is used.
#
#   ./setup.sh            build (safe to re-run; it only adds what's missing)
#
# Uses uv (fast) when available, otherwise plain python3 -m venv + pip.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if command -v uv >/dev/null; then
  uv venv --quiet --allow-existing --python "$PYTHON_VERSION" .venv
  uv pip install --quiet --python .venv/bin/python -r requirements.txt
else
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

.venv/bin/python -m ipykernel install --user --name "phase5-concepts" --display-name "Phase 5 · Concepts" >/dev/null
echo "Ready. In Jupyter or VS Code choose the kernel: Phase 5 · Concepts"
