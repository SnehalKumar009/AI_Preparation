#!/usr/bin/env bash
# Build Phase 5's isolated Python environments and register each one as a
# Jupyter kernel. Frameworks pin conflicting versions of pydantic / openai /
# httpx / opentelemetry, so each framework gets its own environment instead of
# sharing one .venv. Needs `uv` (https://docs.astral.sh/uv/).
#
#   ./setup.sh                  build every environment
#   ./setup.sh core langgraph   build only the ones named
#   ./setup.sh --check          import-test every environment that exists
#   ./setup.sh --list           show environments and their kernels
#
# Environments live in PHASE_5_ORCHESTRATION/.venvs/<name>; the kernels are
# registered for your user as "Phase 5 · <name>" (id: phase5-<name>).
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_VERSION="${PHASE5_PYTHON:-3.12}"
ALL_ENVS=(core langgraph openai_agents crewai msaf adk)

# The module each environment must be able to import (used by --check).
declare -A PROBE=(
  [core]="temporalio"
  [langgraph]="langgraph"
  [openai_agents]="agents"
  [crewai]="crewai"
  [msaf]="agent_framework"
  [adk]="google.adk"
)

venv_python() { echo ".venvs/$1/bin/python"; }

build_env() {
  local name="$1"
  local req="requirements/${name}.txt"
  [[ -f "$req" ]] || { echo "unknown environment: $name (no $req)"; exit 1; }

  echo "==> [$name] creating .venvs/$name (Python $PYTHON_VERSION)"
  uv venv --quiet --allow-existing --python "$PYTHON_VERSION" ".venvs/$name"

  echo "==> [$name] installing $req"
  uv pip install --quiet --python "$(venv_python "$name")" -r "$req"

  echo "==> [$name] registering kernel phase5-$name"
  "$(venv_python "$name")" -m ipykernel install --user \
    --name "phase5-$name" --display-name "Phase 5 · $name" >/dev/null
}

check_env() {
  local name="$1"
  local py
  py="$(venv_python "$name")"
  if [[ ! -x "$py" ]]; then
    printf '  %-14s not built\n' "$name"
    return
  fi
  # Every env must import the shared lab packages plus its own framework.
  if "$py" - "${PROBE[$name]}" <<'EOF' >/dev/null 2>&1
import importlib, sys
sys.path.insert(0, ".")
import orch_lab  # bootstraps study_buddy, rag_lab, agents_lab, mcp_lab
for mod in ("study_buddy", "agents_lab", "mcp_lab", sys.argv[1]):
    importlib.import_module(mod)
EOF
  then
    printf '  %-14s ok   (%s)\n' "$name" "${PROBE[$name]}"
  else
    printf '  %-14s FAIL (run: %s -c "import orch_lab, %s")\n' "$name" "$py" "${PROBE[$name]}"
  fi
}

case "${1:-}" in
  --check)
    echo "Phase 5 environments:"
    for name in "${ALL_ENVS[@]}"; do check_env "$name"; done
    ;;
  --list)
    for name in "${ALL_ENVS[@]}"; do
      printf '  %-14s kernel "Phase 5 · %s"   requirements/%s.txt\n' "$name" "$name" "$name"
    done
    ;;
  -h|--help)
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
    ;;
  *)
    command -v uv >/dev/null || { echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
    if [[ $# -gt 0 ]]; then envs=("$@"); else envs=("${ALL_ENVS[@]}"); fi
    for name in "${envs[@]}"; do build_env "$name"; done
    echo
    "$0" --check
    ;;
esac
