"""Which Phase 5 environment is this notebook running in?

Every framework lives in its own Python environment (built by ``setup.sh``) and
shows up in Jupyter as a kernel called ``Phase 5 · <env>``. A notebook calls
:func:`requires` at the top; if the framework it teaches isn't importable, the
student gets a one-line fix instead of a stack trace, and the framework cells
can skip cleanly — the same way cloud cells skip when an API key is missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def current_env() -> str:
    """Name of the Phase 5 environment running this kernel, or ``'other'``.

    Environments live in ``PHASE_5_ORCHESTRATION/.venvs/<name>``, so the name is
    the folder that contains the running interpreter's virtualenv.
    """
    prefix = Path(sys.prefix)
    if prefix.parent.name == ".venvs":
        return prefix.name
    return "other"


def requires(*modules: str, env: str) -> bool:
    """Return True if every module is importable; otherwise explain the fix.

    ``env`` is the Phase 5 environment that provides the modules, e.g.
    ``requires("langgraph", env="langgraph")``.
    """
    missing = [m for m in modules if importlib.util.find_spec(m.split(".")[0]) is None]
    if not missing:
        return True
    print(
        f"Missing {', '.join(missing)} — this notebook needs the kernel "
        f"'Phase 5 · {env}' (you are on '{current_env()}').\n"
        f"  1. Build it once:   ./setup.sh {env}      (from PHASE_5_ORCHESTRATION/)\n"
        f"  2. Switch kernel:   Kernel ▸ Change Kernel ▸ Phase 5 · {env}\n"
        "Cells that need it will be skipped."
    )
    return False
