from __future__ import annotations

import os
from pathlib import Path


def configure_matplotlib_env() -> Path:
    """Point Matplotlib at a writable repo-local config/cache directory."""
    configured = os.environ.get("MPLCONFIGDIR")
    if configured:
        target = Path(configured).expanduser()
    else:
        target = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
        os.environ["MPLCONFIGDIR"] = str(target)

    target.mkdir(parents=True, exist_ok=True)
    return target
