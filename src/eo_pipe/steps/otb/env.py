"""One-time OTB environment activation.

Sources ``otbenv.profile`` in a subshell and caches the resulting environment.
The env is applied only to OTB subprocesses (not ``os.environ``) so that
OTB's ``GDAL_DATA`` / ``PROJ_LIB`` / ``LD_LIBRARY_PATH`` don't leak into the
in-process rasterio/GDAL.

Profile path: ``EO_PIPE_OTB_PROFILE`` env var, else :data:`DEFAULT_OTB_PROFILE`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_OTB_PROFILE = Path("/home/brustt/Applications/OTB_9.1.1/otbenv.profile")

_otb_env: Optional[Dict[str, str]] = None
_activated = False


def activate_otb() -> Optional[Dict[str, str]]:
    """Activate OTB once; return the env for OTB subprocesses (``None`` = use current env).

    Skipped when ``otbcli_*`` is already on PATH or the profile is missing.
    Subsequent calls are no-ops.
    """
    global _otb_env, _activated
    if _activated:
        return _otb_env
    _activated = True

    if shutil.which("otbcli_OrthoRectification") is not None:
        logger.debug("OTB already available on PATH; skipping activation")
        return None

    profile = Path(os.environ.get("EO_PIPE_OTB_PROFILE", DEFAULT_OTB_PROFILE))
    if not profile.is_file():
        logger.warning("OTB profile not found at %s; OTB steps unavailable", profile)
        return None

    try:
        # otbenv.profile is POSIX sh but uses BASH vars to locate itself -> use bash.
        out = subprocess.run(
            ["bash", "-c", f'source "{profile}" >/dev/null 2>&1; env -0'],
            capture_output=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Failed to source OTB profile %s: %s", profile, exc)
        return None

    env = dict(os.environ)
    for item in out.decode(errors="replace").split("\0"):
        key, sep, value = item.partition("=")
        if sep and key:
            env[key] = value
    _otb_env = env
    logger.debug("OTB environment activated from %s", profile)
    return _otb_env


def otb_env() -> Optional[Dict[str, str]]:
    """Env to pass as ``subprocess.run(env=...)`` (activates on first call)."""
    return activate_otb()


def which_otb(cmd: str) -> Optional[str]:
    """Resolve *cmd* using the activated OTB PATH."""
    env = otb_env()
    return shutil.which(cmd, path=env["PATH"] if env else None)
