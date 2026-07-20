"""Guarantee sys.stdout / sys.stderr are real writable streams.

pythonw.exe (used for a console-less launch, e.g. run_studio.bat) starts the
interpreter with **no console**, so it sets ``sys.stdout`` and ``sys.stderr`` to
``None``. Qt, pyqtgraph and — critically — ``torch.hub.load`` (the DINOv2
backbone loader) all call ``sys.stderr.write(...)`` for warnings/progress, which
under pythonw crashes with::

    AttributeError: 'NoneType' object has no attribute 'write'

Import this and call :func:`ensure_streams` before importing those libraries.
"""
from __future__ import annotations

import os
import sys
import tempfile


def ensure_streams(tag: str = "fs_studio") -> None:
    """If stdout/stderr are None (pythonw), point them at a log file (or devnull)."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        sink = open(
            os.path.join(tempfile.gettempdir(), f"{tag}.log"),
            "a", buffering=1, encoding="utf-8", errors="replace",
        )
    except Exception:
        sink = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = sink
    if sys.stderr is None:
        sys.stderr = sink
    # faulthandler / crash handlers write to the __ originals — keep them valid too
    if getattr(sys, "__stdout__", None) is None:
        sys.__stdout__ = sink
    if getattr(sys, "__stderr__", None) is None:
        sys.__stderr__ = sink
