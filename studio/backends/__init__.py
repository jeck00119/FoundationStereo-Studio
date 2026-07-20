"""Stereo model backends.

Each backend adapts one stereo network to a common interface (see base.py) so
the GUI, worker and engine child can drive any model the same way. The registry
(registry.py) holds metadata-only descriptors — importable without torch or any
model's code — so the GUI can list/enable models cheaply; the heavy adapter is
imported only inside the engine child, in that backend's own environment.
"""
from __future__ import annotations

from .base import (BackendSpec, CheckpointSpec, ParamSpec, StereoBackend)
from .registry import BACKENDS, DEFAULT_BACKEND, get_spec, load_backend

__all__ = [
    "StereoBackend", "BackendSpec", "CheckpointSpec", "ParamSpec",
    "BACKENDS", "DEFAULT_BACKEND", "get_spec", "load_backend",
]
