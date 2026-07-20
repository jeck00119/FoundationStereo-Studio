"""Backward-compatibility facade over the backend architecture.

Historically ``StereoEngine`` WAS the FoundationStereo engine. It is now a thin
wrapper around the FoundationStereo backend (studio/backends/) plus the shared,
model-agnostic ``run_inference`` (studio/infer.py) and ``build_cloud``
(studio/cloud.py). Existing callers — the engine child, the CLI, tests — keep
working while the GUI moves to driving backends by key. Shared data types +
unit constants are re-exported here so ``from .engine import StereoParams`` etc.
still resolve.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from . import cloud
from .dtypes import (ANGLE_DECIMALS, UNIT_DECIMALS, UNIT_PER_M, CloudResult,
                     DisparityResult, InferResult, Progress, StereoParams)

__all__ = [
    "StereoParams", "InferResult", "CloudResult", "DisparityResult",
    "UNIT_PER_M", "UNIT_DECIMALS", "ANGLE_DECIMALS", "Progress", "StereoEngine",
    "DEFAULT_CKPT", "REPO_ROOT",
]

# --- make the FoundationStereo repo importable (core.*, Utils) -------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DEFAULT_CKPT = os.path.join(
    REPO_ROOT, "pretrained_models", "23-51-11", "model_best_bp2.pth"
)


class StereoEngine:
    """Loads FoundationStereo once, then serves inference + cloud building.
    (Facade — delegates to the FoundationStereo backend + shared pipeline.)"""

    def __init__(self) -> None:
        from .backends.foundation_stereo import FoundationStereoBackend

        self._backend = FoundationStereoBackend()
        self.ckpt_dir: Optional[str] = None

    # ------------------------------------------------------------------ load
    def load(self, ckpt_dir: str = DEFAULT_CKPT, progress: Progress = None) -> None:
        self._backend.load(ckpt_dir, None, progress)
        self.ckpt_dir = ckpt_dir

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded

    def device_name(self) -> str:
        return self._backend.device_name()

    def vram_gb(self):
        return self._backend.vram_gb()

    def unload(self) -> None:
        self._backend.unload()

    # ----------------------------------------------------------------- infer
    def infer(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        params: StereoParams,
        progress: Progress = None,
    ) -> InferResult:
        from .infer import run_inference

        return run_inference(self._backend, left_rgb, right_rgb, params, progress)

    # ----------------------------------------------------------- build cloud
    def build_cloud(
        self, result: InferResult, params: StereoParams, progress: Progress = None
    ) -> Optional[CloudResult]:
        """Delegates to the model-agnostic cloud builder (studio.cloud)."""
        return cloud.build_cloud(result, params, progress)

    # --------------------------------------------------------------- exports
    @staticmethod
    def save_cloud(path: str, cloud_result: CloudResult) -> None:
        cloud.save_cloud(path, cloud_result)

    @staticmethod
    def colorize_disparity(disp: np.ndarray, cmap_name: str = "TURBO",
                           vmin=None, vmax=None) -> np.ndarray:
        """RGB uint8 visualization using the repo's own colormap logic."""
        import cv2
        from Utils import vis_disparity

        cmaps = {
            "TURBO": cv2.COLORMAP_TURBO,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "JET": cv2.COLORMAP_JET,
            "PLASMA": cv2.COLORMAP_PLASMA,
        }
        code = cmaps.get(cmap_name.upper(), cv2.COLORMAP_TURBO)
        return vis_disparity(disp, min_val=vmin, max_val=vmax, color_map=code)
