"""Shared data types + unit constants for the studio pipeline.

These are model-agnostic and dependency-light (numpy only), so every layer —
the GUI, the worker, the engine child, and each model backend — can import them
without pulling in torch or any model's code. Kept in their own module (rather
than in engine.py) so backends and cloud.py can share them without import
cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# --- progress callback ------------------------------------------------------
Progress = Optional[Callable[[str], None]]


def tick(cb: Progress, msg: str) -> None:
    if cb is not None:
        cb(msg)


# --- units ------------------------------------------------------------------
# The pipeline is unit-agnostic: depth = fx*baseline/disparity and the whole
# back-projected cloud come out in the SAME unit as `baseline` (fx and disparity
# are both pixels, so they cancel). The UI picks a display unit and feeds
# baseline / z_near / z_far in that unit, so nothing here needs to convert.
# K.txt is metres by FoundationStereo convention and is converted on load.
UNIT_PER_M = {"m": 1.0, "mm": 1000.0, "µm": 1_000_000.0}   # ×this to convert a metre value
# Readout precision, set to the FLOAT32 FLOOR so nothing meaningful is rounded away:
# depth/cloud are stored float32 (~7 significant digits), so at PCB depths (~60 mm)
# 0.1 µm is the finest digit that still carries signal — anything finer is pure storage
# noise. So each unit shows to 0.1 µm: mm→4 decimals, µm→1 decimal, m→6 (→1 µm, its
# own floor at this magnitude). Absolute accuracy is far coarser (disparity quantisation
# ≈ z²/(fx·B), ~0.28 mm/px at 60 mm); the extra decimals are for RELATIVE / repeatability
# comparisons (the whole point of µm here — small pins, run-to-run), not absolute truth.
UNIT_DECIMALS = {"m": 6, "mm": 4, "µm": 1}   # readout decimals — all ≈0.1 µm resolution
ANGLE_DECIMALS = 3                            # angle readouts (0.001°)


@dataclass
class StereoParams:
    """Everything the user can tune, in one place.

    Model-AGNOSTIC fields (scale, calibration, dual-reference, cloud) sit
    alongside a generic ``model_params`` dict whose keys come from the active
    backend's declared ParamSpec schema (e.g. FoundationStereo:
    valid_iters / hierarchical / mixed_precision / low_memory). This is what lets
    one params object drive any stereo model.
    """

    # model-agnostic inference.
    # 0.5, matching the panel's default: full resolution needs several times more
    # VRAM than a 12 GB card has on a normal pair, and the driver spills to system
    # RAM rather than erroring — so a 1.0 default is a silent 30x slowdown, not a
    # sharper result. Every GUI run passes this explicitly (panels.current_params);
    # this default is what a programmatic caller gets, and it should be the safe one.
    scale: float = 0.5

    # backend-specific knobs (keys defined by the active backend's ParamSpecs)
    model_params: dict = field(default_factory=dict)

    # calibration (metric depth needs fx + baseline)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    baseline: float = 0.0  # world unit (mm or m); depth & cloud come out in this unit

    # dual reference — also run the right image as reference (flip trick) and
    # merge both clouds so occluded/silhouette regions get filled from both eyes
    dual_reference: bool = False

    # point cloud (z_near/z_far are in the same world unit as baseline)
    z_near: float = 0.0     # hide points closer than this (0 = no near clip)
    z_far: float = 10.0
    remove_invisible: bool = True
    denoise: bool = True
    denoise_nb_points: int = 20     # k nearest neighbors for statistical denoise
    denoise_std: float = 2.0        # keep points within mean + std_ratio*std

    @property
    def has_calibration(self) -> bool:
        return self.fx > 0 and self.baseline > 0

    def intrinsics(self, scale: float = 1.0) -> np.ndarray:
        K = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=np.float32,
        )
        K[:2] *= scale
        return K


@dataclass
class DisparityResult:
    """What a backend returns from one forward pass — a left-referenced
    disparity map plus whatever extra maps the model natively produces.

    Only ``disp`` is required. ``confidence`` / ``occlusion`` are optional
    (e.g. S²M² produces them); when present they can drive the reliability view
    directly instead of our left-right-consistency estimate."""

    disp: np.ndarray                        # (H,W) float32 disparity of the LEFT (reference) image
    confidence: Optional[np.ndarray] = None  # (H,W) float32 0..1, higher = more reliable
    occlusion: Optional[np.ndarray] = None   # (H,W) bool/float, True/high = occluded


@dataclass
class InferResult:
    disp: np.ndarray                    # (H,W) float32 disparity, working scale
    depth: Optional[np.ndarray]         # (H,W) float32 world-unit depth (0 = invalid) or None
    rgb: np.ndarray                     # (H,W,3) uint8 left image, working scale
    H: int
    W: int
    scale: float
    timing: dict
    K: Optional[np.ndarray] = None
    baseline: float = 0.0               # world unit, the value used to compute depth
    # dual-reference extras (only when params.dual_reference) — right image as
    # reference, in the RIGHT camera's own pixel grid / frame
    disp_right: Optional[np.ndarray] = None   # (H,W) float32 right-ref disparity
    rgb_right: Optional[np.ndarray] = None     # (H,W,3) uint8 right image, working scale
    # optional native per-pixel reliability from the backend (left reference)
    confidence: Optional[np.ndarray] = None    # (H,W) float32 0..1 or None


@dataclass
class CloudResult:
    points: np.ndarray                  # (N,3) float32
    colors: np.ndarray                  # (N,3) uint8
    n: int = 0
    origin: Optional[np.ndarray] = None    # (N,) uint8: 0=left eye, 1=right eye
    reliable: Optional[np.ndarray] = None  # (N,) bool: left-right consistent (not occluded)
