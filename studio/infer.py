"""Model-agnostic inference orchestration.

Everything ``StereoEngine.infer`` used to do EXCEPT the network call itself:
scale the pair, run the backend for the left reference, optionally run the
right reference via the horizontal-flip trick, and turn disparity into
world-unit depth. The one model-specific step — producing disparity — is the
backend's ``disparity()``. Works with any StereoBackend.
"""
from __future__ import annotations

import time

import numpy as np

from .backends.base import StereoBackend
from .dtypes import InferResult, Progress, StereoParams, tick


def run_inference(
    backend: StereoBackend,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    params: StereoParams,
    progress: Progress = None,
) -> InferResult:
    if not backend.is_loaded:
        raise RuntimeError("Model not loaded — call load() first.")
    import cv2

    t_all = time.time()
    scale = float(params.scale)
    img0 = np.ascontiguousarray(left_rgb[..., :3])
    img1 = np.ascontiguousarray(right_rgb[..., :3])
    if scale != 1.0:
        img0 = cv2.resize(img0, None, fx=scale, fy=scale)
        img1 = cv2.resize(img1, None, fx=scale, fy=scale)
    H, W = img0.shape[:2]
    rgb_working = img0.copy()

    tick(progress, "Running network…")
    t_net = time.time()
    dres = backend.disparity(img0, img1, params)
    disp = dres.disp
    confidence = dres.confidence

    disp_right = None
    rgb_right = None
    if params.dual_reference:
        tick(progress, "Right pass (both eyes)…")
        # Right image as reference via the horizontal-flip trick: disparity is
        # left-positive, so flipping both images turns the right camera into a
        # "left" reference; flip the disparity back afterwards.
        dres_r = backend.disparity(
            np.ascontiguousarray(img1[:, ::-1]),
            np.ascontiguousarray(img0[:, ::-1]),
            params,
        )
        disp_right = np.ascontiguousarray(dres_r.disp[:, ::-1])
        rgb_right = img1.copy()
    net_s = time.time() - t_net

    depth = None
    K = None
    if params.has_calibration:
        K = params.intrinsics(scale)
        valid = disp > 0
        depth = np.zeros((H, W), np.float32)
        depth[valid] = K[0, 0] * float(params.baseline) / disp[valid]

    timing = {"net_s": net_s, "total_s": time.time() - t_all}
    return InferResult(
        disp=disp, depth=depth, rgb=rgb_working,
        H=H, W=W, scale=scale, timing=timing, K=K,
        baseline=float(params.baseline),
        disp_right=disp_right, rgb_right=rgb_right, confidence=confidence,
    )
