"""Stereo rectification from a loaded calibration — the optional "raw images" path.

The pipeline everywhere else assumes an already-rectified pair. This module is the
one place that can turn a RAW pair into that rectified pair, so nothing downstream
changes. It targets the user's single-camera CNC rig: one camera (shared intrinsics
K + distortion D) moved between two shots with relative pose R, T.

``cv2.stereoRectify`` does two useful things at once: it builds undistort+rectify
maps that remove lens distortion and row-align the pair, AND it yields the rectified
pinhole K and the baseline the depth stage needs — so rectifying a raw pair also
DERIVES its metric calibration (no hand-entered K.txt required). Doing it properly
also removes the small tilt/roll a hand-aligned CNC pair carries.

Baseline unit: ``T`` (and therefore the derived baseline) is read in the unit the
calibration was solved in — assumed MILLIMETRES here, matching the PCB workflow.
"""
from __future__ import annotations

import json
import os

import numpy as np


class CalibrationError(ValueError):
    """A calibration file that couldn't be read or is missing fields."""


def _arr(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64)


class StereoCalibration:
    """K, D, R, T for a single-camera stereo rig (both images share K, D).

    K: 3x3 intrinsics · D: distortion coeffs (any length OpenCV accepts) ·
    R: 3x3 rotation of the second shot relative to the first · T: 3-vector
    translation between them (its magnitude is the baseline). ``image_size`` is
    optional — the loaded image's size is used when the file omits it."""

    def __init__(self, K, D, R, T, image_size=None) -> None:
        self.K = _arr(K).reshape(3, 3)
        self.D = _arr(D).reshape(-1)
        self.R = _arr(R).reshape(3, 3)
        self.T = _arr(T).reshape(3)
        self.image_size = None
        if image_size is not None:
            wh = np.ravel(np.asarray(image_size))
            if wh.size >= 2:
                self.image_size = (int(round(float(wh[0]))), int(round(float(wh[1]))))

    @property
    def baseline_raw(self) -> float:
        """|T| — the physical camera translation, in the file's own unit."""
        return float(np.linalg.norm(self.T))

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str) -> "StereoCalibration":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            data = np.load(path)
            return cls._from_dict({k: data[k] for k in data.files}, path)
        if ext == ".json":
            with open(path, encoding="utf-8") as f:
                return cls._from_dict(json.load(f), path)
        if ext in (".yml", ".yaml", ".xml"):
            return cls._from_filestorage(path)
        # unknown extension — try to PARSE as JSON, else fall to OpenCV FileStorage.
        # Only a parse failure falls through: a valid-JSON file missing a field must
        # surface its clean CalibrationError, not a cryptic cv2 persistence error.
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (ValueError, UnicodeDecodeError, OSError):
            return cls._from_filestorage(path)
        return cls._from_dict(d, path)

    @staticmethod
    def _pick(d: dict, *names):
        low = {k.lower(): v for k, v in d.items()}
        for n in names:
            if n.lower() in low:
                return low[n.lower()]
        return None

    @classmethod
    def _from_dict(cls, d: dict, path: str) -> "StereoCalibration":
        K = cls._pick(d, "K", "camera_matrix", "cameramatrix", "M1", "KK", "mtx")
        D = cls._pick(d, "D", "dist", "distortion", "dist_coeffs", "distcoeffs",
                      "distortion_coefficients")
        R = cls._pick(d, "R", "rotation")
        T = cls._pick(d, "T", "tvec", "translation")
        size = cls._pick(d, "image_size", "imagesize", "size", "resolution")
        w = cls._pick(d, "image_width", "width", "w")
        h = cls._pick(d, "image_height", "height", "h")
        if size is None and w is not None and h is not None:
            size = (float(np.ravel(w)[0]), float(np.ravel(h)[0]))
        missing = [n for n, v in (("K", K), ("D", D), ("R", R), ("T", T)) if v is None]
        if missing:
            raise CalibrationError(
                f"{os.path.basename(path)} is missing: {', '.join(missing)}. "
                "Expected K, D, R, T (single-camera stereo calibration).")
        return cls(K, D, R, T, size)

    @classmethod
    def _from_filestorage(cls, path: str) -> "StereoCalibration":
        import cv2

        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise CalibrationError(f"couldn't open {os.path.basename(path)} as a calibration file")

        def rd(*names):
            for n in names:
                node = fs.getNode(n)
                if node is not None and not node.empty():
                    return node.mat()
            return None

        K = rd("K", "camera_matrix", "cameraMatrix", "M1", "mtx")
        D = rd("D", "dist_coeffs", "distCoeffs", "distortion", "dist")
        R = rd("R", "rotation")
        T = rd("T", "Tvec", "translation")
        wn, hn = fs.getNode("image_width"), fs.getNode("image_height")
        size = None
        if wn is not None and not wn.empty() and hn is not None and not hn.empty():
            size = (wn.real(), hn.real())
        fs.release()
        missing = [n for n, v in (("K", K), ("D", D), ("R", R), ("T", T)) if v is None]
        if missing:
            raise CalibrationError(
                f"{os.path.basename(path)} is missing: {', '.join(missing)}.")
        return cls(K, D, R, T, size)


class Rectifier:
    """Undistort+rectify maps for a FIXED image size, plus the derived rectified
    pinhole (fx/fy/cx/cy) and baseline. Each side is rectified independently: the
    left with map1, the right with map2 (they get different rectifying rotations)."""

    def __init__(self, calib: StereoCalibration, image_size, alpha: float = 0.0) -> None:
        import cv2

        self.size = (int(image_size[0]), int(image_size[1]))   # (w, h)
        # OpenCV is picky about shapes/dtype: K 3x3, D a row vector, R 3x3, T 3x1,
        # all float64
        K = np.ascontiguousarray(calib.K, np.float64)
        # If the calibration was solved at a DIFFERENT resolution than the image we
        # rectify, scale K to the target: focal + principal point scale with pixels
        # (fx,cx by width; fy,cy by height). Distortion coeffs are normalized, so
        # they don't change. Without this, maps built at the wrong size make
        # cv2.remap silently sample the wrong region.
        if calib.image_size is not None and tuple(calib.image_size) != self.size:
            sx = self.size[0] / float(calib.image_size[0])
            sy = self.size[1] / float(calib.image_size[1])
            K = K.copy()
            K[0, :] *= sx
            K[1, :] *= sy
        D = np.ascontiguousarray(calib.D, np.float64).reshape(1, -1)
        R = np.ascontiguousarray(calib.R, np.float64)
        T = np.ascontiguousarray(calib.T, np.float64).reshape(3, 1)
        R1, R2, P1, P2, Q, _roi1, _roi2 = cv2.stereoRectify(
            K, D, K, D, self.size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=float(alpha))
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            K, D, R1, P1, self.size, cv2.CV_32FC1)
        self.map2x, self.map2y = cv2.initUndistortRectifyMap(
            K, D, R2, P2, self.size, cv2.CV_32FC1)
        self.R1, self.R2, self.P1, self.P2, self.Q = R1, R2, P1, P2, Q
        # rectified pinhole (shared by both images after CALIB_ZERO_DISPARITY)
        self.fx = float(P1[0, 0])
        self.fy = float(P1[1, 1])
        self.cx = float(P1[0, 2])
        self.cy = float(P1[1, 2])
        # P2 = [[fx,0,cx,-fx*Tx],...]  ->  |Tx| = |P2[0,3]/fx|  (T's unit). Baseline is
        # a magnitude, so abs() — OpenCV's sign depends on which camera is "camera 1".
        self.baseline = (abs(float(P2[0, 3] / P1[0, 0])) if P1[0, 0]
                         else float(np.linalg.norm(T)))

    def rectify(self, img: np.ndarray, side: str) -> np.ndarray:
        """Undistort + rectify one image. ``side`` is 'L' or 'R'."""
        import cv2

        mx, my = ((self.map1x, self.map1y) if side == "L"
                  else (self.map2x, self.map2y))
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR)
