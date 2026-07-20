"""Qt-free image loading shared by the Input panel and the folder batch.

One loader, imported by both, so a batched pair is BY CONSTRUCTION fed to the
engine exactly as a hand-dropped one (the two used to carry hand-synced copies).
Kept free of PySide6 so headless tools (tools/calibrate.py) can import it
without dragging in the GUI stack.
"""
from __future__ import annotations

import numpy as np

#: every raster format both file dialogs and the batch scanner accept
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm", ".webp"}
IMG_FILTER = "Images (" + " ".join("*" + e for e in sorted(IMG_EXTS)) + ")"


def to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    """Any imageio-loaded raster → (H,W,3) uint8 RGB.

    The old conversion was ``arr[..., :3].astype(np.uint8)``, which wrapped
    16-bit samples modulo 256 (garbage fed straight into inference, no warning)
    and let a 2-channel gray+alpha image through as 2 channels (an out-of-bounds
    stride read in the thumbnail). Depth is scaled first, channels fixed second.
    """
    a = np.asarray(arr)
    # --- sample depth → uint8 -------------------------------------------------
    if a.dtype == np.uint8:
        pass
    elif a.dtype == np.uint16:
        mx = int(a.max()) if a.size else 0
        # true 16-bit data scales by 257 (65535→255); a file that only ever uses
        # 0–255 is 8-bit data in a 16-bit container and must pass through as-is
        a = a.astype(np.uint8) if mx <= 255 else \
            (a.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
    elif np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        a = np.clip(a.astype(np.float32) * (255.0 / max(int(info.max), 1)),
                    0, 255).astype(np.uint8)
    elif np.issubdtype(a.dtype, np.floating):
        mx = float(np.nanmax(a)) if a.size else 0.0
        scale = 255.0 if mx <= 1.0 + 1e-6 else 1.0    # 0–1 floats vs 0–255 floats
        a = np.clip(np.nan_to_num(a) * scale, 0, 255).astype(np.uint8)
    else:
        a = a.astype(np.uint8)
    # --- channels → 3 ---------------------------------------------------------
    if a.ndim == 2:
        return np.ascontiguousarray(np.stack([a] * 3, -1))
    if a.ndim == 3 and a.shape[2] in (1, 2):          # gray / gray+alpha
        g = a[..., 0]
        return np.ascontiguousarray(np.stack([g] * 3, -1))
    if a.ndim == 3 and a.shape[2] >= 3:               # RGB / RGBA
        return np.ascontiguousarray(a[..., :3])
    raise ValueError(f"unsupported image shape {a.shape}")


def load_rgb(path: str) -> np.ndarray:
    """Read an image file as (H,W,3) uint8 RGB. Raises on unreadable input —
    callers decide whether that's a notice (Input panel) or a banked failure
    (batch)."""
    import imageio.v2 as imageio

    return to_rgb_u8(imageio.imread(path))
