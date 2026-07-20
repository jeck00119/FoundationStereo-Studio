"""Qt-free image loading + stereo-pair discovery.

Shared by the Input panel, the folder batch and the calibration CLI: one loader
and one pairing rulebook, so a batched or calibrated pair is BY CONSTRUCTION
read and matched exactly like a hand-dropped one. Kept free of PySide6 so
headless tools (tools/calibrate.py) can import it without the GUI stack.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

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


# ------------------------------------------------------------------ discovery
@dataclass
class PairScan:
    pairs: list                              # [(label, left_path, right_path)]
    method: str                              # human description of how they matched
    unpaired: list = field(default_factory=list)   # leftover image filenames

    def __bool__(self) -> bool:
        return bool(self.pairs)


def _images_in(folder: str) -> list:
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    return sorted(f for f in names
                  if os.path.splitext(f)[1].lower() in IMG_EXTS
                  and os.path.isfile(os.path.join(folder, f)))


# Left/right SUBFOLDER name pairs, most specific first. Each entry is
# (left-name aliases, right-name aliases), all lower-case.
_LR_DIRNAMES = [
    (("left", "cam_left", "camleft", "left_cam", "leftcam"),
     ("right", "cam_right", "camright", "right_cam", "rightcam")),
    (("cam0", "cam_0", "camera0", "cam00", "view0", "im0", "image0"),
     ("cam1", "cam_1", "camera1", "cam01", "view1", "im1", "image1")),
    (("l",), ("r",)),
    (("0",), ("1",)),
]

# Single-folder filename conventions, most specific first. Each regex runs on the
# file STEM (no extension); the 's' group's lower-case value maps to a side.
_SUFFIX_FAMILIES = [
    (re.compile(r"^(?P<key>.+?)[ _.\-]+(?P<s>left|right)$", re.I),
     {"left": "L", "right": "R"}, "name_left / name_right"),
    (re.compile(r"^(?P<key>.+?)[ _.\-]*cam[ _.\-]*(?P<s>0|1)$", re.I),
     {"0": "L", "1": "R"}, "name_cam0 / name_cam1"),
    (re.compile(r"^(?P<key>.+?)[ _.\-]+(?P<s>l|r)$", re.I),
     {"l": "L", "r": "R"}, "name_L / name_R"),
    (re.compile(r"^(?P<key>.+?)[ _.\-]+(?P<s>0|1)$", re.I),
     {"0": "L", "1": "R"}, "name_0 / name_1"),
]
_PREFIX_FAMILIES = [
    (re.compile(r"^(?P<s>left|right)[ _.\-]+(?P<key>.+)$", re.I),
     {"left": "L", "right": "R"}, "left_name / right_name"),
    (re.compile(r"^(?P<s>l|r)[ _.\-]+(?P<key>.+)$", re.I),
     {"l": "L", "r": "R"}, "L_name / R_name"),
]


def _find_lr_subdirs(folder: str):
    """(left_dir, right_dir) if `folder` holds a left/right pair of subfolders."""
    try:
        subs = [d for d in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, d))]
    except OSError:
        return None
    low = {d.lower(): d for d in subs}
    for lefts, rights in _LR_DIRNAMES:
        L = next((low[n] for n in lefts if n in low), None)
        R = next((low[n] for n in rights if n in low), None)
        if L and R:
            return os.path.join(folder, L), os.path.join(folder, R)
    return None


def _sibling_lr(folder: str):
    """The chosen folder may BE the left (or right) folder, with its partner
    sitting next to it. Match by folder name against a sibling."""
    base = os.path.basename(os.path.normpath(folder)).lower()
    parent = os.path.dirname(os.path.normpath(folder))
    if not parent or not os.path.isdir(parent):
        return None
    try:
        sibs = {d.lower(): d for d in os.listdir(parent)
                if os.path.isdir(os.path.join(parent, d))}
    except OSError:
        return None
    for lefts, rights in _LR_DIRNAMES:
        if base in lefts:
            R = next((sibs[n] for n in rights if n in sibs), None)
            if R:
                return folder, os.path.join(parent, R)
        if base in rights:
            L = next((sibs[n] for n in lefts if n in sibs), None)
            if L:
                return os.path.join(parent, L), folder
    return None


def _stem_map(files: list):
    """{stem: filename} plus the files whose stem collided with an earlier one
    (e.g. 1.png and 1.jpg) — those are ambiguous, kept as 'dropped' so they're
    reported rather than silently vanishing.

    Stems are lower-cased for MATCHING (values keep the real filename) — the same
    rule _pair_by_family documents: Windows filesystems are case-insensitive, so
    left/CAP_01.PNG must pair with right/cap_01.png instead of silently falling
    back to positional-order pairing."""
    m, dropped = {}, []
    for f in files:
        stem = os.path.splitext(f)[0].lower()
        if stem in m:
            dropped.append(f)
        else:
            m[stem] = f
    return m, dropped


def _pair_two_dirs(ld: str, rd: str):
    """Pair images across two folders by identical filename stem; fall back to
    positional order only if the counts match but no stems do."""
    limg, ldrop = _stem_map(_images_in(ld))
    rimg, rdrop = _stem_map(_images_in(rd))
    pairs = [(stem, os.path.join(ld, limg[stem]), os.path.join(rd, rimg[stem]))
             for stem in sorted(limg) if stem in rimg]
    if pairs:
        unpaired = ([limg[s] for s in sorted(limg) if s not in rimg]
                    + [rimg[s] for s in sorted(rimg) if s not in limg]
                    + ldrop + rdrop)               # same-stem duplicates, reported not dropped
        return pairs, unpaired, "matched by filename"
    li, ri = _images_in(ld), _images_in(rd)
    if li and len(li) == len(ri):
        pairs = [(os.path.splitext(a)[0], os.path.join(ld, a), os.path.join(rd, b))
                 for a, b in zip(li, ri)]
        return pairs, [], "paired by order — names differ, check the preview"
    return [], li + ri, "no matching filenames"


def _pair_by_family(files: list, regex, sidemap):
    """Group single-folder files into (key, left, right) by one naming family."""
    groups: dict = {}
    for f in files:
        m = regex.match(os.path.splitext(f)[0])
        if not m:
            continue
        side = sidemap.get(m.group("s").lower())
        if side is not None:
            # lower-case the key so IMG_L pairs with img_R (regex is re.I; on
            # Windows the filesystem is case-insensitive so this can't merge two
            # genuinely distinct captures)
            groups.setdefault(m.group("key").lower(), {})[side] = f
    return [(k, g["L"], g["R"]) for k, g in sorted(groups.items())
            if "L" in g and "R" in g]


def find_pairs(folder: str) -> PairScan:
    """Discover stereo pairs in `folder`. Tries, in order: left/right subfolders
    inside it, the folder + a sibling left/right folder, then single-folder
    filename conventions (name_L/name_R, name_left/name_right, cam0/cam1, …)."""
    folder = os.path.normpath(folder)
    if not os.path.isdir(folder):
        return PairScan([], "not a folder")

    lr, where = _find_lr_subdirs(folder), "subfolders"
    if lr is None:
        lr, where = _sibling_lr(folder), "folder + sibling"
    if lr is not None:
        ld, rd = lr
        pairs, unpaired, how = _pair_two_dirs(ld, rd)
        if pairs:
            method = f"{os.path.basename(ld)} / {os.path.basename(rd)} {where}, {how}"
            return PairScan(pairs, method, unpaired)

    files = _images_in(folder)
    for regex, sidemap, desc in _SUFFIX_FAMILIES + _PREFIX_FAMILIES:
        pairs = _pair_by_family(files, regex, sidemap)
        if pairs:
            paths = [(k, os.path.join(folder, l), os.path.join(folder, r))
                     for k, l, r in pairs]
            used = {l for _, l, r in pairs} | {r for _, l, r in pairs}
            unpaired = [f for f in files if f not in used]
            return PairScan(paths, f"filenames: {desc}", unpaired)

    return PairScan([], "no left/right pairs found", files)
