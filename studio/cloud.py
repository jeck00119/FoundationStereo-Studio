"""Model-agnostic point-cloud construction.

Turns a disparity/depth result (from ANY stereo backend) plus calibration into a
colored, filtered 3D point cloud. Depends only on numpy + the shared dtypes (and
open3d, imported lazily) — deliberately NOT on the FoundationStereo repo's
``core.*`` / ``Utils`` modules, so it works unchanged for every backend and in
any of the per-model child environments. ``depth2xyzmap`` and ``toOpen3dCloud``
are vendored verbatim from the FoundationStereo repo's Utils.py.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .dtypes import CloudResult, InferResult, Progress, StereoParams, tick


# --- vendored from FoundationStereo/Utils.py (kept bit-for-bit) --------------
def toOpen3dCloud(points, colors=None, normals=None):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def depth2xyzmap(depth: np.ndarray, K, uvs: np.ndarray = None, zmin=0.1):
    invalid_mask = (depth < zmin)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing="ij")
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map


# --- cloud pipeline ---------------------------------------------------------
def denoise_mask(pts: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    """Statistical outlier removal via Open3D's multithreaded tensor kernel.

    Returns a boolean KEEP mask over ``pts`` (not filtered arrays) so callers can
    carry parallel per-point arrays (colors, origin, reliability) through the same
    filter. Removes points whose mean distance to their k nearest neighbors
    exceeds mean + std_ratio*std. Benchmarked at ~0.4 s on a 470k-pt cloud vs
    ~19 s for the legacy radius-outlier filter, and it keeps FULL point density
    (no downsampling). Radius-outlier was replaced because its per-point
    fixed-radius search is pathologically slow/hangs on dense clouds; k-NN
    statistical is bounded and fast. Falls back to the legacy statistical filter
    if the tensor API is unavailable."""
    import open3d as o3d

    n = len(pts)
    try:
        tpcd = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(np.ascontiguousarray(pts, np.float32), o3d.core.Dtype.Float32)
        )
        out = tpcd.remove_statistical_outliers(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
        )
        mask = out[1].numpy()
        if mask.dtype == np.bool_:
            return mask
        keep = np.zeros(n, bool)  # some builds return kept indices instead
        keep[mask.astype(np.int64)] = True
        return keep
    except Exception:
        pcd = toOpen3dCloud(pts, np.zeros((n, 3), np.uint8))
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio)
        )
        keep = np.zeros(n, bool)
        keep[np.asarray(ind, np.int64)] = True
        return keep


def lrc_mask(disp_ref: np.ndarray, disp_other: np.ndarray, sign: int) -> np.ndarray:
    """Left-right consistency mask for a reference disparity map.

    Returns True where ``disp_ref`` agrees (within 1 px) with the disparity
    sampled from the OTHER image at the matched column. For a left reference the
    match sits at column ``u - d`` (sign=-1); for a right reference at ``u + d``
    (sign=+1). Pixels that fail — occlusions, silhouette edges, off-frame
    matches — come back False."""
    H, W = disp_ref.shape
    xx = np.arange(W, dtype=np.float32)[None, :]        # (1,W), broadcasts (float32)
    valid = disp_ref > 0
    um = np.round(xx + sign * disp_ref).astype(np.int64)   # (H,W) matched columns
    inb = valid & (um >= 0) & (um < W)
    d_at = np.zeros_like(disp_ref)
    vy, vx = np.nonzero(inb)                            # avoid a full H×W row-index array
    d_at[vy, vx] = disp_other[vy, um[vy, vx]]
    return inb & (d_at > 0) & (np.abs(disp_ref - d_at) <= 1.0)


def _project_side(depth, rgb, K, z_near, z_far, tx, origin, reliable):
    """Back-project one depth map to a filtered, tagged point set.

    ``tx`` shifts X into the common (left) camera frame — 0 for the left eye,
    +baseline for the right. Returns (points, colors, origin[], reliable[]) with
    z-range filtering already applied, all arrays aligned per surviving point.

    NOTE: ``depth2xyzmap`` defaults to zmin=0.1 m, which silently deletes any
    geometry closer than 10 cm — fatal for close-range / macro setups (small
    baseline, short working distance). We pass zmin≈0 and do ALL near/far
    clipping ourselves via z_near/z_far so nothing valid is dropped implicitly."""
    # depth2xyzmap already returns fresh float32; rgb is uint8, reliable is bool —
    # so these are views, no redundant full-array copies (the [keep] index copies)
    xyz = depth2xyzmap(depth, K, zmin=1e-6).reshape(-1, 3).astype(np.float32, copy=False)
    if tx:
        xyz[:, 0] += float(tx)   # in-place on the fresh local map — safe, no aliasing
    cols = rgb.reshape(-1, 3)
    rel = reliable.reshape(-1)
    z = xyz[:, 2]
    lo = max(float(z_near), 1e-6)   # >0 so invalid (depth=0) pixels never survive
    keep = (z >= lo) & (z <= float(z_far))
    n = int(keep.sum())
    return xyz[keep], cols[keep], np.full(n, origin, np.uint8), rel[keep]


def build_cloud(
    result: InferResult, params: StereoParams, progress: Progress = None
) -> Optional[CloudResult]:
    """Turn a depth result into a colored point cloud. Independent of the network
    so z_far / denoise changes re-apply instantly, and of the model so any backend
    that yields disparity + calibration reaches here unchanged."""
    if result.depth is None or result.K is None:
        return None

    K = result.K
    fx = float(K[0, 0])
    # use the baseline that produced result.depth (left eye) so the right eye
    # can't diverge if params.baseline was edited without re-inferring
    B = float(result.baseline) if result.baseline else float(params.baseline)
    z_near = float(params.z_near)
    z_far = float(params.z_far)
    disp_L = result.disp
    H, W = disp_L.shape
    xx = np.arange(W, dtype=np.float32)[None, :]   # broadcasts over rows (no repeat)
    dual = (bool(params.dual_reference)
            and result.disp_right is not None and result.rgb_right is not None)

    tick(progress, "Projecting to 3D…")

    # reliability = left-right consistency (needs the right pass to check).
    # Without dual reference we can't detect occlusion, so all valid = reliable.
    rel_L = lrc_mask(disp_L, result.disp_right, sign=-1) if dual else (disp_L > 0)

    # -- left eye -------------------------------------------------------
    depth_L = result.depth.copy()
    if params.remove_invisible:
        depth_L[(xx - disp_L) < 0] = 0
    pts, cols, ori, rel = _project_side(
        depth_L, result.rgb, K, z_near, z_far, tx=0.0, origin=0, reliable=rel_L
    )

    # -- right eye: back-project, then translate +baseline into left frame
    if dual:
        disp_R = result.disp_right
        rel_R = lrc_mask(disp_R, disp_L, sign=+1)
        depth_R = np.zeros((H, W), np.float32)
        vR = disp_R > 0
        depth_R[vR] = fx * B / disp_R[vR]
        if params.remove_invisible:
            depth_R[(xx + disp_R) >= W] = 0   # match runs off the left image
        pR, cR, oR, rR = _project_side(
            depth_R, result.rgb_right, K, z_near, z_far, tx=B, origin=1, reliable=rel_R
        )
        pts = np.concatenate([pts, pR])
        cols = np.concatenate([cols, cR])
        ori = np.concatenate([ori, oR])
        rel = np.concatenate([rel, rR])

    # -- denoise over the merged cloud, carrying every per-point array --
    if params.denoise and len(pts) > 0:
        tick(progress, "Denoising cloud…")
        try:
            keep = denoise_mask(
                pts, nb_neighbors=int(params.denoise_nb_points),
                std_ratio=float(params.denoise_std),
            )
        except ImportError:
            # open3d has no wheel for every platform (Jetson aarch64 gaps) —
            # a missing OPTIONAL filter must degrade, not kill the whole run
            tick(progress, "Denoise skipped — open3d is not installed.")
        else:
            pts, cols, ori, rel = pts[keep], cols[keep], ori[keep], rel[keep]

    return CloudResult(points=pts, colors=cols, n=len(pts), origin=ori, reliable=rel)


def save_cloud(path: str, cloud: CloudResult) -> None:
    import open3d as o3d

    pcd = toOpen3dCloud(cloud.points, cloud.colors)
    o3d.io.write_point_cloud(path, pcd)
