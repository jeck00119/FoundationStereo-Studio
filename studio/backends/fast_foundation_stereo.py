"""Fast-FoundationStereo backend — the real-time-oriented sibling of FoundationStereo.

Key differences from the FoundationStereo adapter:
  * The checkpoint is a FULLY SERIALIZED model object (``torch.load`` returns the
    whole ``FastFoundationStereo`` instance) — there is no cfg-build +
    load_state_dict. cfg.yaml beside it only carries a few args (max_disp…).
  * It runs the pure-PyTorch ``optimize_build_volume='pytorch1'`` cost-volume path
    (the Triton GWC kernel is only a speed optimization), and we DISABLE
    torch.compile / TorchInductor (which would otherwise demand a Triton install
    for its GPU codegen). Result: it runs in the app's own venv on Windows with no
    Triton at all — just a bit slower than the TensorRT/Triton deployment path.

The repo lives beside FoundationStereo (…/Desktop/Fast-FoundationStereo) and shares
the same ``core.*`` / ``Utils`` module names, so — like every backend — this only
ever loads inside its OWN engine child, with the Fast-FS repo prepended to sys.path.
"""
from __future__ import annotations

import importlib.util
import os

# torch.compile is NOT optional here — it is the memory budget.
#
# Fast-FS's cost-volume builder is @torch.compile'd: TorchInductor FUSES the
# expand → multiply → group-reduce so the giant (C, D, H/4, W/4) intermediate is
# never materialized. That fusion needs Triton for GPU codegen. Running it eager
# materializes the intermediates instead. MEASURED on the 3060 (932×806,
# max_disp 192, same pair, identical disparity to the pixel):
#     compiled  1.49 GB peak / 0.35 s      eager  6.29 GB peak / 0.55 s
# i.e. eager costs 4.2× the memory AND is slower. Peak scales linearly with
# pixels × max_disp, so at scale 0.5 (1332×1152) eager needs ~12.8 GB on a 12 GB
# card — Windows then spills VRAM into system RAM rather than raising OOM, and
# thrashes the machine (this crashed the user's system). The readme's own table
# quotes 653 MB peak, which is only reachable compiled.
#
# So: compile whenever Triton is importable; fall back to eager (correct, just
# heavy) only when it is not. The env vars MUST be set before torch is imported,
# hence find_spec here rather than torch.utils._triton.has_triton().
_HAS_TRITON = importlib.util.find_spec("triton") is not None
if not _HAS_TRITON:
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import sys
from typing import Optional

import numpy as np

from ..dtypes import DisparityResult, Progress, StereoParams, tick
from .base import StereoBackend

_HERE = os.path.dirname(os.path.abspath(__file__))
_FS_REPO = os.path.dirname(os.path.dirname(_HERE))        # …/FoundationStereo (repo root)
# Fast-FoundationStereo is cloned as a sibling of FoundationStereo.
FAST_REPO = os.path.join(os.path.dirname(_FS_REPO), "Fast-FoundationStereo")


class FastFoundationStereoBackend(StereoBackend):
    def __init__(self) -> None:
        self.model = None
        self.ckpt_path: Optional[str] = None
        self._padder_cls = None
        self._amp_dtype = None
        self._compiled = False   # set in load(): is @torch.compile actually active?

    def load(self, ckpt_path: str, params: Optional[StereoParams] = None,
             progress: Progress = None) -> None:
        if FAST_REPO not in sys.path:
            sys.path.insert(0, FAST_REPO)   # so `core.*` / `Utils` resolve to Fast-FS
        import torch
        # Triton may be installed but unusable (no compiler, wrong build) — ask
        # torch itself, and only then leave @torch.compile switched on.
        try:
            from torch.utils._triton import has_triton
            self._compiled = bool(_HAS_TRITON and has_triton())
        except Exception:
            self._compiled = False
        if not self._compiled:
            try:
                torch._dynamo.config.disable = True   # belt-and-suspenders vs the env vars
            except Exception:
                pass
        from core.utils.utils import InputPadder
        from Utils import AMP_DTYPE

        self._torch = torch
        self._padder_cls = InputPadder
        self._amp_dtype = AMP_DTYPE

        tick(progress, "Loading Fast-FoundationStereo weights…")
        # the whole model is pickled; unpickling needs the Fast-FS classes on the path
        model = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # NOTE: max_disp is set per-run in disparity() from model_params (default 192,
        # the README's runtime default). The serialized model bakes max_disp=416 (its
        # TRAINED range) — using that at inference makes the eager cost volume ~2×
        # bigger: 104 s + 28 GB vs 2 s + 13 GB at 192 on a 1332×1152 pair. 192 covers
        # any disparity a downscaled close pair produces; only raise it for <0.1 m at
        # full resolution.

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        tick(progress, f"Moving to {self._device.upper()}…")
        model = model.to(self._device).eval()
        torch.autograd.set_grad_enabled(False)

        self.model = model
        self.ckpt_path = ckpt_path
        tick(progress, "Model ready." if self._compiled else
                       "Model ready (no Triton — running eager, uses ~4× the VRAM).")

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        self.model = None
        super().unload()

    def disparity(self, img0: np.ndarray, img1: np.ndarray,
                  params: StereoParams) -> DisparityResult:
        torch = self._torch
        dev = self._device
        mp = params.model_params
        valid_iters = int(mp.get("valid_iters", 8))
        hierarchical = bool(mp.get("hierarchical", False))
        low_memory = bool(mp.get("low_memory", False))
        # cost-volume search range — the dominant cost/memory knob (see load()).
        self.model.args.max_disp = int(mp.get("max_disp", 192))

        H, W = img0.shape[:2]
        t0 = torch.as_tensor(np.ascontiguousarray(img0)).to(dev).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(np.ascontiguousarray(img1)).to(dev).float()[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        t0, t1 = t0.contiguous(), t1.contiguous()
        with torch.amp.autocast("cuda", enabled=True, dtype=self._amp_dtype):
            if hierarchical:
                d = self.model.run_hierachical(
                    t0, t1, iters=valid_iters, test_mode=True,
                    low_memory=low_memory, small_ratio=0.5,
                )
            else:
                d = self.model.forward(
                    t0, t1, iters=valid_iters, test_mode=True,
                    low_memory=low_memory, optimize_build_volume="pytorch1",
                )
        d = padder.unpad(d.float())
        if dev == "cuda":
            torch.cuda.synchronize()
        # clip(0, None) like the repo's run_demo.py: this model can emit small
        # negative disparities (low-texture / sky). Depth ignores them either way
        # (it only maps disp>0), but leaving them in skews the disparity map's
        # colour range and the exported .npy.
        disp = d.data.cpu().numpy().reshape(H, W).clip(0, None).astype(np.float32)
        return DisparityResult(disp=disp)


def make() -> StereoBackend:
    return FastFoundationStereoBackend()
