"""FoundationStereo backend — the reference / highest-accuracy model.

The load + disparity logic is the original StereoEngine code, unchanged, now
behind the StereoBackend interface. torch and the repo's ``core.*`` are imported
lazily inside the methods, so importing this module (e.g. for the registry) is
cheap and torch-free. The BackendSpec metadata lives in registry.py.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

from ..dtypes import DisparityResult, Progress, StereoParams, tick
from .base import StereoBackend

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))   # studio/backends/ -> repo root


class FoundationStereoBackend(StereoBackend):
    def __init__(self) -> None:
        self.model = None
        self.ckpt_path: Optional[str] = None
        self._padder_cls = None

    def load(self, ckpt_path: str, params: Optional[StereoParams] = None,
             progress: Progress = None) -> None:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        import torch
        from omegaconf import OmegaConf
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        self._torch = torch
        self._padder_cls = InputPadder

        cfg_path = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
        tick(progress, "Reading model config…")
        cfg = OmegaConf.load(cfg_path)
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"

        tick(progress, "Building network…")
        model = FoundationStereo(cfg)

        tick(progress, "Loading weights (~3 GB)…")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model"])

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        tick(progress, f"Moving to {self._device.upper()}…")
        model = model.to(self._device).eval()
        torch.autograd.set_grad_enabled(False)

        self.model = model
        self.ckpt_path = ckpt_path
        tick(progress, "Model ready.")

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
        valid_iters = int(mp.get("valid_iters", 32))
        hierarchical = bool(mp.get("hierarchical", False))
        mixed = bool(mp.get("mixed_precision", True))
        low_memory = bool(mp.get("low_memory", False))
        # honor the mixed-precision toggle inside the model's internal autocasts
        try:
            self.model.args["mixed_precision"] = mixed
        except Exception:
            pass
        H, W = img0.shape[:2]
        t0 = torch.as_tensor(np.ascontiguousarray(img0)).to(dev).float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(np.ascontiguousarray(img1)).to(dev).float()[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        t0, t1 = t0.contiguous(), t1.contiguous()
        with torch.amp.autocast("cuda", enabled=mixed):
            if hierarchical:
                d = self.model.run_hierachical(
                    t0, t1, iters=valid_iters, test_mode=True, low_memory=low_memory,
                )
            else:
                d = self.model.forward(
                    t0, t1, iters=valid_iters, test_mode=True, low_memory=low_memory,
                )
        d = padder.unpad(d.float())
        if dev == "cuda":
            torch.cuda.synchronize()
        disp = d.data.cpu().numpy().reshape(H, W).astype(np.float32)
        return DisparityResult(disp=disp)


def make() -> StereoBackend:
    return FoundationStereoBackend()
