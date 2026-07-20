"""S²M² backend — Scalable Stereo Matching Model (ICCV 2025), Junhong Min et al.

A clean ``s2m2.*`` package (no ``core.*``/``Utils`` name collision) whose model
package lives under the repo's ``src/``. Runs in the app's OWN venv (torch 2.7
loads and runs it fine, despite the repo recommending 2.9; the only exotic op is
``F.scaled_dot_product_attention`` — standard PyTorch — and torch.compile is
opt-in, so no Triton). The network natively emits disparity + occlusion +
CONFIDENCE; we pass the confidence through the pipeline's reliability channel.

The four sizes S/M/L/XL are separate checkpoints; feature_channels /
num_transformer are encoded in the filename ``CH{ch}NTR{ntr}.pth``.

Non-commercial research/education license (per the S²M² repo).
"""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

import numpy as np

from ..dtypes import DisparityResult, Progress, StereoParams, tick
from .base import StereoBackend

_HERE = os.path.dirname(os.path.abspath(__file__))
_FS_REPO = os.path.dirname(os.path.dirname(_HERE))       # …/FoundationStereo (repo root)
# S²M² is cloned beside FoundationStereo; its importable package lives under src/.
S2M2_REPO = os.path.join(os.path.dirname(_FS_REPO), "s2m2")
S2M2_SRC = os.path.join(S2M2_REPO, "src")


class S2M2Backend(StereoBackend):
    def __init__(self) -> None:
        self.model = None
        self.ckpt_path: Optional[str] = None
        self._run = None
        self._dev = None

    def load(self, ckpt_path: str, params: Optional[StereoParams] = None,
             progress: Progress = None) -> None:
        if S2M2_SRC not in sys.path:
            sys.path.insert(0, S2M2_SRC)   # so `import s2m2` resolves to this repo
        import torch
        from s2m2.core.model.s2m2 import S2M2
        from s2m2.core.utils.model_utils import run_stereo_matching

        self._torch = torch
        self._run = run_stereo_matching

        # feature_channels / num_transformer are encoded in the filename
        mm = re.search(r"CH(\d+)NTR(\d+)", os.path.basename(ckpt_path))
        if not mm:
            raise ValueError(f"can't parse S2M2 config from {os.path.basename(ckpt_path)!r}")
        ch, ntr = int(mm.group(1)), int(mm.group(2))

        tick(progress, f"Building S²M² (CH{ch}·NTR{ntr})…")
        model = S2M2(feature_channels=ch, dim_expansion=1, num_transformer=ntr,
                     use_positivity=True, refine_iter=3)

        tick(progress, "Loading weights…")
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        model.my_load_state_dict(ckpt["state_dict"])

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dev = torch.device(self._device)
        tick(progress, f"Moving to {self._device.upper()}…")
        model = model.to(self._dev).eval()
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
        mp = params.model_params
        self.model.refine_iter = int(mp.get("refine_iter", 3))   # read at forward time
        H, W = img0.shape[:2]
        # the model normalizes 0..255 internally; feed uint8 like the S²M² demo
        lt = torch.from_numpy(np.ascontiguousarray(img0)).permute(-1, 0, 1).unsqueeze(0).to(self._dev)
        rt = torch.from_numpy(np.ascontiguousarray(img1)).permute(-1, 0, 1).unsqueeze(0).to(self._dev)
        # run_stereo_matching pads to /32, runs in fp16 autocast, crops back to (H,W)
        disp, occ, conf, _score, _t = self._run(self.model, lt, rt, self._dev, N_repeat=1)
        disp = disp.detach().float().cpu().numpy().reshape(H, W).astype(np.float32)
        conf = conf.detach().float().cpu().numpy().reshape(H, W).astype(np.float32)
        occ = occ.detach().float().cpu().numpy().reshape(H, W)
        # S²M² occ: 0 = occluded → our convention (True = occluded)
        occluded = occ < 0.5
        return DisparityResult(disp=disp, confidence=conf, occlusion=occluded)


def make() -> StereoBackend:
    return S2M2Backend()
