"""The backend contract + the declarative descriptors that drive the GUI.

A ``StereoBackend`` is the ONLY model-specific code: given a rectified pair it
returns a left-referenced disparity map (plus optional confidence/occlusion).
Everything else — scaling, the dual-reference flip, depth = fx·B/disp, the point
cloud, units, viewers — is model-agnostic and lives elsewhere.

The descriptors (ParamSpec / CheckpointSpec / BackendSpec) are pure data with no
torch import, so registry.py can be imported by the GUI to populate the model
dropdown, the checkpoint dropdown and a dynamically-built parameter panel
without loading any model.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..dtypes import DisparityResult, Progress, StereoParams


# --------------------------------------------------------------- descriptors
@dataclass
class ParamSpec:
    """One tunable knob, rendered dynamically into the parameter panel.

    kind: "slider" (numeric, uses minv/maxv/step/fmt/suffix),
          "toggle" (bool),
          "choice" (options: list[(value, label)])."""

    key: str                       # goes into StereoParams.model_params[key]
    label: str
    kind: str = "slider"
    default: object = 0
    minv: float = 0.0
    maxv: float = 1.0
    step: float = 1.0
    fmt: str = "{:.0f}"
    suffix: str = ""
    options: list = field(default_factory=list)   # for "choice": [(value, label), ...]
    tooltip: str = ""
    # NOTE: every model param is a 'needs run' param — the panel wires them all
    # to the stale cue unconditionally. A declared-but-never-honored needs_run
    # flag used to live here; add it back only WITH the wiring if a live model
    # param ever exists.


@dataclass
class CheckpointSpec:
    label: str                     # dropdown text, e.g. "23-51-11 · ViT-L"
    path: str                      # abs path to the .pth file (or model dir)
    note: str = ""

    def available(self) -> bool:
        return bool(self.path) and (os.path.isfile(self.path) or os.path.isdir(self.path))


@dataclass
class BackendSpec:
    """Metadata for one model — importable without torch/the model's code."""

    key: str
    display_name: str
    adapter_module: str            # module exposing make() -> StereoBackend (imported in the child)
    checkpoints: list              # list[CheckpointSpec]
    params: list                   # list[ParamSpec]
    repo_dir: Optional[str] = None   # prepended to sys.path in the child before import
    python_exe: Optional[str] = None  # interpreter for the child (None = the app's own)
    description: str = ""
    #: os.name values this backend can run on, or None for any. Checked in
    #: availability() so a backend that CANNOT work here reads as unavailable in
    #: the picker instead of looking ready and failing when the engine child
    #: tries to import its bindings.
    platforms: Optional[tuple] = None

    def default_checkpoint(self) -> Optional[CheckpointSpec]:
        for c in self.checkpoints:
            if c.available():
                return c
        return self.checkpoints[0] if self.checkpoints else None

    def param_defaults(self) -> dict:
        return {p.key: p.default for p in self.params}

    def availability(self) -> tuple[bool, str]:
        """(usable, reason) — best-effort, cheap checks only (no imports)."""
        if self.platforms and os.name not in self.platforms:
            return False, ("not available on this platform "
                           f"(needs {'/'.join(self.platforms)})")
        if self.repo_dir and not os.path.isdir(self.repo_dir):
            return False, f"repo not found: {self.repo_dir}"
        if self.python_exe and not os.path.isfile(self.python_exe):
            return False, f"interpreter not found: {self.python_exe}"
        if not any(c.available() for c in self.checkpoints):
            return False, "no weights found — download a checkpoint"
        return True, "ready"


# ------------------------------------------------------------------ contract
class StereoBackend(ABC):
    """Adapts one stereo network. Implementations live in the child process, in
    that model's own environment; keep torch imports inside the methods."""

    #: filled in by subclasses when loaded, used by the default helpers below
    _torch = None
    _device = "cpu"

    @abstractmethod
    def load(self, ckpt_path: str, params: Optional[StereoParams] = None,
             progress: Progress = None) -> None:
        """Build the network and load weights, leaving it ready on the GPU."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def disparity(self, img0: np.ndarray, img1: np.ndarray,
                  params: StereoParams) -> DisparityResult:
        """Return the (H,W) float32 disparity of img0 (the reference), + any
        native confidence/occlusion. img0/img1 are working-scale HxWx3 uint8."""

    # -- shared torch helpers (override if a backend differs) --------------
    def unload(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def release_cache(self) -> None:
        """Return the caching allocator's spare blocks to the driver after a run.

        PyTorch keeps a run's whole PEAK reserved for reuse. That is the right
        default on a big card, but here a single run leaves the 12 GB 3060
        looking full (measured, 1332×1152: FoundationStereo reserves 10.27 GB,
        S²M² 4.38 GB) — nothing is left for another app, and the reservation
        makes the next run fight for memory. empty_cache() gives nearly all of it
        back (FS: 10.27 → 1.42 GB); the model itself stays resident, so the only
        cost is re-allocating the scratch on the next run (tens of ms).
        """
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def device_name(self) -> str:
        t = self._torch
        if t is None:
            try:
                import torch
                t = torch
            except Exception:
                return "unknown"
        return t.cuda.get_device_name(0) if t.cuda.is_available() else "CPU"

    def reset_peak_vram(self) -> None:
        """Zero the peak-allocation counter so the next run's peak is measurable."""
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()

    # Everything below reports GiB (÷2**30) but is LABELLED "GB", because that is
    # what nvidia-smi, Task Manager and NVIDIA's own marketing all do: a "12 GB"
    # 3060 is 12288 MiB = 12.00 GiB = 12.88 decimal GB. Dividing by 1e9 and
    # rounding printed "13 GB" for a 12 GB card — a whole gigabyte of headroom
    # that does not exist.
    def peak_vram_gb(self) -> float:
        """Highest VRAM this run RESERVED from the driver, since reset_peak_vram().

        RESERVED, not allocated. max_memory_allocated() counts live tensor bytes,
        which is far below what the card actually had to give up: measured at
        1332×1152, FoundationStereo allocates 8.5 but reserves 10.3. The driver
        cannot hand a reservation to anything else, so the reservation is what
        decides whether a run fits — and it is what the Compare column has to show.

        Reporting the allocated figure said "8.5 GB" on a card that was 96% full,
        which read as ~4 GB spare. There was ~0.5 GB. Nothing warned, because by
        this measure nothing looked wrong; the run just quietly spilled to system
        RAM and took 30× longer.
        """
        if self._torch is None or not self._torch.cuda.is_available():
            return 0.0
        return float(self._torch.cuda.max_memory_reserved()) / 2 ** 30

    def vram_gb(self):
        """(used, total) in GiB across the WHOLE card — other apps included;
        (0,0) if no CUDA."""
        t = self._torch
        if t is None or not t.cuda.is_available():
            return (0.0, 0.0)
        free, total = t.cuda.mem_get_info()
        return ((total - free) / 2 ** 30, total / 2 ** 30)
