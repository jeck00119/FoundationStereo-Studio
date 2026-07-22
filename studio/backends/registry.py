"""Backend registry — pure metadata, safe to import from the GUI (no torch, no
model code). Adapter modules are imported only in the engine child via
``load_backend``, so a backend whose deps live in another venv never has to be
importable from the app's own interpreter.

To add a model: append a BackendSpec here + drop an adapter module (make() ->
StereoBackend) next to this file. Nothing else in the app needs to change.
"""
from __future__ import annotations

import importlib
import os
import sys

from .base import BackendSpec, CheckpointSpec, ParamSpec, StereoBackend

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))


FOUNDATION_STEREO = BackendSpec(
    key="foundation_stereo",
    display_name="FoundationStereo · ViT-L",
    adapter_module="studio.backends.foundation_stereo",
    repo_dir=REPO_ROOT,
    python_exe=None,   # runs in the app's own venv
    checkpoints=[
        CheckpointSpec(
            "23-51-11 · ViT-L",
            os.path.join(REPO_ROOT, "pretrained_models", "23-51-11", "model_best_bp2.pth"),
        ),
    ],
    params=[
        ParamSpec("valid_iters", "Refine iters", "slider", 32, 1, 64, 1, "{:.0f}", "",
                  tooltip="Number of refinement steps the network runs. More = slightly "
                          "cleaner edges but slower. 32 is a good default; ~16 is faster."),
        ParamSpec("hierarchical", "Hierarchical (>1K)", "toggle", False,
                  tooltip="For very large images (>1000 px): process in tiles to fit memory. "
                          "Slower — leave OFF unless full resolution runs out of VRAM."),
        ParamSpec("mixed_precision", "Mixed precision", "toggle", True,
                  tooltip="Fast 16-bit GPU math. Keep ON — faster and lighter, no visible "
                          "quality loss."),
        ParamSpec("low_memory", "Low memory", "toggle", False,
                  tooltip="Reduce GPU memory at the cost of speed. Turn ON only if you get "
                          "out-of-memory errors."),
    ],
    description="NVIDIA FoundationStereo — highest accuracy, zero-shot generalization.",
)


# Fast-FoundationStereo — cloned as a sibling of this repo (…/Desktop/Fast-FoundationStereo).
FAST_FS_REPO = os.path.join(os.path.dirname(REPO_ROOT), "Fast-FoundationStereo")

# The runs the readme's trade-off table documents, slowest/most-accurate first.
# The readme's runtimes (49 / 44 / 38 ms) are deliberately NOT in these labels.
# They are its own figures at 640×480 on a 3090 at valid_iters 8, so they only ever
# gave a RELATIVE ordering — and that ordering is the words themselves. Printed as
# absolutes they landed on the Compare column directly above a MEASURED "2.19 s" on
# this machine at 1332×1152, reading as a 44× contradiction with no way to tell
# which number to believe. The card times the run on your hardware; that beats a
# different GPU's readme figure at every job the label had.
# NOTE: the weights' cfg.yaml says ``vit_size: vitl``, but that is a vestigial
# field naming the ViT-L *teacher* these were distilled FROM. The student is an
# EdgeNeXt-based 14.6M-param network (model_card.md) — which the 59-68 MB
# checkpoints confirm (14.6M × 4 B ≈ 58 MB); a real ViT-L is 3.1 GB.
# hf-c-release = NVIDIA's own re-release on Hugging Face
# (nvidia/c-fast-foundationstereo, NVIDIA Open Model Agreement) — the
# "commercial version" the author announced in NVlabs/Fast-FoundationStereo
# issue #53. Its model card pins the tier: "v1.0 … full capabilities,
# UNPRUNED" — the pruned NAS variants are the faster Drive runs, so unpruned
# is the flagship (23-36-37) accuracy class; same training corpus per the
# card, top-of-family file size (67.8 MB of the 59–68 MB span), and measured
# on-device speed in the heavy-run class (Orin bring-up, 2026-07-22). Listed
# first: best accuracy AND the only run a fresh device can fetch without
# fighting Drive's download quota —
#   huggingface.co/nvidia/c-fast-foundationstereo/resolve/main/
#     {cfg.yaml, model_best_bp2_serialize.pth}
# Bit-identity with Drive's 23-36-37 is unverified, so it stays its own
# entry rather than being renamed into that slot.
_FAST_RUNS = [
    ("hf-c-release", "unpruned flagship · official HF release"),
    ("23-36-37", "most accurate (Drive)"),
    ("20-26-39", "balanced"),
    ("20-30-48", "fastest"),
    ("15-44-51", "extra run · not in the readme table"),
]


def _fast_ckpt(run: str, note: str) -> CheckpointSpec:
    return CheckpointSpec(
        f"{run} · {note}",
        os.path.join(FAST_FS_REPO, "weights", run, "model_best_bp2_serialize.pth"),
    )


FAST_FOUNDATION_STEREO = BackendSpec(
    key="fast_foundation_stereo",
    display_name="Fast-FoundationStereo · EdgeNeXt 14.6M",
    adapter_module="studio.backends.fast_foundation_stereo",
    repo_dir=FAST_FS_REPO,
    python_exe=None,   # runs in the app's own venv (torch 2.7 loads it, no Triton)
    checkpoints=[_fast_ckpt(run, note) for run, note in _FAST_RUNS],
    params=[
        ParamSpec("valid_iters", "Refine iters", "slider", 8, 1, 32, 1, "{:.0f}", "",
                  tooltip="Refinement steps. Fast-FoundationStereo is pruned for few "
                          "iterations — 8 is the trained default; more barely helps."),
        ParamSpec("max_disp", "Max disparity", "slider", 192, 64, 416, 32, "{:.0f}", " px",
                  tooltip="Disparity search range, and the model's main memory/speed knob — "
                          "GPU memory scales roughly with pixels × this value. 192 px is the "
                          "trained default and plenty for downscaled or normal pairs; only "
                          "raise it to sense very near objects (<0.1 m) at full resolution, "
                          "and drop it (e.g. 128) if you run out of memory."),
        ParamSpec("hierarchical", "Hierarchical (>1K)", "toggle", False,
                  tooltip="For very large images (>1000 px): coarse-to-fine in two passes "
                          "to fit memory. Slower — leave OFF unless you hit out-of-memory."),
        ParamSpec("low_memory", "Low memory", "toggle", False,
                  tooltip="Trades speed for memory inside the refinement lookups only. "
                          "Measured effect on PEAK memory here: none — the cost volume "
                          "dominates. If you hit out-of-memory, lower Scale or Max disparity "
                          "instead."),
    ],
    description="NVIDIA Fast-FoundationStereo (CVPR 2026) — real-time-oriented: an EdgeNeXt "
                "student (14.6M params) distilled from FoundationStereo, >10× faster at close "
                "to its zero-shot accuracy. Uses the repo's pure-PyTorch cost volume, compiled "
                "via Triton — which is what keeps its memory near the paper's figures. "
                "Research/evaluation use.",
)


# S²M² (ICCV 2025) — cloned beside this repo; its package lives under src/.
S2M2_REPO = os.path.join(os.path.dirname(REPO_ROOT), "s2m2")
S2M2_SRC = os.path.join(S2M2_REPO, "src")


def _s2m2_ckpt(fname: str, label: str) -> CheckpointSpec:
    return CheckpointSpec(label, os.path.join(S2M2_REPO, "weights", "pretrain_weights", fname))


S2M2 = BackendSpec(
    key="s2m2",
    display_name="S²M² · Scalable Stereo",
    adapter_module="studio.backends.s2m2",
    repo_dir=S2M2_SRC,          # the importable `s2m2` package lives under src/
    python_exe=None,            # runs in the app's own venv (torch 2.7, no Triton)
    checkpoints=[
        _s2m2_ckpt("CH256NTR3.pth", "L · CH256  (recommended)"),
        _s2m2_ckpt("CH384NTR3.pth", "XL · CH384  (best)"),
        _s2m2_ckpt("CH192NTR2.pth", "M · CH192  (faster)"),
        _s2m2_ckpt("CH128NTR1.pth", "S · CH128  (fastest)"),
    ],
    params=[
        ParamSpec("refine_iter", "Refine iters", "slider", 3, 1, 8, 1, "{:.0f}", "",
                  tooltip="Local iterative refinement steps. 3 is the trained default; more "
                          "can sharpen edges a little at some cost to speed."),
    ],
    description="S²M² — Scalable Stereo Matching (ICCV 2025). Joint disparity + occlusion + "
                "confidence; the paper reports 1st on ETH3D / Middlebury / Booster. These "
                "public weights are the Booster-benchmark version (its refinement module is a "
                "UNet, swapped in for stable ONNX export) — not the exact Middlebury/ETH3D "
                "entry. Non-commercial research/education use only.",
)


BACKENDS: dict[str, BackendSpec] = {
    b.key: b for b in (FOUNDATION_STEREO, FAST_FOUNDATION_STEREO, S2M2)
}
DEFAULT_BACKEND = "foundation_stereo"


def get_spec(key: str) -> BackendSpec | None:
    return BACKENDS.get(key)


def load_backend(key: str) -> StereoBackend:
    """Import the backend's adapter (in THIS process — i.e. the engine child, in
    that backend's environment) and instantiate it. Prepends the backend's
    repo_dir so the model's own modules import."""
    spec = BACKENDS[key]
    if spec.repo_dir and spec.repo_dir not in sys.path:
        sys.path.insert(0, spec.repo_dir)
    mod = importlib.import_module(spec.adapter_module)
    return mod.make()
