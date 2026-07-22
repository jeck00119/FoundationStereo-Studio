"""Fast-FoundationStereo · TensorRT backend (Jetson-first).

Wraps upstream's single-ONNX export (``scripts/make_single_onnx.py``) and a
minimal TensorRT-10 runner modeled on ``scripts/run_demo_single_trt.py``
(both Apache-2.0). Engines are size-specific, so they are built LAZILY per
(padded working size, valid_iters, max_disp) the first time a run needs one —
several minutes, once per config per device — and cached on disk next to the
checkpoint (``weights/<run>/trt/``). Unlike the torch backend's triton
warm-up, the cache survives reboots: after the one-time build this backend
starts cold in seconds.

The export strips ImageNet normalization out of the graph (see upstream's
docstring), so this runner owes ``(pixel - mean) / std`` to every input —
constants below are upstream's, in 0-255 scale.
"""
from __future__ import annotations

import os

# The ONNX export runs a real traced forward through the EAGER cost volume —
# dynamo/inductor must stay out of it (mirrors upstream make_single_onnx.py).
# Module level on purpose: it has to land before torch's first import in this
# engine child, and this backend never wants compilation anyway.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import subprocess
import sys
import time
from shutil import which
from typing import Optional

import numpy as np

from ..dtypes import DisparityResult, Progress, StereoParams, tick
from .base import StereoBackend
from .fast_foundation_stereo import backfill_pickled_args

_HERE = os.path.dirname(os.path.abspath(__file__))
_FS_REPO = os.path.dirname(os.path.dirname(_HERE))
FAST_REPO = os.path.join(os.path.dirname(_FS_REPO), "Fast-FoundationStereo")

# upstream's ImageNet constants (0-255 scale) — the graph no longer applies
# them, the runner must (make_single_onnx.py strips normalize_image).
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _find_trtexec() -> Optional[str]:
    # JetPack installs trtexec outside PATH; check its home first.
    fixed = "/usr/src/tensorrt/bin/trtexec"
    if os.access(fixed, os.X_OK):
        return fixed
    return which("trtexec")


def _trt_version() -> str:
    import tensorrt as trt
    return ".".join(trt.__version__.split(".")[:2])


class _EngineRunner:
    """One deserialized engine + execution context, torch tensors as buffers.

    Distilled from upstream scripts/run_demo_single_trt.py
    (SingleEngineTrtRunner, Apache-2.0): set_input_shape → allocate outputs →
    set_tensor_address → execute_async_v3 on torch's current stream.
    """

    def __init__(self, engine_path: str, torch_mod) -> None:
        import tensorrt as trt
        self._trt = trt
        self._torch = torch_mod
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Could not deserialize {os.path.basename(engine_path)} — it was "
                f"likely built by a different TensorRT version than the installed "
                f"{trt.__version__}. Delete the file and rerun; it rebuilds.")
        self.context = self.engine.create_execution_context()

    def _dtype(self, dt):
        trt, torch = self._trt, self._torch
        table = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                 trt.DataType.INT32: torch.int32, trt.DataType.INT8: torch.int8,
                 trt.DataType.BOOL: torch.bool}
        bf16 = getattr(trt.DataType, "BF16", None)
        if bf16 is not None:
            table[bf16] = torch.bfloat16
        if dt not in table:
            raise RuntimeError(f"unsupported TRT dtype {dt}")
        return table[dt]

    def infer(self, inputs: dict) -> dict:
        trt, torch = self._trt, self._torch
        for name, t in inputs.items():
            want = self._dtype(self.engine.get_tensor_dtype(name))
            if t.dtype != want:
                t = t.to(want)
            inputs[name] = t.contiguous()
            self.context.set_input_shape(name, tuple(inputs[name].shape))
        outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(self.context.get_tensor_shape(name))
            outputs[name] = torch.empty(
                shape, device="cuda", dtype=self._dtype(self.engine.get_tensor_dtype(name)))
        for name, t in {**inputs, **outputs}.items():
            self.context.set_tensor_address(name, int(t.data_ptr()))
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 returned False")
        return outputs


class FastFsTrtBackend(StereoBackend):
    def __init__(self) -> None:
        self.ckpt_path: Optional[str] = None
        self._runner: Optional[_EngineRunner] = None
        self._runner_key = None
        self._padder_cls = None
        self._progress: Progress = None   # the child's socket-progress closure,
                                          # kept so lazy engine builds can narrate

    # ------------------------------------------------------------------ load
    def load(self, ckpt_path: str, params: Optional[StereoParams] = None,
             progress: Progress = None) -> None:
        if FAST_REPO not in sys.path:
            sys.path.insert(0, FAST_REPO)
        import torch
        try:
            import tensorrt  # noqa: F401 — fail HERE with a useful message
        except ImportError as e:
            raise RuntimeError(
                "TensorRT python bindings are not visible in this environment. "
                "On Jetson, re-run ./setup_jetson.sh (it links the system "
                "bindings into the venv); this backend is Jetson/Linux-only."
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError("The TensorRT backend needs the CUDA device.")
        from core.utils.utils import InputPadder
        self._torch = torch
        self._device = "cuda"
        self._padder_cls = InputPadder
        self._progress = progress
        torch.autograd.set_grad_enabled(False)
        self.ckpt_path = ckpt_path
        tick(progress, "TensorRT backend ready — engines build lazily per "
                       "input size (first run per size takes minutes, then "
                       "it is cached on disk for good).")

    @property
    def is_loaded(self) -> bool:
        return self.ckpt_path is not None

    def unload(self) -> None:
        self._runner = None
        self._runner_key = None
        super().unload()

    # --------------------------------------------------- engine provisioning
    def _engine_path(self, hp: int, wp: int, iters: int, max_disp: int) -> str:
        d = os.path.join(os.path.dirname(self.ckpt_path), "trt")
        return os.path.join(
            d, f"fp16_{hp}x{wp}_i{iters}_d{max_disp}_trt{_trt_version()}.engine")

    def _export_onnx(self, hp: int, wp: int, iters: int, max_disp: int,
                     onnx_path: str) -> None:
        torch = self._torch
        import core.foundation_stereo as _fs_module
        from scripts.make_single_onnx import (
            FastFoundationStereoSingleOnnx, _build_concat_volume_onnx,
            _build_gwc_volume_onnx)

        tick(self._progress, f"TRT: exporting ONNX at {wp}×{hp} on the CPU "
                             f"(one-time; slower but memory-safe)…")
        model = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        backfill_pickled_args(model)          # HF drop predates args.normalize
        model.args.max_disp = int(max_disp)
        model.args.valid_iters = int(iters)
        model.args.mixed_precision = False
        # CPU ON PURPOSE — this is a lesson written in a reboot. The tracer
        # keeps every intermediate of the eager cost volume alive; at rig
        # sizes that is >5 GB, and on the GPU those pages are nvmap-PINNED —
        # unswappable, unreclaimable. The first rig-size export exhausted the
        # box until PID 1 missed the Tegra watchdog's 2-minute deadline and
        # the hardware reset the system (2026-07-22 14:25). On the CPU the
        # same bytes are ordinary swappable RAM and the 16 GB swapfile
        # absorbs the spike; the one-time export merely takes minutes.
        model = model.cpu().eval()
        wrapper = FastFoundationStereoSingleOnnx(model).cpu().eval()

        # the same monkey-patches upstream's __main__ applies before tracing
        _fs_module.normalize_image = lambda img: img
        _fs_module.build_gwc_volume_optimized_pytorch1 = _build_gwc_volume_onnx
        _fs_module.build_concat_volume_optimized_pytorch1 = _build_concat_volume_onnx

        left = torch.randn(1, 3, hp, wp, device="cpu")
        right = torch.randn(1, 3, hp, wp, device="cpu")
        kwargs = dict(opset_version=17,
                      input_names=["left_image", "right_image"],
                      output_names=["disparity"],
                      do_constant_folding=True)
        try:
            # torch ≥2.9 defaults to the dynamo exporter; upstream's export is
            # written for the tracer. Older torch lacks the kwarg — retry bare.
            torch.onnx.export(wrapper, (left, right), onnx_path,
                              dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(wrapper, (left, right), onnx_path, **kwargs)
        finally:
            del wrapper, model, left, right
            torch.cuda.empty_cache()

    def _build_engine(self, onnx_path: str, engine_path: str) -> None:
        exe = _find_trtexec()
        if exe is None:
            raise RuntimeError(
                "trtexec not found (looked in /usr/src/tensorrt/bin and PATH) — "
                "is the JetPack TensorRT package installed?")
        tick(self._progress, "TRT: building the FP16 engine — several minutes, "
                             "one-time for this size; cached on disk afterwards…")
        t0 = time.time()
        # workspace cap: on 8 GB unified memory an uncapped builder times
        # tactics against ALL free device memory and can OOM the box at rig
        # sizes; 3 GiB keeps builds safe and costs at most a marginal tactic.
        r = subprocess.run(
            [exe, f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16",
             "--memPoolSize=workspace:3072MiB"],
            capture_output=True, text=True)
        if r.returncode != 0 or not os.path.isfile(engine_path):
            tail = "\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-15:])
            raise RuntimeError(f"trtexec failed after {time.time() - t0:.0f}s — "
                               f"log tail:\n{tail}")
        tick(self._progress, f"TRT: engine built in {time.time() - t0:.0f}s.")

    def _ensure_runner(self, hp: int, wp: int, iters: int, max_disp: int) -> None:
        key = (hp, wp, iters, max_disp)
        if self._runner is not None and self._runner_key == key:
            return
        engine_path = self._engine_path(hp, wp, iters, max_disp)
        if not os.path.isfile(engine_path):
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            onnx_path = engine_path.replace(".engine", ".onnx")
            try:
                self._export_onnx(hp, wp, iters, max_disp, onnx_path)
                self._build_engine(onnx_path, engine_path)
            finally:
                # the .onnx is only scaffolding for trtexec; a failed build
                # must not leave half-artifacts that mask the error next run
                if os.path.isfile(onnx_path):
                    os.remove(onnx_path)
                if os.path.isfile(engine_path) and os.path.getsize(engine_path) == 0:
                    os.remove(engine_path)
        # one engine resident at a time — each holds weights + workspace on an
        # 8 GB unified-memory device
        self._runner = None
        self._torch.cuda.empty_cache()
        tick(self._progress, "TRT: loading engine…")
        self._runner = _EngineRunner(engine_path, self._torch)
        self._runner_key = key

    # ------------------------------------------------------------- disparity
    def disparity(self, img0: np.ndarray, img1: np.ndarray,
                  params: StereoParams) -> DisparityResult:
        torch = self._torch
        mp = params.model_params
        iters = int(mp.get("valid_iters", 8))
        max_disp = int(mp.get("max_disp", 192))

        H, W = img0.shape[:2]
        t0 = torch.as_tensor((img0[..., :3].astype(np.float32) - _MEAN) / _STD)
        t1 = torch.as_tensor((img1[..., :3].astype(np.float32) - _MEAN) / _STD)
        t0 = t0.to("cuda")[None].permute(0, 3, 1, 2)
        t1 = t1.to("cuda")[None].permute(0, 3, 1, 2)
        padder = self._padder_cls(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        hp, wp = int(t0.shape[-2]), int(t0.shape[-1])

        self._ensure_runner(hp, wp, iters, max_disp)
        out = self._runner.infer({"left_image": t0.contiguous(),
                                  "right_image": t1.contiguous()})
        d = padder.unpad(out["disparity"].float())
        torch.cuda.synchronize()
        # clip(0) for the same reason as the torch adapter: small negative
        # disparities skew the colour range / exported .npy
        disp = d.data.cpu().numpy().reshape(H, W).clip(0, None).astype(np.float32)
        return DisparityResult(disp=disp)

    # TRT allocates the engine's activation memory OUTSIDE torch's allocator,
    # so the inherited reserved-bytes figure alone would under-report what the
    # device really gave up — add what the engine declares it holds.
    def peak_vram_gb(self) -> float:
        base = super().peak_vram_gb()
        if self._runner is not None:
            base += float(getattr(self._runner.engine, "device_memory_size", 0)) / 2 ** 30
        return base


def make() -> StereoBackend:
    return FastFsTrtBackend()
