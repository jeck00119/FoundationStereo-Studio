"""ROI crop + disparity pre-shift ground truth.

The pre-shift is pure geometry bookkeeping — crop the right side Δ px further
left, get back d−Δ, add Δ again — and every step of it is an off-by-Δ waiting to
happen. So the backend here does REAL correspondence: each pixel encodes its own
absolute column, and the fake network recovers disparity by matching those ids.
A wrong crop origin or a missed un-shift therefore shows up as a wrong disparity,
not as a test that still passes because both sides made the same mistake.

The crop is INPUT PREPARATION (rectify.crop_pair / Rectifier.rectify_roi), not
part of inference — it has to happen before the engine socket, or a whole 73 MB
frame crosses it per pair. So these tests prepare the pair the way the app does
and hand run_inference the prepared images.
"""
import numpy as np
import pytest

from studio.backends.base import StereoBackend
from studio.cloud import build_cloud
from studio.dtypes import DisparityResult, StereoParams
from studio.infer import run_inference
from studio.rectify import crop_pair, roi_rects

FULL_W, FULL_H = 1200, 300
FX = 2000.0
BASELINE = 5.0
D_TRUE = 425          # absolute disparity, like the real macro rig
ROI = (400, 60, 512, 128)     # x0, y0, w, h
SHIFT = 400                   # Δ — leaves 25 px of observed disparity


def _encode(cols, H):
    """(H,W,3) uint8 whose every pixel encodes its ABSOLUTE source column."""
    u = np.asarray(cols, np.int64)
    img = np.zeros((H, len(u), 3), np.uint8)
    img[:, :, 0] = (u >> 8) & 255
    img[:, :, 1] = u & 255
    return img


def _decode(img):
    return (img[..., 0].astype(np.int64) << 8) | img[..., 1].astype(np.int64)


def _scene():
    """left[u] carries id u; right[v] carries id v+d — i.e. the scene point seen
    at right column v sits at left column v+d, which IS disparity d."""
    return (_encode(np.arange(FULL_W), FULL_H),
            _encode(np.arange(FULL_W) + D_TRUE, FULL_H))


class CorrespondenceBackend(StereoBackend):
    """Matches the encoded ids exactly and reports the disparity it OBSERVES."""

    def load(self, ckpt_path, params=None, progress=None):
        pass

    @property
    def is_loaded(self):
        return True

    def disparity(self, img0, img1, params):
        H, W = img0.shape[:2]
        lid, rid = _decode(img0)[0], _decode(img1)[0]     # rows are identical
        j = np.searchsorted(rid, lid)
        jj = np.clip(j, 0, W - 1)
        good = (j < W) & (rid[jj] == lid)
        d = np.zeros((H, W), np.float32)
        d[:, good] = (np.arange(W) - jj)[good].astype(np.float32)
        return DisparityResult(disp=d)


def _params(**kw):
    p = StereoParams(scale=1.0, fx=FX, fy=FX, cx=FULL_W / 2, cy=FULL_H / 2,
                     baseline=BASELINE, z_near=0.0, z_far=1e9,
                     remove_invisible=False, denoise=False)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _run(params):
    """Prepare the pair as the app does, then infer — the real two-step chain."""
    left, right = crop_pair(*_scene(), params)
    return run_inference(CorrespondenceBackend(), left, right, params)


# ------------------------------------------------------------------ geometry
def test_preshift_recovers_true_disparity():
    """The network sees d−Δ; run_inference must hand back d."""
    r = _run(_params(roi=ROI, disp_shift=SHIFT))
    assert r.disp.shape == (ROI[3], ROI[2])
    assert r.disp_offset == pytest.approx(SHIFT)
    matched = r.disp[:, D_TRUE - SHIFT:]
    assert np.allclose(matched, D_TRUE), matched[0, :5]


def test_preshift_depth_matches_uncropped():
    """Same physical plane, so the cropped+shifted depth equals the plain one."""
    full = _run(_params())
    crop = _run(_params(roi=ROI, disp_shift=SHIFT))
    z_full = full.depth[full.depth > 0]
    z_crop = crop.depth[crop.depth > 0]
    assert z_full.size and z_crop.size
    assert np.allclose(z_crop.mean(), z_full.mean(), rtol=1e-6)
    assert np.allclose(z_crop.mean(), FX * BASELINE / D_TRUE, rtol=1e-6)


def test_roi_shifts_principal_point_only():
    p = _params(roi=ROI, disp_shift=SHIFT)
    K = p.intrinsics(1.0)
    assert K[0, 0] == pytest.approx(FX) and K[1, 1] == pytest.approx(FX)
    assert K[0, 2] == pytest.approx(FULL_W / 2 - ROI[0])
    assert K[1, 2] == pytest.approx(FULL_H / 2 - ROI[1])
    assert p.intrinsics(0.5)[0, 2] == pytest.approx((FULL_W / 2 - ROI[0]) * 0.5)


def test_cropped_cloud_lands_in_the_same_world_frame():
    """The whole point of shifting cx/cy: a box placed on an uncropped run must
    still sit on the same pin after switching to the ROI."""
    pf, pc = _params(), _params(roi=ROI, disp_shift=SHIFT)
    cf = build_cloud(_run(pf), pf)
    cc = build_cloud(_run(pc), pc)
    assert cf is not None and cc is not None and cc.n > 0
    lo, hi = cc.points.min(0), cc.points.max(0)
    assert lo[0] >= cf.points[:, 0].min() - 1e-3
    assert hi[0] <= cf.points[:, 0].max() + 1e-3
    assert lo[1] >= cf.points[:, 1].min() - 1e-3
    assert hi[1] <= cf.points[:, 1].max() + 1e-3
    assert np.allclose(cc.points[:, 2].mean(), cf.points[:, 2].mean(), rtol=1e-6)


# -------------------------------------------------------------- edge cases
def test_unmatched_pixels_stay_invalid_through_the_shift():
    """disp==0 means 'no match'. Adding Δ to it would invent a depth."""
    r = _run(_params(roi=ROI, disp_shift=SHIFT))
    dead = r.disp[:, :D_TRUE - SHIFT]
    assert dead.size and np.all(dead == 0)
    assert np.all(r.depth[:, :D_TRUE - SHIFT] == 0)


def test_remove_invisible_does_not_eat_the_preshifted_roi():
    """Testing TRUE disparity here would delete the whole 512-px ROI (d=425)."""
    p = _params(roi=ROI, disp_shift=SHIFT, remove_invisible=True)
    c = build_cloud(_run(p), p)
    assert c is not None
    assert c.n > 0.9 * ROI[2] * ROI[3] * (1 - (D_TRUE - SHIFT) / ROI[2])


def test_dual_reference_with_preshift_is_refused():
    with pytest.raises(ValueError, match="pre-shift"):
        _run(_params(roi=ROI, disp_shift=SHIFT, dual_reference=True))


def test_roi_without_shift_still_works():
    """Plain crop, no pre-shift — the naive mode must stay correct."""
    r = _run(_params(roi=(600, 0, 512, 64)))
    assert r.disp_offset == 0.0
    matched = r.disp[r.disp > 0]
    assert matched.size and np.allclose(matched, D_TRUE)


# --------------------------------------- the clamp: one source of truth
def test_effective_shift_is_clamped_to_x0():
    assert _params().effective_shift == 0.0
    assert _params(roi=ROI, disp_shift=SHIFT).effective_shift == pytest.approx(SHIFT)
    # Δ past x0 would crop before column 0
    assert _params(roi=(30, 0, 128, 64), disp_shift=SHIFT).effective_shift == 30.0
    assert _params(roi=ROI, disp_shift=-5).effective_shift == 0.0


def test_clamped_shift_keeps_both_crops_the_same_size():
    p = _params(roi=(30, 0, 128, 64), disp_shift=SHIFT)
    r = _run(p)
    assert r.disp_offset == pytest.approx(30.0)
    assert r.disp.shape == (64, 128)
    left, right = crop_pair(*_scene(), p)
    assert left.shape == right.shape        # a width mismatch would reach the net


def test_roi_rects_never_returns_a_negative_origin():
    """numpy wraps a negative slice start SILENTLY — measured: a right crop at
    rx=-216 comes back 0 columns wide, remapping the wrong region with no error."""
    for x0 in (0, 5, 30, 399, 400):
        for shift in (0, 50, 400, 5000):
            (lx, ly, lw, lh), (rx, ry, rw, rh) = roi_rects(
                _params(roi=(x0, 0, 128, 64), disp_shift=shift), FULL_W, FULL_H)
            assert lx >= 0 and ly >= 0 and rx >= 0 and ry >= 0
            assert (lw, lh) == (rw, rh)
            assert rx + rw <= FULL_W and ry + rh <= FULL_H


def test_roi_is_clamped_into_the_frame():
    p = _params(roi=(FULL_W - 40, FULL_H - 20, 512, 512))
    (lx, ly, lw, lh), _ = roi_rects(p, FULL_W, FULL_H)
    assert lx + lw <= FULL_W and ly + lh <= FULL_H
    left, right = crop_pair(*_scene(), p)
    assert left.shape[:2] == (lh, lw) == right.shape[:2]


def test_crop_pair_is_a_passthrough_without_an_roi():
    left, right = _scene()
    a, b = crop_pair(left, right, _params())
    assert a is left and b is right
    assert roi_rects(_params(), FULL_W, FULL_H) is None


# ------------------------------------------------- planning the ROI from boxes
def test_roi_for_boxes_covers_them_and_collapses_max_disp():
    """The real rig's numbers: fx=21103.6 (rectified), B=5.1785 mm, ~212 mm."""
    from studio.measure import MeasureBox, roi_for_boxes

    fx = 21103.6
    K = np.array([[fx, 0, 1883.5], [0, fx, 1518.0], [0, 0, 1.0]])
    boxes = [MeasureBox(cx=0.0, cy=0.0, cz=212.0, sx=2, sy=2, sz=4),
             MeasureBox(cx=3.0, cy=1.0, cz=211.0, sx=2, sy=2, sz=4)]
    r = roi_for_boxes(boxes, K, 5.1785, (4024, 3036), scale=1.0)
    assert r is not None
    x0, y0, w, h = r["roi"]
    assert 0 <= x0 and 0 <= y0 and x0 + w <= 4024 and y0 + h <= 3036
    assert w % 32 == 0 and h % 32 == 0
    for b in boxes:
        p = np.asarray(b.corners(), float)
        u = fx * p[:, 0] / p[:, 2] + K[0, 2]
        v = fx * p[:, 1] / p[:, 2] + K[1, 2]
        assert u.min() >= x0 and u.max() <= x0 + w
        assert v.min() >= y0 and v.max() <= y0 + h
    assert r["max_disp"] == 64          # collapses from ~515 to the floor
    assert r["disp_shift"] < r["d_min"]  # never clips the far end to zero
    assert not r["shift_clamped"]


def test_roi_for_boxes_reports_a_clamped_shift():
    """A box near the left edge cannot take a 500 px pre-shift — say so."""
    from studio.measure import MeasureBox, roi_for_boxes

    fx = 21103.6
    K = np.array([[fx, 0, 1883.5], [0, fx, 1518.0], [0, 0, 1.0]])
    b = MeasureBox(cx=-18.0, cy=0.0, cz=212.0, sx=2, sy=2, sz=4)
    r = roi_for_boxes([b], K, 5.1785, (4024, 3036), scale=1.0)
    assert r["shift_clamped"] is True
    assert r["disp_shift"] == float(r["roi"][0])


def test_roi_for_boxes_none_when_nothing_is_in_front():
    from studio.measure import MeasureBox, roi_for_boxes

    K = np.array([[1000.0, 0, 100.0], [0, 1000.0, 50.0], [0, 0, 1.0]])
    assert roi_for_boxes([], K, 5.0, (200, 100)) is None
    behind = MeasureBox(cx=0, cy=0, cz=-50.0, sx=1, sy=1, sz=1)
    assert roi_for_boxes([behind], K, 5.0, (200, 100)) is None


# ------------------------------------------- the saturation check vs the shift
def test_saturation_check_uses_observed_not_true_disparity():
    """max_disp bounds what the network SEARCHED. With a pre-shift, comparing the
    TRUE disparity to it reports ~100 % saturation on every run — a permanent
    false alarm, modal outside a batch."""
    from studio.main_window import MainWindow          # static method only

    sat = MainWindow._disparity_saturation
    # a healthy ROI run: true disparity ~516, Δ=492, so the net searched ~24 px
    true_disp = np.full((64, 64), 516.0, np.float32)
    max_disp = 64
    assert sat(true_disp, max_disp) == pytest.approx(1.0)      # the false alarm
    observed = true_disp - 492.0
    assert sat(observed, max_disp) == 0.0                      # the truth

    # and it must still FIRE when the observed range really is saturated
    assert sat(np.full((64, 64), 63.0, np.float32), max_disp) == pytest.approx(1.0)


def test_crop_pair_passthrough_never_touches_the_images():
    """The no-ROI path is the common one and must not dereference the arrays —
    a dispatch before a pair is loaded would otherwise crash on None.shape."""
    a, b = crop_pair(None, None, _params())
    assert a is None and b is None
    a, b = crop_pair(None, None, None)
    assert a is None and b is None


def test_saturation_advice_differs_for_an_unshifted_roi():
    """A cropped run with no pre-shift saturates ANY max_disp — the rig's absolute
    disparity (~500 px) exceeds the slider's own maximum (416). Telling the user
    to raise it is advice that cannot work AND starts an hour-long engine build;
    the fix is Find shift."""
    from studio.main_window import MainWindow

    sat = MainWindow._disparity_saturation
    true_disp = np.full((64, 64), 500.0, np.float32)
    assert sat(true_disp, 64) == pytest.approx(1.0)        # unshifted: pinned
    assert sat(true_disp, 416) == pytest.approx(1.0)       # even at the slider max
    assert sat(true_disp - 470.0, 64) == 0.0               # shifted: fine


# ----------------------------------------------- engine readiness by SIZE
def test_padded_size_matches_the_engine_key():
    """The engine is keyed by the PADDED size, which is what decides whether
    dragging the box costs an hour. Must mirror run_inference + InputPadder."""
    from studio.backends.registry import padded_size

    assert padded_size(512, 1024, 1.0) == (1024, 512)      # already /32
    assert padded_size(800, 704, 1.0) == (704, 800)
    assert padded_size(4024, 3036, 0.30) == (928, 1216)    # the rig's full frame
    assert padded_size(500, 1000, 1.0) == (1024, 512)      # rounds UP to /32


def test_cached_engine_sizes_reads_the_filenames(tmp_path):
    from studio.backends.registry import cached_engine_sizes

    d = tmp_path / "weights" / "run"
    (d / "trt").mkdir(parents=True)
    (d / "trt" / "fp16_1024x512_i8_d64_trt10.3.engine").write_bytes(b"x")
    (d / "trt" / "fp16_704x800_i8_d192_trt10.3.engine").write_bytes(b"x")
    (d / "trt" / "fp16_832x960_i8_d224_trt10.3.engine.build.log").write_text("failed")
    ck = str(d / "model.pth")
    got = cached_engine_sizes(ck)
    assert (1024, 512, 8, 64) in got and (704, 800, 8, 192) in got
    assert (832, 960, 8, 224) not in got        # a build LOG is not an engine
    assert len(got) == 2


def test_cached_engine_sizes_survives_a_missing_directory():
    from studio.backends.registry import cached_engine_sizes

    assert cached_engine_sizes("/nope/model.pth") == set()
    assert cached_engine_sizes("") == set()


# --------------------------------------------- engine building is opt-in
def test_trt_backend_declares_itself_linux_only():
    """It looked READY on Windows and then failed when the child tried to import
    the bindings. A platform it cannot run on should read as unavailable."""
    from studio.backends import BACKENDS

    spec = BACKENDS["fast_foundation_stereo_trt"]
    assert spec.platforms == ("posix",)
    assert "build_engine" in [p.key for p in spec.params]
    assert [p for p in spec.params if p.key == "build_engine"][0].default is False


def test_every_other_backend_runs_anywhere():
    """Only the ENGINE is TensorRT-specific — the ROI crop, the pre-shift, the
    sites and the measurement are backend-agnostic, so nothing else should be
    platform-gated."""
    from studio.backends import BACKENDS

    for key, spec in BACKENDS.items():
        if key != "fast_foundation_stereo_trt":
            assert spec.platforms is None, key


def test_the_roi_pipeline_names_no_backend():
    """The whole point: switching backend must not change the measurement. If a
    backend name appears in these modules, that has stopped being true."""
    import re
    for mod in ("rectify", "infer", "sites_measure", "cloud", "measure"):
        src = open(f"studio/{mod}.py").read()
        hits = re.findall(r"tensorrt|fast_fs|foundation_stereo|backend_key", src)
        assert not hits, f"studio/{mod}.py mentions {set(hits)}"


# ----------------------------------------------------- setup checkability
def test_unavailable_backends_say_what_to_do():
    """The model repos are SIBLINGS, not submodules, so a fresh clone has none
    of them. 'repo not found' in a dropdown is not an instruction."""
    from studio.backends.base import BackendSpec, CheckpointSpec

    gone = BackendSpec(key="x", display_name="X", adapter_module="m",
                       repo_dir="/nonexistent/Some-Repo", checkpoints=[],
                       params=[])
    ok, why = gone.availability()
    assert not ok and "clone it next to this repo" in why and "/nonexistent" in why

    noweights = BackendSpec(key="y", display_name="Y", adapter_module="m",
                            checkpoints=[CheckpointSpec("c", "/nonexistent/w.pth")],
                            params=[])
    ok, why = noweights.availability()
    assert not ok and "/nonexistent/w.pth" in why


def test_requirements_pin_triton_for_windows():
    """Without it Fast-FS runs its cost volume EAGER: 4.2x the peak memory and
    slower. It was installed by hand before, so a fresh clone was silently worse
    than the machine it replaced."""
    req = open("requirements.txt").read()
    assert "triton-windows" in req
    assert 'sys_platform == "win32"' in req


def test_setup_check_runs_and_reports():
    """It must survive being run on a machine where things ARE missing — that is
    the only machine it matters on."""
    import subprocess
    import sys as _s
    r = subprocess.run([_s.executable, "tools/check_setup.py"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode in (0, 1), r.stderr[-500:]
    for expected in ("Python", "torch", "triton", "PySide6", "backend"):
        assert expected in r.stdout, expected


# ------------------------------------------------------------- installer
def _installer():
    import importlib.util
    sp = importlib.util.spec_from_file_location("inst", "install.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def test_installer_resolves_the_sibling_repo_not_a_subdir():
    """The single thing a fresh clone gets wrong: Fast-FoundationStereo is a
    SIBLING, not a submodule or a subdirectory."""
    import os
    m = _installer()
    assert os.path.dirname(m.FAST_REPO) == os.path.dirname(m.REPO)
    assert not m.FAST_REPO.startswith(m.REPO + os.sep)
    assert m.WEIGHTS_DIR.startswith(m.FAST_REPO)


def test_installer_detects_this_platform():
    m = _installer()
    import os
    assert m.IS_WIN == (os.name == "nt")
    assert m.IS_JETSON == (os.name != "nt" and os.path.exists("/etc/nv_tegra_release"))
    # the venv interpreter must be the one this platform actually uses
    assert m.VPY.endswith("python.exe" if m.IS_WIN else "python")


def test_installer_steps_are_no_ops_when_already_satisfied():
    """Idempotent: re-running after a fix must skip what is done, never redo a
    68 MB download or re-clone over an existing repo."""
    import contextlib, io, os
    m = _installer()
    if not os.path.isdir(os.path.join(m.FAST_REPO, ".git")):
        pytest.skip("sibling repo not present on this machine")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.sibling_repo(True)
        m.weights(True)
    out = buf.getvalue()
    assert "present" in out
    assert "downloading" not in out and "git clone" not in out


def test_installer_writes_a_launcher_matching_the_rig(tmp_path, monkeypatch):
    import os
    m = _installer()
    if m.IS_WIN:
        pytest.skip("Linux .desktop path")
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(os.path, "expanduser", lambda p: p.replace("~", str(home)))
    m.desktop_icon()
    entry = home / "Desktop" / "FoundationStereo Studio.desktop"
    assert entry.exists()
    text = entry.read_text()
    for key in ("Type=Application", "Name=FoundationStereo Studio",
                "Exec=", "Path=", "Icon=", "Terminal=false"):
        assert key in text, key
    assert os.access(entry, os.X_OK)      # launchers must be executable


def test_triton_resolves_correctly_on_every_platform():
    """Triton is the memory budget, not a speed knob — eager costs 4.2x. Each
    platform needs a DIFFERENT source, and aarch64 must resolve to nothing here
    because PyPI has no Jetson build (setup_jetson.sh uses the Jetson index)."""
    from packaging.markers import Marker

    lines = [l.split(";") for l in open("requirements.txt")
             if l.startswith("triton")]
    def resolve(env):
        return [l[0].strip() for l in lines if Marker(l[1].strip()).evaluate(env)]

    win = resolve({"sys_platform": "win32", "platform_machine": "AMD64"})
    lin = resolve({"sys_platform": "linux", "platform_machine": "x86_64"})
    jet = resolve({"sys_platform": "linux", "platform_machine": "aarch64"})
    assert len(win) == 1 and win[0].startswith("triton-windows")
    assert len(lin) == 1 and lin[0].startswith("triton>")
    assert jet == [], "aarch64 must NOT pull a PyPI triton — it cannot serve the iGPU"


def test_both_launchers_refuse_without_a_venv():
    """A launcher that silently does nothing is worse than one that says why."""
    sh = open("run_studio.sh").read()
    bat = open("run_studio.bat").read()
    for name, text in (("run_studio.sh", sh), ("run_studio.bat", bat)):
        assert ".venv" in text, name
        assert "install.py" in text or "setup_jetson.sh" in text, name
    assert "exit /b 1" in bat and "exit 1" in sh


def test_launchers_have_the_line_endings_their_shells_need():
    """cmd.exe can misparse a multi-line if() block in an LF-only .bat, and this
    repo's Windows launcher has one. bash is the mirror case: CR would end up in
    the command. Neither is testable from the other platform, so pin it here."""
    bat = open("run_studio.bat", "rb").read()
    sh = open("run_studio.sh", "rb").read()
    assert b"\r\n" in bat, "run_studio.bat must be CRLF"
    assert b"\r" not in sh, "run_studio.sh must be LF only"
    attrs = open(".gitattributes").read()
    assert "*.bat text eol=crlf" in attrs and "*.sh text eol=lf" in attrs
