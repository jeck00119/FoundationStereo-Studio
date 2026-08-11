"""The window's extracted controllers — level, analyze, export, ROI.

Written because splitting MainWindow introduced a bug the whole suite missed:
level.relevel_current() still called win._reset_analyze_overlay() after that
method had moved onto the analyze controller. Nothing exercised the path, so
102 green tests said nothing about it and only running the app found it.

Two guards here, deliberately different in kind:
  * behavioural — actually drive level/analyze over a synthetic cloud
  * structural  — every controller->window and window->controller attribute
    reference must resolve, so the NEXT move-a-method refactor fails loudly
    even if no test happens to walk that line
"""
import re

import numpy as np
import pytest

from studio.dtypes import CloudResult


@pytest.fixture
def win(qapp):
    from studio.main_window import MainWindow
    w = MainWindow()
    yield w
    w.close()


def _tilted_board(n=40000, tilt=0.02, z0=210.0, seed=0):
    """A board plane tilted `tilt` in X — something for the leveller to remove."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-8, 8, (n, 2))
    z = z0 + tilt * xy[:, 0] + rng.normal(0, 0.02, n)
    pts = np.column_stack([xy, z]).astype(np.float32)
    return CloudResult(points=pts, colors=np.full((n, 3), 128, np.uint8), n=n)


# --------------------------------------------------------------- structural
def test_the_window_never_calls_a_method_it_no_longer_has(win):
    """The broadest guard, and the one that was missing.

    Extracting a cluster leaves callers behind in places no test walks. The first
    version of this file only checked window<->controller references, so it
    caught level.relevel_current() calling a moved method but NOT
    _set_units() calling self._update_ref_label() — same bug, one frame further
    out, found only by running the 3D-tab harness. Every self._* reference in
    the window must resolve, full stop."""
    src = open("studio/main_window.py").read()
    refs = sorted(set(re.findall(r"self\.(_[A-Za-z]\w*)", src)))
    missing = [r for r in refs if not hasattr(win, r)]
    assert not missing, f"main_window calls self.{missing} which no longer exists"


def test_tools_do_not_drive_a_moved_window_api(win):
    """tools/ scripts drive the REAL window through its private API, so a rename
    breaks them silently — they are not imported by the suite. verify_3d_tab.py
    was calling win._ingest_level() for exactly this reason."""
    import glob
    missing = {}
    for path in glob.glob("tools/*.py"):
        src = open(path).read()
        refs = set(re.findall(r"\bwin\.(_?[A-Za-z]\w*)", src))
        bad = sorted(r for r in refs if not hasattr(win, r))
        if bad:
            missing[path] = bad
    assert not missing, f"tools reference window members that no longer exist: {missing}"


def test_every_controller_window_reference_resolves(win):
    """The guard that would have caught the level->analyze breakage."""
    missing = {}
    for mod in ("analyze", "level", "export", "roi"):
        src = open(f"studio/window/{mod}.py").read()
        refs = set(re.findall(r"(?:self\.win|win)\.([A-Za-z_]\w*)", src))
        bad = sorted(r for r in refs if not hasattr(win, r))
        if bad:
            missing[f"window/{mod}.py"] = bad
    assert not missing, f"controllers reference window attributes that do not exist: {missing}"


def test_every_window_reference_to_a_controller_resolves(win):
    src = open("studio/main_window.py").read()
    missing = {}
    for name in ("analyze", "level", "export", "roi"):
        obj = getattr(win, name)
        used = set(re.findall(rf"self\.{name}\.([A-Za-z_]\w*)", src))
        bad = sorted(u for u in used if not hasattr(obj, u))
        if bad:
            missing[name] = bad
    assert not missing, f"window calls controller members that do not exist: {missing}"


def test_controllers_own_their_state_not_the_window(win):
    """The point of the split: state moved, it did not get copied."""
    for gone in ("_level_R", "_level_c_m", "_analyze_tool", "_picked", "_dev_on",
                 "_analyze_isolate", "_z_offset_m", "_z_ref_pp_m", "_last_region",
                 "_analyze_last", "_roi", "_disp_shift"):
        assert not hasattr(win, gone), f"{gone} still on the window"
    assert hasattr(win.level, "R") and hasattr(win.analyze, "tool")
    assert hasattr(win.roi, "roi") and hasattr(win.roi, "disp_shift")


# -------------------------------------------------------------- behavioural
def test_level_round_trip_restores_the_raw_cloud(win):
    win.cloud = _tilted_board()
    win.level.ingest(win.cloud)
    raw = win.cloud.raw_points.copy()
    win.level.on_toggled(True)
    assert win.level.R is not None
    # the levelled board is flatter in Z than the tilted one
    assert win.cloud.points[:, 2].std() < raw[:, 2].std()
    win.level.on_toggled(False)
    assert win.level.R is None
    assert np.allclose(win.cloud.points, raw)


def test_board_plane_follows_the_level_state(win):
    """Unlevelled it is a fresh fit (so it carries the board's tilt); levelled it
    is exactly -Z, because that is what the rotation was built to achieve."""
    win.cloud = _tilted_board(tilt=0.02)
    win.level.ingest(win.cloud)
    n_before, _ = win.level.board_plane()
    tilt_deg = np.degrees(np.arccos(min(1.0, abs(n_before[2]))))
    assert tilt_deg == pytest.approx(np.degrees(np.arctan(0.02)), abs=0.1)
    win.level.on_toggled(True)
    n_after, _ = win.level.board_plane()
    assert np.allclose(n_after, [0.0, 0.0, -1.0])
    win.level.on_toggled(False)


def test_analyze_region_measures_and_resets(win):
    win.cloud = _tilted_board()
    win.level.ingest(win.cloud)
    win.analyze.on_tool("region")
    assert win.analyze.tool == "region" and win.analyze.picked == []
    win.analyze.on_point_picked(-2, -2, 210.0)
    win.analyze.on_point_picked(2, 2, 210.1)
    assert win.analyze.last_shown == "region"
    assert win.analyze.last_region is not None
    assert win.analyze.last_region["rms"] > 0
    win.analyze.reset_overlay()
    assert win.analyze.picked == []


def test_export_menu_is_built_from_the_controller(win):
    menu = win.export_btn.menu()
    assert menu is not None
    from studio.window.export import MENU
    assert len(menu.actions()) == len(MENU)


def test_roi_reaches_the_run_params(win):
    assert win._current_params().roi is None
    win.roi.roi = (128, 64, 256, 192)
    win.roi.disp_shift = 400.0
    p = win._current_params()
    assert p.roi == (128, 64, 256, 192)
    assert p.disp_shift == 400.0
    assert p.effective_shift == 128.0          # clamped to x0, one source of truth
