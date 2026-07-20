"""studio.pairs — image conversion + pair discovery (all Qt-free)."""
import numpy as np
import pytest

from studio.pairs import IMG_EXTS, IMG_FILTER, find_pairs, to_rgb_u8


# ------------------------------------------------------------- to_rgb_u8
def test_uint8_rgb_passthrough():
    a = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    out = to_rgb_u8(a)
    assert out.shape == (2, 3, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, a)


def test_uint16_true_range_scales_not_wraps():
    # the old astype(np.uint8) wrapped 65535 -> 255? no: 65535 % 256 == 255, but
    # 256 wrapped to 0 and 65280 to 0 — mid-range values were garbage
    a = np.array([[0, 256, 32768, 65535]], np.uint16)
    out = to_rgb_u8(a)
    assert out[0, 0, 0] == 0
    assert out[0, 1, 0] in (0, 1)          # 256/257 ≈ 1
    assert abs(int(out[0, 2, 0]) - 128) <= 1
    assert out[0, 3, 0] == 255             # full-scale stays full-scale


def test_uint16_8bit_in_16_container_passthrough():
    a = np.full((2, 2), 200, np.uint16)
    assert to_rgb_u8(a).max() == 200


def test_gray_alpha_two_channel():
    la = np.dstack([np.full((2, 2), 7, np.uint8), np.full((2, 2), 255, np.uint8)])
    out = to_rgb_u8(la)
    assert out.shape == (2, 2, 3)
    assert (out == 7).all()                # gray kept, alpha dropped


def test_float_01_and_0255():
    f = np.full((2, 2), 0.5, np.float32)
    assert abs(int(to_rgb_u8(f)[0, 0, 0]) - 127) <= 1
    g = np.full((2, 2), 200.0, np.float64)
    assert to_rgb_u8(g)[0, 0, 0] == 200


def test_rgba_drops_alpha_and_gray_stacks():
    rgba = np.zeros((2, 2, 4), np.uint8)
    assert to_rgb_u8(rgba).shape == (2, 2, 3)
    assert to_rgb_u8(np.zeros((2, 2), np.uint8)).shape == (2, 2, 3)


def test_filter_matches_ext_set():
    for e in IMG_EXTS:
        assert "*" + e in IMG_FILTER


# ------------------------------------------------------------- discovery
def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_two_dir_pairing_case_insensitive(tmp_path):
    # left/CAP_01.PNG must pair with right/cap_01.png on Windows-style naming
    _touch(tmp_path / "left" / "CAP_01.PNG")
    _touch(tmp_path / "right" / "cap_01.png")
    _touch(tmp_path / "left" / "cap_02.png")
    _touch(tmp_path / "right" / "cap_02.png")
    scan = find_pairs(str(tmp_path))
    assert len(scan.pairs) == 2
    assert "matched by filename" in scan.method
    assert scan.unpaired == []


def test_single_folder_family_pairing(tmp_path):
    _touch(tmp_path / "board_left.png")
    _touch(tmp_path / "board_right.png")
    _touch(tmp_path / "stray.png")
    scan = find_pairs(str(tmp_path))
    assert len(scan.pairs) == 1
    assert scan.pairs[0][0] == "board"
    assert scan.unpaired == ["stray.png"]


def test_not_a_folder():
    scan = find_pairs(r"Z:\definitely\not\here")
    assert not scan and scan.pairs == []
