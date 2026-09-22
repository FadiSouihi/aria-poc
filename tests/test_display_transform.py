"""Contract tests: debug-view transforms (mirror/flip + aspect-preserving fit).

Mirroring the debug view must not touch the pipeline, so it lives in pure
functions that are tested here rather than in the GUI thread.
"""
import numpy as np
import pytest

from aria.perception.transform import (FLIP_MODES, apply_flip, fit_to_window,
                                      flip_bbox, normalize_flip)


def _frame():
    # a small frame with a unique corner so flips are unambiguous
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)        # top-left marker
    frame[0, 5] = (0, 255, 0)        # top-right marker
    return frame


def test_horizontal_mirror_swaps_left_and_right():
    frame = _frame()
    flipped = apply_flip(frame, "horizontal")
    assert tuple(flipped[0, 0]) == (0, 255, 0)      # old top-right is now top-left
    assert tuple(flipped[0, 5]) == (255, 0, 0)
    assert flipped.shape == frame.shape
    assert flipped is not frame                     # a copy, never the store's array


def test_vertical_flip_swaps_top_and_bottom():
    frame = _frame()
    flipped = apply_flip(frame, "vertical")
    assert tuple(flipped[3, 0]) == (255, 0, 0)
    assert tuple(flipped[0, 0]) == (0, 0, 0)


def test_both_is_a_180_rotation():
    frame = _frame()
    flipped = apply_flip(frame, "both")
    assert tuple(flipped[3, 5]) == (255, 0, 0)      # top-left marker → bottom-right
    assert tuple(flipped[3, 0]) == (0, 255, 0)


def test_none_and_aliases():
    frame = _frame()
    assert apply_flip(frame, "none") is frame
    assert normalize_flip("mirror") == "horizontal"
    assert normalize_flip("180") == "both"
    assert normalize_flip("nonsense") == "none"
    assert normalize_flip("") == "none"
    assert FLIP_MODES == ("none", "horizontal", "vertical", "both")


def test_bbox_follows_the_mirror():
    width, height = 640, 480
    bbox = (100, 50, 200, 150)                      # x1, y1, x2, y2
    assert flip_bbox(bbox, width, height, "horizontal") == (440, 50, 540, 150)
    assert flip_bbox(bbox, width, height, "vertical") == (100, 330, 200, 430)
    assert flip_bbox(bbox, width, height, "both") == (440, 330, 540, 430)
    assert flip_bbox(bbox, width, height, "none") == (100, 50, 200, 150)


def test_bbox_round_trips():
    bbox = (100, 50, 200, 150)
    there = flip_bbox(bbox, 640, 480, "horizontal")
    back = flip_bbox(there, 640, 480, "horizontal")
    assert back == bbox


def test_fit_preserves_aspect_ratio_with_black_bars():
    cv2 = pytest.importorskip("cv2")
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)     # 4:3
    out = fit_to_window(frame, 960, 540)                    # 16:9 window
    assert out.shape == (540, 960, 3)
    # scaled 4:3 image inside 960x540 is 720x540 → 120 px bars on each side
    assert tuple(out[270, 0]) == (0, 0, 0)                  # left bar
    assert tuple(out[270, 959]) == (0, 0, 0)                # right bar
    assert tuple(out[270, 480]) == (200, 200, 200)          # image in the middle
    # the subject is not squashed: the image band is exactly 540 tall
    assert tuple(out[0, 480]) == (200, 200, 200)


def test_fit_is_a_noop_when_sizes_already_match():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert fit_to_window(frame, 320, 240) is frame          # no work, no cv2 needed


def test_fit_upscales_a_smaller_frame_to_the_window():
    pytest.importorskip("cv2")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert fit_to_window(frame, 640, 480).shape == (480, 640, 3)
