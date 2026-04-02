from __future__ import annotations

import threading

import numpy as np

from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES, Pico4InputProvider
from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache


def _body_poses(offset: float) -> np.ndarray:
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    body_poses[:, 0] = offset
    body_poses[:, 6] = 1.0
    return body_poses


def _make_provider() -> Pico4InputProvider:
    provider = object.__new__(Pico4InputProvider)
    provider._lock = threading.Lock()
    provider._frame_ready = threading.Event()
    provider._frame_cache = RealtimeFrameCache(buffer_size=8, fps_window=30)
    provider._timestamp_gap_reset_s = 0.15
    provider._poll_sleep_s = 0.0
    provider._last_raw_body_poses = None
    provider._last_frame_timestamp = None
    provider._closed = False
    return provider


def test_pico4_provider_drops_duplicate_raw_body_pose() -> None:
    provider = _make_provider()
    body_poses = _body_poses(1.0)

    assert provider._accept_body_poses(body_poses, timestamp=1.0) is True
    assert provider._accept_body_poses(body_poses.copy(), timestamp=1.01) is False
    assert len(provider._frame_cache) == 1


def test_pico4_provider_resets_interpolation_buffer_on_large_timestamp_gap() -> None:
    provider = _make_provider()

    assert provider._accept_body_poses(_body_poses(1.0), timestamp=1.00) is True
    assert provider._accept_body_poses(_body_poses(2.0), timestamp=1.02) is True
    assert len(provider._frame_cache) == 2

    assert provider._accept_body_poses(_body_poses(9.0), timestamp=1.30) is True
    assert len(provider._frame_cache) == 1
    latest_frame, latest_ts, latest_seq = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(latest_frame["Pelvis"][0][0], 9.0, atol=1e-6)
    np.testing.assert_allclose(latest_ts, 1.30, atol=1e-6)
    assert latest_seq == 3
