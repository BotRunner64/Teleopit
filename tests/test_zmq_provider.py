from __future__ import annotations

import threading

import numpy as np

from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.zmq_provider import ZMQInputProvider


def _frame(x: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "Pelvis": (
            np.asarray([x, 0.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
    }


def _make_provider() -> ZMQInputProvider:
    provider = object.__new__(ZMQInputProvider)
    provider._lock = threading.Lock()
    provider._frame_ready = threading.Event()
    provider._frame_cache = RealtimeFrameCache(buffer_size=8, fps_window=30)
    provider._running = True
    provider._clock_offset = None
    provider._last_source_seq = None
    provider._last_buffer_ts = None
    provider._seq_gap_reset_threshold = 4
    return provider


def test_zmq_provider_drops_duplicate_source_seq() -> None:
    provider = _make_provider()

    provider._process_packet(_frame(1.0), source_ts=1.0, source_seq=10, local_ts=11.0)
    provider._process_packet(_frame(2.0), source_ts=1.0, source_seq=10, local_ts=11.1)

    assert len(provider._frame_cache) == 1
    np.testing.assert_allclose(provider._frame_cache.latest()["Pelvis"][0][0], 1.0, atol=1e-6)
    assert provider._frame_cache.frame_seq == 10


def test_zmq_provider_resets_buffer_on_large_seq_gap() -> None:
    provider = _make_provider()

    provider._process_packet(_frame(1.0), source_ts=1.0, source_seq=10, local_ts=11.0)
    provider._process_packet(_frame(2.0), source_ts=1.02, source_seq=11, local_ts=11.02)
    assert len(provider._frame_cache) == 2

    provider._process_packet(_frame(9.0), source_ts=1.20, source_seq=20, local_ts=11.20)

    assert len(provider._frame_cache) == 1
    assert provider._frame_cache.frame_seq == 20
    np.testing.assert_allclose(provider._frame_cache.latest()["Pelvis"][0][0], 9.0, atol=1e-6)


def test_zmq_provider_monotonicizes_missing_source_seq_timestamps() -> None:
    provider = _make_provider()

    provider._process_packet(_frame(1.0), source_ts=None, source_seq=None, local_ts=11.0)
    provider._process_packet(_frame(2.0), source_ts=None, source_seq=None, local_ts=10.5)

    assert len(provider._frame_cache) == 1
    np.testing.assert_allclose(provider._frame_cache.latest()["Pelvis"][0][0], 1.0, atol=1e-6)
