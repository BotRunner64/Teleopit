from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from teleopit.inputs.pico4_provider import BODY_JOINT_NAMES, Pico4InputProvider
from teleopit.inputs.realtime_frame_cache import RealtimeFrameCache
from teleopit.inputs.realtime_packet import ControlEventType


def _body_poses(offset: float) -> np.ndarray:
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    body_poses[:, 0] = offset
    body_poses[:, 6] = 1.0
    return body_poses


def _pico_frame(
    body_poses: np.ndarray,
    *,
    seq: int,
    timestamp: float,
    body_active: bool = True,
    right_primary: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        seq=seq,
        receive_time_s=timestamp,
        body=SimpleNamespace(active=body_active, joints=body_poses),
        controllers=SimpleNamespace(
            left=SimpleNamespace(buttons={}),
            right=SimpleNamespace(buttons={"primaryButton": right_primary}),
        ),
    )


def _make_provider() -> Pico4InputProvider:
    provider = object.__new__(Pico4InputProvider)
    provider._lock = threading.Lock()
    provider._frame_ready = threading.Event()
    provider._frame_cache = RealtimeFrameCache(buffer_size=8, fps_window=30)
    provider._timeout = 1.0
    provider._timestamp_gap_reset_s = 0.15
    provider._pending_control_events = deque()
    provider._pause_button = "A"
    provider._pause_debounce_s = 0.0
    provider._pause_button_path = provider._resolve_button_path(provider._pause_button)
    provider._last_pause_button_pressed = False
    provider._last_pause_toggle_timestamp = None
    provider._last_raw_body_joints = None
    provider._last_frame_timestamp = None
    provider._last_source_seq = None
    provider._ground_lift_offset = None
    provider._closed = False
    return provider


class _FakeBridge:
    instances: list["_FakeBridge"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        _FakeBridge.instances.append(self)

    def start(self) -> None:
        self.started = True

    def wait_frame(self, timeout: float = 0.1, after_seq: int | None = None) -> Any:
        del timeout, after_seq
        raise TimeoutError

    def close(self) -> None:
        self.closed = True


class _LegacyBridge:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        discovery: bool,
        advertise_ip: str | None,
        video: str | None,
        history_size: int,
        start_timeout: float,
    ) -> None:
        del host, port, discovery, advertise_ip, video, history_size, start_timeout


def test_pico4_provider_starts_pico_bridge_receiver_with_config() -> None:
    _FakeBridge.instances.clear()

    provider = Pico4InputProvider(
        timeout=0.01,
        bridge_host="127.0.0.1",
        bridge_port=12345,
        bridge_discovery=False,
        bridge_advertise_ip="127.0.0.1",
        bridge_video="frames",
        bridge_video_enabled=True,
        bridge_start_timeout=1.5,
        bridge_history_size=7,
        bridge_cls=_FakeBridge,
    )

    try:
        bridge = _FakeBridge.instances[-1]
        assert bridge.started is True
        assert bridge.kwargs == {
            "host": "127.0.0.1",
            "port": 12345,
            "discovery": False,
            "advertise_ip": "127.0.0.1",
            "video": "frames",
            "video_enabled": True,
            "history_size": 7,
            "start_timeout": 1.5,
        }
    finally:
        provider.close()

    assert bridge.closed is True


def test_pico4_provider_requires_pico_bridge_0_2_signature() -> None:
    with pytest.raises(RuntimeError, match=r"pico_bridge >= 0\.2\.0"):
        Pico4InputProvider(timeout=0.01, bridge_cls=_LegacyBridge)


def test_pico4_provider_pushes_video_frame_to_bridge() -> None:
    _FakeBridge.instances.clear()

    def push_video_frame(self: _FakeBridge, frame: np.ndarray) -> int:
        self.video_frame = frame
        return 12

    _FakeBridge.push_video_frame = push_video_frame  # type: ignore[attr-defined]
    provider = Pico4InputProvider(timeout=0.01, bridge_video="frames", bridge_video_enabled=True, bridge_cls=_FakeBridge)
    try:
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        assert provider.push_video_frame(frame) == 12
        assert _FakeBridge.instances[-1].video_frame is frame
    finally:
        provider.close()
        delattr(_FakeBridge, "push_video_frame")


def test_pico4_provider_normalizes_pico_bridge_body_pose_convention() -> None:
    body_poses = _body_poses(0.0)
    body_poses[0] = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]

    converted = Pico4InputProvider._normalize_pico_bridge_body_joints(body_poses)
    np.testing.assert_allclose(converted[0], [1.0, 2.0, -3.0, 0.1, 0.2, -0.3, -0.9])

    frame = Pico4InputProvider._convert_body_joints_to_frame(body_poses)
    np.testing.assert_allclose(frame["Pelvis"][0], [1.0, 3.0, 2.0], atol=1e-6)


def test_pico4_provider_applies_fixed_ground_lift_from_first_real_frame() -> None:
    provider = _make_provider()
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    pelvis_idx = BODY_JOINT_NAMES.index("Pelvis")
    left_ankle_idx = BODY_JOINT_NAMES.index("Left_Ankle")
    right_ankle_idx = BODY_JOINT_NAMES.index("Right_Ankle")
    body_poses[pelvis_idx, 0:3] = [0.0, 0.8, 0.0]
    body_poses[left_ankle_idx, 0:3] = [0.1, -0.2, 0.0]
    body_poses[right_ankle_idx, 0:3] = [-0.1, 0.1, 0.0]
    body_poses[:, 6] = 1.0

    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=1, timestamp=1.0)) is True
    first_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(first_frame["Pelvis"][0][2], 0.8 + 0.2, atol=1e-6)
    np.testing.assert_allclose(first_frame["Left_Ankle"][0][2], 0.0, atol=1e-6)
    assert provider._ground_lift_offset == pytest.approx(0.2)

    body_poses[:, 1] += 0.3
    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=2, timestamp=1.1)) is True
    second_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(second_frame["Pelvis"][0][2], first_frame["Pelvis"][0][2] + 0.3, atol=1e-6)
    np.testing.assert_allclose(second_frame["Left_Ankle"][0][2], 0.3, atol=1e-6)


def test_pico4_provider_recomputes_ground_lift_after_timestamp_gap_reset() -> None:
    provider = _make_provider()
    body_poses = np.zeros((len(BODY_JOINT_NAMES), 7), dtype=np.float64)
    pelvis_idx = BODY_JOINT_NAMES.index("Pelvis")
    left_ankle_idx = BODY_JOINT_NAMES.index("Left_Ankle")
    right_ankle_idx = BODY_JOINT_NAMES.index("Right_Ankle")
    body_poses[pelvis_idx, 0:3] = [0.0, 0.8, 0.0]
    body_poses[left_ankle_idx, 0:3] = [0.1, -0.2, 0.0]
    body_poses[right_ankle_idx, 0:3] = [-0.1, 0.1, 0.0]
    body_poses[:, 6] = 1.0

    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=1, timestamp=1.0)) is True
    assert provider._ground_lift_offset == pytest.approx(0.2)

    body_poses[pelvis_idx, 1] = 0.7
    body_poses[left_ankle_idx, 1] = -0.5
    body_poses[right_ankle_idx, 1] = 0.2
    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=2, timestamp=1.3)) is True
    latest_frame, _, _ = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(latest_frame["Left_Ankle"][0][2], 0.0, atol=1e-6)
    np.testing.assert_allclose(latest_frame["Pelvis"][0][2], 1.2, atol=1e-6)
    assert provider._ground_lift_offset == pytest.approx(0.5)
    assert len(provider._frame_cache) == 1


def test_pico4_provider_drops_duplicate_raw_body_pose() -> None:
    provider = _make_provider()
    body_poses = _body_poses(1.0)

    assert provider._accept_pico_frame(_pico_frame(body_poses, seq=1, timestamp=1.0)) is True
    assert provider._accept_pico_frame(_pico_frame(body_poses.copy(), seq=2, timestamp=1.01)) is False
    assert len(provider._frame_cache) == 1


def test_pico4_provider_resets_interpolation_buffer_on_large_timestamp_gap() -> None:
    provider = _make_provider()

    assert provider._accept_pico_frame(_pico_frame(_body_poses(1.0), seq=1, timestamp=1.00)) is True
    assert provider._accept_pico_frame(_pico_frame(_body_poses(2.0), seq=2, timestamp=1.02)) is True
    assert len(provider._frame_cache) == 2

    assert provider._accept_pico_frame(_pico_frame(_body_poses(9.0), seq=3, timestamp=1.30)) is True
    assert len(provider._frame_cache) == 1
    latest_frame, latest_ts, latest_seq = provider._frame_cache.latest_packet()
    np.testing.assert_allclose(latest_frame["Pelvis"][0][0], 9.0, atol=1e-6)
    np.testing.assert_allclose(latest_ts, 1.30, atol=1e-6)
    assert latest_seq == 3


def test_pico4_provider_exposes_pause_control_events_once() -> None:
    provider = _make_provider()

    assert provider._accept_pico_frame(
        _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0, right_primary=True)
    ) is True

    packet = provider.get_realtime_input_packet()
    assert [event.event_type for event in packet.control_events] == [ControlEventType.TOGGLE_PAUSE]

    packet = provider.get_realtime_input_packet()
    assert packet.control_events == ()


def test_pico4_provider_marks_controller_present_without_raw_field() -> None:
    provider = _make_provider()
    frame = _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0)
    frame.controllers.left.axis = {"grip": 0.8, "trigger": 0.4}

    assert provider._accept_pico_frame(frame) is True

    snapshot = provider.get_controller_snapshot()
    assert snapshot is not None
    assert snapshot.left.present is True
    assert snapshot.left.raw is False
    assert snapshot.left.grip == pytest.approx(0.8)
    assert snapshot.left.trigger == pytest.approx(0.4)


def test_pico4_provider_reads_pause_control_events_when_body_inactive() -> None:
    provider = _make_provider()

    assert provider._accept_pico_frame(
        _pico_frame(_body_poses(1.0), seq=1, timestamp=1.0, body_active=False, right_primary=True)
    ) is False

    events = provider.pop_control_events()
    assert [event.event_type for event in events] == [ControlEventType.TOGGLE_PAUSE]
    assert provider._last_source_seq == 1
