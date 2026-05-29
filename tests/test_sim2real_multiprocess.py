from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teleopit.runtime.mocap_session import MocapSessionState
from teleopit.sim2real.mp.ipc import HEALTH_TOPIC, REFERENCE_RESET_TOPIC, LatestSubscriber, ZmqPublisher
from teleopit.sim2real.mp import resolve_sim2real_runtime_mode
from teleopit.sim2real.mp.messages import ReferencePacket, ReferenceResetPacket, SharedFrameDescriptor
from teleopit.sim2real.mp.runtime import MultiprocessSim2RealController, _RobotControlWorker, _human_frame_is_valid
from teleopit.sim2real.mp.shm import SharedFrameRingReader, SharedFrameRingWriter


def test_resolve_runtime_auto_uses_multiprocess_for_pico4() -> None:
    cfg = {"sim2real_runtime": "auto", "input": {"provider": "pico4"}}
    assert resolve_sim2real_runtime_mode(cfg) == "multiprocess"


def test_resolve_runtime_auto_uses_single_process_for_bvh() -> None:
    cfg = {"sim2real_runtime": "auto", "input": {"provider": "bvh"}}
    assert resolve_sim2real_runtime_mode(cfg) == "single_process"


def test_multiprocess_requires_pico4_provider() -> None:
    cfg = {"sim2real_runtime": "multiprocess", "input": {"provider": "bvh"}}
    with pytest.raises(ValueError, match="requires input.provider=pico4"):
        resolve_sim2real_runtime_mode(cfg)


def test_shared_frame_ring_roundtrip() -> None:
    writer = SharedFrameRingWriter(shape=(2, 3, 1), dtype=np.uint8, slots=2)
    reader = SharedFrameRingReader()
    try:
        frame0 = np.arange(6, dtype=np.uint8).reshape(2, 3, 1)
        desc0 = writer.write(frame0, timestamp_s=1.0)
        assert isinstance(desc0, SharedFrameDescriptor)
        np.testing.assert_array_equal(reader.read(desc0, copy=True), frame0)

        frame1 = np.full((2, 3, 1), 9, dtype=np.uint8)
        desc1 = writer.write(frame1, timestamp_s=2.0)
        np.testing.assert_array_equal(reader.read(desc1, copy=True), frame1)
        assert desc1.slot != desc0.slot
    finally:
        reader.close()
        writer.close(unlink=True)


def test_multiprocess_rejects_unsupported_video_source() -> None:
    cfg = {
        "sim2real_runtime": "multiprocess",
        "input": {"provider": "pico4", "video": {"enabled": True, "source": "mujoco"}},
    }
    with pytest.raises(ValueError, match="only supports input.video.source=realsense or test-pattern"):
        MultiprocessSim2RealController(cfg)


def test_zmq_endpoint_allows_one_publisher_and_subscribers() -> None:
    endpoint = "inproc://sim2real-health-test"
    import zmq

    context = zmq.Context()
    publisher = ZmqPublisher(endpoint, context=context)
    subscriber = LatestSubscriber(endpoint, HEALTH_TOPIC, context=context)
    try:
        with pytest.raises(zmq.ZMQError):
            ZmqPublisher(endpoint, context=context)
    finally:
        subscriber.close()
        publisher.close()
        context.term()


def test_run_sim2real_single_process_shutdowns_on_exception(monkeypatch) -> None:
    script_path = Path.cwd() / "scripts" / "run" / "run_sim2real.py"
    spec = importlib.util.spec_from_file_location("test_run_sim2real", script_path)
    assert spec is not None and spec.loader is not None
    run_sim2real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_sim2real)

    calls: list[str] = []

    class FailingController:
        def __init__(self, _cfg: object) -> None:
            calls.append("init")

        def run(self) -> None:
            calls.append("run")
            raise RuntimeError("boom")

        def shutdown(self) -> None:
            calls.append("shutdown")

    cfg = SimpleNamespace(
        input={"provider": "bvh"},
        controller=SimpleNamespace(policy_path="policy.onnx"),
    )
    monkeypatch.setattr(run_sim2real, "validate_policy_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_sim2real, "resolve_sim2real_runtime_mode", lambda _cfg: "single_process")
    monkeypatch.setattr(run_sim2real, "Sim2RealController", FailingController)

    with pytest.raises(RuntimeError, match="boom"):
        run_sim2real._run_sim2real(cfg)

    assert calls == ["init", "run", "shutdown"]


def test_multiprocess_run_cleans_up_after_start_failure(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, *, name: str, fail_start: bool = False) -> None:
            self.name = name
            self.exitcode = None
            self.fail_start = fail_start
            self.started = False
            self.join_calls: list[float | None] = []
            self.terminated = False

        def start(self) -> None:
            if self.fail_start:
                raise RuntimeError("start failed")
            self.started = True

        def is_alive(self) -> bool:
            return self.started and not self.terminated

        def join(self, timeout: float | None = None) -> None:
            self.join_calls.append(timeout)

        def terminate(self) -> None:
            self.terminated = True

    started_process = FakeProcess(name="pico_io")

    def fake_start_processes(self: MultiprocessSim2RealController) -> None:
        started_process.started = True
        self._processes.append(started_process)
        raise RuntimeError("start failed")

    cfg = {
        "sim2real_runtime": "multiprocess",
        "input": {"provider": "pico4"},
        "multiprocess": {"shutdown_timeout_s": 0.01},
    }
    controller = MultiprocessSim2RealController(cfg)
    monkeypatch.setattr(controller, "_start_processes", fake_start_processes.__get__(controller))

    with pytest.raises(RuntimeError, match="start failed"):
        controller.run()

    assert started_process.join_calls == [0.01, 1.0]
    assert started_process.terminated is True
    assert controller._processes == []


def test_human_frame_validation_rejects_bad_inputs() -> None:
    valid_frame = {
        "Pelvis": (np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    }
    assert _human_frame_is_valid(valid_frame, max_pos_value=5.0)

    bad_frame = {
        "Pelvis": (np.array([6.0, 0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    }
    assert not _human_frame_is_valid(bad_frame, max_pos_value=5.0)


def test_robot_worker_requires_consecutive_valid_references_and_reset_generation(monkeypatch) -> None:
    worker = object.__new__(_RobotControlWorker)
    worker._reference_reset_seq = 0
    worker._latest_reference = None
    worker._last_reference_seq = -1
    worker._consecutive_valid_references = 0
    worker._check_frames = 2
    worker._max_reference_age_s = 0.25
    worker._reference_age_s = lambda: 0.0
    worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    worker._last_commanded_motion_qpos = np.zeros(36, dtype=np.float64)
    worker._mode_pub_events: list[tuple[str, object]] = []
    worker._mode_pub = SimpleNamespace(
        publish=lambda topic, payload: worker._mode_pub_events.append((topic, payload))
    )

    valid0 = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.0,
        seq=1,
        source_timestamp_s=1.0,
        source_seq=1,
        frame_valid=True,
        reference_reset_seq=0,
    )
    worker._note_reference_packet(valid0)
    assert worker._can_switch_to_mocap() is False

    valid1 = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.1,
        seq=2,
        source_timestamp_s=1.1,
        source_seq=2,
        frame_valid=True,
        reference_reset_seq=0,
    )
    worker._note_reference_packet(valid1)
    assert worker._can_switch_to_mocap() is True

    invalid = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.2,
        seq=3,
        source_timestamp_s=1.2,
        source_seq=3,
        frame_valid=False,
        reference_reset_seq=0,
    )
    worker._note_reference_packet(invalid)
    assert worker._can_switch_to_mocap() is False

    worker._publish_reference_reset("enter_standing")
    assert worker._reference_reset_seq == 1
    assert worker._latest_reference is None
    assert worker._consecutive_valid_references == 0
    assert worker._mode_pub_events
    topic, payload = worker._mode_pub_events[-1]
    assert topic == REFERENCE_RESET_TOPIC
    assert isinstance(payload, ReferenceResetPacket)
    assert payload.seq == 1
    assert payload.reason == "enter_standing"

    old_packet = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.3,
        seq=4,
        source_timestamp_s=1.3,
        source_seq=4,
        frame_valid=True,
        reference_reset_seq=0,
    )
    worker._note_reference_packet(old_packet)
    assert worker._latest_reference is None
    assert worker._consecutive_valid_references == 0

    fresh_packet = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.4,
        seq=5,
        source_timestamp_s=1.4,
        source_seq=5,
        frame_valid=True,
        reference_reset_seq=1,
    )
    worker._note_reference_packet(fresh_packet)
    assert worker._latest_reference == fresh_packet
    assert worker._consecutive_valid_references == 1
