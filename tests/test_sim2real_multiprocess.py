from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from teleopit.runtime.mocap_session import MocapSessionState
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType
from teleopit.runtime.arm_mocap import compose_arm_reference, compose_arm_reference_window
from teleopit.sim2real.mp.ipc import HEALTH_TOPIC, LatestSubscriber, ZmqPublisher
from teleopit.sim2real.mp.messages import ReferencePacket, SharedFrameDescriptor
from teleopit.sim.reference_timeline import ReferenceSample, ReferenceWindow
from teleopit.sim2real.mp.runtime import (
    RobotMode,
    Sim2RealRuntime,
    _LoopTimingReporter,
    _RobotControlWorker,
    _human_frame_is_valid,
)
from teleopit.sim2real.mp.shm import SharedFrameRingReader, SharedFrameRingWriter


def test_loop_timing_reporter_separates_late_sleep_from_work_overrun(caplog) -> None:
    reporter = _LoopTimingReporter(target_period_s=0.02, log_interval_s=1.0, deadline_miss_tolerance_s=0.001)

    with caplog.at_level(logging.INFO, logger="teleopit.sim2real.mp.runtime"):
        reporter.record(loop_start_s=0.0, work_elapsed_s=0.0004, cycle_elapsed_s=0.02006, pico_age_s=None)
        reporter.record(loop_start_s=1.0, work_elapsed_s=0.021, cycle_elapsed_s=0.0212, pico_age_s=None)

    message = caplog.messages[-1]
    assert "late_ms" in message
    assert "deadline_miss(>1.00ms)=1/2" in message
    assert "work_overrun=1/2" in message
    assert " overrun=" not in message


def test_sim2real_runtime_rejects_legacy_runtime_keys() -> None:
    cfg = {"sim2real_runtime": "auto", "input": {"provider": "pico4"}}
    with pytest.raises(ValueError, match="Legacy sim2real config keys"):
        Sim2RealRuntime(cfg)


def test_sim2real_runtime_accepts_bvh_provider() -> None:
    cfg = {"input": {"provider": "bvh"}, "runtime": {"shutdown_timeout_s": 0.01}}
    runtime = Sim2RealRuntime(cfg)
    runtime.shutdown()


def test_sim2real_runtime_rejects_hands_without_pico_provider() -> None:
    cfg = {
        "input": {"provider": "bvh"},
        "runtime": {"shutdown_timeout_s": 0.01},
        "hands": {"enabled": True, "driver": "linkerhand_l6", "mode": "gripper"},
    }
    with pytest.raises(ValueError, match="hands.enabled=true requires input.provider=pico4"):
        Sim2RealRuntime(cfg)


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
        "input": {"provider": "pico4", "video": {"enabled": True, "source": "mujoco"}},
        "runtime": {},
    }
    with pytest.raises(ValueError, match="only supports input.video.source=realsense or test-pattern"):
        Sim2RealRuntime(cfg)


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


def test_run_sim2real_shutdowns_on_exception(monkeypatch) -> None:
    script_path = Path.cwd() / "scripts" / "run" / "run_sim2real.py"
    spec = importlib.util.spec_from_file_location("test_run_sim2real", script_path)
    assert spec is not None and spec.loader is not None
    run_sim2real = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_sim2real)

    calls: list[str] = []

    class FailingRuntime:
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
    monkeypatch.setattr(run_sim2real, "Sim2RealRuntime", FailingRuntime)

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

    started_process = FakeProcess(name="pico_input")

    def fake_start_processes(self: Sim2RealRuntime) -> None:
        started_process.started = True
        self._processes.append(started_process)
        raise RuntimeError("start failed")

    cfg = {
        "input": {"provider": "pico4"},
        "runtime": {"shutdown_timeout_s": 0.01},
    }
    controller = Sim2RealRuntime(cfg)
    monkeypatch.setattr(controller, "_start_processes", fake_start_processes.__get__(controller))

    with pytest.raises(RuntimeError, match="start failed"):
        controller.run()

    assert started_process.join_calls == [0.01, 1.0]
    assert started_process.terminated is True
    assert controller._processes == []


def test_pico_video_does_not_spawn_separate_video_worker() -> None:
    started_names: list[str] = []

    class FakeProcess:
        def __init__(self, *, name: str, target: object, args: tuple[object, ...]) -> None:
            del target, args
            self.name = name
            self.exitcode = 0

        def start(self) -> None:
            started_names.append(self.name)

    class FakeContext:
        def Event(self) -> object:
            return SimpleNamespace(set=lambda: None, is_set=lambda: False)

        def Process(self, *, name: str, target: object, args: tuple[object, ...]) -> FakeProcess:
            return FakeProcess(name=name, target=target, args=args)

    cfg = {
        "input": {"provider": "pico4", "video": {"enabled": True, "source": "realsense"}},
        "runtime": {"shutdown_timeout_s": 0.01},
    }
    runtime = Sim2RealRuntime(cfg)
    runtime._ctx = FakeContext()  # type: ignore[assignment]

    runtime._start_processes()

    assert started_names == ["pico_input", "reference", "robot_control"]


def test_human_frame_validation_rejects_bad_inputs() -> None:
    valid_frame = {
        "Pelvis": (np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    }
    assert _human_frame_is_valid(valid_frame)

    bad_frame = {
        "Pelvis": (np.array([np.nan, 0.0, 0.0], dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)),
    }
    assert not _human_frame_is_valid(bad_frame)


def test_robot_worker_holds_stale_reference_instead_of_damping() -> None:
    worker = object.__new__(_RobotControlWorker)
    hold_qpos = np.zeros(36, dtype=np.float64)
    hold_qpos[3] = 1.0
    hold_qpos[7] = 0.25
    worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    worker._latest_reference = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.0,
        seq=1,
        source_timestamp_s=1.0,
        source_seq=1,
        frame_valid=True,
    )
    worker._reference_age_s = lambda: 0.30
    worker._stale_reference_hold_s = 0.08
    worker._last_mocap_hold_reason = None
    worker._last_commanded_motion_qpos = hold_qpos.copy()
    worker._resolve_mocap_hold_qpos = lambda: hold_qpos.copy()
    held: list[np.ndarray] = []
    worker._run_static_mocap_step = lambda qpos: held.append(np.asarray(qpos, dtype=np.float64).copy())
    worker._enter_damping = lambda: pytest.fail("stale references must not enter damping")

    worker._mocap_step()

    assert len(held) == 1
    np.testing.assert_allclose(held[0], hold_qpos)


def test_robot_worker_requires_consecutive_valid_references(monkeypatch) -> None:
    worker = object.__new__(_RobotControlWorker)
    worker._latest_reference = None
    worker._last_reference_seq = -1
    worker._consecutive_valid_references = 0
    worker._check_frames = 2
    worker._max_reference_age_s = 0.25
    worker.provider_kind = "pico4"
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
    )
    worker._note_reference_packet(invalid)
    assert worker._can_switch_to_mocap() is False

    fresh_packet = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.4,
        seq=4,
        source_timestamp_s=1.4,
        source_seq=4,
        frame_valid=True,
    )
    worker._note_reference_packet(fresh_packet)
    assert worker._latest_reference == fresh_packet
    assert worker._consecutive_valid_references == 1


def test_robot_worker_replays_bvh_on_mocap_entry() -> None:
    worker = object.__new__(_RobotControlWorker)
    commands: list[str] = []
    worker.provider_kind = "bvh"
    worker.robot = SimpleNamespace(get_state=lambda: SimpleNamespace())
    worker._standing_qpos = np.zeros(36, dtype=np.float64)
    worker._mocap_reentry_armed = True
    worker._reset_policy_state = lambda: None
    worker._build_resume_alignment_qpos = lambda _hold, _state: np.ones(36, dtype=np.float64)
    worker._ref_proc = SimpleNamespace(reset_alignment=lambda **_kwargs: None)
    worker._send_reference_command = commands.append

    worker._transition_to_mocap()

    assert worker.mode == RobotMode.MOCAP
    assert commands == ["replay_mocap"]


def test_robot_worker_pauses_when_bvh_reference_reports_paused() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.provider_kind = "bvh"
    worker.mode = RobotMode.MOCAP
    worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    worker._last_reference_seq = -1
    worker._consecutive_valid_references = 0
    paused: list[str] = []

    def pause_active_mocap() -> None:
        paused.append("pause")
        worker._mocap_session.state = MocapSessionState.PAUSED

    worker._pause_active_mocap = pause_active_mocap
    packet = ReferencePacket(
        qpos=np.zeros(36, dtype=np.float64),
        timestamp_s=1.0,
        seq=1,
        source_timestamp_s=0.0,
        source_seq=0,
        playback_paused=True,
        playback_finished=True,
    )

    worker._note_reference_packet(packet)

    assert paused == ["pause"]
    assert worker._latest_reference is packet


def test_robot_worker_composes_arm_reference_from_standing_pose() -> None:
    standing_qpos = np.arange(36, dtype=np.float64)
    retarget = np.full(36, 100.0, dtype=np.float64)
    retarget[7 + 15:7 + 29] = np.arange(14, dtype=np.float64) + 200.0

    composed = compose_arm_reference(
        standing_qpos=standing_qpos,
        retarget_qpos=retarget,
        arm_joint_indices=np.arange(15, 29, dtype=np.int64),
        num_actions=29,
    )

    np.testing.assert_allclose(composed[: 7 + 15], standing_qpos[: 7 + 15])
    np.testing.assert_allclose(composed[7 + 15:7 + 29], retarget[7 + 15:7 + 29])


def test_robot_worker_composes_arm_reference_window_samples() -> None:
    standing_qpos = np.zeros(36, dtype=np.float64)
    qpos0 = np.ones(36, dtype=np.float64)
    qpos1 = np.full(36, 2.0, dtype=np.float64)
    window = ReferenceWindow(
        base_time_s=1.0,
        policy_dt_s=0.02,
        reference_steps=(0, 1),
        samples=(
            ReferenceSample(qpos=qpos0, timestamp_s=1.0, mode="a", used_fallback=False, older_timestamp_s=None, newer_timestamp_s=None, alpha=None),
            ReferenceSample(qpos=qpos1, timestamp_s=1.02, mode="b", used_fallback=False, older_timestamp_s=None, newer_timestamp_s=None, alpha=None),
        ),
    )

    composed = compose_arm_reference_window(
        window,
        standing_qpos=standing_qpos,
        arm_joint_indices=np.arange(15, 29, dtype=np.int64),
        num_actions=29,
    )

    assert composed is not None
    np.testing.assert_allclose(composed.samples[0].qpos[7 + 15:7 + 29], 1.0)
    np.testing.assert_allclose(composed.samples[1].qpos[7 + 15:7 + 29], 2.0)
    np.testing.assert_allclose(composed.samples[0].qpos[:7 + 15], 0.0)
    np.testing.assert_allclose(composed.samples[1].qpos[:7 + 15], 0.0)


def test_robot_worker_pico_arms_event_toggles_mocap_and_arms() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.provider_kind = "pico4"
    worker.mode = RobotMode.MOCAP
    worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    worker.robot = SimpleNamespace(get_state=lambda: SimpleNamespace(base_pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]), qpos=np.zeros(29)))
    worker._last_commanded_motion_qpos = np.zeros(36, dtype=np.float64)
    worker._build_resume_alignment_qpos = lambda _hold, _state: np.ones(36, dtype=np.float64)
    worker._set_default_standing_reference = lambda _state: None
    worker._reset_policy_state = lambda: None
    worker._last_retarget_qpos = np.zeros(36, dtype=np.float64)
    resets: list[np.ndarray] = []
    ramps: list[str] = []
    worker._ref_proc = SimpleNamespace(reset_alignment=lambda *, target_qpos=None: resets.append(np.asarray(target_qpos).copy()))
    worker._standing_return_ramp_duration = 0.5
    worker._standing_return_kp_ramp_floor_ratio = 0.5
    worker._safety = SimpleNamespace(start_kp_ramp=lambda **_kwargs: ramps.append("ramp"))

    event = ControlEvent(event_type=ControlEventType.TOGGLE_ARMS, source="test")
    worker._handle_mocap_control_events((event,))
    assert worker.mode == RobotMode.ARMS

    worker._handle_mocap_control_events((event,))
    assert worker.mode == RobotMode.MOCAP
    assert len(resets) == 2
    assert ramps == ["ramp", "ramp"]


def test_robot_worker_bvh_ignores_pico_arms_event() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.provider_kind = "bvh"
    worker.mode = RobotMode.MOCAP
    worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    worker._handle_mocap_control_events((
        ControlEvent(event_type=ControlEventType.TOGGLE_ARMS, source="test"),
    ))

    assert worker.mode == RobotMode.MOCAP


def test_robot_worker_mode_state_marks_arms_as_mocap_active() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.mode = RobotMode.ARMS
    worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    worker._mode_seq = 0
    published: list[object] = []
    worker._mode_pub = SimpleNamespace(publish=lambda _topic, packet: published.append(packet))

    worker._publish_mode_state()

    packet = published[-1]
    assert packet.mode == "arms"
    assert packet.mocap_active is True
    assert packet.mocap_paused is False
