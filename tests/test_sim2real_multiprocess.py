from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from teleopit.runtime.mocap_session import MocapSessionState
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType
from teleopit.runtime.arm_mocap import compose_arm_reference, compose_arm_reference_window
from teleopit.recording.hdf5 import (
    ACTION_KEY,
    FRAME_INDEX_KEY,
    HAND_ACTION_KEY,
    HDF5_RECORDING_FORMAT,
    IMAGE_KEY,
    MODE_KEY,
    STATE_KEY,
    TIMESTAMP_KEY,
    build_mode_observation,
    build_observation_state,
    build_recording_schema,
    hdf5_schema,
)
from teleopit.sim2real.mp.ipc import HEALTH_TOPIC, LatestSubscriber, ZmqPublisher
from teleopit.sim2real.mp.messages import HandCommandPacket, ModeStatePacket, RecordStepPacket, ReferencePacket, SharedFrameDescriptor
from teleopit.sim.reference_timeline import ReferenceSample, ReferenceWindow
from teleopit.sim2real.mp.runtime import (
    map_recording_key_to_command,
    RobotMode,
    Sim2RealRuntime,
    _LoopTimingReporter,
    _RecordingWorker,
    _RobotControlWorker,
    _configured_open_hand_pose,
    _hand_worker_active_for_mode,
    _human_frame_is_valid,
)
from teleopit.sim2real.mp.shm import SharedFrameRingReader, SharedFrameRingWriter


def test_loop_timing_reporter_separates_late_sleep_from_work_overrun(caplog) -> None:
    reporter = _LoopTimingReporter(target_period_s=0.02, log_interval_s=1.0, deadline_miss_tolerance_s=0.001)

    with caplog.at_level(logging.INFO, logger="teleopit.operator"):
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


def test_sim2real_runtime_rejects_recording_without_pico_provider() -> None:
    cfg = {
        "input": {"provider": "bvh"},
        "runtime": {"shutdown_timeout_s": 0.01},
        "recording": {"enabled": True},
    }
    with pytest.raises(ValueError, match="recording.enabled=true requires input.provider=pico4"):
        Sim2RealRuntime(cfg)


def test_sim2real_runtime_rejects_recording_without_input_video() -> None:
    cfg = {
        "input": {"provider": "pico4", "video": {"enabled": False, "source": "realsense"}},
        "runtime": {"shutdown_timeout_s": 0.01},
        "recording": {"enabled": True},
    }
    with pytest.raises(ValueError, match="recording.enabled=true requires input.video.enabled=true"):
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


def test_recording_enabled_adds_recording_worker(monkeypatch) -> None:
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
        "input": {
            "provider": "pico4",
            "video": {"enabled": True, "source": "realsense", "width": 640, "height": 480, "fps": 30},
        },
        "runtime": {"shutdown_timeout_s": 0.01},
        "recording": {"enabled": True},
    }
    monkeypatch.setattr("teleopit.sim2real.mp.runtime._require_recording_dependencies", lambda: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    runtime = Sim2RealRuntime(cfg)
    runtime._ctx = FakeContext()  # type: ignore[assignment]

    runtime._start_processes()

    assert started_names == ["pico_input", "reference", "robot_control", "recording_worker"]


def test_recording_key_mapping() -> None:
    assert map_recording_key_to_command("R") == "record_start"
    assert map_recording_key_to_command("s") == "record_save"
    assert map_recording_key_to_command("D") == "record_discard"
    assert map_recording_key_to_command("q") == "shutdown"
    assert map_recording_key_to_command("x") is None


def test_hdf5_recording_schema() -> None:
    schema = build_recording_schema({"width": 640, "height": 480, "key": IMAGE_KEY})
    sidecar = hdf5_schema(schema)
    features = sidecar["features"]

    assert sidecar["format"] == HDF5_RECORDING_FORMAT
    assert features[IMAGE_KEY]["type"] == "video"
    assert features[IMAGE_KEY]["format"] == "mp4"
    assert features[IMAGE_KEY]["shape"] == [480, 640, 3]
    assert features[FRAME_INDEX_KEY]["dtype"] == "int64"
    assert features[TIMESTAMP_KEY]["dtype"] == "float64"
    assert features[STATE_KEY]["shape"] == [68]
    assert features[MODE_KEY]["shape"] == [1]
    assert features[ACTION_KEY]["shape"] == [36]
    assert features[HAND_ACTION_KEY]["shape"] == [12]
    assert sidecar["features"][STATE_KEY]["slices"]["joint_pos"] == [0, 29]
    assert sidecar["features"][STATE_KEY]["slices"]["projected_gravity"] == [65, 68]
    assert sidecar["features"][MODE_KEY]["codes"]["pause"] == 3
    assert sidecar["features"][ACTION_KEY]["slices"]["joint_pos"] == [7, 36]
    assert sidecar["features"][HAND_ACTION_KEY]["slices"]["left_pose"] == [0, 6]
    assert sidecar["features"][HAND_ACTION_KEY]["slices"]["right_pose"] == [6, 12]


def test_hdf5_recorder_mp4_sidecar_writes_sync_metadata(tmp_path: Path) -> None:
    from teleopit.recording.hdf5 import MP4VideoConfig, TeleopitHDF5Recorder

    schema = build_recording_schema({"width": 2, "height": 2, "key": IMAGE_KEY})
    recorder = TeleopitHDF5Recorder.create(
        output_dir=tmp_path,
        task="walk",
        fps=30,
        schema=schema,
        video_config=MP4VideoConfig(quality=5),
    )

    recorder.start_episode()
    for idx in range(2):
        recorder.add_frame(
            image=np.full((2, 2, 3), idx * 64, dtype=np.uint8),
            state=np.arange(68, dtype=np.float32),
            mode=build_mode_observation("mocap"),
            action=np.arange(36, dtype=np.float32),
            hand_action=np.arange(12, dtype=np.float32),
            task="walk",
        )
    recorder.save_episode()
    recorder.finalize()

    episodes = sorted((tmp_path / "episodes").glob("*.h5"))
    videos = sorted((tmp_path / "videos" / "observation.images.d435i_rgb").glob("*.mp4"))
    assert len(episodes) == 1
    assert len(videos) == 1
    assert videos[0].stat().st_size > 0
    assert (tmp_path / "schema.json").exists()
    assert not list((tmp_path / ".tmp").glob("*.h5"))

    with h5py.File(episodes[0], "r") as h5:
        assert h5.attrs["format"] == HDF5_RECORDING_FORMAT
        assert h5.attrs["version"] == 1
        assert h5.attrs["task"] == "walk"
        assert h5.attrs["fps"] == 30
        assert h5.attrs["frames"] == 2
        assert h5.attrs["video_path"] == videos[0].relative_to(tmp_path).as_posix()
        assert h5.attrs["video_key"] == IMAGE_KEY
        assert h5.attrs["video_frames"] == 2
        assert h5.attrs["video_fps"] == 30
        assert IMAGE_KEY not in h5
        assert h5[FRAME_INDEX_KEY].shape == (2,)
        assert h5[TIMESTAMP_KEY].shape == (2,)
        np.testing.assert_array_equal(h5[FRAME_INDEX_KEY][...], np.array([0, 1], dtype=np.int64))
        np.testing.assert_allclose(h5[TIMESTAMP_KEY][...], np.array([0.0, 1.0 / 30.0], dtype=np.float64))
        assert h5[STATE_KEY].shape == (2, 68)
        assert h5[MODE_KEY].shape == (2, 1)
        assert h5[ACTION_KEY].shape == (2, 36)
        assert h5[HAND_ACTION_KEY].shape == (2, 12)


def test_hdf5_recorder_cleans_partial_episode_when_video_writer_fails(tmp_path: Path) -> None:
    from teleopit.recording.hdf5 import TeleopitHDF5Recorder

    class FailingVideoRecorder(TeleopitHDF5Recorder):
        def _create_video_writer(self, path: Path) -> object:
            path.write_bytes(b"partial")
            raise RuntimeError("writer failed")

    schema = build_recording_schema({"width": 2, "height": 2, "key": IMAGE_KEY})
    recorder = FailingVideoRecorder.create(output_dir=tmp_path, task="walk", fps=30, schema=schema)

    with pytest.raises(RuntimeError, match="writer failed"):
        recorder.start_episode()

    recorder.finalize()
    assert not list((tmp_path / ".tmp").glob("*.h5"))
    assert not list((tmp_path / ".tmp" / "videos" / "observation.images.d435i_rgb").glob("*.mp4"))
    assert not list((tmp_path / "episodes").glob("*.h5"))
    assert not list((tmp_path / "videos" / "observation.images.d435i_rgb").glob("*.mp4"))


def test_hdf5_recorder_keeps_startup_error_when_partial_cleanup_fails(tmp_path: Path) -> None:
    from teleopit.recording.hdf5 import TeleopitHDF5Recorder

    class BrokenWriter:
        def close(self) -> None:
            raise RuntimeError("cleanup failed")

    class FailingDatasetRecorder(TeleopitHDF5Recorder):
        def _create_video_writer(self, path: Path) -> object:
            return BrokenWriter()

        def _create_datasets(self, h5: h5py.File) -> dict[str, h5py.Dataset]:
            raise RuntimeError("startup failed")

    schema = build_recording_schema({"width": 2, "height": 2, "key": IMAGE_KEY})
    recorder = FailingDatasetRecorder.create(output_dir=tmp_path, task="walk", fps=30, schema=schema)

    with pytest.raises(RuntimeError, match="startup failed"):
        recorder.start_episode()

    recorder.finalize()
    assert not list((tmp_path / ".tmp").glob("*.h5"))


def test_configured_open_hand_pose_matches_linkerhand_l6_parser() -> None:
    left, right = _configured_open_hand_pose(
        {
            "hands": {
                "enabled": True,
                "driver": "linkerhand_l6",
                "mode": "gripper",
                "linkerhand_l6": {
                    "thumb_yaw_center": 42,
                    "open_pose": [250, 99, 250, 250, 250, 250],
                },
            },
        }
    )

    np.testing.assert_allclose(left, np.array([250, 42, 250, 250, 250, 250], dtype=np.float32))
    np.testing.assert_allclose(right, left)


def test_configured_open_hand_pose_defaults_without_enabled_hands() -> None:
    left, right = _configured_open_hand_pose({})

    np.testing.assert_allclose(left, np.array([250, 10, 250, 250, 250, 250], dtype=np.float32))
    np.testing.assert_allclose(right, left)


def test_record_observation_state_concat_order() -> None:
    state = SimpleNamespace(
        qpos=np.arange(29, dtype=np.float32),
        qvel=np.arange(29, dtype=np.float32) + 100.0,
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ang_vel=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    out = build_observation_state(state)

    assert out.shape == (68,)
    np.testing.assert_allclose(out[0:29], state.qpos)
    np.testing.assert_allclose(out[29:58], state.qvel)
    np.testing.assert_allclose(out[58:62], state.quat)
    np.testing.assert_allclose(out[62:65], state.ang_vel)
    np.testing.assert_allclose(out[65:68], np.array([0.0, 0.0, -1.0], dtype=np.float32))


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


def test_robot_worker_standing_reference_uses_default_root_height_without_base_pos() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.default_angles = np.zeros(29, dtype=np.float32)
    worker.num_actions = 29
    worker._default_root_pos = np.array([0.0, 0.0, 0.76], dtype=np.float64)
    worker._standing_qpos = np.zeros(36, dtype=np.float64)
    state = SimpleNamespace(
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        qpos=np.zeros(29, dtype=np.float32),
    )

    worker._set_default_standing_reference(state)

    np.testing.assert_allclose(worker._standing_qpos[0:3], np.array([0.0, 0.0, 0.76]))


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


@pytest.mark.parametrize(
    ("mode", "mocap_active", "mocap_paused"),
    [
        ("standing", False, False),
        ("mocap", True, False),
        ("arms", True, False),
        ("mocap", False, True),
        ("arms", False, True),
        ("damping", False, False),
    ],
)
def test_hand_worker_stays_active_in_all_modes(mode: str, mocap_active: bool, mocap_paused: bool) -> None:
    packet = ModeStatePacket(
        mode=mode,
        mocap_active=mocap_active,
        mocap_paused=mocap_paused,
        timestamp_s=1.0,
        seq=1,
    )

    assert _hand_worker_active_for_mode(packet) is True


def test_hand_worker_active_state_only_updates_from_mode_packets() -> None:
    active = False
    mode_packet = None
    if isinstance(mode_packet, ModeStatePacket):
        active = _hand_worker_active_for_mode(mode_packet)
    assert active is False

    mode_packet = ModeStatePacket(
        mode="standing",
        mocap_active=False,
        mocap_paused=False,
        timestamp_s=1.0,
        seq=1,
    )
    if isinstance(mode_packet, ModeStatePacket):
        active = _hand_worker_active_for_mode(mode_packet)
    assert active is True

    mode_packet = None
    if isinstance(mode_packet, ModeStatePacket):
        active = _hand_worker_active_for_mode(mode_packet)
    assert active is True


def test_robot_worker_publish_record_step() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.mode = RobotMode.ARMS
    worker._mocap_session = SimpleNamespace(state=MocapSessionState.ACTIVE)
    worker._mode_seq = 7
    published: list[tuple[str, object]] = []
    worker._record_pub = SimpleNamespace(publish=lambda topic, packet: published.append((topic, packet)))
    robot_state = SimpleNamespace(
        qpos=np.arange(29, dtype=np.float32),
        qvel=np.arange(29, dtype=np.float32) + 10.0,
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ang_vel=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    reference_qpos = np.arange(36, dtype=np.float64)

    worker._publish_record_step(robot_state=robot_state, reference_qpos=reference_qpos)

    assert len(published) == 1
    packet = published[0][1]
    assert isinstance(packet, RecordStepPacket)
    assert packet.mode == "arms"
    assert packet.mocap_active is True
    assert packet.recordable is True
    assert packet.observation_state.shape == (68,)
    assert packet.observation_mode.shape == (1,)
    assert packet.action_reference_qpos.shape == (36,)
    np.testing.assert_allclose(packet.observation_mode, build_mode_observation("arms"))
    np.testing.assert_allclose(packet.action_reference_qpos, reference_qpos.astype(np.float32))


def test_robot_worker_enter_damping_publishes_non_recordable_packet() -> None:
    worker = object.__new__(_RobotControlWorker)
    worker.mode = RobotMode.MOCAP
    worker._mode_seq = 9
    worker._mocap_reentry_armed = True
    worker._last_commanded_motion_qpos = np.ones(36, dtype=np.float64)
    worker._last_mocap_hold_reason = "stale"
    worker._default_root_pos = np.zeros(3, dtype=np.float64)
    worker.num_actions = 29
    worker._mocap_session = SimpleNamespace(reset=lambda: None)
    worker._ref_proc = SimpleNamespace(last_reference_qpos=np.zeros(36, dtype=np.float64))
    robot_state = SimpleNamespace(
        qpos=np.arange(29, dtype=np.float32),
        qvel=np.arange(29, dtype=np.float32) + 10.0,
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ang_vel=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        base_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
    )
    worker.robot = SimpleNamespace(
        set_damping=lambda: None,
        exit_debug_mode=lambda: None,
        get_state=lambda: robot_state,
    )
    published: list[tuple[str, object]] = []
    worker._record_pub = SimpleNamespace(publish=lambda topic, packet: published.append((topic, packet)))

    worker._enter_damping()

    assert worker.mode == RobotMode.DAMPING
    assert published
    packet = published[-1][1]
    assert isinstance(packet, RecordStepPacket)
    assert packet.mode == "damping"
    assert packet.recordable is False
    assert packet.mocap_active is False
    np.testing.assert_allclose(packet.observation_mode, np.array([-1.0], dtype=np.float32))


def test_recording_worker_start_save_discard_with_fake_adapter() -> None:
    from teleopit.sim2real.mp.ipc import default_endpoints

    calls: list[str] = []
    frames: list[dict[str, np.ndarray]] = []

    class FakeRecorder:
        def start_episode(self) -> None:
            calls.append("start")

        def add_frame(
            self,
            *,
            image: np.ndarray,
            state: np.ndarray,
            mode: np.ndarray,
            action: np.ndarray,
            hand_action: np.ndarray,
            task: str,
        ) -> None:
            calls.append(f"frame:{task}")
            frames.append(
                {
                    "image": image.copy(),
                    "state": state.copy(),
                    "mode": mode.copy(),
                    "action": action.copy(),
                    "hand_action": hand_action.copy(),
                }
            )

        def save_episode(self) -> None:
            calls.append("save")

        def discard_episode(self) -> None:
            calls.append("discard")

        def finalize(self) -> None:
            calls.append("finalize")

    def fake_factory(**_kwargs: object) -> FakeRecorder:
        return FakeRecorder()

    stop_event = SimpleNamespace(is_set=lambda: False, set=lambda: None)
    endpoints = default_endpoints(base_port=39850)
    worker = _RecordingWorker(
        {
            "recording": {
                "enabled": True,
                "task": "walk",
                "fps": 30,
                "min_episode_seconds": 0.0,
                "camera": {"width": 2, "height": 2, "key": IMAGE_KEY},
            }
        },
        endpoints,
        stop_event,  # type: ignore[arg-type]
        recorder_factory=fake_factory,
    )
    writer = SharedFrameRingWriter(shape=(2, 2, 3), dtype=np.uint8, slots=2)
    try:
        worker._latest_record = RecordStepPacket(
            timestamp_s=1.0,
            mode="damping",
            mocap_active=False,
            recordable=False,
            observation_state=np.ones(68, dtype=np.float32),
            observation_mode=build_mode_observation("standing"),
            action_reference_qpos=np.ones(36, dtype=np.float32),
            seq=1,
        )
        worker._start_episode()
        assert calls == []

        worker._latest_record = RecordStepPacket(
            timestamp_s=2.0,
            mode="standing",
            mocap_active=False,
            recordable=True,
            observation_state=np.arange(68, dtype=np.float32),
            observation_mode=build_mode_observation("standing"),
            action_reference_qpos=np.arange(36, dtype=np.float32),
            seq=2,
        )
        worker._start_episode()
        worker._save_episode()
        assert calls == ["start", "discard"]

        worker._start_episode()
        worker._latest_hand_command = HandCommandPacket(
            timestamp_s=2.05,
            driver="linkerhand_l6",
            mode="gripper",
            active=True,
            left_pose=np.arange(6, dtype=np.float32),
            right_pose=np.arange(6, 12, dtype=np.float32),
            seq=1,
        )
        desc = writer.write(np.full((2, 2, 3), 5, dtype=np.uint8), timestamp_s=2.1)
        worker._handle_video(desc)
        worker._save_episode()

        assert calls == ["start", "discard", "start", "frame:walk", "save"]
        assert frames[0]["image"].shape == (2, 2, 3)
        np.testing.assert_allclose(frames[0]["state"], np.arange(68, dtype=np.float32))
        np.testing.assert_allclose(frames[0]["mode"], build_mode_observation("standing"))
        np.testing.assert_allclose(frames[0]["action"], np.arange(36, dtype=np.float32))
        np.testing.assert_allclose(frames[0]["hand_action"], np.arange(12, dtype=np.float32))

        worker._latest_record = RecordStepPacket(
            timestamp_s=3.0,
            mode="pause",
            mocap_active=False,
            recordable=True,
            observation_state=np.zeros(68, dtype=np.float32),
            observation_mode=build_mode_observation("pause"),
            action_reference_qpos=np.zeros(36, dtype=np.float32),
            seq=3,
        )
        worker._start_episode()
        worker._discard_episode("test")
        assert calls[-2:] == ["start", "discard"]
    finally:
        writer.close(unlink=True)
        worker._record_sub.close()
        worker._video_sub.close()
        worker._hand_command_sub.close()
        worker._command_sub.close()
        worker._frame_reader.close()
