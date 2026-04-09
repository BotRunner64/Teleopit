from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_mujoco
from teleopit.bus.in_process import InProcessBus
from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE
from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType, RealtimeInputPacket
from teleopit.interfaces import RobotState
from teleopit.runtime.terminal_keyboard import TerminalKeyEvent


class _DummyRobot:
    def __init__(self) -> None:
        self.num_actions = 2
        self.kps = np.array([1.0, 1.0], dtype=np.float32)
        self.kds = np.array([0.1, 0.1], dtype=np.float32)
        self.torque_limits = np.array([10.0, 10.0], dtype=np.float32)
        self.default_dof_pos = np.array([0.5, -0.5], dtype=np.float32)
        self._last_action = np.zeros(2, dtype=np.float32)
        self._qpos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._qvel = np.zeros(3, dtype=np.float32)
        self._timestamp = 0.0

    def get_state(self) -> RobotState:
        return RobotState(
            qpos=self._qpos.copy(),
            qvel=self._qvel.copy(),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ang_vel=np.zeros(3, dtype=np.float32),
            timestamp=self._timestamp,
            base_pos=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            base_lin_vel=np.zeros(3, dtype=np.float32),
        )

    def set_action(self, action: np.ndarray) -> None:
        self._last_action = np.asarray(action, dtype=np.float32)[:2]

    def step(self) -> None:
        self._qvel[:2] = self._last_action
        self._qpos[:2] += self._last_action * 0.01
        self._timestamp += 0.02


class _DummyController:
    _expected_obs_dim = 4

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        return np.array([0.1, -0.1], dtype=np.float32)

    def reset(self) -> None:
        pass


class _DummyObsBuilder:
    def __init__(self) -> None:
        self.mimic_obs_calls: list[np.ndarray] = []
        self._base = _DummyObsBuilderBase()

    def reset(self) -> None:
        pass

    def build(
        self,
        state: object,
        motion_qpos: np.ndarray,
        motion_joint_vel: np.ndarray,
        last_action: np.ndarray,
        motion_anchor_lin_vel_w: np.ndarray,
        motion_anchor_ang_vel_w: np.ndarray,
    ) -> np.ndarray:
        del state, motion_joint_vel, last_action, motion_anchor_lin_vel_w, motion_anchor_ang_vel_w
        self.mimic_obs_calls.append(np.asarray(motion_qpos[:1], dtype=np.float32).copy())
        return np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)


class _DummyObsBuilderBase:
    def __init__(self) -> None:
        self._anchor_body_id = 0
        self._last_pos = np.zeros(3, dtype=np.float32)
        self._last_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _run_fk(self, pos: np.ndarray, quat: np.ndarray, joints: np.ndarray) -> None:
        del joints
        self._last_pos = np.asarray(pos, dtype=np.float32).copy()
        self._last_quat = np.asarray(quat, dtype=np.float32).copy()

    def _get_body_pos(self, body_id: int) -> np.ndarray:
        del body_id
        return self._last_pos

    def _get_body_quat(self, body_id: int) -> np.ndarray:
        del body_id
        return self._last_quat


class _DummyInputProvider:
    fps = 1

    def __init__(self) -> None:
        self._frames = 0

    def get_frame(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self._frames += 1
        return {"Pelvis": (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))}

    def is_available(self) -> bool:
        return self._frames < 3


class _DummyRetargeter:
    def retarget(self, human_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pelvis_x = float(human_data["Pelvis"][0][0])
        return (
            np.array([pelvis_x, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.zeros(29, dtype=np.float64),
        )

    def reset(self) -> None:
        pass


class _DummyRecorder:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def add_frame(self, data: dict[str, object]) -> None:
        self.frames.append(data)


@requires_mujoco
def test_simulation_loop_runs_and_records_without_viewers() -> None:
    from teleopit.sim.loop import SimulationLoop

    bus = InProcessBus()
    robot = _DummyRobot()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=_DummyObsBuilder(),
        bus=bus,
        cfg={"policy_hz": 50.0, "pd_hz": 50.0, "realtime": False, "transition_duration": 0.0},
        viewers=set(),
    )

    recorder = _DummyRecorder()
    result = loop.run(
        input_provider=_DummyInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=2,
        recorder=recorder,
    )

    assert result["steps"] == 2
    assert len(recorder.frames) == 2

    latest_action = bus.get_latest(TOPIC_ACTION)
    latest_mimic = bus.get_latest(TOPIC_MIMIC_OBS)
    latest_state = bus.get_latest(TOPIC_ROBOT_STATE)

    assert isinstance(latest_action, np.ndarray)
    np.testing.assert_allclose(latest_action, np.array([0.1, -0.1], dtype=np.float32))
    assert isinstance(latest_mimic, np.ndarray)
    assert latest_mimic.shape == (35,)
    assert latest_state is not None

    target = np.asarray(recorder.frames[-1]["target_dof_pos"], dtype=np.float32)
    np.testing.assert_allclose(target, np.array([0.6, -0.6], dtype=np.float32))


@requires_mujoco
def test_simulation_loop_interpolates_realtime_input_with_one_frame_delay(monkeypatch) -> None:
    import teleopit.sim.runtime_components as runtime_components
    from teleopit.sim.loop import SimulationLoop

    class _RealtimeInputProvider:
        fps = 10

        def __init__(self) -> None:
            self._packets = [
                (
                    {"Pelvis": (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))},
                    0.0,
                    0,
                ),
                (
                    {"Pelvis": (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))},
                    0.1,
                    1,
                ),
            ]
            self._idx = 0

        def get_frame_packet(self):
            packet = self._packets[min(self._idx, len(self._packets) - 1)]
            self._idx += 1
            return packet

    monkeypatch.setattr(
        runtime_components,
        "extract_mimic_obs",
        lambda qpos, last_qpos, dt: np.array([qpos[0]], dtype=np.float32),
    )
    monkeypatch.setattr("teleopit.sim.session.time.monotonic", lambda: 0.15)

    bus = InProcessBus()
    robot = _DummyRobot()
    obs_builder = _DummyObsBuilder()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=obs_builder,
        bus=bus,
        cfg={"policy_hz": 50.0, "pd_hz": 50.0, "realtime": False, "transition_duration": 0.0},
        viewers=set(),
    )

    result = loop.run(
        input_provider=_RealtimeInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=2,
    )

    assert result["steps"] == 2
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[0], np.array([0.0], dtype=np.float32))
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[1], np.array([0.5], dtype=np.float32), atol=1e-6)


@requires_mujoco
def test_simulation_loop_rejects_nonzero_reference_steps_without_realtime_timeline() -> None:
    from teleopit.sim.loop import SimulationLoop

    bus = InProcessBus()
    robot = _DummyRobot()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=_DummyObsBuilder(),
        bus=bus,
        cfg={
            "policy_hz": 50.0,
            "pd_hz": 50.0,
            "realtime": False,
            "transition_duration": 0.0,
            "retarget_buffer_enabled": True,
            "reference_steps": [0, 1],
        },
        viewers=set(),
    )

    with pytest.raises(ValueError, match="get_frame_packet"):
        loop.run(
            input_provider=_DummyInputProvider(),
            retargeter=_DummyRetargeter(),
            num_steps=1,
        )


@requires_mujoco
def test_simulation_loop_waits_for_realtime_warmup_before_first_policy_step(monkeypatch) -> None:
    import teleopit.sim.runtime_components as runtime_components
    from teleopit.sim.loop import SimulationLoop

    class _RealtimeInputProvider:
        fps = 10

        def __init__(self) -> None:
            self._packets = [
                (
                    {'Pelvis': (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))},
                    0.0,
                    0,
                ),
                (
                    {'Pelvis': (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))},
                    0.1,
                    1,
                ),
            ]
            self._idx = 0

        def get_frame_packet(self):
            packet = self._packets[min(self._idx, len(self._packets) - 1)]
            self._idx += 1
            return packet

    monkeypatch.setattr(
        runtime_components,
        'extract_mimic_obs',
        lambda qpos, last_qpos, dt: np.array([qpos[0]], dtype=np.float32),
    )
    monkeypatch.setattr('teleopit.sim.session.time.monotonic', lambda: 0.1)
    monkeypatch.setattr('teleopit.sim.session.time.sleep', lambda _seconds: None)

    bus = InProcessBus()
    robot = _DummyRobot()
    obs_builder = _DummyObsBuilder()
    provider = _RealtimeInputProvider()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=obs_builder,
        bus=bus,
        cfg={
            'policy_hz': 50.0,
            'pd_hz': 50.0,
            'realtime': False,
            'transition_duration': 0.0,
            'realtime_buffer_warmup_steps': 2,
        },
        viewers=set(),
    )

    result = loop.run(
        input_provider=provider,
        retargeter=_DummyRetargeter(),
        num_steps=1,
    )

    assert result['steps'] == 1
    assert provider._idx >= 2
    assert len(obs_builder.mimic_obs_calls) == 1
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[0], np.array([0.0], dtype=np.float32), atol=1e-6)


@requires_mujoco
def test_simulation_loop_allows_future_reference_steps_without_explicit_high_watermark() -> None:
    from teleopit.sim.loop import SimulationLoop

    bus = InProcessBus()
    robot = _DummyRobot()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=_DummyObsBuilder(),
        bus=bus,
        cfg={
            "policy_hz": 50.0,
            "pd_hz": 50.0,
            "realtime": False,
            "transition_duration": 0.0,
            "reference_steps": [0, 1, 2, 3, 4],
            "retarget_buffer_delay_s": 0.08,
            "retarget_buffer_window_s": 0.5,
            "realtime_buffer_low_watermark_steps": 0,
        },
        viewers=set(),
    )

    assert loop._ref_cfg.realtime_buffer_low_watermark_steps == 0
    assert loop._ref_cfg.realtime_buffer_high_watermark_steps is None

    class _RealtimeInputProvider:
        fps = 50

        def __init__(self) -> None:
            self._frame = {"Pelvis": (np.zeros(3, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))}

        def get_frame_packet(self):
            return self._frame, 0.0, 0

    result = loop.run(
        input_provider=_RealtimeInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=1,
    )

    assert result["steps"] == 1


@requires_mujoco
def test_simulation_loop_pause_resume_freezes_then_blends_back(monkeypatch) -> None:
    from teleopit.sim.loop import SimulationLoop

    class _RealtimeInputProvider:
        fps = 1

        def __init__(self) -> None:
            self._packets = [
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([0.2, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=0.0,
                    seq=0,
                    control_events=(),
                ),
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([1.0, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=1.0,
                    seq=1,
                    control_events=(
                        ControlEvent(
                            event_type=ControlEventType.TOGGLE_PAUSE,
                            source="pico4:test",
                            timestamp_s=1.0,
                        ),
                    ),
                ),
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([1.0, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=2.0,
                    seq=2,
                    control_events=(
                        ControlEvent(
                            event_type=ControlEventType.TOGGLE_PAUSE,
                            source="pico4:test",
                            timestamp_s=2.0,
                        ),
                    ),
                ),
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([1.0, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=3.0,
                    seq=3,
                    control_events=(),
                ),
            ]
            self._idx = 0

        def get_realtime_input_packet(self):
            packet = self._packets[min(self._idx, len(self._packets) - 1)]
            self._idx += 1
            return packet

    monotonic_values = iter([0.0, 1.0, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1, 5.0, 5.1])
    monkeypatch.setattr("teleopit.sim.session.time.monotonic", lambda: next(monotonic_values))

    bus = InProcessBus()
    robot = _DummyRobot()
    obs_builder = _DummyObsBuilder()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=obs_builder,
        bus=bus,
        cfg={
            "policy_hz": 50.0,
            "pd_hz": 50.0,
            "realtime": False,
            "transition_duration": 0.0,
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
            "velcmd_fixed_ref_yaw_alignment": False,
            "pause_resume_transition_duration": 1.0,
            "pause_resume_warmup_steps": 0,
        },
        viewers=set(),
    )

    result = loop.run(
        input_provider=_RealtimeInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=4,
    )

    assert result["steps"] == 4
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[0], np.array([0.2], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[1], np.array([0.2], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[2], np.array([0.2], dtype=np.float32), atol=1e-6)
    assert obs_builder.mimic_obs_calls[3][0] > 0.2
    assert obs_builder.mimic_obs_calls[3][0] < 1.0


@requires_mujoco
def test_simulation_loop_realtime_keyboard_mode_transitions(monkeypatch) -> None:
    from teleopit.sim.loop import SimulationLoop

    class _RealtimeInputProvider:
        fps = 1

        def __init__(self) -> None:
            self._packets = [
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([0.3, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=0.0,
                    seq=0,
                    control_events=(),
                ),
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([0.6, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=1.0,
                    seq=1,
                    control_events=(),
                ),
            ]
            self._idx = 0

        def get_realtime_input_packet(self):
            packet = self._packets[min(self._idx, len(self._packets) - 1)]
            self._idx += 1
            return packet

    class _KeyboardReader:
        def __init__(self) -> None:
            self._polls = [
                (),
                (TerminalKeyEvent("y"),),
                (TerminalKeyEvent("x"),),
            ]
            self._idx = 0

        @property
        def active(self) -> bool:
            return True

        def poll(self) -> tuple[TerminalKeyEvent, ...]:
            if self._idx >= len(self._polls):
                return ()
            events = self._polls[self._idx]
            self._idx += 1
            return events

        def close(self) -> None:
            pass

    monkeypatch.setattr("teleopit.sim.session.TerminalKeyboardReader", _KeyboardReader)

    bus = InProcessBus()
    robot = _DummyRobot()
    obs_builder = _DummyObsBuilder()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=obs_builder,
        bus=bus,
        cfg={
            "policy_hz": 50.0,
            "pd_hz": 50.0,
            "realtime": False,
            "transition_duration": 0.0,
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
            "velcmd_fixed_ref_yaw_alignment": False,
            "keyboard": {"enabled": True},
        },
        viewers=set(),
    )

    result = loop.run(
        input_provider=_RealtimeInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=3,
    )

    assert result["steps"] == 3
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[0], np.array([0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[1], np.array([0.3], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[2], np.array([0.0], dtype=np.float32), atol=1e-6)


@requires_mujoco
def test_simulation_loop_realtime_keyboard_mode_drains_stale_pause_events(monkeypatch) -> None:
    from teleopit.sim.loop import SimulationLoop

    class _RealtimeInputProvider:
        fps = 1

        def __init__(self) -> None:
            self._packet = RealtimeInputPacket(
                frame={
                    "Pelvis": (
                        np.array([0.4, 0.0, 0.0], dtype=np.float32),
                        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    )
                },
                timestamp_s=0.0,
                seq=0,
                control_events=(
                    ControlEvent(
                        event_type=ControlEventType.TOGGLE_PAUSE,
                        source="pico4:test",
                        timestamp_s=0.0,
                    ),
                ),
            )
            self._pending_control_events = list(self._packet.control_events)

        def has_frame(self) -> bool:
            return True

        def pop_control_events(self):
            events = tuple(self._pending_control_events)
            self._pending_control_events.clear()
            return events

        def get_realtime_input_packet(self):
            packet = RealtimeInputPacket(
                frame=self._packet.frame,
                timestamp_s=self._packet.timestamp_s,
                seq=self._packet.seq,
                control_events=tuple(self._pending_control_events),
            )
            self._pending_control_events.clear()
            return packet

    class _KeyboardReader:
        def __init__(self) -> None:
            self._polls = [
                (TerminalKeyEvent("a"),),
                (TerminalKeyEvent("y"),),
            ]
            self._idx = 0

        @property
        def active(self) -> bool:
            return True

        def poll(self) -> tuple[TerminalKeyEvent, ...]:
            if self._idx >= len(self._polls):
                return ()
            events = self._polls[self._idx]
            self._idx += 1
            return events

        def close(self) -> None:
            pass

    monkeypatch.setattr("teleopit.sim.session.TerminalKeyboardReader", _KeyboardReader)

    bus = InProcessBus()
    robot = _DummyRobot()
    obs_builder = _DummyObsBuilder()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=obs_builder,
        bus=bus,
        cfg={
            "policy_hz": 50.0,
            "pd_hz": 50.0,
            "realtime": False,
            "transition_duration": 0.0,
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
            "velcmd_fixed_ref_yaw_alignment": False,
            "keyboard": {"enabled": True},
        },
        viewers=set(),
    )

    result = loop.run(
        input_provider=_RealtimeInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=3,
    )

    assert result["steps"] == 3
    assert len(obs_builder.mimic_obs_calls) == 3
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[-1], np.array([0.4], dtype=np.float32), atol=1e-6)


@requires_mujoco
def test_simulation_loop_realtime_keyboard_mode_keeps_standing_when_input_not_ready(monkeypatch) -> None:
    from teleopit.sim.loop import SimulationLoop

    class _RealtimeInputProvider:
        fps = 1

        def has_frame(self) -> bool:
            return False

        def pop_control_events(self):
            return ()

    class _KeyboardReader:
        def __init__(self) -> None:
            self._polls = [
                (),
                (TerminalKeyEvent("y"),),
            ]
            self._idx = 0

        @property
        def active(self) -> bool:
            return True

        def poll(self) -> tuple[TerminalKeyEvent, ...]:
            if self._idx >= len(self._polls):
                return ()
            events = self._polls[self._idx]
            self._idx += 1
            return events

        def close(self) -> None:
            pass

    monkeypatch.setattr("teleopit.sim.session.TerminalKeyboardReader", _KeyboardReader)

    bus = InProcessBus()
    robot = _DummyRobot()
    obs_builder = _DummyObsBuilder()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=obs_builder,
        bus=bus,
        cfg={
            "policy_hz": 50.0,
            "pd_hz": 50.0,
            "realtime": False,
            "transition_duration": 0.0,
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
            "velcmd_fixed_ref_yaw_alignment": False,
            "keyboard": {"enabled": True},
        },
        viewers=set(),
    )

    result = loop.run(
        input_provider=_RealtimeInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=2,
    )

    assert result["steps"] == 2
    assert len(obs_builder.mimic_obs_calls) == 2
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[0], np.array([0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[1], np.array([0.0], dtype=np.float32), atol=1e-6)
