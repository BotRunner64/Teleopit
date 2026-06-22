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
        self.motion_qpos_calls: list[np.ndarray] = []
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
        self.motion_qpos_calls.append(np.asarray(motion_qpos, dtype=np.float32).copy())
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
    def __init__(self) -> None:
        self.reset_calls = 0

    def retarget(self, human_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pelvis_x = float(human_data["Pelvis"][0][0])
        return (
            np.array([pelvis_x, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.zeros(29, dtype=np.float64),
        )

    def reset(self) -> None:
        self.reset_calls += 1


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def test_standing_qpos_keeps_yaw_but_drops_roll() -> None:
    from teleopit.sim.loop import SimulationLoop

    bus = InProcessBus()
    robot = _DummyRobot()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=_DummyObsBuilder(),
        bus=bus,
        cfg={"policy_hz": 50.0, "pd_hz": 50.0, "realtime": False},
        viewers=set(),
    )

    yaw = np.float32(np.pi / 2.0)
    roll = np.float32(np.pi / 6.0)
    yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32)
    roll_quat = np.array([np.cos(roll / 2.0), np.sin(roll / 2.0), 0.0, 0.0], dtype=np.float32)
    tilted_quat = _quat_mul(yaw_quat, roll_quat)
    state = RobotState(
        qpos=np.array([0.2, -0.1], dtype=np.float32),
        qvel=np.zeros(2, dtype=np.float32),
        quat=tilted_quat,
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=0.0,
        base_pos=np.array([1.0, 2.0, 0.9], dtype=np.float32),
        base_lin_vel=np.zeros(3, dtype=np.float32),
    )

    standing_qpos = loop._build_standing_qpos(state)

    np.testing.assert_allclose(standing_qpos[0:3], state.base_pos, atol=1e-6)
    np.testing.assert_allclose(standing_qpos[3:7], yaw_quat, atol=1e-6)
    np.testing.assert_allclose(standing_qpos[7:9], robot.default_dof_pos, atol=1e-6)


def test_standing_reference_is_fixed_after_initialization() -> None:
    from teleopit.sim.loop import SimulationLoop

    bus = InProcessBus()
    robot = _DummyRobot()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=_DummyObsBuilder(),
        bus=bus,
        cfg={"policy_hz": 50.0, "pd_hz": 50.0, "realtime": False},
        viewers=set(),
    )

    first_state = RobotState(
        qpos=np.zeros(2, dtype=np.float32),
        qvel=np.zeros(2, dtype=np.float32),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=0.0,
        base_pos=np.array([1.0, 2.0, 0.9], dtype=np.float32),
        base_lin_vel=np.zeros(3, dtype=np.float32),
    )
    first = loop._set_standing_reference(first_state)

    drifted_state = RobotState(
        qpos=np.zeros(2, dtype=np.float32),
        qvel=np.zeros(2, dtype=np.float32),
        quat=np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=1.0,
        base_pos=np.array([5.0, 6.0, 0.9], dtype=np.float32),
        base_lin_vel=np.zeros(3, dtype=np.float32),
    )
    live = loop._build_standing_qpos(drifted_state)

    np.testing.assert_allclose(loop._standing_qpos, first, atol=1e-6)
    assert not np.allclose(live[0:7], first[0:7])


@requires_mujoco
def test_simulation_loop_runs_without_viewers() -> None:
    from teleopit.sim.loop import SimulationLoop

    bus = InProcessBus()
    robot = _DummyRobot()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=_DummyObsBuilder(),
        bus=bus,
        cfg={"policy_hz": 50.0, "pd_hz": 50.0, "realtime": False},
        viewers=set(),
    )

    result = loop.run(
        input_provider=_DummyInputProvider(),
        retargeter=_DummyRetargeter(),
        num_steps=2,
    )

    assert result["steps"] == 2

    latest_action = bus.get_latest(TOPIC_ACTION)
    latest_mimic = bus.get_latest(TOPIC_MIMIC_OBS)
    latest_state = bus.get_latest(TOPIC_ROBOT_STATE)

    assert isinstance(latest_action, np.ndarray)
    np.testing.assert_allclose(latest_action, np.array([0.1, -0.1], dtype=np.float32))
    assert isinstance(latest_mimic, np.ndarray)
    assert latest_mimic.shape == (35,)
    assert latest_state is not None


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
        cfg={"policy_hz": 50.0, "pd_hz": 50.0, "realtime": False},
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
def test_simulation_loop_allows_future_reference_steps() -> None:
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
            "reference_steps": [0, 1, 2, 3, 4],
            "retarget_buffer_delay_s": 0.08,
            "retarget_buffer_window_s": 0.5,
        },
        viewers=set(),
    )

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
def test_simulation_loop_pause_resume_freezes_then_reanchors_live_retarget(monkeypatch) -> None:
    """Sim2sim pause/resume follows realtime retarget-reset semantics.

    Pause freezes the reference at the hold pose. Resume resets policy/reference
    state and directly accepts the live retarget pose, with root XY reanchored
    to the current robot state instead of interpolating from the stale hold pose.
    """
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
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
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
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[2], np.array([0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(obs_builder.mimic_obs_calls[3], np.array([0.0], dtype=np.float32), atol=1e-6)
    assert loop._step_runner.last_retarget_qpos is not None


@requires_mujoco
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
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
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
def test_simulation_loop_pico_arms_mode_composes_standing_body_with_live_arm(monkeypatch) -> None:
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
                            np.array([0.9, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=1.0,
                    seq=1,
                    control_events=(),
                ),
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([0.9, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=2.0,
                    seq=2,
                    control_events=(ControlEvent(event_type=ControlEventType.TOGGLE_ARMS, source="pico4:test"),),
                ),
                RealtimeInputPacket(
                    frame={
                        "Pelvis": (
                            np.array([1.2, 0.0, 0.0], dtype=np.float32),
                            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        )
                    },
                    timestamp_s=3.0,
                    seq=3,
                    control_events=(ControlEvent(event_type=ControlEventType.TOGGLE_ARMS, source="pico4:test"),),
                ),
            ]
            self._idx = 0

        def get_realtime_input_packet(self):
            packet = self._packets[min(self._idx, len(self._packets) - 1)]
            self._idx += 1
            return packet

    class _Retargeter:
        def retarget(self, frame: object) -> np.ndarray:
            pelvis = np.asarray(frame["Pelvis"][0], dtype=np.float64)
            qpos = np.zeros(36, dtype=np.float64)
            qpos[0] = pelvis[0]
            qpos[3] = 1.0
            qpos[7] = pelvis[0]
            qpos[8] = pelvis[0] + 10.0
            return qpos

    class _KeyboardReader:
        def __init__(self) -> None:
            self._polls = [
                (TerminalKeyEvent("y"),),
                (),
                (),
                (),
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

    robot = _DummyRobot()
    obs_builder = _DummyObsBuilder()
    loop = SimulationLoop(
        robot=robot,
        controller=_DummyController(),
        obs_builder=obs_builder,
        bus=InProcessBus(),
        cfg={
            "policy_hz": 50.0,
            "pd_hz": 50.0,
            "realtime": False,
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
            "keyboard": {"enabled": True},
            "arm_mocap": {"controlled_joint_indices": [1]},
        },
        viewers=set(),
    )

    result = loop.run(
        input_provider=_RealtimeInputProvider(),
        retargeter=_Retargeter(),
        num_steps=4,
    )

    assert result["steps"] == 4
    # Step 2 is ARMS: root/non-arm stays at standing while arm index 1 follows retarget.
    np.testing.assert_allclose(obs_builder.motion_qpos_calls[2][0], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs_builder.motion_qpos_calls[2][7], 0.5, atol=1e-6)
    np.testing.assert_allclose(obs_builder.motion_qpos_calls[2][8], 10.9, atol=1e-6)
    # Step 3 toggles back to full-body MOCAP; root XY is reanchored, while non-arm joints follow retarget again.
    np.testing.assert_allclose(obs_builder.motion_qpos_calls[3][0], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs_builder.motion_qpos_calls[3][7], 1.2, atol=1e-6)


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
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
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
            "retarget_buffer_enabled": False,
            "realtime_input_delay_s": 0.0,
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
