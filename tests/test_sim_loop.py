from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_mujoco
from teleopit.bus.in_process import InProcessBus
from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE
from teleopit.interfaces import RobotState


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


class _DummyRecorder:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def record_step(self, data: dict[str, object]) -> None:
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
    monkeypatch.setattr("teleopit.sim.loop.time.monotonic", lambda: 0.15)

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
