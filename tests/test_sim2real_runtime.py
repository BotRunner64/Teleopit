from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


class DummyRobot:
    def __init__(self, _cfg: object) -> None:
        self._state = SimpleNamespace(
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            qpos=np.zeros(29, dtype=np.float32),
            qvel=np.zeros(29, dtype=np.float32),
            ang_vel=np.zeros(3, dtype=np.float32),
        )
        self.sent_positions: list[np.ndarray] = []

    def enter_debug_mode(self) -> bool:
        return True

    def lock_all_joints(self) -> None:
        pass

    def get_state(self) -> SimpleNamespace:
        return self._state

    def send_positions(self, target_dof_pos: np.ndarray) -> None:
        self.sent_positions.append(np.asarray(target_dof_pos, dtype=np.float32))

    def set_damping(self) -> None:
        pass

    def exit_debug_mode(self) -> None:
        pass


class DummyRemote:
    def __init__(self) -> None:
        self.LB = SimpleNamespace(pressed=False, on_pressed=False)
        self.RB = SimpleNamespace(pressed=False, on_pressed=False)
        self.start = SimpleNamespace(pressed=False, on_pressed=False)
        self.Y = SimpleNamespace(pressed=False, on_pressed=False)
        self.X = SimpleNamespace(pressed=False, on_pressed=False)


def _set_button(button: SimpleNamespace, *, pressed: bool, on_pressed: bool) -> None:
    button.pressed = pressed
    button.on_pressed = on_pressed


class DummyProvider:
    def __init__(self) -> None:
        self._xrt = SimpleNamespace(is_body_data_available=lambda: True)
        self._frame = {"Pelvis": (np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))}
        self.fps = 30
        self._frame_seq = 0
        self._frame_timestamp = 1.0

    def is_available(self) -> bool:
        return True

    def get_frame(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self._frame

    def get_frame_packet(self) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float, int]:
        return self._frame, self._frame_timestamp, self._frame_seq


class DummyRetargeter:
    def __init__(self, qpos: np.ndarray) -> None:
        self._qpos = np.asarray(qpos, dtype=np.float64)

    def retarget(self, _frame: object) -> np.ndarray:
        return self._qpos.copy()


class DummyPolicy:
    def __init__(self, expected_obs_dim: int = 166) -> None:
        self._expected_obs_dim = expected_obs_dim
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def compute_action(self, _obs: np.ndarray) -> np.ndarray:
        return np.zeros(29, dtype=np.float32)

    def get_target_dof_pos(self, _action: np.ndarray) -> np.ndarray:
        return np.zeros(29, dtype=np.float32)


class DummyVelCmdObservationBuilder:
    def __init__(self) -> None:
        self.total_obs_size = 166
        self.reset_calls = 0
        self.build_calls: list[dict[str, np.ndarray]] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def build(
        self,
        _robot_state: object,
        motion_qpos: np.ndarray,
        _motion_joint_vel: np.ndarray,
        _last_action: np.ndarray,
        motion_anchor_lin_vel_w: np.ndarray,
        motion_anchor_ang_vel_w: np.ndarray,
    ) -> np.ndarray:
        self.build_calls.append(
            {
                "motion_qpos": np.asarray(motion_qpos, dtype=np.float32).copy(),
                "motion_anchor_lin_vel_w": np.asarray(motion_anchor_lin_vel_w, dtype=np.float32).copy(),
                "motion_anchor_ang_vel_w": np.asarray(motion_anchor_ang_vel_w, dtype=np.float32).copy(),
            }
        )
        return np.zeros(self.total_obs_size, dtype=np.float32)


def _make_cfg(transition_duration: float = 1.0) -> dict[str, object]:
    return {
        "policy_hz": 50.0,
        "transition_duration": transition_duration,
        "real_robot": {},
        "mocap_switch": {"check_frames": 1},
        "robot": {
            "default_angles": [0.0] * 29,
            "num_actions": 29,
        },
        "controller": {},
        "input": {"provider": "pico4"},
    }


def _install_controller_mocks(monkeypatch, *, policy: DummyPolicy, obs_builder: DummyVelCmdObservationBuilder, qpos: np.ndarray) -> None:
    import teleopit.sim2real.controller as controller_mod

    monkeypatch.setattr(controller_mod, "UnitreeG1Robot", DummyRobot)
    monkeypatch.setattr(controller_mod, "UnitreeRemote", DummyRemote)
    monkeypatch.setattr(controller_mod, "VelCmdObservationBuilder", DummyVelCmdObservationBuilder)
    monkeypatch.setattr(controller_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        controller_mod,
        "build_sim2real_mocap_components",
        lambda *args, **kwargs: SimpleNamespace(
            input_provider=DummyProvider(),
            retargeter=DummyRetargeter(qpos),
            controller=policy,
            obs_builder=obs_builder,
        ),
    )


def test_mode_transitions_reset_stateful_policy(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))

    ctrl = Sim2RealController(_make_cfg())
    ctrl._enter_standing()
    ctrl._transition_to_mocap()

    assert policy.reset_calls == 2
    assert obs_builder.reset_calls == 2


def test_reset_policy_state_clears_reference_timeline(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))

    ctrl = Sim2RealController(_make_cfg())
    assert ctrl._reference_timeline is not None
    ctrl._reference_timeline.append(np.zeros(36, dtype=np.float64), 1.0)
    ctrl._last_live_packet_seq = 7

    ctrl._reset_policy_state()

    assert len(ctrl._reference_timeline) == 0
    assert ctrl._last_live_packet_seq == -1


def test_sim2real_rejects_nonzero_reference_steps_without_buffer(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))

    cfg = _make_cfg()
    cfg["retarget_buffer_enabled"] = False
    cfg["reference_steps"] = [0, 1]

    with pytest.raises(ValueError, match="retarget_buffer_enabled=true"):
        Sim2RealController(cfg)


def test_sim2real_rejects_reference_horizon_with_insufficient_delay(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))

    cfg = _make_cfg()
    cfg["reference_steps"] = [0, 2]
    cfg["retarget_buffer_delay_s"] = 0.01

    with pytest.raises(ValueError, match="retarget_buffer_delay_s"):
        Sim2RealController(cfg)


def test_state_machine_allows_mocap_reentry_after_returning_to_standing(monkeypatch) -> None:
    from teleopit.sim2real.controller import RobotMode, Sim2RealController

    policy = DummyPolicy(expected_obs_dim=154)
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(
        monkeypatch,
        policy=policy,
        obs_builder=obs_builder,
        qpos=np.zeros(36, dtype=np.float64),
    )

    ctrl = Sim2RealController(_make_cfg())

    _set_button(ctrl.remote.start, pressed=True, on_pressed=True)
    ctrl._handle_transitions()
    assert ctrl.mode == RobotMode.STANDING

    _set_button(ctrl.remote.start, pressed=False, on_pressed=False)
    _set_button(ctrl.remote.Y, pressed=True, on_pressed=True)
    ctrl._handle_transitions()
    assert ctrl.mode == RobotMode.MOCAP

    _set_button(ctrl.remote.Y, pressed=False, on_pressed=False)
    _set_button(ctrl.remote.X, pressed=True, on_pressed=True)
    ctrl._handle_transitions()
    assert ctrl.mode == RobotMode.STANDING

    _set_button(ctrl.remote.X, pressed=False, on_pressed=False)
    _set_button(ctrl.remote.Y, pressed=True, on_pressed=False)
    ctrl._handle_transitions()
    assert ctrl.mode == RobotMode.MOCAP


def test_mocap_step_ramps_velcmd_during_transition(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[0] = 0.3
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    ctrl = Sim2RealController(_make_cfg(transition_duration=2.0))
    ctrl._transition_to_mocap()
    monkeypatch.setattr(
        ctrl,
        "_compute_anchor_velocities",
        lambda _qpos: (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ),
    )

    ctrl._mocap_step()
    ctrl._mocap_step()

    assert len(obs_builder.build_calls) == 2
    np.testing.assert_allclose(
        obs_builder.build_calls[0]["motion_anchor_lin_vel_w"],
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_builder.build_calls[0]["motion_anchor_ang_vel_w"],
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_builder.build_calls[1]["motion_anchor_lin_vel_w"],
        np.array([0.01, 0.02, 0.03], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        obs_builder.build_calls[1]["motion_anchor_ang_vel_w"],
        np.array([0.04, 0.05, 0.06], dtype=np.float32),
        atol=1e-6,
    )


def test_mocap_step_velcmd_applies_fixed_initial_yaw_alignment(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    ctrl = Sim2RealController(_make_cfg(transition_duration=0.0))
    ctrl.robot._state.quat = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)
    monkeypatch.setattr(
        ctrl,
        "_compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    ctrl._mocap_step()

    np.testing.assert_allclose(
        obs_builder.build_calls[0]["motion_qpos"][3:7],
        np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
        atol=1e-6,
    )


def test_mocap_step_velcmd_keeps_fixed_yaw_after_start(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    ctrl = Sim2RealController(_make_cfg(transition_duration=0.0))
    monkeypatch.setattr(
        ctrl,
        "_compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    ctrl.robot._state.quat = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)
    ctrl._mocap_step()
    ctrl.robot._state.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    ctrl._mocap_step()

    np.testing.assert_allclose(
        obs_builder.build_calls[1]["motion_qpos"][3:7],
        np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
        atol=1e-6,
    )


def test_mocap_step_velcmd_can_disable_fixed_yaw_alignment(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    cfg = _make_cfg(transition_duration=0.0)
    cfg["velcmd_fixed_ref_yaw_alignment"] = False
    ctrl = Sim2RealController(cfg)
    ctrl.robot._state.quat = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)
    monkeypatch.setattr(
        ctrl,
        "_compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    ctrl._mocap_step()

    np.testing.assert_allclose(
        obs_builder.build_calls[0]["motion_qpos"][3:7],
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
