from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from teleopit.inputs.realtime_packet import ControlEvent, ControlEventType, RealtimeInputPacket


class DummyRobot:
    def __init__(self, _cfg: object) -> None:
        self._state = SimpleNamespace(
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            qpos=np.zeros(29, dtype=np.float32),
            qvel=np.zeros(29, dtype=np.float32),
            ang_vel=np.zeros(3, dtype=np.float32),
        )
        self.sent_positions: list[np.ndarray] = []
        self.sent_gains: list[tuple[np.ndarray | None, np.ndarray | None]] = []
        self.lock_calls = 0

    def enter_debug_mode(self) -> bool:
        return True

    def lock_all_joints(self) -> None:
        self.lock_calls += 1

    def get_state(self) -> SimpleNamespace:
        return self._state

    def send_positions(self, target_dof_pos: np.ndarray, kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        self.sent_positions.append(np.asarray(target_dof_pos, dtype=np.float32))
        self.sent_gains.append((
            None if kp is None else np.asarray(kp, dtype=np.float32),
            None if kd is None else np.asarray(kd, dtype=np.float32),
        ))

    def set_damping(self) -> None:
        pass

    def exit_debug_mode(self) -> None:
        pass


class DummyRemote:
    def __init__(self) -> None:
        self.LB = SimpleNamespace(pressed=False, on_pressed=False)
        self.RB = SimpleNamespace(pressed=False, on_pressed=False)
        self.start = SimpleNamespace(pressed=False, on_pressed=False)
        self.A = SimpleNamespace(pressed=False, on_pressed=False)
        self.B = SimpleNamespace(pressed=False, on_pressed=False)
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
        self._control_events: tuple[ControlEvent, ...] = ()

    def is_available(self) -> bool:
        return True

    def get_frame(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return self._frame

    def get_frame_packet(self) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float, int]:
        return self._frame, self._frame_timestamp, self._frame_seq

    def get_realtime_input_packet(self) -> RealtimeInputPacket[dict[str, tuple[np.ndarray, np.ndarray]]]:
        control_events = tuple(self._control_events)
        self._control_events = ()
        return RealtimeInputPacket(
            frame=self._frame,
            timestamp_s=self._frame_timestamp,
            seq=self._frame_seq,
            control_events=control_events,
        )


class DummyRetargeter:
    def __init__(self, qpos: np.ndarray) -> None:
        self._qpos = np.asarray(qpos, dtype=np.float64)
        self.reset_calls = 0

    def retarget(self, _frame: object) -> np.ndarray:
        return self._qpos.copy()

    def reset(self) -> None:
        self.reset_calls += 1


class DummyPolicy:
    def __init__(self, expected_obs_dim: int = 166) -> None:
        self._expected_obs_dim = expected_obs_dim
        self._multi_input = True
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
        motion_joint_vel: np.ndarray,
        _last_action: np.ndarray,
        motion_anchor_lin_vel_w: np.ndarray,
        motion_anchor_ang_vel_w: np.ndarray,
    ) -> np.ndarray:
        self.build_calls.append(
            {
                "motion_qpos": np.asarray(motion_qpos, dtype=np.float32).copy(),
                "motion_joint_vel": np.asarray(motion_joint_vel, dtype=np.float32).copy(),
                "motion_anchor_lin_vel_w": np.asarray(motion_anchor_lin_vel_w, dtype=np.float32).copy(),
                "motion_anchor_ang_vel_w": np.asarray(motion_anchor_ang_vel_w, dtype=np.float32).copy(),
            }
        )
        return np.zeros(self.total_obs_size, dtype=np.float32)


class DummyHandRuntime:
    def __init__(self) -> None:
        self.active_flags: list[bool] = []
        self.close_calls = 0

    def start(self) -> None:
        pass

    def tick(self, *, active: bool) -> None:
        self.active_flags.append(active)

    def close(self) -> None:
        self.close_calls += 1


class FailingHandRuntime(DummyHandRuntime):
    def tick(self, *, active: bool) -> None:
        super().tick(active=active)
        raise RuntimeError("hand send failed")


def _make_cfg() -> dict[str, object]:
    return {
        "policy_hz": 50.0,
        "real_robot": {
            "kp_real": [100.0] * 29,
            "kd_real": [2.0] * 29,
        },
        "standing_return_ramp_duration": 0.5,
        "standing_return_kp_ramp_floor_ratio": 0.5,
        "mocap_switch": {"check_frames": 1},
        "robot": {
            "default_angles": [0.0] * 29,
            "num_actions": 29,
            "xml_path": "robot.xml",
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

    # Both transitions now do soft episode-reset and preserve retargeter warm-start.
    assert policy.reset_calls == 2
    assert obs_builder.reset_calls == 2
    assert ctrl.retargeter.reset_calls == 0


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


def test_sim2real_retarget_viewer_defaults_off(monkeypatch) -> None:
    import teleopit.sim2real.controller as controller_mod
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))

    starts: list[tuple[object, ...]] = []
    monkeypatch.setattr(controller_mod, "start_robot_viewer", lambda *args, **kwargs: starts.append(args))

    Sim2RealController(_make_cfg())

    assert starts == []


def test_sim2real_retarget_viewer_writes_reference_qpos(monkeypatch) -> None:
    import multiprocessing as mp

    import teleopit.sim2real.controller as controller_mod
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[0] = 0.25
    target_qpos[3] = 1.0
    target_qpos[7] = 0.5
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    arr = mp.Array("d", 36)
    alive = mp.Value("i", 1)
    shutdown = mp.Event()
    proc = SimpleNamespace(join=lambda timeout=None: None, is_alive=lambda: False, terminate=lambda: None)
    starts: list[tuple[object, ...]] = []

    def fake_start_robot_viewer(*args: object, **_kwargs: object) -> tuple[object, object, object, object]:
        starts.append(args)
        return proc, arr, alive, shutdown

    monkeypatch.setattr(controller_mod, "start_robot_viewer", fake_start_robot_viewer)
    cfg = _make_cfg()
    cfg["retarget_buffer_enabled"] = False
    cfg["viewers"] = "retarget"
    ctrl = Sim2RealController(cfg)
    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    ctrl._mocap_step()

    assert starts
    with arr.get_lock():
        written = np.asarray(arr[:], dtype=np.float64)
    np.testing.assert_allclose(written[[0, 3, 7]], target_qpos[[0, 3, 7]], atol=1e-6)


def test_sim2real_retarget_viewer_rejects_sim_viewers(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))

    cfg = _make_cfg()
    cfg["viewers"] = ["retarget", "sim2sim"]

    with pytest.raises(ValueError, match="supports only the optional 'retarget' viewer"):
        Sim2RealController(cfg)


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


def test_return_to_standing_uses_default_pose_and_stronger_ramp_without_relock(monkeypatch) -> None:
    from teleopit.sim2real.controller import RobotMode, Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(
        monkeypatch,
        policy=policy,
        obs_builder=obs_builder,
        qpos=np.zeros(36, dtype=np.float64),
    )

    cfg = _make_cfg()
    cfg["robot"]["default_angles"] = [0.2] * 29
    ctrl = Sim2RealController(cfg)
    ctrl.mode = RobotMode.MOCAP
    ctrl.robot._state.qpos = np.ones(29, dtype=np.float32)
    ctrl._last_commanded_motion_qpos = np.ones(36, dtype=np.float64)

    ctrl._enter_standing()
    ctrl._standing_step()

    assert ctrl.mode == RobotMode.STANDING
    assert ctrl.robot.lock_calls == 0
    np.testing.assert_allclose(ctrl._standing_qpos[7:36], np.full(29, 0.2, dtype=np.float64))
    kp, kd = ctrl.robot.sent_gains[-1]
    assert kp is not None
    assert kd is not None
    np.testing.assert_allclose(kp, np.full(29, 50.0, dtype=np.float32))


def test_dexterous_hand_ticks_only_during_active_mocap(monkeypatch) -> None:
    import teleopit.sim2real.controller as controller_mod
    from teleopit.runtime.mocap_session import MocapSessionState
    from teleopit.sim2real.controller import RobotMode, Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    hand_runtime = DummyHandRuntime()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))
    monkeypatch.setattr(controller_mod, "build_linkerhand_runtime", lambda _cfg, _provider: hand_runtime)

    ctrl = Sim2RealController(_make_cfg())

    ctrl.mode = RobotMode.STANDING
    ctrl._tick_dexterous_hand()

    ctrl.mode = RobotMode.MOCAP
    ctrl._mocap_session.reset()
    assert ctrl._mocap_session.state == MocapSessionState.ACTIVE
    ctrl._tick_dexterous_hand()

    ctrl._mocap_session.pause(np.zeros(36, dtype=np.float64))
    ctrl._tick_dexterous_hand()

    assert hand_runtime.active_flags == [False, True, False]


def test_dexterous_hand_failure_does_not_enter_damping(monkeypatch) -> None:
    import teleopit.sim2real.controller as controller_mod
    from teleopit.sim2real.controller import RobotMode, Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    hand_runtime = FailingHandRuntime()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))
    monkeypatch.setattr(controller_mod, "build_linkerhand_runtime", lambda _cfg, _provider: hand_runtime)

    ctrl = Sim2RealController(_make_cfg())
    ctrl.mode = RobotMode.MOCAP
    ctrl._mocap_session.reset()

    ctrl._tick_dexterous_hand()

    assert ctrl.mode == RobotMode.MOCAP
    assert hand_runtime.active_flags == [True]


def test_can_switch_to_mocap_returns_false_without_blocking_when_realtime_has_no_frame(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(
        monkeypatch,
        policy=policy,
        obs_builder=obs_builder,
        qpos=np.zeros(36, dtype=np.float64),
    )

    ctrl = Sim2RealController(_make_cfg())
    get_frame_calls = 0

    def blocking_get_frame() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        nonlocal get_frame_calls
        get_frame_calls += 1
        raise AssertionError("get_frame should not be called before a realtime frame is available")

    ctrl.input_provider.has_frame = lambda: False
    ctrl.input_provider.get_frame = blocking_get_frame

    assert ctrl._can_switch_to_mocap() is False
    assert get_frame_calls == 0


def test_mocap_step_episode_reset_on_transition(monkeypatch) -> None:
    """After _transition_to_mocap, the first mocap step starts with zero joint velocity."""
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[0] = 0.3
    target_qpos[7] = 1.0
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    cfg = _make_cfg()
    cfg["retarget_buffer_enabled"] = False
    ctrl = Sim2RealController(cfg)
    ctrl._transition_to_mocap()
    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
        lambda _qpos: (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ),
    )

    ctrl._mocap_step()

    assert len(obs_builder.build_calls) == 1
    np.testing.assert_allclose(obs_builder.build_calls[0]["motion_joint_vel"], np.zeros(29, dtype=np.float32))
    np.testing.assert_allclose(obs_builder.build_calls[0]["motion_qpos"][7], 1.0, atol=1e-6)


def test_mocap_step_velcmd_applies_fixed_initial_yaw_alignment(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    ctrl = Sim2RealController(_make_cfg())
    ctrl.robot._state.quat = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)
    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
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

    ctrl = Sim2RealController(_make_cfg())
    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
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


def test_transition_to_mocap_uses_resume_style_alignment_and_zero_velocity(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[0] = 0.25
    target_qpos[7] = 0.75
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    cfg = _make_cfg()
    cfg["retarget_buffer_enabled"] = False
    ctrl = Sim2RealController(cfg)
    ctrl.robot._state.base_pos = np.array([1.0, 2.0, 0.0], dtype=np.float32)
    ctrl.robot._state.quat = np.array([0.9238795, 0.0, 0.0, 0.38268343], dtype=np.float32)
    ctrl._transition_to_mocap()

    assert ctrl._last_retarget_qpos is None

    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )
    ctrl._mocap_step()

    np.testing.assert_allclose(obs_builder.build_calls[0]["motion_joint_vel"], np.zeros(29, dtype=np.float32))
    np.testing.assert_allclose(obs_builder.build_calls[0]["motion_qpos"][0:2], np.array([1.0, 2.0], dtype=np.float32), atol=1e-6)


def test_mocap_step_waits_for_realtime_warmup_before_running_policy(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[3] = 1.0
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    cfg = _make_cfg()
    cfg['realtime_buffer_warmup_steps'] = 2
    ctrl = Sim2RealController(cfg)
    monkeypatch.setattr(
        ctrl._ref_proc,
        'compute_anchor_velocities',
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )
    monkeypatch.setattr('teleopit.sim2real.controller.time.monotonic', lambda: 1.1)

    ctrl._mocap_step()

    assert len(obs_builder.build_calls) == 0
    assert ctrl.robot.sent_positions == []

    ctrl.input_provider._frame_seq = 1
    ctrl.input_provider._frame_timestamp = 1.03
    ctrl._mocap_step()

    assert len(obs_builder.build_calls) == 1
    assert len(ctrl.robot.sent_positions) == 1


def test_sim2real_allows_future_reference_steps(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=np.zeros(36, dtype=np.float64))

    cfg = _make_cfg()
    cfg["reference_steps"] = [0, 1, 2, 3, 4]
    cfg["retarget_buffer_delay_s"] = 0.08
    cfg["retarget_buffer_window_s"] = 0.5

    Sim2RealController(cfg)


def test_mocap_step_uses_current_reference_qpos(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[3] = 1.0
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    cfg = _make_cfg()
    cfg["retarget_buffer_enabled"] = False
    ctrl = Sim2RealController(cfg)
    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    ctrl._mocap_step()
    ctrl.retargeter._qpos[0] = 1.0
    ctrl._mocap_step()

    assert len(obs_builder.build_calls) == 2
    np.testing.assert_allclose(obs_builder.build_calls[0]["motion_qpos"][0], 0.0, atol=1e-6)
    np.testing.assert_allclose(obs_builder.build_calls[1]["motion_qpos"][0], 1.0, atol=1e-6)


def test_mocap_pause_freezes_reference_and_zeroes_velocities(monkeypatch) -> None:
    from teleopit.sim2real.controller import Sim2RealController
    from teleopit.runtime.mocap_session import MocapSessionState

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[0] = 0.2
    target_qpos[3] = 1.0
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    cfg = _make_cfg()
    cfg["retarget_buffer_enabled"] = False
    ctrl = Sim2RealController(cfg)
    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    ctrl._mocap_step()
    ctrl.input_provider._control_events = (
        ControlEvent(
            event_type=ControlEventType.TOGGLE_PAUSE,
            source="pico4:test",
            timestamp_s=1.1,
        ),
    )
    ctrl.retargeter._qpos[0] = 1.0
    ctrl.input_provider._frame_seq = 1
    ctrl.input_provider._frame_timestamp = 1.1
    ctrl._mocap_step()

    assert ctrl._mocap_session.state == MocapSessionState.PAUSED
    np.testing.assert_allclose(obs_builder.build_calls[-1]["motion_qpos"][0], 0.2, atol=1e-6)
    np.testing.assert_allclose(obs_builder.build_calls[-1]["motion_joint_vel"], np.zeros(29, dtype=np.float32))
    np.testing.assert_allclose(
        obs_builder.build_calls[-1]["motion_anchor_lin_vel_w"],
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_allclose(
        obs_builder.build_calls[-1]["motion_anchor_ang_vel_w"],
        np.zeros(3, dtype=np.float32),
    )


def test_mocap_resume_uses_episode_reset_semantics(monkeypatch) -> None:
    """Resume does an episode reset and reanchors live mocap root XY."""
    from teleopit.sim2real.controller import Sim2RealController
    from teleopit.runtime.mocap_session import MocapSessionState

    policy = DummyPolicy()
    obs_builder = DummyVelCmdObservationBuilder()
    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[0] = 0.2
    target_qpos[3] = 1.0
    _install_controller_mocks(monkeypatch, policy=policy, obs_builder=obs_builder, qpos=target_qpos)

    cfg = _make_cfg()
    cfg["retarget_buffer_enabled"] = False
    ctrl = Sim2RealController(cfg)
    monkeypatch.setattr(
        ctrl._ref_proc,
        "compute_anchor_velocities",
        lambda _qpos: (
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        ),
    )

    # Run one step, then pause
    ctrl._mocap_step()
    ctrl.input_provider._control_events = (
        ControlEvent(
            event_type=ControlEventType.TOGGLE_PAUSE,
            source="pico4:test",
            timestamp_s=1.1,
        ),
    )
    ctrl.input_provider._frame_seq = 1
    ctrl.input_provider._frame_timestamp = 1.1
    ctrl._mocap_step()
    assert ctrl._mocap_session.state == MocapSessionState.PAUSED

    # Resume: should stay on the ACTIVE/PAUSED state model.
    ctrl.retargeter._qpos[0] = 1.0
    ctrl.retargeter._qpos[7] = 1.0
    ctrl.input_provider._control_events = (
        ControlEvent(
            event_type=ControlEventType.TOGGLE_PAUSE,
            source="pico4:test",
            timestamp_s=1.2,
        ),
    )
    ctrl.input_provider._frame_seq = 2
    ctrl.input_provider._frame_timestamp = 1.2
    ctrl._mocap_step()

    # Episode-reset resume goes straight to ACTIVE
    assert ctrl._mocap_session.state == MocapSessionState.ACTIVE
    # Policy was reset (last_action zeroed, history cleared)
    assert np.allclose(ctrl._last_action, 0.0)
    assert ctrl.retargeter.reset_calls == 0
    # Retarget reference jumps to the live mocap pose (joint 0), while root XY
    # is reanchored to the paused reference because real-robot XY is unobserved.
    np.testing.assert_allclose(obs_builder.build_calls[-1]["motion_qpos"][0], 0.2, atol=1e-6)
    np.testing.assert_allclose(obs_builder.build_calls[-1]["motion_qpos"][7], 1.0, atol=1e-6)
    np.testing.assert_allclose(obs_builder.build_calls[-1]["motion_joint_vel"], np.zeros(29, dtype=np.float32))
