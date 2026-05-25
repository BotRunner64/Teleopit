from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np

from teleopit.controllers.observation import VelCmdObservationBuilder
from teleopit.sim.runtime_components import PolicyStepRunner


def test_runtime_package_reexports_public_helpers() -> None:
    from teleopit.runtime import build_inference_components, cfg_get, validate_policy_path

    assert callable(build_inference_components)
    assert callable(cfg_get)
    assert callable(validate_policy_path)


def test_observation_module_imports_without_runtime_cycle() -> None:
    module = importlib.import_module("teleopit.controllers.observation")
    assert hasattr(module, "VelCmdObservationBuilder")


def _make_runner(
    obs_builder: object,
    *,
    reference_velocity_smoothing_alpha: float = 1.0,
) -> PolicyStepRunner:
    return PolicyStepRunner(
        robot=SimpleNamespace(),
        controller=SimpleNamespace(),
        obs_builder=obs_builder,
        policy_hz=50.0,
        decimation=1,
        num_actions=29,
        kps=np.ones(29, dtype=np.float32),
        kds=np.ones(29, dtype=np.float32),
        torque_limits=np.ones(29, dtype=np.float32),
        default_dof_pos=np.zeros(29, dtype=np.float32),
        reference_velocity_smoothing_alpha=reference_velocity_smoothing_alpha,
    )


def test_prepare_motion_command_velcmd_applies_fixed_initial_yaw_alignment(monkeypatch) -> None:
    velcmd_builder = VelCmdObservationBuilder.__new__(VelCmdObservationBuilder)
    runner = _make_runner(velcmd_builder)
    monkeypatch.setattr(
        runner,
        '_compute_anchor_velocities',
        lambda qpos: (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
    )

    retarget_qpos = np.zeros(36, dtype=np.float64)
    retarget_qpos[:3] = np.array([1.5, 4.5, 0.82], dtype=np.float64)
    retarget_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state = SimpleNamespace(
        base_pos=np.array([10.0, 20.0, 0.8], dtype=np.float32),
        quat=np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
    )

    prep = runner.prepare_motion_command(retarget_qpos, state)

    np.testing.assert_allclose(prep.qpos[:2], np.array([1.5, 4.5]), atol=1e-6)
    np.testing.assert_allclose(prep.qpos[3:7], state.quat, atol=1e-6)


def test_prepare_motion_command_velcmd_keeps_fixed_yaw_after_start(monkeypatch) -> None:
    velcmd_builder = VelCmdObservationBuilder.__new__(VelCmdObservationBuilder)
    runner = _make_runner(velcmd_builder)
    monkeypatch.setattr(
        runner,
        '_compute_anchor_velocities',
        lambda qpos: (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
    )

    retarget_qpos = np.zeros(36, dtype=np.float64)
    retarget_qpos[:3] = np.array([1.5, 4.5, 0.82], dtype=np.float64)
    retarget_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    first_state = SimpleNamespace(
        base_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
        quat=np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
        qpos=np.zeros(29, dtype=np.float32),
    )
    second_state = SimpleNamespace(
        base_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
        quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        qpos=np.zeros(29, dtype=np.float32),
    )

    first = runner.prepare_motion_command(retarget_qpos, first_state)
    runner.finish_step(np.zeros(29, dtype=np.float32), first.qpos)
    second = runner.prepare_motion_command(retarget_qpos, second_state)

    np.testing.assert_allclose(second.qpos[:2], np.array([1.5, 4.5]), atol=1e-6)
    np.testing.assert_allclose(second.qpos[3:7], first_state.quat, atol=1e-6)


def test_prepare_motion_command_can_reanchor_reference_xy(monkeypatch) -> None:
    velcmd_builder = VelCmdObservationBuilder.__new__(VelCmdObservationBuilder)
    runner = _make_runner(velcmd_builder)
    monkeypatch.setattr(
        runner,
        '_compute_anchor_velocities',
        lambda qpos: (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
    )

    target_qpos = np.zeros(36, dtype=np.float64)
    target_qpos[:2] = np.array([10.0, 20.0], dtype=np.float64)
    runner.reset_reference_alignment(target_qpos)

    retarget_qpos = np.zeros(36, dtype=np.float64)
    retarget_qpos[:3] = np.array([1.5, 4.5, 0.82], dtype=np.float64)
    retarget_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state = SimpleNamespace(
        base_pos=np.array([10.0, 20.0, 0.8], dtype=np.float32),
        quat=np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
        qpos=np.zeros(29, dtype=np.float32),
    )

    first = runner.prepare_motion_command(retarget_qpos, state)
    runner.finish_step(np.zeros(29, dtype=np.float32), first.qpos)

    moved_qpos = retarget_qpos.copy()
    moved_qpos[0] += 1.0
    second = runner.prepare_motion_command(moved_qpos, state)

    np.testing.assert_allclose(first.qpos[:2], np.array([10.0, 20.0]), atol=1e-6)
    np.testing.assert_allclose(second.qpos[:2], np.array([10.0, 21.0]), atol=1e-6)
    np.testing.assert_allclose(first.qpos[3:7], state.quat, atol=1e-6)


def test_prepare_motion_command_smooths_joint_velocity_spikes(monkeypatch) -> None:
    velcmd_builder = VelCmdObservationBuilder.__new__(VelCmdObservationBuilder)
    runner = _make_runner(
        velcmd_builder,
        reference_velocity_smoothing_alpha=0.5,
    )
    monkeypatch.setattr(
        runner,
        '_compute_anchor_velocities',
        lambda qpos: (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
    )

    state = SimpleNamespace(
        base_pos=np.array([0.0, 0.0, 0.8], dtype=np.float32),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        qpos=np.zeros(29, dtype=np.float32),
    )
    qpos0 = np.zeros(36, dtype=np.float64)
    qpos0[3] = 1.0
    qpos1 = qpos0.copy()
    qpos1[7] = 1.0

    first = runner.prepare_motion_command(qpos0, state)
    runner.finish_step(np.zeros(29, dtype=np.float32), first.qpos)

    second = runner.prepare_motion_command(qpos1, state)
    runner.finish_step(np.zeros(29, dtype=np.float32), second.qpos)

    third = runner.prepare_motion_command(qpos1, state)

    np.testing.assert_allclose(second.raw_motion_joint_vel[0], 50.0, atol=1e-6)
    np.testing.assert_allclose(second.motion_joint_vel[0], 25.0, atol=1e-6)
    np.testing.assert_allclose(third.raw_motion_joint_vel[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(third.motion_joint_vel[0], 12.5, atol=1e-6)
