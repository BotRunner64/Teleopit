from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from teleopit.controllers.observation import VelCmdObservationBuilder
from teleopit.controllers.qpos_interpolator import QposInterpolator
from teleopit.sim.runtime_components import PolicyStepRunner


def _make_runner(obs_builder: object, *, fixed_ref_yaw_alignment: bool = True) -> PolicyStepRunner:
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
        qpos_interpolator=QposInterpolator(0.0, 50.0),
        fixed_ref_yaw_alignment=fixed_ref_yaw_alignment,
    )


def test_prepare_motion_command_velcmd_applies_fixed_initial_yaw_alignment(monkeypatch) -> None:
    velcmd_builder = VelCmdObservationBuilder.__new__(VelCmdObservationBuilder)
    runner = _make_runner(velcmd_builder)
    monkeypatch.setattr(
        runner,
        "_compute_anchor_velocities",
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
        "_compute_anchor_velocities",
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


def test_prepare_motion_command_non_velcmd_aligns_root_xy_to_robot() -> None:
    runner = _make_runner(SimpleNamespace())

    retarget_qpos = np.zeros(36, dtype=np.float64)
    retarget_qpos[:3] = np.array([1.5, 4.5, 0.82], dtype=np.float64)
    retarget_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    state = SimpleNamespace(
        base_pos=np.array([10.0, 20.0, 0.8], dtype=np.float32),
        quat=np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32),
    )

    prep = runner.prepare_motion_command(retarget_qpos, state)

    np.testing.assert_allclose(prep.qpos[:2], np.array([10.0, 20.0]), atol=1e-6)
    np.testing.assert_allclose(prep.qpos[3:7], state.quat, atol=1e-6)
