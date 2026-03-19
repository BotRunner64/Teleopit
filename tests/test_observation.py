"""Tests for VelCmdHistory observation helpers and builder."""

from __future__ import annotations

import numpy as np
import pytest

from teleopit.controllers.observation import (
    VelCmdObservationBuilder,
    _quat_inv_np,
    _quat_rotate_np,
    compute_fixed_yaw_alignment_quat,
    rotate_motion_qpos_by_yaw,
)
from teleopit.interfaces import RobotState
from conftest import DEFAULT_ANGLES, NUM_ACTIONS, find_g1_xml_path, requires_mujoco


_XML_PATH = find_g1_xml_path()
_skip_no_xml = pytest.mark.skipif(_XML_PATH is None, reason="Robot XML not found")


def _velcmd_cfg() -> dict[str, object]:
    return {
        "num_actions": NUM_ACTIONS,
        "default_dof_pos": DEFAULT_ANGLES.tolist(),
        "xml_path": _XML_PATH or "",
        "anchor_body_name": "torso_link",
    }


def _make_motion_qpos() -> np.ndarray:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[3] = 1.0
    return qpos


def _make_state() -> RobotState:
    return RobotState(
        qpos=np.zeros(NUM_ACTIONS, dtype=np.float32),
        qvel=np.zeros(NUM_ACTIONS, dtype=np.float32),
        quat=np.array([1, 0, 0, 0], dtype=np.float32),
        ang_vel=np.zeros(3, dtype=np.float32),
        timestamp=0.0,
        base_pos=np.zeros(3, dtype=np.float32),
    )


def test_compute_fixed_yaw_alignment_quat_extracts_yaw_only() -> None:
    robot_quat = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)
    motion_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    yaw_offset = compute_fixed_yaw_alignment_quat(robot_quat, motion_quat)

    np.testing.assert_allclose(yaw_offset, robot_quat, atol=1e-6)


def test_rotate_motion_qpos_by_yaw_keeps_first_frame_position_with_pivot() -> None:
    qpos = np.zeros(36, dtype=np.float32)
    qpos[0:3] = np.array([1.5, 4.5, 0.82], dtype=np.float32)
    qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    yaw_offset = np.array([0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)

    rotate_motion_qpos_by_yaw(qpos, yaw_offset, pivot_pos_w=np.array([1.5, 4.5, 0.82], dtype=np.float32))

    np.testing.assert_allclose(qpos[0:3], np.array([1.5, 4.5, 0.82], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(qpos[3:7], yaw_offset, atol=1e-6)


@requires_mujoco
@_skip_no_xml
class TestVelCmdObservationBuilder:
    def test_output_dimension_is_166(self) -> None:
        builder = VelCmdObservationBuilder(_velcmd_cfg())
        obs = builder.build(
            _make_state(),
            _make_motion_qpos(),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
        assert builder.total_obs_size == 166
        assert obs.shape == (166,)
        assert obs.dtype == np.float32

    def test_projected_gravity_uses_root_quat_not_anchor_quat(self) -> None:
        builder = VelCmdObservationBuilder(_velcmd_cfg())
        builder._base.build = lambda *args, **kwargs: np.zeros(154, dtype=np.float32)  # type: ignore[method-assign]
        builder._base._run_fk = lambda *args, **kwargs: None  # type: ignore[method-assign]
        builder._base._anchor_body_id = 0
        builder._base._get_body_quat = lambda _body_id: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # type: ignore[method-assign]

        root_quat = np.array(
            [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0],
            dtype=np.float32,
        )
        state = RobotState(
            qpos=np.zeros(NUM_ACTIONS, dtype=np.float32),
            qvel=np.zeros(NUM_ACTIONS, dtype=np.float32),
            quat=root_quat,
            ang_vel=np.zeros(3, dtype=np.float32),
            timestamp=0.0,
            base_pos=np.zeros(3, dtype=np.float32),
        )

        obs = builder.build(
            state,
            _make_motion_qpos(),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(NUM_ACTIONS, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )

        expected_projected_gravity = _quat_rotate_np(
            _quat_inv_np(root_quat),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        np.testing.assert_allclose(obs[154:157], expected_projected_gravity, atol=1e-6)
