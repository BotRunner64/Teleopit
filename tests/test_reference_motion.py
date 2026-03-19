from __future__ import annotations

import numpy as np

from teleopit.sim.reference_motion import (
    OfflineReferenceMotion,
    interpolate_retarget_qpos,
    slerp_quat_wxyz,
)


def test_slerp_quat_wxyz_half_turn_midpoint_is_normalized() -> None:
    quat0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    out = slerp_quat_wxyz(quat0, quat1, 0.5)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)
    np.testing.assert_allclose(out, np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]), atol=1e-6)


def test_interpolate_retarget_qpos_slerps_root_quaternion_and_lerps_joints() -> None:
    qpos0 = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    qpos1 = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 10.0], dtype=np.float64)
    out = interpolate_retarget_qpos(qpos0, qpos1, 0.25)
    np.testing.assert_allclose(out[:3], np.array([0.25, 0.5, 1.5]), atol=1e-6)
    np.testing.assert_allclose(out[7], 2.5, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out[3:7]), 1.0, atol=1e-6)


class _FakeInputProvider:
    fps = 30

    def __init__(self) -> None:
        self.frames = [
            {"pelvis": (np.array([float(i), 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))}
            for i in range(3)
        ]

    def __len__(self) -> int:
        return len(self.frames)

    def get_frame_by_index(self, index: int):
        frame = self.frames[index]
        return {
            key: (np.asarray(value[0], dtype=np.float64).copy(), np.asarray(value[1], dtype=np.float64).copy())
            for key, value in frame.items()
        }


class _FakeRetargeter:
    def retarget(self, human_frame):
        pelvis_x = float(human_frame["pelvis"][0][0])
        return np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, pelvis_x], dtype=np.float64)


def test_offline_reference_motion_samples_interpolated_qpos_at_policy_time() -> None:
    sampler = OfflineReferenceMotion(_FakeInputProvider(), _FakeRetargeter())
    sample = sampler.sample(0.02)
    assert sample is not None
    np.testing.assert_allclose(sample.qpos[7], 0.6, atol=1e-6)
    np.testing.assert_allclose(sample.human_frame["pelvis"][0], np.array([0.6, 0.0, 0.0]), atol=1e-6)
