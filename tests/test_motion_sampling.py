from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from train_mimic.data.dataset_lib import merge_clip_dicts
from train_mimic.tasks.tracking.mdp.commands import MotionLib


def _clip_dict(num_frames: int = 6, fps: int = 1) -> dict[str, object]:
    time = np.arange(num_frames, dtype=np.float32)
    joint_pos = np.zeros((num_frames, 29), dtype=np.float32)
    joint_pos[:, 0] = time
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:, 0] = 1.0

    body_pos_w = np.zeros((num_frames, 3, 3), dtype=np.float32)
    body_pos_w[:, :, 0] = time[:, None]
    body_quat_w = np.zeros((num_frames, 3, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, 3, 3), dtype=np.float32)
    body_lin_vel_w[:, :, 0] = 1.0
    body_ang_vel_w = np.zeros((num_frames, 3, 3), dtype=np.float32)
    body_names = np.asarray(
        ["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"],
        dtype=str,
    )
    return {
        "fps": fps,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "body_names": body_names,
    }


def test_motion_lib_sample_times_respect_window_steps(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    merge_clip_dicts([_clip_dict()], motion_path, window_steps=(0, 2, -1))

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0, 2, -1),
    )
    motion_ids = torch.zeros(128, dtype=torch.long)
    times = motion.sample_times(motion_ids)
    sampled_frames = times * motion.clip_fps[motion_ids]

    assert float(torch.min(sampled_frames)) >= 1.0
    assert float(torch.max(sampled_frames)) < 3.0


def test_motion_lib_default_window_does_not_sample_wraparound_tail(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion_default.npz"
    merge_clip_dicts([_clip_dict()], motion_path, window_steps=(0,))

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0,),
    )
    motion_ids = torch.zeros(256, dtype=torch.long)
    times = motion.sample_times(motion_ids)
    sampled_frames = times * motion.clip_fps[motion_ids]

    assert float(torch.min(sampled_frames)) >= 0.0
    assert float(torch.max(sampled_frames)) < 5.0


def test_motion_lib_get_window_frames_returns_requested_offsets(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    merge_clip_dicts([_clip_dict()], motion_path, window_steps=(0, 2, -1))

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0, 2, -1),
    )
    frames = motion.get_window_frames(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([2.0], dtype=torch.float32),
    )

    assert frames["joint_pos"].shape == (1, 3, 29)
    assert torch.allclose(
        frames["joint_pos"][0, :, 0],
        torch.tensor([2.0, 4.0, 1.0], dtype=torch.float32),
    )
    assert frames["body_pos_w"].shape == (1, 3, 2, 3)
    assert torch.allclose(
        frames["body_pos_w"][0, :, 0, 0],
        torch.tensor([2.0, 4.0, 1.0], dtype=torch.float32),
    )

    current = motion.get_frames(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([2.0], dtype=torch.float32),
    )
    assert current["joint_pos"].shape == (1, 29)
    assert torch.allclose(current["joint_pos"][0, :1], torch.tensor([2.0], dtype=torch.float32))


def test_motion_lib_window_start_and_end_times_follow_valid_center_range(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion_windowed.npz"
    merge_clip_dicts([_clip_dict()], motion_path, window_steps=(0, 2, -1))

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0, 2, -1),
    )
    motion_ids = torch.tensor([0], dtype=torch.long)

    assert torch.allclose(motion.sample_start_times(motion_ids), torch.tensor([1.0]))
    assert torch.allclose(motion.clip_sample_end_s[motion_ids], torch.tensor([3.0]))
