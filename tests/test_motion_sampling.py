from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_mimic.data.dataset_lib import merge_clip_dicts
from train_mimic.tasks.tracking.mdp.commands import MotionCommand, MotionLib


def _clip_dict(num_frames: int = 6, fps: int = 1) -> dict[str, object]:
    time = np.arange(num_frames, dtype=np.float32)
    joint_pos = np.zeros((num_frames, 29), dtype=np.float32)
    joint_pos[:, 0] = time
    joint_vel = np.zeros_like(joint_pos)
    joint_vel[:, 0] = 1.0

    body_pos_w = np.zeros((num_frames, 3, 3), dtype=np.float32)
    body_pos_w[:, :, 0] = time[:, None]
    body_pos_w[:, :, 1] = np.arange(3, dtype=np.float32)[None, :]
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


def _write_shard_dir(
    path: Path,
    clip_dicts: list[dict[str, object]],
    *,
    weights: list[float] | None = None,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    merge_clip_dicts(clip_dicts, path / "shard_000.npz", weights=weights)
    return path


def test_motion_lib_sample_times_respect_window_steps(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(tmp_path / "motion", [_clip_dict()])

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
    motion_path = _write_shard_dir(tmp_path / "motion_default", [_clip_dict()])

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
    motion_path = _write_shard_dir(tmp_path / "motion", [_clip_dict()])

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


def test_motion_lib_selects_bodies_by_dataset_names(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(tmp_path / "motion_named_bodies", [_clip_dict()])

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([99, 0], dtype=torch.long),
        body_names=["right_ankle_roll_link", "pelvis"],
        window_steps=(0,),
    )
    frames = motion.get_frames(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([2.0], dtype=torch.float32),
    )

    assert frames["body_pos_w"].shape == (1, 2, 3)
    assert torch.allclose(
        frames["body_pos_w"][0, :, 1],
        torch.tensor([2.0, 0.0], dtype=torch.float32),
    )


def test_motion_lib_window_start_and_end_times_follow_valid_center_range(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(tmp_path / "motion_windowed", [_clip_dict()])

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0, 2, -1),
    )
    motion_ids = torch.tensor([0], dtype=torch.long)

    assert torch.allclose(motion.sample_start_times(motion_ids), torch.tensor([1.0]))
    assert torch.allclose(motion.clip_sample_end_s[motion_ids], torch.tensor([3.0]))


def test_motion_lib_adaptive_bins_are_clip_local(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(
        tmp_path / "motion_adaptive_bins",
        [_clip_dict(num_frames=6), _clip_dict(num_frames=8)],
        weights=[1.0, 3.0],
    )
    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0,),
    )

    num_bins = motion.prepare_adaptive_sampling(bin_size_frames=2)

    assert num_bins == 7
    assert motion.adaptive_bin_clip_ids.tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert motion.adaptive_bin_start_frames.tolist() == [0, 2, 4, 0, 2, 4, 6]
    assert motion.adaptive_bin_end_frames.tolist() == [2, 4, 5, 2, 4, 6, 7]

    clip0_mass = motion.adaptive_bin_base_probs[:3].sum()
    clip1_mass = motion.adaptive_bin_base_probs[3:].sum()
    assert torch.allclose(clip0_mass, torch.tensor(0.25), atol=1e-6)
    assert torch.allclose(clip1_mass, torch.tensor(0.75), atol=1e-6)


def test_motion_lib_adaptive_sampling_never_crosses_clip_boundaries(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(
        tmp_path / "motion_adaptive_sample",
        [_clip_dict(num_frames=6), _clip_dict(num_frames=8)],
    )
    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0,),
    )
    motion.prepare_adaptive_sampling(bin_size_frames=2)

    motion_ids, motion_times, bins = motion.sample_adaptive_times(
        motion.adaptive_bin_base_probs,
        512,
    )
    sampled_frames = motion_times * motion.clip_fps[motion_ids]

    assert torch.all(sampled_frames >= motion.clip_sample_starts[motion_ids])
    assert torch.all(sampled_frames < motion.clip_sample_ends[motion_ids])
    assert torch.equal(motion.adaptive_bin_clip_ids[bins], motion_ids)
    assert torch.equal(motion.adaptive_bins_for(motion_ids, motion_times), bins)


def test_motion_command_adaptive_sampling_state_round_trips() -> None:
    source = MotionCommand.__new__(MotionCommand)
    source.cfg = SimpleNamespace(sampling_mode="adaptive", adaptive_bin_size_frames=2)
    source.adaptive_bin_failed_count = torch.tensor([0.0, 2.0, 4.0])
    source._current_adaptive_bin_failed = torch.tensor([1.0, 0.0, 3.0])

    target = MotionCommand.__new__(MotionCommand)
    target.cfg = SimpleNamespace(sampling_mode="adaptive", adaptive_bin_size_frames=2)
    target._env = SimpleNamespace(device="cpu")
    target.adaptive_bin_failed_count = torch.zeros(3)
    target._current_adaptive_bin_failed = torch.zeros(3)

    state = source.get_adaptive_sampling_state()
    assert state is not None
    target.load_adaptive_sampling_state(state)

    assert torch.equal(target.adaptive_bin_failed_count, source.adaptive_bin_failed_count)
    assert torch.equal(
        target._current_adaptive_bin_failed,
        source._current_adaptive_bin_failed,
    )
