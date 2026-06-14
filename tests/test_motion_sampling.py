from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from train_mimic.data.dataset_lib import write_hdf5_manifest, write_hdf5_motion_shard
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
    array_keys = [
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    ]
    clip_lengths = np.asarray([np.asarray(cd["joint_pos"]).shape[0] for cd in clip_dicts], dtype=np.int64)
    clip_starts = np.zeros(len(clip_lengths), dtype=np.int64)
    if len(clip_lengths) > 1:
        clip_starts[1:] = np.cumsum(clip_lengths[:-1])
    merged = {key: np.concatenate([np.asarray(cd[key]) for cd in clip_dicts], axis=0) for key in array_keys}
    merged["fps"] = int(clip_dicts[0]["fps"])
    merged["body_names"] = np.asarray(clip_dicts[0]["body_names"])
    merged["clip_starts"] = clip_starts
    merged["clip_lengths"] = clip_lengths
    merged["clip_fps"] = np.full(len(clip_dicts), int(clip_dicts[0]["fps"]), dtype=np.int64)
    merged["clip_weights"] = np.asarray(weights if weights is not None else [1.0] * len(clip_dicts), dtype=np.float64)
    shard_info = write_hdf5_motion_shard(merged, path / "shard_000.h5")
    write_hdf5_manifest(
        path,
        shard_infos=[shard_info],
        fps=int(clip_dicts[0]["fps"]),
        body_names=np.asarray(clip_dicts[0]["body_names"]),
    )
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


def test_motion_lib_rejects_shard_body_name_mismatch(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion_mismatch"
    clip = _clip_dict()
    shard0 = _write_shard_dir(motion_path, [clip])

    clip_bad = _clip_dict()
    clip_bad["body_names"] = np.asarray(
        ["pelvis", "right_ankle_roll_link", "left_ankle_roll_link"],
        dtype=str,
    )
    array_keys = [
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    ]
    merged = {key: np.asarray(clip_bad[key]) for key in array_keys}
    merged["fps"] = int(clip_bad["fps"])
    merged["body_names"] = np.asarray(clip_bad["body_names"])
    merged["clip_starts"] = np.asarray([0], dtype=np.int64)
    merged["clip_lengths"] = np.asarray([np.asarray(clip_bad["joint_pos"]).shape[0]], dtype=np.int64)
    merged["clip_fps"] = np.asarray([int(clip_bad["fps"])], dtype=np.int64)
    merged["clip_weights"] = np.asarray([1.0], dtype=np.float64)
    bad_info = write_hdf5_motion_shard(merged, motion_path / "shard_001.h5")

    (motion_path / "manifest.json").write_text(
        json.dumps({
            "format": "teleopit_motion_hdf5",
            "version": 1,
            "fps": 1,
            "body_names": np.asarray(clip["body_names"]).tolist(),
            "shards": [
                {"path": "shard_000.h5", "clips": 1, "frames": 6},
                {"path": "shard_001.h5", "clips": 1, "frames": int(bad_info["frames"])},
            ],
            "clips": 2,
            "frames": 12,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="body_names mismatch"):
        MotionLib(
            str(shard0),
            body_indexes=torch.tensor([0, 1], dtype=torch.long),
            window_steps=(0,),
        )


def test_write_hdf5_manifest_accepts_relative_shard_paths(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(tmp_path / "motion_relative_manifest", [_clip_dict()])
    manifest_path = write_hdf5_manifest(
        motion_path,
        shard_infos=[{"path": "shard_000.h5", "clips": 1, "frames": 6}],
        fps=1,
        body_names=np.asarray(["pelvis", "left_ankle_roll_link", "right_ankle_roll_link"]),
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["shards"][0]["path"] == "shard_000.h5"


class _FakeMotion:
    def __init__(self) -> None:
        self.clip_sample_start_s = torch.tensor([0.0, 1.0, 2.0])

    def sample_motion_ids(self, n: int) -> torch.Tensor:
        return torch.full((n,), 2, dtype=torch.long)

    def sample_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return torch.full_like(motion_ids, 9.0, dtype=torch.float32)


def test_motion_command_rewind_sampling_uses_failed_env_previous_time() -> None:
    command = MotionCommand.__new__(MotionCommand)
    command.cfg = SimpleNamespace(
        sampling_mode="rewind",
        rewind_prob=1.0,
        rewind_min_steps=2,
        rewind_max_steps=2,
    )
    command._env = SimpleNamespace(
        device="cpu",
        termination_manager=SimpleNamespace(
            terminated=torch.tensor([True, False, True])
        ),
    )
    command.motion = _FakeMotion()
    command.motion_ids = torch.tensor([0, 1, 1], dtype=torch.long)
    command.motion_times = torch.tensor([5.0, 6.0, 1.25], dtype=torch.float32)
    command._step_dt = 0.5

    command._rewind_sampling(torch.tensor([0, 1, 2], dtype=torch.long))

    assert torch.equal(command.motion_ids, torch.tensor([0, 2, 1]))
    assert torch.allclose(command.motion_times, torch.tensor([4.0, 9.0, 1.0]))


def test_motion_command_rewind_sampling_falls_back_to_uniform_when_disabled() -> None:
    command = MotionCommand.__new__(MotionCommand)
    command.cfg = SimpleNamespace(
        sampling_mode="rewind",
        rewind_prob=0.0,
        rewind_min_steps=2,
        rewind_max_steps=2,
    )
    command._env = SimpleNamespace(
        device="cpu",
        termination_manager=SimpleNamespace(terminated=torch.tensor([True])),
    )
    command.motion = _FakeMotion()
    command.motion_ids = torch.tensor([0], dtype=torch.long)
    command.motion_times = torch.tensor([5.0], dtype=torch.float32)
    command._step_dt = 0.5

    command._rewind_sampling(torch.tensor([0], dtype=torch.long))

    assert torch.equal(command.motion_ids, torch.tensor([2]))
    assert torch.allclose(command.motion_times, torch.tensor([9.0]))


def test_motion_command_rewind_sampling_rejects_invalid_step_range() -> None:
    command = MotionCommand.__new__(MotionCommand)
    command.cfg = SimpleNamespace(
        sampling_mode="rewind",
        rewind_prob=1.0,
        rewind_min_steps=3,
        rewind_max_steps=2,
    )
    command.motion_ids = torch.tensor([0], dtype=torch.long)
    command.motion_times = torch.tensor([5.0], dtype=torch.float32)
    command.motion = _FakeMotion()

    with pytest.raises(ValueError, match="rewind_max_steps"):
        command._rewind_sampling(torch.tensor([0], dtype=torch.long))
