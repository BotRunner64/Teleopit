from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import h5py

from train_mimic.data.dataset_lib import (
    PRECOMPUTED_MOTION_VERSION,
    compute_dataset_stats,
    write_precomputed_motion_shard,
    write_hdf5_motion_shard,
)
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
        "root_pos": body_pos_w[:, 0],
        "root_quat_w": body_quat_w[:, 0],
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
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    array_keys = [
        "root_pos", "root_quat_w",
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
    shard_path = path / "shard_000.h5"
    _write_precomputed_from_merged(shard_path, merged)
    return path


def _write_precomputed_from_merged(shard_path: Path, merged: dict[str, object]) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    clip_starts = np.asarray(merged["clip_starts"], dtype=np.int64)
    clip_lengths = np.asarray(merged["clip_lengths"], dtype=np.int64)
    clip_fps = np.asarray(merged["clip_fps"], dtype=np.int64)
    with h5py.File(shard_path, "w") as h5:
        h5.attrs["format"] = "teleopit_precomputed_motion_hdf5"
        h5.attrs["version"] = PRECOMPUTED_MOTION_VERSION
        h5.create_dataset(
            "body_names",
            data=np.asarray(merged["body_names"]).astype(str).astype(object),
            dtype=str_dt,
        )
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            h5.create_dataset(key, data=np.asarray(merged[key], dtype=np.float32), chunks=True)
        h5.create_dataset("clip_starts", data=clip_starts)
        h5.create_dataset("clip_lengths", data=clip_lengths)
        h5.create_dataset("clip_fps", data=clip_fps)
        h5.create_dataset("source_clip_ids", data=np.arange(len(clip_lengths), dtype=np.int64))
        h5.create_dataset("source_start_frames", data=np.zeros(len(clip_lengths), dtype=np.int64))
        h5.create_dataset("source_clip_starts", data=clip_starts)
        h5.create_dataset("source_clip_lengths", data=clip_lengths)
        h5.create_dataset("source_clip_fps", data=clip_fps)


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

    current = motion.get_frames(
        torch.tensor([0], dtype=torch.long),
        torch.tensor([2.0], dtype=torch.float32),
    )
    assert current["joint_pos"].shape == (1, 29)
    assert torch.allclose(current["joint_pos"][0, :1], torch.tensor([2.0], dtype=torch.float32))


def test_motion_lib_frame_offsets_select_requested_clip(tmp_path: Path) -> None:
    clip0 = _clip_dict(num_frames=3)
    clip1 = _clip_dict(num_frames=7)
    clip1["root_pos"] = np.asarray(clip1["root_pos"]).copy()
    clip1["root_pos"][:, 0] += 100.0
    clip1["joint_pos"] = np.asarray(clip1["joint_pos"]).copy()
    clip1["joint_pos"][:, 0] += 100.0
    clip1["body_pos_w"] = np.asarray(clip1["body_pos_w"]).copy()
    clip1["body_pos_w"][:, :, 0] += 100.0

    motion_path = _write_shard_dir(tmp_path / "motion_flat_offsets", [clip0, clip1])
    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0,),
    )

    frames = motion.get_frames(
        torch.tensor([1], dtype=torch.long),
        torch.tensor([2.0], dtype=torch.float32),
    )

    assert torch.allclose(frames["joint_pos"][0, :1], torch.tensor([102.0]))
    assert torch.allclose(frames["body_pos_w"][0, 0], torch.tensor([102.0, 0.0, 0.0]))


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
    assert torch.isfinite(frames["body_pos_w"]).all()


def test_motion_lib_rejects_minimal_motion_shards(tmp_path: Path) -> None:
    path = tmp_path / "motion_minimal"
    path.mkdir()
    clip = _clip_dict()
    merged = {
        "fps": int(clip["fps"]),
        "root_pos": np.asarray(clip["root_pos"]),
        "root_quat_w": np.asarray(clip["root_quat_w"]),
        "joint_pos": np.asarray(clip["joint_pos"]),
        "body_names": np.asarray(clip["body_names"]),
        "clip_starts": np.asarray([0], dtype=np.int64),
        "clip_lengths": np.asarray([np.asarray(clip["joint_pos"]).shape[0]], dtype=np.int64),
        "clip_fps": np.asarray([int(clip["fps"])], dtype=np.int64),
    }
    write_hdf5_motion_shard(merged, path / "shard_000.h5")

    with pytest.raises(FileNotFoundError, match="precomputed Teleopit"):
        MotionLib(
            str(path),
            body_indexes=torch.tensor([0, 1], dtype=torch.long),
            window_steps=(0,),
        )


def test_precomputed_stats_preserve_source_clip_ids_for_windowed_shards(tmp_path: Path) -> None:
    clip = _clip_dict(num_frames=12, fps=30)
    shard_path = tmp_path / "motion" / "shard_000.h5"
    shard_path.parent.mkdir(parents=True)
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(shard_path, "w") as h5:
        h5.attrs["format"] = "teleopit_precomputed_motion_hdf5"
        h5.attrs["version"] = PRECOMPUTED_MOTION_VERSION
        h5.create_dataset("body_names", data=np.asarray(clip["body_names"]).astype(object), dtype=str_dt)
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            h5.create_dataset(key, data=np.asarray(clip[key], dtype=np.float32), chunks=True)
        h5.create_dataset("clip_starts", data=np.asarray([0, 4, 8], dtype=np.int64))
        h5.create_dataset("clip_lengths", data=np.asarray([6, 6, 4], dtype=np.int64))
        h5.create_dataset("clip_fps", data=np.asarray([30, 30, 30], dtype=np.int64))
        h5.create_dataset("source_clip_ids", data=np.asarray([0, 0, 0], dtype=np.int64))
        h5.create_dataset("source_start_frames", data=np.asarray([0, 4, 8], dtype=np.int64))
        h5.create_dataset("source_clip_starts", data=np.asarray([0], dtype=np.int64))
        h5.create_dataset("source_clip_lengths", data=np.asarray([12], dtype=np.int64))
        h5.create_dataset("source_clip_fps", data=np.asarray([30], dtype=np.int64))

    stats = compute_dataset_stats(shard_path.parent, precomputed=True)

    assert stats["windows"] == 3
    assert stats["source_clips"] == 1
    assert stats["duration_s"] == pytest.approx(12 / 30)


def test_write_precomputed_motion_shard_copies_window_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMotionFkExtractor:
        def __init__(self, model_path: object = None) -> None:
            del model_path

        def extract(
            self,
            root_pos: np.ndarray,
            root_quat_w: np.ndarray,
            joint_pos: np.ndarray,
            body_names: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            del joint_pos
            bodies = int(len(body_names))
            body_pos_w = np.repeat(np.asarray(root_pos, dtype=np.float32)[:, None, :], bodies, axis=1)
            body_quat_w = np.repeat(np.asarray(root_quat_w, dtype=np.float32)[:, None, :], bodies, axis=1)
            return body_pos_w, body_quat_w

    monkeypatch.setattr("train_mimic.data.motion_fk.MotionFkExtractor", _FakeMotionFkExtractor)
    clip = _clip_dict(num_frames=12, fps=30)
    merged = {
        "fps": int(clip["fps"]),
        "root_pos": np.asarray(clip["root_pos"], dtype=np.float32),
        "root_quat_w": np.asarray(clip["root_quat_w"], dtype=np.float32),
        "joint_pos": np.asarray(clip["joint_pos"], dtype=np.float32),
        "body_names": np.asarray(clip["body_names"]).astype(str),
        "clip_starts": np.asarray([0], dtype=np.int64),
        "clip_lengths": np.asarray([12], dtype=np.int64),
        "clip_fps": np.asarray([30], dtype=np.int64),
    }
    minimal_path = tmp_path / "minimal" / "shard_000.h5"
    precomputed_path = tmp_path / "precomputed" / "shard_000.h5"
    write_hdf5_motion_shard(
        merged,
        minimal_path,
        max_window_frames=6,
        overlap_frames=2,
    )

    write_precomputed_motion_shard(minimal_path, precomputed_path)

    with h5py.File(precomputed_path, "r") as h5:
        assert h5.attrs["version"] == PRECOMPUTED_MOTION_VERSION
        assert h5["clip_starts"][()].tolist() == [0, 4, 6]
        assert h5["clip_lengths"][()].tolist() == [6, 6, 6]
        assert h5["source_clip_ids"][()].tolist() == [0, 0, 0]
        assert h5["source_start_frames"][()].tolist() == [0, 4, 6]
        assert h5["source_clip_starts"][()].tolist() == [0]
        assert h5["source_clip_lengths"][()].tolist() == [12]


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


def test_motion_lib_sampling_weights_follow_valid_duration(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(
        tmp_path / "motion_weighted",
        [
            _clip_dict(num_frames=3, fps=10),
            _clip_dict(num_frames=6, fps=10),
            _clip_dict(num_frames=11, fps=10),
        ],
    )

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0,),
    )

    assert torch.allclose(
        motion.sample_weights,
        torch.tensor([0.2 / 1.7, 0.5 / 1.7, 1.0 / 1.7], dtype=torch.float32),
    )


def test_motion_lib_samples_ids_randomly_by_valid_duration(tmp_path: Path) -> None:
    motion_path = _write_shard_dir(
        tmp_path / "motion_weighted_global",
        [
            _clip_dict(num_frames=3, fps=10),
            _clip_dict(num_frames=11, fps=10),
        ],
    )

    motion = MotionLib(
        str(motion_path),
        body_indexes=torch.tensor([0, 1], dtype=torch.long),
        window_steps=(0,),
    )

    ids = motion.sample_motion_ids(2048)
    counts = torch.bincount(ids.cpu(), minlength=2).float()

    assert counts[1] > counts[0] * 3.0


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
        "root_pos", "root_quat_w",
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    ]
    merged = {key: np.asarray(clip_bad[key]) for key in array_keys}
    merged["fps"] = int(clip_bad["fps"])
    merged["body_names"] = np.asarray(clip_bad["body_names"])
    merged["clip_starts"] = np.asarray([0], dtype=np.int64)
    merged["clip_lengths"] = np.asarray([np.asarray(clip_bad["joint_pos"]).shape[0]], dtype=np.int64)
    merged["clip_fps"] = np.asarray([int(clip_bad["fps"])], dtype=np.int64)
    bad_shard = motion_path / "shard_001.h5"
    _write_precomputed_from_merged(bad_shard, merged)

    with pytest.raises(ValueError, match="body_names"):
        MotionLib(
            str(shard0),
            body_indexes=torch.tensor([0, 1], dtype=torch.long),
            window_steps=(0,),
        )


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
