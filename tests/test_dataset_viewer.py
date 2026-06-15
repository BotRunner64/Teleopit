from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.view.view_dataset import discover_dataset_clips
from train_mimic.data.dataset_lib import merge_clip_dicts_payload, write_hdf5_motion_shard


def _clip_dict(num_frames: int, fps: int = 30) -> dict[str, object]:
    root_quat_w = np.zeros((num_frames, 4), dtype=np.float32)
    root_quat_w[:, 0] = 1.0
    body_quat_w = np.zeros((num_frames, 1, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    return {
        "fps": fps,
        "root_pos": np.zeros((num_frames, 3), dtype=np.float32),
        "root_quat_w": root_quat_w,
        "joint_pos": np.zeros((num_frames, 29), dtype=np.float32),
        "joint_vel": np.zeros((num_frames, 29), dtype=np.float32),
        "body_pos_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((num_frames, 1, 3), dtype=np.float32),
        "body_names": np.asarray(["pelvis"], dtype=str),
    }


def test_discover_dataset_clips_reads_source_clip_metadata(tmp_path: Path) -> None:
    payload = merge_clip_dicts_payload([
        _clip_dict(4, fps=30),
        _clip_dict(6, fps=30),
    ])
    write_hdf5_motion_shard(payload, tmp_path / "nested" / "shard_000.h5")

    clips = discover_dataset_clips(tmp_path)

    assert [clip.clip_id for clip in clips] == ["nested/shard_000.h5#0", "nested/shard_000.h5#1"]
    assert [clip.num_frames for clip in clips] == [4, 6]
    assert [clip.clip_index for clip in clips] == [0, 1]


def test_discover_dataset_clips_uses_filename_for_single_h5_input(tmp_path: Path) -> None:
    shard_path = tmp_path / "shard_000.h5"
    payload = merge_clip_dicts_payload([
        _clip_dict(4, fps=30),
        _clip_dict(6, fps=30),
    ])
    write_hdf5_motion_shard(payload, shard_path)

    clips = discover_dataset_clips(shard_path)

    assert [clip.clip_id for clip in clips] == ["shard_000.h5#0", "shard_000.h5#1"]
