from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure project root is on sys.path so `scripts.review` is importable
# even when running `pytest` directly (without `python -m pytest`).
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import h5py

from scripts.review import build_dataset_from_review
from scripts.review import export_reviewed_manifest
from scripts.review import init_review_manifest
from train_mimic.data.dataset_lib import (
    merge_clip_dicts_payload,
    write_hdf5_manifest,
    write_hdf5_motion_shard,
)
from train_mimic.data.review_lib import ReviewRow, load_review_state, save_review_state


BODY_NAMES = np.array(["pelvis", "torso"], dtype=str)


def _write_manifest(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "source",
                "file_rel",
                "num_frames",
                "fps",
                "resolved_split",
                "resolved_npz_path",
                "weight",
                "clip_index",
            ]
        )
        writer.writerow(
            [
                "src:clip_train",
                "src",
                "cache/clip_train.npz",
                4,
                24,
                "train",
                "/tmp/placeholder_train.npz",
                2.5,
                -1,
            ]
        )
        writer.writerow(
            [
                "src:clip_val",
                "src",
                "cache/clip_val.npz",
                5,
                30,
                "val",
                "/tmp/placeholder_val.npz",
                0.75,
                -1,
            ]
        )


def _write_npz(path: Path, *, num_frames: int, fps: int) -> None:
    joint_pos = np.linspace(0.0, 0.2, num_frames * 29, dtype=np.float32).reshape(num_frames, 29)
    joint_vel = np.gradient(joint_pos, axis=0).astype(np.float32)
    body_pos_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    body_pos_w[:, 0, 2] = np.linspace(0.75, 0.8, num_frames, dtype=np.float32)
    body_pos_w[:, 1, 2] = body_pos_w[:, 0, 2] + 0.3
    body_quat_w = np.zeros((num_frames, len(BODY_NAMES), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    np.savez(
        path,
        fps=fps,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        body_names=BODY_NAMES,
    )


def _clip_dict(*, num_frames: int, fps: int) -> dict[str, object]:
    joint_pos = np.linspace(0.0, 0.2, num_frames * 29, dtype=np.float32).reshape(num_frames, 29)
    joint_vel = np.gradient(joint_pos, axis=0).astype(np.float32)
    body_pos_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    body_pos_w[:, 0, 2] = np.linspace(0.75, 0.8, num_frames, dtype=np.float32)
    body_pos_w[:, 1, 2] = body_pos_w[:, 0, 2] + 0.3
    body_quat_w = np.zeros((num_frames, len(BODY_NAMES), 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((num_frames, len(BODY_NAMES), 3), dtype=np.float32)
    return {
        "fps": fps,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "body_names": BODY_NAMES,
    }


def _write_h5_split(path: Path, clip: dict[str, object]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    payload = merge_clip_dicts_payload([clip])
    shard_path = path / "shard_000.h5"
    h5_info = write_hdf5_motion_shard(payload, shard_path)
    write_hdf5_manifest(
        path,
        shard_infos=[h5_info],
        fps=int(payload["fps"]),
        body_names=np.asarray(payload["body_names"]),
    )
    return shard_path


def test_init_review_manifest_preserves_weight(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest_resolved.csv"
    review_path = tmp_path / "review_state.csv"
    _write_manifest(manifest_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_review_manifest.py",
            "--dataset",
            "demo",
            "--manifest",
            str(manifest_path),
            "--output",
            str(review_path),
        ],
    )

    init_review_manifest.main()

    rows = load_review_state(review_path)
    assert [row.weight for row in rows] == [2.5, 0.75]
    assert [row.decision for row in rows] == ["", ""]


def test_export_reviewed_manifest_preserves_weight_and_filters_keep(
    tmp_path: Path,
    monkeypatch,
) -> None:
    review_path = tmp_path / "review_state.csv"
    output_path = tmp_path / "filtered_manifest.csv"
    summary_path = tmp_path / "review_summary.json"
    save_review_state(
        [
            ReviewRow(
                clip_id="src:clip_train",
                source="src",
                file_rel="cache/clip_train.npz",
                resolved_npz_path="cache/clip_train.npz",
                resolved_split="train",
                num_frames=4,
                fps=24,
                duration_s=4 / 24,
                weight=2.5,
                decision="keep",
            ),
            ReviewRow(
                clip_id="src:clip_val",
                source="src",
                file_rel="cache/clip_val.npz",
                resolved_npz_path="cache/clip_val.npz",
                resolved_split="val",
                num_frames=5,
                fps=30,
                duration_s=5 / 30,
                weight=0.75,
                decision="drop",
            ),
        ],
        review_path,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_reviewed_manifest.py",
            "--review",
            str(review_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
        ],
    )

    export_reviewed_manifest.main()

    with output_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    assert rows[0]["clip_id"] == "src:clip_train"
    assert float(rows[0]["weight"]) == 2.5


def test_build_dataset_from_review_resamples_mixed_fps_and_preserves_weights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_h5 = _write_h5_split(tmp_path / "source_train", _clip_dict(num_frames=4, fps=24))
    val_h5 = _write_h5_split(tmp_path / "source_val", _clip_dict(num_frames=5, fps=30))

    filtered_manifest = tmp_path / "filtered_manifest.csv"
    with filtered_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "source",
                "file_rel",
                "num_frames",
                "fps",
                "resolved_split",
                "resolved_npz_path",
                "weight",
                "clip_index",
            ]
        )
        writer.writerow(
            [
                "src:clip_train",
                "src",
                str(train_h5),
                4,
                24,
                "train",
                str(train_h5),
                2.5,
                0,
            ]
        )
        writer.writerow(
            [
                "src:clip_val",
                "src",
                str(val_h5),
                5,
                30,
                "val",
                str(val_h5),
                0.75,
                0,
            ]
        )

    output_dir = tmp_path / "twist2_cleaned"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_dataset_from_review.py",
            "--filtered_manifest",
            str(filtered_manifest),
            "--output_dir",
            str(output_dir),
            "--target_fps",
            "30",
        ],
    )

    build_dataset_from_review.main()

    assert (output_dir / "train" / "manifest.json").is_file()
    assert (output_dir / "val" / "manifest.json").is_file()
    with h5py.File(output_dir / "train" / "shard_000.h5", "r") as train:
        assert int(train.attrs["fps"]) == 30
        assert train["clip_weights"][()].tolist() == [2.5]
        assert train["source_clip_lengths"][()].tolist() == [5]
    with h5py.File(output_dir / "val" / "shard_000.h5", "r") as val:
        assert int(val.attrs["fps"]) == 30
        assert val["clip_weights"][()].tolist() == [0.75]

    with (output_dir / "manifest_resolved.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["resolved_npz_path"].endswith("train/shard_000.h5")
    assert rows[0]["clip_index"] == "0"


def test_init_review_manifest_preserves_weight_metadata(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest_resolved.csv"
    review_path = tmp_path / "review_state.csv"
    _write_manifest(manifest_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "init_review_manifest.py",
            "--dataset",
            "demo",
            "--manifest",
            str(manifest_path),
            "--output",
            str(review_path),
        ],
    )

    init_review_manifest.main()

    rows = load_review_state(review_path)
    assert rows[0].weight == 2.5
    assert rows[1].weight == 0.75
