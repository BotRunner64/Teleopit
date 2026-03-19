"""Dataset clip preprocessing and filtering utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from train_mimic.data.dataset_lib import inspect_clip_dict

GROUND_ALIGN_MODES = {"none", "clip_min_foot", "frame_min_foot"}


@dataclass(frozen=True)
class DatasetPreprocessSpec:
    normalize_root_xy: bool = False
    root_body_name: str = "pelvis"
    ground_align: str = "none"
    foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    min_frames: int = 1
    max_root_lin_vel: float | None = None
    min_peak_body_height: float | None = None
    max_all_off_ground_s: float | None = None
    off_ground_height: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_preprocess_spec(spec: DatasetPreprocessSpec) -> DatasetPreprocessSpec:
    if spec.ground_align not in GROUND_ALIGN_MODES:
        raise ValueError(
            f"preprocess.ground_align must be one of {sorted(GROUND_ALIGN_MODES)}, got {spec.ground_align!r}"
        )
    if len(spec.foot_body_names) != 2:
        raise ValueError("preprocess.foot_body_names must contain exactly two body names")
    if spec.min_frames <= 0:
        raise ValueError(f"preprocess.min_frames must be > 0, got {spec.min_frames}")
    if spec.max_root_lin_vel is not None and spec.max_root_lin_vel <= 0.0:
        raise ValueError(
            f"preprocess.max_root_lin_vel must be > 0, got {spec.max_root_lin_vel}"
        )
    if spec.min_peak_body_height is not None and spec.min_peak_body_height < 0.0:
        raise ValueError(
            "preprocess.min_peak_body_height must be >= 0, "
            f"got {spec.min_peak_body_height}"
        )
    if spec.max_all_off_ground_s is not None and spec.max_all_off_ground_s <= 0.0:
        raise ValueError(
            "preprocess.max_all_off_ground_s must be > 0, "
            f"got {spec.max_all_off_ground_s}"
        )
    if spec.off_ground_height < 0.0:
        raise ValueError(
            f"preprocess.off_ground_height must be >= 0, got {spec.off_ground_height}"
        )
    return spec


def _body_index(body_names: np.ndarray, body_name: str, *, label: str) -> int:
    names = [str(name) for name in body_names.tolist()]
    try:
        return names.index(body_name)
    except ValueError as exc:
        raise ValueError(f"{label} body {body_name!r} not found in body_names") from exc


def _longest_true_run(mask: np.ndarray) -> int:
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1-D, got {mask.shape}")
    if mask.size == 0 or not np.any(mask):
        return 0
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return int(np.max(ends - starts)) if starts.size else 0


def preprocess_clip_dict(
    payload: Mapping[str, Any],
    *,
    spec: DatasetPreprocessSpec,
    clip_label: str,
) -> dict[str, Any]:
    """Apply motion_tracking-style preprocessing to a standard clip dict."""
    spec = validate_preprocess_spec(spec)

    result = {
        "fps": int(payload["fps"]),
        "joint_pos": np.asarray(payload["joint_pos"]).copy(),
        "joint_vel": np.asarray(payload["joint_vel"]).copy(),
        "body_pos_w": np.asarray(payload["body_pos_w"]).copy(),
        "body_quat_w": np.asarray(payload["body_quat_w"]).copy(),
        "body_lin_vel_w": np.asarray(payload["body_lin_vel_w"]).copy(),
        "body_ang_vel_w": np.asarray(payload["body_ang_vel_w"]).copy(),
        "body_names": np.asarray(payload["body_names"]).copy(),
    }
    inspect_clip_dict(result)

    fps = int(result["fps"])
    body_pos_w = result["body_pos_w"]
    body_lin_vel_w = result["body_lin_vel_w"]
    body_names = np.asarray(result["body_names"])
    num_frames = int(result["joint_pos"].shape[0])

    if num_frames < spec.min_frames:
        raise ValueError(
            f"{clip_label}: clip too short after conversion ({num_frames} < {spec.min_frames})"
        )

    root_index: int | None = None
    if spec.normalize_root_xy or spec.max_root_lin_vel is not None:
        root_index = _body_index(body_names, spec.root_body_name, label="root")

    if spec.normalize_root_xy:
        assert root_index is not None
        offset_xy = body_pos_w[0, root_index, :2].copy()
        body_pos_w[..., 0] -= offset_xy[0]
        body_pos_w[..., 1] -= offset_xy[1]

    foot_indices: tuple[int, int] | None = None
    if spec.ground_align != "none":
        foot_indices = (
            _body_index(body_names, spec.foot_body_names[0], label="foot"),
            _body_index(body_names, spec.foot_body_names[1], label="foot"),
        )
        foot_z = body_pos_w[:, foot_indices, 2]
        if spec.ground_align == "clip_min_foot":
            body_pos_w[..., 2] -= float(np.min(foot_z))
        elif spec.ground_align == "frame_min_foot":
            body_pos_w[..., 2] -= np.min(foot_z, axis=1, keepdims=True)

    if spec.max_root_lin_vel is not None:
        assert root_index is not None
        peak_root_lin_vel = float(np.max(np.abs(body_lin_vel_w[:, root_index, :])))
        if peak_root_lin_vel > spec.max_root_lin_vel:
            raise ValueError(
                f"{clip_label}: root linear velocity spike {peak_root_lin_vel:.4f} exceeds "
                f"{spec.max_root_lin_vel:.4f}"
            )

    if spec.max_all_off_ground_s is not None:
        min_body_z = np.min(body_pos_w[:, :, 2], axis=1)
        all_off = min_body_z > spec.off_ground_height
        longest_run = _longest_true_run(all_off)
        if longest_run > int(round(spec.max_all_off_ground_s * fps)):
            raise ValueError(
                f"{clip_label}: all bodies off ground for {longest_run / fps:.3f}s, exceeds "
                f"{spec.max_all_off_ground_s:.3f}s"
            )

    if spec.min_peak_body_height is not None:
        peak_height = float(np.max(body_pos_w[:, :, 2]))
        if peak_height < spec.min_peak_body_height:
            raise ValueError(
                f"{clip_label}: peak body height {peak_height:.4f} below "
                f"{spec.min_peak_body_height:.4f}"
            )

    inspect_clip_dict(result)
    return result
