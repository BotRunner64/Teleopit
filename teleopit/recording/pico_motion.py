"""Helpers for recording Pico-retargeted G1 motion clips."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable

import numpy as np

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS, ROOT_DIM
from teleopit.runtime.assets import PROJECT_ROOT
from train_mimic.data.dataset_lib import inspect_clip_dict
from train_mimic.data.motion_fk import MotionFkExtractor, compute_body_velocities, finite_diff_velocity
from train_mimic.scripts.convert_pkl_to_npz import _MJLAB_G1_BODY_NAMES


@dataclass(frozen=True)
class PicoDatasetSpec:
    """Dataset spec defaults for Pico-recorded NPZ clips."""

    dataset_name: str = "pico_recorded"
    target_fps: int = 30
    source_name: str = "pico_clips"


class RecordingState:
    """Thread-safe state shared by terminal UI and retarget worker."""

    def __init__(self, clip_name: str | None = None) -> None:
        self._lock = threading.Lock()
        self.clip_name = clip_name
        self.recording = False
        self.qpos_buffer: list[np.ndarray] = []
        self.record_start_s: float | None = None

    def set_clip_name(self, clip_name: str) -> None:
        with self._lock:
            if self.recording:
                raise RuntimeError("cannot change clip name while recording")
            self.clip_name = clip_name
            self.qpos_buffer.clear()
            self.record_start_s = None

    def start(self) -> str:
        with self._lock:
            if self.clip_name is None:
                raise RuntimeError("clip name must be set before recording")
            if self.recording:
                return self.clip_name
            self.qpos_buffer.clear()
            self.recording = True
            self.record_start_s = time.monotonic()
            return self.clip_name

    def discard(self) -> tuple[str | None, int]:
        with self._lock:
            clip_name = self.clip_name
            frame_count = len(self.qpos_buffer)
            self.recording = False
            self.qpos_buffer.clear()
            self.record_start_s = None
            return clip_name, frame_count

    def snapshot(self) -> tuple[str | None, bool, list[np.ndarray]]:
        with self._lock:
            clip_name = self.clip_name
            recording = self.recording
            frames = [frame.copy() for frame in self.qpos_buffer]
            return clip_name, recording, frames

    def mark_saved(self) -> None:
        with self._lock:
            self.recording = False
            self.qpos_buffer.clear()
            self.record_start_s = None

    def append(self, qpos: np.ndarray) -> None:
        with self._lock:
            if self.recording:
                self.qpos_buffer.append(qpos.copy())

    def status(self) -> tuple[str | None, bool, int, float | None]:
        with self._lock:
            elapsed = None if self.record_start_s is None else time.monotonic() - self.record_start_s
            return self.clip_name, self.recording, len(self.qpos_buffer), elapsed


def sanitize_clip_name(raw_name: str) -> str:
    """Return a filesystem-friendly semantic clip name."""
    name = raw_name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_.-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        raise ValueError("clip name must contain at least one letter or digit")
    return name


def timestamp_suffix(now: datetime | None = None) -> str:
    current = now or datetime.now()
    return current.strftime("%Y%m%d_%H%M%S")


def unique_clip_path(output_dir: str | Path, clip_name: str, *, now: datetime | None = None) -> Path:
    """Build a non-overwriting NPZ path for one semantic clip."""
    out_dir = Path(output_dir).expanduser()
    safe_name = sanitize_clip_name(clip_name)
    stem = f"{safe_name}_{timestamp_suffix(now)}"
    candidate = out_dir / f"{stem}.npz"
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = out_dir / f"{stem}_{index:03d}.npz"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not allocate a unique clip path in {out_dir}")


def _display_path(path: Path, *, project_root: Path = PROJECT_ROOT) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def ensure_pico_dataset_spec(
    spec_path: str | Path,
    clips_dir: str | Path,
    *,
    spec: PicoDatasetSpec = PicoDatasetSpec(),
    overwrite: bool = False,
) -> Path:
    """Create the dataset YAML spec used to merge recorded Pico clips.

    Existing specs are preserved by default so hand-edited settings are not lost.
    """
    path = Path(spec_path).expanduser()
    if path.exists() and not overwrite:
        return path

    clips_path = Path(clips_dir).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    clips_display = _display_path(clips_path)
    content = (
        f"name: {spec.dataset_name}\n"
        f"target_fps: {int(spec.target_fps)}\n"
        "preprocess:\n"
        "  normalize_root_xy: true\n"
        "  ground_align: first_frame_foot\n"
        "sources:\n"
        f"  - name: {spec.source_name}\n"
        "    type: npz\n"
        f"    input: {clips_display}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def qpos_sequence_to_motion_clip(
    qpos_sequence: Iterable[np.ndarray] | np.ndarray,
    *,
    fps: int,
    extractor: Any | None = None,
    body_names: list[str] | None = None,
) -> dict[str, Any]:
    """Convert retargeted G1 qpos frames into a standard training motion clip."""
    qpos = np.asarray(list(qpos_sequence) if not isinstance(qpos_sequence, np.ndarray) else qpos_sequence, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != FULL_QPOS_DIM:
        raise ValueError(f"qpos_sequence must have shape (T,{FULL_QPOS_DIM}), got {qpos.shape}")
    if qpos.shape[0] < 2:
        raise ValueError("qpos_sequence must contain at least 2 frames")
    if int(fps) <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos_sequence contains NaN/Inf")

    names = list(_MJLAB_G1_BODY_NAMES if body_names is None else body_names)
    root_pos = qpos[:, 0:3].astype(np.float32, copy=False)
    root_quat_wxyz = qpos[:, 3:7].astype(np.float32, copy=False)
    joint_pos = qpos[:, ROOT_DIM:ROOT_DIM + NUM_JOINTS].astype(np.float32, copy=False)
    if joint_pos.shape[1] != NUM_JOINTS:
        raise ValueError(f"joint_pos must have {NUM_JOINTS} columns, got {joint_pos.shape}")

    dt = 1.0 / float(fps)
    joint_vel = finite_diff_velocity(joint_pos, dt)

    fk_extractor = extractor or MotionFkExtractor()
    body_pos_w, body_quat_w = fk_extractor.extract(root_pos, root_quat_wxyz, joint_pos, names)
    body_lin_vel_w, body_ang_vel_w = compute_body_velocities(body_pos_w, body_quat_w, dt)

    clip = {
        "fps": int(fps),
        "root_pos": root_pos.astype(np.float32, copy=False),
        "root_quat_w": root_quat_wxyz.astype(np.float32, copy=False),
        "joint_pos": joint_pos.astype(np.float32, copy=False),
        "joint_vel": joint_vel.astype(np.float32, copy=False),
        "body_pos_w": np.asarray(body_pos_w, dtype=np.float32),
        "body_quat_w": np.asarray(body_quat_w, dtype=np.float32),
        "body_lin_vel_w": np.asarray(body_lin_vel_w, dtype=np.float32),
        "body_ang_vel_w": np.asarray(body_ang_vel_w, dtype=np.float32),
        "body_names": np.asarray(names, dtype=str),
    }
    inspect_clip_dict(clip)
    return clip


def write_motion_clip_npz(
    output_path: str | Path,
    qpos_sequence: Iterable[np.ndarray] | np.ndarray,
    *,
    fps: int,
    extractor: Any | None = None,
) -> Path:
    """Write a retargeted qpos sequence as an atomically replaced motion NPZ."""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    clip = qpos_sequence_to_motion_clip(qpos_sequence, fps=fps, extractor=extractor)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.savez(handle, **clip)
        path_tmp = tmp_path
        path_tmp.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path
