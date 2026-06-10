from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS
from teleopit.recording.pico_motion import (
    PicoDatasetSpec,
    RecordingState,
    ensure_pico_dataset_spec,
    qpos_sequence_to_motion_clip,
    sanitize_clip_name,
    unique_clip_path,
    write_motion_clip_npz,
)
from train_mimic.data.dataset_builder import load_dataset_spec
from train_mimic.data.dataset_lib import inspect_npz


class _FakeFkExtractor:
    num_actions = NUM_JOINTS

    def extract(
        self,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        joint_pos: np.ndarray,
        body_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        del root_quat_wxyz, joint_pos
        offsets = np.zeros((len(body_names), 3), dtype=np.float32)
        offsets[:, 2] = np.linspace(0.0, 0.2, len(body_names), dtype=np.float32)
        body_pos_w = root_pos[:, None, :] + offsets[None, :, :]
        body_quat_w = np.zeros((root_pos.shape[0], len(body_names), 4), dtype=np.float32)
        body_quat_w[..., 0] = 1.0
        return body_pos_w.astype(np.float32), body_quat_w


def _qpos_sequence(num_frames: int = 4) -> np.ndarray:
    qpos = np.zeros((num_frames, FULL_QPOS_DIM), dtype=np.float32)
    qpos[:, 0] = np.linspace(0.0, 0.3, num_frames, dtype=np.float32)
    qpos[:, 2] = 0.76
    qpos[:, 3] = 1.0
    qpos[:, 7] = np.linspace(0.0, 0.2, num_frames, dtype=np.float32)
    return qpos


def test_sanitize_clip_name_keeps_semantic_label_filesystem_safe() -> None:
    assert sanitize_clip_name(" Walk Forward Slow ") == "walk_forward_slow"
    assert sanitize_clip_name("turn-left/fast") == "turn-left_fast"
    with pytest.raises(ValueError, match="clip name"):
        sanitize_clip_name("...")


def test_unique_clip_path_adds_timestamp_and_avoids_overwrite(tmp_path: Path) -> None:
    now = datetime(2026, 6, 10, 14, 22, 33)
    first = unique_clip_path(tmp_path, "walk forward", now=now)
    assert first.name == "walk_forward_20260610_142233.npz"
    first.write_bytes(b"placeholder")

    second = unique_clip_path(tmp_path, "walk forward", now=now)
    assert second.name == "walk_forward_20260610_142233_001.npz"


def test_qpos_sequence_to_motion_clip_writes_standard_npz_fields() -> None:
    clip = qpos_sequence_to_motion_clip(_qpos_sequence(), fps=30, extractor=_FakeFkExtractor())
    assert int(clip["fps"]) == 30
    assert clip["joint_pos"].shape == (4, NUM_JOINTS)
    assert clip["joint_vel"].shape == (4, NUM_JOINTS)
    assert clip["body_pos_w"].shape[0] == 4
    assert clip["body_quat_w"].shape[-1] == 4


def test_qpos_sequence_to_motion_clip_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match=r"shape"):
        qpos_sequence_to_motion_clip(np.zeros((3, FULL_QPOS_DIM - 1)), fps=30, extractor=_FakeFkExtractor())

    bad = _qpos_sequence()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        qpos_sequence_to_motion_clip(bad, fps=30, extractor=_FakeFkExtractor())

    with pytest.raises(ValueError, match="at least 2"):
        qpos_sequence_to_motion_clip(_qpos_sequence(1), fps=30, extractor=_FakeFkExtractor())


def test_write_motion_clip_npz_and_inspect(tmp_path: Path) -> None:
    out = tmp_path / "clip.npz"
    write_motion_clip_npz(out, _qpos_sequence(), fps=30, extractor=_FakeFkExtractor())
    assert out.exists()
    meta = inspect_npz(out)
    assert meta.fps == 30
    assert meta.num_frames == 4


def test_ensure_pico_dataset_spec_preserves_existing_file(tmp_path: Path) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    spec_path = tmp_path / "pico_recorded.yaml"

    ensure_pico_dataset_spec(
        spec_path,
        clips_dir,
        spec=PicoDatasetSpec(dataset_name="pico_recorded", target_fps=30),
    )
    spec = load_dataset_spec(spec_path)
    assert spec.name == "pico_recorded"
    assert spec.target_fps == 30
    assert spec.sources[0].type == "npz"
    assert spec.sources[0].input == str(clips_dir)

    spec_path.write_text("name: hand_edited\n", encoding="utf-8")
    ensure_pico_dataset_spec(spec_path, clips_dir)
    assert spec_path.read_text(encoding="utf-8") == "name: hand_edited\n"


def test_recording_state_snapshot_does_not_clear_buffer() -> None:
    state = RecordingState("walk")
    state.start()
    state.append(_qpos_sequence(2)[0])
    state.append(_qpos_sequence(2)[1])

    clip_name, recording, frames = state.snapshot()
    assert clip_name == "walk"
    assert recording is True
    assert len(frames) == 2
    assert state.status()[2] == 2

    state.mark_saved()
    assert state.status()[1] is False
    assert state.status()[2] == 0


def test_recording_state_discard_clears_buffer() -> None:
    state = RecordingState("turn")
    state.start()
    state.append(_qpos_sequence(2)[0])

    clip_name, frame_count = state.discard()
    assert clip_name == "turn"
    assert frame_count == 1
    assert state.status()[1] is False
    assert state.status()[2] == 0
