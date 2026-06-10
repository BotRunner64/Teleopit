"""Interactively record Pico-retargeted G1 motion clips as training NPZ files."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from teleopit.constants import FULL_QPOS_DIM
from teleopit.inputs.pico4_provider import Pico4InputProvider
from teleopit.recording.pico_motion import (
    PicoDatasetSpec,
    RecordingState,
    ensure_pico_dataset_spec,
    sanitize_clip_name,
    unique_clip_path,
    write_motion_clip_npz,
)
from teleopit.retargeting.core import RetargetingModule
from teleopit.runtime.assets import PROJECT_ROOT, UNITREE_G1_MJLAB_XML, missing_gmr_assets_message
from teleopit.runtime.common import cfg_get
from teleopit.runtime.terminal_keyboard import TerminalKeyboardReader
from teleopit.sim.viewer_subprocess import start_robot_viewer


class RetargetPreview:
    """Small wrapper around the existing MuJoCo retarget viewer subprocess."""

    def __init__(self, xml_path: str | Path = UNITREE_G1_MJLAB_XML, *, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._proc = None
        self._arr = None
        self._alive = None
        self._shutdown = None
        self._qpos_len = FULL_QPOS_DIM
        if not self.enabled:
            return

        path = Path(xml_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(missing_gmr_assets_message(path, label="G1 MuJoCo XML for retarget viewer"))

        import mujoco

        model = mujoco.MjModel.from_xml_path(str(path))
        self._qpos_len = int(model.nq)
        self._proc, self._arr, self._alive, self._shutdown = start_robot_viewer(
            str(path),
            self._qpos_len,
            True,
            "Retarget",
            900,
            50,
        )
        initial = np.zeros((self._qpos_len,), dtype=np.float64)
        if self._qpos_len > 3:
            initial[3] = 1.0
        self.update(initial)

    def update(self, qpos: np.ndarray) -> None:
        if not self.enabled or self._arr is None:
            return
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos_arr.shape[0] > self._qpos_len:
            raise ValueError(f"retarget qpos has {qpos_arr.shape[0]} values, viewer accepts {self._qpos_len}")
        out = np.zeros((self._qpos_len,), dtype=np.float64)
        out[: qpos_arr.shape[0]] = qpos_arr
        with self._arr.get_lock():
            self._arr[: self._qpos_len] = out.tolist()

    def close(self) -> None:
        if self._shutdown is not None:
            self._shutdown.set()
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()


class RetargetRecordingWorker:
    """Continuously retarget Pico frames without blocking terminal input."""

    def __init__(
        self,
        *,
        provider: Pico4InputProvider,
        retargeter: RetargetingModule,
        preview: RetargetPreview,
        state: RecordingState,
        target_fps: int,
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        self._provider = provider
        self._retargeter = retargeter
        self._preview = preview
        self._state = state
        self._target_fps = int(target_fps)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pico_retarget_recorder", daemon=True)
        self._last_error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)

    def raise_if_failed(self) -> None:
        if self._last_error is not None:
            raise RuntimeError(f"retarget worker failed: {self._last_error}") from self._last_error

    def _run(self) -> None:
        last_seq: int | None = None
        dt = 1.0 / float(self._target_fps)
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_tick:
                    self._stop_event.wait(timeout=min(next_tick - now, 0.01))
                    continue
                next_tick = now + dt
                qpos, last_seq = _retarget_latest_frame(self._provider, self._retargeter, last_seq=last_seq)
                if qpos is None:
                    continue
                self._preview.update(qpos)
                self._state.append(qpos)
        except BaseException as exc:
            self._last_error = exc


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _build_provider(cfg: DictConfig) -> Pico4InputProvider:
    input_cfg = cfg_get(cfg, "input", {}) or {}
    return Pico4InputProvider(
        human_format=str(cfg_get(input_cfg, "human_format", "pico_bridge")),
        timeout=float(cfg_get(input_cfg, "pico4_timeout", 60.0)),
        buffer_size=int(cfg_get(input_cfg, "pico4_buffer_size", 60)),
        timestamp_gap_reset_s=float(cfg_get(input_cfg, "pico4_timestamp_gap_reset_s", 0.15)),
        pause_button=cfg_get(input_cfg, "pause_button", "A"),
        pause_debounce_s=float(cfg_get(input_cfg, "pause_debounce_s", 0.25)),
        bridge_host=str(cfg_get(input_cfg, "bridge_host", "0.0.0.0")),
        bridge_port=int(cfg_get(input_cfg, "bridge_port", 63901)),
        bridge_discovery=bool(cfg_get(input_cfg, "bridge_discovery", True)),
        bridge_advertise_ip=cfg_get(input_cfg, "bridge_advertise_ip", None),
        bridge_video=None,
        bridge_video_enabled=False,
        bridge_start_timeout=float(cfg_get(input_cfg, "bridge_start_timeout", 10.0)),
        bridge_history_size=int(cfg_get(input_cfg, "bridge_history_size", 120)),
    )


def _build_retargeter(cfg: DictConfig, provider: Pico4InputProvider) -> RetargetingModule:
    input_cfg = cfg_get(cfg, "input", {}) or {}
    human_height = cfg_get(input_cfg, "human_height", 1.75)
    return RetargetingModule(
        robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
        human_format=str(provider.human_format),
        actual_human_height=float(human_height),
    )


def _prompt_clip_name() -> str | None:
    while True:
        raw = input("\nClip name (semantic label, or q to quit): ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        try:
            return sanitize_clip_name(raw)
        except ValueError as exc:
            print(f"Invalid clip name: {exc}")


def _retarget_latest_frame(
    provider: Pico4InputProvider,
    retargeter: RetargetingModule,
    *,
    last_seq: int | None,
) -> tuple[np.ndarray | None, int | None]:
    if not provider.has_frame():
        return None, last_seq
    frame, _timestamp_s, seq = provider.get_frame_packet()
    seq = int(seq)
    if last_seq is not None and seq == last_seq:
        return None, last_seq
    qpos = np.asarray(retargeter.retarget(frame), dtype=np.float64).reshape(-1)
    if qpos.shape[0] != FULL_QPOS_DIM:
        raise ValueError(f"retarget qpos must be {FULL_QPOS_DIM}D, got {qpos.shape[0]}")
    if not np.isfinite(qpos).all():
        raise ValueError("retarget qpos contains NaN/Inf")
    return qpos, seq


def _record_one_clip(
    *,
    cfg: DictConfig,
    state: RecordingState,
) -> str:
    record_cfg = cfg_get(cfg, "record", {}) or {}
    target_fps = int(cfg_get(record_cfg, "target_fps", 30))
    min_frames = int(cfg_get(record_cfg, "min_frames", 30))
    max_duration_s = float(cfg_get(record_cfg, "max_duration_s", 0.0))
    output_dir = _project_path(str(cfg_get(record_cfg, "output_dir", "data/pico_motion/clips")))
    if target_fps <= 0:
        raise ValueError(f"record.target_fps must be > 0, got {target_fps}")
    if min_frames < 2:
        raise ValueError(f"record.min_frames must be >= 2, got {min_frames}")

    keyboard = TerminalKeyboardReader()
    if not keyboard.active:
        keyboard.close()
        raise RuntimeError("record_pico_motion.py requires an interactive TTY for keyboard controls")

    print("Controls: R=start  S=save  D=discard  N=new name  Q=quit")
    clip_name, recording, frame_count, _elapsed = state.status()
    if clip_name is None:
        print("Preview is running. Press N to enter a clip name before recording.")
    else:
        print(f"Ready: {clip_name}")

    try:
        while True:
            for event in keyboard.poll():
                key = event.key.lower()
                if key == "q":
                    _clip_name, was_recording, _frame_count, _elapsed = state.status()
                    if was_recording:
                        discarded_name, _ = state.discard()
                        print(f"\nDiscarded unsaved clip: {discarded_name}")
                    return "quit"
                if key == "n":
                    _clip_name, was_recording, _frame_count, _elapsed = state.status()
                    if was_recording:
                        print("\nPress S to save or D to discard before changing clip name.")
                        continue
                    return "next"
                if key == "r":
                    current_name, _recording, _frame_count, _elapsed = state.status()
                    if current_name is None:
                        print("\nPress N and enter a clip name before recording.")
                        continue
                    clip_name = state.start()
                    print(f"\nRecording: {clip_name}")
                elif key == "d":
                    clip_name, frame_count = state.discard()
                    label = "<unnamed>" if clip_name is None else clip_name
                    print(f"\nDiscarded: {label} ({frame_count} frames)")
                    return "next"
                elif key == "s":
                    clip_name, _was_recording, frames = state.snapshot()
                    if clip_name is None:
                        print("\nPress N and enter a clip name before saving.")
                        continue
                    if len(frames) < min_frames:
                        print(f"\nNeed at least {min_frames} frames before saving; current={len(frames)}")
                        continue
                    output_path = unique_clip_path(output_dir, clip_name)
                    write_motion_clip_npz(output_path, frames, fps=target_fps)
                    state.mark_saved()
                    duration_s = len(frames) / float(target_fps)
                    print(f"\nSaved: {output_path} ({len(frames)} frames, {duration_s:.2f}s)")
                    return "next"

            clip_name, recording, frame_count, elapsed = state.status()
            if recording and clip_name is not None and max_duration_s > 0.0 and elapsed is not None:
                if elapsed >= max_duration_s:
                    clip_name, _was_recording, frames = state.snapshot()
                    if clip_name is None:
                        raise RuntimeError("recording reached max_duration_s without a clip name")
                    if len(frames) < min_frames:
                        raise ValueError(
                            f"max_duration_s reached but only recorded {len(frames)} frames; "
                            f"min_frames={min_frames}"
                        )
                    output_path = unique_clip_path(output_dir, clip_name)
                    write_motion_clip_npz(output_path, frames, fps=target_fps)
                    state.mark_saved()
                    print(f"\nSaved by max_duration_s: {output_path}")
                    return "next"

            time.sleep(0.02)
    finally:
        keyboard.close()


def _maybe_write_dataset_spec(cfg: DictConfig) -> None:
    record_cfg = cfg_get(cfg, "record", {}) or {}
    if not bool(cfg_get(record_cfg, "write_dataset_spec", True)):
        return
    output_dir = _project_path(str(cfg_get(record_cfg, "output_dir", "data/pico_motion/clips")))
    spec_path = _project_path(str(cfg_get(record_cfg, "dataset_spec_path", "data/pico_motion/pico_recorded.yaml")))
    dataset_name = str(cfg_get(record_cfg, "dataset_name", "pico_recorded"))
    target_fps = int(cfg_get(record_cfg, "target_fps", 30))
    overwrite = bool(cfg_get(record_cfg, "overwrite_dataset_spec", False))
    path = ensure_pico_dataset_spec(
        spec_path,
        output_dir,
        spec=PicoDatasetSpec(dataset_name=dataset_name, target_fps=target_fps),
        overwrite=overwrite,
    )
    print(f"Dataset spec: {path}")


def _configure_logging(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    record_cfg = cfg_get(cfg, "record", {}) or {}
    if bool(cfg_get(record_cfg, "quiet_logs", True)):
        logging.getLogger("pico_bridge").setLevel(logging.WARNING)
        logging.getLogger("teleopit.inputs.pico4_provider").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="../../teleopit/configs", config_name="pico4_record")
def main(cfg: DictConfig) -> None:
    _configure_logging(cfg)
    _maybe_write_dataset_spec(cfg)

    print("Starting Pico receiver; waiting for body tracking...")
    provider = _build_provider(cfg)
    preview = RetargetPreview(enabled=False)
    try:
        preview.close()
        preview = RetargetPreview(enabled=bool(cfg_get(cfg_get(cfg, "record", {}) or {}, "viewer_enabled", True)))
        retargeter = _build_retargeter(cfg, provider)
        print("Pico recorder is ready. Retarget viewer will update after the first frame.")
        record_cfg = cfg_get(cfg, "record", {}) or {}
        target_fps = int(cfg_get(record_cfg, "target_fps", 30))
        state = RecordingState()
        worker = RetargetRecordingWorker(
            provider=provider,
            retargeter=retargeter,
            preview=preview,
            state=state,
            target_fps=target_fps,
        )
        worker.start()
        try:
            clip_name = _prompt_clip_name()
            if clip_name is not None:
                state.set_clip_name(clip_name)
            while True:
                worker.raise_if_failed()
                if clip_name is None:
                    break
                command = _record_one_clip(cfg=cfg, state=state)
                worker.raise_if_failed()
                if command == "quit":
                    break
                clip_name = _prompt_clip_name()
                if clip_name is not None:
                    state.set_clip_name(clip_name)
        finally:
            worker.stop()
            worker.raise_if_failed()
    except KeyboardInterrupt:
        print("\nInterrupted; unsaved in-progress clip was discarded.")
    finally:
        preview.close()
        provider.close()


if __name__ == "__main__":
    main()
