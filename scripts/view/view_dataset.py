#!/usr/bin/env python3
"""Read-only web viewer for Teleopit HDF5 motion datasets."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import mujoco
import numpy as np
import viser

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mjlab.viewer.viser import ViserMujocoScene

from teleopit.runtime.assets import UNITREE_G1_XML, missing_gmr_assets_message
from train_mimic.data.dataset_lib import find_motion_shards, read_motion_clip

DEFAULT_XML = UNITREE_G1_XML


@dataclass(frozen=True)
class DatasetClip:
    clip_id: str
    shard_path: Path
    clip_index: int
    num_frames: int
    fps: int

    @property
    def duration_s(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0


def discover_dataset_clips(dataset_path: Path) -> list[DatasetClip]:
    clips: list[DatasetClip] = []
    for shard_path in find_motion_shards(dataset_path):
        with h5py.File(shard_path, "r") as h5:
            required = ["source_clip_lengths", "source_clip_fps"]
            missing = [key for key in required if key not in h5]
            if missing:
                raise ValueError(
                    f"HDF5 shard {shard_path} missing source clip metadata {missing}. "
                    "Rebuild the dataset with the current HDF5 writer."
                )
            lengths = np.asarray(h5["source_clip_lengths"], dtype=np.int64)
            fps = np.asarray(h5["source_clip_fps"], dtype=np.int64)

        if dataset_path.is_file():
            shard_rel = Path(shard_path.name)
        else:
            try:
                shard_rel = shard_path.relative_to(dataset_path)
            except ValueError:
                shard_rel = Path(shard_path.name)
        for clip_index, (num_frames, clip_fps) in enumerate(zip(lengths, fps)):
            clips.append(
                DatasetClip(
                    clip_id=f"{shard_rel.as_posix()}#{clip_index}",
                    shard_path=shard_path,
                    clip_index=int(clip_index),
                    num_frames=int(num_frames),
                    fps=int(clip_fps),
                )
            )
    if not clips:
        raise ValueError(f"dataset has no source clips: {dataset_path}")
    return clips


class ClipPlayer:
    """Loads one source clip and drives MuJoCo qpos frame by frame."""

    def __init__(self, mj_model: mujoco.MjModel) -> None:
        self.model = mj_model
        self.data = mujoco.MjData(mj_model)
        self._joint_pos: np.ndarray | None = None
        self._pelvis_pos: np.ndarray | None = None
        self._pelvis_quat: np.ndarray | None = None
        self._fps: int = 30
        self._num_frames: int = 0

    def load_clip(self, clip: DatasetClip) -> None:
        d = read_motion_clip(clip.shard_path, clip.clip_index)
        self._joint_pos = np.asarray(d["joint_pos"])
        body_pos_w = np.asarray(d["body_pos_w"])
        body_quat_w = np.asarray(d["body_quat_w"])
        self._pelvis_pos = body_pos_w[:, 0, :]
        self._pelvis_quat = body_quat_w[:, 0, :]
        self._fps = int(d["fps"])
        self._num_frames = int(self._joint_pos.shape[0])

    def set_frame(self, frame_idx: int) -> None:
        if self._joint_pos is None or self._pelvis_pos is None or self._pelvis_quat is None:
            return
        idx = max(0, min(frame_idx, self._num_frames - 1))
        self.data.qpos[:3] = self._pelvis_pos[idx]
        self.data.qpos[3:7] = self._pelvis_quat[idx]
        self.data.qpos[7:] = self._joint_pos[idx]
        mujoco.mj_forward(self.model, self.data)

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def num_frames(self) -> int:
        return self._num_frames


class DatasetSession:
    def __init__(self, clips: list[DatasetClip], sort_mode: str) -> None:
        self._clips = clips
        self._order = list(range(len(clips)))
        if sort_mode == "duration_desc":
            self._order.sort(key=lambda i: -clips[i].duration_s)
        elif sort_mode == "shard":
            self._order.sort(key=lambda i: clips[i].clip_id)
        self._cursor = 0

    @property
    def total(self) -> int:
        return len(self._clips)

    @property
    def cursor_display(self) -> int:
        return self._cursor + 1

    @property
    def total_duration_s(self) -> float:
        return sum(clip.duration_s for clip in self._clips)

    def current_clip(self) -> DatasetClip:
        return self._clips[self._order[self._cursor]]

    def go_next(self) -> None:
        if self._cursor < len(self._order) - 1:
            self._cursor += 1

    def go_prev(self) -> None:
        if self._cursor > 0:
            self._cursor -= 1

    def jump_to(self, position: int) -> None:
        self._cursor = max(0, min(position - 1, len(self._order) - 1))


class DatasetViewerApp:
    def __init__(
        self,
        *,
        dataset_path: Path,
        xml_path: Path,
        port: int,
        sort_mode: str,
    ) -> None:
        self._dataset_path = dataset_path
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._player = ClipPlayer(self._model)
        self._session = DatasetSession(discover_dataset_clips(dataset_path), sort_mode)

        self._server = viser.ViserServer(port=port, label="Dataset Viewer")
        self._scene = ViserMujocoScene(
            server=self._server,
            mj_model=self._model,
            num_envs=1,
        )
        mujoco.mj_forward(self._model, self._player.data)
        self._scene.update_from_mjdata(self._player.data)

        self._playing = False
        self._speed = 1.0
        self._current_frame = 0
        self._pending_actions: list[str] = []
        self._pending_frame_scrub: int | None = None
        self._pending_jump: int | None = None
        self._hotkey_clearing = False

    def setup_gui(self) -> None:
        gui = self._server.gui

        with gui.add_folder("Clip", order=0):
            self._info_html = gui.add_html("<em>Loading...</em>")

        with gui.add_folder("Playback", order=1):
            self._play_btn = gui.add_button("Play", color="green")
            self._frame_slider = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
            self._speed_group = gui.add_button_group(
                "Speed",
                options=["0.25x", "0.5x", "1x", "2x"],
            )
            self._restart_btn = gui.add_button("Restart")

            @self._play_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("toggle_play")

            @self._frame_slider.on_update
            def _(_) -> None:
                self._pending_frame_scrub = int(self._frame_slider.value)

            @self._speed_group.on_click
            def _(event) -> None:
                self._speed = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}.get(
                    event.target.value,
                    1.0,
                )

            @self._restart_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("restart")

        with gui.add_folder("Navigation", order=2):
            self._prev_btn = gui.add_button("Prev")
            self._next_btn = gui.add_button("Next")
            self._jump_input = gui.add_number(
                "Jump to #",
                initial_value=1,
                min=1,
                max=self._session.total,
                step=1,
            )

            @self._prev_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("prev")

            @self._next_btn.on_click
            def _(_) -> None:
                self._pending_actions.append("next")

            @self._jump_input.on_update
            def _(_) -> None:
                self._pending_jump = int(self._jump_input.value)

        with gui.add_folder("Dataset", order=3, expand_by_default=False):
            self._stats_html = gui.add_html("<em>Loading...</em>")

        with gui.add_folder("Shortcuts", order=4):
            self._hotkey_input = gui.add_text("Hotkey (click here, type key)", initial_value="")
            gui.add_markdown("**N**=Next  **P**=Prev  **Space**=Play/Pause  **R**=Restart  **F**=Speed Up")

            @self._hotkey_input.on_update
            def _(_) -> None:
                if self._hotkey_clearing:
                    return
                raw = self._hotkey_input.value
                self._hotkey_clearing = True
                self._hotkey_input.value = ""
                self._hotkey_clearing = False
                if not raw:
                    return
                action = {
                    "n": "next",
                    "p": "prev",
                    " ": "toggle_play",
                    "r": "restart",
                    "f": "speed_up",
                }.get(raw[-1].lower())
                if action:
                    self._pending_actions.append(action)

        self._scene.create_visualization_gui()

    def _load_current_clip(self) -> None:
        clip = self._session.current_clip()
        try:
            self._player.load_clip(clip)
        except Exception as exc:
            self._info_html.content = f"<strong style='color:red'>Error loading clip:</strong><br/>{exc}"
            return

        self._current_frame = 0
        self._playing = False
        self._player.set_frame(0)
        self._scene.update_from_mjdata(self._player.data)

        self._play_btn.label = "Play"
        self._play_btn.color = "green"
        self._frame_slider.max = max(0, self._player.num_frames - 1)
        self._frame_slider.value = 0
        self._jump_input.value = self._session.cursor_display

        self._pending_frame_scrub = None
        self._pending_jump = None
        self._pending_actions.clear()
        self._update_info_display()
        self._update_stats_display()

    def _update_frame(self) -> None:
        self._player.set_frame(self._current_frame)
        self._scene.update_from_mjdata(self._player.data)

    def _update_info_display(self) -> None:
        clip = self._session.current_clip()
        self._info_html.content = f"""
        <div style="font-size: 0.85em; line-height: 1.4; padding: 0.5em;">
          <strong>#{self._session.cursor_display}/{self._session.total}</strong><br/>
          <strong>clip:</strong> {clip.clip_id}<br/>
          <strong>shard:</strong> {clip.shard_path}<br/>
          <strong>source clip index:</strong> {clip.clip_index}<br/>
          <strong>frames:</strong> {clip.num_frames} | <strong>fps:</strong> {clip.fps}
          | <strong>duration:</strong> {clip.duration_s:.2f}s
        </div>
        """

    def _update_stats_display(self) -> None:
        total_min = self._session.total_duration_s / 60.0
        self._stats_html.content = f"""
        <div style="font-size: 0.85em; line-height: 1.4; padding: 0.5em;">
          <strong>root:</strong> {self._dataset_path}<br/>
          <strong>source clips:</strong> {self._session.total}<br/>
          <strong>duration:</strong> {total_min:.1f} min
        </div>
        """

    def _process_pending(self) -> None:
        scrub = self._pending_frame_scrub
        if scrub is not None and not self._playing:
            self._pending_frame_scrub = None
            self._current_frame = scrub
            self._update_frame()

        jump = self._pending_jump
        if jump is not None:
            self._pending_jump = None
            self._session.jump_to(jump)
            self._load_current_clip()
            return

        while self._pending_actions:
            action = self._pending_actions.pop(0)
            if action == "toggle_play":
                self._playing = not self._playing
                self._play_btn.label = "Pause" if self._playing else "Play"
                self._play_btn.color = "red" if self._playing else "green"
            elif action == "restart":
                self._current_frame = 0
                self._update_frame()
                self._frame_slider.value = 0
                self._pending_frame_scrub = None
            elif action == "prev":
                self._session.go_prev()
                self._load_current_clip()
                return
            elif action == "next":
                self._session.go_next()
                self._load_current_clip()
                return
            elif action == "speed_up":
                levels = [1.0, 2.0, 4.0, 8.0]
                self._speed = levels[(levels.index(self._speed) + 1) % len(levels)] if self._speed in levels else 1.0
                label = f"{self._speed:g}x"
                if label in ("0.25x", "0.5x", "1x", "2x"):
                    self._speed_group.value = label

    def run(self) -> None:
        self.setup_gui()
        self._load_current_clip()

        print(f"\nDataset viewer ready at http://localhost:{self._server.get_port()}")
        print("Press Ctrl+C to exit.\n")

        last_frame_time = time.time()
        try:
            while True:
                now = time.time()
                if self._scene.needs_update:
                    self._scene.refresh_visualization()

                self._process_pending()

                if self._playing and self._player.num_frames > 0:
                    dt = now - last_frame_time
                    frames_to_advance = dt * self._player.fps * self._speed
                    if frames_to_advance >= 1.0:
                        self._current_frame += int(frames_to_advance)
                        if self._current_frame >= self._player.num_frames:
                            self._current_frame = 0
                            self._playing = False
                            self._play_btn.label = "Play"
                            self._play_btn.color = "green"
                        self._update_frame()
                        self._frame_slider.value = self._current_frame
                        self._pending_frame_scrub = None
                        last_frame_time = now
                else:
                    last_frame_time = now

                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self._server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only web viewer for Teleopit HDF5 datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset root directory or a single Teleopit .h5 shard",
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="Robot XML path",
    )
    parser.add_argument("--port", type=int, default=8012, help="Viser server port")
    parser.add_argument(
        "--sort",
        type=str,
        default="shard",
        choices=["shard", "duration_desc"],
        help="Initial clip sort order",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        print(f"ERROR: dataset path not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    xml_path = Path(args.xml) if args.xml else DEFAULT_XML
    if not xml_path.is_file():
        print(
            "ERROR: " + missing_gmr_assets_message(xml_path, label="Robot XML"),
            file=sys.stderr,
        )
        sys.exit(1)

    app = DatasetViewerApp(
        dataset_path=dataset_path,
        xml_path=xml_path,
        port=args.port,
        sort_mode=args.sort,
    )
    app.run()


if __name__ == "__main__":
    main()
