#!/usr/bin/env python3
"""Play or export a standardized Teleopit motion NPZ clip."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import mujoco as mj
import numpy as np

from teleopit.retargeting.gmr.params import (
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a standardized Teleopit motion NPZ clip in MuJoCo.",
    )
    parser.add_argument("--npz", type=str, required=True, help="Path to a single motion NPZ clip")
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="Robot viewer type. For lafan1_v1 clips this should usually stay as unitree_g1.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--end", type=int, default=-1, help="Exclusive end frame index; -1 means full clip")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame")
    parser.add_argument("--loop", action="store_true", help="Loop the clip")
    parser.add_argument("--record_video", action="store_true", help="Export an mp4 while rendering")
    parser.add_argument("--video_path", type=str, default=None, help="Output mp4 path")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render offscreen to mp4 without opening a GLFW window",
    )
    parser.add_argument("--width", type=int, default=1280, help="Output width for --headless video")
    parser.add_argument("--height", type=int, default=720, help="Output height for --headless video")
    parser.add_argument(
        "--show_body_frames",
        action="store_true",
        help="Overlay FK body frames stored in the NPZ clip",
    )
    parser.add_argument(
        "--show_body_names",
        action="store_true",
        help="Show body names when --show_body_frames is enabled",
    )
    parser.add_argument(
        "--no_rate_limit",
        action="store_true",
        help="Render as fast as possible instead of the clip FPS",
    )
    return parser.parse_args()


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return q / norm


def load_clip(path: Path) -> dict[str, np.ndarray | int]:
    data = np.load(path, allow_pickle=True)
    required = ["fps", "joint_pos", "body_pos_w", "body_quat_w", "body_names"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"NPZ missing required keys: {missing}")

    body_names = np.asarray(data["body_names"]).astype(str)
    if "pelvis" not in body_names:
        raise ValueError("NPZ body_names must contain 'pelvis' as the root body")
    pelvis_idx = int(np.where(body_names == "pelvis")[0][0])

    fps = int(np.asarray(data["fps"]).item())
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_quat_w = _normalize_quaternion(np.asarray(data["body_quat_w"], dtype=np.float32))

    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos must be rank-2, got {joint_pos.shape}")
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"body_pos_w must be (T, nb, 3), got {body_pos_w.shape}")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"body_quat_w must be (T, nb, 4), got {body_quat_w.shape}")
    if not (joint_pos.shape[0] == body_pos_w.shape[0] == body_quat_w.shape[0]):
        raise ValueError("joint_pos, body_pos_w, and body_quat_w must share the same frame count")

    return {
        "fps": fps,
        "joint_pos": joint_pos,
        "root_pos": body_pos_w[:, pelvis_idx, :],
        "root_quat_wxyz": body_quat_w[:, pelvis_idx, :],
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_names": body_names,
    }


def build_overlay_frame(
    body_names: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    frame_idx: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        str(name): (body_pos_w[frame_idx, i], body_quat_w[frame_idx, i])
        for i, name in enumerate(body_names)
    }


def render_headless_video(
    *,
    robot: str,
    frame_indices: np.ndarray,
    render_fps: int,
    video_path: str,
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    width: int,
    height: int,
) -> None:
    xml_path = ROBOT_XML_DICT[robot]
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    try:
        renderer = mj.Renderer(model, height=height, width=width)
    except Exception as exc:
        raise RuntimeError(
            "Failed to create a MuJoCo offscreen renderer. "
            "This environment likely lacks a usable headless OpenGL backend. "
            "Try setting MUJOCO_GL=egl or MUJOCO_GL=osmesa on a machine with those backends available."
        ) from exc
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.distance = VIEWER_CAM_DISTANCE_DICT[robot]
    cam.elevation = -10
    base_body_id = model.body(ROBOT_BASE_DICT[robot]).id

    video_file = Path(video_path).expanduser().resolve()
    video_file.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(video_file, fps=render_fps) as writer:
        for frame_idx in frame_indices:
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[:3] = root_pos[frame_idx]
            data.qpos[3:7] = root_quat_wxyz[frame_idx]
            data.qpos[7:] = joint_pos[frame_idx]
            mj.mj_forward(model, data)
            cam.lookat[:] = data.xpos[base_body_id]
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

    renderer.close()


def main() -> int:
    args = parse_args()
    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ clip not found: {npz_path}")
    if args.stride <= 0:
        raise ValueError(f"--stride must be > 0, got {args.stride}")

    clip = load_clip(npz_path)
    total_frames = int(clip["joint_pos"].shape[0])  # type: ignore[index]
    start = max(0, args.start)
    end = total_frames if args.end < 0 else min(total_frames, args.end)
    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}, total={total_frames}")

    frame_indices = np.arange(start, end, args.stride, dtype=np.int64)
    render_fps = max(1, int(round(int(clip["fps"]) / args.stride)))
    video_path = args.video_path
    if args.record_video and video_path is None:
        video_path = str(npz_path.with_suffix(".mp4"))
    if args.headless and not args.record_video:
        raise ValueError("--headless requires --record_video so there is an output target")
    if args.headless and (args.width <= 0 or args.height <= 0):
        raise ValueError("--width and --height must be > 0")

    joint_pos = clip["joint_pos"]
    root_pos = clip["root_pos"]
    root_quat_wxyz = clip["root_quat_wxyz"]
    body_pos_w = clip["body_pos_w"]
    body_quat_w = clip["body_quat_w"]
    body_names = clip["body_names"]

    print(f"Rendering {npz_path}")
    print(f"frames={len(frame_indices)} / {total_frames}, fps={render_fps}, robot={args.robot}")
    if args.record_video:
        print(f"video={Path(video_path).expanduser().resolve()}")

    if args.headless:
        if args.show_body_frames or args.show_body_names:
            raise ValueError("--show_body_frames/--show_body_names are only supported in interactive mode")
        if args.loop:
            raise ValueError("--loop is not supported together with --headless")
        render_headless_video(
            robot=args.robot,
            frame_indices=frame_indices,
            render_fps=render_fps,
            video_path=video_path,
            root_pos=root_pos,
            root_quat_wxyz=root_quat_wxyz,
            joint_pos=joint_pos,
            width=args.width,
            height=args.height,
        )
        return 0

    if "DISPLAY" not in os.environ:
        raise RuntimeError("DISPLAY is missing. Use --headless --record_video to export an mp4 offscreen.")

    from teleopit.retargeting.gmr.robot_motion_viewer import RobotMotionViewer

    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=render_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=video_path,
    )

    try:
        while True:
            for frame_idx in frame_indices:
                overlay = None
                if args.show_body_frames:
                    overlay = build_overlay_frame(body_names, body_pos_w, body_quat_w, frame_idx)
                viewer.step(
                    root_pos=root_pos[frame_idx],
                    root_rot=root_quat_wxyz[frame_idx],
                    dof_pos=joint_pos[frame_idx],
                    human_motion_data=overlay,
                    show_human_body_name=args.show_body_names,
                    rate_limit=not args.no_rate_limit,
                    follow_camera=True,
                )
            if not args.loop:
                break
    finally:
        viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
