#!/usr/bin/env python3
"""Send BVH motion data over UDP for testing the realtime sim2sim pipeline.

Usage:
    python scripts/send_bvh_udp.py --bvh data/sample_bvh/hc_mocap/walk.bvh
"""
from __future__ import annotations

import argparse
import socket
import time


def _read_motion_lines(bvh_path: str) -> tuple[list[str], float]:
    """Read the MOTION section of a BVH file. Returns (lines, frame_time)."""
    lines: list[str] = []
    frame_time = 1.0 / 30.0
    in_motion = False

    with open(bvh_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "MOTION":
                in_motion = True
                continue
            if not in_motion:
                continue
            if stripped.startswith("Frames:"):
                continue
            if stripped.startswith("Frame Time:"):
                frame_time = float(stripped.split(":")[1].strip())
                continue
            if stripped:
                lines.append(stripped)

    return lines, frame_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Send BVH frames over UDP")
    parser.add_argument("--bvh", required=True, help="Path to BVH file")
    parser.add_argument("--host", default="127.0.0.1", help="Target host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=1118, help="Target port (default: 1118)")
    parser.add_argument("--fps", type=float, default=0, help="Override FPS (0 = use BVH frame time)")
    parser.add_argument("--downsample", type=int, default=1, help="Send every Nth frame")
    args = parser.parse_args()

    motion_lines, frame_time = _read_motion_lines(args.bvh)
    if not motion_lines:
        print(f"No motion data found in {args.bvh}")
        return

    if args.fps > 0:
        frame_time = 1.0 / args.fps

    send_interval = frame_time * args.downsample

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    print(f"Sending {len(motion_lines)} frames to {args.host}:{args.port}")
    print(f"Frame time: {frame_time:.4f}s ({1.0/frame_time:.1f} fps), downsample: {args.downsample}x")

    try:
        for i in range(0, len(motion_lines), args.downsample):
            t0 = time.monotonic()
            sock.sendto(motion_lines[i].encode("utf-8"), addr)
            elapsed = time.monotonic() - t0
            sleep_time = send_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("Done — all frames sent.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
