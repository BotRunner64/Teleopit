#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from train_mimic.data.dataset_lib import compute_dataset_stats


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d} ({seconds / 3600.0:.2f} h)"
    return f"{minutes:d}:{secs:02d} ({seconds:.1f} s)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a Teleopit motion dataset root.")
    parser.add_argument("dataset", type=str, help="Dataset root directory or a single .h5 shard")
    parser.add_argument("--json", action="store_true", help="Print full JSON statistics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = compute_dataset_stats(args.dataset)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0

    print(f"root: {stats['root']}")
    print(f"shards: {stats['shards']}")
    print(f"windows: {stats['windows']}")
    print(f"source_clips: {stats['source_clips']}")
    print(f"frames: {stats['frames']}")
    print(f"duration: {_format_duration(float(stats['duration_s']))}")
    print(f"fps: {stats['fps']}")
    print(f"bodies: {len(stats['body_names'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
