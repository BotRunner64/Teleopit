#!/usr/bin/env python3
"""Rebuild train/val HDF5 shard directories from a filtered manifest.

Reads filtered_manifest.csv (output of export_reviewed_manifest.py),
verifies all referenced motion files exist, and rebuilds cleaned train/val HDF5 splits.

Usage:
    python scripts/data/build_dataset_from_review.py \
        --filtered_manifest data/datasets/review/twist2/filtered_manifest.csv \
        --output_dir data/datasets/builds/twist2_cleaned
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train_mimic.data.dataset_lib import (
    merge_clip_dicts_payload,
    read_motion_clip,
    utc_now_iso,
    write_hdf5_manifest,
    write_hdf5_motion_shard,
    write_json,
)
from train_mimic.data.dataset_builder import DatasetClipRow, write_manifest_resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cleaned dataset from filtered manifest"
    )
    parser.add_argument(
        "--filtered_manifest", type=str, required=True,
        help="Path to filtered_manifest.csv",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory, e.g. data/datasets/builds/twist2_cleaned",
    )
    parser.add_argument(
        "--target_fps", type=int, default=None,
        help="Resample all clips to this FPS. "
        "Required when source clips have mixed FPS values.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.filtered_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    if not manifest_path.is_file():
        print(f"ERROR: filtered manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Read filtered manifest
    # Columns: clip_id, source, file_rel, num_frames, fps, resolved_split, resolved_npz_path,
    # weight, clip_index
    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        has_clip_index = "clip_index" in (reader.fieldnames or [])
        for idx, raw in enumerate(reader, start=2):
            rows.append({
                "clip_id": raw["clip_id"].strip(),
                "source": raw["source"].strip(),
                "file_rel": raw["file_rel"].strip(),
                "num_frames": int(raw["num_frames"]),
                "fps": int(raw["fps"]),
                "resolved_split": raw["resolved_split"].strip(),
                "resolved_npz_path": raw["resolved_npz_path"].strip(),
                "weight": float(raw["weight"]),
                "clip_index": int(raw["clip_index"]) if has_clip_index else -1,
                "line_no": idx,
            })

    if not rows:
        print("ERROR: filtered manifest has no data rows", file=sys.stderr)
        sys.exit(1)

    # Verify all referenced motion files exist
    missing = []
    for row in rows:
        p = Path(row["resolved_npz_path"])
        if not p.is_file():
            # Try resolving from file_rel
            alt = Path(row["file_rel"])
            if not alt.is_absolute():
                alt = PROJECT_ROOT / alt
            if alt.is_file():
                row["resolved_npz_path"] = str(alt.resolve())
            else:
                missing.append(f"  line {row['line_no']}: {row['clip_id']} -> {row['resolved_npz_path']}")

    if missing:
        print(f"ERROR: {len(missing)} motion files not found:", file=sys.stderr)
        for m in missing[:20]:
            print(m, file=sys.stderr)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more", file=sys.stderr)
        sys.exit(1)

    # Check for mixed FPS
    fps_values = sorted(set(r["fps"] for r in rows))
    if len(fps_values) > 1:
        print(f"[INFO] Mixed FPS detected: {fps_values}. Resampling all clips to {args.target_fps} FPS.")
        if args.target_fps is None:
            print(
                "ERROR: clips have mixed FPS values but --target_fps is not set.\n"
                "Please specify --target_fps (e.g. --target_fps 30).",
                file=sys.stderr,
            )
            sys.exit(1)

    # Split into train / val
    train_rows = [r for r in rows if r["resolved_split"] == "train"]
    val_rows = [r for r in rows if r["resolved_split"] == "val"]

    if not train_rows:
        print("ERROR: no train clips in filtered manifest", file=sys.stderr)
        sys.exit(1)
    if not val_rows:
        print("WARNING: no val clips in filtered manifest", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_output_rows: list[DatasetClipRow] = []

    def _merge_split(split_rows: list[dict], split_name: str) -> dict | None:
        if not split_rows:
            return None
        print(f"Merging {len(split_rows)} {split_name} clips...")
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        out = split_dir / "shard_000.h5"

        clip_dicts = [
            read_motion_clip(Path(r["resolved_npz_path"]), int(r["clip_index"]))
            for r in split_rows
        ]
        weights_list = [r["weight"] for r in split_rows]
        payload = merge_clip_dicts_payload(
            clip_dicts,
            target_fps=args.target_fps,
            weights=weights_list,
        )
        h5_info = write_hdf5_motion_shard(payload, out)
        write_hdf5_manifest(
            split_dir,
            shard_infos=[h5_info],
            fps=int(payload["fps"]),
            body_names=np.asarray(payload["body_names"]),
        )

        source_lengths = np.asarray(payload["clip_lengths"], dtype=np.int64)
        for clip_index, (r, num_frames) in enumerate(zip(split_rows, source_lengths)):
            all_output_rows.append(DatasetClipRow(
                clip_id=r["clip_id"],
                source=r["source"],
                file_rel=r["file_rel"],
                num_frames=int(num_frames),
                fps=int(payload["fps"]),
                resolved_split=split_name,
                resolved_npz_path=str(out),
                weight=float(r["weight"]),
                clip_index=clip_index,
            ))

        total_frames = int(np.asarray(payload["joint_pos"]).shape[0])
        stats = {
            "output": str(split_dir),
            "shards": 1,
            "clips": int(h5_info["clips"]),
            "num_clips": int(h5_info["clips"]),
            "source_clips": int(h5_info["source_clips"]),
            "frames": total_frames,
            "fps": int(payload["fps"]),
            "duration_s": float(total_frames / max(int(payload["fps"]), 1)),
        }
        print(f"  {split_name}/: {stats['frames']} frames, {stats['duration_s'] / 60:.1f} min")
        return stats

    train_stats = _merge_split(train_rows, "train")
    val_stats = _merge_split(val_rows, "val")

    resolved_manifest = write_manifest_resolved(all_output_rows, output_dir)

    # Write build info
    report = {
        "built_at_utc": utc_now_iso(),
        "source_manifest": str(manifest_path),
        "manifest_resolved": str(resolved_manifest),
        "output_dir": str(output_dir),
        "target_fps": args.target_fps,
        "source_rows": [asdict(row) for row in all_output_rows],
        "clip_counts": {
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "splits": {
            "train": train_stats,
            "val": val_stats,
        },
    }
    write_json(output_dir / "build_info.json", report)

    print(f"\nCleaned dataset built at: {output_dir}")
    print(f"  Total clips: {len(rows)} (train={len(train_rows)}, val={len(val_rows)})")


if __name__ == "__main__":
    main()
