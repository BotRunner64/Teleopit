#!/usr/bin/env python3
"""Precompute derived FK and velocity arrays for minimal Teleopit HDF5 shards."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from train_mimic.data.dataset_lib import (
    find_motion_shards,
    write_precomputed_motion_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute FK and velocity arrays for a minimal Teleopit HDF5 dataset."
    )
    parser.add_argument("dataset", type=str, help="Dataset root directory or a single .h5 shard")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output training dataset directory. Defaults to <dataset>_precomputed.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing precomputed shards")
    parser.add_argument("--jobs", type=int, default=1, help="Number of shard-level worker processes")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional MuJoCo XML override for FK precompute",
    )
    parser.add_argument("--json", action="store_true", help="Print final report JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError(f"--jobs must be positive, got {args.jobs}")

    shard_paths = find_motion_shards(args.dataset)
    source_root = Path(args.dataset).expanduser().resolve()
    if source_root.is_file():
        default_outdir = source_root.parent.with_name(f"{source_root.parent.name}_precomputed")
    else:
        default_outdir = source_root.with_name(f"{source_root.name}_precomputed")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir is not None else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    def output_path_for(source_path: Path) -> Path:
        rel = Path(source_path.name) if source_root.is_file() else source_path.relative_to(source_root)
        return outdir / rel

    results: list[dict[str, Any]] = []

    if args.jobs == 1:
        for shard_path in shard_paths:
            result = write_precomputed_motion_shard(
                shard_path,
                output_path_for(shard_path),
                force=args.force,
                model_path=args.model_path,
            )
            results.append(result)
            print(f"[{result['status'].upper()}] {result['output']}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    write_precomputed_motion_shard,
                    shard_path,
                    output_path_for(shard_path),
                    force=args.force,
                    model_path=args.model_path,
                ): shard_path
                for shard_path in shard_paths
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{result['status'].upper()}] {result['output']}")

    report = {
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "outdir": str(outdir),
        "shards": len(shard_paths),
        "written": sum(1 for item in results if item["status"] == "written"),
        "existing": sum(1 for item in results if item["status"] == "existing"),
        "results": sorted(results, key=lambda item: item["shard"]),
    }
    print(
        f"[DONE] shards={report['shards']} written={report['written']} "
        f"existing={report['existing']}"
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
