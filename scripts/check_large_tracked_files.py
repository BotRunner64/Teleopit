#!/usr/bin/env python3
"""Fail when heavyweight binaries are committed to Git."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_BLOCKED_EXTENSIONS = (
    ".gif",
    ".mp4",
    ".obj",
    ".dae",
    ".stl",
    ".npz",
    ".pt",
    ".pth",
    ".onnx",
    ".h5",
    ".pkl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject heavyweight tracked files before they land in Git history."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root to scan (default: project root).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum allowed size for any tracked file.",
    )
    parser.add_argument(
        "--blocked-ext",
        nargs="*",
        default=list(DEFAULT_BLOCKED_EXTENSIONS),
        help="Extensions that must never be tracked, regardless of size.",
    )
    return parser.parse_args()


def tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=False,
    )
    return [
        repo_root / Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    blocked_ext = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in args.blocked_ext
    }

    blocked: list[str] = []
    oversized: list[str] = []
    for path in tracked_files(repo_root):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root)
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if suffix in blocked_ext:
            blocked.append(f"{rel} ({size} bytes)")
        elif size > args.max_bytes:
            oversized.append(f"{rel} ({size} bytes)")

    if not blocked and not oversized:
        print("OK: no blocked or oversized tracked files found.")
        return 0

    print("Repository hygiene check failed.", file=sys.stderr)
    if blocked:
        print("\nBlocked file types must stay outside Git history:", file=sys.stderr)
        for entry in blocked:
            print(f"  - {entry}", file=sys.stderr)
    if oversized:
        print(
            f"\nTracked files larger than {args.max_bytes} bytes:",
            file=sys.stderr,
        )
        for entry in oversized:
            print(f"  - {entry}", file=sys.stderr)
    print(
        "\nMove large assets to external downloads or Releases instead of committing them.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
