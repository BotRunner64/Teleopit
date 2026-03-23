#!/usr/bin/env python3
"""Sample a balanced subset from SEED metadata for dataset building.

Produces a filtered metadata CSV that can be used as ``metadata_csv`` in a
dataset YAML spec.  The existing ``build_dataset.py`` pipeline consumes this
CSV unchanged – no pipeline modifications required.

Example
-------
    python train_mimic/scripts/data/sample_seed_dataset.py --hours 3.0
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

SEED_FPS = 120  # raw SEED capture rate
TARGET_FPS = 30
MIN_FRAMES_AT_TARGET = 22  # preprocess.min_frames in YAML spec
MIN_FRAMES_AT_SOURCE = MIN_FRAMES_AT_TARGET * (SEED_FPS // TARGET_FPS)  # 88

DEFAULT_INPUT = "data/SEED/seed_metadata_v003.csv"
DEFAULT_OUTPUT = "train_mimic/data/seed/seed_metadata_v003_3h.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample balanced SEED subset by category.")
    p.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Source metadata CSV")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output filtered CSV")
    p.add_argument("--hours", type=float, default=3.0, help="Target total duration in hours")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def _duration_s(row: dict) -> float:
    return int(row["move_duration_frames"]) / SEED_FPS


def main() -> int:
    args = parse_args()
    total_budget = args.hours * 3600.0  # seconds

    # ── 1. Read & filter ──────────────────────────────────────────────
    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    rows = [
        r
        for r in all_rows
        if r["is_mirror"] == "False"
        and int(r["move_duration_frames"]) >= MIN_FRAMES_AT_SOURCE
    ]
    print(f"[INFO] Loaded {len(all_rows)} rows, {len(rows)} after is_mirror=False & min_frames filter")

    # ── 2. Group by category ──────────────────────────────────────────
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    categories = sorted(by_cat.keys())
    num_cats = len(categories)
    cat_durations = {cat: sum(_duration_s(r) for r in clips) for cat, clips in by_cat.items()}

    print(f"[INFO] {num_cats} categories, total {sum(cat_durations.values()) / 3600:.1f}h available")

    # ── 3. Uniform-with-redistribution budget ─────────────────────────
    base_quota = total_budget / num_cats
    budget: dict[str, float] = {}
    small_cats: set[str] = set()

    for cat in categories:
        if cat_durations[cat] <= base_quota:
            budget[cat] = cat_durations[cat]
            small_cats.add(cat)

    remaining = total_budget - sum(budget.values())
    large_cats = [c for c in categories if c not in small_cats]
    if large_cats:
        quota_large = remaining / len(large_cats)
        for cat in large_cats:
            budget[cat] = quota_large

    # ── 4. Sample clips per category ──────────────────────────────────
    rng = random.Random(args.seed)
    selected: list[dict] = []

    print()
    print(f"{'Category':<30} {'Clips':>8} {'Sel':>6} {'Avail(s)':>10} {'Budget(s)':>10} {'Sampled(s)':>10}")
    print("-" * 80)

    for cat in categories:
        clips = by_cat[cat]
        cat_budget = budget[cat]

        if cat in small_cats:
            chosen = clips
        else:
            shuffled = list(clips)
            rng.shuffle(shuffled)
            chosen = []
            acc = 0.0
            for clip in shuffled:
                if acc >= cat_budget:
                    break
                chosen.append(clip)
                acc += _duration_s(clip)

        sel_dur = sum(_duration_s(c) for c in chosen)
        print(
            f"{cat:<30} {len(clips):>8} {len(chosen):>6} "
            f"{cat_durations[cat]:>10.1f} {cat_budget:>10.1f} {sel_dur:>10.1f}"
        )
        selected.extend(chosen)

    total_sel_dur = sum(_duration_s(r) for r in selected)
    print("-" * 80)
    print(f"{'TOTAL':<30} {len(rows):>8} {len(selected):>6} {'':>10} {total_budget:>10.1f} {total_sel_dur:>10.1f}")
    print(f"\n[INFO] Selected {len(selected)} clips, {total_sel_dur / 3600:.2f}h")

    # ── 5. Write output CSV ───────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"[DONE] Wrote {len(selected)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
