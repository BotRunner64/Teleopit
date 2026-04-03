"""Benchmark DDS LowState reception rate and jitter.

Run on both PC (wired) and NX (onboard) to compare.

Usage:
    python scripts/bench_dds.py                          # default eth0
    python scripts/bench_dds.py --interface enp130s0     # wired PC
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import g1_bridge_sdk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to measure")
    args = parser.parse_args()

    print(f"Interface: {args.interface}")
    print(f"Duration:  {args.duration}s")
    print()

    bridge = g1_bridge_sdk.G1Bridge(args.interface)

    print("Waiting for first LowState...")
    if not bridge.wait_for_state(5.0):
        print("ERROR: No LowState received within 5s. Check interface name.")
        return

    print("First LowState received")
    print(f"Collecting data for {args.duration}s...")
    print()

    # Poll a monotonically increasing LowState receive counter so we measure
    # actual message arrivals instead of qpos changes.
    timestamps: list[float] = []
    poll_interval = 0.0005  # 0.5ms poll
    end_time = time.monotonic() + args.duration
    prev_count = int(bridge.get_state_counter())
    prev_poll_time = time.monotonic()
    skipped_batches = 0

    while time.monotonic() < end_time:
        now = time.monotonic()
        count = int(bridge.get_state_counter())
        if count > prev_count:
            delta = count - prev_count
            if delta > 1:
                skipped_batches += 1
            interval = (now - prev_poll_time) / delta
            timestamps.extend(prev_poll_time + interval * (idx + 1) for idx in range(delta))
            prev_count = count
        prev_poll_time = now
        time.sleep(poll_interval)

    if len(timestamps) < 2:
        print(f"ERROR: Only {len(timestamps)} LowState messages detected.")
        return

    # Analyze
    ts = np.array(timestamps)
    intervals = np.diff(ts) * 1000  # ms

    total_msgs = len(ts)
    total_time = ts[-1] - ts[0]
    avg_hz = (total_msgs - 1) / total_time if total_time > 0 else 0

    print("=" * 60)
    print("DDS LowState Reception Statistics")
    print("=" * 60)
    print(f"  LowState messages:      {total_msgs}")
    print(f"  Duration:               {total_time:.3f}s")
    print(f"  Average rate:           {avg_hz:.1f} Hz")
    print(f"  Batched polls (>1 msg): {skipped_batches}")
    print()
    print(f"  Interval (ms):")
    print(f"    mean:   {np.mean(intervals):.2f}")
    print(f"    std:    {np.std(intervals):.2f}")
    print(f"    min:    {np.min(intervals):.2f}")
    print(f"    max:    {np.max(intervals):.2f}")
    print(f"    p50:    {np.percentile(intervals, 50):.2f}")
    print(f"    p95:    {np.percentile(intervals, 95):.2f}")
    print(f"    p99:    {np.percentile(intervals, 99):.2f}")
    print()

    # Check for gaps (> 10ms = missed at least one message at 500Hz)
    gaps = intervals[intervals > 10]
    print(f"  Gaps > 10ms:  {len(gaps)} / {len(intervals)} ({100*len(gaps)/len(intervals):.1f}%)")
    gaps_20 = intervals[intervals > 20]
    print(f"  Gaps > 20ms:  {len(gaps_20)} / {len(intervals)} ({100*len(gaps_20)/len(intervals):.1f}%)")
    print("=" * 60)

    # Verdict
    if avg_hz < 100:
        print("\nWARNING: LowState rate very low. DDS communication may be degraded.")
    elif avg_hz < 400:
        print(f"\nNOTE: LowState at {avg_hz:.0f}Hz (expected ~500Hz). Acceptable but not ideal.")
    else:
        print(f"\nOK: LowState at {avg_hz:.0f}Hz.")


if __name__ == "__main__":
    main()
