"""Benchmark DDS LowState reception rate and jitter.

Run on both PC (wired) and NX (onboard) to compare.

Usage:
    python scripts/bench_dds.py                          # default eth0
    python scripts/bench_dds.py --interface enp130s0     # wired PC
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add unitree SDK submodule
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from teleopit.runtime.cli import add_unitree_sdk_submodule
add_unitree_sdk_submodule(Path(__file__).resolve().parent.parent)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HG_LowState

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to measure")
    args = parser.parse_args()

    print(f"Interface: {args.interface}")
    print(f"Duration:  {args.duration}s")
    print()

    ChannelFactoryInitialize(0, args.interface)

    timestamps: list[float] = []
    mode_machines: list[int] = []

    def on_lowstate(msg):
        timestamps.append(time.monotonic())
        mode_machines.append(msg.mode_machine)

    sub = ChannelSubscriber("rt/lowstate", HG_LowState)
    sub.Init(on_lowstate, 10)

    # Wait for first message
    print("Waiting for first LowState...")
    deadline = time.monotonic() + 5.0
    while not timestamps and time.monotonic() < deadline:
        time.sleep(0.01)

    if not timestamps:
        print("ERROR: No LowState received within 5s. Check interface name.")
        return

    print(f"First LowState received (mode_machine={mode_machines[0]})")
    print(f"Collecting data for {args.duration}s...")
    print()

    timestamps.clear()
    mode_machines.clear()
    time.sleep(args.duration)

    if len(timestamps) < 2:
        print(f"ERROR: Only {len(timestamps)} messages received.")
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
    print(f"  Messages received: {total_msgs}")
    print(f"  Duration:          {total_time:.3f}s")
    print(f"  Average rate:      {avg_hz:.1f} Hz")
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

    # mode_machine consistency
    unique_mm = set(mode_machines)
    print(f"\n  mode_machine values: {unique_mm}")
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
