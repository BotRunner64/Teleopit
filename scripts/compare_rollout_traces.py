#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_KEYS = (
    "obs",
    "obs_history",
    "action",
    "motion_joint_vel",
    "motion_anchor_lin_vel_w",
    "motion_anchor_ang_vel_w",
)


def _load_trace(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _metadata(trace: dict[str, np.ndarray]) -> dict[str, object]:
    raw = trace.get("metadata_json")
    if raw is None:
        return {}
    return json.loads(str(raw.item()))


def _first_divergent_step(diff: np.ndarray, threshold: float) -> int | None:
    if diff.ndim == 1:
        mask = diff > threshold
    else:
        axes = tuple(range(1, diff.ndim))
        mask = np.any(diff > threshold, axis=axes)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def _compare_key(
    lhs: np.ndarray,
    rhs: np.ndarray,
    *,
    key: str,
    atol: float,
) -> str:
    num_steps = min(lhs.shape[0], rhs.shape[0])
    lhs_trim = np.asarray(lhs[:num_steps], dtype=np.float64)
    rhs_trim = np.asarray(rhs[:num_steps], dtype=np.float64)
    diff = np.abs(lhs_trim - rhs_trim)
    first_bad = _first_divergent_step(diff, atol)
    mean_abs = float(diff.mean())
    max_abs = float(diff.max())
    if first_bad is None:
        return f"{key:<24} mean_abs={mean_abs:.6g} max_abs={max_abs:.6g} first_bad=none"
    return f"{key:<24} mean_abs={mean_abs:.6g} max_abs={max_abs:.6g} first_bad={first_bad}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark and sim2sim rollout traces.")
    parser.add_argument("--lhs", required=True, help="First .npz trace path")
    parser.add_argument("--rhs", required=True, help="Second .npz trace path")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(DEFAULT_KEYS),
        help="Array keys to compare (default: common policy/reference keys)",
    )
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute divergence threshold")
    args = parser.parse_args()

    lhs = _load_trace(args.lhs)
    rhs = _load_trace(args.rhs)

    lhs_meta = _metadata(lhs)
    rhs_meta = _metadata(rhs)
    if lhs_meta:
        print(f"LHS metadata: {lhs_meta}")
    if rhs_meta:
        print(f"RHS metadata: {rhs_meta}")

    lhs_steps = int(lhs.get("step", np.empty((0,))).shape[0])
    rhs_steps = int(rhs.get("step", np.empty((0,))).shape[0])
    print(f"LHS steps: {lhs_steps}")
    print(f"RHS steps: {rhs_steps}")

    compared = 0
    for key in args.keys:
        if key not in lhs or key not in rhs:
            print(f"{key:<24} skipped (missing in one trace)")
            continue
        print(_compare_key(lhs[key], rhs[key], key=key, atol=args.atol))
        compared += 1

    if compared == 0:
        raise ValueError("No comparable keys found. Check the trace paths and --keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
