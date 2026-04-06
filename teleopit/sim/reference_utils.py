"""Shared reference-window utilities used by both offline simulation and sim2real.

These are pure functions extracted from ``SimLoop`` and ``Sim2RealController`` to avoid
code duplication.  Each function takes explicit arguments instead of ``self``.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from teleopit.sim.reference_motion import OfflineReferenceMotion
from teleopit.sim.reference_timeline import ReferenceSample, ReferenceWindow, ReferenceWindowBuilder

Float64Array = NDArray[np.float64]


def obs_builder_requires_reference_window(obs_builder: object) -> bool:
    """Return True if *obs_builder* consumes a reference window."""
    return bool(getattr(obs_builder, "requires_reference_window", False)) or callable(
        getattr(obs_builder, "build_with_reference_window", None)
    )


def sample_offline_reference_at(
    offline_reference: OfflineReferenceMotion,
    target_time_s: float,
) -> ReferenceSample:
    """Sample *offline_reference* at *target_time_s*, clamping at boundaries."""
    fallback_mode: str | None = None
    sample_time_s = float(target_time_s)
    if sample_time_s < 0.0:
        sample_time_s = 0.0
        fallback_mode = "fallback_oldest"
    elif sample_time_s >= offline_reference.duration_s:
        sample_time_s = float(np.nextafter(offline_reference.duration_s, 0.0))
        fallback_mode = "fallback_latest"

    sampled = offline_reference.sample(sample_time_s)
    if sampled is None:
        sample_time_s = float(np.nextafter(offline_reference.duration_s, 0.0))
        sampled = offline_reference.sample(sample_time_s)
        fallback_mode = "fallback_latest"
    if sampled is None:
        raise RuntimeError("OfflineReferenceMotion could not sample a valid reference window frame")

    if fallback_mode is not None:
        return ReferenceSample(
            qpos=np.asarray(sampled.qpos, dtype=np.float64).copy(),
            timestamp_s=sample_time_s,
            mode=fallback_mode,
            used_fallback=True,
            older_timestamp_s=sample_time_s,
            newer_timestamp_s=sample_time_s,
            alpha=None,
        )

    older_timestamp_s = float(sampled.frame_idx0) / float(offline_reference.fps)
    newer_timestamp_s = float(sampled.frame_idx1) / float(offline_reference.fps)
    mode = "single_frame" if sampled.frame_idx0 == sampled.frame_idx1 else "interpolate"
    return ReferenceSample(
        qpos=np.asarray(sampled.qpos, dtype=np.float64).copy(),
        timestamp_s=float(sample_time_s),
        mode=mode,
        used_fallback=False,
        older_timestamp_s=older_timestamp_s,
        newer_timestamp_s=newer_timestamp_s,
        alpha=float(sampled.alpha),
    )


def build_offline_reference_window(
    offline_reference: OfflineReferenceMotion,
    base_time_s: float,
    reference_window_builder: ReferenceWindowBuilder,
    policy_hz: float,
) -> ReferenceWindow:
    """Build a :class:`ReferenceWindow` from *offline_reference* anchored at *base_time_s*."""
    reference_steps = tuple(reference_window_builder.reference_steps)
    samples = tuple(
        sample_offline_reference_at(
            offline_reference,
            float(base_time_s) + float(step) * (1.0 / policy_hz),
        )
        for step in reference_steps
    )
    return ReferenceWindow(
        base_time_s=float(base_time_s),
        policy_dt_s=1.0 / policy_hz,
        reference_steps=reference_steps,
        samples=samples,
    )


def build_static_reference_window(
    qpos: Float64Array,
    reference_window_builder: ReferenceWindowBuilder,
    policy_hz: float,
) -> ReferenceWindow:
    """Build a static :class:`ReferenceWindow` where every sample holds *qpos*."""
    base_time_s = time.monotonic()
    reference_steps = tuple(reference_window_builder.reference_steps)
    qpos_copy = np.asarray(qpos, dtype=np.float64).reshape(-1).copy()
    samples = tuple(
        ReferenceSample(
            qpos=qpos_copy.copy(),
            timestamp_s=base_time_s + float(step) / policy_hz,
            mode="static_reference",
            used_fallback=False,
            older_timestamp_s=base_time_s + float(step) / policy_hz,
            newer_timestamp_s=base_time_s + float(step) / policy_hz,
            alpha=None,
        )
        for step in reference_steps
    )
    return ReferenceWindow(
        base_time_s=base_time_s,
        policy_dt_s=1.0 / policy_hz,
        reference_steps=reference_steps,
        samples=samples,
    )
