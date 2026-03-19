from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from teleopit.sim.reference_motion import interpolate_retarget_qpos

Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class ReferenceSample:
    qpos: Float64Array
    timestamp_s: float
    mode: str
    used_fallback: bool
    older_timestamp_s: float | None
    newer_timestamp_s: float | None
    alpha: float | None


@dataclass(frozen=True)
class ReferenceWindow:
    base_time_s: float
    policy_dt_s: float
    reference_steps: tuple[int, ...]
    samples: tuple[ReferenceSample, ...]

    def current_sample(self) -> ReferenceSample:
        try:
            idx = self.reference_steps.index(0)
        except ValueError as exc:
            raise ValueError("reference_steps must contain 0") from exc
        return self.samples[idx]

    def modes(self) -> tuple[str, ...]:
        return tuple(sample.mode for sample in self.samples)

    def alphas(self) -> tuple[float, ...]:
        return tuple(-1.0 if sample.alpha is None else float(sample.alpha) for sample in self.samples)

    def fallback_mask(self) -> tuple[bool, ...]:
        return tuple(bool(sample.used_fallback) for sample in self.samples)

    def timestamps(self) -> tuple[float, ...]:
        return tuple(float(sample.timestamp_s) for sample in self.samples)


@dataclass(frozen=True)
class _TimelineFrame:
    timestamp_s: float
    qpos: Float64Array


def parse_reference_steps(raw: object | None) -> tuple[int, ...]:
    if raw is None:
        return (0,)
    if isinstance(raw, np.ndarray):
        values = raw.reshape(-1).tolist()
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        raise ValueError(f"reference_steps must be a sequence of ints, got {raw}")
    if not values:
        raise ValueError("reference_steps must contain at least one step")

    steps: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"reference_steps entries must be ints, got {value}")
        steps.append(int(value))
    if 0 not in steps:
        raise ValueError(f"reference_steps must contain 0, got {steps}")

    seen_negative = False
    for step in steps[1:]:
        if step < 0:
            seen_negative = True
        elif seen_negative:
            raise ValueError(
                "reference_steps format must be [0, ...future/non-negative, ...history/negative]. "
                f"Got {steps}."
            )
    return tuple(steps)


class ReferenceTimeline:
    def __init__(self, *, window_s: float, max_frames: int | None = None) -> None:
        if not np.isfinite(window_s) or window_s <= 0.0:
            raise ValueError(f"window_s must be > 0, got {window_s}")
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive when set, got {max_frames}")
        self._window_s = float(window_s)
        self._max_frames = max_frames
        self._frames: deque[_TimelineFrame] = deque()

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def latest_timestamp(self) -> float | None:
        if not self._frames:
            return None
        return float(self._frames[-1].timestamp_s)

    def append(self, qpos: Float64Array, timestamp_s: float) -> None:
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos_arr.shape[0] < 7:
            raise ValueError(f"Reference qpos must contain at least root pose, got {qpos_arr.shape[0]}")
        ts = float(timestamp_s)
        if not np.isfinite(ts):
            raise ValueError(f"timestamp_s must be finite, got {timestamp_s}")

        frame = _TimelineFrame(timestamp_s=ts, qpos=qpos_arr.copy())
        if self._frames and ts <= self._frames[-1].timestamp_s + 1e-9:
            self._frames[-1] = _TimelineFrame(
                timestamp_s=float(self._frames[-1].timestamp_s),
                qpos=frame.qpos,
            )
        else:
            self._frames.append(frame)
        self._trim()

    def sample_at(self, target_time_s: float) -> ReferenceSample:
        if not self._frames:
            raise RuntimeError("ReferenceTimeline is empty")
        target = float(target_time_s)
        if not np.isfinite(target):
            raise ValueError(f"target_time_s must be finite, got {target_time_s}")

        frames = tuple(self._frames)
        if len(frames) == 1:
            only = frames[0]
            return ReferenceSample(
                qpos=only.qpos.copy(),
                timestamp_s=float(only.timestamp_s),
                mode="single_frame",
                used_fallback=True,
                older_timestamp_s=float(only.timestamp_s),
                newer_timestamp_s=float(only.timestamp_s),
                alpha=None,
            )

        oldest = frames[0]
        newest = frames[-1]
        if target <= oldest.timestamp_s:
            return ReferenceSample(
                qpos=oldest.qpos.copy(),
                timestamp_s=float(oldest.timestamp_s),
                mode="fallback_oldest",
                used_fallback=True,
                older_timestamp_s=float(oldest.timestamp_s),
                newer_timestamp_s=float(oldest.timestamp_s),
                alpha=None,
            )
        if target >= newest.timestamp_s:
            return ReferenceSample(
                qpos=newest.qpos.copy(),
                timestamp_s=float(newest.timestamp_s),
                mode="fallback_latest",
                used_fallback=True,
                older_timestamp_s=float(newest.timestamp_s),
                newer_timestamp_s=float(newest.timestamp_s),
                alpha=None,
            )

        for idx in range(1, len(frames)):
            older = frames[idx - 1]
            newer = frames[idx]
            if target > newer.timestamp_s:
                continue
            dt = float(newer.timestamp_s - older.timestamp_s)
            if dt <= 1e-9:
                return ReferenceSample(
                    qpos=newer.qpos.copy(),
                    timestamp_s=float(newer.timestamp_s),
                    mode="single_frame",
                    used_fallback=True,
                    older_timestamp_s=float(newer.timestamp_s),
                    newer_timestamp_s=float(newer.timestamp_s),
                    alpha=None,
                )
            alpha = float(np.clip((target - older.timestamp_s) / dt, 0.0, 1.0))
            return ReferenceSample(
                qpos=interpolate_retarget_qpos(older.qpos, newer.qpos, alpha),
                timestamp_s=float(target),
                mode="interpolate",
                used_fallback=False,
                older_timestamp_s=float(older.timestamp_s),
                newer_timestamp_s=float(newer.timestamp_s),
                alpha=alpha,
            )

        return ReferenceSample(
            qpos=newest.qpos.copy(),
            timestamp_s=float(newest.timestamp_s),
            mode="fallback_latest",
            used_fallback=True,
            older_timestamp_s=float(newest.timestamp_s),
            newer_timestamp_s=float(newest.timestamp_s),
            alpha=None,
        )

    def sample_many(self, target_times_s: Sequence[float]) -> tuple[ReferenceSample, ...]:
        return tuple(self.sample_at(target_time_s) for target_time_s in target_times_s)

    def _trim(self) -> None:
        if not self._frames:
            return
        latest_ts = float(self._frames[-1].timestamp_s)
        cutoff_ts = latest_ts - self._window_s
        while len(self._frames) > 1 and self._frames[0].timestamp_s < cutoff_ts:
            self._frames.popleft()
        if self._max_frames is not None:
            while len(self._frames) > self._max_frames:
                self._frames.popleft()


class ReferenceWindowBuilder:
    def __init__(self, *, policy_dt_s: float, reference_steps: Sequence[int]) -> None:
        if not np.isfinite(policy_dt_s) or policy_dt_s <= 0.0:
            raise ValueError(f"policy_dt_s must be > 0, got {policy_dt_s}")
        self._policy_dt_s = float(policy_dt_s)
        self._reference_steps = parse_reference_steps(reference_steps)

    @property
    def reference_steps(self) -> tuple[int, ...]:
        return self._reference_steps

    @property
    def policy_dt_s(self) -> float:
        return self._policy_dt_s

    @property
    def max_future_step(self) -> int:
        return max((step for step in self._reference_steps if step > 0), default=0)

    @property
    def min_history_step(self) -> int:
        return min((step for step in self._reference_steps if step < 0), default=0)

    @property
    def requires_timeline(self) -> bool:
        return self.max_future_step > 0 or self.min_history_step < 0

    def required_delay_s(self) -> float:
        return float(self.max_future_step) * self._policy_dt_s

    def required_window_s(self, delay_s: float) -> float:
        delay = float(delay_s)
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError(f"delay_s must be finite and >= 0, got {delay_s}")
        return delay + float(abs(self.min_history_step)) * self._policy_dt_s

    def validate_runtime_support(
        self,
        *,
        delay_s: float,
        window_s: float,
        config_label: str = "Reference timeline",
    ) -> None:
        delay = float(delay_s)
        if not np.isfinite(delay) or delay < 0.0:
            raise ValueError(f"{config_label}: retarget_buffer_delay_s must be finite and >= 0, got {delay_s}")

        window = float(window_s)
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError(f"{config_label}: retarget_buffer_window_s must be > 0, got {window_s}")

        required_delay = self.required_delay_s()
        if delay + 1e-9 < required_delay:
            raise ValueError(
                f"{config_label}: retarget_buffer_delay_s={delay:.6f}s is too small for "
                f"reference_steps={list(self._reference_steps)} at policy_dt={self._policy_dt_s:.6f}s; "
                f"need >= {required_delay:.6f}s to sample the furthest future step. "
                "Increase retarget_buffer_delay_s or remove positive reference_steps."
            )

        required_window = self.required_window_s(delay)
        if window + 1e-9 < required_window:
            raise ValueError(
                f"{config_label}: retarget_buffer_window_s={window:.6f}s is too small for "
                f"reference_steps={list(self._reference_steps)} with retarget_buffer_delay_s={delay:.6f}s; "
                f"need >= {required_window:.6f}s to retain the requested history/current sample window. "
                "Increase retarget_buffer_window_s or reduce negative reference_steps."
            )

    def sample(self, timeline: ReferenceTimeline, base_time_s: float) -> ReferenceWindow:
        target_times_s = tuple(
            float(base_time_s) + float(step) * self._policy_dt_s
            for step in self._reference_steps
        )
        return ReferenceWindow(
            base_time_s=float(base_time_s),
            policy_dt_s=self._policy_dt_s,
            reference_steps=self._reference_steps,
            samples=timeline.sample_many(target_times_s),
        )
