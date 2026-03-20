from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from teleopit.sim.reference_timeline import (
    ReferenceSample,
    ReferenceTimeline,
    ReferenceWindow,
    ReferenceWindowBuilder,
)

Float32Array = NDArray[np.float32]


class ExponentialVecSmoother:
    def __init__(self, alpha: float) -> None:
        alpha_f = float(alpha)
        if not np.isfinite(alpha_f) or alpha_f <= 0.0 or alpha_f > 1.0:
            raise ValueError(f"alpha must be finite and in (0, 1], got {alpha}")
        self._alpha = alpha_f
        self._state: Float32Array | None = None

    def reset(self) -> None:
        self._state = None

    def apply(self, value: Float32Array | NDArray[np.generic]) -> Float32Array:
        cur = np.asarray(value, dtype=np.float32).reshape(-1)
        if self._state is None or self._state.shape != cur.shape or self._alpha >= 1.0 - 1e-6:
            self._state = cur.copy()
            return self._state.copy()

        self._state = np.asarray(
            (1.0 - self._alpha) * self._state + self._alpha * cur,
            dtype=np.float32,
        )
        return self._state.copy()


@dataclass(frozen=True)
class RealtimeReferenceDiagnostics:
    future_horizon_steps: int
    real_frame_count: int
    warmup_done: bool
    used_repeat_padding: bool
    padding_active: bool


class RealtimeReferenceManager:
    def __init__(
        self,
        *,
        reference_window_builder: ReferenceWindowBuilder,
        low_watermark_steps: int = 0,
        high_watermark_steps: int | None = None,
        warmup_steps: int = 0,
    ) -> None:
        self._builder = reference_window_builder
        self._low_watermark_steps = max(int(low_watermark_steps), self._builder.max_future_step)
        if high_watermark_steps is None:
            self._high_watermark_steps = self._low_watermark_steps
        else:
            self._high_watermark_steps = int(high_watermark_steps)
        if self._low_watermark_steps < 0:
            raise ValueError("low_watermark_steps must be >= 0")
        if self._high_watermark_steps < self._low_watermark_steps:
            raise ValueError(
                "high_watermark_steps must be >= low_watermark_steps, "
                f"got {self._high_watermark_steps} < {self._low_watermark_steps}"
            )
        self._warmup_steps = max(int(warmup_steps), 0)
        self.reset()

    def reset(self) -> None:
        self._padding_active = False
        self._real_frame_count = 0

    def note_realtime_frame(self) -> None:
        self._real_frame_count += 1

    @property
    def real_frame_count(self) -> int:
        return self._real_frame_count

    @property
    def warmup_done(self) -> bool:
        return self._real_frame_count >= self._warmup_steps

    def future_horizon_steps(
        self,
        timeline: ReferenceTimeline,
        base_time_s: float,
    ) -> int:
        latest_timestamp = timeline.latest_timestamp()
        if latest_timestamp is None:
            return 0
        horizon_s = max(0.0, float(latest_timestamp) - float(base_time_s))
        return max(0, int(np.floor(horizon_s / self._builder.policy_dt_s + 1e-6)))

    def sample(
        self,
        timeline: ReferenceTimeline,
        base_time_s: float,
    ) -> tuple[ReferenceWindow, RealtimeReferenceDiagnostics]:
        window = self._builder.sample(timeline, base_time_s)
        future_horizon_steps = self.future_horizon_steps(timeline, base_time_s)
        self._update_padding_state(future_horizon_steps)
        target_padding_horizon = (
            self._high_watermark_steps
            if self._padding_active
            else self._builder.max_future_step
        )
        padded_window, used_repeat_padding = self._pad_future_window(
            timeline,
            window,
            base_time_s,
            target_padding_horizon,
        )
        return padded_window, RealtimeReferenceDiagnostics(
            future_horizon_steps=future_horizon_steps,
            real_frame_count=self._real_frame_count,
            warmup_done=self.warmup_done,
            used_repeat_padding=used_repeat_padding,
            padding_active=self._padding_active,
        )

    def _update_padding_state(self, future_horizon_steps: int) -> None:
        if self._high_watermark_steps <= 0:
            self._padding_active = False
            return
        if future_horizon_steps <= self._low_watermark_steps:
            self._padding_active = True
        elif future_horizon_steps >= self._high_watermark_steps:
            self._padding_active = False

    def _pad_future_window(
        self,
        timeline: ReferenceTimeline,
        window: ReferenceWindow,
        base_time_s: float,
        target_padding_horizon: int,
    ) -> tuple[ReferenceWindow, bool]:
        latest_timestamp = timeline.latest_timestamp()
        if latest_timestamp is None or target_padding_horizon <= 0:
            return window, False

        latest_sample = timeline.sample_at(latest_timestamp)
        latest_ts = float(latest_timestamp)
        policy_dt_s = self._builder.policy_dt_s
        samples: list[ReferenceSample] = []
        used_repeat_padding = False

        for step, sample in zip(window.reference_steps, window.samples):
            target_time_s = float(base_time_s) + float(step) * policy_dt_s
            if step > 0 and step <= target_padding_horizon and target_time_s > latest_ts + 1e-9:
                used_repeat_padding = True
                samples.append(
                    ReferenceSample(
                        qpos=np.asarray(latest_sample.qpos, dtype=np.float64).copy(),
                        timestamp_s=target_time_s,
                        mode="repeat_latest",
                        used_fallback=False,
                        older_timestamp_s=latest_ts,
                        newer_timestamp_s=latest_ts,
                        alpha=None,
                    )
                )
            else:
                samples.append(sample)

        if not used_repeat_padding:
            return window, False

        return ReferenceWindow(
            base_time_s=float(window.base_time_s),
            policy_dt_s=float(window.policy_dt_s),
            reference_steps=tuple(window.reference_steps),
            samples=tuple(samples),
        ), True
