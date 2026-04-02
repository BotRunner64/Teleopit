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
    requested_base_time_s: float
    effective_base_time_s: float
    latest_timestamp_s: float | None
    used_catchup: bool
    catchup_active: bool


class RealtimeReferenceManager:
    def __init__(
        self,
        *,
        reference_window_builder: ReferenceWindowBuilder,
        low_watermark_steps: int = 0,
        high_watermark_steps: int | None = None,
        warmup_steps: int = 0,
        catchup_enabled: bool = False,
        catchup_trigger_steps: int | None = None,
        catchup_release_steps: int | None = None,
        catchup_target_delay_s: float | None = None,
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
        self._catchup_enabled = bool(catchup_enabled)
        self._catchup_trigger_steps = (
            max(int(catchup_trigger_steps), 0)
            if catchup_trigger_steps is not None
            else max(self._high_watermark_steps, self._low_watermark_steps + 2)
        )
        self._catchup_release_steps = (
            max(int(catchup_release_steps), 0)
            if catchup_release_steps is not None
            else max(self._low_watermark_steps, self._builder.max_future_step)
        )
        if self._catchup_release_steps > self._catchup_trigger_steps:
            raise ValueError(
                "catchup_release_steps must be <= catchup_trigger_steps, "
                f"got {self._catchup_release_steps} > {self._catchup_trigger_steps}"
            )
        self._catchup_target_delay_s = (
            float(catchup_target_delay_s)
            if catchup_target_delay_s is not None
            else float(self._builder.policy_dt_s * max(self._builder.max_future_step, 1))
        )
        if (
            not np.isfinite(self._catchup_target_delay_s)
            or self._catchup_target_delay_s < 0.0
        ):
            raise ValueError(
                "catchup_target_delay_s must be finite and >= 0, "
                f"got {catchup_target_delay_s}"
            )
        self.reset()

    def reset(self) -> None:
        self._padding_active = False
        self._catchup_active = False
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
        requested_base_time_s = float(base_time_s)
        latest_timestamp = timeline.latest_timestamp()
        requested_future_horizon_steps = self.future_horizon_steps(timeline, requested_base_time_s)
        effective_base_time_s, used_catchup = self._apply_catchup(
            requested_base_time_s,
            latest_timestamp,
            requested_future_horizon_steps,
        )
        window = self._builder.sample(timeline, effective_base_time_s)
        future_horizon_steps = self.future_horizon_steps(timeline, effective_base_time_s)
        self._update_padding_state(future_horizon_steps)
        target_padding_horizon = (
            self._high_watermark_steps
            if self._padding_active
            else self._builder.max_future_step
        )
        padded_window, used_repeat_padding = self._pad_future_window(
            timeline,
            window,
            effective_base_time_s,
            target_padding_horizon,
        )
        return padded_window, RealtimeReferenceDiagnostics(
            future_horizon_steps=future_horizon_steps,
            real_frame_count=self._real_frame_count,
            warmup_done=self.warmup_done,
            used_repeat_padding=used_repeat_padding,
            padding_active=self._padding_active,
            requested_base_time_s=requested_base_time_s,
            effective_base_time_s=effective_base_time_s,
            latest_timestamp_s=None if latest_timestamp is None else float(latest_timestamp),
            used_catchup=used_catchup,
            catchup_active=self._catchup_active,
        )

    def _apply_catchup(
        self,
        requested_base_time_s: float,
        latest_timestamp: float | None,
        requested_future_horizon_steps: int,
    ) -> tuple[float, bool]:
        if not self._catchup_enabled or latest_timestamp is None:
            self._catchup_active = False
            return requested_base_time_s, False

        if self._catchup_active:
            if requested_future_horizon_steps <= self._catchup_release_steps:
                self._catchup_active = False
        elif requested_future_horizon_steps >= self._catchup_trigger_steps:
            self._catchup_active = True

        if not self._catchup_active:
            return requested_base_time_s, False

        target_base_time_s = max(
            requested_base_time_s,
            float(latest_timestamp) - self._catchup_target_delay_s,
        )
        used_catchup = target_base_time_s > requested_base_time_s + 1e-9
        return target_base_time_s, used_catchup

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
