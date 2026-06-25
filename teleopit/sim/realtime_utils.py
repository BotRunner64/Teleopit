from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from teleopit.sim.reference_timeline import ReferenceTimeline, ReferenceWindow, ReferenceWindowBuilder

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
    requested_base_time_s: float
    effective_base_time_s: float
    latest_timestamp_s: float | None


class RealtimeReferenceManager:
    def __init__(
        self,
        *,
        reference_window_builder: ReferenceWindowBuilder,
        warmup_steps: int = 0,
    ) -> None:
        self._builder = reference_window_builder
        self._warmup_steps = max(int(warmup_steps), 0)
        self.reset()

    def reset(self) -> None:
        self._real_frame_count = 0

    def set_warmup_steps(self, warmup_steps: int) -> None:
        self._warmup_steps = max(int(warmup_steps), 0)

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
        effective_base_time_s = requested_base_time_s
        window = self._builder.sample(timeline, effective_base_time_s)
        future_horizon_steps = self.future_horizon_steps(timeline, effective_base_time_s)
        return window, RealtimeReferenceDiagnostics(
            future_horizon_steps=future_horizon_steps,
            real_frame_count=self._real_frame_count,
            warmup_done=self.warmup_done,
            requested_base_time_s=requested_base_time_s,
            effective_base_time_s=effective_base_time_s,
            latest_timestamp_s=None if latest_timestamp is None else float(latest_timestamp),
        )
