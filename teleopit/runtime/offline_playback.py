from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OfflinePlaybackController:
    duration_s: float
    step_dt_s: float
    pause_on_end: bool = False

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be > 0, got {self.duration_s}")
        if self.step_dt_s <= 0.0:
            raise ValueError(f"step_dt_s must be > 0, got {self.step_dt_s}")
        self._last_sample_time_s = float(np.nextafter(self.duration_s, 0.0))
        self._time_s = 0.0
        self._paused = False
        self._finished = False

    @property
    def current_time_s(self) -> float:
        return self._time_s

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def finished(self) -> bool:
        return self._finished

    def replay(self) -> None:
        self._time_s = 0.0
        self._paused = False
        self._finished = False

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        if self._finished:
            raise RuntimeError("Offline playback already finished; replay from the beginning instead.")
        self._paused = False

    def finish(self) -> None:
        self._time_s = self._last_sample_time_s
        self._finished = True
        self._paused = bool(self.pause_on_end)

    def advance(self) -> bool:
        if self._paused or self._finished:
            return False
        next_time_s = self._time_s + self.step_dt_s
        if next_time_s >= self.duration_s:
            self.finish()
            return True
        self._time_s = next_time_s
        return False
