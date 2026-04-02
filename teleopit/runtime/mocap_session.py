from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray


Float64Array = NDArray[np.float64]


class MocapSessionState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    RESUMING = "resuming"


class MocapSessionManager:
    def __init__(self) -> None:
        self._state = MocapSessionState.ACTIVE
        self._hold_qpos: Float64Array | None = None

    @property
    def state(self) -> MocapSessionState:
        return self._state

    @property
    def hold_qpos(self) -> Float64Array | None:
        if self._hold_qpos is None:
            return None
        return self._hold_qpos.copy()

    def reset(self) -> None:
        self._state = MocapSessionState.ACTIVE
        self._hold_qpos = None

    def pause(self, qpos: Float64Array) -> None:
        self._hold_qpos = np.asarray(qpos, dtype=np.float64).reshape(-1).copy()
        self._state = MocapSessionState.PAUSED

    def begin_resume(self) -> Float64Array:
        if self._hold_qpos is None:
            raise RuntimeError("Cannot resume mocap without a paused hold qpos")
        self._state = MocapSessionState.RESUMING
        return self._hold_qpos.copy()

    def finish_resume(self) -> None:
        self._state = MocapSessionState.ACTIVE
