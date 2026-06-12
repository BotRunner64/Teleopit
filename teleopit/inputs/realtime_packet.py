from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray


FrameT = TypeVar("FrameT")

# Canonical type alias for a per-joint pose frame: joint_name → (pos_xyz, quat_wxyz)
HumanFrame = Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]


class ControlEventType(str, Enum):
    TOGGLE_PAUSE = "toggle_pause"
    TOGGLE_ARMS = "toggle_arms"


@dataclass(frozen=True)
class ControlEvent:
    event_type: ControlEventType
    source: str
    timestamp_s: float | None = None


@dataclass(frozen=True)
class RealtimeInputPacket(Generic[FrameT]):
    frame: FrameT
    timestamp_s: float
    seq: int
    control_events: tuple[ControlEvent, ...] = ()
