"""Core abstract interfaces for Teleopit using typing.Protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class RobotState:
    """Robot state snapshot."""
    qpos: np.ndarray  # Joint positions
    qvel: np.ndarray  # Joint velocities
    quat: np.ndarray  # Base orientation quaternion
    ang_vel: np.ndarray  # Base angular velocity
    timestamp: float  # Timestamp in seconds
    base_pos: np.ndarray | None = None  # Base position in world frame
    base_lin_vel: np.ndarray | None = None  # Base linear velocity in body frame


@runtime_checkable
class InputProvider(Protocol):
    """Minimal interface shared by all human-motion input sources.

    Satisfied by both offline providers (``BVHInputProvider``)
    and realtime providers such as ``Pico4InputProvider``.
    """

    @property
    def fps(self) -> float: ...

    @property
    def bone_names(self) -> list[str]: ...

    @property
    def bone_parents(self) -> NDArray[np.int32]: ...

    def is_available(self) -> bool: ...

    def get_frame(self) -> Any: ...


@runtime_checkable
class RealtimeInputProvider(InputProvider, Protocol):
    """Extended interface for realtime streaming providers such as Pico4.

    Adds methods required by the realtime buffering pipeline that offline
    providers (BVH, looping wrappers) do not implement.
    """

    @property
    def human_format(self) -> str: ...

    def get_frame_packet(self) -> tuple[Any, float, int]: ...

    def get_realtime_input_packet(self) -> Any: ...

    def sample_frame(self, query_time_s: float, delay_s: float) -> Any: ...

    def close(self) -> None: ...


@runtime_checkable
class Retargeter(Protocol):
    """Retargets human motion to robot motion."""

    def retarget(self, human_data: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Retarget human motion to robot. Returns (base_pos, base_rot, joint_pos)."""
        ...

    def reset(self) -> None:
        """Reset retargeter state."""
        ...


@runtime_checkable
class Controller(Protocol):
    """Generates robot actions from observations."""
    
    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute action from observation."""
        ...
    
    def reset(self) -> None:
        """Reset controller state."""
        ...


@runtime_checkable
class Robot(Protocol):
    """Robot interface for state and control."""

    def get_state(self) -> RobotState:
        """Get current robot state."""
        ...

    def set_action(self, action: np.ndarray) -> None:
        """Set robot action."""
        ...

    def step(self) -> None:
        """Step robot simulation/hardware."""
        ...

    def reset(self, qpos: np.ndarray | None = None) -> None:
        """Reset robot to initial or specified state."""
        ...


@runtime_checkable
class MessageBus(Protocol):
    """Message bus for inter-component communication."""
    
    def publish(self, topic: str, data: Any) -> None:
        """Publish data to topic."""
        ...
    
    def subscribe(self, topic: str) -> Any:
        """Subscribe to topic and get latest data."""
        ...


@runtime_checkable
class Recorder(Protocol):
    """Records teleoperation data."""

    def add_frame(self, data: Dict[str, Any]) -> None:
        """Record a single frame of data."""
        ...


@runtime_checkable
class ObservationBuilder(Protocol):
    """Builds observations for controller from robot state."""

    def build_observation(self, state: RobotState, history: list[np.ndarray], action_mimic: np.ndarray) -> np.ndarray:
        """Build observation from state, history, and mimic action."""
        ...

    def reset(self) -> None:
        """Reset observation builder state."""
        ...


@runtime_checkable
class SupportsReferenceWindow(Protocol):
    """ObservationBuilder that can build observations using a reference window."""

    def build_with_reference_window(
        self, state: RobotState, reference_window: Any, motion_qpos: np.ndarray, last_action: np.ndarray,
    ) -> np.ndarray:
        """Build observation using reference window instead of velocity commands."""
        ...
