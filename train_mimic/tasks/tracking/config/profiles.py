"""Internal tracking task profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackingTaskProfile:
    name: str
    experiment_name: str
    sampling_mode: str
    episode_length_s: float = 10.0
    anchor_pos_threshold: float = 0.4
    anchor_ori_threshold: float = 1.0
    ee_body_pos_threshold: float = 0.4
    adaptive_kernel_size: int | None = None
    adaptive_uniform_ratio: float | None = None
    adaptive_alpha: float | None = None


OFFICIAL_UNIFORM_PROFILE = TrackingTaskProfile(
    name="official_uniform_no_state_est",
    experiment_name="g1_tracking",
    sampling_mode="uniform",
)

VELCMD_HISTORY_ADAPTIVE_PROFILE = TrackingTaskProfile(
    name="velcmd_history_adaptive",
    experiment_name="g1_tracking_velcmd_history_adaptive",
    sampling_mode="adaptive",
)
