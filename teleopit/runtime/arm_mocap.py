from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.constants import ROOT_DIM
from teleopit.runtime.common import cfg_get
from teleopit.sim.reference_timeline import ReferenceSample, ReferenceWindow


Float64Array = NDArray[np.float64]


def parse_arm_joint_indices(cfg: Any, *, num_actions: int) -> NDArray[np.int64]:
    arm_mocap_cfg = cfg_get(cfg, "arm_mocap", {}) or {}
    raw = cfg_get(arm_mocap_cfg, "controlled_joint_indices", None)
    default = list(range(15, num_actions)) if num_actions > 15 else [num_actions - 1]
    indices = np.asarray(default if raw is None else raw, dtype=np.int64).reshape(-1)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= num_actions):
        raise ValueError(
            f"arm_mocap.controlled_joint_indices must contain valid joint indices in [0, {num_actions}), "
            f"got {indices.tolist()}"
        )
    if np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError("arm_mocap.controlled_joint_indices must not contain duplicates")
    return indices


def compose_arm_reference(
    *,
    standing_qpos: Float64Array,
    retarget_qpos: Float64Array,
    arm_joint_indices: NDArray[np.int64],
    num_actions: int,
) -> Float64Array:
    retarget = np.asarray(retarget_qpos, dtype=np.float64).reshape(-1)
    if retarget.shape[0] < ROOT_DIM + num_actions:
        raise ValueError(f"Retargeted qpos too short: {retarget.shape[0]} (need >= {ROOT_DIM + num_actions})")
    composed = np.asarray(standing_qpos, dtype=np.float64).reshape(-1).copy()
    joint_indices = ROOT_DIM + np.asarray(arm_joint_indices, dtype=np.int64).reshape(-1)
    composed[joint_indices] = retarget[joint_indices]
    return composed


def compose_arm_reference_window(
    reference_window: ReferenceWindow | None,
    *,
    standing_qpos: Float64Array,
    arm_joint_indices: NDArray[np.int64],
    num_actions: int,
) -> ReferenceWindow | None:
    if reference_window is None:
        return None
    samples = tuple(
        ReferenceSample(
            qpos=compose_arm_reference(
                standing_qpos=standing_qpos,
                retarget_qpos=sample.qpos,
                arm_joint_indices=arm_joint_indices,
                num_actions=num_actions,
            ),
            timestamp_s=float(sample.timestamp_s),
            mode=str(sample.mode),
            used_fallback=bool(sample.used_fallback),
            older_timestamp_s=sample.older_timestamp_s,
            newer_timestamp_s=sample.newer_timestamp_s,
            alpha=sample.alpha,
        )
        for sample in reference_window.samples
    )
    return ReferenceWindow(
        base_time_s=float(reference_window.base_time_s),
        policy_dt_s=float(reference_window.policy_dt_s),
        reference_steps=tuple(reference_window.reference_steps),
        samples=samples,
    )
