from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

HumanFrame = dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]
Float64Array = NDArray[np.float64]


def _normalize_quat(quat: NDArray[np.floating[Any]]) -> Float64Array:
    out = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return out / norm


def slerp_quat_wxyz(
    quat0: NDArray[np.floating[Any]],
    quat1: NDArray[np.floating[Any]],
    alpha: float,
) -> Float64Array:
    q0 = _normalize_quat(quat0)
    q1 = _normalize_quat(quat1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        blended = q0 + alpha * (q1 - q0)
        return _normalize_quat(blended)

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    if sin_theta_0 <= 1e-8:
        return q0.copy()

    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return _normalize_quat(s0 * q0 + s1 * q1)


def interpolate_human_frames(frame0: HumanFrame, frame1: HumanFrame, alpha: float) -> HumanFrame:
    if alpha <= 0.0:
        return {
            name: (
                np.asarray(data[0], dtype=np.float64).copy(),
                _normalize_quat(data[1]),
            )
            for name, data in frame0.items()
        }
    if alpha >= 1.0:
        return {
            name: (
                np.asarray(data[0], dtype=np.float64).copy(),
                _normalize_quat(data[1]),
            )
            for name, data in frame1.items()
        }

    out: HumanFrame = {}
    for name, data0 in frame0.items():
        if name not in frame1:
            raise KeyError(f"Human frame interpolation missing body '{name}' in second frame")
        pos0 = np.asarray(data0[0], dtype=np.float64).reshape(3)
        quat0 = np.asarray(data0[1], dtype=np.float64).reshape(4)
        pos1 = np.asarray(frame1[name][0], dtype=np.float64).reshape(3)
        quat1 = np.asarray(frame1[name][1], dtype=np.float64).reshape(4)
        out[name] = (
            pos0 + alpha * (pos1 - pos0),
            slerp_quat_wxyz(quat0, quat1, alpha),
        )
    return out


def interpolate_retarget_qpos(qpos0: Float64Array, qpos1: Float64Array, alpha: float) -> Float64Array:
    cur0 = np.asarray(qpos0, dtype=np.float64).reshape(-1)
    cur1 = np.asarray(qpos1, dtype=np.float64).reshape(-1)
    if cur0.shape != cur1.shape:
        raise ValueError(
            f"Retarget qpos shapes must match for interpolation, got {cur0.shape} vs {cur1.shape}"
        )
    if cur0.shape[0] < 7:
        raise ValueError(f"Retarget qpos must contain at least root pose, got {cur0.shape[0]}")

    if alpha <= 0.0:
        return cur0.copy()
    if alpha >= 1.0:
        return cur1.copy()

    out = cur0 + alpha * (cur1 - cur0)
    out[3:7] = slerp_quat_wxyz(cur0[3:7], cur1[3:7], alpha)
    return out.astype(np.float64, copy=False)


@dataclass(frozen=True)
class OfflineReferenceSample:
    human_frame: HumanFrame
    qpos: Float64Array
    frame_f: float
    frame_idx0: int
    frame_idx1: int
    alpha: float


class OfflineReferenceMotion:
    """Precompute sequential retargeted motion for offline BVH playback.

    The benchmark path interpolates motion clips at the policy step rate.
    For offline BVH sim2sim we mirror that by retargeting each source BVH frame
    once in chronological order, then interpolating the retargeted qpos at the
    policy timestamp.
    """

    def __init__(self, input_provider: Any, retargeter: Any) -> None:
        if not hasattr(input_provider, "__len__") or not hasattr(input_provider, "get_frame_by_index"):
            raise TypeError(
                "OfflineReferenceMotion requires an input provider with __len__() "
                "and get_frame_by_index()."
            )
        raw_fps = getattr(input_provider, "fps", None)
        if not isinstance(raw_fps, (int, float)) or float(raw_fps) <= 0.0:
            raise ValueError(f"OfflineReferenceMotion requires a positive provider fps, got {raw_fps}")

        self._input_provider = input_provider
        self._retargeter = retargeter
        self._fps = float(raw_fps)
        self._num_frames = int(len(input_provider))
        if self._num_frames <= 0:
            raise ValueError("OfflineReferenceMotion requires at least one source frame")
        self._human_frame_cache: dict[int, HumanFrame] = {}
        self._retarget_qpos_cache: dict[int, Float64Array] = {}

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def duration_s(self) -> float:
        return float(self._num_frames) / self._fps

    def sample(self, policy_time_s: float) -> OfflineReferenceSample | None:
        if policy_time_s < 0.0:
            raise ValueError(f"policy_time_s must be non-negative, got {policy_time_s}")
        if policy_time_s >= self.duration_s:
            return None

        frame_f = min(policy_time_s * self._fps, float(self._num_frames - 1))
        frame_idx0 = int(np.floor(frame_f))
        frame_idx1 = min(frame_idx0 + 1, self._num_frames - 1)
        alpha = float(frame_f - frame_idx0)

        human_frame = interpolate_human_frames(
            self._get_human_frame(frame_idx0),
            self._get_human_frame(frame_idx1),
            alpha,
        )
        qpos = interpolate_retarget_qpos(
            self._get_retarget_qpos(frame_idx0),
            self._get_retarget_qpos(frame_idx1),
            alpha,
        )
        return OfflineReferenceSample(
            human_frame=human_frame,
            qpos=qpos,
            frame_f=frame_f,
            frame_idx0=frame_idx0,
            frame_idx1=frame_idx1,
            alpha=alpha,
        )

    def _get_human_frame(self, index: int) -> HumanFrame:
        cached = self._human_frame_cache.get(index)
        if cached is not None:
            return cached
        frame = self._input_provider.get_frame_by_index(index)
        self._human_frame_cache[index] = frame
        return frame

    def _get_retarget_qpos(self, index: int) -> Float64Array:
        cached = self._retarget_qpos_cache.get(index)
        if cached is not None:
            return cached
        human_frame = self._get_human_frame(index)
        qpos = np.asarray(self._retargeter.retarget(human_frame), dtype=np.float64).reshape(-1)
        self._retarget_qpos_cache[index] = qpos.copy()
        return qpos
