"""Shared quaternion and geometry helpers (wxyz convention)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]


def quat_inv_np(q: Float32Array) -> Float32Array:
    """Conjugate (inverse for unit quaternions), wxyz layout."""
    inv = q.copy()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def quat_mul_np(q1: Float32Array, q2: Float32Array) -> Float32Array:
    """Hamilton product of two wxyz quaternions."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1).astype(np.float32)
