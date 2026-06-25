from __future__ import annotations

import numpy as np

PICO_BRIDGE_TO_MEDIAPIPE = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    9,
    10,
    12,
    13,
    14,
    15,
    17,
    18,
    19,
    20,
    22,
    23,
    24,
    25,
)

PICO_NATIVE_TO_RH = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def pico_hand_to_landmarks(hand_state: object) -> np.ndarray:
    state = np.asarray(hand_state, dtype=np.float64)
    if state.shape != (26, 7):
        state = state.reshape(26, 7)
    positions = state[:, :3] @ PICO_NATIVE_TO_RH.T
    landmarks = np.empty((21, 3), dtype=np.float64)
    for mp_index, pico_index in enumerate(PICO_BRIDGE_TO_MEDIAPIPE):
        landmarks[mp_index] = positions[pico_index]
    return landmarks
