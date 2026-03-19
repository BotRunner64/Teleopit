"""Pico4 VR full-body motion capture input provider.

Uses xrobotoolkit_sdk to receive real-time body tracking data from a Pico4
headset. Outputs the same ``{joint_name: (position, quat)}`` dict format as
the BVH providers, so downstream retargeting is unchanged.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from teleopit.inputs.rot_utils import quat_mul_np
from teleopit.sim.reference_motion import interpolate_human_frames

logger = logging.getLogger(__name__)

# Pre-compute the provider-space to Teleopit-space transform.
_INPUT_TO_TELEOPIT_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
_INPUT_TO_TELEOPIT_QUAT = R.from_matrix(_INPUT_TO_TELEOPIT_MATRIX).as_quat(scalar_first=True)

# SMPL-X 24 body joint names from xrobotoolkit_sdk
BODY_JOINT_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
]

HumanFrame = Dict[str, Tuple[NDArray[np.float64], NDArray[np.float64]]]


def _coordinate_transform_input(body_pose_dict: dict) -> dict:
    """Transform provider-space poses into Teleopit's expected coordinates."""
    for body_name, value in body_pose_dict.items():
        x, y, z = value[0]
        qw, qx, qy, qz = value[1]

        orientation = quat_mul_np(
            _INPUT_TO_TELEOPIT_QUAT, np.array([qw, qx, qy, qz]), scalar_first=True
        )
        position = np.array([x, y, z]) @ _INPUT_TO_TELEOPIT_MATRIX.T

        body_pose_dict[body_name] = [position.tolist(), orientation.tolist()]

    return body_pose_dict


class Pico4InputProvider:
    """Real-time input provider using Pico4 full-body tracking via xrobotoolkit_sdk.

    Implements the same interface as ``BVHInputProvider`` / ``UDPBVHInputProvider``
    so it plugs directly into the Teleopit pipeline.
    """

    def __init__(self, human_format: str = "xrobot", timeout: float = 60.0) -> None:
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise ImportError(
                "xrobotoolkit_sdk is required for Pico4 input. "
                "Install the Pico4 SDK or use a different input provider."
            ) from exc

        self._xrt = xrt
        self._xrt.init()
        self._human_format = human_format
        self._timeout = timeout
        self._closed = False
        self._frame_ready = threading.Event()
        self._lock = threading.Lock()
        self._previous_frame: HumanFrame | None = None
        self._latest_frame: HumanFrame | None = None
        self._previous_timestamp: float | None = None
        self._latest_timestamp: float | None = None
        self._frame_seq: int = 0
        self._last_raw_body_poses: NDArray[np.float64] | None = None
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="pico4_input")
        self._poll_thread.start()
        logger.info("Pico4InputProvider initialized (xrobotoolkit_sdk)")

    @property
    def fps(self) -> int:
        """Pico4 body tracking runs at ~30 fps."""
        return 30

    @property
    def human_format(self) -> str:
        return self._human_format

    def is_available(self) -> bool:
        """Whether the provider is healthy and can be polled.

        Returns ``True`` as long as the provider has not been closed.
        This matches ``UDPBVHInputProvider`` semantics where
        ``is_available()`` means "provider is alive" (not "data ready
        right now").  The actual blocking-until-data logic lives in
        ``get_frame()``.
        """
        return not self._closed and self._poll_thread.is_alive()

    def get_frame(self) -> HumanFrame:
        """Get the current body tracking frame.

        Blocks up to ``timeout`` seconds waiting for the first tracked
        frame, then returns the latest cached realtime frame.

        Returns:
            Dict mapping joint name to (position_3d, quaternion_wxyz) tuples
            after Teleopit's input-space transform.

        Raises:
            TimeoutError: If no body data arrives within the timeout.
        """
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No Pico4 body data received within {self._timeout:.1f}s timeout"
            )
        with self._lock:
            if self._latest_frame is None:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._latest_frame

    def get_frame_packet(self) -> tuple[HumanFrame, float, int]:
        """Return the latest cached frame together with timestamp and sequence."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No Pico4 body data received within {self._timeout:.1f}s timeout"
            )
        with self._lock:
            if self._latest_frame is None or self._latest_timestamp is None:
                raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
            return self._latest_frame, float(self._latest_timestamp), int(self._frame_seq)

    def sample_frame(self, query_time_s: float, delay_s: float) -> HumanFrame:
        """Sample a delayed interpolated realtime frame from the polling buffer."""
        if not self._frame_ready.wait(timeout=self._timeout):
            raise TimeoutError(
                f"No Pico4 body data received within {self._timeout:.1f}s timeout"
            )
        with self._lock:
            latest_frame = self._latest_frame
            latest_timestamp = self._latest_timestamp
            previous_frame = self._previous_frame
            previous_timestamp = self._previous_timestamp

        if latest_frame is None or latest_timestamp is None:
            raise RuntimeError("Pico4 frame buffer signaled ready without a latest frame")
        if (
            previous_frame is None
            or previous_timestamp is None
            or latest_timestamp <= previous_timestamp + 1e-6
        ):
            return latest_frame

        target_time = float(query_time_s - max(delay_s, 0.0))
        alpha = (target_time - previous_timestamp) / (latest_timestamp - previous_timestamp)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return interpolate_human_frames(previous_frame, latest_frame, alpha)

    def close(self) -> None:
        """Mark provider as closed."""
        self._closed = True
        self._poll_thread.join(timeout=1.0)
        logger.info("Pico4InputProvider closed")

    def _poll_loop(self) -> None:
        poll_sleep = 1.0 / max(float(self.fps), 1.0)
        while not self._closed:
            if not self._xrt.is_body_data_available():
                time.sleep(0.01)
                continue

            try:
                body_poses = np.asarray(self._xrt.get_body_joints_pose(), dtype=np.float64)
            except Exception:
                logger.exception("Failed to read Pico4 body data")
                time.sleep(0.05)
                continue

            if self._last_raw_body_poses is not None and np.array_equal(body_poses, self._last_raw_body_poses):
                time.sleep(0.005)
                continue

            frame = self._convert_body_poses_to_frame(body_poses)
            timestamp = time.monotonic()
            with self._lock:
                self._previous_frame = self._latest_frame
                self._previous_timestamp = self._latest_timestamp
                self._latest_frame = frame
                self._latest_timestamp = timestamp
                self._frame_seq += 1
                self._last_raw_body_poses = body_poses.copy()
            self._frame_ready.set()
            time.sleep(poll_sleep)

    @staticmethod
    def _convert_body_poses_to_frame(body_poses: NDArray[np.float64]) -> HumanFrame:
        body_pose_dict: dict[str, list] = {}
        for i, joint_name in enumerate(BODY_JOINT_NAMES):
            pos = [body_poses[i][0], body_poses[i][1], body_poses[i][2]]
            # SDK returns [x,y,z,qx,qy,qz,qw] — convert to scalar-first [w,x,y,z]
            rot = [body_poses[i][6], body_poses[i][3], body_poses[i][4], body_poses[i][5]]
            body_pose_dict[joint_name] = [pos, rot]

        body_pose_dict = _coordinate_transform_input(body_pose_dict)

        result: HumanFrame = {}
        for name, (pos, quat) in body_pose_dict.items():
            result[name] = (np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64))
        return result
