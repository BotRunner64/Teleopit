"""Unitree G1 robot interface via g1_bridge_sdk (C++ DDS bridge).

Wraps g1_bridge_sdk.G1Bridge to provide a RobotState-compatible API
for the sim2real controller. All realtime DDS communication happens
in native C++ threads; Python only reads/writes buffers.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from teleopit.constants import NUM_JOINTS
from teleopit.interfaces import RobotState
from teleopit.runtime.common import cfg_get

logger = logging.getLogger(__name__)


class UnitreeG1Robot:
    """Control a physical G1 robot via g1_bridge_sdk (C++ DDS bridge).

    Two operating paradigms:
    - High-level (ai mode): LocoClient handles walking, no LowCmd publishing
    - Low-level (debug mode): C++ bridge publishes LowCmd at 200Hz
    """

    def __init__(self, cfg: Any) -> None:
        self._network_interface: str = str(cfg_get(cfg, "network_interface", "eth0"))
        self._kp = np.asarray(cfg_get(cfg, "kp_real", [100] * NUM_JOINTS), dtype=np.float32)
        self._kd = np.asarray(cfg_get(cfg, "kd_real", [2] * NUM_JOINTS), dtype=np.float32)

        if self._kp.shape[0] != NUM_JOINTS:
            raise ValueError(f"kp_real must have {NUM_JOINTS} entries")
        if self._kd.shape[0] != NUM_JOINTS:
            raise ValueError(f"kd_real must have {NUM_JOINTS} entries")

        import g1_bridge_sdk

        self._bridge = g1_bridge_sdk.G1Bridge(self._network_interface)
        self._publishing: bool = False

        if not self._bridge.wait_for_state(3.0):
            logger.warning("No LowState received within 3s -- robot may not be connected")
        else:
            logger.info("UnitreeG1Robot: state received")

        logger.info(
            "UnitreeG1Robot initialised on %s via C++ bridge",
            self._network_interface,
        )

    # ------------------------------------------------------------------
    # Public API: state reading
    # ------------------------------------------------------------------

    def get_state(self) -> RobotState:
        """Read latest LowState -> RobotState(qpos[29], qvel[29], quat[4], ang_vel[3])."""
        qpos, qvel, quat, ang_vel = self._bridge.get_state()
        return RobotState(
            qpos=qpos,
            qvel=qvel,
            quat=quat,
            ang_vel=ang_vel,
            timestamp=time.time(),
        )

    def get_wireless_remote(self) -> bytes:
        """Return 40-byte wireless remote data."""
        return self._bridge.get_wireless_remote()

    # ------------------------------------------------------------------
    # Public API: low-level commands
    # ------------------------------------------------------------------

    def _ensure_publishing(self) -> None:
        """Start the C++ publish thread if not already running."""
        if not self._publishing:
            self._bridge.start_publish()
            self._publishing = True
            logger.info("C++ bridge publish thread started")

    def lock_all_joints(self) -> None:
        """Lock ALL joints to current position with PD gains."""
        self._ensure_publishing()

        qpos, _, _, _ = self._bridge.get_state()
        self._bridge.set_target(qpos, self._kp, self._kd)
        logger.info("All %d joints locked to current position", NUM_JOINTS)

    def send_positions(
        self,
        target_pos: np.ndarray,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        """Update 29-DOF position targets (published by C++ thread)."""
        self._ensure_publishing()

        if target_pos.shape[0] != NUM_JOINTS:
            raise ValueError(f"target_pos must have {NUM_JOINTS} entries")

        if kp is None:
            kp = self._kp
        if kd is None:
            kd = self._kd

        self._bridge.set_target(
            np.asarray(target_pos, dtype=np.float32),
            np.asarray(kp, dtype=np.float32),
            np.asarray(kd, dtype=np.float32),
        )

    def set_damping(self) -> None:
        """Set all motors to damping mode (kp=0, kd=8.0)."""
        self._ensure_publishing()
        self._bridge.set_damping()

    # ------------------------------------------------------------------
    # motion_switcher: high/low level switching
    # ------------------------------------------------------------------

    def enter_debug_mode(self) -> bool:
        """Enter debug mode: release all modes until none active."""
        try:
            for attempt in range(10):
                code, name = self._bridge.check_mode()
                logger.info("enter_debug_mode: check_mode -> code=%s, name=%s", code, name)
                if code != 0:
                    logger.error("enter_debug_mode: check_mode RPC failed with code=%s", code)
                    return False
                if not name:
                    logger.info("enter_debug_mode: no active mode -- debug mode ready")
                    return True
                release_code = self._bridge.release_mode()
                logger.info("enter_debug_mode: release_mode('%s') -> code=%s", name, release_code)
                if release_code != 0:
                    logger.error(
                        "enter_debug_mode: failed to release active mode '%s' (code=%s)",
                        name,
                        release_code,
                    )
                    return False
                time.sleep(1)
            logger.warning("enter_debug_mode: could not fully release after 10 attempts")
            return False
        except Exception as exc:
            logger.error("enter_debug_mode failed: %s", exc)
            return False

    def exit_debug_mode(self) -> bool:
        """Exit debug mode: stop publishing then SelectMode('ai')."""
        self._bridge.stop_publish()
        self._publishing = False

        try:
            code = self._bridge.select_mode("ai")
            logger.info("exit_debug_mode: select_mode('ai') -> code=%s", code)
            return code == 0
        except Exception as exc:
            logger.error("exit_debug_mode failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up."""
        self._bridge.stop_publish()
        self._publishing = False
        logger.info("UnitreeG1Robot closed")
