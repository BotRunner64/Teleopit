#!/usr/bin/env python3
"""Standalone G1 standing script -- no RL policy, no xrobotoolkit dependency.

Sends fixed standing-pose PD commands directly via Unitree SDK2 DDS.
Usage:
    python scripts/standalone_standing.py --network-interface enp130s0

Flow:
    1. Init DDS, subscribe to rt/lowstate
    2. Enter debug mode (release MotionSwitcher modes)
    3. Lock joints to current position
    4. Ramp from current position to standing pose over 2s
    5. Hold standing pose until Ctrl-C
    6. On exit: set damping, restore ai mode
"""

from __future__ import annotations

import argparse
import logging
import signal
import struct
import threading
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- G1 29-DOF Constants ----
NUM_JOINTS = 29
NUM_MOTORS = 35
MODE_PR = 0
MODE_MACHINE = 5
PUBLISH_HZ = 250
POS_STOP_F = 2146000000.0
VEL_STOP_F = 16000.0

# Default standing pose (from g1_constants.py HOME_KEYFRAME)
DEFAULT_ANGLES = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,       # Left leg (0-5)
    -0.312, 0, 0, 0.669, -0.363, 0,       # Right leg (6-11)
    0, 0, 0,                                # Waist (12-14)
    0.2, 0.2, 0, 0.6, 0, 0, 0,            # Left arm (15-21)
    0.2, -0.2, 0, 0.6, 0, 0, 0,           # Right arm (22-28)
], dtype=np.float32)

# PD gains (from mjlab g1_constants.py)
KP = np.array([
    40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
    40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
    40.2, 28.5, 28.5,
    14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
    14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
], dtype=np.float32)

KD = np.array([
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
    2.6, 1.8, 1.8,
    0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
    0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
], dtype=np.float32)

KD_DAMPING = 8.0
RAMP_DURATION = 2.0  # seconds to ramp from current to standing pose

# Identity joint map: logical joint i -> motor slot i
JOINT_MAP = list(range(NUM_JOINTS))

# ---- Wireless remote button parsing (for L1+R1 emergency stop) ----
# Byte offsets in the 40-byte wireless_remote payload
_KEYS_OFFSET = 2  # uint16 at byte 2


def _parse_keys(remote_bytes: bytes) -> int:
    """Extract 16-bit key bitmask from 40-byte wireless remote payload."""
    if len(remote_bytes) < 4:
        return 0
    return struct.unpack_from("<H", remote_bytes, _KEYS_OFFSET)[0]


# Button masks (matching Unitree wireless remote)
_R1 = 0x0001
_L1 = 0x0002


class StandingController:
    """Minimal controller: ramp to standing pose and hold."""

    def __init__(self, network_interface: str, ramp_duration: float = RAMP_DURATION) -> None:
        self._network_interface = network_interface
        self._ramp_duration = ramp_duration
        self._shutdown = False

        # ---- SDK imports ----
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            LowCmd_ as HG_LowCmd,
            LowState_ as HG_LowState,
        )
        from unitree_sdk2py.utils.crc import CRC

        self._LowCmd_Factory = unitree_hg_msg_dds__LowCmd_
        self._crc = CRC()

        # ---- DDS init ----
        ChannelFactoryInitialize(0, self._network_interface)

        # Subscriber: rt/lowstate
        self._lowstate = None
        self._state_sub = ChannelSubscriber("rt/lowstate", HG_LowState)
        self._state_sub.Init(self._on_lowstate, 10)

        # Publisher: rt/lowcmd
        self._cmd_pub = ChannelPublisher("rt/lowcmd", HG_LowCmd)
        self._cmd_pub.Init()

        # Build default LowCmd message
        self._cmd = self._LowCmd_Factory()
        self._cmd.mode_pr = MODE_PR
        self._cmd.mode_machine = MODE_MACHINE
        self._cmd.level_flag = 0xFF

        # Init all motor slots
        for i in range(NUM_MOTORS):
            self._cmd.motor_cmd[i].mode = 0x01
            if i < NUM_JOINTS:
                self._cmd.motor_cmd[i].q = 0.0
                self._cmd.motor_cmd[i].kp = 0.0
                self._cmd.motor_cmd[i].dq = 0.0
                self._cmd.motor_cmd[i].kd = KD_DAMPING
                self._cmd.motor_cmd[i].tau = 0.0
            else:
                self._cmd.motor_cmd[i].q = POS_STOP_F
                self._cmd.motor_cmd[i].kp = 0.0
                self._cmd.motor_cmd[i].dq = VEL_STOP_F
                self._cmd.motor_cmd[i].kd = 0.0
                self._cmd.motor_cmd[i].tau = 0.0

        self._cmd_lock = threading.Lock()
        self._publish_thread = None
        self._publish_running = False

        # Motion switcher (lazy)
        self._motion_switcher = None

        # Wait for first state
        logger.info("Waiting for LowState on %s ...", self._network_interface)
        deadline = time.monotonic() + 5.0
        while self._lowstate is None and time.monotonic() < deadline:
            time.sleep(0.05)

        if self._lowstate is None:
            raise RuntimeError("No LowState received within 5s -- check network and robot power")
        logger.info("LowState received, robot connected")

    def _on_lowstate(self, msg) -> None:
        self._lowstate = msg

    # ---- 250Hz publish thread ----

    def _start_publish(self) -> None:
        if self._publish_thread is not None:
            return
        self._publish_running = True
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

    def _stop_publish(self) -> None:
        self._publish_running = False
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=1.0)
            self._publish_thread = None

    def _publish_loop(self) -> None:
        dt = 1.0 / PUBLISH_HZ
        while self._publish_running:
            t0 = time.monotonic()
            with self._cmd_lock:
                self._cmd.mode_machine = MODE_MACHINE
                self._cmd.crc = self._crc.Crc(self._cmd)
                self._cmd_pub.Write(self._cmd)
            elapsed = time.monotonic() - t0
            if (dt - elapsed) > 0:
                time.sleep(dt - elapsed)

    # ---- Motion switcher ----

    def _get_motion_switcher(self):
        if self._motion_switcher is not None:
            return self._motion_switcher
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        client = MotionSwitcherClient()
        client.SetTimeout(1.0)
        client.Init()
        self._motion_switcher = client
        return client

    def enter_debug_mode(self) -> bool:
        """Release all MotionSwitcher modes to enable rt/lowcmd control."""
        try:
            client = self._get_motion_switcher()
            for _ in range(10):
                code, result = client.CheckMode()
                mode_name = result.get("name", "") if isinstance(result, dict) else ""
                if not mode_name:
                    logger.info("Debug mode ready (no active mode)")
                    return True
                logger.info("Releasing mode: %s", mode_name)
                client.ReleaseMode()
                time.sleep(1)
            logger.warning("Could not fully release modes after 10 attempts")
            return False
        except Exception as exc:
            logger.error("enter_debug_mode failed: %s", exc)
            return False

    def exit_debug_mode(self) -> bool:
        """Restore ai mode via MotionSwitcher."""
        self._stop_publish()
        try:
            client = self._get_motion_switcher()
            code, result = client.SelectMode("ai")
            logger.info("SelectMode('ai') -> code=%s", code)
            return code == 0
        except Exception as exc:
            logger.error("exit_debug_mode failed: %s", exc)
            return False

    # ---- Motor commands ----

    def _get_current_qpos(self) -> np.ndarray:
        """Read current joint positions from LowState."""
        ls = self._lowstate
        qpos = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            qpos[i] = ls.motor_state[JOINT_MAP[i]].q
        return qpos

    def _send_positions(self, target: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        """Update motor commands for all 29 joints."""
        with self._cmd_lock:
            for i in range(NUM_JOINTS):
                idx = JOINT_MAP[i]
                self._cmd.motor_cmd[idx].mode = 1
                self._cmd.motor_cmd[idx].q = float(target[i])
                self._cmd.motor_cmd[idx].kp = float(kp[i])
                self._cmd.motor_cmd[idx].dq = 0.0
                self._cmd.motor_cmd[idx].kd = float(kd[i])
                self._cmd.motor_cmd[idx].tau = 0.0

    def _set_damping(self) -> None:
        """All motors to damping mode."""
        with self._cmd_lock:
            for i in range(NUM_MOTORS):
                self._cmd.motor_cmd[i].mode = 1
                self._cmd.motor_cmd[i].q = 0.0
                self._cmd.motor_cmd[i].kp = 0.0
                self._cmd.motor_cmd[i].dq = 0.0
                self._cmd.motor_cmd[i].kd = KD_DAMPING
                self._cmd.motor_cmd[i].tau = 0.0

    def _lock_joints(self) -> None:
        """Lock all joints to current position."""
        ls = self._lowstate
        with self._cmd_lock:
            for i in range(NUM_JOINTS):
                idx = JOINT_MAP[i]
                self._cmd.motor_cmd[idx].mode = 1
                self._cmd.motor_cmd[idx].q = ls.motor_state[idx].q
                self._cmd.motor_cmd[idx].kp = float(KP[i])
                self._cmd.motor_cmd[idx].dq = 0.0
                self._cmd.motor_cmd[idx].kd = float(KD[i])
                self._cmd.motor_cmd[idx].tau = 0.0

    def _check_emergency_stop(self) -> bool:
        """Check if L1+R1 are both pressed on the wireless remote."""
        ls = self._lowstate
        if ls is None:
            return False
        try:
            remote_bytes = bytes(ls.wireless_remote)
            keys = _parse_keys(remote_bytes)
            return bool(keys & _R1) and bool(keys & _L1)
        except Exception:
            return False

    # ---- Main loop ----

    def run(self) -> None:
        """Enter debug mode, ramp to standing, hold until Ctrl-C or L1+R1."""
        # Register signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # 1. Enter debug mode
            logger.info("Entering debug mode ...")
            if not self.enter_debug_mode():
                logger.error("Failed to enter debug mode, aborting")
                return

            # 2. Start 250Hz publish thread
            self._start_publish()

            # 3. Lock joints to current position first
            logger.info("Locking joints to current position ...")
            self._lock_joints()
            time.sleep(0.5)  # Brief hold to stabilize

            # 4. Ramp from current position to standing pose
            logger.info("Ramping to standing pose over %.1fs ...", self._ramp_duration)
            start_qpos = self._get_current_qpos()
            ramp_start = time.monotonic()
            dt = 1.0 / 50.0  # 50Hz control loop (matches policy_hz)

            while not self._shutdown:
                t0 = time.monotonic()
                elapsed = t0 - ramp_start

                # Emergency stop check
                if self._check_emergency_stop():
                    logger.warning("L1+R1 pressed -- emergency damping!")
                    self._set_damping()
                    time.sleep(1.0)
                    break

                # Compute interpolation alpha
                if elapsed < self._ramp_duration:
                    alpha = elapsed / self._ramp_duration
                    # Smooth cubic interpolation (ease in/out)
                    alpha = 3 * alpha**2 - 2 * alpha**3
                    target = start_qpos + alpha * (DEFAULT_ANGLES - start_qpos)
                else:
                    target = DEFAULT_ANGLES

                self._send_positions(target, KP, KD)

                # Sleep to maintain 50Hz
                sleep_time = dt - (time.monotonic() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if not self._shutdown:
                return  # Emergency stop triggered, cleanup below

            # 5. Hold standing (already at default angles) until shutdown
            # (We land here after ramp if no emergency stop, but _shutdown was set)

        except Exception as exc:
            logger.error("Error in main loop: %s", exc)
        finally:
            self._cleanup()

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Shutdown signal received")
        self._shutdown = True

    def _cleanup(self) -> None:
        """Safely shut down: damping -> restore ai mode."""
        logger.info("Shutting down: setting damping ...")
        self._set_damping()
        time.sleep(0.5)

        logger.info("Restoring ai mode ...")
        self.exit_debug_mode()
        logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="G1 standalone standing (no RL policy)")
    parser.add_argument(
        "--network-interface", type=str, default="eth0",
        help="Network interface for DDS (e.g. eth0, enp130s0)",
    )
    parser.add_argument(
        "--ramp-duration", type=float, default=RAMP_DURATION,
        help="Seconds to ramp from current to standing pose (default: 2.0)",
    )
    args = parser.parse_args()

    controller = StandingController(
        network_interface=args.network_interface,
        ramp_duration=args.ramp_duration,
    )
    controller.run()


if __name__ == "__main__":
    main()
