"""Sim2Real controller -- state machine + dual-mode control loop.

Supports two operating modes for a physical Unitree G1:
- **Gamepad**: LocoClient sends velocity commands (robot stays in "ai" mode)
- **Mocap**: Debug mode + direct LowCmd joint control via RL policy

Design based on unitreerobotics/xr_teleoperate patterns:
- LocoClient with fire-and-forget timeout (0.0001s) for velocity commands
- No SetFsmId() call -- robot in "ai" mode is already walk-ready
- Debug mode via MotionSwitcher.Enter_Debug_Mode() for low-level control
- 250Hz continuous LowCmd publishing thread (robot needs constant commands)
- Lock all joints to current position when entering debug mode
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from teleopit.controllers.observation import TWIST2ObservationBuilder
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.inputs.udp_bvh_provider import UDPBVHInputProvider
from teleopit.interfaces import RobotState
from teleopit.retargeting.core import RetargetingModule, extract_mimic_obs
from teleopit.sim2real.remote import UnitreeRemote
from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

logger = logging.getLogger(__name__)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        value = cfg.get(key)
        return default if value is None else value
    return getattr(cfg, key, default)


class RobotMode(Enum):
    DAMPING = "damping"
    PREPARATION = "preparation"
    STANDING = "standing"
    GAMEPAD = "gamepad"
    MOCAP = "mocap"


class Sim2RealController:
    """G1 real-robot controller -- gamepad/mocap dual mode with state machine.

    Gamepad mode: robot stays in "ai" mode, LocoClient sends velocity commands.
    Mocap mode: enter debug mode, direct LowCmd via RL policy at 250Hz.

    Based on xr_teleoperate:
    - LocoClient.SetTimeout(0.0001) -- fire-and-forget, no blocking
    - No SetFsmId() needed -- "ai" mode is already walk-ready
    - Debug mode via CheckMode/ReleaseMode loop for low-level control
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.mode = RobotMode.DAMPING

        self.policy_hz: float = float(_cfg_get(cfg, "policy_hz", 50.0))
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # ---- Real robot (SDK) ----
        real_cfg = _cfg_get(cfg, "real_robot")
        self.robot = UnitreeG1Robot(real_cfg)
        self.remote = UnitreeRemote()

        # ---- LocoClient (created in _enter_preparation) ----
        self._loco_client: Any = None

        # ---- Gamepad speed limits ----
        gp_cfg = _cfg_get(cfg, "gamepad", {})
        self._max_vx: float = float(_cfg_get(gp_cfg, "max_vx", 0.5))
        self._max_vy: float = float(_cfg_get(gp_cfg, "max_vy", 0.3))
        self._max_vyaw: float = float(_cfg_get(gp_cfg, "max_vyaw", 0.5))

        # ---- Mocap pipeline (reuse existing components) ----
        input_cfg = _cfg_get(cfg, "input")
        controller_cfg = _cfg_get(cfg, "controller")
        robot_cfg = _cfg_get(cfg, "robot")

        # Resolve reference BVH path
        ref_bvh = str(_cfg_get(input_cfg, "reference_bvh", ""))
        if ref_bvh and not Path(ref_bvh).is_absolute():
            ref_bvh = str((self._project_root / ref_bvh).resolve())

        self.udp_provider = UDPBVHInputProvider(
            reference_bvh=ref_bvh,
            host=str(_cfg_get(input_cfg, "udp_host", "")),
            port=int(_cfg_get(input_cfg, "udp_port", 1118)),
            human_format=str(_cfg_get(input_cfg, "bvh_format", "hc_mocap")),
            timeout=float(_cfg_get(input_cfg, "udp_timeout", 30.0)),
        )

        human_format = _cfg_get(input_cfg, "human_format", None)
        if not human_format or str(human_format) == "null":
            bvh_format = str(_cfg_get(input_cfg, "bvh_format", "hc_mocap"))
            human_format = f"bvh_{bvh_format}"

        self.retargeter = RetargetingModule(
            robot_name=str(_cfg_get(input_cfg, "robot_name", "unitree_g1")),
            human_format=str(human_format),
            actual_human_height=float(_cfg_get(input_cfg, "human_height", 1.75)),
        )

        # Ensure controller has default_dof_pos
        if _cfg_get(controller_cfg, "default_dof_pos", None) is None:
            default_angles = _cfg_get(robot_cfg, "default_angles", None)
            if default_angles is not None:
                if hasattr(controller_cfg, "__setattr__"):
                    controller_cfg.default_dof_pos = list(default_angles)
                elif isinstance(controller_cfg, dict):
                    controller_cfg["default_dof_pos"] = list(default_angles)

        # Resolve policy path
        policy_path = str(_cfg_get(controller_cfg, "policy_path", ""))
        if policy_path and not Path(policy_path).is_absolute():
            resolved = (self._project_root / policy_path).resolve()
            if resolved.exists():
                if hasattr(controller_cfg, "__setattr__"):
                    controller_cfg.policy_path = str(resolved)
                elif isinstance(controller_cfg, dict):
                    controller_cfg["policy_path"] = str(resolved)

        self.policy = RLPolicyController(controller_cfg)

        # ObservationBuilder
        obs_cfg = {
            "num_actions": int(_cfg_get(robot_cfg, "num_actions", 29)),
            "ang_vel_scale": float(_cfg_get(robot_cfg, "ang_vel_scale", 0.25)),
            "dof_pos_scale": float(_cfg_get(robot_cfg, "dof_pos_scale", 1.0)),
            "dof_vel_scale": float(_cfg_get(robot_cfg, "dof_vel_scale", 0.05)),
            "ankle_idx": list(_cfg_get(robot_cfg, "ankle_idx", [4, 5, 10, 11])),
            "default_dof_pos": list(_cfg_get(robot_cfg, "default_angles")),
        }
        self.obs_builder = TWIST2ObservationBuilder(obs_cfg)

        # Default standing pose (29-DOF)
        self.default_angles = np.asarray(
            _cfg_get(robot_cfg, "default_angles"), dtype=np.float32
        )
        self.num_actions: int = int(_cfg_get(robot_cfg, "num_actions", 29))

        # ---- Mocap mode state ----
        self._last_action: Float32Array = np.zeros(self.num_actions, dtype=np.float32)
        self._last_retarget_qpos: Float64Array | None = None

        # ---- Mocap switch safety ----
        mocap_sw = _cfg_get(cfg, "mocap_switch", {})
        self._check_frames: int = int(_cfg_get(mocap_sw, "check_frames", 10))
        self._max_pos_value: float = float(_cfg_get(mocap_sw, "max_position_value", 5.0))

        # ---- Startup: ensure "ai" mode is active ----
        self._ensure_ai_mode()

        logger.info(
            "Sim2RealController ready | policy_hz=%.0f | "
            "max_vx=%.1f max_vy=%.1f max_vyaw=%.1f",
            self.policy_hz, self._max_vx, self._max_vy, self._max_vyaw,
        )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _ensure_ai_mode(self) -> None:
        """Ensure robot is in 'ai' mode at startup.

        If the robot was left in debug/released state from a previous run,
        restore 'ai' mode and wait for the robot to initialize (stand up).
        """
        logger.info("=== Startup: checking robot mode ===")
        result = self.robot.check_mode()
        if result is None:
            logger.warning("Could not query motion mode")
            return

        mode_name = result.get("name", "") if isinstance(result, dict) else str(result)
        logger.info("Current motion mode: '%s'", mode_name)

        if mode_name:
            logger.info("Robot already in '%s' mode -- ready", mode_name)
            return

        # No active mode -- restore "ai" mode
        logger.warning("No active mode -- restoring 'ai' mode...")
        self.robot.exit_debug_mode()
        logger.info("Waiting 5s for robot to initialize and stand up...")
        time.sleep(5.0)

        # Verify
        result = self.robot.check_mode()
        if result is not None:
            mode_name = result.get("name", "") if isinstance(result, dict) else str(result)
            logger.info("Mode after restore: '%s'", mode_name)

    # ------------------------------------------------------------------
    # LocoClient management (fire-and-forget, matching xr_teleoperate)
    # ------------------------------------------------------------------

    def _create_loco_client(self) -> Any:
        """Create LocoClient with fire-and-forget timeout.

        xr_teleoperate uses SetTimeout(0.0001) -- commands are sent but
        responses are not waited for. This prevents blocking the control loop.
        """
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        client = LocoClient()
        client.SetTimeout(0.0001)  # Fire-and-forget (matching xr_teleoperate)
        client.Init()
        logger.info("LocoClient created (fire-and-forget, timeout=0.0001s)")
        return client

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop at policy_hz."""
        logger.info("Starting control loop -- press Start to begin")
        dt = 1.0 / self.policy_hz

        try:
            while True:
                t0 = time.monotonic()

                # 1. Read remote state
                ls = self.robot.get_lowstate()
                if ls is not None:
                    remote_bytes = bytes(ls.wireless_remote)
                    self.remote.update(remote_bytes)

                # 2. Emergency stop (highest priority)
                if self._check_emergency_stop():
                    if self.mode != RobotMode.DAMPING:
                        logger.warning("EMERGENCY STOP (L1+R1)")
                        self._enter_damping()
                    self._sleep_until(t0, dt)
                    continue

                # 3. Mode transitions
                self._handle_transitions()

                # 4. Execute current mode
                if self.mode == RobotMode.GAMEPAD:
                    self._gamepad_step()
                elif self.mode == RobotMode.MOCAP:
                    self._mocap_step()

                # 5. Rate control
                self._sleep_until(t0, dt)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt -- shutting down")

    # ------------------------------------------------------------------
    # Mode execution
    # ------------------------------------------------------------------

    def _gamepad_step(self) -> None:
        """Gamepad mode: read joystick -> LocoClient.Move(vx, vy, vyaw).

        LocoClient uses fire-and-forget timeout so Move() never blocks.
        The robot's "ai" mode handles the actual locomotion.
        """
        if self._loco_client is None:
            return

        vx = self.remote.ly * self._max_vx
        vy = self.remote.lx * self._max_vy
        vyaw = self.remote.rx * self._max_vyaw

        try:
            self._loco_client.Move(vx, vy, vyaw)
        except Exception:
            pass  # Fire-and-forget -- errors are expected (timeout)

    def _mocap_step(self) -> None:
        """Mocap mode: UDP BVH -> retarget -> policy -> update LowCmd targets.

        The 250Hz publish thread in UnitreeG1Robot handles actual publishing.
        This method only updates the position targets at policy_hz (50Hz).
        """
        if not self.udp_provider.is_available():
            logger.warning("UDP provider unavailable -- entering damping")
            self._enter_damping()
            return

        try:
            human_frame = self.udp_provider.get_frame()
        except TimeoutError:
            logger.warning("UDP timeout -- entering damping")
            self._enter_damping()
            return

        # Retarget -> mimic observation (35D)
        retargeted = self.retargeter.retarget(human_frame)
        qpos = self._retarget_to_qpos(retargeted)
        mimic_obs = extract_mimic_obs(
            qpos=qpos,
            last_qpos=self._last_retarget_qpos,
            dt=1.0 / self.policy_hz,
        )

        # Robot state from SDK
        robot_state = self.robot.get_state()

        # Build observation (1402D) -> policy inference
        obs = self.obs_builder.build(robot_state, mimic_obs, self._last_action)
        obs = self._adapt_observation_for_policy(obs)
        action = self.policy.compute_action(obs)
        target_dof_pos = self.policy.get_target_dof_pos(action)

        # Update position targets (250Hz thread handles publishing)
        self.robot.send_positions(target_dof_pos)

        # Update state
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_retarget_qpos = qpos.copy()

    # ------------------------------------------------------------------
    # State machine transitions
    # ------------------------------------------------------------------

    def _check_emergency_stop(self) -> bool:
        """L1 + R1 pressed simultaneously -> emergency stop."""
        return self.remote.LB.pressed and self.remote.RB.pressed

    def _enter_damping(self) -> None:
        """Enter damping mode from any state.

        From GAMEPAD: LocoClient.Damp() + destroy LocoClient
        From MOCAP: set_damping() via LowCmd + exit debug mode
        """
        if self.mode == RobotMode.MOCAP:
            # In debug mode with LowCmd publisher -- send damping
            logger.info("DAMPING: sending LowCmd damping (from MOCAP)...")
            self.robot.set_damping()
            time.sleep(0.5)
            # Restore "ai" mode
            logger.info("DAMPING: restoring 'ai' mode...")
            self.robot.exit_debug_mode()
            time.sleep(3.0)
        elif self._loco_client is not None:
            # In high-level mode -- use LocoClient to damp
            logger.info("DAMPING: LocoClient.Damp() (from GAMEPAD)...")
            try:
                self._loco_client.Damp()
            except Exception:
                pass

        self._loco_client = None
        self.mode = RobotMode.DAMPING
        logger.info("Mode -> DAMPING")

    def _handle_transitions(self) -> None:
        """Handle remote-triggered mode transitions."""
        if self.mode == RobotMode.DAMPING:
            if self.remote.start.on_pressed:
                logger.info("Start pressed -> PREPARATION")
                self._enter_preparation()

        elif self.mode == RobotMode.PREPARATION:
            if self.remote.A.on_pressed:
                logger.info("A pressed -> GAMEPAD")
                self.mode = RobotMode.STANDING

        elif self.mode == RobotMode.STANDING:
            self.mode = RobotMode.GAMEPAD
            logger.info("Mode -> GAMEPAD (LocoClient active, joystick controls walking)")

        elif self.mode == RobotMode.GAMEPAD:
            if self.remote.Y.on_pressed:
                if self._can_switch_to_mocap():
                    logger.info("Y pressed -> MOCAP")
                    self._transition_to_mocap()
                else:
                    logger.warning("Cannot switch to MOCAP -- UDP check failed")

        elif self.mode == RobotMode.MOCAP:
            if self.remote.X.on_pressed:
                logger.info("X pressed -> GAMEPAD (from MOCAP)")
                self._transition_to_gamepad()

    # ------------------------------------------------------------------
    # PREPARATION
    # ------------------------------------------------------------------

    def _enter_preparation(self) -> None:
        """DAMPING -> PREPARATION: ensure 'ai' mode, create LocoClient.

        Based on xr_teleoperate: no SetFsmId() needed. The robot in "ai"
        mode is already walk-ready. Just create LocoClient and start
        sending Move commands.
        """
        logger.info("=== PREPARATION ===")

        # Ensure "ai" mode is active
        result = self.robot.check_mode()
        mode_name = ""
        if result is not None:
            mode_name = result.get("name", "") if isinstance(result, dict) else str(result)
        logger.info("Current mode: '%s'", mode_name)

        if not mode_name:
            logger.info("No active mode -- restoring 'ai' mode...")
            self.robot.exit_debug_mode()
            logger.info("Waiting 5s for robot to stand up...")
            time.sleep(5.0)

            result = self.robot.check_mode()
            if result is not None:
                mode_name = result.get("name", "") if isinstance(result, dict) else str(result)
            if not mode_name:
                logger.error("Failed to restore 'ai' mode -- stay in DAMPING")
                return

        # Create LocoClient (fire-and-forget, matching xr_teleoperate)
        logger.info("Creating LocoClient...")
        try:
            self._loco_client = self._create_loco_client()
        except Exception as exc:
            logger.error("Failed to create LocoClient: %s", exc)
            return

        self.mode = RobotMode.PREPARATION
        logger.info("=== PREPARATION complete -- press A to enter GAMEPAD ===")

    # ------------------------------------------------------------------
    # Mode switching: gamepad <-> mocap
    # ------------------------------------------------------------------

    def _can_switch_to_mocap(self) -> bool:
        """Verify UDP signal is stable and values are reasonable."""
        if not self.udp_provider.is_available():
            logger.warning("Mocap check: UDP provider not available")
            return False

        if not self.udp_provider._frame_ready.is_set():
            logger.warning("Mocap check: no UDP data received yet")
            return False

        valid_count = 0
        for _ in range(self._check_frames + 5):
            try:
                frame = self.udp_provider.get_frame()
            except TimeoutError:
                return False

            all_valid = True
            for bone_name, (pos, quat) in frame.items():
                if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
                    all_valid = False
                    break
                if np.any(np.abs(pos) > self._max_pos_value):
                    all_valid = False
                    break
                if np.any(np.isnan(quat)) or np.any(np.isinf(quat)):
                    all_valid = False
                    break

            if all_valid:
                valid_count += 1
            else:
                valid_count = 0

            if valid_count >= self._check_frames:
                return True

            time.sleep(0.02)

        logger.warning("Mocap check: only %d/%d valid frames", valid_count, self._check_frames)
        return False

    def _transition_to_mocap(self) -> None:
        """Switch from gamepad -> mocap mode.

        1. Stop walking via LocoClient
        2. Enter debug mode (ReleaseMode loop)
        3. Create LowCmd publisher + lock all joints
        4. Read current state as initial retarget reference
        """
        # Stop walking
        if self._loco_client is not None:
            try:
                self._loco_client.StopMove()
            except Exception:
                pass
            time.sleep(0.5)
        self._loco_client = None

        # Enter debug mode (release "ai" -> low-level control)
        logger.info("Entering debug mode for low-level control...")
        self.robot.enter_debug_mode()
        time.sleep(0.5)

        # Lock all joints to current position (prevent collapse)
        logger.info("Locking joints to current position...")
        self.robot.lock_all_joints()
        time.sleep(0.5)

        # Read current state as initial
        state = self.robot.get_state()
        init_qpos = np.zeros(36, dtype=np.float64)
        init_qpos[3:7] = state.quat.astype(np.float64)
        init_qpos[7:36] = state.qpos.astype(np.float64)
        self._last_retarget_qpos = init_qpos
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)

        # Reset observation builder history
        self.obs_builder.reset()

        self.mode = RobotMode.MOCAP
        logger.info("Mode -> MOCAP (debug mode, 250Hz LowCmd publishing)")

    def _transition_to_gamepad(self) -> None:
        """Switch from mocap -> gamepad mode.

        1. Smooth interpolation to default standing pose (2s)
        2. Exit debug mode (SelectMode("ai"))
        3. Recreate LocoClient
        """
        logger.info("Smooth transition to default pose (2s)...")
        self._smooth_to_default_pose(duration=2.0)

        # Exit debug mode -> restore "ai" locomotion
        logger.info("Exiting debug mode -> restoring 'ai' mode...")
        self.robot.exit_debug_mode()
        time.sleep(3.0)

        # Recreate LocoClient
        logger.info("Recreating LocoClient...")
        try:
            self._loco_client = self._create_loco_client()
        except Exception as exc:
            logger.error("Failed to recreate LocoClient: %s", exc)
            self._loco_client = None

        self.mode = RobotMode.GAMEPAD
        logger.info("Mode -> GAMEPAD (LocoClient restored)")

    def _smooth_to_default_pose(self, duration: float = 2.0) -> None:
        """Linear interpolation from current joint positions to default standing pose."""
        state = self.robot.get_state()
        start_pos = state.qpos.copy().astype(np.float32)
        target_pos = self.default_angles.copy()
        steps = max(int(duration * self.policy_hz), 1)
        dt = 1.0 / self.policy_hz

        for i in range(steps):
            alpha = (i + 1) / steps
            interp = start_pos * (1.0 - alpha) + target_pos * alpha
            self.robot.send_positions(interp)
            time.sleep(dt)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retarget_to_qpos(self, retargeted: object) -> Float64Array:
        """Convert retarget output to 36D qpos (7D root + 29D joints)."""
        if isinstance(retargeted, tuple) and len(retargeted) == 3:
            base_pos = np.asarray(retargeted[0], dtype=np.float64).reshape(-1)
            base_rot = np.asarray(retargeted[1], dtype=np.float64).reshape(-1)
            joint_pos = np.asarray(retargeted[2], dtype=np.float64).reshape(-1)
            qpos = np.concatenate((base_pos, base_rot, joint_pos))
        else:
            qpos = np.asarray(retargeted, dtype=np.float64).reshape(-1)
        if qpos.shape[0] < 36:
            raise ValueError(f"Retargeted qpos too short: {qpos.shape[0]} (need >= 36)")
        return qpos

    def _adapt_observation_for_policy(self, obs: Float32Array) -> Float32Array:
        """Pad or truncate observation to match policy input dimension."""
        expected = getattr(self.policy, "_expected_obs_dim", None)
        if not isinstance(expected, int) or expected <= 0:
            return obs
        if obs.shape[0] == expected:
            return obs
        if obs.shape[0] > expected:
            return obs[:expected]
        pad = np.zeros(expected - obs.shape[0], dtype=np.float32)
        return np.concatenate((obs, pad), dtype=np.float32)

    @staticmethod
    def _sleep_until(t0: float, dt: float) -> None:
        """Sleep to maintain control frequency."""
        elapsed = time.monotonic() - t0
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean shutdown: restore 'ai' mode so robot is ready for next run."""
        logger.info("Shutting down Sim2RealController")
        # If in mocap/debug mode, restore "ai" mode
        if self.mode == RobotMode.MOCAP:
            try:
                self.robot.set_damping()
                time.sleep(0.5)
            except Exception:
                pass
        try:
            self.robot.exit_debug_mode()
        except Exception:
            pass
        try:
            self.udp_provider.close()
        except Exception:
            pass
        try:
            self.robot.close()
        except Exception:
            pass
