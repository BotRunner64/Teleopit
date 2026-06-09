#!/usr/bin/env python3
"""Standalone G1 standing script with RL policy -- no Teleopit/Pico dependency.

Uses ONNX RL policy inference to maintain balanced standing, matching the
STANDING mode used by the sim2real robot-control runtime. Only depends on:
  - g1_bridge_sdk (C++ DDS bridge)
  - onnxruntime
  - mujoco
  - numpy

Usage:
    python scripts/run/standalone_standing.py \
        --policy track.onnx \
        --network-interface enp130s0

Flow:
    1. Init DDS, subscribe to rt/lowstate
    2. Load ONNX policy + MuJoCo model for observation building
    3. Enter debug mode (release MotionSwitcher modes)
    4. Lock joints, then run RL policy standing loop
    5. Hold standing until Ctrl-C or L1+R1
    6. On exit: set damping, restore ai mode
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

import mujoco
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---- G1 29-DOF Constants ----
NUM_JOINTS = 29
NUM_MOTORS = 35
MODE_PR = 0
MODE_MACHINE = 5
PUBLISH_HZ = 200
POLICY_HZ = 50.0
POS_STOP_F = 2146000000.0
VEL_STOP_F = 16000.0
KD_DAMPING = 8.0
JOINT_VEL_LIMIT = 10.0
DEFAULT_KP_RAMP_DURATION = 2.0
DEFAULT_KP_RAMP_FLOOR_RATIO = 0.1

# Default standing pose (from g1_constants.py HOME_KEYFRAME)
DEFAULT_ANGLES = np.array([
    -0.312, 0, 0, 0.669, -0.363, 0,       # Left leg (0-5)
    -0.312, 0, 0, 0.669, -0.363, 0,       # Right leg (6-11)
    0, 0, 0,                                # Waist (12-14)
    0.2, 0.2, 0, 0.6, 0, 0, 0,            # Left arm (15-21)
    0.2, -0.2, 0, 0.6, 0, 0, 0,           # Right arm (22-28)
], dtype=np.float32)

# Action scale (from g1.yaml)
ACTION_SCALE = np.array([
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.3507, 0.5475, 0.3507, 0.4386, 0.4386,
    0.5475, 0.4386, 0.4386,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
    0.4386, 0.4386, 0.4386, 0.4386, 0.4386, 0.0745, 0.0745,
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

# Joint position limits (from pico4_sim2real.yaml)
JOINT_POS_LOWER = np.array([
    -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618,
    -2.618, -0.52, -0.52,
    -3.0892, -1.5882, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443,
    -3.0892, -2.2515, -2.618, -1.0472, -1.972222054, -1.61443, -1.61443,
], dtype=np.float32)

JOINT_POS_UPPER = np.array([
    2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618,
    2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618,
    2.618, 0.52, 0.52,
    2.6704, 2.2515, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443,
    2.6704, 1.5882, 2.618, 2.0944, 1.972222054, 1.61443, 1.61443,
], dtype=np.float32)

# MuJoCo XML path for FK
MJCF_PATH = _REPO_ROOT / "teleopit" / "retargeting" / "gmr" / "assets" / "unitree_g1" / "g1_mjlab.xml"

JOINT_MAP = list(range(NUM_JOINTS))
GRAVITY_UNIT_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# Wireless remote button masks
_KEYS_OFFSET = 2
_R1 = 0x0001
_L1 = 0x0002


# =====================================================================
# Quaternion helpers (from teleopit.controllers.observation)
# =====================================================================

def quat_inv(q):
    inv = q.copy()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1).astype(np.float32)


def quat_rotate(q, v):
    v_quat = np.zeros((*v.shape[:-1], 4), dtype=np.float32)
    v_quat[..., 1:4] = v
    result = quat_mul(quat_mul(q, v_quat), quat_inv(q))
    return result[..., 1:4]


def yaw_quat(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    out = np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float32)
    out /= max(float(np.linalg.norm(out)), 1e-8)
    return out


def align_motion_qpos_yaw(robot_quat_wxyz, motion_qpos):
    """Align motion_qpos[3:7] quaternion yaw to match robot yaw."""
    motion_quat = np.asarray(motion_qpos[3:7], dtype=np.float32)
    robot_quat = np.asarray(robot_quat_wxyz, dtype=np.float32)
    delta = quat_mul(robot_quat, quat_inv(motion_quat))
    delta_yaw = yaw_quat(delta)
    motion_qpos[3:7] = quat_mul(delta_yaw, motion_quat).astype(motion_qpos.dtype)
    return motion_qpos


def quat_to_rot6d(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    r00 = 1 - 2 * (y*y + z*z)
    r01 = 2 * (x*y - w*z)
    r10 = 2 * (x*y + w*z)
    r11 = 1 - 2 * (x*x + z*z)
    r20 = 2 * (x*z - w*y)
    r21 = 2 * (y*z + w*x)
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float32)


# =====================================================================
# Observation builder (from teleopit.controllers.observation)
# =====================================================================

class ObservationBuilder:
    """166D VelCmd observation builder using MuJoCo FK."""

    def __init__(self, xml_path: str):
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._mj_data = mujoco.MjData(self._mj_model)

        # Find anchor body (torso_link)
        body_names = {self._mj_model.body(i).name: i for i in range(self._mj_model.nbody)}
        self._anchor_body_id = body_names["torso_link"]

        # Base obs: command(29+29) + anchor_ori_b(6) + ang_vel(3) + joint_pos_rel(29) + qvel(29) + last_act(29)
        # = 154
        # VelCmd extra: projected_gravity(3) + ref_lin_vel_b(3) + ref_ang_vel_b(3) + ref_proj_gravity(3)
        # = 12
        # Total = 166
        self.total_obs_size = NUM_JOINTS * 2 + 6 + 3 + NUM_JOINTS * 3 + 12

        # Precompute motion torso offset for standing (DEFAULT_ANGLES with identity base)
        # torso_quat_world = quat_mul(base_quat, torso_offset) for constant joint angles
        self._run_fk(np.zeros(3), np.array([1, 0, 0, 0], dtype=np.float32), DEFAULT_ANGLES)
        self._standing_torso_offset = self._get_body_quat(self._anchor_body_id).copy()

    def _run_fk(self, base_pos, base_quat, joint_pos):
        self._mj_data.qpos[:] = 0.0
        self._mj_data.qpos[0:3] = np.asarray(base_pos, dtype=np.float64).reshape(3)
        quat = np.asarray(base_quat, dtype=np.float64).reshape(4)
        quat = quat / max(np.linalg.norm(quat), 1e-8)
        self._mj_data.qpos[3:7] = quat
        n = min(len(joint_pos), self._mj_model.nq - 7)
        self._mj_data.qpos[7:7 + n] = np.asarray(joint_pos, dtype=np.float64)[:n]
        mujoco.mj_kinematics(self._mj_model, self._mj_data)

    def _get_body_quat(self, body_id):
        return np.asarray(self._mj_data.xquat[body_id], dtype=np.float32).copy()

    def build(self, robot_qpos, robot_qvel, robot_quat, robot_ang_vel,
              motion_qpos, motion_joint_vel, last_action):
        """Build 166D observation for VelCmd policy.

        Args:
            robot_qpos: (29,) current joint positions
            robot_qvel: (29,) current joint velocities
            robot_quat: (4,) base orientation quaternion (w,x,y,z)
            robot_ang_vel: (3,) base angular velocity
            motion_qpos: (36,) reference motion [pos(3) + quat(4) + joints(29)]
            motion_joint_vel: (29,) reference joint velocity
            last_action: (29,) previous policy action
        """
        qpos = np.asarray(robot_qpos, dtype=np.float32)[:NUM_JOINTS]
        qvel = np.asarray(robot_qvel, dtype=np.float32)[:NUM_JOINTS]
        robot_q = np.asarray(robot_quat, dtype=np.float32)
        ang_vel = np.asarray(robot_ang_vel, dtype=np.float32)
        motion = np.asarray(motion_qpos, dtype=np.float32)
        m_joint_vel = np.asarray(motion_joint_vel, dtype=np.float32)[:NUM_JOINTS]
        last_act = np.asarray(last_action, dtype=np.float32)

        # Anchor quaternions: skip MuJoCo FK, use base quat * precomputed offset
        # For standing, waist joints ≈ 0 so torso ≈ base orientation
        robot_anchor_quat = quat_mul(robot_q, self._standing_torso_offset)

        motion_base_quat = motion[3:7]
        motion_joint_pos = motion[7:7 + NUM_JOINTS]
        motion_anchor_quat = quat_mul(motion_base_quat, self._standing_torso_offset)

        # Base observation (154D)
        command = np.concatenate((motion_joint_pos, m_joint_vel), dtype=np.float32)
        rel_quat = quat_mul(quat_inv(robot_anchor_quat), motion_anchor_quat)
        motion_anchor_ori_b = quat_to_rot6d(rel_quat)
        joint_pos_rel = qpos - DEFAULT_ANGLES

        base_obs = np.concatenate([
            command,            # 29 + 29 = 58
            motion_anchor_ori_b,  # 6
            ang_vel,            # 3
            joint_pos_rel,      # 29
            qvel,               # 29
            last_act,           # 29
        ], dtype=np.float32)

        # VelCmd extra (12D) -- standing has zero reference velocities
        projected_gravity = quat_rotate(quat_inv(robot_q), GRAVITY_UNIT_W)
        robot_inv = quat_inv(robot_anchor_quat)
        # Zero reference velocities for standing
        ref_lin_vel_b = np.zeros(3, dtype=np.float32)
        ref_ang_vel_b = np.zeros(3, dtype=np.float32)
        ref_proj_gravity = quat_rotate(quat_inv(motion_anchor_quat), GRAVITY_UNIT_W)

        velcmd_obs = np.concatenate([
            projected_gravity,      # 3
            ref_lin_vel_b,          # 3
            ref_ang_vel_b,          # 3
            ref_proj_gravity,       # 3
        ], dtype=np.float32)

        obs = np.concatenate([base_obs, velcmd_obs], dtype=np.float32)
        return obs


# =====================================================================
# ONNX Policy wrapper (from teleopit.controllers.rl_policy)
# =====================================================================

class PolicyInference:
    """Minimal ONNX policy inference wrapper."""

    def __init__(self, policy_path: str):
        providers = ["CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")

        self._session = ort.InferenceSession(policy_path, providers=providers)
        onnx_inputs = self._session.get_inputs()
        self._input_name = onnx_inputs[0].name
        self._output_name = self._session.get_outputs()[0].name

        # Check for dual-input (obs + obs_history) model
        self._multi_input = False
        self._history_length = 0
        self._history_buf: deque[np.ndarray] = deque()
        if len(onnx_inputs) == 2 and onnx_inputs[1].name == "obs_history":
            self._multi_input = True
            self._history_length = int(onnx_inputs[1].shape[1])
            self._history_buf = deque(maxlen=self._history_length)

        # Extract expected obs dim
        self._expected_obs_dim = None
        if len(onnx_inputs[0].shape) >= 2 and isinstance(onnx_inputs[0].shape[-1], int):
            self._expected_obs_dim = int(onnx_inputs[0].shape[-1])

        logger.info(
            "Policy loaded: input=%s, obs_dim=%s, multi_input=%s, history_len=%d",
            self._input_name, self._expected_obs_dim, self._multi_input, self._history_length,
        )

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        obs_flat = obs.reshape(-1)

        if self._multi_input:
            if len(self._history_buf) == 0:
                for _ in range(self._history_length):
                    self._history_buf.append(obs_flat.copy())
            else:
                self._history_buf.append(obs_flat.copy())
            obs_history = np.stack(list(self._history_buf), axis=0)[np.newaxis].astype(np.float32)
            feed = {self._input_name: obs, "obs_history": obs_history}
        else:
            feed = {self._input_name: obs}

        raw_action = np.asarray(
            self._session.run([self._output_name], feed)[0], dtype=np.float32
        ).reshape(-1)
        return raw_action

    def get_target_dof_pos(self, raw_action: np.ndarray) -> np.ndarray:
        clipped = np.clip(raw_action, -10.0, 10.0)
        scaled = clipped * ACTION_SCALE
        return scaled + DEFAULT_ANGLES

    def reset(self):
        self._history_buf.clear()


# =====================================================================
# Main controller
# =====================================================================

class StandingController:
    """RL-policy-based standing controller matching sim2real STANDING behavior."""

    def __init__(self, network_interface: str, policy_path: str,
                 no_policy: bool = False,
                 publish_hz: int = 250,
                 obs_delay: float = 0.0,
                 command_delay: float = 0.0,
                 kp_ramp_duration: float = DEFAULT_KP_RAMP_DURATION,
                 kp_ramp_floor_ratio: float = DEFAULT_KP_RAMP_FLOOR_RATIO) -> None:
        self._network_interface = network_interface
        self._shutdown = False
        if obs_delay < 0.0:
            raise ValueError("obs_delay must be >= 0")
        if command_delay < 0.0:
            raise ValueError("command_delay must be >= 0")
        if kp_ramp_duration < 0.0:
            raise ValueError("kp_ramp_duration must be >= 0")
        if not 0.0 <= kp_ramp_floor_ratio <= 1.0:
            raise ValueError("kp_ramp_floor_ratio must be in [0, 1]")

        # ---- Load policy and observation builder ----
        self._policy = PolicyInference(policy_path)
        xml_path = str(MJCF_PATH)
        if not MJCF_PATH.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        self._obs_builder = ObservationBuilder(xml_path)

        # Verify observation dimension matches policy expectation
        if (self._policy._expected_obs_dim is not None and
                self._policy._expected_obs_dim != self._obs_builder.total_obs_size):
            raise ValueError(
                f"Obs dimension mismatch: builder={self._obs_builder.total_obs_size}, "
                f"policy={self._policy._expected_obs_dim}"
            )

        self._no_policy = no_policy
        self._dry_run = False
        self._state_delay = 0.0
        self._obs_delay = float(obs_delay)
        self._command_delay = float(command_delay)
        self._state_history: deque[tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = deque(maxlen=512)
        self._state_history_lock = threading.Lock()
        self._state_sampler_thread: threading.Thread | None = None
        self._state_sampler_running = False
        self._pending_targets: deque[tuple[float, np.ndarray]] = deque()
        self._pending_targets_cv = threading.Condition()
        self._command_sender_thread: threading.Thread | None = None
        self._command_sender_running = False
        self._last_obs_age_s = 0.0
        self._last_command_queue_len = 0

        # ---- Policy state ----
        self._step_count = 0
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        # Standing reference qpos: [pos(3), quat(4), joints(29)] = 36D
        self._standing_qpos = np.zeros(36, dtype=np.float64)
        self._standing_qpos[3] = 1.0  # identity quaternion w=1
        self._standing_qpos[7:36] = DEFAULT_ANGLES.astype(np.float64)

        # ---- Pipeline state ----
        self._inference_thread: threading.Thread | None = None
        self._inference_running = False

        self._publish_hz = publish_hz
        self._kp_ramp_duration_steps = max(1, int(kp_ramp_duration * POLICY_HZ))
        self._kp_ramp_floor_ratio = float(kp_ramp_floor_ratio)
        self._kp_ramp_step = 0
        self._kp_ramp_active = False

        self._init_cpp_backend()

    # ==================================================================
    # Backend init
    # ==================================================================

    def _init_cpp_backend(self) -> None:
        import g1_bridge_sdk
        logger.info("Using C++ bridge backend (%dHz publish)", self._publish_hz)
        self._bridge = g1_bridge_sdk.G1Bridge(self._network_interface, self._publish_hz)

        logger.info("Waiting for LowState on %s ...", self._network_interface)
        if not self._bridge.wait_for_state(5.0):
            raise RuntimeError("No LowState received within 5s -- check network and robot power")
        logger.info("LowState received, robot connected")

    # ==================================================================
    # Robot state reading
    # ==================================================================

    def _get_robot_state(self):
        return self._read_robot_state()

    def _read_robot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        qpos, qvel, quat, ang_vel = self._bridge.get_state()
        return (
            np.asarray(qpos, dtype=np.float32).copy(),
            np.asarray(qvel, dtype=np.float32).copy(),
            np.asarray(quat, dtype=np.float32).copy(),
            np.asarray(ang_vel, dtype=np.float32).copy(),
        )

    def _record_robot_state(self) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        now = time.monotonic()
        state = self._read_robot_state()
        with self._state_history_lock:
            self._state_history.append((now, state))
        return now, state

    def _get_observation_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        now, current = self._record_robot_state()
        if self._obs_delay <= 0.0:
            self._last_obs_age_s = 0.0
            return current

        target_time = now - self._obs_delay
        with self._state_history_lock:
            history = list(self._state_history)
        selected_time, selected_state = history[0]
        for sample_time, sample_state in reversed(history):
            if sample_time <= target_time:
                selected_time, selected_state = sample_time, sample_state
                break
        self._last_obs_age_s = max(0.0, now - selected_time)
        return selected_state

    def _start_state_sampler(self) -> None:
        if self._obs_delay <= 0.0 or self._state_sampler_thread is not None:
            return
        self._state_sampler_running = True
        self._state_sampler_thread = threading.Thread(target=self._state_sampler_loop, daemon=True)
        self._state_sampler_thread.start()

    def _stop_state_sampler(self) -> None:
        self._state_sampler_running = False
        if self._state_sampler_thread is not None:
            self._state_sampler_thread.join(timeout=1.0)
            self._state_sampler_thread = None

    def _state_sampler_loop(self) -> None:
        sample_dt = 1.0 / max(float(self._publish_hz), POLICY_HZ)
        while self._state_sampler_running and not self._shutdown:
            self._record_robot_state()
            time.sleep(sample_dt)

    # ==================================================================
    # Publish thread
    # ==================================================================

    def _start_publish(self) -> None:
        self._bridge.start_publish()

    def _stop_publish(self) -> None:
        self._bridge.stop_publish()

    # ==================================================================
    # Motion switcher
    # ==================================================================

    def enter_debug_mode(self) -> bool:
        try:
            for _ in range(10):
                code, name = self._bridge.check_mode()
                logger.info("check_mode -> code=%s, name=%s", code, name)
                if code != 0:
                    logger.error("check_mode RPC failed with code=%s", code)
                    return False
                if not name:
                    logger.info("Debug mode ready (no active mode)")
                    return True
                logger.info("Releasing mode: %s", name)
                release_code = self._bridge.release_mode()
                logger.info("release_mode('%s') -> code=%s", name, release_code)
                if release_code != 0:
                    logger.error("Failed to release mode '%s' (code=%s)", name, release_code)
                    return False
                time.sleep(1)
            logger.warning("Could not release modes after 10 attempts")
            return False
        except Exception as exc:
            logger.error("enter_debug_mode failed: %s", exc)
            return False

    def exit_debug_mode(self) -> bool:
        self._stop_publish()
        try:
            code = self._bridge.select_mode("ai")
            logger.info("select_mode('ai') -> code=%s", code)
            return code == 0
        except Exception as exc:
            logger.error("exit_debug_mode failed: %s", exc)
            return False

    # ==================================================================
    # Motor commands
    # ==================================================================

    def _set_damping(self) -> None:
        self._bridge.set_damping()

    def _lock_joints(self) -> None:
        qpos, _, _, _ = self._get_robot_state()
        self._bridge.set_target(qpos, KP, KD)
        self._bridge.lock_joints()

    def _start_kp_ramp(self) -> None:
        self._kp_ramp_step = 0
        self._kp_ramp_active = True
        logger.info(
            "Kp ramp armed: %d steps (%.1fs), floor_ratio=%.2f",
            self._kp_ramp_duration_steps,
            self._kp_ramp_duration_steps / POLICY_HZ,
            self._kp_ramp_floor_ratio,
        )

    def _compute_kp_ramp_gains(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._kp_ramp_active:
            return None

        factor = min(1.0, self._kp_ramp_step / self._kp_ramp_duration_steps)
        kp = KP * (self._kp_ramp_floor_ratio + (1.0 - self._kp_ramp_floor_ratio) * factor)

        self._kp_ramp_step += 1
        if self._kp_ramp_step >= self._kp_ramp_duration_steps:
            self._kp_ramp_active = False
            logger.info("Kp ramp complete (%d steps)", self._kp_ramp_duration_steps)

        return np.asarray(kp, dtype=np.float32), KD.copy()

    def _write_target_now(self, target: np.ndarray) -> None:
        gains = self._compute_kp_ramp_gains()
        if gains is None:
            self._bridge.set_target(target, KP, KD)
            return
        kp, kd = gains
        self._bridge.set_target(target, kp, kd)

    def _pop_due_targets(self, now: float) -> list[np.ndarray]:
        due: list[np.ndarray] = []
        with self._pending_targets_cv:
            while self._pending_targets and self._pending_targets[0][0] <= now:
                _, target = self._pending_targets.popleft()
                due.append(target)
            self._last_command_queue_len = len(self._pending_targets)
        return due

    def _flush_pending_targets(self, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        due = self._pop_due_targets(now)
        for target in due:
            self._write_target_now(target)
        if len(due) > 1:
            logger.warning("Flushed %d delayed targets in one control tick", len(due))

    def _send_target(self, target: np.ndarray) -> None:
        if self._command_delay <= 0.0:
            self._write_target_now(target)
            self._last_command_queue_len = len(self._pending_targets)
            return
        with self._pending_targets_cv:
            self._pending_targets.append((time.monotonic() + self._command_delay, np.asarray(target, dtype=np.float32).copy()))
            self._last_command_queue_len = len(self._pending_targets)
            self._pending_targets_cv.notify()

    def _start_command_sender(self) -> None:
        if self._command_delay <= 0.0 or self._command_sender_thread is not None:
            return
        self._command_sender_running = True
        self._command_sender_thread = threading.Thread(target=self._command_sender_loop, daemon=True)
        self._command_sender_thread.start()

    def _stop_command_sender(self) -> None:
        self._command_sender_running = False
        with self._pending_targets_cv:
            self._pending_targets_cv.notify_all()
        if self._command_sender_thread is not None:
            self._command_sender_thread.join(timeout=1.0)
            self._command_sender_thread = None
        with self._pending_targets_cv:
            self._pending_targets.clear()
            self._last_command_queue_len = 0

    def _command_sender_loop(self) -> None:
        while self._command_sender_running and not self._shutdown:
            now = time.monotonic()
            due = self._pop_due_targets(now)
            if due:
                self._write_target_now(due[-1])
                if len(due) > 1:
                    logger.warning("Dropped %d stale delayed targets", len(due) - 1)
                continue

            with self._pending_targets_cv:
                if not self._pending_targets:
                    self._pending_targets_cv.wait(timeout=0.02)
                    continue
                wait_s = max(0.0, self._pending_targets[0][0] - time.monotonic())
                self._pending_targets_cv.wait(timeout=min(wait_s, 0.02))

    # ==================================================================
    # Safety checks
    # ==================================================================

    def _check_emergency_stop(self) -> bool:
        try:
            remote_bytes = self._bridge.get_wireless_remote()
            if len(remote_bytes) < 4:
                return False
            keys = struct.unpack_from("<H", remote_bytes, _KEYS_OFFSET)[0]
            return bool(keys & _R1) and bool(keys & _L1)
        except Exception:
            return False

    def _check_joint_vel_safety(self, qvel: np.ndarray) -> bool:
        max_vel = np.max(np.abs(qvel))
        if max_vel > JOINT_VEL_LIMIT:
            logger.error("SAFETY: joint vel %.2f rad/s > limit %.2f -- damping!", max_vel, JOINT_VEL_LIMIT)
            return True
        return False

    # ---- Standing step (matches sim2real robot-control standing step) ----

    def _standing_step(self) -> np.ndarray:
        """One step of RL policy standing inference. Returns target joint positions."""
        _t0 = time.monotonic()
        qpos, qvel, quat, ang_vel = self._get_observation_state()

        # Build standing reference aligned to robot's current yaw
        ref_qpos = self._standing_qpos.copy()
        align_motion_qpos_yaw(quat, ref_qpos)

        # Zero joint velocity reference for standing
        motion_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        motion_qpos = np.asarray(ref_qpos[:7 + NUM_JOINTS], dtype=np.float32)

        _t1 = time.monotonic()
        # Build observation
        obs = self._obs_builder.build(
            robot_qpos=qpos,
            robot_qvel=qvel,
            robot_quat=quat,
            robot_ang_vel=ang_vel,
            motion_qpos=ref_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
        )

        _t2 = time.monotonic()
        # Policy inference
        action = self._policy.compute_action(obs)

        target_dof_pos = self._policy.get_target_dof_pos(action)
        _t3 = time.monotonic()

        # Diagnostic
        self._step_count += 1
        step_ms = (_t3 - _t0) * 1000
        if self._step_count % 25 == 1 or step_ms > (1000.0 / POLICY_HZ):
            tag = "OVERRUN" if step_ms > (1000.0 / POLICY_HZ) else "DIAG"
            logger.info(
                "%s step=%d | state=%.2fms obs=%.2fms infer=%.2fms total=%.1fms | "
                "obs_age=%.1fms cmd_q=%d | qvel_norm=%.4f | action_norm=%.4f | "
                "target[:6]=%s | qpos[:6]=%s",
                tag, self._step_count,
                (_t1 - _t0) * 1000, (_t2 - _t1) * 1000, (_t3 - _t2) * 1000,
                step_ms,
                self._last_obs_age_s * 1000,
                self._last_command_queue_len,
                float(np.linalg.norm(qvel)),
                float(np.linalg.norm(action)),
                np.array2string(target_dof_pos[:6], precision=4, separator=','),
                np.array2string(qpos[:6], precision=4, separator=','),
            )

        # Joint limits
        target_dof_pos = np.clip(target_dof_pos, JOINT_POS_LOWER, JOINT_POS_UPPER)

        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        return target_dof_pos

    # ---- Inference thread ----

    def _start_inference(self) -> None:
        if self._inference_thread is not None:
            return
        self._inference_running = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def _stop_inference(self) -> None:
        self._inference_running = False
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2.0)
            self._inference_thread = None

    def _inference_loop(self) -> None:
        """~50Hz soft-realtime inference loop.

        Runs policy inference and writes target positions to _target_buf.
        The 250Hz publish thread reads _target_buf independently — even if
        inference takes 30ms, the robot keeps receiving 250Hz commands with
        the last good target.
        """
        dt = 1.0 / POLICY_HZ
        loop_count = 0
        overrun_count = 0
        max_elapsed = 0.0
        elapsed_sum = 0.0

        while self._inference_running and not self._shutdown:
            t0 = time.monotonic()
            if self._command_delay <= 0.0:
                self._flush_pending_targets(t0)

            # Emergency stop check
            if self._check_emergency_stop():
                logger.warning("L1+R1 pressed -- emergency damping!")
                self._set_damping()
                self._shutdown = True
                break

            # Joint velocity safety check (warning only, no shutdown)
            _, qvel, _, _ = self._get_robot_state()
            self._check_joint_vel_safety(qvel)

            # Artificial state delay (for debugging)
            if self._state_delay > 0:
                time.sleep(self._state_delay)

            # Policy step
            if self._no_policy:
                target = np.clip(DEFAULT_ANGLES.copy(), JOINT_POS_LOWER, JOINT_POS_UPPER)
            else:
                target = self._standing_step()

            # Write target to publish thread. Kp ramps after standing entry;
            # policy targets stay unchanged, matching sim2real STANDING.
            self._send_target(target)

            # Timing diagnostics (informational only — not a control failure)
            elapsed = time.monotonic() - t0
            loop_count += 1
            elapsed_sum += elapsed
            max_elapsed = max(max_elapsed, elapsed)
            if elapsed > dt:
                overrun_count += 1

            if loop_count % 50 == 0:
                avg_ms = (elapsed_sum / loop_count) * 1000
                logger.info(
                    "Inference stats: avg=%.1fms, max=%.1fms, overruns=%d/%d (target=%.1fms)",
                    avg_ms, max_elapsed * 1000, overrun_count, loop_count, dt * 1000,
                )

            remain = dt - (time.monotonic() - t0)
            if remain > 0:
                time.sleep(remain)

        if self._command_delay <= 0.0:
            self._flush_pending_targets()

    # ---- Main loop ----

    def _run_dry(self) -> None:
        """Dry-run: read state + build obs + infer, no motor commands. Safe for timing tests."""
        logger.info("=== DRY-RUN MODE: no motor commands will be sent ===")
        self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._policy.reset()

        dt = 1.0 / POLICY_HZ
        loop_count = 0
        overrun_count = 0
        elapsed_sum = 0.0
        max_elapsed = 0.0
        state_sum = 0.0
        obs_sum = 0.0
        infer_sum = 0.0

        self._start_state_sampler()
        while not self._shutdown:
            t0 = time.monotonic()

            # 1. Read state
            qpos, qvel, quat, ang_vel = self._get_observation_state()
            t1 = time.monotonic()

            # 2. Build reference
            ref_qpos = self._standing_qpos.copy()
            align_motion_qpos_yaw(quat, ref_qpos)
            motion_joint_vel = np.zeros(NUM_JOINTS, dtype=np.float32)

            # 3. Build observation
            obs = self._obs_builder.build(
                robot_qpos=qpos, robot_qvel=qvel,
                robot_quat=quat, robot_ang_vel=ang_vel,
                motion_qpos=ref_qpos, motion_joint_vel=motion_joint_vel,
                last_action=self._last_action,
            )
            t2 = time.monotonic()

            # 4. Policy inference
            raw_action = self._policy.compute_action(obs)
            target = self._policy.get_target_dof_pos(raw_action)
            t3 = time.monotonic()

            self._last_action = raw_action.copy()
            loop_count += 1
            e = t3 - t0
            elapsed_sum += e
            max_elapsed = max(max_elapsed, e)
            state_sum += (t1 - t0)
            obs_sum += (t2 - t1)
            infer_sum += (t3 - t2)
            if e > dt:
                overrun_count += 1

            if loop_count % 50 == 0:
                n = loop_count
                logger.info(
                    "DRY step=%d | state=%.2fms obs=%.2fms infer=%.2fms total=%.2fms | "
                    "max=%.2fms overruns=%d/%d | obs_age=%.1fms | target[:6]=%s",
                    n,
                    (state_sum / n) * 1000, (obs_sum / n) * 1000,
                    (infer_sum / n) * 1000, (elapsed_sum / n) * 1000,
                    max_elapsed * 1000, overrun_count, n,
                    self._last_obs_age_s * 1000,
                    np.array2string(target[:6], precision=4, separator=','),
                )

            remain = dt - (time.monotonic() - t0)
            if remain > 0:
                time.sleep(remain)

        self._stop_state_sampler()
        logger.info(
            "DRY-RUN finished: %d steps, avg total=%.2fms "
            "(state=%.2f obs=%.2f infer=%.2f) max=%.2fms overruns=%d",
            loop_count,
            (elapsed_sum / max(loop_count, 1)) * 1000,
            (state_sum / max(loop_count, 1)) * 1000,
            (obs_sum / max(loop_count, 1)) * 1000,
            (infer_sum / max(loop_count, 1)) * 1000,
            max_elapsed * 1000, overrun_count,
        )

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if self._dry_run:
            self._run_dry()
            return

        try:
            # 1. Enter debug mode
            logger.info("Entering debug mode ...")
            if not self.enter_debug_mode():
                logger.error("Failed to enter debug mode, aborting")
                return
            time.sleep(0.5)

            # 2. Start publish thread (C++: 500Hz, Python: 250Hz)
            self._start_publish()

            # 3. Lock joints to current position
            logger.info("Locking joints to current position ...")
            self._lock_joints()
            time.sleep(0.3)

            # 4. ONNX warmup — eliminate first-inference spike
            logger.info("Warming up ONNX runtime ...")
            dummy_obs = np.zeros(self._obs_builder.total_obs_size, dtype=np.float32)
            for _ in range(3):
                self._policy.compute_action(dummy_obs)
            self._policy.reset()
            logger.info("ONNX warmup complete")

            # 5. Initialize policy state
            self._last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
            self._start_kp_ramp()
            self._start_state_sampler()
            self._start_command_sender()

            logger.info("Starting RL policy standing (pipelined)")

            # 6. Start inference thread (~50Hz, soft deadline)
            self._start_inference()

            # 7. Main thread waits for shutdown signal
            while not self._shutdown:
                time.sleep(0.1)

            # 8. Stop inference thread
            self._stop_inference()

        except Exception as exc:
            logger.error("Error in main loop: %s", exc)
        finally:
            self._cleanup()

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Shutdown signal received")
        self._shutdown = True

    def _cleanup(self) -> None:
        self._inference_running = False
        self._stop_state_sampler()
        self._stop_command_sender()
        logger.info("Shutting down: setting damping ...")
        self._set_damping()
        time.sleep(0.5)
        logger.info("Stopping publish and restoring ai mode ...")
        self.exit_debug_mode()
        logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="G1 standalone standing with RL policy")
    parser.add_argument(
        "--policy", type=str, required=True,
        help="Path to ONNX policy file (e.g. track.onnx)",
    )
    parser.add_argument(
        "--network-interface", type=str, default="eth0",
        help="Network interface for DDS (e.g. eth0, enp130s0)",
    )
    parser.add_argument(
        "--no-policy", action="store_true",
        help="Skip RL policy, just send fixed DEFAULT_ANGLES (diagnostic mode)",
    )
    parser.add_argument(
        "--state-delay", type=float, default=0.0,
        help=(
            "Legacy loop delay before the policy step. This consumes timing budget but does not make "
            "the observation stale; prefer --obs-delay or --command-delay for latency tests."
        ),
    )
    parser.add_argument(
        "--obs-delay", type=float, default=0.0,
        help="Use LowState sampled this many seconds in the past when building the policy observation.",
    )
    parser.add_argument(
        "--command-delay", type=float, default=0.0,
        help="Delay writing each computed target to the C++ publish thread by this many seconds.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read state + build obs + infer only, no motor commands (safe timing test)",
    )
    parser.add_argument(
        "--publish-hz", type=int, default=200,
        help="C++ publish frequency in Hz (default: 200, matching training pd_hz)",
    )
    parser.add_argument(
        "--kp-ramp-duration", type=float, default=DEFAULT_KP_RAMP_DURATION,
        help="Seconds to ramp Kp after entering standing (default: 2.0, matches sim2real)",
    )
    parser.add_argument(
        "--kp-ramp-floor-ratio", type=float, default=DEFAULT_KP_RAMP_FLOOR_RATIO,
        help="Initial Kp ratio for the standing ramp (default: 0.1, matches sim2real)",
    )
    args = parser.parse_args()

    controller = StandingController(
        network_interface=args.network_interface,
        policy_path=args.policy,
        no_policy=args.no_policy,
        publish_hz=args.publish_hz,
        obs_delay=args.obs_delay,
        command_delay=args.command_delay,
        kp_ramp_duration=args.kp_ramp_duration,
        kp_ramp_floor_ratio=args.kp_ramp_floor_ratio,
    )
    controller._state_delay = args.state_delay
    controller._dry_run = args.dry_run
    controller.run()


if __name__ == "__main__":
    main()
