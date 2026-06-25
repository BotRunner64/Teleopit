#!/usr/bin/env python3
"""Standalone G1 standing controller using the sim2real standing path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from teleopit.constants import FULL_QPOS_DIM, NUM_JOINTS, ROOT_DIM
from teleopit.controllers.observation import align_motion_qpos_yaw
from teleopit.controllers.rl_policy import RLPolicyController
from teleopit.runtime.common import cfg_get
from teleopit.runtime.factory import _build_policy_components, build_simulation_cfg
from teleopit.sim.reference_timeline import ReferenceWindowBuilder
from teleopit.sim.reference_utils import build_static_reference_window, obs_builder_requires_reference_window
from teleopit.sim2real.reference_processor import Sim2RealReferenceProcessor
from teleopit.sim2real.remote import UnitreeRemote
from teleopit.sim2real.safety import Sim2RealSafetyManager
from teleopit.sim2real.unitree_g1 import UnitreeG1Robot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_HZ = 50.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StepTiming:
    state_s: float
    obs_s: float
    infer_s: float
    target_s: float
    send_s: float
    action_delta: float | None
    target_delta: float | None
    qvel_norm: float
    ang_vel_norm: float


class _StandaloneTimingReporter:
    def __init__(
        self,
        *,
        target_period_s: float,
        log_interval_s: float = 1.0,
        deadline_miss_tolerance_s: float = 0.001,
        enabled: bool = True,
    ) -> None:
        self._target_period_s = float(target_period_s)
        self._log_interval_s = float(log_interval_s)
        self._deadline_miss_tolerance_s = float(deadline_miss_tolerance_s)
        self._enabled = bool(enabled)
        self._window_start_s: float | None = None
        self._loop_ms: list[float] = []
        self._late_ms: list[float] = []
        self._work_ms: list[float] = []
        self._state_ms: list[float] = []
        self._obs_ms: list[float] = []
        self._infer_ms: list[float] = []
        self._target_ms: list[float] = []
        self._send_ms: list[float] = []
        self._action_delta: list[float] = []
        self._target_delta: list[float] = []
        self._qvel_norm: list[float] = []
        self._ang_vel_norm: list[float] = []
        self._deadline_miss_count = 0
        self._work_overrun_count = 0

    def record(
        self,
        *,
        loop_start_s: float,
        work_elapsed_s: float,
        cycle_elapsed_s: float,
        step: _StepTiming,
    ) -> None:
        if self._window_start_s is None:
            self._window_start_s = float(loop_start_s)
        self._loop_ms.append(float(cycle_elapsed_s) * 1000.0)
        self._late_ms.append(max(0.0, float(cycle_elapsed_s) - self._target_period_s) * 1000.0)
        self._work_ms.append(float(work_elapsed_s) * 1000.0)
        self._state_ms.append(float(step.state_s) * 1000.0)
        self._obs_ms.append(float(step.obs_s) * 1000.0)
        self._infer_ms.append(float(step.infer_s) * 1000.0)
        self._target_ms.append(float(step.target_s) * 1000.0)
        self._send_ms.append(float(step.send_s) * 1000.0)
        if step.action_delta is not None:
            self._action_delta.append(float(step.action_delta))
        if step.target_delta is not None:
            self._target_delta.append(float(step.target_delta))
        self._qvel_norm.append(float(step.qvel_norm))
        self._ang_vel_norm.append(float(step.ang_vel_norm))
        if cycle_elapsed_s > self._target_period_s + self._deadline_miss_tolerance_s:
            self._deadline_miss_count += 1
        if work_elapsed_s > self._target_period_s + 1e-9:
            self._work_overrun_count += 1
        if loop_start_s - self._window_start_s >= self._log_interval_s:
            self._emit(loop_start_s)

    def _emit(self, end_s: float) -> None:
        sample_count = len(self._loop_ms)
        if sample_count <= 0:
            self._reset(end_s)
            return
        if not self._enabled:
            self._reset(end_s)
            return
        logger.info(
            "Standalone timing | samples=%d window=%.1fs | "
            "loop_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "late_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f deadline_miss(>%.2fms)=%d/%d | "
            "work_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f work_overrun=%d/%d | "
            "state_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "obs_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "infer_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "target_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "send_ms p50=%.2f p95=%.2f p99=%.2f max=%.2f | "
            "action_delta p50=%.4f p95=%.4f p99=%.4f max=%.4f | "
            "target_delta p50=%.4f p95=%.4f p99=%.4f max=%.4f | "
            "qvel_norm p50=%.4f p95=%.4f p99=%.4f max=%.4f | "
            "ang_vel_norm p50=%.4f p95=%.4f p99=%.4f max=%.4f",
            sample_count,
            end_s - float(self._window_start_s),
            *self._summarize(self._loop_ms),
            *self._summarize(self._late_ms),
            self._deadline_miss_tolerance_s * 1000.0,
            self._deadline_miss_count,
            sample_count,
            *self._summarize(self._work_ms),
            self._work_overrun_count,
            sample_count,
            *self._summarize(self._state_ms),
            *self._summarize(self._obs_ms),
            *self._summarize(self._infer_ms),
            *self._summarize(self._target_ms),
            *self._summarize(self._send_ms),
            *self._summarize(self._action_delta),
            *self._summarize(self._target_delta),
            *self._summarize(self._qvel_norm),
            *self._summarize(self._ang_vel_norm),
        )
        self._reset(end_s)

    def _reset(self, window_start_s: float) -> None:
        self._window_start_s = float(window_start_s)
        self._loop_ms.clear()
        self._late_ms.clear()
        self._work_ms.clear()
        self._state_ms.clear()
        self._obs_ms.clear()
        self._infer_ms.clear()
        self._target_ms.clear()
        self._send_ms.clear()
        self._action_delta.clear()
        self._target_delta.clear()
        self._qvel_norm.clear()
        self._ang_vel_norm.clear()
        self._deadline_miss_count = 0
        self._work_overrun_count = 0

    @staticmethod
    def _summarize(samples: list[float]) -> tuple[float, float, float, float]:
        if not samples:
            return 0.0, 0.0, 0.0, 0.0
        values = np.asarray(samples, dtype=np.float64)
        p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
        return float(p50), float(p95), float(p99), float(np.max(values))


class StandaloneStandingController:
    """Small wrapper around the production sim2real STANDING implementation."""

    def __init__(
        self,
        cfg: Any,
        *,
        dry_run: bool = False,
        no_policy: bool = False,
        obs_delay_s: float = 0.0,
        command_delay_s: float = 0.0,
    ) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.no_policy = no_policy
        self.obs_delay_s = self._validate_delay_s(obs_delay_s, name="obs_delay_s")
        self.command_delay_s = self._validate_delay_s(command_delay_s, name="command_delay_s")
        self.shutdown_requested = False

        self.policy_hz = float(cfg_get(cfg, "policy_hz", DEFAULT_POLICY_HZ))
        self.dt = 1.0 / self.policy_hz
        self.robot_cfg = cfg_get(cfg, "robot")
        self.real_robot_cfg = cfg_get(cfg, "real_robot")
        self.default_angles = np.asarray(cfg_get(self.robot_cfg, "default_angles"), dtype=np.float32)
        self.num_actions = int(cfg_get(self.robot_cfg, "num_actions", NUM_JOINTS))

        default_root_qpos = np.asarray(
            cfg_get(self.robot_cfg, "mujoco_default_qpos", [0.0, 0.0, 0.0]), dtype=np.float64
        ).reshape(-1)
        self._default_root_pos = np.zeros(3, dtype=np.float64)
        if default_root_qpos.shape[0] >= 3:
            self._default_root_pos[:] = default_root_qpos[:3]

        self.policy, self.obs_builder = self._build_policy_and_obs()
        self.robot = UnitreeG1Robot(self.real_robot_cfg)
        self.remote = UnitreeRemote()
        self.safety = Sim2RealSafetyManager(cfg, self.robot, self.policy_hz, self.num_actions)
        self._entered_debug = False

        sim_cfg = build_simulation_cfg(cfg)
        self.ref_proc = Sim2RealReferenceProcessor(
            obs_builder=self.obs_builder,
            policy=self.policy,
            policy_hz=self.policy_hz,
            num_actions=self.num_actions,
            reference_velocity_smoothing_alpha=float(sim_cfg["reference_velocity_smoothing_alpha"]),
            reference_anchor_velocity_smoothing_alpha=float(sim_cfg["reference_anchor_velocity_smoothing_alpha"]),
        )
        self.reference_window_builder = ReferenceWindowBuilder(
            policy_dt_s=self.dt,
            reference_steps=cfg_get(cfg, "reference_steps", [0]),
        )

        self._standing_qpos = np.zeros(FULL_QPOS_DIM, dtype=np.float64)
        self._standing_qpos[3] = 1.0
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self._last_target: np.ndarray | None = None
        self._step_count = 0
        console_cfg = cfg_get(cfg, "console", {}) or {}
        self._timing = _StandaloneTimingReporter(
            target_period_s=self.dt,
            log_interval_s=float(cfg_get(console_cfg, "timing_log_interval_s", 10.0)),
            enabled=bool(cfg_get(console_cfg, "show_timing", False)),
        )

        if self.obs_delay_s > 0.0 or self.command_delay_s > 0.0:
            logger.info(
                "Diagnostic delay injection enabled | obs_delay=%.3fms command_delay=%.3fms",
                self.obs_delay_s * 1000.0,
                self.command_delay_s * 1000.0,
            )

    @staticmethod
    def _validate_delay_s(value: float, *, name: str) -> float:
        delay_s = float(value)
        if not np.isfinite(delay_s) or delay_s < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        return delay_s

    def _build_policy_and_obs(self) -> tuple[Any, Any]:
        controller_cfg = cfg_get(self.cfg, "controller")
        policy, obs_builder = _build_policy_components(
            robot_cfg=self.robot_cfg,
            controller_cfg=controller_cfg,
            sim_cfg=build_simulation_cfg(self.cfg),
            project_root=PROJECT_ROOT,
            controller_cls=RLPolicyController,
        )
        if not bool(getattr(policy, "_multi_input", False)):
            raise ValueError("Standalone standing requires a dual-input ONNX policy ('obs', 'obs_history').")
        return policy, obs_builder

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            if self.dry_run:
                self._run_dry()
                return
            self._enter_standing()
            self._run_control_loop()
        finally:
            self._cleanup()

    def _enter_standing(self) -> None:
        logger.info("Entering debug mode...")
        if not self.robot.enter_debug_mode():
            raise RuntimeError("Failed to enter debug mode")
        self._entered_debug = True
        time.sleep(0.5)

        logger.info("Locking joints to current position...")
        self.robot.lock_all_joints()
        time.sleep(0.3)

        self._warmup_policy()
        state = self.robot.get_state()
        self._set_default_standing_reference(state)
        self._reset_policy_state()
        self.safety.start_kp_ramp()
        logger.info("Mode -> STANDING (standalone)")

    def _run_control_loop(self) -> None:
        while not self.shutdown_requested:
            t0 = time.monotonic()
            self.remote.update(self.robot.get_wireless_remote())
            if self.remote.LB.pressed and self.remote.RB.pressed:
                logger.warning("L1+R1 pressed -- damping")
                self.shutdown_requested = True
                break
            if self.safety.check_joint_velocity_safety():
                self.robot.set_damping()
                self.shutdown_requested = True
                break

            step_timing = self._standing_step()
            work_elapsed_s = time.monotonic() - t0
            cycle_elapsed_s = self._sleep_until(t0)
            self._timing.record(
                loop_start_s=t0,
                work_elapsed_s=work_elapsed_s,
                cycle_elapsed_s=cycle_elapsed_s,
                step=step_timing,
            )

    def _standing_step(self) -> _StepTiming:
        t0 = time.monotonic()
        robot_state = self.robot.get_state()
        t_state = time.monotonic()
        if self.obs_delay_s > 0.0:
            time.sleep(self.obs_delay_s)
        qpos = self._standing_qpos.copy()
        motion_joint_vel = np.zeros(self.num_actions, dtype=np.float32)
        motion_qpos = np.asarray(qpos[: ROOT_DIM + self.num_actions], dtype=np.float32)
        reference_window = None
        if obs_builder_requires_reference_window(self.obs_builder):
            reference_window = build_static_reference_window(qpos, self.reference_window_builder, self.policy_hz)
        obs = self.ref_proc.build_observation(
            robot_state=robot_state,
            motion_qpos=motion_qpos,
            motion_joint_vel=motion_joint_vel,
            last_action=self._last_action,
            anchor_lin_vel_w=np.zeros(3, dtype=np.float32),
            anchor_ang_vel_w=np.zeros(3, dtype=np.float32),
            reference_window=reference_window,
        )
        obs = self.ref_proc.validate_observation(obs)
        t_obs = time.monotonic()
        action = np.zeros(self.num_actions, dtype=np.float32) if self.no_policy else self.policy.compute_action(obs)
        t_infer = time.monotonic()
        target_dof_pos = self.safety.clip_to_joint_limits(self.policy.get_target_dof_pos(action))
        t_target = time.monotonic()
        action_delta = float(np.linalg.norm(action - self._last_action))
        target_delta = None if self._last_target is None else float(np.linalg.norm(target_dof_pos - self._last_target))
        if self.dry_run:
            self._log_step(robot_state.qvel, action, target_dof_pos, dry=True)
        else:
            if self.command_delay_s > 0.0:
                time.sleep(self.command_delay_s)
            self.safety.send_positions(target_dof_pos)
            self._log_step(robot_state.qvel, action, target_dof_pos, dry=False)
        t_send = time.monotonic()
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._last_target = np.asarray(target_dof_pos, dtype=np.float32).reshape(-1).copy()
        return _StepTiming(
            state_s=t_state - t0,
            obs_s=t_obs - t_state,
            infer_s=t_infer - t_obs,
            target_s=t_target - t_infer,
            send_s=t_send - t_target,
            action_delta=action_delta,
            target_delta=target_delta,
            qvel_norm=float(np.linalg.norm(robot_state.qvel)),
            ang_vel_norm=float(np.linalg.norm(robot_state.ang_vel)),
        )

    def _run_dry(self) -> None:
        logger.info("=== DRY-RUN MODE: no motor commands will be sent ===")
        self._warmup_policy()
        state = self.robot.get_state()
        self._set_default_standing_reference(state)
        self._reset_policy_state()
        while not self.shutdown_requested:
            t0 = time.monotonic()
            step_timing = self._standing_step()
            work_elapsed_s = time.monotonic() - t0
            cycle_elapsed_s = self._sleep_until(t0)
            self._timing.record(
                loop_start_s=t0,
                work_elapsed_s=work_elapsed_s,
                cycle_elapsed_s=cycle_elapsed_s,
                step=step_timing,
            )

    def _set_default_standing_reference(self, state: object) -> None:
        self._standing_qpos[:] = 0.0
        self._standing_qpos[0:3] = self._default_root_pos
        self._standing_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        align_motion_qpos_yaw(np.asarray(getattr(state, "quat"), dtype=np.float32), self._standing_qpos)
        self._standing_qpos[ROOT_DIM:FULL_QPOS_DIM] = self.default_angles.astype(np.float64)

    def _reset_policy_state(self) -> None:
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.ref_proc.reset_smoothers()
        self.ref_proc.reset_alignment()
        self.policy.reset()
        self.obs_builder.reset()
        self._last_target = None

    def _warmup_policy(self) -> None:
        if self.no_policy:
            return
        logger.info("Warming up ONNX runtime...")
        dummy_obs = np.zeros(int(self.obs_builder.total_obs_size), dtype=np.float32)
        for _ in range(3):
            self.policy.compute_action(dummy_obs)
        self.policy.reset()
        logger.info("ONNX warmup complete")

    def _log_step(self, qvel: np.ndarray, action: np.ndarray, target: np.ndarray, *, dry: bool) -> None:
        self._step_count += 1
        if self._step_count % 50 != 1:
            return
        tag = "DRY" if dry else "STANDING"
        logger.info(
            "%s step=%d | qvel_norm=%.4f | action_norm=%.4f | target[:6]=%s",
            tag,
            self._step_count,
            float(np.linalg.norm(qvel)),
            float(np.linalg.norm(action)),
            np.array2string(target[:6], precision=4, separator=","),
        )

    def _sleep_until(self, t0: float) -> float:
        remaining = self.dt - (time.monotonic() - t0)
        if remaining > 0.0:
            time.sleep(remaining)
        return time.monotonic() - t0

    def _signal_handler(self, _signum: int, _frame: object) -> None:
        logger.info("Shutdown signal received")
        self.shutdown_requested = True

    def _cleanup(self) -> None:
        if self.dry_run:
            self.robot.close()
            return
        if not self._entered_debug:
            self.robot.close()
            return
        try:
            logger.info("Shutting down: setting damping...")
            self.robot.set_damping()
            time.sleep(0.5)
        finally:
            logger.info("Stopping publish and restoring ai mode...")
            self.robot.exit_debug_mode()
            self.robot.close()
            logger.info("Done.")


def _build_cfg(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.create(
        {
            "policy_hz": DEFAULT_POLICY_HZ,
            "startup_ramp_duration": args.kp_ramp_duration,
            "kp_ramp_floor_ratio": args.kp_ramp_floor_ratio,
            "joint_vel_limit": args.joint_vel_limit,
            "reference_steps": [0],
            "reference_velocity_smoothing_alpha": 1.0,
            "reference_anchor_velocity_smoothing_alpha": 1.0,
            "robot": OmegaConf.load(PROJECT_ROOT / "teleopit" / "configs" / "robot" / "g1.yaml"),
            "controller": OmegaConf.load(PROJECT_ROOT / "teleopit" / "configs" / "controller" / "rl_policy.yaml"),
            "real_robot": OmegaConf.load(PROJECT_ROOT / "teleopit" / "configs" / "sim2real.yaml").real_robot,
            "console": {
                "show_timing": bool(args.show_timing),
                "timing_log_interval_s": float(args.timing_log_interval_s),
            },
        }
    )
    cfg.controller.policy_path = str(args.policy)
    cfg.real_robot.network_interface = str(args.network_interface)
    cfg.real_robot.publish_hz = int(args.publish_hz)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 standalone standing using sim2real standing implementation")
    parser.add_argument("--policy", type=str, required=True, help="Path to a dual-input ONNX policy")
    parser.add_argument("--network-interface", type=str, default="eth0", help="DDS network interface")
    parser.add_argument("--dry-run", action="store_true", help="Read state and run policy without motor commands")
    parser.add_argument("--no-policy", action="store_true", help="Send zero-action standing target")
    parser.add_argument("--publish-hz", type=int, default=200, help="C++ bridge publish frequency")
    parser.add_argument("--kp-ramp-duration", type=float, default=2.0, help="Startup Kp ramp duration in seconds")
    parser.add_argument("--kp-ramp-floor-ratio", type=float, default=0.1, help="Initial Kp ratio during startup")
    parser.add_argument("--joint-vel-limit", type=float, default=10.0, help="Damp if any joint exceeds this velocity")
    parser.add_argument("--show-timing", action="store_true", help="Print periodic timing diagnostics")
    parser.add_argument(
        "--timing-log-interval-s",
        type=float,
        default=10.0,
        help="Timing diagnostic print interval when --show-timing is set",
    )
    parser.add_argument(
        "--obs-delay-ms",
        type=float,
        default=0.0,
        help="Diagnostic delay after LowState read, before observation build/inference",
    )
    parser.add_argument(
        "--command-delay-ms",
        type=float,
        default=0.0,
        help="Diagnostic delay after target computation, before C++ bridge set_target",
    )
    args = parser.parse_args()

    controller = StandaloneStandingController(
        _build_cfg(args),
        dry_run=bool(args.dry_run),
        no_policy=bool(args.no_policy),
        obs_delay_s=float(args.obs_delay_ms) / 1000.0,
        command_delay_s=float(args.command_delay_ms) / 1000.0,
    )
    controller.run()


if __name__ == "__main__":
    main()
