from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teleopit.controllers.observation import (
    MotionTrackingObservationBuilder,
    VelCmdObservationBuilder,
)

from .common import cfg_get, cfg_set, normalize_path_in_cfg, parse_viewers, require_section


@dataclass(frozen=True)
class InferenceComponents:
    robot: Any
    controller: Any
    obs_builder: Any
    input_provider: Any
    retargeter: Any
    sim_cfg: dict[str, object]
    viewers: set[str]


@dataclass(frozen=True)
class MocapComponents:
    input_provider: Any
    retargeter: Any
    controller: Any
    obs_builder: Any


class _LoopingInputProvider:
    def __init__(
        self, provider: Any, on_reset: Callable[[], None] | None = None
    ) -> None:
        self._provider = provider
        self._on_reset = on_reset

    def get_frame(self) -> dict[str, tuple[Any, Any]]:
        if not self._provider.is_available():
            self._provider.reset()
            if self._on_reset is not None:
                self._on_reset()
        return self._provider.get_frame()

    def is_available(self) -> bool:
        return True

    @property
    def fps(self) -> int:
        return self._provider.fps

    @property
    def bone_names(self) -> list[str]:
        return self._provider.bone_names

    @property
    def bone_parents(self) -> Any:
        return self._provider.bone_parents



def build_simulation_cfg(cfg: Any) -> dict[str, object]:
    return {
        "policy_hz": float(cfg_get(cfg, "policy_hz", 50.0)),
        "pd_hz": float(cfg_get(cfg, "pd_hz", 1000.0)),
        "transition_duration": float(cfg_get(cfg, "transition_duration", 0.0) or 0.0),
        "velcmd_fixed_ref_yaw_alignment": bool(cfg_get(cfg, "velcmd_fixed_ref_yaw_alignment", True)),
        "retarget_buffer_enabled": bool(cfg_get(cfg, "retarget_buffer_enabled", True)),
        "retarget_buffer_window_s": float(cfg_get(cfg, "retarget_buffer_window_s", 0.5)),
        "retarget_buffer_delay_s": cfg_get(cfg, "retarget_buffer_delay_s", None),
        "reference_steps": cfg_get(cfg, "reference_steps", [0]),
        "reference_debug_log": bool(cfg_get(cfg, "reference_debug_log", False)),
        "realtime_input_delay_s": cfg_get(cfg, "realtime_input_delay_s", None),
        "realtime_buffer_low_watermark_steps": cfg_get(cfg, "realtime_buffer_low_watermark_steps", None),
        "realtime_buffer_high_watermark_steps": cfg_get(cfg, "realtime_buffer_high_watermark_steps", None),
        "realtime_buffer_warmup_steps": cfg_get(cfg, "realtime_buffer_warmup_steps", None),
        "reference_velocity_smoothing_alpha": float(cfg_get(cfg, "reference_velocity_smoothing_alpha", 1.0)),
        "reference_anchor_velocity_smoothing_alpha": float(
            cfg_get(cfg, "reference_anchor_velocity_smoothing_alpha", 1.0)
        ),
        "realtime": bool(cfg_get(cfg, "realtime", False)),
        "debug_trace_path": cfg_get(cfg, "debug_trace_path", None),
    }



def propagate_controller_defaults(controller_cfg: Any, robot_cfg: Any) -> None:
    if cfg_get(controller_cfg, "default_dof_pos", None) is None:
        default_angles = cfg_get(robot_cfg, "default_angles", None)
        if default_angles is not None:
            cfg_set(controller_cfg, "default_dof_pos", list(default_angles))

    if cfg_get(controller_cfg, "action_scale", None) is None:
        robot_action_scale = cfg_get(robot_cfg, "action_scale", None)
        if robot_action_scale is not None:
            try:
                cfg_set(controller_cfg, "action_scale", list(robot_action_scale))
            except TypeError:
                cfg_set(controller_cfg, "action_scale", robot_action_scale)



def _prepare_policy_paths(robot_cfg: Any, controller_cfg: Any, project_root: Path) -> None:
    normalize_path_in_cfg(
        robot_cfg,
        "xml_path",
        base_dir=project_root,
        required=True,
        missing_message="robot.xml_path must be set",
    )
    normalize_path_in_cfg(
        controller_cfg,
        "policy_path",
        base_dir=project_root,
        required=True,
        missing_message="controller.policy_path must be set to an exported ONNX policy",
    )



def _build_obs_builder(robot_cfg: Any, controller_cfg: Any, sim_cfg: dict[str, object]) -> Any:
    observation_type = str(cfg_get(controller_cfg, "observation_type", "velcmd_history")).lower()
    reference_steps = [int(v) for v in sim_cfg.get("reference_steps", [0])]
    future_steps_raw = cfg_get(controller_cfg, "future_steps", None)
    future_steps = reference_steps if future_steps_raw is None else [int(v) for v in future_steps_raw]
    if future_steps != reference_steps:
        raise ValueError(
            "Motion/reference window mismatch at startup: "
            f"controller.future_steps={future_steps} but top-level reference_steps={reference_steps}. "
            "Set them to the same sequence so runtime sampling matches the policy observation definition."
        )
    obs_cfg = {
        "num_actions": int(cfg_get(robot_cfg, "num_actions")),
        "default_dof_pos": list(cfg_get(robot_cfg, "default_angles")),
        "xml_path": str(cfg_get(robot_cfg, "xml_path")),
        "anchor_body_name": cfg_get(robot_cfg, "anchor_body_name", "torso_link"),
        "reference_steps": reference_steps,
        "future_steps": future_steps,
        "prev_action_steps": int(cfg_get(controller_cfg, "prev_action_steps", 8)),
        "root_angvel_history_steps": list(cfg_get(controller_cfg, "root_angvel_history_steps", [0, 1, 2, 3, 4, 8, 12, 16, 20])),
        "projected_gravity_history_steps": list(cfg_get(controller_cfg, "projected_gravity_history_steps", [0, 1, 2, 3, 4, 8, 12, 16, 20])),
        "joint_pos_history_steps": list(cfg_get(controller_cfg, "joint_pos_history_steps", [0, 1, 2, 3, 4, 8, 12, 16, 20])),
        "joint_vel_history_steps": list(cfg_get(controller_cfg, "joint_vel_history_steps", [0, 1, 2, 3, 4, 8, 12, 16, 20])),
        "compliance_flag_value": float(cfg_get(controller_cfg, "compliance_flag_value", 1.0)),
        "compliance_flag_threshold": float(cfg_get(controller_cfg, "compliance_flag_threshold", 10.0)),
        "robot_joint_names": list(cfg_get(controller_cfg, "robot_joint_names", cfg_get(controller_cfg, "joint_names", []))),
        "target_joint_names": cfg_get(controller_cfg, "target_joint_names", None),
    }
    if observation_type == "velcmd_history":
        return VelCmdObservationBuilder(obs_cfg)
    if observation_type in {"motion_tracking", "motion_tracking_deploy"}:
        return MotionTrackingObservationBuilder(obs_cfg)
    raise ValueError(
        f"Unsupported controller.observation_type='{observation_type}'. "
        "Supported values are velcmd_history and motion_tracking_deploy."
    )



def _build_policy_components(
    *,
    robot_cfg: Any,
    controller_cfg: Any,
    sim_cfg: dict[str, object],
    project_root: Path,
    controller_cls: type[Any],
) -> tuple[Any, Any]:
    _prepare_policy_paths(robot_cfg, controller_cfg, project_root)
    propagate_controller_defaults(controller_cfg, robot_cfg)

    controller = controller_cls(controller_cfg)
    obs_builder = _build_obs_builder(robot_cfg, controller_cfg, sim_cfg)
    policy_dim = getattr(controller, "_expected_obs_dim", None)
    builder_dim = getattr(obs_builder, "total_obs_size", None)
    if policy_dim is not None and builder_dim is not None and policy_dim != builder_dim:
        raise ValueError(
            f"Observation dimension mismatch at startup: obs_builder produces {builder_dim}D "
            f"but policy expects {policy_dim}D. Use a matching ONNX model."
        )
    return controller, obs_builder



def _prepare_input_cfg(input_cfg: Any, project_root: Path, *, sim2real: bool) -> str:
    provider_kind = str(cfg_get(input_cfg, "provider", "udp_bvh" if sim2real else "bvh")).lower()
    if provider_kind in ("bvh", "vr_stub"):
        normalize_path_in_cfg(
            input_cfg,
            "bvh_file",
            base_dir=project_root,
            required=True,
            missing_message="input.bvh_file must be set for offline BVH input",
        )
    elif provider_kind == "udp_bvh":
        normalize_path_in_cfg(
            input_cfg,
            "reference_bvh",
            base_dir=project_root,
            required=True,
            missing_message="input.reference_bvh must be set for udp_bvh provider",
        )
    elif provider_kind != "pico4":
        raise ValueError(
            f"Unsupported input.provider='{provider_kind}'. "
            "Supported providers are bvh, vr_stub, udp_bvh, pico4."
        )

    if sim2real and provider_kind not in ("udp_bvh", "pico4"):
        raise ValueError(
            f"Sim2real only supports udp_bvh or pico4 input providers; got '{provider_kind}'."
        )
    return provider_kind



def _build_input_provider(
    *,
    input_cfg: Any,
    provider_kind: str,
    bvh_input_cls: type[Any],
    pico4_input_cls: type[Any],
    udp_bvh_input_cls: type[Any],
) -> Any:
    if provider_kind == "pico4":
        return pico4_input_cls(
            human_format=str(cfg_get(input_cfg, "human_format", "xrobot")),
            timeout=float(cfg_get(input_cfg, "pico4_timeout", 60.0)),
        )

    if provider_kind == "udp_bvh":
        return udp_bvh_input_cls(
            reference_bvh=str(cfg_get(input_cfg, "reference_bvh")),
            host=str(cfg_get(input_cfg, "udp_host", "")),
            port=int(cfg_get(input_cfg, "udp_port", 1118)),
            human_format=str(cfg_get(input_cfg, "bvh_format", "hc_mocap")),
            timeout=float(cfg_get(input_cfg, "udp_timeout", 30.0)),
        )

    provider = bvh_input_cls(
        bvh_path=str(cfg_get(input_cfg, "bvh_file")),
        human_format=str(cfg_get(input_cfg, "bvh_format", "lafan1")),
    )
    if provider_kind == "vr_stub":
        return _LoopingInputProvider(provider)
    return provider



def _resolve_human_format(input_cfg: Any, input_provider: Any) -> str:
    if hasattr(input_provider, "human_format"):
        provider_format = input_provider.human_format
        provider_kind = str(cfg_get(input_cfg, "provider", "bvh")).lower()
        if provider_kind == "pico4":
            return str(provider_format)
        return f"bvh_{provider_format}"

    human_format = cfg_get(input_cfg, "human_format", None)
    if human_format and str(human_format) != "null":
        return str(human_format)

    if str(cfg_get(input_cfg, "provider", "bvh")).lower() == "pico4":
        return str(cfg_get(input_cfg, "human_format", "xrobot"))
    return f"bvh_{cfg_get(input_cfg, 'bvh_format', 'lafan1')}"



def _build_retargeter(input_cfg: Any, input_provider: Any, retargeter_cls: type[Any]) -> Any:
    return retargeter_cls(
        robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
        human_format=_resolve_human_format(input_cfg, input_provider),
        actual_human_height=float(cfg_get(input_cfg, "human_height", 1.75)),
    )



def build_inference_components(
    cfg: Any,
    project_root: Path,
    *,
    robot_cls: type[Any],
    controller_cls: type[Any],
    obs_builder_cls: type[Any],
    bvh_input_cls: type[Any],
    pico4_input_cls: type[Any],
    udp_bvh_input_cls: type[Any],
    retargeter_cls: type[Any],
) -> InferenceComponents:
    del obs_builder_cls
    robot_cfg = require_section(cfg, "robot")
    controller_cfg = require_section(cfg, "controller")
    input_cfg = require_section(cfg, "input")
    sim_cfg = build_simulation_cfg(cfg)

    provider_kind = _prepare_input_cfg(input_cfg, project_root, sim2real=False)
    controller, obs_builder = _build_policy_components(
        robot_cfg=robot_cfg,
        controller_cfg=controller_cfg,
        sim_cfg=sim_cfg,
        project_root=project_root,
        controller_cls=controller_cls,
    )
    robot = robot_cls(robot_cfg)
    input_provider = _build_input_provider(
        input_cfg=input_cfg,
        provider_kind=provider_kind,
        bvh_input_cls=bvh_input_cls,
        pico4_input_cls=pico4_input_cls,
        udp_bvh_input_cls=udp_bvh_input_cls,
    )
    if isinstance(input_provider, _LoopingInputProvider) and hasattr(controller, "reset"):
        input_provider._on_reset = controller.reset
    retargeter = _build_retargeter(input_cfg, input_provider, retargeter_cls)
    return InferenceComponents(
        robot=robot,
        controller=controller,
        obs_builder=obs_builder,
        input_provider=input_provider,
        retargeter=retargeter,
        sim_cfg=sim_cfg,
        viewers=parse_viewers(cfg),
    )



def build_sim2real_mocap_components(
    cfg: Any,
    project_root: Path,
    *,
    controller_cls: type[Any],
    obs_builder_cls: type[Any],
    pico4_input_cls: type[Any],
    udp_bvh_input_cls: type[Any],
    retargeter_cls: type[Any],
) -> MocapComponents:
    del obs_builder_cls
    robot_cfg = require_section(cfg, "robot")
    controller_cfg = require_section(cfg, "controller")
    input_cfg = require_section(cfg, "input")
    sim_cfg = build_simulation_cfg(cfg)

    provider_kind = _prepare_input_cfg(input_cfg, project_root, sim2real=True)
    controller, obs_builder = _build_policy_components(
        robot_cfg=robot_cfg,
        controller_cfg=controller_cfg,
        sim_cfg=sim_cfg,
        project_root=project_root,
        controller_cls=controller_cls,
    )
    input_provider = _build_input_provider(
        input_cfg=input_cfg,
        provider_kind=provider_kind,
        bvh_input_cls=object,
        pico4_input_cls=pico4_input_cls,
        udp_bvh_input_cls=udp_bvh_input_cls,
    )
    retargeter = _build_retargeter(input_cfg, input_provider, retargeter_cls)
    return MocapComponents(
        input_provider=input_provider,
        retargeter=retargeter,
        controller=controller,
        obs_builder=obs_builder,
    )
