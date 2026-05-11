from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from teleopit.controllers.observation import VelCmdObservationBuilder
from teleopit.inputs.pico_video import bridge_video_source, parse_pico_video_config

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



def build_simulation_cfg(cfg: Any) -> dict[str, object]:
    playback_cfg = cfg_get(cfg, "playback", {}) or {}
    playback_keyboard_cfg = cfg_get(playback_cfg, "keyboard", {}) or {}
    realtime_keyboard_cfg = cfg_get(cfg, "keyboard", {}) or {}
    return {
        "policy_hz": float(cfg_get(cfg, "policy_hz", 50.0)),
        "pd_hz": float(cfg_get(cfg, "pd_hz", 1000.0)),
        "transition_duration": float(cfg_get(cfg, "transition_duration", 0.0) or 0.0),
        "pause_resume_transition_duration": float(
            cfg_get(cfg, "pause_resume_transition_duration", cfg_get(cfg, "transition_duration", 0.0)) or 0.0
        ),
        "pause_resume_warmup_steps": cfg_get(cfg, "pause_resume_warmup_steps", None),
        "retarget_buffer_enabled": bool(cfg_get(cfg, "retarget_buffer_enabled", True)),
        "retarget_buffer_window_s": float(cfg_get(cfg, "retarget_buffer_window_s", 0.5)),
        "retarget_buffer_delay_s": cfg_get(cfg, "retarget_buffer_delay_s", None),
        "reference_steps": cfg_get(cfg, "reference_steps", [0]),
        "reference_debug_log": bool(cfg_get(cfg, "reference_debug_log", False)),
        "realtime_input_delay_s": cfg_get(cfg, "realtime_input_delay_s", None),
        "realtime_buffer_low_watermark_steps": cfg_get(cfg, "realtime_buffer_low_watermark_steps", None),
        "realtime_buffer_high_watermark_steps": cfg_get(cfg, "realtime_buffer_high_watermark_steps", None),
        "realtime_buffer_warmup_steps": cfg_get(cfg, "realtime_buffer_warmup_steps", None),
        "realtime_catchup_enabled": bool(cfg_get(cfg, "realtime_catchup_enabled", False)),
        "realtime_catchup_trigger_steps": cfg_get(cfg, "realtime_catchup_trigger_steps", None),
        "realtime_catchup_release_steps": cfg_get(cfg, "realtime_catchup_release_steps", None),
        "realtime_catchup_target_delay_s": cfg_get(cfg, "realtime_catchup_target_delay_s", None),
        "reference_qpos_smoothing_alpha": float(cfg_get(cfg, "reference_qpos_smoothing_alpha", 1.0)),
        "reference_velocity_smoothing_alpha": float(cfg_get(cfg, "reference_velocity_smoothing_alpha", 1.0)),
        "reference_anchor_velocity_smoothing_alpha": float(
            cfg_get(cfg, "reference_anchor_velocity_smoothing_alpha", 1.0)
        ),
        "realtime": bool(cfg_get(cfg, "realtime", False)),
        "debug_trace_path": cfg_get(cfg, "debug_trace_path", None),
        "playback": {
            "pause_on_end": bool(cfg_get(playback_cfg, "pause_on_end", False)),
            "keyboard": {
                "enabled": bool(cfg_get(playback_keyboard_cfg, "enabled", False)),
            },
        },
        "keyboard": {
            "enabled": bool(cfg_get(realtime_keyboard_cfg, "enabled", False)),
        },
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
        "robot_joint_names": list(cfg_get(controller_cfg, "robot_joint_names", cfg_get(controller_cfg, "joint_names", []))),
        "target_joint_names": cfg_get(controller_cfg, "target_joint_names", None),
    }
    if observation_type == "velcmd_history":
        return VelCmdObservationBuilder(obs_cfg)
    raise ValueError(
        f"Unsupported controller.observation_type='{observation_type}'. "
        "Supported value is velcmd_history."
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
    if not bool(getattr(controller, "_multi_input", False)):
        raise ValueError(
            "Only dual inputs ONNX policies are supported here; expected inputs named 'obs' and 'obs_history'."
        )
    obs_builder = _build_obs_builder(robot_cfg, controller_cfg, sim_cfg)
    policy_dim = getattr(controller, "_expected_obs_dim", None)
    builder_dim = getattr(obs_builder, "total_obs_size", None)
    if policy_dim is not None and builder_dim is not None and policy_dim != builder_dim:
        if builder_dim == 166:
            raise ValueError(
                f"Only 166D velcmd_history ONNX policies are supported here; "
                f"obs_builder produces 166D but policy expects {policy_dim}D."
            )
        raise ValueError(
            f"Observation dimension mismatch at startup: obs_builder produces {builder_dim}D "
            f"but policy expects {policy_dim}D. Use a matching ONNX model."
        )
    return controller, obs_builder



def _prepare_input_cfg(input_cfg: Any, project_root: Path, *, sim2real: bool) -> str:
    provider_kind = str(cfg_get(input_cfg, "provider", "bvh")).lower()
    if provider_kind == "bvh":
        normalize_path_in_cfg(
            input_cfg,
            "bvh_file",
            base_dir=project_root,
            required=True,
            missing_message="input.bvh_file must be set for offline BVH input",
        )
    elif provider_kind == "udp_bvh":
        pass  # skeleton resolved automatically from bvh_format
    elif provider_kind != "pico4":
        raise ValueError(
            f"Unsupported input.provider='{provider_kind}'. "
            "Supported providers are bvh, pico4, udp_bvh."
        )

    if sim2real and provider_kind not in ("bvh", "pico4", "udp_bvh"):
        raise ValueError(
            f"Sim2real only supports bvh, pico4, or udp_bvh input providers; got '{provider_kind}'."
        )
    return provider_kind



def _build_input_provider(
    *,
    input_cfg: Any,
    provider_kind: str,
    bvh_input_cls: type[Any],
    pico4_input_cls: type[Any],
) -> Any:
    if provider_kind == "udp_bvh":
        from teleopit.inputs.udp_bvh_provider import UDPBVHInputProvider

        return UDPBVHInputProvider(
            bvh_format=str(cfg_get(input_cfg, "bvh_format", "hc_mocap")),
            human_height=float(cfg_get(input_cfg, "human_height", 1.75)),
            udp_host=str(cfg_get(input_cfg, "udp_host", "")),
            udp_port=int(cfg_get(input_cfg, "udp_port", 1118)),
            udp_timeout=float(cfg_get(input_cfg, "udp_timeout", 30.0)),
        )

    if provider_kind == "pico4":
        video_cfg = parse_pico_video_config(input_cfg)
        video_source = bridge_video_source(video_cfg)
        return pico4_input_cls(
            human_format=str(cfg_get(input_cfg, "human_format", "pico_bridge")),
            timeout=float(cfg_get(input_cfg, "pico4_timeout", 60.0)),
            buffer_size=int(cfg_get(input_cfg, "pico4_buffer_size", 60)),
            timestamp_gap_reset_s=float(cfg_get(input_cfg, "pico4_timestamp_gap_reset_s", 0.15)),
            pause_button=cfg_get(input_cfg, "pause_button", "A"),
            pause_debounce_s=float(cfg_get(input_cfg, "pause_debounce_s", 0.25)),
            bridge_host=str(cfg_get(input_cfg, "bridge_host", "0.0.0.0")),
            bridge_port=int(cfg_get(input_cfg, "bridge_port", 63901)),
            bridge_discovery=bool(cfg_get(input_cfg, "bridge_discovery", True)),
            bridge_advertise_ip=cfg_get(input_cfg, "bridge_advertise_ip", None),
            bridge_video=video_source,
            bridge_video_enabled=video_cfg.enabled,
            bridge_start_timeout=float(cfg_get(input_cfg, "bridge_start_timeout", 10.0)),
            bridge_history_size=int(cfg_get(input_cfg, "bridge_history_size", 120)),
        )

    return bvh_input_cls(
        bvh_path=str(cfg_get(input_cfg, "bvh_file")),
        human_format=str(cfg_get(input_cfg, "bvh_format", "lafan1")),
    )



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
        return str(cfg_get(input_cfg, "human_format", "pico_bridge"))
    return f"bvh_{cfg_get(input_cfg, 'bvh_format', 'lafan1')}"


def _resolve_actual_human_height(input_cfg: Any, input_provider: Any) -> float:
    del input_provider
    configured_height = cfg_get(input_cfg, "human_height", None)
    if configured_height not in (None, "", "null"):
        actual_human_height = float(configured_height)
    else:
        actual_human_height = 1.75

    if actual_human_height <= 0.0:
        raise ValueError(
            f"Resolved human height must be > 0, got {actual_human_height}. "
            "Set input.human_height or provide a valid input provider human_height."
        )
    return actual_human_height



def _build_retargeter(input_cfg: Any, input_provider: Any, retargeter_cls: type[Any]) -> Any:
    return retargeter_cls(
        robot_name=str(cfg_get(input_cfg, "robot_name", "unitree_g1")),
        human_format=_resolve_human_format(input_cfg, input_provider),
        actual_human_height=_resolve_actual_human_height(input_cfg, input_provider),
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
    )
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
    bvh_input_cls: type[Any],
    pico4_input_cls: type[Any],
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
        bvh_input_cls=bvh_input_cls,
        pico4_input_cls=pico4_input_cls,
    )
    retargeter = _build_retargeter(input_cfg, input_provider, retargeter_cls)
    return MocapComponents(
        input_provider=input_provider,
        retargeter=retargeter,
        controller=controller,
        obs_builder=obs_builder,
    )
