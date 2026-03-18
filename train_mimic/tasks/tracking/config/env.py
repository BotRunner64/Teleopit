"""Tracking task environment builders."""

from __future__ import annotations

from copy import deepcopy

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from train_mimic.tasks.tracking import mdp
from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE
from train_mimic.tasks.tracking.config.profiles import (
    OFFICIAL_UNIFORM_PROFILE,
    TrackingTaskProfile,
    VELCMD_HISTORY_ADAPTIVE_PROFILE,
)
from train_mimic.tasks.tracking.mdp import MotionCommandCfg
from train_mimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

_TRACKING_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


def _prune_actor_terms_for_no_state_estimation(cfg: ManagerBasedRlEnvCfg) -> None:
    new_actor_terms = {
        k: v
        for k, v in cfg.observations["actor"].terms.items()
        if k not in {"motion_anchor_pos_b", "base_lin_vel"}
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=new_actor_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
    )


def _apply_play_mode_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"


_ROOT_POSE_REWARD_KEYS = ("motion_global_root_pos", "motion_global_root_ori")


def _swap_obs_for_yaw_only(cfg: ManagerBasedRlEnvCfg) -> None:
    """Replace observation terms with yaw-only variants."""
    _YAW_SWAP = {
        "motion_anchor_pos_b": mdp.motion_anchor_pos_b_yaw,
        "motion_anchor_ori_b": mdp.motion_anchor_ori_b_yaw,
        "body_pos": mdp.robot_body_pos_b_yaw,
        "body_ori": mdp.robot_body_ori_b_yaw,
    }
    for group_key in ("actor", "critic"):
        group = cfg.observations[group_key]
        for term_name, yaw_func in _YAW_SWAP.items():
            if term_name in group.terms:
                group.terms[term_name].func = yaw_func


def make_tracking_env_cfg_for_profile(
    profile: TrackingTaskProfile = OFFICIAL_UNIFORM_PROFILE,
    *,
    play: bool = False,
    no_root_pose: bool = False,
    yaw_only: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the G1 tracking env for a single internal profile."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_g1_robot_cfg()}
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=4,
        ),
    )

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.body_names = _TRACKING_BODY_NAMES
    motion_cmd.motion_file = DEFAULT_TRAIN_MOTION_FILE
    motion_cmd.sampling_mode = profile.sampling_mode
    if profile.adaptive_kernel_size is not None:
        motion_cmd.adaptive_kernel_size = profile.adaptive_kernel_size
    if profile.adaptive_uniform_ratio is not None:
        motion_cmd.adaptive_uniform_ratio = profile.adaptive_uniform_ratio
    if profile.adaptive_alpha is not None:
        motion_cmd.adaptive_alpha = profile.adaptive_alpha

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    cfg.terminations["anchor_pos"].params["threshold"] = profile.anchor_pos_threshold
    cfg.terminations["anchor_ori"].params["threshold"] = profile.anchor_ori_threshold
    cfg.terminations["ee_body_pos"].params["threshold"] = profile.ee_body_pos_threshold
    cfg.viewer.body_name = "torso_link"
    cfg.episode_length_s = profile.episode_length_s
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    _prune_actor_terms_for_no_state_estimation(cfg)
    if yaw_only:
        _swap_obs_for_yaw_only(cfg)
    if no_root_pose:
        for key in _ROOT_POSE_REWARD_KEYS:
            cfg.rewards.pop(key)
    if play:
        _apply_play_mode_overrides(cfg)

    return cfg


def _add_history_obs_groups(
    cfg: ManagerBasedRlEnvCfg, history_length: int = 10
) -> None:
    """Add ``actor_history`` and ``critic_history`` observation groups.

    Each history group mirrors the corresponding base group's terms but uses
    group-level ``history_length`` with ``flatten_history_dim=False`` so the
    observation manager produces ``(B, T, D)`` tensors.
    """
    cfg.observations["actor_history"] = ObservationGroupCfg(
        terms=deepcopy(cfg.observations["actor"].terms),
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
        history_length=history_length,
        flatten_history_dim=False,
    )
    cfg.observations["critic_history"] = ObservationGroupCfg(
        terms=deepcopy(cfg.observations["critic"].terms),
        concatenate_terms=True,
        enable_corruption=False,
        history_length=history_length,
        flatten_history_dim=False,
    )


def make_history_cnn_tracking_env_cfg(
    *, play: bool = False, yaw_only: bool = False
) -> ManagerBasedRlEnvCfg:
    """Create the G1 tracking env with history CNN observation groups."""
    cfg = make_tracking_env_cfg_for_profile(
        OFFICIAL_UNIFORM_PROFILE, play=play, yaw_only=yaw_only
    )
    _add_history_obs_groups(cfg)
    # In play mode the base factory disables corruption on actor; do the same
    # for the history groups.
    if play:
        cfg.observations["actor_history"].enable_corruption = False
        cfg.observations["critic_history"].enable_corruption = False
    return cfg


_VELCMD_ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    "projected_gravity": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ref_base_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_base_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_base_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_base_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
}

_VELCMD_CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "ref_base_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_base_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_base_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_base_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
}


def make_velcmd_history_tracking_env_cfg(
    *, play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the G1 tracking env with velocity-command obs + history CNN.

    Extends the base HistoryCNN config by appending projected_gravity,
    ref_base_lin_vel_b, ref_base_ang_vel_b, and ref_projected_gravity_b
    to both actor and critic observation groups.
    """
    cfg = make_tracking_env_cfg_for_profile(OFFICIAL_UNIFORM_PROFILE, play=play)

    # Append velocity-command terms to actor and critic groups.
    cfg.observations["actor"].terms.update(deepcopy(_VELCMD_ACTOR_TERMS))
    cfg.observations["critic"].terms.update(deepcopy(_VELCMD_CRITIC_TERMS))

    _add_history_obs_groups(cfg)

    if play:
        cfg.observations["actor_history"].enable_corruption = False
        cfg.observations["critic_history"].enable_corruption = False

    return cfg


def make_velcmd_history_adaptive_tracking_env_cfg(
    *, play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the adaptive VelCmd + HistoryCNN tracking env."""
    cfg = make_tracking_env_cfg_for_profile(VELCMD_HISTORY_ADAPTIVE_PROFILE, play=play)

    cfg.observations["actor"].terms.update(deepcopy(_VELCMD_ACTOR_TERMS))
    cfg.observations["critic"].terms.update(deepcopy(_VELCMD_CRITIC_TERMS))

    _add_history_obs_groups(cfg)

    if play:
        cfg.observations["actor_history"].enable_corruption = False
        cfg.observations["critic_history"].enable_corruption = False

    return cfg
