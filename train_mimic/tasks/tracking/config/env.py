"""Tracking task environment builders."""

from __future__ import annotations

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from train_mimic.tasks.tracking import mdp
from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE
from train_mimic.tasks.tracking.config.profiles import (
    OFFICIAL_UNIFORM_PROFILE,
    TrackingTaskProfile,
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


def make_tracking_env_cfg_for_profile(
    profile: TrackingTaskProfile = OFFICIAL_UNIFORM_PROFILE,
    *,
    play: bool = False,
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
    if play:
        _apply_play_mode_overrides(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Velocity-driven tracking task
# ---------------------------------------------------------------------------

_EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)


def _apply_velocity_driven_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.rewards = {
        "root_lin_vel": RewardTermCfg(
            func=mdp.root_lin_vel_tracking_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.5},
        ),
        "root_yaw_rate": RewardTermCfg(
            func=mdp.root_yaw_rate_tracking_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.5},
        ),
        "root_height": RewardTermCfg(
            func=mdp.root_height_tracking_exp,
            weight=0.3,
            params={"command_name": "motion", "std": 0.1},
        ),
        "motion_body_pos": RewardTermCfg(
            func=mdp.motion_local_body_position_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.3},
        ),
        "motion_body_ori": RewardTermCfg(
            func=mdp.motion_local_body_orientation_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.4},
        ),
        "motion_body_lin_vel": RewardTermCfg(
            func=mdp.motion_local_body_linear_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 1.0},
        ),
        "motion_body_ang_vel": RewardTermCfg(
            func=mdp.motion_local_body_angular_velocity_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 3.14},
        ),
        "action_rate_l2": cfg.rewards["action_rate_l2"],
        "joint_limit": cfg.rewards["joint_limit"],
        "self_collisions": cfg.rewards["self_collisions"],
    }


def _apply_velocity_driven_observations(cfg: ManagerBasedRlEnvCfg) -> None:
    from mjlab.managers.observation_manager import ObservationTermCfg

    actor_terms = dict(cfg.observations["actor"].terms)
    critic_terms = dict(cfg.observations["critic"].terms)

    for terms in (actor_terms, critic_terms):
        terms.pop("motion_anchor_pos_b", None)
        terms.pop("motion_anchor_ori_b", None)

    vel_cmd_term = ObservationTermCfg(
        func=mdp.root_vel_cmd,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )
    yaw_cmd_term = ObservationTermCfg(
        func=mdp.root_yaw_rate_cmd,
        params={"command_name": "motion"},
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )

    actor_terms["root_vel_cmd"] = vel_cmd_term
    actor_terms["root_yaw_rate_cmd"] = yaw_cmd_term

    critic_vel_cmd = ObservationTermCfg(
        func=mdp.root_vel_cmd,
        params={"command_name": "motion"},
    )
    critic_yaw_cmd = ObservationTermCfg(
        func=mdp.root_yaw_rate_cmd,
        params={"command_name": "motion"},
    )
    critic_terms["root_vel_cmd"] = critic_vel_cmd
    critic_terms["root_yaw_rate_cmd"] = critic_yaw_cmd

    cfg.observations["actor"] = ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
    )
    cfg.observations["critic"] = ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
    )


def _apply_velocity_driven_terminations(
    cfg: ManagerBasedRlEnvCfg, profile: TrackingTaskProfile
) -> None:
    cfg.terminations["ee_body_pos"] = TerminationTermCfg(
        func=mdp.bad_local_body_pos_z,
        params={
            "command_name": "motion",
            "threshold": profile.ee_body_pos_threshold,
            "body_names": _EE_BODY_NAMES,
        },
    )


def make_vel_tracking_env_cfg_for_profile(
    profile: TrackingTaskProfile,
    *,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the G1 velocity-driven tracking env."""
    cfg = make_tracking_env_cfg()

    # --- Robot-specific (same as original) ---
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
    cfg.terminations["anchor_pos"].params["threshold"] = profile.anchor_pos_threshold
    cfg.terminations["anchor_ori"].params["threshold"] = profile.anchor_ori_threshold
    cfg.viewer.body_name = "torso_link"
    cfg.episode_length_s = profile.episode_length_s
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    # --- Velocity-driven overrides ---
    _apply_velocity_driven_rewards(cfg)
    _apply_velocity_driven_observations(cfg)
    _apply_velocity_driven_terminations(cfg, profile)

    if play:
        _apply_play_mode_overrides(cfg)

    return cfg
