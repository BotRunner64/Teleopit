"""Environment builder for the single supported VelCmdHistory task."""

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


def _apply_play_mode_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"


def _add_history_obs_groups(
    cfg: ManagerBasedRlEnvCfg, history_length: int = 10
) -> None:
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
    """Create the single supported G1 VelCmdHistory training env."""
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
    motion_cmd.sampling_mode = "uniform"

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
    cfg.terminations["anchor_pos"].params["threshold"] = 0.4
    cfg.terminations["anchor_ori"].params["threshold"] = 1.0
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.4
    cfg.viewer.body_name = "torso_link"
    cfg.episode_length_s = 10.0
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    actor_terms = {
        key: value
        for key, value in cfg.observations["actor"].terms.items()
        if key not in {"motion_anchor_pos_b", "base_lin_vel"}
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
    )

    cfg.observations["actor"].terms.update(deepcopy(_VELCMD_ACTOR_TERMS))
    cfg.observations["critic"].terms.update(deepcopy(_VELCMD_CRITIC_TERMS))

    _add_history_obs_groups(cfg)

    if play:
        _apply_play_mode_overrides(cfg)
        cfg.observations["actor_history"].enable_corruption = False
        cfg.observations["critic_history"].enable_corruption = False

    return cfg
