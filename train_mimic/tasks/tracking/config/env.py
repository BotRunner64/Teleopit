"""Environment builder for the General-Tracking-G1 task."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from train_mimic.tasks.tracking import mdp
from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE
from train_mimic.tasks.tracking.mdp import MotionCommandCfg
from train_mimic.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from teleopit.runtime.assets import UNITREE_G1_XML, missing_gmr_assets_message

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

_TRAIN_ONLY_EVENTS = (
    "push_robot",
    "base_com",
    "add_joint_default_pos",
    "physics_material",
    "randomize_rigid_body_mass",
)


def resolve_g1_training_xml(robot_xml: str | Path | None = None) -> Path:
    """Resolve the MuJoCo XML used for G1 policy training."""
    if robot_xml is None or str(robot_xml).strip() == "":
        return UNITREE_G1_XML.resolve()

    path = Path(robot_xml).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _get_g1_training_spec(robot_xml: str | Path | None = None) -> mujoco.MjSpec:
    xml_path = resolve_g1_training_xml(robot_xml)
    if not xml_path.is_file():
        raise FileNotFoundError(
            missing_gmr_assets_message(xml_path, label="G1 training MuJoCo XML")
        )
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def make_g1_training_robot_cfg(robot_xml: str | Path | None = None):
    robot_cfg = get_g1_robot_cfg()
    xml_path = resolve_g1_training_xml(robot_xml)
    robot_cfg.spec_fn = partial(_get_g1_training_spec, xml_path)
    return robot_cfg


def _apply_play_mode_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    for event_name in _TRAIN_ONLY_EVENTS:
        cfg.events.pop(event_name, None)
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
    "robot_projected_gravity_b": ObservationTermCfg(
        func=mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "ref_anchor_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_height": ObservationTermCfg(
        func=mdp.ref_anchor_height,
        params={"command_name": "motion"},
    ),
}

_VELCMD_CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "robot_projected_gravity_b": ObservationTermCfg(func=mdp.projected_gravity),
    "ref_anchor_lin_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_lin_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_ang_vel_b": ObservationTermCfg(
        func=mdp.ref_anchor_ang_vel_b,
        params={"command_name": "motion"},
    ),
    "ref_projected_gravity_b": ObservationTermCfg(
        func=mdp.ref_projected_gravity_b,
        params={"command_name": "motion"},
    ),
    "ref_anchor_height": ObservationTermCfg(
        func=mdp.ref_anchor_height,
        params={"command_name": "motion"},
    ),
}


def _configure_self_collision_reward(cfg: ManagerBasedRlEnvCfg) -> None:
    excluded_body_names = (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    cfg.scene.sensors = (
        *tuple(getattr(cfg.scene, "sensors", ()) or ()),
        ContactSensorCfg(
            name="self_collision",
            # Exclude only primary wrist bodies; wrist vs torso is still caught by torso.
            primary=ContactMatch(
                mode="body",
                pattern=r".*",
                entity="robot",
                exclude=excluded_body_names,
            ),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            history_length=4,
        ),
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={
            "sensor_name": "self_collision",
            "force_threshold": 1.0,
        },
    )


def _configure_feet_acc_reward(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.rewards["feet_acc"] = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-6,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=r".*ankle.*"),
        },
    )


def _configure_additional_wrist_pos_reward(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.rewards["additional_wrist_pos"] = RewardTermCfg(
        func=mdp.motion_relative_body_point_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.12,
            "body_names": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
            "body_offsets": ((0.18, -0.025, 0.0), (0.18, 0.025, 0.0)),
        },
    )


def make_general_tracking_env_cfg(
    *, play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the General-Tracking-G1 training env."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": make_g1_training_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = G1_ACTION_SCALE

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    motion_cmd.body_names = _TRACKING_BODY_NAMES
    motion_cmd.motion_file = DEFAULT_TRAIN_MOTION_FILE
    motion_cmd.sampling_mode = "rewind"
    motion_cmd.window_steps = (0,)

    cfg.events["physics_material"].params[
        "asset_cfg"
    ].geom_names = r".*_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.events["randomize_rigid_body_mass"].params[
        "asset_cfg"
    ].body_names = "torso_link"
    _configure_self_collision_reward(cfg)
    _configure_feet_acc_reward(cfg)
    _configure_additional_wrist_pos_reward(cfg)
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    cfg.terminations["anchor_pos"].params["threshold"] = 0.25
    cfg.terminations["anchor_ori"].params["threshold"] = 1.0
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.25
    cfg.viewer.body_name = "torso_link"
    cfg.episode_length_s = 10.0
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    actor_terms = {
        key: value
        for key, value in cfg.observations["actor"].terms.items()
        if key not in {"ref_anchor_pos_b", "robot_base_lin_vel_b"}
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
