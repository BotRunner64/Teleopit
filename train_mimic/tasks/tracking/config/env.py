"""Environment builders for supported tracking tasks."""

from __future__ import annotations

from copy import deepcopy

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from train_mimic.tasks.tracking import mdp
from train_mimic.tasks.tracking.config.constants import DEFAULT_TRAIN_MOTION_FILE
from train_mimic.tasks.tracking.mdp import MotionCommandCfg, MotionTrackingCommandCfg
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

_MOTION_KEYPOINT_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_knee_link",
    "right_knee_link",
)

_MOTION_LOWER_KEYPOINT_BODY_NAMES = (
    "pelvis",
    "left_knee_link",
    "right_knee_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)

_MOTION_UPPER_KEYPOINT_BODY_NAMES = (
    "torso_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

_MOTION_BODY_Z_TERMINATE_NAMES = (
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

_MOTION_TRACKING_DEPLOY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

_MOTION_TRACKING_DEPLOY_ACTION_SCALE = dict(
    zip(
        _MOTION_TRACKING_DEPLOY_JOINT_NAMES,
        (
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ),
        strict=True,
    )
)
_MOTION_TRACKING_DEPLOY_REFERENCE_STEPS = (0, 1, 2, 3, 4, -1, -2, -4, -8, -12, -16)
_MOTION_TRACKING_DEPLOY_HISTORY_STEPS = (0, 1, 2, 3, 4, 8, 12, 16, 20)
_MOTION_TRACKING_DEPLOY_PREV_ACTION_STEPS = 8
_MOTION_TRACKING_DEPLOY_COMPLIANCE_VALUE = 1.0
_MOTION_TRACKING_DEPLOY_COMPLIANCE_THRESHOLD = 10.0


def _apply_play_mode_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MotionTrackingCommandCfg))

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

_MOTION_DEPLOY_ACTOR_TERMS: dict[str, ObservationTermCfg] = {
    "boot_indicator": ObservationTermCfg(func=mdp.motion_tracking_boot_indicator),
    "tracking_command": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_command_obs,
        params={"command_name": "motion"},
    ),
    "compliance_flag": ObservationTermCfg(
        func=mdp.motion_tracking_compliance_flag,
        params={
            "value": _MOTION_TRACKING_DEPLOY_COMPLIANCE_VALUE,
            "threshold": _MOTION_TRACKING_DEPLOY_COMPLIANCE_THRESHOLD,
        },
    ),
    "target_joint_pos": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_target_joint_pos,
        params={"command_name": "motion"},
    ),
    "target_root_z": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_target_root_z,
        params={"command_name": "motion", "offset": 0.035},
    ),
    "target_projected_gravity_b": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_target_projected_gravity,
        params={"command_name": "motion"},
    ),
    "root_ang_vel_history": ObservationTermCfg(
        func=mdp.MotionTrackingDeployHistoryTerm,
        params={"source": "root_ang_vel_b", "steps": _MOTION_TRACKING_DEPLOY_HISTORY_STEPS},
    ),
    "projected_gravity_history": ObservationTermCfg(
        func=mdp.MotionTrackingDeployHistoryTerm,
        params={"source": "projected_gravity_b", "steps": _MOTION_TRACKING_DEPLOY_HISTORY_STEPS},
    ),
    "joint_pos_history": ObservationTermCfg(
        func=mdp.MotionTrackingDeployHistoryTerm,
        params={"source": "joint_pos", "steps": _MOTION_TRACKING_DEPLOY_HISTORY_STEPS},
    ),
    "joint_vel_history": ObservationTermCfg(
        func=mdp.MotionTrackingDeployHistoryTerm,
        params={"source": "joint_vel", "steps": _MOTION_TRACKING_DEPLOY_HISTORY_STEPS},
    ),
    "prev_actions": ObservationTermCfg(
        func=mdp.MotionTrackingPrevActionHistoryTerm,
        params={"steps": _MOTION_TRACKING_DEPLOY_PREV_ACTION_STEPS},
    ),
}

_MOTION_DEPLOY_CRITIC_TERMS: dict[str, ObservationTermCfg] = {
    "boot_indicator": ObservationTermCfg(func=mdp.motion_tracking_boot_indicator),
    "tracking_command": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_command_obs,
        params={"command_name": "motion"},
    ),
    "compliance_flag": ObservationTermCfg(
        func=mdp.motion_tracking_compliance_flag,
        params={
            "value": _MOTION_TRACKING_DEPLOY_COMPLIANCE_VALUE,
            "threshold": _MOTION_TRACKING_DEPLOY_COMPLIANCE_THRESHOLD,
        },
    ),
    "target_joint_pos": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_target_joint_pos,
        params={"command_name": "motion"},
    ),
    "target_root_z": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_target_root_z,
        params={"command_name": "motion", "offset": 0.035},
    ),
    "target_projected_gravity_b": ObservationTermCfg(
        func=mdp.motion_tracking_deploy_target_projected_gravity,
        params={"command_name": "motion"},
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.motion_tracking_projected_gravity),
    "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(func=mdp.motion_tracking_joint_pos_abs),
    "joint_vel": ObservationTermCfg(func=mdp.motion_tracking_joint_vel_abs),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "current_keypoint_pos_b": ObservationTermCfg(
        func=mdp.motion_tracking_current_keypoint_pos_b,
        params={"command_name": "motion"},
    ),
    "target_keypoint_pos_b": ObservationTermCfg(
        func=mdp.motion_tracking_target_keypoint_pos_b,
        params={"command_name": "motion"},
    ),
    "current_keypoint_rot_b": ObservationTermCfg(
        func=mdp.motion_tracking_current_keypoint_rot_b,
        params={"command_name": "motion"},
    ),
    "target_keypoint_rot_b": ObservationTermCfg(
        func=mdp.motion_tracking_target_keypoint_rot_b,
        params={"command_name": "motion"},
    ),
    "current_keypoint_linvel_b": ObservationTermCfg(
        func=mdp.motion_tracking_current_keypoint_linvel_b,
        params={"command_name": "motion"},
    ),
    "target_keypoint_linvel_b": ObservationTermCfg(
        func=mdp.motion_tracking_target_keypoint_linvel_b,
        params={"command_name": "motion"},
    ),
    "current_keypoint_angvel_b": ObservationTermCfg(
        func=mdp.motion_tracking_current_keypoint_angvel_b,
        params={"command_name": "motion"},
    ),
    "target_keypoint_angvel_b": ObservationTermCfg(
        func=mdp.motion_tracking_target_keypoint_angvel_b,
        params={"command_name": "motion"},
    ),
    "body_height": ObservationTermCfg(
        func=mdp.motion_tracking_body_height_obs,
        params={"command_name": "motion"},
    ),
}


def make_velcmd_history_tracking_env_cfg(
    *, play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the legacy G1 VelCmdHistory training env."""
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
    motion_cmd.window_steps = (0,)

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


def make_motion_tracking_deploy_env_cfg(
    *, play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create a single-stage actor observation aligned to the deployed motion-tracking policy."""
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
    joint_pos_action.scale = _MOTION_TRACKING_DEPLOY_ACTION_SCALE

    cfg.commands["motion"] = MotionTrackingCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        motion_file=DEFAULT_TRAIN_MOTION_FILE,
        anchor_body_name="torso_link",
        root_body_name="pelvis",
        body_names=_TRACKING_BODY_NAMES,
        keypoint_body_names=_MOTION_KEYPOINT_BODY_NAMES,
        lower_keypoint_body_names=_MOTION_LOWER_KEYPOINT_BODY_NAMES,
        upper_keypoint_body_names=_MOTION_UPPER_KEYPOINT_BODY_NAMES,
        feet_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        body_z_terminate_body_names=_MOTION_BODY_Z_TERMINATE_NAMES,
        tracking_joint_names=_MOTION_TRACKING_DEPLOY_JOINT_NAMES,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(-0.1, 0.1),
        sampling_mode="uniform",
        window_steps=_MOTION_TRACKING_DEPLOY_REFERENCE_STEPS,
        target_joint_pos_bias_range=(-0.1, 0.1),
        root_drift_vel_xy_max=0.25,
        root_drift_vel_z_max=0.05,
        root_z_offset_range=(-0.03, 0.03),
        feet_standing_z_threshold=0.18,
        feet_standing_vxy_threshold=0.2,
        feet_standing_vz_threshold=0.15,
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.viewer.body_name = "torso_link"
    cfg.episode_length_s = 10.0
    if cfg.sim.njmax < 500:
        cfg.sim.njmax = 500

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=deepcopy(_MOTION_DEPLOY_ACTOR_TERMS),
            concatenate_terms=True,
            enable_corruption=True,
        ),
        "critic": ObservationGroupCfg(
            terms=deepcopy(_MOTION_DEPLOY_CRITIC_TERMS),
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    cfg.rewards = {
        "root_pos_tracking": RewardTermCfg(
            func=mdp.motion_tracking_root_pos_tracking,
            weight=0.5,
            params={"command_name": "motion", "sigma": [0.3]},
        ),
        "root_rot_tracking": RewardTermCfg(
            func=mdp.motion_tracking_root_rot_tracking,
            weight=0.5,
            params={"command_name": "motion", "sigma": [0.4]},
        ),
        "root_vel_tracking": RewardTermCfg(
            func=mdp.motion_tracking_root_vel_tracking,
            weight=1.0,
            params={"command_name": "motion", "sigma": [1.0]},
        ),
        "root_ang_vel_tracking": RewardTermCfg(
            func=mdp.motion_tracking_root_ang_vel_tracking,
            weight=1.0,
            params={"command_name": "motion", "sigma": [3.0]},
        ),
        "keypoint_tracking": RewardTermCfg(
            func=mdp.motion_tracking_keypoint_tracking,
            weight=1.0,
            params={"command_name": "motion", "sigma": [0.3]},
        ),
        "keypoint_vel_tracking": RewardTermCfg(
            func=mdp.motion_tracking_keypoint_vel_tracking,
            weight=1.0,
            params={"command_name": "motion", "sigma": [1.0]},
        ),
        "keypoint_rot_tracking": RewardTermCfg(
            func=mdp.motion_tracking_keypoint_rot_tracking,
            weight=1.0,
            params={"command_name": "motion", "sigma": [0.4]},
        ),
        "keypoint_angvel_tracking": RewardTermCfg(
            func=mdp.motion_tracking_keypoint_angvel_tracking,
            weight=1.0,
            params={"command_name": "motion", "sigma": [3.0]},
        ),
        "lower_keypoint_tracking": RewardTermCfg(
            func=mdp.motion_tracking_lower_keypoint_tracking,
            weight=0.5,
            params={"command_name": "motion", "sigma": [0.2]},
        ),
        "upper_keypoint_tracking": RewardTermCfg(
            func=mdp.motion_tracking_upper_keypoint_tracking,
            weight=0.5,
            params={"command_name": "motion", "sigma": [0.3]},
        ),
        "joint_pos_tracking": RewardTermCfg(
            func=mdp.motion_tracking_joint_pos_tracking,
            weight=1.0,
            params={"command_name": "motion", "sigma": [0.5]},
        ),
        "joint_vel_tracking": RewardTermCfg(
            func=mdp.motion_tracking_joint_vel_tracking,
            weight=0.5,
            params={"command_name": "motion", "sigma": [3.0]},
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1.0e-1),
        "joint_limit": RewardTermCfg(
            func=mdp.joint_pos_limits,
            weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-10.0,
            params={"sensor_name": "self_collision", "force_threshold": 10.0},
        ),
    }

    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "body_z_tracking_failure": TerminationTermCfg(
            func=mdp.motion_tracking_body_z_failure,
            params={
                "command_name": "motion",
                "threshold": 0.25,
                "body_names": _MOTION_BODY_Z_TERMINATE_NAMES,
            },
        ),
        "gravity_tracking_failure": TerminationTermCfg(
            func=mdp.motion_tracking_gravity_failure,
            params={"command_name": "motion", "threshold": 0.5},
        ),
    }

    if play:
        _apply_play_mode_overrides(cfg)
        cfg.observations["actor"].enable_corruption = False

    return cfg
