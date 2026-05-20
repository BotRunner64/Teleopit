from __future__ import annotations

from train_mimic.app import DEFAULT_TASK
from train_mimic.tasks.tracking import mdp


def test_general_tracking_domain_randomization_matches_gr00t_active_set() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.envs.mdp import dr
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(DEFAULT_TASK)
    events = env_cfg.events

    assert set(events) == {
        "push_robot",
        "base_com",
        "encoder_bias",
        "add_joint_default_pos",
        "motor_params_implicit_upper_body_pd",
        "motor_params_implicit_lower_body_pd",
        "motor_params_implicit_armature",
        "physics_material",
        "randomize_rigid_body_mass",
        "randomize_dexhand_payload_mass",
        "randomize_gimbal_payload_mass",
        "randomize_dexhand_payload_pos",
        "randomize_gimbal_payload_pos",
    }

    push_robot = events["push_robot"]
    assert push_robot.func is mdp.push_by_setting_velocity
    assert push_robot.mode == "interval"
    assert push_robot.interval_range_s == (4.0, 6.0)
    assert push_robot.params["velocity_range"] == {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.2, 0.2),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
    }

    base_com = events["base_com"]
    assert base_com.func is dr.body_com_offset
    assert base_com.mode == "startup"
    assert base_com.params["asset_cfg"].body_names == ("torso_link",)
    assert base_com.params["operation"] == "add"
    assert base_com.params["ranges"] == {
        0: (-0.025, 0.025),
        1: (-0.05, 0.05),
        2: (-0.05, 0.05),
    }

    encoder_bias = events["encoder_bias"]
    assert encoder_bias.func is dr.encoder_bias
    assert encoder_bias.mode == "startup"
    assert encoder_bias.params["bias_range"] == (-0.01, 0.01)

    add_joint_default_pos = events["add_joint_default_pos"]
    assert add_joint_default_pos.func is dr.joint_default_pos
    assert add_joint_default_pos.mode == "startup"
    assert add_joint_default_pos.params["asset_cfg"].joint_names == ".*"
    assert add_joint_default_pos.params["operation"] == "add"
    assert add_joint_default_pos.params["ranges"] == (-0.01, 0.01)

    upper_motor_pd = events["motor_params_implicit_upper_body_pd"]
    assert upper_motor_pd.func is dr.pd_gains
    assert upper_motor_pd.mode == "reset"
    assert upper_motor_pd.params["asset_cfg"].actuator_names is None
    assert upper_motor_pd.params["asset_cfg"].actuator_ids == [0, 3]
    assert upper_motor_pd.params["kp_range"] == (0.9, 1.1)
    assert upper_motor_pd.params["kd_range"] == (0.9, 1.1)
    assert upper_motor_pd.params["distribution"] == "log_uniform"
    assert upper_motor_pd.params["operation"] == "scale"

    lower_motor_pd = events["motor_params_implicit_lower_body_pd"]
    assert lower_motor_pd.func is dr.pd_gains
    assert lower_motor_pd.mode == "reset"
    assert lower_motor_pd.params["asset_cfg"].actuator_names is None
    assert lower_motor_pd.params["asset_cfg"].actuator_ids == [1, 2, 4, 5]
    assert lower_motor_pd.params["kp_range"] == (0.5, 2.0)
    assert lower_motor_pd.params["kd_range"] == (0.5, 2.0)
    assert lower_motor_pd.params["distribution"] == "log_uniform"
    assert lower_motor_pd.params["operation"] == "scale"

    motor_armature = events["motor_params_implicit_armature"]
    assert motor_armature.func is dr.joint_armature
    assert motor_armature.mode == "startup"
    assert motor_armature.params["asset_cfg"].joint_names == ".*"
    assert motor_armature.params["ranges"] == (0.75, 1.25)
    assert motor_armature.params["distribution"] == "log_uniform"
    assert motor_armature.params["operation"] == "scale"

    physics_material = events["physics_material"]
    assert physics_material.func is dr.geom_friction
    assert physics_material.mode == "startup"
    assert physics_material.params["asset_cfg"].geom_names == r".*_collision$"
    assert physics_material.params["operation"] == "abs"
    assert physics_material.params["ranges"] == (0.3, 1.6)

    mass = events["randomize_rigid_body_mass"]
    assert mass.func is dr.pseudo_inertia
    assert mass.mode == "startup"
    assert mass.params["asset_cfg"].body_names == "torso_link"
    assert mass.params["alpha_range"] == (-0.1, 0.45)

    dexhand_mass = events["randomize_dexhand_payload_mass"]
    assert dexhand_mass.func is dr.pseudo_inertia
    assert dexhand_mass.mode == "startup"
    assert dexhand_mass.params["asset_cfg"].body_names == (
        "left_dexhand_payload",
        "right_dexhand_payload",
    )
    assert dexhand_mass.params["alpha_range"] == (-8.0, 0.34657359027997264)

    gimbal_mass = events["randomize_gimbal_payload_mass"]
    assert gimbal_mass.func is dr.pseudo_inertia
    assert gimbal_mass.mode == "startup"
    assert gimbal_mass.params["asset_cfg"].body_names == ("head_gimbal_payload",)
    assert gimbal_mass.params["alpha_range"] == (-8.0, 0.34657359027997264)

    dexhand_pos = events["randomize_dexhand_payload_pos"]
    assert dexhand_pos.func is dr.body_pos
    assert dexhand_pos.mode == "startup"
    assert dexhand_pos.params["asset_cfg"].body_names == (
        "left_dexhand_payload",
        "right_dexhand_payload",
    )
    assert dexhand_pos.params["operation"] == "abs"
    assert dexhand_pos.params["ranges"] == {
        0: (0.04, 0.12),
        1: (-0.03, 0.03),
        2: (-0.03, 0.03),
    }

    gimbal_pos = events["randomize_gimbal_payload_pos"]
    assert gimbal_pos.func is dr.body_pos
    assert gimbal_pos.mode == "startup"
    assert gimbal_pos.params["asset_cfg"].body_names == ("head_gimbal_payload",)
    assert gimbal_pos.params["operation"] == "abs"
    assert gimbal_pos.params["ranges"] == {
        0: (0.03, 0.12),
        1: (-0.03, 0.03),
        2: (0.40, 0.50),
    }


def test_play_env_disables_training_only_domain_randomization() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    play_cfg = load_env_cfg(DEFAULT_TASK, play=True)

    assert "push_robot" not in play_cfg.events
    assert "base_com" not in play_cfg.events
    assert "encoder_bias" not in play_cfg.events
    assert "add_joint_default_pos" not in play_cfg.events
    assert "motor_params_implicit_upper_body_pd" not in play_cfg.events
    assert "motor_params_implicit_lower_body_pd" not in play_cfg.events
    assert "motor_params_implicit_armature" not in play_cfg.events
    assert "physics_material" not in play_cfg.events
    assert "randomize_rigid_body_mass" not in play_cfg.events
    assert "randomize_dexhand_payload_mass" not in play_cfg.events
    assert "randomize_gimbal_payload_mass" not in play_cfg.events
    assert "randomize_dexhand_payload_pos" not in play_cfg.events
    assert "randomize_gimbal_payload_pos" not in play_cfg.events
    assert play_cfg.events == {}
