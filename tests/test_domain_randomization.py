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
        "add_joint_default_pos",
        "physics_material",
        "randomize_rigid_body_mass",
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

    add_joint_default_pos = events["add_joint_default_pos"]
    assert add_joint_default_pos.func is dr.joint_default_pos
    assert add_joint_default_pos.mode == "startup"
    assert add_joint_default_pos.params["asset_cfg"].joint_names == ".*"
    assert add_joint_default_pos.params["operation"] == "add"
    assert add_joint_default_pos.params["ranges"] == (-0.01, 0.01)

    physics_material = events["physics_material"]
    assert physics_material.func is dr.geom_friction
    assert physics_material.mode == "startup"
    assert physics_material.params["asset_cfg"].geom_names == r".*_collision$"
    assert physics_material.params["operation"] == "abs"
    assert physics_material.params["ranges"] == (0.3, 1.6)

    mass = events["randomize_rigid_body_mass"]
    assert mass.func is dr.pseudo_inertia
    assert mass.mode == "startup"
    assert mass.params["asset_cfg"].body_names == (
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    assert mass.params["alpha_range"] == (-0.1, 0.45)


def test_play_env_disables_training_only_domain_randomization() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    play_cfg = load_env_cfg(DEFAULT_TASK, play=True)

    assert "push_robot" not in play_cfg.events
    assert "base_com" not in play_cfg.events
    assert "add_joint_default_pos" not in play_cfg.events
    assert "physics_material" not in play_cfg.events
    assert "randomize_rigid_body_mass" not in play_cfg.events
    assert play_cfg.events == {}
