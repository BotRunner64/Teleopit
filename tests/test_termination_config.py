from __future__ import annotations

from train_mimic.app import DEFAULT_TASK
from train_mimic.tasks.tracking import mdp


def test_general_tracking_termination_config_matches_baseline_policy() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(DEFAULT_TASK)
    terminations = env_cfg.terminations

    assert set(terminations) == {
        "time_out",
        "anchor_pos",
        "anchor_ori",
        "ee_body_pos",
    }

    anchor_pos = terminations["anchor_pos"]
    assert anchor_pos.func is mdp.bad_anchor_pos_z_only
    assert anchor_pos.params == {
        "command_name": "motion",
        "threshold": 0.4,
    }

    anchor_ori = terminations["anchor_ori"]
    assert anchor_ori.func is mdp.bad_anchor_ori
    assert anchor_ori.params["threshold"] == 1.0

    ee_body_pos = terminations["ee_body_pos"]
    assert ee_body_pos.func is mdp.bad_motion_body_pos_z_only
    assert ee_body_pos.params == {
        "command_name": "motion",
        "threshold": 0.4,
        "body_names": (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
    }
