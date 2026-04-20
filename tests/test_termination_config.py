from __future__ import annotations

from types import SimpleNamespace

import torch

from train_mimic.app import DEFAULT_TASK
from train_mimic.tasks.tracking import mdp


def test_general_tracking_termination_config_matches_requested_policy() -> None:
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
    assert anchor_pos.func is mdp.bad_anchor_pos_z_only_adaptive
    assert anchor_pos.params == {
        "command_name": "motion",
        "threshold": 0.15,
        "down_threshold": 0.4,
        "root_height_threshold": 0.5,
    }

    anchor_ori = terminations["anchor_ori"]
    assert anchor_ori.func is mdp.bad_anchor_ori
    assert anchor_ori.params["threshold"] == 1.0

    ee_body_pos = terminations["ee_body_pos"]
    assert ee_body_pos.func is mdp.bad_motion_body_pos_z_only_adaptive
    assert ee_body_pos.params == {
        "command_name": "motion",
        "threshold": 0.15,
        "down_threshold": 0.4,
        "root_height_threshold": 0.5,
        "body_names": (
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
    }

def test_adaptive_height_termination_uses_relaxed_threshold_for_low_reference() -> None:
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=("left_ankle_roll_link",)),
        anchor_pos_w=torch.tensor([[0.0, 0.0, 0.3], [0.0, 0.0, 0.8]], dtype=torch.float32),
        robot_anchor_pos_w=torch.tensor([[0.0, 0.0, -0.09], [0.0, 0.0, 0.55]], dtype=torch.float32),
        body_pos_relative_w=torch.tensor(
            [
                [[0.0, 0.0, 0.30]],
                [[0.0, 0.0, 0.80]],
            ],
            dtype=torch.float32,
        ),
        robot_body_pos_w=torch.tensor(
            [
                [[0.0, 0.0, -0.09]],
                [[0.0, 0.0, 0.55]],
            ],
            dtype=torch.float32,
        ),
    )
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _name: command),
    )

    anchor_done = mdp.bad_anchor_pos_z_only_adaptive(
        env,
        "motion",
        threshold=0.15,
        down_threshold=0.4,
        root_height_threshold=0.5,
    )
    ee_done = mdp.bad_motion_body_pos_z_only_adaptive(
        env,
        "motion",
        threshold=0.15,
        down_threshold=0.4,
        root_height_threshold=0.5,
        body_names=("left_ankle_roll_link",),
    )

    assert anchor_done.tolist() == [False, True]
    assert ee_done.tolist() == [False, True]
