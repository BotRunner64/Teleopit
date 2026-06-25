from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from teleopit.bus.topics import TOPIC_ACTION, TOPIC_MIMIC_OBS, TOPIC_ROBOT_STATE
from conftest import find_g1_xml_path


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


requires_mujoco = pytest.mark.skipif(not _has_module("mujoco"), reason="mujoco not installed")
requires_onnxruntime = pytest.mark.skipif(not _has_module("onnxruntime"), reason="onnxruntime not installed")
requires_mink = pytest.mark.skipif(not _has_module("mink"), reason="mink not installed")


def _asset_paths(project_root: Path) -> tuple[Path, Path, Path]:
    policy_raw = os.environ.get("TELEOPIT_TEST_POLICY_ONNX", "").strip()
    policy_env = Path(policy_raw).expanduser() if policy_raw else Path("__missing_policy__.onnx")
    bvh = project_root / "teleopit" / "retargeting" / "gmr" / "assets" / "xsens_bvh_test" / "251021_04_boxing_120Hz_cm_3DsMax.bvh"
    xml_raw = find_g1_xml_path()
    xml = Path(xml_raw) if xml_raw is not None else Path("__missing_robot__.xml")
    return policy_env, bvh, xml


@requires_mujoco
@requires_onnxruntime
@requires_mink
def test_bvh_to_mujoco_pipeline_stands(project_root: Path) -> None:
    from omegaconf import OmegaConf

    from teleopit.pipeline import TeleopPipeline

    policy_path, bvh_path, xml_path = _asset_paths(project_root)
    if not policy_path.exists() or not bvh_path.exists() or not xml_path.exists():
        pytest.skip("set TELEOPIT_TEST_POLICY_ONNX to a compatible 167D ONNX policy to run this e2e test")

    robot_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "robot" / "g1.yaml")
    controller_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "controller" / "rl_policy.yaml")
    input_cfg = OmegaConf.load(project_root / "teleopit" / "configs" / "input" / "bvh.yaml")

    robot_cfg.xml_path = str(xml_path)
    controller_cfg.policy_path = str(policy_path)
    controller_cfg.default_dof_pos = list(robot_cfg.default_angles)
    controller_cfg.action_scale = 0.0
    controller_cfg.clip_range = [-10.0, 10.0]

    input_cfg.bvh_file = str(bvh_path)
    input_cfg.provider = "bvh"
    input_cfg.bvh_format = "lafan1"
    input_cfg.human_format = "bvh_xsens"
    input_cfg.robot_name = "unitree_g1"

    cfg = OmegaConf.create(
        {
            "robot": robot_cfg,
            "controller": controller_cfg,
            "input": input_cfg,
                "policy_hz": 50,
                "pd_hz": 50,
        }
    )

    pipeline = TeleopPipeline(cfg)

    action_count: list[int] = []
    mimic_count: list[int] = []
    state_count: list[int] = []

    pipeline.bus.subscribe(TOPIC_ACTION, lambda _: action_count.append(1))
    pipeline.bus.subscribe(TOPIC_MIMIC_OBS, lambda _: mimic_count.append(1))
    pipeline.bus.subscribe(TOPIC_ROBOT_STATE, lambda _: state_count.append(1))

    result = pipeline.run(num_steps=100)

    assert float(result["root_height"]) > 0.3
    assert int(result["steps"]) == 100

    assert len(action_count) == 100
    assert len(mimic_count) == 100
    assert len(state_count) == 100

    latest_action = pipeline.bus.get_latest(TOPIC_ACTION)
    latest_mimic = pipeline.bus.get_latest(TOPIC_MIMIC_OBS)
    latest_state = pipeline.bus.get_latest(TOPIC_ROBOT_STATE)

    assert isinstance(latest_action, np.ndarray)
    assert latest_action.shape == (29,)
    assert isinstance(latest_mimic, np.ndarray)
    assert latest_mimic.shape == (35,)
    assert latest_state is not None
