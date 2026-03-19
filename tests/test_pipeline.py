from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from teleopit.pipeline import TeleopPipeline

from conftest import find_g1_xml_path, requires_mujoco


def test_pipeline_inherits_robot_action_decode_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            captured["robot_cfg"] = cfg

    class DummyController:
        _expected_obs_dim = 166
        _multi_input = True

        def __init__(self, cfg: object) -> None:
            captured["controller_cfg"] = cfg

        def reset(self) -> None:
            pass

    class DummyObsBuilder:
        total_obs_size = 166

        def __init__(self, cfg: object) -> None:
            captured["obs_cfg"] = cfg

        def reset(self) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"

        def __init__(self, **kwargs: object) -> None:
            captured["input_kwargs"] = kwargs

    class DummyRetargeter:
        def __init__(self, **kwargs: object) -> None:
            captured["retarget_kwargs"] = kwargs

    class DummyLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["loop_args"] = args
            captured["loop_kwargs"] = kwargs

    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"dummy")
    bvh_path = tmp_path / "input.bvh"
    bvh_path.write_text("HIERARCHY\n", encoding="utf-8")
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("<mujoco model='dummy'/>", encoding="utf-8")

    robot_action_scale = [0.1, 0.2, 0.3]
    robot_default_angles = [-0.4, 0.5, -0.6]
    cfg = OmegaConf.create(
        {
            "robot": {
                "num_actions": 3,
                "xml_path": str(xml_path),
                "default_angles": robot_default_angles,
                "action_scale": robot_action_scale,
            },
            "controller": {
                "policy_path": str(policy_path),
                "action_scale": None,
                "default_dof_pos": None,
            },
            "input": {
                "provider": "bvh",
                "bvh_file": str(bvh_path),
                "bvh_format": "lafan1",
                "robot_name": "unitree_g1",
            },
            "policy_hz": 50,
            "pd_hz": 1000,
            "transition_duration": 1.5,
            "realtime": True,
            "viewers": ["retarget", "sim2sim"],
        }
    )

    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController)
    monkeypatch.setattr("teleopit.runtime.factory.VelCmdObservationBuilder", DummyObsBuilder)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    TeleopPipeline(cfg)

    controller_cfg = captured["controller_cfg"]
    loop_cfg = captured["loop_args"][4]
    assert list(controller_cfg.default_dof_pos) == robot_default_angles
    assert list(controller_cfg.action_scale) == robot_action_scale
    assert loop_cfg["transition_duration"] == pytest.approx(1.5)
    assert loop_cfg["realtime"] is True
    assert captured["loop_kwargs"]["viewers"] == {"retarget", "sim2sim"}


def test_pipeline_rejects_legacy_viewer_key(monkeypatch, tmp_path: Path) -> None:
    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            pass

    class DummyController:
        _expected_obs_dim = 166
        _multi_input = True

        def __init__(self, cfg: object) -> None:
            pass

        def reset(self) -> None:
            pass

    class DummyObsBuilder:
        total_obs_size = 166

        def __init__(self, cfg: object) -> None:
            pass

        def reset(self) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"

        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyRetargeter:
        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"dummy")
    bvh_path = tmp_path / "input.bvh"
    bvh_path.write_text("HIERARCHY\n", encoding="utf-8")
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("<mujoco model='dummy'/>", encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "robot": {"num_actions": 3, "xml_path": str(xml_path), "default_angles": [0.0, 0.0, 0.0]},
            "controller": {"policy_path": str(policy_path)},
            "input": {"provider": "bvh", "bvh_file": str(bvh_path)},
            "policy_hz": 50,
            "pd_hz": 1000,
            "viewer": True,
        }
    )

    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController)
    monkeypatch.setattr("teleopit.runtime.factory.VelCmdObservationBuilder", DummyObsBuilder)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    with pytest.raises(ValueError, match="Legacy config key 'viewer'"):
        TeleopPipeline(cfg)


_XML_PATH = find_g1_xml_path()
_skip_no_xml = pytest.mark.skipif(_XML_PATH is None, reason="Robot XML not found")

_DEFAULT_ANGLES_29 = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]


def _pipeline_cfg(tmp_path: Path) -> dict:
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"dummy")
    bvh_path = tmp_path / "input.bvh"
    bvh_path.write_text("HIERARCHY\n", encoding="utf-8")

    return {
        "robot": {
            "num_actions": 29,
            "xml_path": _XML_PATH or "",
            "default_angles": _DEFAULT_ANGLES_29,
            "action_scale": [0.5] * 29,
        },
        "controller": {
            "policy_path": str(policy_path),
            "action_scale": None,
            "default_dof_pos": None,
        },
        "input": {
            "provider": "bvh",
            "bvh_file": str(bvh_path),
            "bvh_format": "lafan1",
            "robot_name": "unitree_g1",
        },
        "policy_hz": 50,
        "pd_hz": 1000,
    }


@requires_mujoco
@_skip_no_xml
def test_pipeline_166d_policy_required(monkeypatch, tmp_path: Path) -> None:
    class DummyController160:
        _expected_obs_dim = 160
        _multi_input = True

        def __init__(self, cfg: object) -> None:
            pass

        def reset(self) -> None:
            pass

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"

        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyRetargeter:
        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController160)
    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    cfg = OmegaConf.create(_pipeline_cfg(tmp_path))
    with pytest.raises(ValueError, match="Only 166D"):
        TeleopPipeline(cfg)


@requires_mujoco
@_skip_no_xml
def test_pipeline_requires_dual_input_policy(monkeypatch, tmp_path: Path) -> None:
    class DummyControllerSingle:
        _expected_obs_dim = 166
        _multi_input = False

        def __init__(self, cfg: object) -> None:
            pass

        def reset(self) -> None:
            pass

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"

        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyRetargeter:
        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyControllerSingle)
    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    cfg = OmegaConf.create(_pipeline_cfg(tmp_path))
    with pytest.raises(ValueError, match="dual inputs"):
        TeleopPipeline(cfg)


@requires_mujoco
@_skip_no_xml
def test_pipeline_dim_match_passes(monkeypatch, tmp_path: Path) -> None:
    class DummyController166:
        _expected_obs_dim = 166
        _multi_input = True

        def __init__(self, cfg: object) -> None:
            pass

        def reset(self) -> None:
            pass

    class DummyRobot:
        def __init__(self, cfg: object) -> None:
            pass

    class DummyInputProvider:
        human_format = "lafan1"

        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyRetargeter:
        def __init__(self, **kwargs: object) -> None:
            pass

    class DummyLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr("teleopit.pipeline.RLPolicyController", DummyController166)
    monkeypatch.setattr("teleopit.pipeline.MuJoCoRobot", DummyRobot)
    monkeypatch.setattr("teleopit.pipeline.BVHInputProvider", DummyInputProvider)
    monkeypatch.setattr("teleopit.pipeline.RetargetingModule", DummyRetargeter)
    monkeypatch.setattr("teleopit.pipeline.SimulationLoop", DummyLoop)

    cfg = OmegaConf.create(_pipeline_cfg(tmp_path))
    pipeline = TeleopPipeline(cfg)
    assert pipeline.obs_builder.total_obs_size == 166
