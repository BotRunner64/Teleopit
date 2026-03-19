"""Integration tests for Sim2RealController VelCmdHistory startup validation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from conftest import find_g1_xml_path, requires_mujoco


_XML_PATH = find_g1_xml_path()
_skip_no_xml = pytest.mark.skipif(_XML_PATH is None, reason="Robot XML not found")

_DEFAULT_ANGLES_29 = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]


def _make_dummy_policy(expected_obs_dim: int, *, multi_input: bool = True) -> MagicMock:
    policy = MagicMock()
    policy._expected_obs_dim = expected_obs_dim
    policy._multi_input = multi_input
    return policy


def _make_sim2real_cfg(tmp_path: Path) -> dict:
    policy_path = tmp_path / "policy.onnx"
    policy_path.write_bytes(b"dummy")
    ref_bvh = tmp_path / "ref.bvh"
    ref_bvh.write_text("HIERARCHY\n", encoding="utf-8")

    return {
        "policy_hz": 50.0,
        "real_robot": {"network_interface": "lo"},
        "gamepad": {},
        "mocap_switch": {},
        "input": {
            "reference_bvh": str(ref_bvh),
            "udp_host": "",
            "udp_port": 1118,
            "bvh_format": "hc_mocap",
            "udp_timeout": 0.1,
            "robot_name": "unitree_g1",
        },
        "controller": {
            "policy_path": str(policy_path),
            "action_scale": [0.5] * 29,
            "default_dof_pos": list(_DEFAULT_ANGLES_29),
        },
        "robot": {
            "num_actions": 29,
            "xml_path": _XML_PATH or "",
            "default_angles": _DEFAULT_ANGLES_29,
            "action_scale": [0.5] * 29,
            "anchor_body_name": "torso_link",
        },
    }


def _apply_sim2real_mocks(monkeypatch, policy_mock: MagicMock) -> None:
    monkeypatch.setattr(
        "teleopit.sim2real.controller.UnitreeG1Robot",
        lambda cfg: MagicMock(check_mode=MagicMock(return_value={"name": "mock"})),
    )
    monkeypatch.setattr(
        "teleopit.sim2real.controller.UnitreeRemote",
        MagicMock,
    )

    dummy_provider = MagicMock()
    dummy_provider.human_format = "hc_mocap"
    monkeypatch.setattr(
        "teleopit.sim2real.controller.UDPBVHInputProvider",
        lambda **kw: dummy_provider,
    )
    monkeypatch.setattr(
        "teleopit.sim2real.controller.RetargetingModule",
        lambda **kw: MagicMock(),
    )
    monkeypatch.setattr(
        "teleopit.sim2real.controller.RLPolicyController",
        lambda cfg: policy_mock,
    )


@requires_mujoco
@_skip_no_xml
class TestSim2RealStartupDim:
    def test_velcmd_history_startup_builds_166d_obs(self, monkeypatch, tmp_path: Path) -> None:
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(166, multi_input=True)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path)

        ctrl = Sim2RealController(cfg)
        assert ctrl.obs_builder.total_obs_size == 166

    def test_non_166_policy_is_rejected(self, monkeypatch, tmp_path: Path) -> None:
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(160, multi_input=True)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path)

        with pytest.raises(ValueError, match="Only 166D"):
            Sim2RealController(cfg)

    def test_single_input_policy_is_rejected(self, monkeypatch, tmp_path: Path) -> None:
        from teleopit.sim2real.controller import Sim2RealController

        policy_mock = _make_dummy_policy(166, multi_input=False)
        _apply_sim2real_mocks(monkeypatch, policy_mock)
        cfg = _make_sim2real_cfg(tmp_path)

        with pytest.raises(ValueError, match="dual inputs"):
            Sim2RealController(cfg)
