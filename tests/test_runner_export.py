from __future__ import annotations

from pathlib import Path

import torch

from train_mimic.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner


class _DummyOnnxModel(torch.nn.Module):
    input_names = ["obs"]
    output_names = ["actions"]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, :2]

    def get_dummy_inputs(self):
        return (torch.zeros(1, 3),)


class _DummyPolicy:
    def as_onnx(self, verbose: bool = False) -> _DummyOnnxModel:
        _ = verbose
        return _DummyOnnxModel()


class _DummyAlg:
    def get_policy(self) -> _DummyPolicy:
        return _DummyPolicy()


def test_runner_export_policy_to_onnx_defaults_to_deploy_signature(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_export(model, args, output_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["output_path"] = output_path
        captured["input_names"] = kwargs["input_names"]
        captured["output_names"] = kwargs["output_names"]
        captured["arg_shapes"] = [tuple(arg.shape) for arg in args]

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    runner = MotionTrackingOnPolicyRunner.__new__(MotionTrackingOnPolicyRunner)
    runner.alg = _DummyAlg()

    runner.export_policy_to_onnx(str(tmp_path))

    assert captured["output_path"] == str(tmp_path / "policy.onnx")
    assert captured["input_names"] == ["obs"]
    assert captured["output_names"] == ["actions"]
    assert captured["arg_shapes"] == [(1, 3)]
