from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

from train_mimic.scripts.save_onnx import export_policy_as_onnx
from train_mimic.tasks.tracking.rl.temporal_cnn_model import TemporalCNNModel


def _build_temporal_actor_state_dict(obs_dim: int = 166, history_length: int = 10) -> dict[str, torch.Tensor]:
    obs = TensorDict(
        {
            "actor": torch.zeros(1, obs_dim),
            "actor_history": torch.zeros(1, history_length, obs_dim),
        },
        batch_size=[1],
    )
    model = TemporalCNNModel(
        obs=obs,
        obs_groups={"actor": ("actor", "actor_history")},
        obs_set="actor",
        output_dim=29,
        hidden_dims=(32, 16),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        cnn_cfg={
            "output_channels": (8, 4),
            "kernel_size": 3,
            "activation": "elu",
            "global_pool": "avg",
        },
    )
    return model.state_dict()


def _write_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    torch.save(checkpoint, path)


def test_export_temporal_cnn_layout_a_checkpoint(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_export(model, args, output_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["output_path"] = output_path
        captured["input_names"] = kwargs["input_names"]
        captured["arg_shapes"] = [tuple(arg.shape) for arg in args]

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    actor_state = _build_temporal_actor_state_dict()
    checkpoint = {"model_state_dict": {f"actor.{key}": value for key, value in actor_state.items()}}
    ckpt_path = tmp_path / "temporal_layout_a.pt"
    _write_checkpoint(ckpt_path, checkpoint)

    export_policy_as_onnx(str(ckpt_path), str(tmp_path / "policy.onnx"), history_length=10)

    assert captured["output_path"] == str(tmp_path / "policy.onnx")
    assert captured["input_names"] == ["obs", "obs_history"]
    assert captured["arg_shapes"] == [(1, 166), (1, 10, 166)]


def test_export_temporal_cnn_accepts_normalizer_mean_var_keys(monkeypatch, tmp_path: Path) -> None:
    called = False

    def fake_export(model, args, output_path, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    actor_state = _build_temporal_actor_state_dict()
    actor_state["obs_normalizer.mean"] = actor_state.pop("obs_normalizer._mean")
    actor_state["obs_normalizer.var"] = actor_state.pop("obs_normalizer._var")
    ckpt_path = tmp_path / "temporal_mean_var.pt"
    _write_checkpoint(ckpt_path, {"actor_state_dict": actor_state})

    export_policy_as_onnx(str(ckpt_path), str(tmp_path / "policy.onnx"), history_length=10)

    assert called is True
