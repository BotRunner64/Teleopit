from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

from train_mimic.scripts.save_onnx import export_policy_as_onnx
from train_mimic.tasks.tracking.rl.temporal_cnn_model import TemporalCNNModel


def _build_temporal_actor_model(
    *,
    obs_dim: int = 166,
    history_length: int = 10,
    ref_window_dim: int | None = None,
    ref_window_length: int = 20,
    output_dim: int = 29,
) -> TemporalCNNModel:
    obs_dict: dict[str, torch.Tensor] = {
        "actor": torch.zeros(1, obs_dim),
        "actor_history": torch.zeros(1, history_length, obs_dim),
    }
    obs_groups: tuple[str, ...]
    if ref_window_dim is None:
        obs_groups = ("actor", "actor_history")
    else:
        obs_dict["actor_ref_window"] = torch.zeros(1, ref_window_length, ref_window_dim)
        obs_groups = ("actor", "actor_history", "actor_ref_window")

    obs = TensorDict(obs_dict, batch_size=[1])
    return TemporalCNNModel(
        obs=obs,
        obs_groups={"actor": obs_groups},
        obs_set="actor",
        output_dim=output_dim,
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


def _build_temporal_actor_state_dict(obs_dim: int = 166, history_length: int = 10) -> dict[str, torch.Tensor]:
    return _build_temporal_actor_model(obs_dim=obs_dim, history_length=history_length).state_dict()


def _build_multi_group_temporal_actor_state_dict(
    *,
    obs_dim: int = 8,
    history_length: int = 10,
    ref_window_dim: int = 6,
    ref_window_length: int = 20,
    output_dim: int = 4,
) -> dict[str, torch.Tensor]:
    return _build_temporal_actor_model(
        obs_dim=obs_dim,
        history_length=history_length,
        ref_window_dim=ref_window_dim,
        ref_window_length=ref_window_length,
        output_dim=output_dim,
    ).state_dict()


def _write_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    torch.save(checkpoint, path)


def test_temporal_cnn_onnx_wrapper_supports_multiple_temporal_groups() -> None:
    model = _build_temporal_actor_model(
        obs_dim=8,
        history_length=10,
        ref_window_dim=6,
        ref_window_length=20,
        output_dim=4,
    )

    onnx_model = model.as_onnx(verbose=False)
    dummy_inputs = onnx_model.get_dummy_inputs()
    output = onnx_model(*dummy_inputs)

    assert onnx_model.input_names == ["obs", "obs_history", "obs_ref_window"]
    assert [tuple(arg.shape) for arg in dummy_inputs] == [(1, 8), (1, 10, 8), (1, 20, 6)]
    assert tuple(output.shape) == (1, 4)


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


def test_export_temporal_cnn_multi_group_checkpoint(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_export(model, args, output_path, **kwargs):  # type: ignore[no-untyped-def]
        captured["output_path"] = output_path
        captured["input_names"] = kwargs["input_names"]
        captured["arg_shapes"] = [tuple(arg.shape) for arg in args]

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    actor_state = _build_multi_group_temporal_actor_state_dict()
    ckpt_path = tmp_path / "temporal_multi_group.pt"
    _write_checkpoint(ckpt_path, {"actor_state_dict": actor_state})

    export_policy_as_onnx(str(ckpt_path), str(tmp_path / "policy.onnx"), history_length=10)

    assert captured["output_path"] == str(tmp_path / "policy.onnx")
    assert captured["input_names"] == ["obs", "obs_history", "obs_ref_window"]
    assert captured["arg_shapes"] == [(1, 8), (1, 10, 8), (1, 20, 6)]


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
