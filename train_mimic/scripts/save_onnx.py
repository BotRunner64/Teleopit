#!/usr/bin/env python3
"""Export a VelCmdHistory policy checkpoint to ONNX."""

from __future__ import annotations

import argparse

import torch

from train_mimic.app import validate_checkpoint_path


def _extract_full_actor_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract the raw actor state dict from supported checkpoint layouts."""
    if not isinstance(checkpoint, dict):
        return {}

    if "actor_state_dict" in checkpoint and isinstance(checkpoint["actor_state_dict"], dict):
        return dict(checkpoint["actor_state_dict"])

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        return {}

    actor_state: dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if key.startswith("actor."):
            actor_state[key[len("actor."):]] = val
    return actor_state


def _has_temporal_cnn_keys(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("cnn_encoders.") for k in state_dict)


def _resolve_normalizer_stats(state_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mean_keys = ("obs_normalizer._mean", "obs_normalizer.mean")
    var_keys = ("obs_normalizer._var", "obs_normalizer.var")

    mean = next((state_dict[key] for key in mean_keys if key in state_dict), None)
    var = next((state_dict[key] for key in var_keys if key in state_dict), None)
    if mean is None or var is None:
        raise KeyError(
            "TemporalCNN checkpoint missing obs normalizer stats. "
            f"Tried mean keys {mean_keys} and var keys {var_keys}."
        )
    return mean, var


def _canonicalize_temporal_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    if "obs_normalizer.mean" in normalized and "obs_normalizer._mean" not in normalized:
        normalized["obs_normalizer._mean"] = normalized.pop("obs_normalizer.mean")
    if "obs_normalizer.var" in normalized and "obs_normalizer._var" not in normalized:
        normalized["obs_normalizer._var"] = normalized.pop("obs_normalizer.var")
    return normalized


def _export_temporal_cnn_policy(
    actor_state_dict: dict[str, torch.Tensor],
    output_path: str,
    history_length: int,
    activation: str,
    opset_version: int,
) -> None:
    from tensordict import TensorDict

    from train_mimic.tasks.tracking.rl.temporal_cnn_model import TemporalCNNModel

    sd = _canonicalize_temporal_state_dict(actor_state_dict)
    obs_mean, _obs_var = _resolve_normalizer_stats(sd)
    obs_dim = int(obs_mean.shape[-1])

    mlp_weights: list[tuple[int, torch.Tensor]] = []
    for key, val in sd.items():
        if key.startswith("mlp.") and key.endswith(".weight"):
            idx = int(key.split(".")[1])
            mlp_weights.append((idx, val))
    mlp_weights.sort()
    hidden_dims = tuple(int(w.shape[0]) for _, w in mlp_weights[:-1])
    output_dim = int(mlp_weights[-1][1].shape[0])

    cnn_conv_weights: list[tuple[int, torch.Tensor]] = []
    for key, val in sd.items():
        if "cnn_encoders." in key and ".net." in key and key.endswith(".weight") and val.ndim == 3:
            idx = int(key.split(".net.")[1].split(".")[0])
            cnn_conv_weights.append((idx, val))
    cnn_conv_weights.sort()
    if not cnn_conv_weights:
        raise RuntimeError("No Conv1d weights found in checkpoint despite cnn_encoders keys.")
    output_channels = tuple(int(w.shape[0]) for _, w in cnn_conv_weights)
    kernel_size = int(cnn_conv_weights[0][1].shape[2])

    dist_cfg = None
    if any("distribution." in k for k in sd):
        dist_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }

    print(
        f"  Architecture: obs_dim={obs_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}"
    )
    print(
        f"  CNN: output_channels={output_channels}, kernel_size={kernel_size}, "
        f"history_length={history_length}"
    )

    dummy_obs = TensorDict(
        {
            "actor": torch.zeros(1, obs_dim),
            "actor_history": torch.zeros(1, history_length, obs_dim),
        },
        batch_size=[1],
    )

    model = TemporalCNNModel(
        obs=dummy_obs,
        obs_groups={"actor": ("actor", "actor_history")},
        obs_set="actor",
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        obs_normalization=True,
        distribution_cfg=dist_cfg,
        cnn_cfg={
            "output_channels": output_channels,
            "kernel_size": kernel_size,
            "activation": activation,
            "global_pool": "avg",
        },
    )
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded TemporalCNNModel weights: {len(sd)} tensors")

    onnx_model = model.as_onnx()
    onnx_model.eval()
    dummy_inputs = onnx_model.get_dummy_inputs()
    input_names = onnx_model.input_names
    output_names = onnx_model.output_names

    torch.onnx.export(
        onnx_model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={},
        dynamo=False,
    )

    shapes = [tuple(d.shape) for d in dummy_inputs]
    print(f"Exported TemporalCNN ONNX model to: {output_path}")
    print(f"  Inputs:  {list(zip(input_names, shapes))}")
    print(f"  Outputs: {output_names}")


def export_policy_as_onnx(
    checkpoint_path: str,
    output_path: str,
    activation: str = "elu",
    opset_version: int = 18,
    history_length: int = 10,
) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Cannot load checkpoint — missing module '{e.name}'.\n"
                "This is likely an old Isaac Lab checkpoint (train_mimic.rsl_rl). "
                "Only mjlab rsl_rl 5.x checkpoints from logs/rsl_rl/g1_tracking_velcmd_history/ are supported."
            ) from e

    full_actor_sd = _extract_full_actor_state_dict(checkpoint)
    if not _has_temporal_cnn_keys(full_actor_sd):
        raise RuntimeError(
            "Only VelCmdHistory TemporalCNN checkpoints are supported. "
            "Expected dual-input actor weights with cnn_encoders.* keys."
        )

    print("Detected VelCmdHistory TemporalCNN checkpoint.")
    _export_temporal_cnn_policy(
        actor_state_dict=full_actor_sd,
        output_path=output_path,
        history_length=history_length,
        activation=activation,
        opset_version=opset_version,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export VelCmdHistory policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--opset_version", type=int, default=18)
    parser.add_argument(
        "--history_length",
        type=int,
        default=10,
        help="History length for VelCmdHistory checkpoints (default: 10)",
    )
    args = parser.parse_args()

    try:
        validate_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    export_policy_as_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        activation=args.activation,
        opset_version=args.opset_version,
        history_length=args.history_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
