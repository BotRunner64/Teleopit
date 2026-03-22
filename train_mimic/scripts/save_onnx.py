#!/usr/bin/env python3
"""Export a tracking policy checkpoint to ONNX."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from train_mimic.app import validate_checkpoint_path

_DEFAULT_REF_WINDOW_LENGTH = 20


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
            "Checkpoint missing obs normalizer stats. "
            f"Tried mean keys {mean_keys} and var keys {var_keys}."
        )
    return mean, var


def _canonicalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = dict(state_dict)
    if "obs_normalizer.mean" in normalized and "obs_normalizer._mean" not in normalized:
        normalized["obs_normalizer._mean"] = normalized.pop("obs_normalizer.mean")
    if "obs_normalizer.var" in normalized and "obs_normalizer._var" not in normalized:
        normalized["obs_normalizer._var"] = normalized.pop("obs_normalizer.var")

    for key in list(normalized):
        if key.startswith("obs_normalizers_3d.") and key.endswith(".mean"):
            canonical = key[: -len(".mean")] + "._mean"
            normalized.setdefault(canonical, normalized.pop(key))
        elif key.startswith("obs_normalizers_3d.") and key.endswith(".var"):
            canonical = key[: -len(".var")] + "._var"
            normalized.setdefault(canonical, normalized.pop(key))
    return normalized


def _extract_mlp_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, tuple[int, ...], int]:
    obs_mean, _obs_var = _resolve_normalizer_stats(state_dict)
    obs_dim = int(obs_mean.shape[-1])
    mlp_weights: list[tuple[int, torch.Tensor]] = []
    for key, val in state_dict.items():
        if key.startswith("mlp.") and key.endswith(".weight"):
            idx = int(key.split(".")[1])
            mlp_weights.append((idx, val))
    mlp_weights.sort()
    if not mlp_weights:
        raise RuntimeError("Checkpoint does not contain any mlp.*.weight tensors.")
    hidden_dims = tuple(int(w.shape[0]) for _, w in mlp_weights[:-1])
    output_dim = int(mlp_weights[-1][1].shape[0])
    return obs_dim, hidden_dims, output_dim


def _extract_temporal_group_names(state_dict: dict[str, torch.Tensor]) -> list[str]:
    group_names: list[str] = []
    for key in state_dict:
        if not key.startswith("cnn_encoders."):
            continue
        group_name = key.split(".", 2)[1]
        if group_name not in group_names:
            group_names.append(group_name)
    return group_names


def _extract_temporal_group_dims(
    state_dict: dict[str, torch.Tensor], group_names: Sequence[str]
) -> dict[str, int]:
    dims: dict[str, int] = {}
    for group_name in group_names:
        mean_key = f"obs_normalizers_3d.{group_name}._mean"
        if mean_key in state_dict:
            dims[group_name] = int(state_dict[mean_key].shape[-1])
            continue

        conv_prefix = f"cnn_encoders.{group_name}.net."
        conv_weight = next(
            (
                val
                for key, val in state_dict.items()
                if key.startswith(conv_prefix) and key.endswith(".weight") and val.ndim == 3
            ),
            None,
        )
        if conv_weight is None:
            raise RuntimeError(f"Unable to infer feature dim for temporal group '{group_name}'.")
        dims[group_name] = int(conv_weight.shape[1])
    return dims


def _parse_temporal_group_length_overrides(items: Sequence[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(
                "Invalid --temporal_group_length value '" + item + "'. Expected GROUP=LENGTH."
            )
        group_name, length_str = item.split("=", 1)
        group_name = group_name.strip()
        if not group_name:
            raise ValueError(f"Invalid --temporal_group_length value '{item}'. Empty group name.")
        try:
            length = int(length_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid temporal length '{length_str}' for group '{group_name}'."
            ) from exc
        if length <= 0:
            raise ValueError(
                f"Temporal length for group '{group_name}' must be positive, got {length}."
            )
        overrides[group_name] = length
    return overrides


def _resolve_temporal_group_lengths(
    group_names: Sequence[str],
    history_length: int,
    overrides: dict[str, int] | None = None,
) -> dict[str, int]:
    overrides = {} if overrides is None else dict(overrides)
    resolved: dict[str, int] = {}
    for group_name in group_names:
        if group_name in overrides:
            resolved[group_name] = overrides.pop(group_name)
        elif group_name.endswith("_history"):
            resolved[group_name] = history_length
        elif group_name.endswith("_ref_window"):
            resolved[group_name] = _DEFAULT_REF_WINDOW_LENGTH
        else:
            raise RuntimeError(
                "Unable to infer temporal length for group '"
                + group_name
                + "'. Pass --temporal_group_length "
                + group_name
                + "=<length>."
            )
    if overrides:
        raise ValueError(
            "Unused --temporal_group_length override(s): " + ", ".join(sorted(overrides))
        )
    return resolved


def _export_temporal_cnn_policy(
    actor_state_dict: dict[str, torch.Tensor],
    output_path: str,
    history_length: int,
    activation: str,
    opset_version: int,
    temporal_group_lengths: dict[str, int] | None = None,
) -> None:
    from tensordict import TensorDict

    from train_mimic.tasks.tracking.rl.temporal_cnn_model import TemporalCNNModel

    sd = _canonicalize_state_dict(actor_state_dict)
    obs_dim, hidden_dims, output_dim = _extract_mlp_dims(sd)

    temporal_group_names = _extract_temporal_group_names(sd)
    if not temporal_group_names:
        raise RuntimeError("No temporal groups found in checkpoint despite cnn_encoders keys.")

    cnn_conv_weights: list[tuple[int, torch.Tensor]] = []
    conv_prefix = f"cnn_encoders.{temporal_group_names[0]}.net."
    for key, val in sd.items():
        if key.startswith(conv_prefix) and key.endswith(".weight") and val.ndim == 3:
            idx = int(key.split(".net.")[1].split(".")[0])
            cnn_conv_weights.append((idx, val))
    cnn_conv_weights.sort()
    if not cnn_conv_weights:
        raise RuntimeError(
            f"No Conv1d weights found for temporal group '{temporal_group_names[0]}'."
        )
    output_channels = tuple(int(w.shape[0]) for _, w in cnn_conv_weights)
    kernel_size = int(cnn_conv_weights[0][1].shape[2])

    temporal_group_dims = _extract_temporal_group_dims(sd, temporal_group_names)
    resolved_group_lengths = _resolve_temporal_group_lengths(
        temporal_group_names,
        history_length=history_length,
        overrides=temporal_group_lengths,
    )

    dist_cfg = None
    if any("distribution." in k for k in sd):
        dist_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }

    temporal_group_summary = ", ".join(
        f"{group}=(T={resolved_group_lengths[group]}, D={temporal_group_dims[group]})"
        for group in temporal_group_names
    )

    print(
        f"  Architecture: obs_dim={obs_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}"
    )
    print(
        f"  CNN: output_channels={output_channels}, kernel_size={kernel_size}, groups=[{temporal_group_summary}]"
    )

    dummy_obs_dict: dict[str, torch.Tensor] = {"actor": torch.zeros(1, obs_dim)}
    for group_name in temporal_group_names:
        dummy_obs_dict[group_name] = torch.zeros(
            1, resolved_group_lengths[group_name], temporal_group_dims[group_name]
        )
    dummy_obs = TensorDict(dummy_obs_dict, batch_size=[1])

    model = TemporalCNNModel(
        obs=dummy_obs,
        obs_groups={"actor": ("actor", *temporal_group_names)},
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

    onnx_model = model.as_onnx(verbose=False)
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


def _export_mlp_policy(
    actor_state_dict: dict[str, torch.Tensor],
    output_path: str,
    activation: str,
    opset_version: int,
) -> None:
    from tensordict import TensorDict
    from rsl_rl.models.mlp_model import MLPModel

    sd = _canonicalize_state_dict(actor_state_dict)
    obs_dim, hidden_dims, output_dim = _extract_mlp_dims(sd)
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

    dummy_obs = TensorDict({"actor": torch.zeros(1, obs_dim)}, batch_size=[1])
    model = MLPModel(
        obs=dummy_obs,
        obs_groups={"actor": ("actor",)},
        obs_set="actor",
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        obs_normalization=True,
        distribution_cfg=dist_cfg,
    )
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded MLPModel weights: {len(sd)} tensors")

    onnx_model = model.as_onnx(verbose=False)
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
    print(f"Exported MLP ONNX model to: {output_path}")
    print(f"  Inputs:  {list(zip(input_names, shapes))}")
    print(f"  Outputs: {output_names}")


def export_policy_as_onnx(
    checkpoint_path: str,
    output_path: str,
    activation: str = "elu",
    opset_version: int = 18,
    history_length: int = 10,
    temporal_group_lengths: dict[str, int] | None = None,
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
                "Only mjlab rsl_rl 5.x tracking checkpoints are supported."
            ) from e

    full_actor_sd = _extract_full_actor_state_dict(checkpoint)
    if _has_temporal_cnn_keys(full_actor_sd):
        print("Detected TemporalCNN tracking checkpoint.")
        _export_temporal_cnn_policy(
            actor_state_dict=full_actor_sd,
            output_path=output_path,
            history_length=history_length,
            activation=activation,
            opset_version=opset_version,
            temporal_group_lengths=temporal_group_lengths,
        )
        return

    print("Detected single-input MLP tracking checkpoint.")
    _export_mlp_policy(
        actor_state_dict=full_actor_sd,
        output_path=output_path,
        activation=activation,
        opset_version=opset_version,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export tracking policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--opset_version", type=int, default=18)
    parser.add_argument(
        "--history_length",
        type=int,
        default=10,
        help="History length for TemporalCNN *_history groups (default: 10)",
    )
    parser.add_argument(
        "--temporal_group_length",
        action="append",
        default=[],
        metavar="GROUP=LENGTH",
        help=(
            "Override the length of a named TemporalCNN temporal group during export. "
            "Can be passed multiple times, e.g. --temporal_group_length actor_ref_window=20"
        ),
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
        temporal_group_lengths=_parse_temporal_group_length_overrides(args.temporal_group_length),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
