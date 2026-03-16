#!/usr/bin/env python3
"""Export trained tracking policy to ONNX format for hardware deployment.

Supports both standard MLP and TemporalCNNModel (history-CNN) checkpoints.
Loads rsl_rl checkpoint and exports actor network with empirical
normalization baked in.

Usage:
    python train_mimic/scripts/save_onnx.py \
        --checkpoint logs/rsl_rl/g1_tracking/.../model_30000.pt \
        --output policy.onnx

    # History-CNN checkpoint (auto-detected, or specify --history_length):
    python train_mimic/scripts/save_onnx.py \
        --checkpoint logs/rsl_rl/g1_tracking_history_cnn/.../model_30000.pt \
        --output policy.onnx --history_length 10
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from train_mimic.app import validate_checkpoint_path


class PolicyExportWrapper(nn.Module):
    """Wraps rsl_rl ActorCritic actor for ONNX export with normalization."""

    def __init__(self, actor: nn.Module, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.actor(obs)


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
    """Detect whether a state dict belongs to a TemporalCNNModel."""
    return any(k.startswith("cnn_encoders.") for k in state_dict)


def _resolve_normalizer_stats(state_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve observation-normalizer mean/variance tensors across naming variants."""
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
    """Normalize compatible key variants to the names expected by the model."""
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
    """Reconstruct a TemporalCNNModel from its state dict and export to ONNX."""
    from tensordict import TensorDict

    from train_mimic.tasks.tracking.rl.temporal_cnn_model import TemporalCNNModel

    sd = _canonicalize_temporal_state_dict(actor_state_dict)

    # --- Infer architecture from state dict ---
    # obs_dim from normalizer shape
    obs_mean, _obs_var = _resolve_normalizer_stats(sd)
    obs_dim = int(obs_mean.shape[-1])

    # MLP architecture from mlp.*.weight keys
    mlp_weights: list[tuple[int, torch.Tensor]] = []
    for key, val in sd.items():
        if key.startswith("mlp.") and key.endswith(".weight"):
            idx = int(key.split(".")[1])
            mlp_weights.append((idx, val))
    mlp_weights.sort()
    hidden_dims = tuple(int(w.shape[0]) for _, w in mlp_weights[:-1])
    output_dim = int(mlp_weights[-1][1].shape[0])

    # CNN architecture from cnn_encoders.*.net.*.weight keys
    cnn_conv_weights: list[tuple[int, torch.Tensor]] = []
    for key, val in sd.items():
        if "cnn_encoders." in key and ".net." in key and key.endswith(".weight") and val.ndim == 3:
            # Conv1d weight shape: (out_ch, in_ch, kernel_size)
            idx = int(key.split(".net.")[1].split(".")[0])
            cnn_conv_weights.append((idx, val))
    cnn_conv_weights.sort()
    if not cnn_conv_weights:
        raise RuntimeError("No Conv1d weights found in checkpoint despite cnn_encoders keys.")
    output_channels = tuple(int(w.shape[0]) for _, w in cnn_conv_weights)
    kernel_size = int(cnn_conv_weights[0][1].shape[2])

    # Check for distribution
    has_distribution = any("distribution." in k for k in sd)
    dist_cfg = None
    if has_distribution:
        dist_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        }

    print(f"  Architecture: obs_dim={obs_dim}, hidden_dims={hidden_dims}, "
          f"output_dim={output_dim}")
    print(f"  CNN: output_channels={output_channels}, kernel_size={kernel_size}, "
          f"history_length={history_length}")

    # Build dummy obs TensorDict for model construction
    B = 1
    dummy_obs = TensorDict(
        {
            "actor": torch.zeros(B, obs_dim),
            "actor_history": torch.zeros(B, history_length, obs_dim),
        },
        batch_size=[B],
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
    num_observations: int = 189,
    num_actions: int = 29,
    actor_hidden_dims: list[int] | None = None,
    activation: str = "elu",
    opset_version: int = 18,
    history_length: int = 10,
) -> None:
    """Export policy to ONNX.

    Args:
        checkpoint_path: Path to rsl_rl checkpoint (.pt)
        output_path: Output ONNX file path
        num_observations: Observation dimension (~189 for mjlab tracking)
        num_actions: Action dimension (29 for G1)
        actor_hidden_dims: MLP hidden dimensions
        activation: Activation function name
    """
    if actor_hidden_dims is None:
        actor_hidden_dims = [512, 256, 128]

    # Load checkpoint — try weights_only=True first (avoids pickle issues),
    # fall back to weights_only=False for older torch-saved objects.
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Cannot load checkpoint — missing module '{e.name}'.\n"
                "This is likely an old Isaac Lab checkpoint (train_mimic.rsl_rl). "
                "Only mjlab rsl_rl 5.x checkpoints from logs/rsl_rl/g1_tracking/ are supported."
            ) from e

    # --- Resolve actor state dict and obs normalizer params ---
    # Supported checkpoint layouts:
    #
    #   Layout A (rsl_rl 5.x OnPolicyRunner.save):
    #     checkpoint["model_state_dict"]["actor.model.0.weight"] ...
    #     checkpoint["model_state_dict"]["obs_normalizer.mean"] ...
    #
    #   Layout B (some rsl_rl 5.x variants save actor separately):
    #     checkpoint["actor_state_dict"]["model.0.weight"] ...
    #     checkpoint["obs_normalizer_state_dict"]["mean"] ...

    actor_state: dict[str, torch.Tensor] = {}
    obs_mean: torch.Tensor | None = None
    obs_var: torch.Tensor | None = None

    if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
        # Layout B — actor saved separately (common in current g1_tracking checkpoints).
        # Typical keys:
        #   mlp.0.weight, mlp.0.bias, ...
        #   distribution.std_param
        #   obs_normalizer._mean, obs_normalizer._var, ...
        raw = checkpoint["actor_state_dict"]
        for key, val in raw.items():
            # Actor MLP weights
            if key.startswith("mlp."):
                actor_state[key[len("mlp."):]] = val
            # Some variants may store model.* instead of mlp.*
            elif key.startswith("model."):
                actor_state[key[len("model."):]] = val

            # Empirical normalizer often lives inside actor_state_dict
            if key in ("obs_normalizer._mean", "obs_normalizer.mean"):
                obs_mean = val
            elif key in ("obs_normalizer._var", "obs_normalizer.var"):
                obs_var = val

        # Normalizer may be in a separate entry
        for norm_key in ("obs_normalizer_state_dict", "obs_rms_state_dict"):
            if norm_key in checkpoint and isinstance(checkpoint[norm_key], dict):
                sd = checkpoint[norm_key]
                if "mean" in sd and "var" in sd:
                    obs_mean, obs_var = sd["mean"], sd["var"]
                    break
        print(f"Detected layout B (actor_state_dict). Keys sample: {list(raw.keys())[:4]}")
    else:
        # Layout A — full model state dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # bare state dict

        for key, val in state_dict.items():
            if key.startswith("actor.model."):
                actor_state[key[len("actor.model."):]] = val
            elif key.startswith("actor."):
                actor_state[key[len("actor."):]] = val

        # Normalizer stored inline (obs_normalizer.* or obs_rms.*)
        for prefix in ("obs_normalizer.", "obs_rms."):
            if f"{prefix}mean" in state_dict and f"{prefix}var" in state_dict:
                obs_mean = state_dict[f"{prefix}mean"]
                obs_var = state_dict[f"{prefix}var"]
                break
        print(f"Detected layout A (model_state_dict). Keys sample: {list(state_dict.keys())[:4]}")

    # --- Detect TemporalCNNModel and branch to dedicated export ---
    full_actor_sd = _extract_full_actor_state_dict(checkpoint)
    if _has_temporal_cnn_keys(full_actor_sd):
        print("Detected TemporalCNNModel checkpoint — using temporal CNN export path.")
        _export_temporal_cnn_policy(
            actor_state_dict=full_actor_sd,
            output_path=output_path,
            history_length=history_length,
            activation=activation,
            opset_version=opset_version,
        )
        return

    if not actor_state:
        top_keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else ["<not a dict>"]
        raise RuntimeError(
            f"No actor weights found in checkpoint.\n"
            f"  Top-level keys: {top_keys}\n"
            "Expected 'actor_state_dict' or 'model_state_dict' with 'actor.*' sub-keys."
        )

    # Infer actor dimensions from checkpoint to avoid CLI/config mismatch.
    # Actor state keys are expected like: "0.weight", "0.bias", "2.weight", ...
    linear_weight_items: list[tuple[int, torch.Tensor]] = []
    for key, val in actor_state.items():
        if key.endswith(".weight"):
            layer_name = key.split(".", maxsplit=1)[0]
            if layer_name.isdigit():
                linear_weight_items.append((int(layer_name), val))
    linear_weight_items.sort(key=lambda x: x[0])
    if not linear_weight_items:
        raise RuntimeError("No linear layer weights found in actor state dict.")

    ckpt_num_observations = int(linear_weight_items[0][1].shape[1])
    ckpt_hidden_dims = [int(w.shape[0]) for _, w in linear_weight_items[:-1]]
    ckpt_num_actions = int(linear_weight_items[-1][1].shape[0])

    if num_observations != ckpt_num_observations:
        print(f"Override num_observations from {num_observations} -> {ckpt_num_observations} (from checkpoint)")
        num_observations = ckpt_num_observations
    if actor_hidden_dims != ckpt_hidden_dims:
        print(f"Override actor_hidden_dims from {actor_hidden_dims} -> {ckpt_hidden_dims} (from checkpoint)")
        actor_hidden_dims = ckpt_hidden_dims
    if num_actions != ckpt_num_actions:
        print(f"Override num_actions from {num_actions} -> {ckpt_num_actions} (from checkpoint)")
        num_actions = ckpt_num_actions

    # Build actor MLP matching checkpoint structure.
    activation_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    in_dim = num_observations
    for hidden_dim in actor_hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_fn())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, num_actions))
    actor = nn.Sequential(*layers)

    actor.load_state_dict(actor_state, strict=True)
    print(f"Loaded actor weights: {len(actor_state)} tensors")
    actor.eval()

    # Build empirical normalizer if params were found
    normalizer = None
    if obs_mean is not None and obs_var is not None:
        class EmpiricalNormalizer(nn.Module):
            def __init__(self, m: torch.Tensor, v: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("mean", m)
                self.register_buffer("std", torch.sqrt(v + 1e-8))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x - self.mean) / self.std

        normalizer = EmpiricalNormalizer(obs_mean, obs_var)
        print("Embedded empirical normalization into ONNX model.")
    else:
        print("No obs normalizer found in checkpoint — exporting without normalization.")

    # Wrap and export
    export_model = PolicyExportWrapper(actor, normalizer)
    export_model.eval()

    dummy_input = torch.randn(1, num_observations)
    batch_size_dim = torch.export.Dim("batch_size")
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=True,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_shapes={
            "obs": {0: batch_size_dim},
        },
    )

    print(f"Exported ONNX model to: {output_path}")
    print(f"  Input:  observations ({num_observations}D)")
    print(f"  Output: actions ({num_actions}D)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export tracking policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--num_observations", type=int, default=189)
    parser.add_argument("--num_actions", type=int, default=29)
    parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--activation", type=str, default="elu")
    parser.add_argument("--opset_version", type=int, default=18)
    parser.add_argument("--history_length", type=int, default=10,
                        help="History length for TemporalCNNModel checkpoints (default: 10)")
    args = parser.parse_args()

    try:
        validate_checkpoint_path(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    export_policy_as_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_observations=args.num_observations,
        num_actions=args.num_actions,
        actor_hidden_dims=args.actor_hidden_dims,
        activation=args.activation,
        opset_version=args.opset_version,
        history_length=args.history_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
