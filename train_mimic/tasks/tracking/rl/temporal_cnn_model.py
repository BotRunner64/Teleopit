"""Temporal CNN model: 1-D CNN encoder for history observations + MLP head.

Follows the same extension pattern as ``rsl_rl.models.cnn_model.CNNModel``
but encodes 3-D ``(B, T, D)`` history groups instead of 4-D image groups.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState

from train_mimic.tasks.tracking.rl.conv1d_encoder import Conv1dEncoder


def _export_input_name(group_name: str) -> str:
    """Map observation-group names to stable export input names."""
    if group_name in {"actor", "critic"}:
        return "obs"
    for prefix in ("actor_", "critic_"):
        if group_name.startswith(prefix):
            return "obs_" + group_name[len(prefix):]
    return group_name


class TemporalCNNModel(MLPModel):
    """MLP model extended with 1-D CNN encoders for temporal observation history.

    3-D observation groups ``(B, T, D)`` are encoded by per-group ``Conv1dEncoder``
    instances.  1-D groups ``(B, D)`` are handled by the parent ``MLPModel``.

    Each 3-D group gets its own ``EmpiricalNormalization`` whose shape matches
    the group's feature dimension, so groups with different D values are handled
    correctly.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg: dict[str, Any] | None = None,
    ) -> None:
        # --- separate 1-D and 3-D groups before super().__init__ ---
        self._get_obs_dim(obs, obs_groups, obs_set)

        # Build Conv1d encoders for each 3-D group
        if cnn_cfg is None:
            cnn_cfg = {}
        self.cnn_encoders_dict: dict[str, Conv1dEncoder] = {}
        self.cnn_latent_dim = 0
        for group_name, obs_dim in zip(self.obs_groups_3d, self.obs_dims_3d):
            encoder = Conv1dEncoder(input_channels=obs_dim, **cnn_cfg)
            self.cnn_encoders_dict[group_name] = encoder
            self.cnn_latent_dim += encoder.output_dim

        # Stash flag before super().__init__ so we can build per-group normalizers
        self._obs_normalization_3d = obs_normalization

        # Now let parent build MLP (uses _get_latent_dim for MLP input size)
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # Register encoders as proper sub-modules
        self.cnn_encoders = nn.ModuleDict(self.cnn_encoders_dict)

        # Per-3D-group normalizers (each sized to its own feature dim)
        if obs_normalization:
            normalizers_3d: dict[str, nn.Module] = {}
            for group_name, dim_3d in zip(self.obs_groups_3d, self.obs_dims_3d):
                normalizers_3d[group_name] = EmpiricalNormalization(dim_3d)
            self.obs_normalizers_3d = nn.ModuleDict(normalizers_3d)
        else:
            self.obs_normalizers_3d = nn.ModuleDict(
                {g: nn.Identity() for g in self.obs_groups_3d}
            )

    # ------------------------------------------------------------------
    # Override: split observation groups into 1-D and 3-D
    # ------------------------------------------------------------------
    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        active = obs_groups[obs_set]
        obs_dim_1d = 0
        groups_1d: list[str] = []
        groups_3d: list[str] = []
        dims_3d: list[int] = []
        history_lengths: list[int] = []

        for g in active:
            ndim = len(obs[g].shape)
            if ndim == 2:  # (B, D)
                groups_1d.append(g)
                obs_dim_1d += obs[g].shape[-1]
            elif ndim == 3:  # (B, T, D)
                groups_3d.append(g)
                dims_3d.append(obs[g].shape[-1])
                history_lengths.append(obs[g].shape[1])
            else:
                raise ValueError(
                    f"TemporalCNNModel expects 1-D (B,D) or 3-D (B,T,D) obs, "
                    f"got shape {obs[g].shape} for '{g}'."
                )

        self.obs_groups_3d = groups_3d
        self.obs_dims_3d = dims_3d
        self.history_lengths = history_lengths
        self.obs_groups_1d = groups_1d
        return groups_1d, obs_dim_1d

    # ------------------------------------------------------------------
    # Override: MLP input = 1-D obs dim + CNN latent dim
    # ------------------------------------------------------------------
    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.cnn_latent_dim

    # ------------------------------------------------------------------
    # Override: build latent from 1-D + CNN-encoded 3-D groups
    # ------------------------------------------------------------------
    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        # 1-D path (concatenate + normalize via parent)
        latent_1d = super().get_latent(obs, masks, hidden_state)

        # 3-D path: per-group normalize → permute → Conv1d encode
        latent_parts = [latent_1d]
        for group_name in self.obs_groups_3d:
            h = obs[group_name]  # (B, T, D)
            h = self.obs_normalizers_3d[group_name](h)
            h = h.permute(0, 2, 1)  # (B, D, T) — channels-first for Conv1d
            latent_parts.append(self.cnn_encoders[group_name](h))

        return torch.cat(latent_parts, dim=-1)

    # ------------------------------------------------------------------
    # Override: update normalizers for 3-D groups too
    # ------------------------------------------------------------------
    def update_normalization(self, obs: TensorDict) -> None:
        super().update_normalization(obs)
        if self._obs_normalization_3d:
            for group_name in self.obs_groups_3d:
                h = obs[group_name]  # (B, T, D)
                # Flatten to (B*T, D) for running-stats update
                B, T, D = h.shape
                self.obs_normalizers_3d[group_name].update(h.reshape(B * T, D))  # type: ignore

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    def as_jit(self) -> nn.Module:
        return _TorchTemporalCNNModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxTemporalCNNModel(self, verbose)


# ======================================================================
# JIT-export wrapper
# ======================================================================
class _TorchTemporalCNNModel(nn.Module):
    def __init__(self, model: TemporalCNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnn_normalizers = nn.ModuleList(
            [copy.deepcopy(model.obs_normalizers_3d[g]) for g in model.obs_groups_3d]
        )
        self.cnn_encoders = nn.ModuleList(
            [copy.deepcopy(model.cnn_encoders[g]) for g in model.obs_groups_3d]
        )
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.temporal_input_names = [_export_input_name(g) for g in model.obs_groups_3d]

    def _encode_temporal_inputs(self, obs_temporal: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(obs_temporal) != len(self.cnn_encoders):
            raise ValueError(
                f"Expected {len(self.cnn_encoders)} temporal inputs "
                f"({self.temporal_input_names}), got {len(obs_temporal)}."
            )
        latent_parts: list[torch.Tensor] = []
        for obs_group, normalizer, encoder in zip(
            obs_temporal, self.cnn_normalizers, self.cnn_encoders, strict=True
        ):
            h = normalizer(obs_group)
            h = h.permute(0, 2, 1)
            latent_parts.append(encoder(h))
        return latent_parts

    def forward(self, obs_1d: torch.Tensor, *obs_temporal: torch.Tensor) -> torch.Tensor:
        latent_1d = self.obs_normalizer(obs_1d)
        latent = torch.cat(
            [latent_1d, *self._encode_temporal_inputs(obs_temporal)], dim=-1
        )
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        pass


# ======================================================================
# ONNX-export wrapper
# ======================================================================
class _OnnxTemporalCNNModel(nn.Module):
    def __init__(self, model: TemporalCNNModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.cnn_normalizers = nn.ModuleList(
            [copy.deepcopy(model.obs_normalizers_3d[g]) for g in model.obs_groups_3d]
        )
        self.cnn_encoders = nn.ModuleList(
            [copy.deepcopy(model.cnn_encoders[g]) for g in model.obs_groups_3d]
        )
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self._obs_dim_1d = model.obs_dim
        self._obs_dims_3d = model.obs_dims_3d
        self._history_lengths = model.history_lengths
        self._temporal_input_names = [_export_input_name(g) for g in model.obs_groups_3d]

    def _encode_temporal_inputs(self, obs_temporal: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(obs_temporal) != len(self.cnn_encoders):
            raise ValueError(
                f"Expected {len(self.cnn_encoders)} temporal inputs "
                f"({self._temporal_input_names}), got {len(obs_temporal)}."
            )
        latent_parts: list[torch.Tensor] = []
        for obs_group, normalizer, encoder in zip(
            obs_temporal, self.cnn_normalizers, self.cnn_encoders, strict=True
        ):
            h = normalizer(obs_group)
            h = h.permute(0, 2, 1)
            latent_parts.append(encoder(h))
        return latent_parts

    def forward(self, obs_1d: torch.Tensor, *obs_temporal: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export.

        Args:
            obs_1d: ``(1, D)`` current-frame observation.
            *obs_temporal: ``(1, T_i, D_i)`` temporal observation groups.
        """
        latent_1d = self.obs_normalizer(obs_1d)
        latent = torch.cat(
            [latent_1d, *self._encode_temporal_inputs(obs_temporal)], dim=-1
        )
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        dummy_1d = torch.zeros(1, self._obs_dim_1d)
        dummy_temporal = tuple(
            torch.zeros(1, hist_len, obs_dim)
            for hist_len, obs_dim in zip(self._history_lengths, self._obs_dims_3d, strict=True)
        )
        return (dummy_1d, *dummy_temporal)

    @property
    def input_names(self) -> list[str]:
        return ["obs", *self._temporal_input_names]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
