"""1-D convolutional encoder for temporal observation sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}

_POOLS = {
    "avg": nn.AdaptiveAvgPool1d,
    "max": nn.AdaptiveMaxPool1d,
}


class Conv1dEncoder(nn.Module):
    """Encode a ``(B, input_channels, seq_len)`` sequence into ``(B, C)``."""

    def __init__(
        self,
        input_channels: int,
        output_channels: tuple[int, ...] = (64, 32),
        kernel_size: int = 3,
        activation: str = "elu",
        global_pool: str = "avg",
    ) -> None:
        super().__init__()

        act_cls = _ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in output_channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
            layers.append(act_cls())
            in_ch = out_ch
        pool_cls = _POOLS[global_pool]
        layers.append(pool_cls(1))
        layers.append(nn.Flatten(start_dim=1))
        self.net = nn.Sequential(*layers)
        self._output_dim = output_channels[-1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, input_channels, seq_len)``

        Returns:
            ``(B, output_dim)``
        """
        return self.net(x)
