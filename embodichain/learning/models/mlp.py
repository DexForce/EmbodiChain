# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Trainable MLP shared by supervised learning and RL policy builders."""

from __future__ import annotations

from functools import reduce
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

__all__ = ["MLP", "build_mlp_from_cfg"]

ActivationName = str | None


def _resolve_activation(name: ActivationName) -> nn.Module:
    if name is None:
        return nn.Identity()
    name_l = str(name).lower()
    if name_l in ("relu",):
        return nn.ReLU()
    if name_l in ("elu",):
        return nn.ELU()
    if name_l in ("tanh",):
        return nn.Tanh()
    if name_l in ("gelu",):
        return nn.GELU()
    if name_l in ("silu", "swish"):
        return nn.SiLU()
    # fallback
    return nn.ReLU()


class MLP(nn.Sequential):
    """General MLP supporting custom last activation, orthogonal init, and output reshape.

    Args:
        input_dim: input dimension
        output_dim: output dimension (int or shape tuple/list)
        hidden_dims: hidden layer sizes, e.g. [256, 256]
        activation: hidden layer activation name (relu/elu/tanh/gelu/silu)
        last_activation: last-layer activation name or None for linear
        use_layernorm: whether to add LayerNorm after each hidden linear layer
        dropout_p: dropout probability for hidden layers (0 disables)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | Sequence[int],
        hidden_dims: Sequence[int],
        activation: ActivationName = "elu",
        last_activation: ActivationName = None,
        use_layernorm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()

        act = lambda: _resolve_activation(activation)
        last_act = (
            _resolve_activation(last_activation)
            if last_activation is not None
            else None
        )

        layers: list[nn.Module] = []
        dims = [input_dim] + list(hidden_dims)

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            if use_layernorm:
                layers.append(nn.LayerNorm(out_d))
            layers.append(act())
            if dropout_p and dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))

        # Output layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(dims[-1], output_dim))
        else:
            total_out = int(reduce(lambda a, b: a * b, output_dim))
            layers.append(nn.Linear(dims[-1], total_out))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=tuple(output_dim)))

        if last_act is not None:
            layers.append(last_act)

        for idx, layer in enumerate(layers):
            self.add_module(str(idx), layer)

    def init_orthogonal(self, scales: float | Sequence[float] = 1.0) -> None:
        """Orthogonal-initialize linear layers and zero the bias.

        scales: single gain value or a sequence with length equal to the
        number of linear layers.
        """

        def get_scale(i: int) -> float:
            if isinstance(scales, (list, tuple)):
                return float(scales[i])
            return float(scales)

        lin_idx = 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=get_scale(lin_idx))
                nn.init.zeros_(m.bias)
                lin_idx += 1


def build_mlp_from_cfg(module_cfg: Mapping[str, Any], in_dim: int, out_dim: int) -> MLP:
    """Construct an MLP from a model configuration.

    Args:
        module_cfg: Mapping with ``type="mlp"`` and ``network_cfg`` containing
            ``hidden_sizes`` plus optional activation, normalization, dropout
            and orthogonal-initialization settings.
        in_dim: Number of input features.
        out_dim: Number of output features.

    Returns:
        Configured MLP module.
    """
    if module_cfg.get("type", "").lower() != "mlp":
        raise ValueError("Only 'mlp' type is supported by this network builder.")

    network_cfg = module_cfg["network_cfg"]
    model = MLP(
        in_dim,
        out_dim,
        network_cfg["hidden_sizes"],
        network_cfg.get("activation", "elu"),
        last_activation=network_cfg.get("last_activation"),
        use_layernorm=bool(network_cfg.get("use_layernorm", False)),
        dropout_p=float(network_cfg.get("dropout_p", 0.0)),
    )
    if "orthogonal_init" in network_cfg:
        model.init_orthogonal(network_cfg["orthogonal_init"])
    return model
