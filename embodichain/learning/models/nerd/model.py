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

"""NeRD causal Transformer compatible with the official network parameter names.

Architecture adapted from NVlabs/neural-robot-dynamics (Apache-2.0), whose
Transformer derives from Andrej Karpathy's nanoGPT (MIT). See THIRD_PARTY_LICENSES.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations
import math
from collections.abc import Mapping
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from embodichain.utils import configclass

__all__ = ["NeRDModelCfg", "NeRDModel", "FeatureNormalizer"]


@configclass
class NeRDModelCfg:
    """Network architecture and ordered input schema for NeRD.

    The default architecture matches the official Cartpole checkpoint; input_dims
    and output_dim must be supplied. Ant uses n_embd=384 and its contact schema.
    """

    input_dims: dict[str, int] = {}
    output_dim: int = 4
    n_embd: int = 192
    n_layer: int = 6
    n_head: int = 12
    block_size: int = 32
    history_length: int = 10
    head_hidden_dims: tuple[int, ...] = (64,)
    bias: bool = False
    dropout: float = 0.0
    normalize_input: bool = True
    normalize_output: bool = True
    output_tanh: bool = False


class FeatureNormalizer(nn.Module):
    """Frozen mean/variance buffers with the exact official sqrt(var + 1e-5)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("var", torch.ones(width))

    def forward(self, value: Tensor, inverse: bool = False) -> Tensor:
        """Normalize or denormalize without modifying the caller's tensor."""
        scale = torch.sqrt(self.var + 1e-5)
        return value * scale + self.mean if inverse else (value - self.mean) / scale


class _LayerNorm(nn.Module):
    def __init__(self, width: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width)) if bias else None

    def forward(self, value: Tensor) -> Tensor:
        return F.layer_norm(value, self.weight.shape, self.weight, self.bias, 1e-5)


class _Attention(nn.Module):
    def __init__(self, cfg: NeRDModelCfg) -> None:
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.n_head, self.n_embd, self.dropout = cfg.n_head, cfg.n_embd, cfg.dropout

    def forward(self, value: Tensor) -> Tensor:
        batch, length, width = value.shape
        q, k, v = self.c_attn(value).split(self.n_embd, -1)
        q, k, v = [
            x.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for x in (q, k, v)
        ]
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_dropout(self.c_proj(out))


class _MLP(nn.Module):
    def __init__(self, cfg: NeRDModelCfg) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, value: Tensor) -> Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(value))))


class _Block(nn.Module):
    def __init__(self, cfg: NeRDModelCfg) -> None:
        super().__init__()
        self.ln_1 = _LayerNorm(cfg.n_embd, cfg.bias)
        self.attn = _Attention(cfg)
        self.ln_2 = _LayerNorm(cfg.n_embd, cfg.bias)
        self.mlp = _MLP(cfg)

    def forward(self, value: Tensor) -> Tensor:
        value = value + self.attn(self.ln_1(value))
        return value + self.mlp(self.ln_2(value))


class _GPT(nn.Module):
    def __init__(self, cfg: NeRDModelCfg) -> None:
        super().__init__()
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Linear(sum(cfg.input_dims.values()), cfg.n_embd),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layer)]),
                ln_f=_LayerNorm(cfg.n_embd, cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer)
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, value: Tensor) -> Tensor:
        pos = torch.arange(value.shape[1], device=value.device)
        value = self.transformer.drop(
            self.transformer.wte(value) + self.transformer.wpe(pos)
        )
        for block in self.transformer.h:
            value = block(value)
        return self.lm_head(self.transformer.ln_f(value))


class _Head(nn.Module):
    def __init__(self, cfg: NeRDModelCfg) -> None:
        super().__init__()
        modules = []
        width = cfg.n_embd
        for next_width in cfg.head_hidden_dims:
            modules.extend([nn.Linear(width, next_width), nn.ReLU()])
            width = next_width
        self.feature_net = nn.Module()
        self.feature_net.body = nn.Sequential(*modules)
        self.output_net = nn.Linear(width, cfg.output_dim)

    def forward(self, value: Tensor) -> Tensor:
        return self.output_net(self.feature_net.body(value))


class NeRDModel(nn.Module):
    """Causal NeRD network with immutable named inputs and portable statistics.

    Args:
        cfg: Explicit input layout and Transformer configuration.

    All inputs have shape (batch, time, width). Right-padding is permitted:
    causal attention preserves prefix outputs, and the caller masks padded loss.
    ``evaluate`` gathers the final valid prefix token independently per row.
    Neither training nor evaluation implicitly detaches gradients.
    """

    def __init__(self, cfg: NeRDModelCfg) -> None:
        super().__init__()
        if not cfg.input_dims or any(v <= 0 for v in cfg.input_dims.values()):
            raise ValueError("input_dims must specify positive ordered feature widths")
        if any("." in k or not k for k in cfg.input_dims):
            raise ValueError("Feature names must be nonempty and cannot contain dots")
        if (
            cfg.n_head <= 0
            or cfg.n_embd <= 0
            or cfg.n_embd % cfg.n_head
            or cfg.n_layer <= 0
        ):
            raise ValueError(
                "Positive embedding must be divisible by positive attention heads"
            )
        if (
            not 1 <= cfg.history_length <= cfg.block_size
            or cfg.output_dim <= 0
            or not 0 <= cfg.dropout < 1
        ):
            raise ValueError("Invalid history, output dimension or dropout")
        self.cfg = cfg.copy()
        self.encoders = nn.ModuleDict({"low_dim": nn.Module()})
        self.encoders["low_dim"].body = nn.Sequential()
        self.transformer_model = _GPT(cfg)
        self.model = _Head(cfg)
        self.input_rms = nn.ModuleDict(
            {name: FeatureNormalizer(width) for name, width in cfg.input_dims.items()}
        )
        self.output_rms = FeatureNormalizer(cfg.output_dim)

    def set_normalization(
        self,
        input_stats: Mapping[str, tuple[Tensor, Tensor]],
        output_mean: Tensor,
        output_var: Tensor,
    ) -> None:
        """Set frozen statistics; missing features and invalid variances fail early."""
        if set(input_stats) != set(self.cfg.input_dims):
            raise ValueError("Normalization keys must exactly match the feature schema")
        entries = [(self.input_rms[k], *input_stats[k]) for k in input_stats]
        entries.append((self.output_rms, output_mean, output_var))
        for rms, mean, var in entries:
            if (
                mean.shape != rms.mean.shape
                or var.shape != rms.var.shape
                or not torch.isfinite(mean).all()
                or not torch.isfinite(var).all()
                or (var < 0).any()
            ):
                raise ValueError("Invalid normalizer mean/variance")
        with torch.no_grad():
            for rms, mean, var in entries:
                rms.mean.copy_(mean)
                rms.var.copy_(var)

    def forward(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Predict every sequence token in physical target units."""
        values = []
        shape = None
        for name, width in self.cfg.input_dims.items():
            value = inputs[name]
            if value.ndim != 3 or value.shape[-1] != width:
                raise ValueError(f"{name} must have shape (batch, time, {width})")
            if shape is not None and value.shape[:2] != shape:
                raise ValueError("All model features must share batch/time axes")
            shape = value.shape[:2]
            values.append(
                self.input_rms[name](value) if self.cfg.normalize_input else value
            )
        if not 1 <= shape[1] <= self.cfg.block_size:
            raise ValueError("Sequence length is outside the model block_size")
        value = self.model(self.transformer_model(torch.cat(values, -1)))
        if self.cfg.output_tanh:
            value = torch.tanh(value)
        return (
            self.output_rms(value, inverse=True) if self.cfg.normalize_output else value
        )

    def evaluate(
        self, inputs: Mapping[str, Tensor], valid_lengths: Tensor | None = None
    ) -> Tensor:
        """Predict the last valid token, shape (batch, 1, output_dim)."""
        value = self(inputs)
        if valid_lengths is None:
            return value[:, -1:]
        if (
            valid_lengths.shape != (value.shape[0],)
            or valid_lengths.dtype != torch.long
        ):
            raise ValueError(
                "valid_lengths must be long values with one length per row"
            )
        valid = ((valid_lengths >= 1) & (valid_lengths <= value.shape[1])).all()
        if valid_lengths.device.type == "cpu":
            if not valid:
                raise ValueError("valid_lengths must be in [1, sequence_length]")
        else:
            # Keep per-step history validation on device; do not synchronize CUDA.
            torch._assert_async(valid, "valid_lengths must be in [1, sequence_length]")
        index = (valid_lengths - 1)[:, None, None].expand(-1, 1, value.shape[-1])
        return value.gather(1, index)
