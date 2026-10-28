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

"""Visual ACT: a conditional VAE and a Transformer action-chunk decoder.

The architecture follows Zhao et al., https://arxiv.org/abs/2304.13705.
This implementation uses PyTorch and torchvision rather than vendoring ACT.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from embodichain.learning.sl import LossResult, SupervisedBatch
from embodichain.utils import configclass

__all__ = ["ACTCfg", "ACTPolicy", "ACTObjective"]


@configclass
class ACTCfg:
    """Dimensions and architecture for a visual ACT policy.

    Args:
        state_dim: Width of measured robot state.
        action_dim: Width of each action command.
        horizon: Number of actions predicted per observation.
        hidden_dim: Transformer embedding width, divisible by four and heads.
        heads: Number of attention heads.
        encoder_layers: Layers in each observation and posterior encoder.
        decoder_layers: Action decoder layers.
        feedforward_dim: Transformer feedforward width.
        latent_dim: CVAE latent width.
        dropout: Transformer dropout rate.
        pretrained_backbone: Initialize ResNet-18 from torchvision ImageNet weights.
    """

    state_dim: int = 9
    action_dim: int = 8
    horizon: int = 30
    hidden_dim: int = 256
    heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 4
    feedforward_dim: int = 512
    latent_dim: int = 32
    dropout: float = 0.1
    pretrained_backbone: bool = True


def _position_1d(length: int, width: int, *, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    scale = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / width)
    )
    values = position * scale
    return torch.stack((values.sin(), values.cos()), dim=-1).flatten(-2)


def _position_2d(
    height: int, width: int, dim: int, *, device: torch.device
) -> torch.Tensor:
    y = _position_1d(height, dim // 2, device=device)[:, None].expand(-1, width, -1)
    x = _position_1d(width, dim // 2, device=device)[None].expand(height, -1, -1)
    return torch.cat((y, x), dim=-1).reshape(height * width, dim)


class ACTPolicy(nn.Module):
    """Predict normalized action chunks from RGB and proprioception.

    The posterior consumes demonstrated actions only during training. When no
    actions are provided, inference uses the prior mean, ``z=0``. State and
    action statistics are model buffers and must be set from training data.

    Args:
        cfg: ACT architecture and dimensions.
    """

    def __init__(self, cfg: ACTCfg | None = None) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.cfg = cfg or ACTCfg()
        self.cfg.validate()
        c = self.cfg
        positive = (
            c.state_dim,
            c.action_dim,
            c.horizon,
            c.hidden_dim,
            c.heads,
            c.encoder_layers,
            c.decoder_layers,
            c.feedforward_dim,
            c.latent_dim,
        )
        if any(type(n) is not int or n < 1 for n in positive):
            raise ValueError("ACT dimensions must be positive integers")
        if c.hidden_dim % 4 or c.hidden_dim % c.heads or not 0 <= c.dropout < 1:
            raise ValueError("Invalid ACT attention width, head count or dropout")
        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if c.pretrained_backbone else None
        )
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.image_projection = nn.Conv2d(512, c.hidden_dim, 1)
        self.state_projection = nn.Linear(c.state_dim, c.hidden_dim)
        self.action_projection = nn.Linear(c.action_dim, c.hidden_dim)
        self.latent_projection = nn.Linear(c.latent_dim, c.hidden_dim)
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, c.hidden_dim))
        self.posterior_state = nn.Linear(c.state_dim, c.hidden_dim)
        self.posterior_stats = nn.Linear(c.hidden_dim, 2 * c.latent_dim)
        self.encoder_tokens = nn.Parameter(torch.zeros(2, c.hidden_dim))
        self.queries = nn.Parameter(torch.randn(c.horizon, c.hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            c.hidden_dim, c.heads, c.feedforward_dim, c.dropout, batch_first=True
        )
        posterior_layer = nn.TransformerEncoderLayer(
            c.hidden_dim, c.heads, c.feedforward_dim, c.dropout, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            c.hidden_dim, c.heads, c.feedforward_dim, c.dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, c.encoder_layers, enable_nested_tensor=False
        )
        self.posterior = nn.TransformerEncoder(
            posterior_layer, c.encoder_layers, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, c.decoder_layers)
        self.action_head = nn.Linear(c.hidden_dim, c.action_dim)
        self.register_buffer("state_mean", torch.zeros(c.state_dim))
        self.register_buffer("state_std", torch.ones(c.state_dim))
        self.register_buffer("action_mean", torch.zeros(c.action_dim))
        self.register_buffer("action_std", torch.ones(c.action_dim))
        self.register_buffer(
            "rgb_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "rgb_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def set_normalization(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
    ) -> None:
        """Freeze training-set statistics used by both loss and rollout.

        Args:
            state_mean: Mean of each measured state feature.
            state_std: Positive scale for each state feature.
            action_mean: Mean of each action feature.
            action_std: Positive scale for each action feature.
        """
        for name, value in (
            ("state_mean", state_mean),
            ("state_std", state_std),
            ("action_mean", action_mean),
            ("action_std", action_std),
        ):
            buffer = getattr(self, name)
            if value.shape != buffer.shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid ACT normalization {name}")
            if name.endswith("std") and not bool((value > 0).all()):
                raise ValueError(f"ACT normalization {name} must be positive")
            buffer.copy_(value.to(buffer))

    def forward(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        *,
        actions: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return normalized predictions and optional posterior parameters.

        Args:
            rgb: RGB ``[B, 3, H, W]``, uint8 or floating point in [0, 1].
            state: Measured robot state ``[B, state_dim]``.
            actions: Physical demonstration commands ``[B, horizon, action_dim]``.
            valid: Boolean mask for real action frames, required with actions.

        Returns:
            Predicted normalized commands and posterior mean/log variance.
        """
        c = self.cfg
        if (
            rgb.ndim != 4
            or rgb.shape[1] != 3
            or state.shape != (rgb.shape[0], c.state_dim)
        ):
            raise ValueError("ACT expects RGB [B,3,H,W] and state [B,state_dim]")
        batch = state.shape[0]
        state = (state.float() - self.state_mean) / self.state_std
        image = rgb.float().div(255) if rgb.dtype == torch.uint8 else rgb.float()
        features = self.image_projection(
            self.backbone((image - self.rgb_mean) / self.rgb_std)
        )
        height, width = features.shape[-2:]
        visual = features.flatten(2).transpose(1, 2)
        visual = visual + _position_2d(
            height, width, c.hidden_dim, device=rgb.device
        ).to(visual)
        mu = logvar = None
        if actions is not None:
            if (
                actions.shape != (batch, c.horizon, c.action_dim)
                or valid is None
                or valid.shape != (batch, c.horizon)
                or valid.dtype != torch.bool
                or not bool(valid.any(dim=1).all())
            ):
                raise ValueError("ACT training requires a nonempty boolean action mask")
            normalized = (actions - self.action_mean) / self.action_std
            tokens = torch.cat(
                (
                    self.posterior_cls.expand(batch, -1, -1),
                    self.posterior_state(state)[:, None],
                    self.action_projection(normalized),
                ),
                dim=1,
            )
            tokens = tokens + _position_1d(
                c.horizon + 2, c.hidden_dim, device=rgb.device
            ).to(tokens)
            padding = torch.cat(
                (torch.zeros(batch, 2, dtype=torch.bool, device=valid.device), ~valid),
                dim=1,
            )
            posterior = self.posterior(tokens, src_key_padding_mask=padding)[:, 0]
            mu, logvar = self.posterior_stats(posterior).chunk(2, dim=-1)
            latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            if valid is not None:
                raise ValueError("ACT action mask requires actions")
            latent = state.new_zeros(batch, c.latent_dim)
        extra = torch.stack(
            (self.latent_projection(latent), self.state_projection(state)), dim=1
        )
        memory = self.encoder(torch.cat((extra + self.encoder_tokens, visual), dim=1))
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        return self.action_head(self.decoder(queries, memory)), mu, logvar

    def predict(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Predict physical action commands using the deterministic prior.

        Args:
            rgb: Current RGB batch.
            state: Current robot state batch.

        Returns:
            Denormalized action chunk of shape ``[B, horizon, action_dim]``.
        """
        predicted, _, _ = self(rgb, state)
        return predicted * self.action_std + self.action_mean


class ACTObjective:
    """Masked action L1 and standard-normal CVAE KL loss.

    Args:
        kl_weight: Multiplier for the per-sample latent KL term.
    """

    def __init__(self, kl_weight: float = 10.0) -> None:
        if not math.isfinite(kl_weight) or kl_weight < 0:
            raise ValueError("kl_weight must be finite and nonnegative")
        self.kl_weight = kl_weight

    def compute_loss(self, model: nn.Module, batch: SupervisedBatch) -> LossResult:
        """Compute the loss from causal inputs and masked action targets.

        Args:
            model: ACT policy.
            batch: Inputs ``rgb``/``state``, action targets and ``valid`` mask.

        Returns:
            Differentiable supervised loss and detached scalar metrics.
        """
        valid = batch.masks["valid"]
        actions = batch.targets
        predicted, mu, logvar = model(
            batch.inputs["rgb"], batch.inputs["state"], actions=actions, valid=valid
        )
        assert mu is not None and logvar is not None
        target = (actions - model.action_mean) / model.action_std
        # Equal per-example weighting also preserves SL's distributed contract.
        frame_error = (predicted - target).abs().mean(-1)
        l1 = ((frame_error * valid).sum(1) / valid.sum(1)).mean()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(-1)).mean()
        return LossResult(
            l1 + self.kl_weight * kl,
            {"action_l1": l1.detach(), "kl": kl.detach()},
            weight=float(len(actions)),
        )
