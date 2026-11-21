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

"""Image-conditioned LeWM dynamics with explicit action history."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from embodichain.utils import configclass

from ._architecture import ARPredictor, Embedder, MLP

__all__ = ["LeWMCfg", "LeWM", "preprocess_rgb"]


@configclass
class LeWMCfg:
    """Configure the reference-compatible visual encoder and predictor.

    Args:
        action_dim: Number of scalar controls in one model step, including all
            concatenated raw controls when ``action_block`` is greater than one.
        image_size: Square encoder input size.
        patch_size: ViT patch size.
        embed_dim: Latent and transformer width.
        history_size: Maximum number of observed frames used by the predictor,
            sampled ``action_block`` raw timesteps apart.
        vision_depth: Number of ViT blocks.
        vision_heads: Number of ViT attention heads.
        predictor_depth: Number of causal predictor blocks.
        predictor_heads: Number of predictor attention heads.
        predictor_head_dim: Width per predictor attention head.
        mlp_dim: Width of predictor and projector hidden layers.
        dropout: Predictor dropout probability.
        action_block: Number of consecutive raw controls concatenated into one
            model action. ``action_dim`` must be divisible by this number.
    """

    action_dim: int = 10
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 192
    history_size: int = 3
    vision_depth: int = 12
    vision_heads: int = 3
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_head_dim: int = 64
    mlp_dim: int = 2048
    dropout: float = 0.1
    action_block: int = 1


def preprocess_rgb(images: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Resize uint8 RGB/RGBA frames and apply the reference ImageNet transform.

    Args:
        images: Tensor with shape ``(..., H, W, 3|4)`` and uint8 dtype.
        image_size: Square output dimension.

    Returns:
        Float32 tensor with shape ``(..., 3, image_size, image_size)``.
    """
    if images.dtype != torch.uint8 or images.ndim < 4 or images.shape[-1] not in (3, 4):
        raise ValueError("images must be uint8 RGB or RGBA frames in HWC layout")
    if type(image_size) is not int or image_size < 1:
        raise ValueError("image_size must be a positive integer")
    shape = images.shape[:-3]
    rgb = images[..., :3].reshape(-1, *images.shape[-3:-1], 3)
    rgb = rgb.permute(0, 3, 1, 2).float().div_(255.0)
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    rgb = (rgb - mean) / std
    if rgb.shape[-2:] != (image_size, image_size):
        rgb = F.interpolate(
            rgb,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return rgb.reshape(*shape, 3, image_size, image_size)


class LeWM(nn.Module):
    """Predict action-conditioned latent observations, independent of simulation.

    Args:
        cfg: Encoder, action and temporal dimensions.

    Inputs to :meth:`encode` are normalized CHW images. Actions to prediction
    methods are in the recorded controller's raw units; frozen action statistics
    are applied internally. The two statistics are included in checkpoints.
    """

    def __init__(self, cfg: LeWMCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LeWMCfg()
        self.cfg.validate()
        c = self.cfg
        sizes = (
            c.action_dim,
            c.image_size,
            c.patch_size,
            c.embed_dim,
            c.history_size,
            c.vision_depth,
            c.vision_heads,
            c.predictor_depth,
            c.predictor_heads,
            c.predictor_head_dim,
            c.mlp_dim,
            c.action_block,
        )
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError("LeWM dimensions must be positive integers")
        if (
            c.image_size % c.patch_size
            or c.action_dim % c.action_block
            or c.embed_dim % c.vision_heads
            or not 0 <= c.dropout < 1
        ):
            raise ValueError("Invalid LeWM image, attention or dropout configuration")
        try:
            from transformers import ViTConfig, ViTModel
        except ImportError as exc:
            raise ImportError(
                "LeWM requires the 'lewm' extra: pip install embodichain[lewm]"
            ) from exc
        vit = ViTConfig(
            image_size=c.image_size,
            patch_size=c.patch_size,
            hidden_size=c.embed_dim,
            num_hidden_layers=c.vision_depth,
            num_attention_heads=c.vision_heads,
            intermediate_size=4 * c.embed_dim,
        )
        self.encoder = ViTModel(vit, add_pooling_layer=False, use_mask_token=False)
        self.predictor = ARPredictor(
            num_frames=c.history_size,
            depth=c.predictor_depth,
            heads=c.predictor_heads,
            mlp_dim=c.mlp_dim,
            input_dim=c.embed_dim,
            hidden_dim=c.embed_dim,
            output_dim=c.embed_dim,
            dim_head=c.predictor_head_dim,
            dropout=c.dropout,
        )
        self.action_encoder = Embedder(input_dim=c.action_dim, emb_dim=c.embed_dim)
        self.projector = MLP(
            c.embed_dim, c.mlp_dim, c.embed_dim, norm_fn=nn.BatchNorm1d
        )
        self.pred_proj = MLP(
            c.embed_dim, c.mlp_dim, c.embed_dim, norm_fn=nn.BatchNorm1d
        )
        self.register_buffer("action_mean", torch.zeros(c.action_dim))
        self.register_buffer("action_std", torch.ones(c.action_dim))

    def set_action_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Freeze train-only action moments used for training and planning.

        Args:
            mean: One mean per raw action component.
            std: Positive standard deviation per component.
        """
        if mean.shape != self.action_mean.shape or std.shape != self.action_std.shape:
            raise ValueError("action statistics have the wrong width")
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or not (std > 0).all()
        ):
            raise ValueError("action statistics must be finite with positive std")
        self.action_mean.copy_(mean.to(self.action_mean))
        self.action_std.copy_(std.to(self.action_std))

    def load_reference_weights(self, state: dict[str, torch.Tensor]) -> None:
        """Strictly load a reference weights-only state dictionary.

        Args:
            state: Original LeWM parameters and buffers without action moments.
        """
        if "action_mean" in state or "action_std" in state:
            raise ValueError(
                "reference weights must not contain local action statistics"
            )
        self.load_state_dict(
            {**state, "action_mean": self.action_mean, "action_std": self.action_std},
            strict=True,
        )

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a batch of normalized image histories into latent vectors.

        Args:
            pixels: Float images shaped ``(B, T, 3, H, W)``.

        Returns:
            Latents shaped ``(B, T, D)``.
        """
        if pixels.ndim != 5 or pixels.shape[2] != 3:
            raise ValueError("pixels must have shape (B, T, 3, H, W)")
        b, t = pixels.shape[:2]
        features = self.encoder(
            pixels.flatten(0, 1).to(next(self.encoder.parameters()).dtype),
            interpolate_pos_encoding=True,
        ).last_hidden_state[:, 0]
        return self.projector(features).reshape(b, t, -1)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode raw actions using the frozen train-only moments.

        Args:
            actions: Raw action tensor of shape ``(B, T, A)``.

        Returns:
            Encoded actions of shape ``(B, T, D)``.
        """
        if actions.ndim != 3 or actions.shape[-1] != self.cfg.action_dim:
            raise ValueError("actions must have shape (B, T, action_dim)")
        return self.action_encoder(
            (actions.float() - self.action_mean) / self.action_std
        )

    def predict(self, embeddings: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict the next latent for each aligned frame and outgoing action.

        Args:
            embeddings: Latents shaped ``(B, T, D)``.
            actions: Outgoing raw actions shaped ``(B, T, A)``.

        Returns:
            Predicted next latents shaped ``(B, T, D)``.
        """
        if (
            embeddings.ndim != 3
            or embeddings.shape[:2] != actions.shape[:2]
            or embeddings.shape[-1] != self.cfg.embed_dim
        ):
            raise ValueError("embedding and action histories are not aligned")
        if not 1 <= embeddings.shape[1] <= self.cfg.history_size:
            raise ValueError("history exceeds the configured predictor context")
        pred = self.predictor(embeddings, self.encode_actions(actions))
        return self.pred_proj(pred.flatten(0, 1)).reshape_as(pred)

    def rollout(
        self,
        history: torch.Tensor,
        past_actions: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Roll out strictly future candidate actions from observed history.

        Args:
            history: Observed latents ``(B, H, D)``.
            past_actions: Executed actions between those frames ``(B, H-1, A)``.
            candidates: Future actions ``(B, S, T, A)``.

        Returns:
            Predicted future latents ``(B, S, T, D)``.
        """
        if history.ndim != 3 or candidates.ndim != 4 or past_actions.ndim != 3:
            raise ValueError(
                "rollout requires history, past actions and candidate batches"
            )
        b, h, d = history.shape
        if not 1 <= h <= self.cfg.history_size or d != self.cfg.embed_dim:
            raise ValueError("invalid initial latent history")
        if (
            candidates.shape[0] != b
            or candidates.shape[2] < 1
            or candidates.shape[-1] != self.cfg.action_dim
        ):
            raise ValueError("invalid candidate actions")
        if past_actions.shape != (b, h - 1, self.cfg.action_dim):
            raise ValueError("past_actions must contain exactly H-1 actions")
        s, horizon = candidates.shape[1:3]
        if s < 1:
            raise ValueError("at least one candidate is required")
        latent = list(
            history[:, None].expand(b, s, h, d).reshape(b * s, h, d).unbind(1)
        )
        old = (
            past_actions[:, None]
            .expand(b, s, h - 1, self.cfg.action_dim)
            .reshape(b * s, h - 1, self.cfg.action_dim)
        )
        future = candidates.reshape(b * s, horizon, self.cfg.action_dim)
        actions = torch.cat((old, future), dim=1)
        for step in range(horizon):
            end = h + step
            start = max(0, end - self.cfg.history_size)
            pred = self.predict(
                torch.stack(latent[start:end], dim=1), actions[:, start:end]
            )[:, -1]
            latent.append(pred)
        return torch.stack(latent[h:], dim=1).reshape(b, s, horizon, d)

    @staticmethod
    def goal_cost(predictions: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Compute final latent squared distance for each candidate.

        Args:
            predictions: Predicted futures ``(B, S, T, D)``.
            goal: Goal embeddings ``(B, D)``.

        Returns:
            Final costs ``(B, S)``.
        """
        if predictions.ndim != 4 or goal.shape != (
            predictions.shape[0],
            predictions.shape[-1],
        ):
            raise ValueError("goal and prediction batches are incompatible")
        return (predictions[:, :, -1] - goal[:, None]).square().sum(-1)
