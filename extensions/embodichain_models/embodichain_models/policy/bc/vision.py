# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Copyright 2025 The RLinf Authors.
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

"""Trainable visual features adapted from RLInf's Apache-2.0 ResNet encoder.

Reference: RLinf/RLinf, commit ``91a302ec50ddacd1fdabc5859b85e3661d768236``,
``rlinf/models/embodiment/modules/resnet_utils.py``. Modifications remove the
classifier, pretrained checkpoint requirement, frozen/detached backbone and
dropout. This module owns ImageNet normalization; callers supply RGB floats in
``[0, 1]``. Spatial pooling contracts directly over pixels, avoiding the
reference implementation's expanded ``[N, C, H, W, 8]`` intermediate.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

__all__ = ["ResNet10Encoder"]


class _SpatialResNet10(ResNet):
    def __init__(self) -> None:
        super().__init__(
            BasicBlock,
            [1, 1, 1, 1],
            norm_layer=partial(nn.GroupNorm, 4, eps=1e-5),
        )
        del self.avgpool
        del self.fc

    def _forward_impl(self, images: torch.Tensor) -> torch.Tensor:
        features = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return self.layer4(features)


class _SpatialLearnedEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kernel = nn.Parameter(torch.randn(512, 8, 8, 8))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Same channel-major, embedding-minor order as the pinned reference.
        return torch.einsum("bchw,chwf->bcf", features, self.kernel).flatten(1)


class ResNet10Encoder(nn.Module):
    """Encode 256-pixel RGB images into trainable 256-dimensional features.

    The encoder has four single-block residual stages with GroupNorm, eight
    learned spatial embeddings, and a Linear/LayerNorm/Tanh projection. It
    initializes all parameters locally and never downloads weights. Every
    stage remains differentiable in both training and evaluation modes.

    Input is float32 ``[N, 3, 256, 256]`` RGB in ``[0, 1]`` on the model's
    device. Value range and finiteness are the data adapter's responsibility;
    forward checks tensor metadata without synchronizing CUDA for pixel scans.
    Noncontiguous inputs are accepted and are never modified in place.
    """

    output_dim = 256
    image_size = 256

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.resnet_backbone = _SpatialResNet10()
        self.pooling_layer = _SpatialLearnedEmbeddings()
        self.mlp = nn.Sequential(
            nn.Linear(512 * 8, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.Tanh(),
        )
        # Pinned RLInf init_mlp_weights(..., nonlinearity="tanh").
        nn.init.kaiming_normal_(self.mlp[0].weight, mode="fan_out", nonlinearity="tanh")
        nn.init.zeros_(self.mlp[0].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``[N, 256]`` features from causal RGB float inputs.

        Args:
            images: Nonempty float32 NCHW images, already scaled to ``[0, 1]``.

        Returns:
            Independently owned feature tensor connected to inputs and weights.

        Raises:
            ValueError: Image shape, dtype or device does not match the contract.
        """
        if (
            images.ndim != 4
            or images.shape[0] < 1
            or images.shape[1:] != (3, self.image_size, self.image_size)
            or images.dtype != torch.float32
            or images.device != self.image_mean.device
        ):
            raise ValueError(
                "ResNet10Encoder requires nonempty float32 [N, 3, 256, 256] "
                "RGB images on the model device"
            )
        normalized = (images - self.image_mean) / self.image_std
        features = self.resnet_backbone(normalized)
        return self.mlp(self.pooling_layer(features))
