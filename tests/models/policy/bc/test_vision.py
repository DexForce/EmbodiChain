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

"""CPU contracts for the trainable RGB encoder shared by learning clients."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn
from torchvision.models.resnet import BasicBlock, ResNet

from embodichain_models.policy.bc import ResNet10Encoder


@pytest.fixture(scope="module", autouse=True)
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def test_reference_structure_and_all_stages_receive_gradients():
    torch.manual_seed(12)
    encoder = ResNet10Encoder()
    assert [
        len(getattr(encoder.resnet_backbone, f"layer{i}")) for i in range(1, 5)
    ] == [1] * 4
    assert not any(
        isinstance(module, (nn.BatchNorm2d, nn.Dropout)) for module in encoder.modules()
    )
    assert not any(
        name.startswith("resnet_backbone.fc") for name in encoder.state_dict()
    )
    for module in encoder.modules():
        if isinstance(module, nn.GroupNorm):
            assert module.num_groups == 4 and module.eps == 1e-5
    images = torch.rand(1, 3, 256, 256, requires_grad=True)
    output = encoder(images)
    assert output.shape == (1, 256) and output.abs().max() <= 1
    output.square().mean().backward()
    assert images.grad is not None and images.grad.norm() > 0
    for name, parameter in encoder.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.norm() > 0, name


def test_reference_forward_intermediates_and_gradient_parity():
    """Independent torchvision backbone and pinned RLInf's explicit pooling."""
    torch.manual_seed(91)
    encoder = ResNet10Encoder()
    torchvision_backbone = ResNet(
        BasicBlock, [1, 1, 1, 1], norm_layer=partial(nn.GroupNorm, 4, eps=1e-5)
    )
    reference_backbone = nn.Sequential(
        OrderedDict(list(torchvision_backbone.named_children())[:-2])
    )
    reference_backbone.load_state_dict(encoder.resnet_backbone.state_dict())
    reference_pool = encoder.pooling_layer.kernel.detach().clone().requires_grad_()
    reference_head = deepcopy(encoder.mlp)
    inputs = torch.rand(1, 3, 256, 256, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    actual = encoder(inputs)
    features = reference_backbone(
        (reference_inputs - encoder.image_mean) / encoder.image_std
    )
    assert features.shape == (1, 512, 8, 8)
    torch.testing.assert_close(
        features,
        encoder.resnet_backbone((inputs - encoder.image_mean) / encoder.image_std),
        atol=0,
        rtol=0,
    )
    reference_embedding = (
        (features.unsqueeze(-1) * reference_pool.unsqueeze(0))
        .sum((2, 3))
        .reshape(1, -1)
    )
    torch.testing.assert_close(
        encoder.pooling_layer(features), reference_embedding, atol=5e-5, rtol=2e-5
    )
    expected = reference_head(reference_embedding)
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    weights = torch.linspace(-1, 1, 256).reshape(1, -1)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad, atol=2e-5, rtol=5e-4)
    torch.testing.assert_close(
        encoder.pooling_layer.kernel.grad, reference_pool.grad, atol=2e-5, rtol=5e-4
    )
    for (_, actual_parameter), (_, expected_parameter) in zip(
        encoder.resnet_backbone.named_parameters(),
        reference_backbone.named_parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, atol=3e-4, rtol=5e-4
        )


def test_rgb_normalization_once_and_noncontiguous_input_ownership():
    encoder = ResNet10Encoder().eval()
    rgb = torch.tensor([0.0, 0.5, 1.0]).view(1, 3, 1, 1).expand(1, 3, 256, 256)
    assert not rgb.is_contiguous()
    before = rgb.clone()
    seen = []
    handle = encoder.resnet_backbone.conv1.register_forward_pre_hook(
        lambda _module, arguments: seen.append(arguments[0].detach().clone())
    )
    with torch.no_grad():
        original_output = encoder(rgb)
        preserved = original_output.clone()
        encoder(torch.zeros_like(rgb))
    handle.remove()
    torch.testing.assert_close(seen[0], (rgb - encoder.image_mean) / encoder.image_std)
    torch.testing.assert_close(rgb, before, atol=0, rtol=0)
    torch.testing.assert_close(original_output, preserved, atol=0, rtol=0)


def test_batch32_and_individual_image_encoding_agree():
    torch.manual_seed(4)
    encoder = ResNet10Encoder().eval()
    image = torch.rand(1, 3, 256, 256)
    noncontiguous = image.expand(32, -1, -1, -1)
    with torch.no_grad():
        batch_features = encoder(noncontiguous)
        single = encoder(image)
    assert batch_features.shape == (32, 256)
    torch.testing.assert_close(
        batch_features, single.expand_as(batch_features), atol=5e-6, rtol=2e-5
    )


def test_checkpoint_roundtrip_preserves_outputs_and_trainability():
    torch.manual_seed(3)
    encoder = ResNet10Encoder().eval()
    image = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        expected = encoder(image)
    restored = ResNet10Encoder().eval()
    restored.load_state_dict(deepcopy(encoder.state_dict()))
    with torch.no_grad():
        torch.testing.assert_close(restored(image), expected, atol=0, rtol=0)
    assert all(parameter.requires_grad for parameter in restored.parameters())


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((0, 3, 256, 256), torch.float32),
        ((1, 256, 256, 3), torch.float32),
        ((1, 4, 256, 256), torch.float32),
        ((1, 3, 128, 128), torch.float32),
        ((1, 3, 256, 256), torch.uint8),
        ((1, 3, 256, 256), torch.float64),
    ],
)
def test_wrong_image_metadata_is_rejected(shape, dtype):
    encoder = ResNet10Encoder()
    with pytest.raises(ValueError, match="float32"):
        encoder(torch.zeros(shape, dtype=dtype))
