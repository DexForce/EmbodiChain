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

"""General learning components shared by supervised models and RL."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from embodichain.learning.models import EmpiricalNormalizer, MLP, build_mlp_from_cfg


def test_configured_network_supports_supervised_gradients_and_initialization():
    model = build_mlp_from_cfg(
        {
            "type": "mlp",
            "network_cfg": {
                "hidden_sizes": [6],
                "activation": "silu",
                "last_activation": "tanh",
                "use_layernorm": True,
                "orthogonal_init": [1.0, 0.5],
            },
        },
        3,
        2,
    ).double()
    assert isinstance(model, MLP)
    assert any(isinstance(module, nn.LayerNorm) for module in model)
    inputs = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.4, 0.5, 0.6]], dtype=torch.float64, requires_grad=True
    )
    prediction = model(inputs)
    assert prediction.shape == (2, 2)
    assert prediction.abs().max() <= 1
    prediction.square().sum().backward()
    assert torch.isfinite(inputs.grad).all() and inputs.grad.norm() > 0
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    linear = next(module for module in model if isinstance(module, nn.Linear))
    torch.testing.assert_close(
        linear.weight.T @ linear.weight,
        torch.eye(3, dtype=torch.float64),
        atol=1e-6,
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="Only 'mlp'"):
        build_mlp_from_cfg({"type": "unknown"}, 3, 2)


def test_feature_statistics_roundtrip_and_preserve_input_gradients():
    features = torch.tensor(
        [[1.0, 2.0], [3.0, 6.0], [5.0, 4.0], [7.0, 8.0]], dtype=torch.float64
    )
    normalizer = EmpiricalNormalizer(2).double()
    normalizer.update(features[:2])
    normalizer.update(features[2:])
    torch.testing.assert_close(normalizer.mean, features.mean(0, keepdim=True))
    torch.testing.assert_close(
        normalizer.variance, features.var(0, unbiased=False, keepdim=True)
    )
    restored = EmpiricalNormalizer(2).double().eval()
    restored.load_state_dict(deepcopy(normalizer.state_dict()))
    before = deepcopy(restored.state_dict())
    restored.update(features * 10)
    for name, value in before.items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    inputs = features.clone().requires_grad_()
    result = restored(inputs)
    torch.testing.assert_close(result, normalizer(features))
    gradient = torch.autograd.grad(result.sum(), inputs)[0]
    torch.testing.assert_close(
        gradient,
        (1 / (restored.standard_deviation + restored.epsilon)).expand_as(features),
    )
