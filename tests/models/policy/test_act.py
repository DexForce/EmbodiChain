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

"""Behavioral contracts for ACT padding, gradients and temporal execution."""

from __future__ import annotations

import math
import subprocess
import sys

import torch

from embodichain.learning.sl import SupervisedBatch
from embodichain_models.policy.act import (
    ACTCfg,
    ACTObjective,
    ACTPolicy,
    ACTTemporalEnsembler,
)


def _model() -> ACTPolicy:
    return ACTPolicy(
        ACTCfg(
            state_dim=3,
            action_dim=2,
            horizon=4,
            hidden_dim=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            feedforward_dim=64,
            latent_dim=4,
            dropout=0.0,
            pretrained_backbone=False,
        )
    )


def test_public_policy_import_does_not_load_simulation():
    code = (
        "import sys; from embodichain_models.policy.act import ACTPolicy; "
        "assert not any(n == 'dexsim' or n.startswith(('dexsim.', 'embodichain.lab.sim')) "
        "for n in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_padded_actions_do_not_affect_posterior_or_loss_and_gradients_reach_model():
    torch.manual_seed(12)
    model = _model().eval()
    rgb = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    state = torch.randn(2, 3)
    actions = torch.randn(2, 4, 2)
    valid = torch.tensor([[True, True, False, False], [True, True, True, False]])
    altered = actions.clone()
    altered[~valid] = 1000.0
    objective = ACTObjective(kl_weight=10)

    torch.manual_seed(3)
    first = objective.compute_loss(
        model, SupervisedBatch({"rgb": rgb, "state": state}, actions, {"valid": valid})
    )
    torch.manual_seed(3)
    second = objective.compute_loss(
        model, SupervisedBatch({"rgb": rgb, "state": state}, altered, {"valid": valid})
    )
    torch.testing.assert_close(first.loss, second.loss)
    first.loss.backward()
    assert model.backbone[0].weight.grad is not None
    assert model.backbone[0].weight.grad.abs().sum() > 0
    assert model.posterior_stats.weight.grad is not None
    assert model.posterior_stats.weight.grad.abs().sum() > 0


def test_inference_is_deterministic_and_denormalizes_actions():
    torch.manual_seed(8)
    model = _model().eval()
    model.set_normalization(
        torch.zeros(3),
        torch.ones(3),
        torch.tensor([2.0, -1.0]),
        torch.tensor([0.5, 2.0]),
    )
    rgb = torch.zeros(1, 3, 64, 64, dtype=torch.uint8)
    state = torch.zeros(1, 3)
    with torch.no_grad():
        normalized, mu, logvar = model(rgb, state)
        actual = model.predict(rgb, state)
    assert mu is None and logvar is None
    torch.testing.assert_close(
        actual, normalized * torch.tensor([0.5, 2.0]) + torch.tensor([2.0, -1.0])
    )


def test_objective_matches_independent_masked_l1_and_kl_values():
    class KnownPrediction(torch.nn.Module):
        action_mean = torch.zeros(2)
        action_std = torch.ones(2)

        def forward(self, rgb, state, *, actions, valid):
            return (
                torch.zeros_like(actions),
                torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                torch.zeros(2, 2),
            )

    actions = torch.tensor([[[2.0, 4.0], [999.0, 999.0]], [[1.0, 1.0], [3.0, 3.0]]])
    valid = torch.tensor([[True, False], [True, True]])
    result = ACTObjective(kl_weight=10.0).compute_loss(
        KnownPrediction(),
        SupervisedBatch(
            {"rgb": torch.empty(2, 3, 1, 1), "state": torch.empty(2, 1)},
            actions,
            {"valid": valid},
        ),
    )
    torch.testing.assert_close(result.metrics["action_l1"], torch.tensor(2.5))
    torch.testing.assert_close(result.metrics["kl"], torch.tensor(0.25))
    torch.testing.assert_close(result.loss, torch.tensor(5.0))


def test_ensemble_aligns_overlapping_chunks_and_resets_individual_rows():
    ensemble = ACTTemporalEnsembler(3, 2, 1, decay=0.2)
    first = torch.tensor([[[0.0], [2.0], [4.0]], [[5.0], [7.0], [9.0]]])
    second = torch.tensor([[[10.0], [12.0], [14.0]], [[15.0], [17.0], [19.0]]])
    torch.testing.assert_close(ensemble.update(first), torch.tensor([[0.0], [5.0]]))
    ensemble.reset(torch.tensor([1]))
    actual = ensemble.update(second)
    expected_first = (2.0 * math.exp(0.2) + 10.0) / (math.exp(0.2) + 1.0)
    torch.testing.assert_close(actual, torch.tensor([[expected_first], [15.0]]))
    ensemble.reset()
    torch.testing.assert_close(ensemble.update(first), torch.tensor([[0.0], [5.0]]))

    extreme = ACTTemporalEnsembler(3, 1, 1, decay=10000.0)
    extreme.update(first[:1])
    torch.testing.assert_close(extreme.update(second[:1]), torch.tensor([[2.0]]))
