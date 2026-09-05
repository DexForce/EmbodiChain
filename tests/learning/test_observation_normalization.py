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

"""Tests for semantic-mask-aware running observation normalization."""

from __future__ import annotations

import pytest
import torch

from embodichain.learning.rl.normalization import RunningObservationNormalizer


def test_running_normalizer_matches_combined_population_statistics() -> None:
    normalizer = RunningObservationNormalizer(3, "cpu")
    first = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    second = torch.tensor([[5.0, 6.0, 7.0]])

    normalizer.update(first)
    normalizer.update(second)

    combined = torch.cat([first, second])
    # The small pseudo-count has zero mean, unit variance, matching the NMG
    # training reference rather than an exact batch-only moment.
    expected_count = 3.0001
    expected_mean = combined.sum(dim=0) / expected_count
    expected_m2 = (
        combined.square().sum(dim=0) + 1.0e-4 - expected_count * expected_mean.square()
    )
    torch.testing.assert_close(normalizer.mean, expected_mean)
    torch.testing.assert_close(normalizer.var, expected_m2 / expected_count)
    assert normalizer.count == pytest.approx(expected_count)


def test_running_normalizer_preserves_semantic_fields_and_gradients() -> None:
    normalizer = RunningObservationNormalizer(
        3,
        "cpu",
        normalize_mask=torch.tensor([True, False, True]),
    )
    normalizer.update(torch.tensor([[1.0, 0.0, 3.0], [3.0, 1.0, 7.0]]))
    observation = torch.tensor([[2.0, 1.0, 5.0]], requires_grad=True)

    normalized = normalizer.normalize(observation)
    normalized.sum().backward()

    assert normalized[0, 1] == 1.0
    assert observation.grad is not None
    assert torch.isfinite(observation.grad).all()
    assert observation.grad[0, 1] == 1.0


def test_running_normalizer_checkpoint_round_trip() -> None:
    source = RunningObservationNormalizer(2, "cpu", torch.tensor([True, False]))
    source.update(torch.tensor([[2.0, 1.0], [4.0, 0.0]]))
    restored = RunningObservationNormalizer(2, "cpu")

    restored.load_state_dict(source.state_dict())

    torch.testing.assert_close(restored.mean, source.mean)
    torch.testing.assert_close(restored.var, source.var)
    assert restored.count == source.count
    assert torch.equal(restored.normalize_mask, source.normalize_mask)
