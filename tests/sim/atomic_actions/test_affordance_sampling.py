# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Tests for reproducible simulation-level Affordance sampling values."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
    AffordanceSample,
    AffordanceSamplingContext,
)


def _poses(batch_size: int, candidate_count: int | None = None) -> torch.Tensor:
    if candidate_count is None:
        return torch.eye(4).repeat(batch_size, 1, 1)
    return torch.eye(4).repeat(batch_size, candidate_count, 1, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("count", 0),
        ("count", True),
        ("seed", -1),
        ("episode_id", 1.5),
        ("attempt_id", -1),
    ],
)
def test_sampling_context_rejects_invalid_stream_identity(field: str, value: object):
    values = {"count": 2, "seed": 3, "episode_id": 4, "attempt_id": 5}
    values[field] = value
    with pytest.raises(ValueError):
        AffordanceSamplingContext(**values)


def test_sampling_context_uses_private_reproducible_generators():
    context = AffordanceSamplingContext(count=4, seed=3, episode_id=2, attempt_id=1)
    global_state = torch.random.get_rng_state().clone()

    first = torch.rand(5, generator=context.generator("grasp", group=7))
    second = torch.rand(5, generator=context.generator("grasp", group=7))
    changed = torch.rand(
        5,
        generator=replace(context, attempt_id=2).generator("grasp", group=7),
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_pose_candidates_from_rows_preserves_empty_and_nonfinite_validity():
    finite = _poses(2)
    nonfinite = finite.clone()
    nonfinite[1, 0, 0] = torch.nan
    candidates = AffordancePoseCandidates.from_rows(
        [
            (nonfinite, torch.tensor([0.2, 0.1])),
            (torch.empty(0, 4, 4), torch.empty(0)),
        ],
        device="cpu",
    )

    assert candidates.poses.shape == (2, 2, 4, 4)
    assert candidates.valid.tolist() == [[True, False], [False, False]]
    assert torch.isinf(candidates.costs[1]).all()
    torch.testing.assert_close(candidates.poses[1], _poses(2))


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "poses": torch.eye(4),
            "costs": torch.zeros(1, 1),
            "valid": torch.ones(1, 1, dtype=torch.bool),
        },
        {
            "poses": _poses(1, 2),
            "costs": torch.zeros(1, 1),
            "valid": torch.ones(1, 2, dtype=torch.bool),
        },
        {
            "poses": _poses(1, 2),
            "costs": torch.zeros(1, 2),
            "valid": torch.ones(1, 2),
        },
    ],
)
def test_pose_candidates_rejects_invalid_tensor_contract(kwargs: dict[str, object]):
    with pytest.raises((TypeError, ValueError)):
        AffordancePoseCandidates(**kwargs)


def test_affordance_sample_validates_and_owns_result_values():
    success = torch.tensor([True, False])
    poses = _poses(2)
    metadata = {"candidate_ids": [0, -1]}
    sample = AffordanceSample(success=success, poses=poses, metadata=metadata)

    success[0] = False
    poses[0, 0, 0] = 7.0
    metadata["candidate_ids"] = [9, 9]

    assert sample.success.tolist() == [True, False]
    assert sample.poses[0, 0, 0] == 1.0
    assert sample.metadata == {"candidate_ids": [0, -1]}


@pytest.mark.parametrize(
    "success,poses,metadata",
    [
        (torch.ones(2), _poses(2), {}),
        (torch.ones(1, dtype=torch.bool), _poses(2), {}),
        (torch.ones(2, dtype=torch.bool), _poses(2).to(torch.int64), {}),
        (torch.ones(2, dtype=torch.bool), _poses(2), []),
    ],
)
def test_affordance_sample_rejects_invalid_result_contract(
    success: torch.Tensor,
    poses: torch.Tensor,
    metadata: object,
):
    with pytest.raises((TypeError, ValueError)):
        AffordanceSample(success=success, poses=poses, metadata=metadata)
