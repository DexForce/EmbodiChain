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

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from embodichain.compute.kinematics.ik_path import resolve_seeded_ik_path

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/seeded_ik_reference.json").read_text()
)


@pytest.mark.parametrize("case", _REFERENCE["cases"], ids=lambda case: case["mode"])
def test_seeded_path_matches_frozen_solver_calls_and_branch_selection(case):
    calls = iter(case["calls"])
    counts = Counter()

    def replay(kind, pose, seed):
        expected = next(calls)
        assert expected["kind"] == kind
        torch.testing.assert_close(pose, torch.tensor(expected["pose"]), rtol=0, atol=0)
        torch.testing.assert_close(seed, torch.tensor(expected["seed"]), rtol=0, atol=0)
        return torch.tensor(expected["valid"]), torch.tensor(expected["qpos"])

    result = resolve_seeded_ik_path(
        torch.tensor(case["poses"]),
        torch.tensor(case["start"]),
        solve=lambda pose, seed: replay("solve", pose, seed),
        solve_many=(
            (lambda pose, seed: replay("many", pose, seed))
            if case["mode"] == "batched"
            else None
        ),
        lower=torch.tensor(case["lower"]),
        upper=torch.tensor(case["upper"]),
        limits=torch.tensor(case["limits"]),
        dt=case["dt"],
        record_operation=lambda name, count: counts.update({name: count}),
    )
    assert next(calls, None) is None
    assert dict(counts) == case["counters"]
    if case["result"] is None:
        assert result is None
    else:
        torch.testing.assert_close(result, torch.tensor(case["result"]), rtol=0, atol=0)


def test_seeded_path_preserves_single_environment_and_candidate_filter_boundary():
    kwargs = dict(
        poses=torch.eye(4)[None],
        start=torch.zeros(1, 1),
        solve=lambda pose, seed: (torch.tensor([True]), seed),
        lower=torch.tensor([[-1.0]]),
        upper=torch.tensor([[1.0]]),
        limits=torch.ones(1, 1),
        dt=0.1,
    )
    with pytest.raises(ValueError, match="one environment"):
        resolve_seeded_ik_path(**{**kwargs, "start": torch.zeros(2, 1)})
    with pytest.raises(ValueError, match="serial solver"):
        resolve_seeded_ik_path(
            **kwargs,
            solve_many=kwargs["solve"],
            candidate_predicate=lambda candidate: False,
        )
    assert (
        resolve_seeded_ik_path(**kwargs, candidate_predicate=lambda candidate: False)
        is None
    )


def test_seeded_path_rejects_final_fk_failure_and_zero_velocity():
    kwargs = dict(
        poses=torch.eye(4)[None],
        start=torch.zeros(1, 1),
        solve=lambda pose, seed: (torch.tensor([True]), seed),
        lower=torch.tensor([[-1.0]]),
        upper=torch.tensor([[1.0]]),
        limits=torch.ones(1, 1),
        dt=0.1,
    )
    assert (
        resolve_seeded_ik_path(
            **kwargs, pose_valid=lambda qpos, poses: torch.tensor([False])
        )
        is None
    )
    with pytest.raises(ValueError, match="positive velocity"):
        resolve_seeded_ik_path(**{**kwargs, "limits": torch.zeros(1, 1)})
