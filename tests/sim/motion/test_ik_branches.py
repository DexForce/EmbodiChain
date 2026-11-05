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

"""Direct CPU branch/path calls retain the frozen source candidate and RNG law."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from embodichain.compute.kinematics.seed_selection import (
    select_separated_joint_branches,
)
from embodichain.lab.sim.motion.ik_branches import CpuIKBranchSearch
from embodichain.lab.sim.motion.search import search_cost_scope
from _ik_branch_fixture import branch_fixture


def trace_case(tmp_path, mode):
    robot, solver, root, calls, reads = branch_fixture(
        tmp_path, failed=mode == "failed"
    )
    search = CpuIKBranchSearch.capture(robot, "arm", 0.04)
    start = (
        torch.tensor([[1.98, 0.0]])
        if mode == "near_limit"
        else torch.tensor([[0.0, 0.1]])
    )
    target = torch.eye(4)[None]
    target[0, 0, 3] = 1.99 if mode == "near_limit" else 0.3
    target[0, 2, 3] = 0.2
    target = search.to_root(root @ target)
    torch.manual_seed(217)
    before = torch.get_rng_state().clone()
    endpoints = search.endpoints(target, start)
    assert torch.equal(before, torch.get_rng_state())
    path_targets = target.repeat(2, 1, 1)
    path_targets[:, 0, 3] = (
        torch.tensor([1.99, 1.98]) if mode == "near_limit" else torch.tensor([0.2, 0.3])
    )
    path = search.path(path_targets, start)
    assert torch.equal(before, torch.get_rng_state())
    return {
        "endpoints": [[i, q.tolist()] for i, q in endpoints],
        "path": None if path is None else path.tolist(),
        "calls": [
            {**c, "rng": hashlib.sha256(c["rng"].numpy().tobytes()).hexdigest()}
            for c in calls
        ],
        "pose_reads": reads,
    }


@pytest.mark.parametrize("mode", ["ordinary", "near_limit", "failed"])
def test_cpu_branch_search_matches_source_oracle(tmp_path, monkeypatch, mode):
    # Keep this CPU unit test from touching busy CUDA contexts; retain and assert
    # the source manual_seed_all invocation separately from CPU state equality.
    cuda_seeds = []
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda seed: cuda_seeds.append(seed)
    )
    with search_cost_scope() as costs:
        actual = trace_case(tmp_path, mode)
    expected = json.loads(
        (Path(__file__).parent / "data/ik_branches_source.json").read_text()
    )[mode]
    assert actual == expected
    assert cuda_seeds == [217, 0, 0]
    assert costs.counters["cpu_ik_seeds"] >= 32
    assert costs.counters["cpu_ik_selected_branches"] == len(actual["endpoints"])


@pytest.mark.parametrize(
    "dtype, expected", [(torch.float64, (0, 1, 2)), (torch.float32, (0,))]
)
def test_branch_selector_keeps_stable_ties_and_strict_separation(dtype, expected):
    # float32's representation of 0.02 is strictly below the source double
    # threshold; retain that boundary rather than silently adding an epsilon.
    q = torch.tensor([[0.0], [0.02], [-0.02], [0.019], [0.02]], dtype=dtype)
    assert (
        select_separated_joint_branches(
            q, torch.ones(5, dtype=torch.bool), torch.zeros(1, 1, dtype=dtype)
        )
        == expected
    )
    q[3] = float("nan")
    assert (
        select_separated_joint_branches(
            q,
            torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool),
            torch.zeros(1, 1, dtype=dtype),
        )
        == expected
    )


def test_unsupported_seed_sampler_rejects_before_live_root_or_ik(tmp_path):
    robot, solver, root, calls, reads = branch_fixture(tmp_path)
    solver._seed_sampler = object()
    assert not CpuIKBranchSearch.supports(robot, "arm")
    with pytest.raises(ValueError, match="custom seed sampler"):
        CpuIKBranchSearch.capture(robot, "arm", 0.04)
    assert not calls and not reads
