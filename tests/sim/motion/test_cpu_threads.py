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

from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.motion.cpu_threads import (
    cpu_planning_threads,
    candidate_ik_threads,
)
from embodichain.lab.sim.motion.solvers.pytorch_solver import PytorchSolver


def thread_fixture(monkeypatch, initial):
    state = {"threads": initial, "sets": [], "reads": 0, "solver_reads": 0}

    def get():
        state["reads"] += 1
        return state["threads"]

    def set_count(value):
        state["threads"] = value
        state["sets"].append(value)

    monkeypatch.setattr(torch, "get_num_threads", get)
    monkeypatch.setattr(torch, "set_num_threads", set_count)
    solver = object.__new__(PytorchSolver)

    def get_solver(part):
        state["solver_reads"] += 1
        assert part == "arm"
        return solver

    return state, SimpleNamespace(get_solver=get_solver)


@pytest.mark.parametrize("initial,batch", [(1, 1), (4, 4), (16, 8)])
def test_cpu_scope_captures_source_capacity_and_restores_nested_batch_threads(
    monkeypatch, initial, batch
):
    state, robot = thread_fixture(monkeypatch, initial)
    poses = torch.empty(1, 10, 10, 4, 4)
    with cpu_planning_threads(device="cpu", num_envs=1) as budget:
        assert (
            budget.initial_threads,
            budget.serial_threads,
            budget.batch_threads,
        ) == (initial, 1, batch)
        assert state["threads"] == 1
        with candidate_ik_threads(robot, "arm", poses):
            assert state["threads"] == batch
            with candidate_ik_threads(robot, "arm", poses):
                assert state["threads"] == batch
        assert state["threads"] == 1
    assert state["threads"] == initial
    assert state["sets"] == ([1, 1] if initial == 1 else [1, batch, 1, initial])
    reads = state["solver_reads"]
    with candidate_ik_threads(robot, "arm", poses):
        assert state["threads"] == initial
    assert state["solver_reads"] == reads


@pytest.mark.parametrize(
    "shape,device,solver",
    [
        ((1, 1, 99, 4, 4), "cpu", "pytorch"),
        ((2, 10, 10, 4, 4), "cpu", "pytorch"),
        ((1, 10, 10, 4, 4), "meta", "pytorch"),
        ((1, 10, 10, 4, 4), "cpu", "other"),
    ],
)
def test_small_batched_non_cpu_and_other_solver_paths_keep_serial_threads(
    monkeypatch, shape, device, solver
):
    state, robot = thread_fixture(monkeypatch, 16)
    if solver == "other":
        robot.get_solver = lambda _: object()
    with cpu_planning_threads(device="cpu", num_envs=1):
        with candidate_ik_threads(robot, "arm", torch.empty(shape, device=device)):
            assert state["threads"] == 1
    assert state["sets"] == [1, 16]


def test_cpu_thread_scope_restores_after_solver_failure_and_never_lowers_higher_budget(
    monkeypatch,
):
    state, robot = thread_fixture(monkeypatch, 16)
    poses = torch.empty(1, 50, 2, 4, 4)
    with pytest.raises(RuntimeError, match="solver failed"):
        with cpu_planning_threads(device="cpu", num_envs=1):
            with candidate_ik_threads(robot, "arm", poses):
                assert state["threads"] == 8
                raise RuntimeError("solver failed")
    assert state["threads"] == 16 and state["sets"] == [1, 8, 1, 16]
    state["sets"].clear()
    with cpu_planning_threads(device="cpu", num_envs=1):
        state["threads"] = 12
        with candidate_ik_threads(robot, "arm", poses):
            assert state["threads"] == 12
    assert state["sets"] == [1, 16]


@pytest.mark.parametrize(
    "device,count",
    [
        pytest.param("cuda:0", 1, id="non_cpu_metadata"),
        pytest.param("cpu", 2, id="multiple_environments"),
    ],
)
def test_non_source_worker_scope_does_not_observe_or_change_thread_pool(
    monkeypatch, device, count
):
    state, _ = thread_fixture(monkeypatch, 16)
    with cpu_planning_threads(device=device, num_envs=count) as budget:
        assert budget is None
    assert state["reads"] == 0 and state["sets"] == []
