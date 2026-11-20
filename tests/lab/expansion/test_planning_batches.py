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


"""Regression coverage for cooperative planning execution contexts."""

from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.expansion._planning_batches import batched_planning


@pytest.mark.parametrize("fail", [False, True])
def test_autocast_rows_preserve_precision_and_restore_owner(fail):
    class Solver:
        def get_fk(self, qpos):
            if fail and qpos[0, 0] < 0:
                raise ValueError("solver failure")
            return qpos @ torch.eye(2)

    solver = Solver()
    robot = SimpleNamespace(device=torch.device("cpu"), _solvers={"arm": solver})
    inputs = torch.tensor([[1.1234, 2.3456]])

    class Cursor:
        def __init__(self, enabled, value):
            self.enabled, self.value = enabled, value

        def next(self):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.enabled):
                first = solver.get_fk(self.value)
                # Resuming a row must restore the same context for subsequent calls.
                second = solver.get_fk(self.value)
                return first + second

    cursors = {
        0: Cursor(True, inputs),
        1: Cursor(False, inputs),
        2: Cursor(True, inputs),
    }
    expected = {row: cursor.next() for row, cursor in cursors.items()}
    cpu_rng = torch.get_rng_state().clone()
    with torch.autocast("cpu", enabled=False):
        with batched_planning(robot, 7) as planner:
            if fail:
                cursors[1].value = -inputs
                with pytest.raises(ValueError, match="solver failure"):
                    planner.gather(cursors)
            else:
                actual = planner.gather(cursors)
                for row in actual:
                    torch.testing.assert_close(
                        actual[row], expected[row], rtol=0, atol=0
                    )
                saved = actual[2].clone()
                actual[0].zero_()
                torch.testing.assert_close(actual[2], saved)
            assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cpu")
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert "get_fk" not in solver.__dict__


@pytest.mark.gpu
@pytest.mark.parametrize("robot_device", ["cpu", "cuda:0"])
def test_planning_scope_leaves_external_rng_generators_unchanged(robot_device):
    robot = SimpleNamespace(device=torch.device(robot_device), _solvers={})
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(123)
        cpu = torch.get_rng_state().clone()
        cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
        with batched_planning(robot, 7):
            assert torch.equal(cpu, torch.get_rng_state())
            for before, after in zip(cuda, torch.cuda.get_rng_state_all()):
                assert torch.equal(before, after)
        assert torch.equal(cpu, torch.get_rng_state())
        for before, after in zip(cuda, torch.cuda.get_rng_state_all()):
            assert torch.equal(before, after)


def test_full_motion_grouping_preserves_autocast(monkeypatch):
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenerator,
        MotionGenOptions,
    )
    from embodichain.lab.sim.motion.planners.utils import PlanState, PlanResult

    robot = SimpleNamespace(device=torch.device("cpu"), _solvers={})

    def plan(self, targets, options):
        positions = (options.start_qpos @ torch.eye(2))[:, None]
        return PlanResult(
            success=torch.ones(len(positions), dtype=torch.bool),
            positions=positions,
            dt=torch.zeros(len(positions), 1),
        )

    monkeypatch.setattr(MotionGenerator, "_generate_once", plan)

    class Cursor:
        def __init__(self, row, enabled):
            self.enabled = enabled
            self.generator = object.__new__(MotionGenerator)
            self.generator.robot = robot
            self.generator.planner = SimpleNamespace(
                cfg=None, collision_world_info=None
            )
            self.generator._env_ids = (row,)

        def next(self):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.enabled):
                start = torch.tensor([[1.1234, 2.3456]])
                return self.generator.generate(
                    [PlanState.from_qpos(start)],
                    MotionGenOptions(
                        strategy="ik_interp",
                        interpolation_mode="fixed_grid",
                        sample_count=2,
                        interpolation_dt=0.1,
                        start_qpos=start,
                    ),
                )

    cursors = {0: Cursor(0, True), 1: Cursor(1, False), 2: Cursor(2, True)}
    expected = {row: cursor.next() for row, cursor in cursors.items()}
    with batched_planning(robot, 7) as planner:
        actual = planner.gather(cursors)
    for row in actual:
        torch.testing.assert_close(
            actual[row].positions, expected[row].positions, rtol=0, atol=0
        )
    saved = actual[2].positions.clone()
    actual[0].positions.zero_()
    torch.testing.assert_close(actual[2].positions, saved)
