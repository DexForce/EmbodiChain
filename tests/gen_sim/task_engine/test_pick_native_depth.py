# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
import torch
import pytest

from embodichain.gen_sim.task_engine._task_program import actions
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordanceSample,
    AffordancePoseCandidates,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp, PickUpOptions


def setup(
    monkeypatch,
    *,
    endpoint_clear: bool,
    path_success: bool = True,
    consume_rng: bool = False,
):
    action = actions.GenSimPickUp()
    action.pick_purposes = {"probe": "ordinary"}
    action._planning_services = SimpleNamespace(device=torch.device("cpu"))
    direction = torch.tensor([[0.0, 0.0, -1.0]])
    calls = []
    options = PickUpOptions()

    def feasible(poses, start, obj, arm, opt, approach):
        if consume_rng:
            torch.rand(2)
        return poses @ opt.grasp_frame_to_eef, torch.ones(1, 1, dtype=torch.bool)

    action._select_feasible_grasp_variants = feasible

    def choose(*args, **kwargs):
        pool = kwargs.get(
            "candidate_pool",
            AffordancePoseCandidates(
                torch.eye(4)[None, None],
                torch.zeros(1, 1),
                torch.ones(1, 1, dtype=torch.bool),
            ),
        )
        kwargs["raw_candidate_observer"](pool)
        poses, valid = kwargs["feasible"](pool.poses, direction)
        kwargs["candidate_observer"](
            AffordancePoseCandidates(poses, pool.costs, pool.valid & valid)
        )
        return (
            AffordanceSample(valid[:, 0], poses[:, 0], {"candidate_ids": [0]}),
            direction,
        )

    monkeypatch.setattr(actions, "select_grasp", choose)

    def planner(self, pose, start, last, policy, opt, *args):
        if consume_rng:
            torch.rand(3)
        depth = -float(opt.grasp_frame_to_eef[2, 3])
        calls.append(depth)
        return (
            torch.tensor([path_success]),
            torch.full((1, 3, 4), depth),
            {"approach": 1, "close": 1, "lift": 1},
        )

    monkeypatch.setattr(PickUp, "_get_full_pickup_trajectory", planner)
    action._native_pick_endpoint_clear = lambda request, context, path, lengths: (
        endpoint_clear or float(path[0, 0, 0]) > 0
    )
    grasp = SimpleNamespace(
        require_target=lambda kind: object(),
        joint_positions=lambda *a, **kw: torch.zeros(1, 1),
    )
    request = SimpleNamespace(
        binding=SimpleNamespace(endpoint=lambda *a: grasp),
        motion_policy=object(),
        invocation_id="probe",
    )
    screen = SimpleNamespace(
        revision=6, depth_offsets=lambda: (0.0, 0.015, 0.03, 0.045, 0.06, 0.075)
    )
    selection = actions._PickSelection(request=request, closing_geometry=screen)
    context = SimpleNamespace(
        batch_size=1,
        robot=SimpleNamespace(qpos=torch.zeros(1, 4)),
        last_qpos=torch.zeros(1, 4),
        require_control_dt=lambda: 0.04,
    )

    def run():
        return action._select_adaptive_pick(
            object(),
            object(),
            torch.eye(4)[None],
            torch.zeros(1, 1),
            object(),
            options,
            context,
            "probe",
            (
                SimpleNamespace(
                    direction=direction, axis=None, positive=torch.tensor([True])
                ),
            ),
            selection,
        )

    return action, selection, options, calls, run


def test_depth_search_selects_first_safe_endpoint_and_keeps_same_path(monkeypatch):
    action, selection, options, calls, run = setup(monkeypatch, endpoint_clear=False)
    before = options.grasp_frame_to_eef.clone()
    sample = run()
    assert sample.success.tolist() == [True]
    assert calls == pytest.approx([0.0, 0.015])
    assert selection.depth_recovery_attempts == 1
    assert sample.metadata["adaptive_grasp"]["selected_retreat_m"] == pytest.approx(
        [0.015]
    )
    assert torch.equal(options.grasp_frame_to_eef, before)
    assert torch.allclose(selection.trajectory[1], torch.full((1, 3, 4), 0.015))
    token = actions._PICK_SELECTION.set(selection)
    try:
        assert (
            action._get_full_pickup_trajectory(*([None] * 11)) is selection.trajectory
        )
    finally:
        actions._PICK_SELECTION.reset(token)


def test_qualified_original_does_not_run_any_depth_alternative(monkeypatch):
    _, selection, _, calls, run = setup(monkeypatch, endpoint_clear=True)
    sample = run()
    assert sample.success.all()
    assert calls == [0.0]
    assert selection.depth_recovery_attempts == 0


def test_revision_seven_retains_existing_depth_budget_and_first_safe_selection(
    monkeypatch,
):
    _, selection, _, calls, run = setup(monkeypatch, endpoint_clear=False)
    selection.closing_geometry.revision = 7
    assert run().success.all()
    assert calls == pytest.approx([0.0, 0.015])
    assert selection.depth_recovery_attempts == 1


def test_depth_paths_share_existing_four_recovery_attempts(monkeypatch):
    action, selection, _, calls, run = setup(
        monkeypatch, endpoint_clear=False, path_success=False
    )
    assert not run().success.any()
    assert len(calls) == 5
    assert selection.depth_recovery_attempts == 4
    first = SimpleNamespace(
        plan_success=torch.tensor([False]), diagnostics=SimpleNamespace(metadata={})
    )
    assert action._recover_raw_pick_candidates(None, None, first, selection) is first
    assert action._retry_ordinary_pick_candidates(None, None, first, selection) is first


def test_extra_depths_reuse_original_ik_rng_without_advancing_global_stream(
    monkeypatch,
):
    _, _, _, calls, run = setup(monkeypatch, endpoint_clear=False, consume_rng=True)
    torch.manual_seed(123)
    torch.rand(5)
    expected = torch.random.get_rng_state().clone()
    torch.manual_seed(123)
    assert run().success.all()
    assert calls == pytest.approx([0.0, 0.015])
    assert torch.equal(torch.random.get_rng_state(), expected)


def test_cached_depth_pool_never_calls_upstream_generator() -> None:
    from embodichain.gen_sim.task_engine._task_program.adaptive_grasp import (
        select_grasp,
        GraspStrategy,
    )

    pool = AffordancePoseCandidates(
        torch.eye(4)[None, None], torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool)
    )
    source = SimpleNamespace(
        get_grasp_candidates=Mock(side_effect=AssertionError("regenerated")),
        sample_candidates=Mock(
            return_value=AffordanceSample(
                torch.tensor([True]), torch.eye(4)[None], {"candidate_ids": [0]}
            )
        ),
    )
    choices = (
        GraspStrategy(torch.tensor([[0.0, 0.0, -1.0]]), None, torch.tensor([True])),
    )
    context = SimpleNamespace(affordance_sampling=None, env_ids=torch.tensor([0]))
    sample, _ = select_grasp(
        source,
        object(),
        torch.eye(4)[None],
        choices,
        context,
        "same",
        candidate_pool=pool,
    )
    assert sample.success.all()
    source.get_grasp_candidates.assert_not_called()
    with pytest.raises(ValueError, match="exactly one"):
        select_grasp(
            source,
            object(),
            torch.eye(4)[None],
            choices * 2,
            context,
            "same",
            candidate_pool=pool,
        )
