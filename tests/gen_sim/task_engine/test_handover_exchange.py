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

from __future__ import annotations

import json
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.handover_exchange import (
    select_exchange,
)
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
)
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget


@pytest.fixture
def case():
    station = torch.eye(4)[None]
    station[:, :3, 3] = torch.tensor([0.1, 0.0, 0.7])
    poses = station[:, None].repeat(1, 3, 1, 1)
    poses[0, :, 0, 3] += torch.tensor([0.0, 0.01, 0.02])
    pool = AffordancePoseCandidates(
        poses, torch.tensor([[2.0, 1.0, 3.0]]), torch.tensor([[True, True, True]])
    )
    original_mask = torch.zeros(1, 3, dtype=torch.bool)
    source, receive = JointPositionTarget("source", (0, 1)), JointPositionTarget(
        "receive", (2, 3)
    )
    resources = SimpleNamespace(
        first=SimpleNamespace(arm=source), second=SimpleNamespace(arm=receive)
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        affordance_sampling=None,
        robot=SimpleNamespace(qpos=torch.zeros(1, 4)),
    )
    request = SimpleNamespace(goal=SimpleNamespace(target_pose=torch.eye(4)))
    offset = torch.eye(4)[None]
    offset[:, 2, 3] = 0.2
    held = SimpleNamespace(object_to_eef=offset)
    solver = SimpleNamespace(
        lower_qpos_limits=torch.full((2,), -0.8),
        upper_qpos_limits=torch.full((2,), 0.8),
    )

    def ik(**kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        return torch.tensor([True]), kwargs["joint_seed"] + 0.01

    robot = SimpleNamespace(
        device="cpu",
        get_qpos_limits=Mock(return_value=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])),
        get_solver=Mock(return_value=solver),
        compute_ik=Mock(side_effect=ik),
    )

    def proposal(shift=0.1, *, mask=None, axis="+free_horizontal"):
        return {
            "translation": torch.tensor([[shift, 0.0, 0.0]]),
            "mask": torch.tensor([[True, True, False]]) if mask is None else mask,
            "report": {"scope": "native_nominal_approach", "qualified": True},
            "axis": axis,
            "norm_m": abs(shift),
        }

    screen = SimpleNamespace(
        exchange_candidates=Mock(return_value=([proposal()], {"max_proposals": 3}))
    )
    roots = torch.eye(4).repeat(1, 1, 1)
    return SimpleNamespace(
        screen=screen,
        robot=robot,
        request=request,
        context=context,
        resources=resources,
        held=held,
        station=station,
        roots=roots,
        pool=pool,
        mask=original_mask,
        proposal=proposal,
        solver=solver,
    )


def select(case):
    return select_exchange(
        case.screen,
        case.robot,
        case.request,
        case.context,
        case.resources,
        case.held,
        case.station,
        case.roots,
        case.roots,
        case.pool,
        case.mask,
        92,
        0.15,
    )


def assert_unchanged(result, case):
    station, poses, mask, _ = result
    assert station is case.station and poses is case.pool.poses and mask is case.mask


def rng_snapshot():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.random.get_rng_state().clone(),
    )


def assert_rng_equal(before):
    after = rng_snapshot()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and before[1][2:] == after[1][2:]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert torch.equal(before[2], after[2])


def test_safe_original_has_no_proposals_ik_or_rng_change(case):
    case.mask[0, 1] = True
    rng = rng_snapshot()
    result = select(case)
    assert_unchanged(result, case)
    assert result[3]["reason"] == "original_station_qualified"
    case.screen.exchange_candidates.assert_not_called()
    case.robot.compute_ik.assert_not_called()
    case.robot.get_solver.assert_not_called()
    assert_rng_equal(rng)


def test_empty_original_pool_does_not_generate_station_or_new_candidates(case):
    case.pool.valid.zero_()
    result = select(case)
    assert_unchanged(result, case)
    assert result[3]["reason"] == "original_pool_empty"
    case.screen.exchange_candidates.assert_not_called()
    case.robot.compute_ik.assert_not_called()


def test_shift_is_rigid_preserves_pool_costs_state_and_original_seed_rng(case):
    old_pose, old_costs, old_valid = (
        case.pool.poses.clone(),
        case.pool.costs.clone(),
        case.pool.valid.clone(),
    )
    state, target, offset = (
        case.context.robot.qpos.clone(),
        case.request.goal.target_pose.clone(),
        case.held.object_to_eef.clone(),
    )
    rng = rng_snapshot()
    station, poses, mask, evidence = select(case)
    assert evidence["accepted"] is True and evidence["selected_pool_index"] == 1
    assert evidence["ik_queries"] == 3
    assert torch.equal(poses[..., :3, :3], old_pose[..., :3, :3])
    torch.testing.assert_close(
        torch.linalg.inv(station[:, None]) @ poses,
        torch.linalg.inv(case.station[:, None]) @ old_pose,
    )
    assert (
        torch.equal(case.pool.poses, old_pose)
        and torch.equal(case.pool.costs, old_costs)
        and torch.equal(case.pool.valid, old_valid)
    )
    assert (
        torch.equal(case.context.robot.qpos, state)
        and torch.equal(case.request.goal.target_pose, target)
        and torch.equal(case.held.object_to_eef, offset)
    )
    calls = case.robot.compute_ik.call_args_list
    assert [call.kwargs["name"] for call in calls] == ["source", "receive", "receive"]
    torch.testing.assert_close(calls[0].kwargs["pose"], station @ offset)
    torch.testing.assert_close(
        calls[1].kwargs["pose"][:, :3, 3],
        poses[:, 1, :3, 3] - poses[:, 1, :3, 2] * 0.15,
    )
    torch.testing.assert_close(calls[2].kwargs["joint_seed"], torch.full((1, 2), 0.01))
    assert mask.tolist() == [[True, True, False]]
    assert_rng_equal(rng)
    json.dumps(evidence, allow_nan=False)


def test_stations_are_ordered_by_geometry_norm_not_input_order(case):
    case.screen.exchange_candidates.return_value = (
        [case.proposal(0.3), case.proposal(0.1), case.proposal(-0.2)],
        {},
    )
    result = select(case)
    assert result[3]["selected_translation"][0][0] == pytest.approx(0.1)
    assert (
        len(result[3]["station_attempts"]) == 1
        and case.robot.compute_ik.call_count == 3
    )


def test_three_stations_and_nine_queries_are_the_complete_budget(case):
    case.screen.exchange_candidates.return_value = (
        [case.proposal(x) for x in (0.3, 0.1, 0.2)],
        {},
    )
    count = 0

    def reject_grasp(**kwargs):
        nonlocal count
        count += 1
        return torch.tensor([count % 3 != 0]), kwargs["joint_seed"]

    case.robot.compute_ik.side_effect = reject_grasp
    result = select(case)
    assert_unchanged(result, case)
    assert result[3]["ik_queries"] == 9 and case.robot.compute_ik.call_count == 9
    assert len(result[3]["station_attempts"]) == 3
    assert result[3]["reason"] == "no_qualified_exchange_station"


@pytest.mark.parametrize("damage", ["runtime", "solver", "nonfinite", "false", "shape"])
def test_endpoint_ik_and_intersection_of_original_limits_are_required(case, damage):
    answer = torch.full((1, 2), 0.5)
    success = torch.tensor([True])
    if damage == "runtime":
        case.robot.get_qpos_limits.return_value[:] = torch.tensor([-0.4, 0.4])
    elif damage == "solver":
        case.solver.upper_qpos_limits.fill_(0.4)
    elif damage == "nonfinite":
        answer[0, 0] = float("nan")
    elif damage == "false":
        success[0] = False
    else:
        answer = torch.zeros(1, 3)
    case.robot.compute_ik.side_effect = lambda **kwargs: (success, answer)
    result = select(case)
    assert_unchanged(result, case)
    assert result[3]["accepted"] is False and case.robot.compute_ik.call_count == 1
    json.dumps(result[3], allow_nan=False)


def test_rejected_native_paths_do_not_spend_ik_queries(case):
    case.screen.exchange_candidates.return_value = (
        [case.proposal(mask=torch.zeros_like(case.mask))],
        {},
    )
    result = select(case)
    assert_unchanged(result, case)
    assert result[3]["ik_queries"] == 0
    case.robot.compute_ik.assert_not_called()


@pytest.mark.parametrize(
    "damage", ["too_many", "mask", "translation", "limits", "screen", "ik"]
)
def test_unavailable_qualification_retains_failed_original_and_restores_rng(
    case, damage
):
    if damage == "too_many":
        case.screen.exchange_candidates.return_value = (
            [case.proposal() for _ in range(4)],
            {},
        )
    elif damage == "mask":
        case.screen.exchange_candidates.return_value[0][0]["mask"] = torch.ones(
            1, 2, dtype=torch.bool
        )
    elif damage == "translation":
        case.screen.exchange_candidates.return_value[0][0]["translation"].fill_(
            float("nan")
        )
    elif damage == "limits":
        case.solver.lower_qpos_limits.fill_(float("nan"))
    elif damage == "screen":
        case.screen.exchange_candidates.side_effect = ValueError(
            "native shape unavailable"
        )
    else:

        def broken(**kwargs):
            torch.rand(1)
            raise RuntimeError("IK API unavailable")

        case.robot.compute_ik.side_effect = broken
    rng = rng_snapshot()
    result = select(case)
    assert_unchanged(result, case)
    assert result[3]["reason"] == "exchange_qualification_unavailable"
    assert_rng_equal(rng)


def test_sampler_enabled_scope_remains_unmodified(case):
    case.context.affordance_sampling = object()
    result = select(case)
    assert_unchanged(result, case)
    assert result[3]["reason"] == "unsupported_exchange_scope"
    case.screen.exchange_candidates.assert_not_called()


def test_original_finite_sampler_eligibility_cannot_be_restored_by_proposal(case):
    case.pool.costs[0, 1] = float("nan")
    result = select(case)
    assert_unchanged(result, case)
    eligible = case.screen.exchange_candidates.call_args.args[2]
    assert eligible.tolist() == [[True, False, True]]
    assert result[3]["reason"] == "exchange_qualification_unavailable"
    case.robot.compute_ik.assert_not_called()
