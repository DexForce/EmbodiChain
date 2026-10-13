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

from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    actions,
    motion,
    pick_path,
    place_recovery,
)
from embodichain.lab.sim.atomic_actions import (
    AffordancePoseCandidates,
    AntipodalAffordance,
    JointPositionTarget,
    PlannerDiagnostics,
    StateDelta,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp, PickUpOptions
from embodichain.lab.sim.atomic_actions.primitives.place import (
    Place,
    PlaceGoal,
    PlaceOptions,
)
from embodichain.lab.sim.atomic_actions.core import ObjectSemantics
from embodichain.lab.sim.atomic_actions.state import HeldObjectState


@dataclass(frozen=True)
class Goal:
    grasp_xpos: torch.Tensor | None = None


@dataclass(frozen=True)
class Request:
    goal: object
    skill_options: object
    motion_policy: object
    binding: object
    invocation_id: str = "pick:0"


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    expected_effects: StateDelta
    diagnostics: PlannerDiagnostics
    joint_trajectory: object = None
    segments: tuple = ()
    scene_dependencies: tuple = ("bin",)
    scene_dependency_monitor_until: object = None
    scene_dependency_end_segment: object = None


def _pick(monkeypatch):
    action = actions.GenSimPickUp()
    action._planning_services = NS(
        robot=NS(device=torch.device("cpu")), device=torch.device("cpu")
    )
    request = Request(
        Goal(),
        PickUpOptions(),
        object(),
        NS(endpoint=lambda *_: NS(task_state_key="arm")),
    )
    first = Plan(torch.tensor([False]), StateDelta(), PlannerDiagnostics("test"))
    pool = torch.eye(4)[None, None].repeat(1, 6, 1, 1)
    pool[0, :, 0, 3] = torch.arange(6) * 0.1
    direction = torch.tensor([[0.0, 0.0, -1.0]])
    selection = actions._PickSelection(request=request, direction=direction)
    selection.candidate_routes = [
        (
            0,
            AffordancePoseCandidates(
                pool,
                torch.arange(6)[None].float(),
                torch.tensor([[False, True, True, True, True, True]]),
            ),
            direction,
        )
    ]
    built = []

    def build(_, selected, context):
        torch.rand(1)
        built.append(selected)
        held = HeldObjectState(
            semantics=ObjectSemantics(
                AntipodalAffordance(), geometry={}, entity_id="item"
            ),
            grasp_xpos=selected.goal.grasp_xpos.clone(),
            object_to_eef=torch.eye(4)[None],
        )
        return replace(
            first,
            plan_success=torch.tensor([True]),
            expected_effects=StateDelta(held_object_updates={"arm": held}),
        )

    monkeypatch.setattr(PickUp, "_plan", build)
    return action, request, NS(), first, selection, built


def test_pick_candidate_retry_certifies_each_complete_plan_and_preserves_rng(
    monkeypatch,
):
    action, request, context, first, selection, built = _pick(monkeypatch)
    checked = []

    def certify(_, selected, ctx, plan, sel):
        checked.append(plan)
        assert ctx is context
        assert sel.trajectory is None
        return replace(plan, plan_success=torch.tensor([len(checked) == 2]))

    monkeypatch.setattr(pick_path, "rebuild", certify)
    before = torch.get_rng_state().clone()
    result = action._retry_ordinary_pick_candidates(request, context, first, selection)
    assert len(built) == len(checked) == 2
    assert result.plan_success.all()
    torch.testing.assert_close(
        result.expected_effects.held_object_updates["arm"].grasp_xpos,
        built[1].goal.grasp_xpos,
    )
    assert all(r.motion_policy is request.motion_policy for r in built)
    assert selection.request is request
    assert torch.equal(torch.get_rng_state(), before)
    assert result.scene_dependencies == first.scene_dependencies


def test_pick_candidate_budget_rejects_all_unsafe_paths(monkeypatch):
    action, request, context, first, selection, built = _pick(monkeypatch)
    monkeypatch.setattr(pick_path, "rebuild", lambda *args: first)
    result = action._retry_ordinary_pick_candidates(request, context, first, selection)
    assert not result.plan_success.any()
    assert len(built) == 4
    assert (
        result.diagnostics.metadata["gen_sim_pick_candidate_retry"]["accepted"] is False
    )


def test_safe_pick_never_enumerates_alternatives(monkeypatch):
    data = _pick(monkeypatch)
    safe = replace(data[3], plan_success=torch.tensor([True]))
    assert (
        data[0]._retry_ordinary_pick_candidates(data[1], data[2], safe, data[4]) is safe
    )
    assert not data[5]


def test_pick_retry_prefers_distinct_wrist_orientations_before_near_duplicates(
    monkeypatch,
):
    action, request, context, first, selection, built = _pick(monkeypatch)
    pool = selection.candidate_routes[0][1]
    poses = torch.eye(4)[None, None].repeat(1, 6, 1, 1)
    for i, angle in enumerate(
        (0.0, 0.0, 0.001, torch.pi / 2, torch.pi, 3 * torch.pi / 2)
    ):
        c, s = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        poses[0, i, :2, :2] = torch.tensor([[c, -s], [s, c]])
    selection.candidate_routes = [
        (
            0,
            AffordancePoseCandidates(poses, pool.costs, pool.valid),
            selection.direction,
        )
    ]
    monkeypatch.setattr(pick_path, "rebuild", lambda *args: first)
    result = action._retry_ordinary_pick_candidates(request, context, first, selection)
    attempts = result.diagnostics.metadata["gen_sim_pick_candidate_retry"]["attempts"]
    assert [a["candidate_id"] for a in attempts] == [1, 3, 4, 5]


@pytest.mark.parametrize(
    "damage", [None, "velocity", "release", "explicit_cartesian", "z_cap"]
)
def test_place_candidates_keep_release_and_must_pass_certificate(monkeypatch, damage):
    action = actions.GenSimPlace()
    target = torch.eye(4)[None]
    target[:, 0, 3] = 0.2
    start = torch.eye(4)[None]
    start[:, 2, 3] = 0.4
    arm = JointPositionTarget("arm", (0,))
    robot = NS(
        device=torch.device("cpu"),
        compute_fk=lambda **kw: (start if not built else target).clone(),
        get_solver=lambda _: NS(
            get_iteration_params=lambda: {"pos_eps": 0.005, "rot_eps": 0.0005}
        ),
    )
    action._planning_services = NS(robot=robot, device=torch.device("cpu"))
    action.recovery_scene_vertices = {"table": torch.zeros(1, 3)}
    endpoint = NS(task_state_key="arm", require_target=lambda _: arm)
    options = PlaceOptions(max_approach_retract_z=0.5 if damage == "z_cap" else None)
    request = Request(
        PlaceGoal(target),
        options,
        NS(strategy="ik_interp", dynamic_collision_mode="off"),
        NS(endpoint=lambda *_: endpoint),
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=torch.zeros(1, 1)),
        get_held_object=lambda _: NS(semantics=NS(affordance=AntipodalAffordance())),
    )
    first = Plan(
        torch.tensor([False]),
        StateDelta(held_object_updates={"arm": None}),
        PlannerDiagnostics("test"),
    )
    built = []

    def build(_, selected, ctx):
        torch.rand(1)
        built.append(selected)
        return replace(
            first,
            plan_success=torch.tensor([True]),
            segments=(NS(name="release", start=1),),
            joint_trajectory=NS(positions=torch.zeros(1, 3, 1)),
        )

    monkeypatch.setattr(Place, "_plan", build)
    monkeypatch.setattr(actions, "_inspect_pick_motion", lambda *args: {"valid": True})
    monkeypatch.setattr(
        place_recovery, "inspect", lambda *args: {"valid": damage != "velocity"}
    )
    if damage == "release":
        robot.compute_fk = lambda **kw: start.clone()
    before = torch.get_rng_state().clone()
    token = motion._CARTESIAN_APPROACH.set(damage == "explicit_cartesian")
    try:
        result = action._retry_place_routes(request, context, first)
    finally:
        motion._CARTESIAN_APPROACH.reset(token)
    assert result.plan_success.all().item() is (damage is None)
    assert len(built) == (
        0 if damage in {"explicit_cartesian", "z_cap"} else (1 if damage is None else 2)
    )
    for selected in built:
        torch.testing.assert_close(selected.goal.xpos[:, 1:], target[:, None])
        assert selected.skill_options is request.skill_options
        assert selected.motion_policy is request.motion_policy
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("damage", [None, "jump", "limit", "nan", "pose"])
def test_seeded_resolver_enforces_step_limits_and_pose(damage):
    start = torch.zeros(1, 1)
    poses = torch.tensor([0.01, 0.02])[:, None]

    def solve(target, seed):
        q = target.clone()
        if damage == "jump":
            q += 0.5
        if damage == "limit":
            q += 2
        if damage == "nan":
            q[:] = float("nan")
        return torch.tensor([True]), q

    result = motion.resolve_seeded_ik_path(
        poses,
        start,
        solve=solve,
        lower=torch.tensor([-1.0]),
        upper=torch.tensor([1.0]),
        limits=torch.ones(1),
        dt=0.04,
        pose_valid=lambda q, p: torch.full((len(p),), damage != "pose"),
    )
    assert (result is not None) is (damage is None)
    if result is not None:
        torch.testing.assert_close(result, torch.tensor([[0.0], [0.01], [0.02]]))


def test_e6_uses_the_same_seeded_resolver(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import articulation_recovery

    search = object.__new__(articulation_recovery._Search)
    search.solver = NS(get_ik=lambda *a, **kw: None)
    search.lower, search.upper, search.limits, search.dt = (
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        torch.ones(1),
        0.04,
    )
    search.pose_valid, search.solve_seeds = Mock(), Mock()
    resolved = torch.zeros(3, 1)
    resolver = Mock(return_value=resolved)
    monkeypatch.setattr(articulation_recovery, "resolve_seeded_ik_path", resolver)
    assert search.path(torch.zeros(2, 4, 4), torch.zeros(1, 1)) is resolved
    resolver.assert_called_once()
