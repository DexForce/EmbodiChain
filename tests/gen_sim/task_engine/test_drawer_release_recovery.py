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

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import drawer_runtime as drawer
from embodichain.lab.sim.atomic_actions import PlaceGoal, SceneEntityPose
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
from embodichain.gen_sim.task_engine._task_program.articulation_binding import (
    PrismaticBinding,
)
from embodichain.gen_sim.task_engine._task_program.drawer_binding import DrawerRoute


def test_reduced_release_keeps_outside_rim_and_payload_target_unchanged() -> None:
    observation = object.__new__(drawer.DrawerObservation)
    observation.route = NS(
        binding=NS(axis=(1.0, 0.0, 0.0), axis_sign=1),
        lower=(-0.1, -0.1, 0.0),
        upper=(0.1, 0.1, 0.1),
        upright=False,
    )
    observation.vertices = [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]
    observation.poses = lambda: (torch.eye(4)[None], torch.eye(4)[None])
    target = torch.eye(4)[None]
    target[:, 2, 3] = 0.015
    before = target.clone()
    nominal = observation._entry_poses_for_target(target)
    lower = observation._entry_poses_for_target(target, release_lift=0.0)
    assert torch.equal(nominal[0], lower[0]) and torch.equal(nominal[1], lower[1])
    assert torch.equal(lower[2], lower[1])
    assert nominal[2][0, 2, 3] - lower[2][0, 2, 3] == pytest.approx(0.12)
    assert torch.equal(target, before)


@dataclass
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics
    joint_trajectory: object = None
    scene_dependencies: tuple = ()


@pytest.mark.parametrize(
    "nominal_success,nominal_count,expected_lifts",
    [
        (True, 1, [0.12]),
        (False, 1, [0.12, 0.0]),
        (False, 2, [0.12]),
    ],
)
def test_drawer_release_fallback_preserves_nominal_and_total_full_plan_budget(
    monkeypatch,
    nominal_success,
    nominal_count,
    expected_lifts,
) -> None:
    generated, planned = [], []
    generator = NS(placement_path=lambda target: nullcontext())

    class Engine(drawer.DrawerPlacementEngine):
        def __init__(self):
            self._drawer_observations = (
                NS(route=NS(affordance="drawer", upright=False)),
            )

        @property
        def robot(self):
            return object()

        @property
        def motion_generator(self):
            return generator

    def prepare(request, context, observations, *, release_lift=0.12):
        generated.append(release_lift)
        return tuple(
            NS(goal=NS(xpos=torch.full((1, 3, 4, 4), release_lift)))
            for _ in range(nominal_count)
        )

    def prefilter(request, context, candidates, robot):
        return tuple(enumerate(candidates)), []

    def build(engine, request, context, **kwargs):
        lift = float(request.goal.xpos[0, 0, 0, 0])
        planned.append(lift)
        return Plan(
            torch.tensor([nominal_success or lift == 0.0]),
            PlannerDiagnostics(backend="test"),
        )

    monkeypatch.setattr(drawer, "prepare_place_candidates", prepare)
    monkeypatch.setattr(drawer, "_prefilter_place_candidates", prefilter)
    monkeypatch.setattr(
        drawer, "_validate_drawer_plan_velocity", lambda plan, robot: plan
    )
    monkeypatch.setattr(drawer.GenSimActionEngine, "_plan_request", build)
    request = NS(goal=PlaceGoal(SceneEntityPose("drawer")))
    result = Engine()._plan_request(request, NS(batch_size=1))
    assert generated == expected_lifts
    assert len(planned) <= drawer.DRAWER_MAX_FULL_PLAN_CANDIDATES
    assert result.plan_success.tolist() == [nominal_success or 0.0 in expected_lifts]
    assert isinstance(request.goal.xpos, SceneEntityPose)


def test_explicit_upright_release_descends_to_support_after_crossing_the_rim() -> None:
    observation = object.__new__(drawer.DrawerObservation)
    observation.route = NS(
        binding=NS(axis=(1.0, 0.0, 0.0), axis_sign=1),
        lower=(-0.1, -0.1, 0.0),
        upper=(0.1, 0.1, 0.1),
        upright=True,
    )
    observation.vertices = [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]
    observation.poses = lambda: (torch.eye(4)[None], torch.eye(4)[None])
    target = torch.eye(4)[None]
    target[:, 2, 3] = 0.015
    outside, rim, release = observation._entry_poses_for_target(
        target, release_lift=0.0
    )
    assert torch.equal(release, target)
    assert rim[0, 2, 3] > release[0, 2, 3]
    assert torch.equal(outside[:, :3, :3], target[:, :3, :3])
    assert rim[0, 2, 3] - 0.01 == pytest.approx(observation.route.upper[2] + 0.025)


def test_explicit_upright_candidate_does_not_tip_an_isotropic_payload() -> None:
    import numpy as np
    from embodichain.gen_sim.task_engine._task_program.drawer_geometry import (
        placement_pose_candidates,
    )

    vertices = np.array([[-0.04, -0.04, -0.04], [0.04, 0.04, 0.04]])
    inputs = (
        vertices,
        np.eye(3),
        np.eye(4),
        np.array([-0.2, -0.2, 0.0]),
        np.array([0.2, 0.2, 0.2]),
    )
    original = placement_pose_candidates(*inputs)
    upright = placement_pose_candidates(*inputs, preserve_vertical=True)
    assert any(not np.allclose(pose[:3, 2], [0, 0, 1]) for pose in original)
    assert all(np.allclose(pose[:3, 2], [0, 0, 1]) for pose in upright)


@pytest.mark.parametrize("upright", [False, True])
def test_drawer_upright_contract_roundtrip_preserves_legacy_payload(
    upright: bool,
) -> None:
    binding = PrismaticBinding(
        "cabinet",
        "slide",
        "drawer",
        "body",
        "/drawer/handle",
        "0" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (-0.2, 0.0),
        part_id="part_one",
    )
    route = DrawerRoute(
        "inside__cabinet__object__part_one",
        "object",
        binding,
        (-0.1, -0.1, 0.0),
        (0.1, 0.1, 0.1),
        upright=upright,
    )
    payload = route.payload()
    assert ("upright" in payload) is upright
    assert DrawerRoute.decode(payload) == route
    with pytest.raises(ValueError, match="boolean"):
        DrawerRoute.decode({**payload, "upright": 1})
    with pytest.raises(ValueError, match="exactly"):
        DrawerRoute.decode({**payload, "unknown": True})


@pytest.mark.parametrize("clear", [False, True])
def test_upright_drawer_checks_the_complete_emitted_path(clear: bool) -> None:
    positions = torch.zeros(1, 5, 2)
    plan = Plan(
        torch.tensor([True]),
        PlannerDiagnostics(backend="test"),
        NS(positions=positions),
    )
    seen = []

    def check(path):
        seen.append(path)
        return {"valid": clear, "frame": 3, "pairs": [["finger", "drawer"]]}

    result = drawer._validate_drawer_articulation(plan, check)
    assert torch.equal(seen[0], positions[0])
    assert result.plan_success.tolist() == [clear]
    assert plan.plan_success.tolist() == [True]
    assert drawer._validate_drawer_articulation(plan, None) is plan


@pytest.mark.parametrize("clear_index", [None, 2])
def test_upright_endpoint_ranking_preserves_budget_and_keeps_colliding_fallback(
    clear_index,
):
    from embodichain.lab.sim.atomic_actions import JointPositionTarget

    target = JointPositionTarget("arm", (0,))
    hand = NS(
        require_target=lambda cls: JointPositionTarget("hand", (1,)),
        joint_positions=lambda *a, **kw: torch.ones(1, 1),
    )
    request = NS(
        binding=NS(
            endpoint=lambda *a: (
                hand if a[-1] == "grasp" else NS(require_target=lambda cls: target)
            )
        )
    )
    context = NS(batch_size=1, robot=NS(qpos=torch.zeros(1, 2)))
    candidates = []
    for index in range(3):
        poses = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
        poses[:, :, 0, 3] = index + 1
        candidates.append(NS(goal=NS(xpos=poses)))
    robot = NS(compute_ik=lambda **kw: (torch.tensor([True]), kw["pose"][:, :1, 3]))
    selected, records = drawer._prefilter_place_candidates(
        request,
        context,
        tuple(candidates),
        robot,
        articulation_checker=lambda positions: {
            "valid": int(positions[0, 0]) - 1 == clear_index
        },
    )
    assert [i for i, _ in selected] == ([0, 1] if clear_index is None else [2, 0])
    assert len(selected) == drawer.DRAWER_MAX_FULL_PLAN_CANDIDATES
    assert all(record["accepted"] for record in records)


def test_upright_candidates_remain_bounded_and_preserve_vertical_axis():
    observation = object.__new__(drawer.DrawerObservation)
    observation.route = NS(
        binding=NS(
            axis=(1.0, 0.0, 0.0), axis_sign=1, joint="slide", target=lambda _: 0.0
        ),
        lower=(-0.1, -0.1, 0.0),
        upper=(0.1, 0.1, 0.1),
        upright=True,
    )
    observation.vertices = [[-0.01, -0.01, -0.01], [0.01, 0.01, 0.01]]
    observation.art = NS(
        joint_names=["slide"], get_qpos=lambda: torch.tensor([[-0.16]])
    )
    observation.poses = lambda: (torch.eye(4)[None], torch.eye(4)[None])
    candidates = observation._target_poses()
    assert 1 <= len(candidates) <= 5
    for pose in candidates:
        points = torch.tensor(observation.vertices) @ pose[0, :3, :3].T + pose[0, :3, 3]
        assert points[:, 0].amin() >= -0.095 - 1e-6
        assert points[:, 0].amax() <= 0.095 + 1e-6
        torch.testing.assert_close(pose[0, :3, 2], torch.tensor([0.0, 0.0, 1.0]))


def test_transport_heading_queries_do_not_rewrite_transport_options(monkeypatch):
    from embodichain.lab.sim.atomic_actions import (
        HeldObjectPoseGoal,
        JointPositionTarget,
    )
    from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
    from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
        MoveHeldObjectOptions,
    )
    from embodichain.gen_sim.task_engine._task_program import articulation_collision

    relation = torch.eye(4)[None]
    relation[:, 0, 3] = 0.02
    held = NS(semantics=NS(entity_id="payload"), object_to_eef=relation)
    context = NS(
        batch_size=1,
        robot=NS(qpos=torch.zeros(1, 2)),
        task=NS(get_held_object=lambda _: held),
    )
    poses = [torch.eye(4)[None] for _ in range(2)]
    poses[1][:, 0, 3] = 0.3
    observation = NS(
        route=NS(
            upright=True,
            object_id="payload",
            affordance="drawer",
            binding=NS(link_id="link"),
        ),
        transport_pose=lambda: poses[0],
        entry_pose_candidates=lambda **kw: tuple((pose, pose, pose) for pose in poses),
    )
    generator = NS(
        _simulation=object(), transport=lambda *a: nullcontext(), transport_metadata={}
    )

    class Engine(drawer.DrawerPlacementEngine):
        def __init__(self):
            self._drawer_observations = (observation,)

        @property
        def robot(self):
            return object()

        @property
        def motion_generator(self):
            return generator

    @dataclass
    class Request:
        goal: object
        skill_options: object
        motion_policy: object
        binding: object

    options = MoveHeldObjectOptions()
    binding = NS(
        endpoint=lambda *a: NS(
            task_state_key="left",
            require_target=lambda _: JointPositionTarget("arm", (0,)),
        )
    )
    request = Request(
        HeldObjectPoseGoal(SceneEntityPose("drawer")), options, MotionPolicy(), binding
    )

    def rank(request, context, candidates, robot, **kwargs):
        assert all(candidate.skill_options is options for candidate in candidates)
        assert isinstance(request.goal, HeldObjectPoseGoal)
        return ((1, candidates[1]),), [{"accepted": True}]

    def build(engine, selected, context, **kwargs):
        assert selected.skill_options is options
        torch.testing.assert_close(selected.goal.object_target_pose, poses[1])
        return Plan(torch.tensor([True]), PlannerDiagnostics(backend="test"))

    monkeypatch.setattr(drawer, "_prefilter_place_candidates", rank)
    monkeypatch.setattr(
        articulation_collision, "articulation_motion_checker", lambda *a: object()
    )
    monkeypatch.setattr(drawer.GenSimActionEngine, "_plan_request", build)
    result = Engine()._plan_request(request, context)
    assert result.plan_success.tolist() == [True]
    assert result.diagnostics.metadata["drawer_transport_heading"]["candidate"] == 1


@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("side", [0, -1, 1])
def test_upright_drawer_landing_avoids_a_measured_prior_payload(occupied, side):
    import numpy as np

    observation = object.__new__(drawer.DrawerObservation)
    observation.route = NS(
        lower=(-0.15, -0.15, 0.0),
        upper=(0.15, 0.15, 0.2),
        binding=NS(axis=(1.0, 0.0, 0.0)),
    )
    vertices = np.array(
        [
            [x, y, z]
            for x in (-0.025, 0.025)
            for y in (-0.025, 0.025)
            for z in (0.0, 0.1)
        ]
    )
    observation.vertices = vertices
    observation.other_payloads = (
        NS(
            containment_report=lambda: {"accepted": occupied},
            vertices=vertices,
            poses=lambda: (torch.eye(4)[None], torch.eye(4)[None]),
        ),
    )
    target = torch.eye(4)
    target[2, 3] = 0.008
    result = observation._unoccupied_target(target, torch.eye(4), side=side)
    if occupied:
        assert torch.linalg.vector_norm(result[:2, 3]) > 0.07
        assert (result[:2, 3].abs() + 0.025 < 0.15).all()
        assert result[2, 3] == target[2, 3]
        assert torch.equal(result[:3, :3], target[:3, :3])
    else:
        assert result is target


@pytest.mark.parametrize("minimum_height", [0.0, 0.025, 0.05, 1.0])
def test_upright_release_search_defers_colliding_endpoints_without_spending_full_budget(
    monkeypatch, minimum_height
):
    from embodichain.lab.sim.atomic_actions import JointPositionTarget
    from embodichain.gen_sim.task_engine._task_program import articulation_collision

    planned = []
    generator = NS(_simulation=object(), placement_path=lambda _: nullcontext())

    class Engine(drawer.DrawerPlacementEngine):
        def __init__(self):
            self._drawer_observations = (
                NS(route=NS(affordance="drawer", upright=True)),
            )

        @property
        def robot(self):
            return object()

        @property
        def motion_generator(self):
            return generator

    def prepare(request, context, observations, *, release_lift):
        return (NS(goal=NS(xpos=torch.full((1, 3, 4, 4), release_lift))),)

    def rank(request, context, candidates, robot, **kwargs):
        height = float(candidates[0].goal.xpos[0, 0, 0, 0])
        return ((0, candidates[0]),), [
            {
                "accepted": True,
                "articulation_endpoints": [{"valid": height + 1e-6 >= minimum_height}],
            }
        ]

    def build(engine, request, context, **kwargs):
        planned.append(float(request.goal.xpos[0, 0, 0, 0]))
        return Plan(torch.tensor([True]), PlannerDiagnostics(backend="test"))

    monkeypatch.setattr(drawer, "prepare_place_candidates", prepare)
    monkeypatch.setattr(drawer, "_prefilter_place_candidates", rank)
    monkeypatch.setattr(drawer, "_validate_drawer_plan_velocity", lambda p, r: p)
    monkeypatch.setattr(drawer, "_validate_drawer_articulation", lambda p, r: p)
    monkeypatch.setattr(drawer.GenSimActionEngine, "_plan_request", build)
    monkeypatch.setattr(
        articulation_collision, "articulation_motion_checker", lambda *a: object()
    )
    request = NS(
        goal=PlaceGoal(SceneEntityPose("drawer")),
        binding=NS(
            endpoint=lambda *a: NS(
                require_target=lambda cls: JointPositionTarget("arm", (0,))
            )
        ),
    )
    result = Engine()._plan_request(request, NS(batch_size=1))
    assert planned == pytest.approx(
        [minimum_height if minimum_height <= 0.05 else 0.12]
    )
    assert result.plan_success.tolist() == [True]
