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

"""GenSim-only E6 recovery scope, immutable requests and fail-closed gates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace as NS
import random

import numpy as np
import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    articulation_recovery as recovery,
)
from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
    bind_articulation_calls,
    GenSimActionEngine,
)
from embodichain.lab.sim.atomic_actions import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanState, PlanResult


@dataclass
class Diagnostics:
    metadata: dict = field(default_factory=dict)
    failure: str = "original_failure"


@dataclass
class Plan:
    plan_success: torch.Tensor = field(default_factory=lambda: torch.tensor([False]))
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    joint_trajectory: object = None
    segments: tuple = (NS(name="pull"),)


def test_binding_is_by_exact_registered_call_not_skill_or_object():
    segments = [
        NS(
            name=f"s{i}",
            segment_id=f"segment-{i}",
            calls=(NS(segment_call_index=0, call=NS(call_id=call)),),
        )
        for i, call in enumerate(
            (
                "gen_sim.articulation_slide",
                "gen_sim.articulation_withdraw",
                "gen_sim.articulation_park",
                "ordinary.move_end_effector",
                "gen_sim.articulation_slide",
            )
        )
    ]
    program = NS(program_id="p", iter_segments=lambda: iter(segments))
    assert bind_articulation_calls(program) == (
        ("p/segment-0:0", "slide"),
        ("p/segment-1:0", "withdraw"),
        ("p/segment-4:0", "slide"),
    )


@pytest.mark.parametrize(
    "original,expected", [(140, 210), (260, 390), (300, 390), (390, 390), (400, None)]
)
def test_frame_budget_is_bounded_without_shortening_caller(original, expected):
    assert recovery.slide_budget(original) == expected


def test_scope_is_generator_local_nested_and_exception_safe():
    first, second = object(), object()
    assert recovery.motion_scope(first) is None
    with recovery.planning_scope(first, "slide") as outer:
        assert recovery.motion_scope(first) is outer
        assert recovery.motion_scope(second) is None
        with pytest.raises(RuntimeError):
            with recovery.planning_scope(second, "withdraw") as inner:
                assert recovery.motion_scope(first) is None
                assert recovery.motion_scope(second) is inner
                raise RuntimeError("injected")
        assert recovery.motion_scope(first) is outer
    assert recovery.motion_scope(first) is None


@pytest.mark.parametrize(
    "call,enabled,batch,strategy,collision,expected",
    [
        ("bound", True, 1, "ik_interp", "off", True),
        ("ordinary", True, 1, "ik_interp", "off", False),
        ("bound", False, 1, "ik_interp", "off", False),
        ("bound", True, 2, "ik_interp", "off", False),
        ("bound", True, 1, "motion_gen", "off", False),
        ("bound", True, 1, "ik_interp", "revalidate", False),
    ],
)
def test_engine_gates_recovery_before_touching_other_planning(
    monkeypatch, call, enabled, batch, strategy, collision, expected
):
    generator = object()
    engine = object.__new__(GenSimActionEngine)
    engine._planning_services = NS(motion_generator=generator)
    engine._articulation_calls = {"bound": "withdraw"}
    engine._cartesian_calls = {call} if enabled else set()
    seen = []

    transform = lambda request, context, plan: plan

    def original(self, request, context=None, *, plan_transform=None):
        assert plan_transform is transform
        seen.append(recovery.motion_scope(generator) is not None)
        return Plan(plan_success=torch.tensor([True]))

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", original)
    request = NS(
        invocation_id=call,
        motion_policy=NS(
            strategy=strategy, dynamic_collision_mode=collision, sample_count=260
        ),
    )
    result = engine._plan_request(
        request, NS(batch_size=batch), plan_transform=transform
    )
    assert seen == [expected]
    assert result.plan_success.tolist() == [True]
    assert recovery.motion_scope(generator) is None


def test_successful_plan_is_not_rebuilt_or_searched():
    original = Plan(plan_success=torch.tensor([True]))
    scope = recovery.PlanningScope(object(), "slide")
    assert (
        recovery.recover(
            scope,
            object(),
            object(),
            original,
            lambda _: pytest.fail("unexpected rebuild"),
        )
        is original
    )


def test_capture_owns_targets_and_start_snapshot():
    pose = torch.eye(4)[None]
    options = MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 2),
        sample_count=2,
        interpolation_dt=0.04,
    )
    scope = recovery.PlanningScope(object(), "withdraw")
    scope.record([PlanState(move_type=MoveType.EEF_MOVE, xpos=pose)], options)
    pose[:, 0, 3] = 10
    options.start_qpos[:] = 10
    assert scope.calls[0].targets[0].xpos[0, 0, 3] == 0
    assert scope.calls[0].options.start_qpos.tolist() == [[0, 0]]


def test_unsupported_motion_capture_does_not_break_a_normal_plan():
    scope = recovery.PlanningScope(object(), "slide")
    scope.record(
        [PlanState(move_type=MoveType.JOINT_MOVE, qpos=torch.zeros(1, 2))], None
    )
    original = Plan(plan_success=torch.tensor([True]))
    assert not scope.compatible
    assert (
        recovery.recover(scope, object(), object(), original, lambda _: pytest.fail())
        is original
    )


@pytest.mark.parametrize("change", ["start", "target", "count", "dt", "part", "fk"])
def test_replay_refuses_changed_goals_timing_or_invalid_fk(change):
    pose = torch.eye(4)[None]
    generator = NS(robot=NS(compute_fk=lambda **kw: pose))
    scope = recovery.PlanningScope(generator, "withdraw")
    options = MotionGenOptions(
        strategy="ik_interp",
        control_part="arm",
        start_qpos=torch.zeros(1, 2),
        sample_count=2,
        interpolation_dt=0.04,
    )
    targets = [PlanState(move_type=MoveType.EEF_MOVE, xpos=pose)]
    scope.record(targets, options)
    scope.tracks = (torch.zeros(2, 2),)
    scope.search = NS(
        to_root=lambda p: p, pose_valid=lambda q, p: torch.tensor([change != "fk"])
    )
    changes = {
        "start": {"start_qpos": torch.ones(1, 2)},
        "count": {"sample_count": 3},
        "dt": {"interpolation_dt": 0.08},
        "part": {"control_part": "different"},
    }
    if change in changes:
        options = replace(options, **changes[change])
    if change == "target":
        moved = pose.clone()
        moved[:, 0, 3] = 1
        targets = [replace(targets[0], xpos=moved)]
    with pytest.raises(recovery._ReplayMismatch):
        scope.replay(targets, options)


@pytest.mark.parametrize(
    "failure",
    [None, "velocity", "collision", "missing_dependency", "mismatch", "budget"],
)
def test_only_fully_valid_recovery_is_accepted_and_rng_is_restored(
    monkeypatch, failure
):
    solver = NS(device=torch.device("cpu"), _seed_sampler=None)
    monkeypatch.setattr(recovery, "PytorchSolver", type(solver))
    robot = NS(get_solver=lambda _: solver, get_qpos=lambda: torch.zeros(1, 2))
    generator = NS(robot=robot, planner=NS(cfg=NS(sim_instance_id=7)))
    scope = recovery.PlanningScope(generator, "slide")
    options = NS(control_part="arm", start_qpos=torch.zeros(1, 2))
    scope.calls = [recovery.MotionCall([], options) for _ in range(4)]
    request = NS(
        skill_options=NS(release_retreat_distance=0.04),
        motion_policy=MotionPolicy(sample_count=260),
    )
    original = Plan()
    chosen = Plan(
        plan_success=torch.tensor([True]),
        diagnostics=Diagnostics(failure=None),
        joint_trajectory=NS(
            positions=torch.zeros(1, 3, 2), dt=torch.tensor([[0.0, 0.04, 0.04]])
        ),
    )
    if failure == "velocity":
        chosen.joint_trajectory.positions[0, 1, 0] = 1
    if failure == "budget":
        chosen.joint_trajectory.positions = torch.zeros(1, 391, 2)
        chosen.joint_trajectory.dt = torch.full((1, 391), 0.04)

    def search(*args):
        torch.rand(3)
        return NS()

    monkeypatch.setattr(recovery, "_Search", search)
    monkeypatch.setattr(
        recovery,
        "_tracks",
        lambda *a: iter(((request, (torch.zeros(3, 2),) * 4, {"frames": 390}),)),
    )
    monkeypatch.setattr(recovery, "_joint_velocity_limits", lambda *a: torch.ones(1, 2))
    monkeypatch.setattr(recovery.SimulationManager, "get_instance", lambda _: object())

    def collision(*args):
        if failure == "missing_dependency":
            raise ImportError("optional FCL missing")
        return {"valid": failure != "collision"}

    monkeypatch.setattr(recovery, "check_free_motion", collision)

    def build(value):
        assert value is request
        torch.rand(5)
        random.random()
        np.random.rand(4)
        scope.cursor = 4
        if failure == "mismatch":
            raise recovery._ReplayMismatch("changed target")
        return chosen

    rng = torch.get_rng_state().clone()
    py_rng, np_rng = random.getstate(), np.random.get_state()
    result = recovery.recover(
        scope,
        request,
        NS(batch_size=1, require_control_dt=lambda: 0.04),
        original,
        build,
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == py_rng
    assert np.array_equal(np.random.get_state()[1], np_rng[1])
    assert np.random.get_state()[2:] == np_rng[2:]
    assert scope.tracks is None and scope.search is None
    assert result.plan_success.tolist() == [failure is None]
    assert original.plan_success.tolist() == [False]
    assert result.diagnostics.metadata["gen_sim_articulation_recovery"]["status"] == (
        "accepted" if failure is None else "exhausted"
    )
    if failure is not None:
        assert result.diagnostics.failure == "original_failure"


def test_local_path_filters_fast_candidates_without_changing_solver_defaults():
    search = object.__new__(recovery._Search)
    search.lower = torch.tensor([-2.0, -2.0])
    search.upper = -search.lower
    search.limits = torch.ones(2)
    search.dt = 0.04
    search.solver = NS(
        get_ik=lambda pose, qpos_seed: (
            torch.tensor([True]),
            torch.tensor([[[0.8, 0.8]]]),
        )
    )
    search.solve_seeds = lambda pose, seeds: (
        torch.tensor([True, True]),
        torch.tensor([[0.01, 0.01], [0.8, 0.8]]),
    )
    search.pose_valid = lambda q, poses: torch.ones(len(q), dtype=torch.bool)
    start = torch.zeros(1, 2)
    result = search.path(torch.eye(4)[None], start)
    torch.testing.assert_close(result, torch.tensor([[0.0, 0.0], [0.01, 0.01]]))
    assert torch.equal(start, torch.zeros_like(start))


@pytest.mark.parametrize("obstacle_x,accepted", [(0.5, False), (5.0, True)])
def test_collision_screen_uses_physical_meshes_and_arbitrary_link_names(
    tmp_path, obstacle_x, accepted
):
    pytest.importorskip("fcl")
    pytest.importorskip("yourdfpy")
    from embodichain.gen_sim.task_engine._task_program.articulation_collision import (
        check_free_motion,
    )

    urdf = tmp_path / "arm.urdf"
    urdf.write_text(
        """<robot name="fixture"><link name="world"/><link name="chain_root"/>
<joint name="mount" type="fixed"><parent link="world"/><child link="chain_root"/><origin xyz="0.5 0 0"/></joint>
<link name="tool"><collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
<joint name="spin" type="revolute"><parent link="chain_root"/><child link="tool"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>"""
    )
    eye = torch.eye(4)[None]
    tool = eye.clone()
    tool[:, 0, 3] = 0.5
    obstacle = eye.clone()
    obstacle[:, 0, 3] = obstacle_x
    robot = NS(
        cfg=NS(fpath=str(urdf)),
        joint_names=["spin"],
        get_qpos=lambda: torch.zeros(1, 1),
        get_solver=lambda _: NS(root_link_name="chain_root", end_link_name="tool"),
        get_local_pose=lambda **kw: eye,
        get_link_pose=lambda *a, **kw: tool,
    )
    shape = NS(
        vertices=None,
        triangles=None,
        half_extents=torch.ones(3) * 0.1,
        local_pose=torch.eye(4),
    )
    obj = NS(get_local_pose=lambda **kw: obstacle, get_collision_shapes=lambda: [shape])
    sim = NS(
        get_rigid_object_uid_list=lambda: ["obstacle"],
        get_rigid_object=lambda _: obj,
        get_articulation_uid_list=lambda: [],
    )
    result = check_free_motion(
        robot, sim, torch.tensor([[0.0], [0.1]]), "arbitrary_control_part"
    )
    assert result["valid"] is accepted
    if not accepted:
        assert result["kind"] == "scene" and result["frame"] == 0
