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

"""A nominally valid Slide must leave its paired withdrawal executable."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    articulation_recovery as recovery,
)
from embodichain.lab.sim.atomic_actions.policies import MotionPolicy


@dataclass
class Diagnostics:
    metadata: dict = field(default_factory=dict)
    failure: object = None


@dataclass
class Plan:
    plan_success: torch.Tensor
    joint_trajectory: object
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    segments: tuple = (NS(name="pull"),)


def _scope(monkeypatch):
    solver = NS(device=torch.device("cpu"), _seed_sampler=None)
    monkeypatch.setattr(recovery, "PytorchSolver", type(solver))
    robot = NS(get_solver=lambda _: solver, get_qpos=lambda: torch.zeros(1, 2))
    generator = NS(robot=robot, planner=NS(cfg=NS(sim_instance_id=7)))
    scope = recovery.PlanningScope(generator, "slide", withdrawal_samples=260)
    options = NS(control_part="arm", start_qpos=torch.zeros(1, 2))
    scope.calls = [recovery.MotionCall([], options) for _ in range(4)]
    request = NS(
        skill_options=NS(release_retreat_distance=0.04),
        motion_policy=MotionPolicy(sample_count=260),
    )
    monkeypatch.setattr(recovery, "_Search", lambda *args: NS())
    monkeypatch.setattr(
        recovery, "_joint_velocity_limits", lambda *args: torch.ones(1, 2)
    )
    monkeypatch.setattr(recovery.SimulationManager, "get_instance", lambda _: object())
    monkeypatch.setattr(recovery, "check_free_motion", lambda *args: {"valid": True})
    return scope, request, NS(batch_size=1, require_control_dt=lambda: 0.04)


def _trajectory(index):
    return NS(
        positions=torch.full((1, 3, 2), index * 0.01),
        dt=torch.tensor([[0.0, 0.04, 0.04]]),
    )


@pytest.mark.parametrize("nominal", [False, True])
@pytest.mark.parametrize("accepted", [0, 1, None])
def test_reselect_terminal_when_paired_withdrawal_is_unreachable(
    monkeypatch, nominal, accepted
):
    scope, request, context = _scope(monkeypatch)
    original = Plan(torch.tensor([nominal]), _trajectory(-1) if nominal else None)
    monkeypatch.setattr(
        recovery,
        "_tracks",
        lambda *args: iter(
            (request, (torch.full((3, 2), i * 0.01),) * 4, {"endpoint_seed_index": i})
            for i in range(2)
        ),
    )
    checked = []

    def preview(scope, request, context, search, trajectory, **kwargs):
        index = round(float(trajectory.positions[0, -1, 0]) * 100)
        checked.append(index)
        return {"valid": index == accepted, "reason": "no_continuous_ik"}

    monkeypatch.setattr(recovery, "_preview_withdrawal", preview)

    def build(selected):
        scope.cursor = 4
        return Plan(
            torch.tensor([True]),
            NS(positions=scope.tracks[0][None], dt=torch.tensor([[0.0, 0.04, 0.04]])),
        )

    result = recovery.recover(scope, request, context, original, build)
    assert checked == ([-1] if nominal else []) + ([0] if accepted == 0 else [0, 1]) + (
        [0, 1] if accepted is None else []
    )
    assert result.plan_success.tolist() == [accepted is not None]
    assert original.plan_success.tolist() == [nominal]
    if accepted is not None:
        evidence = result.diagnostics.metadata["gen_sim_articulation_recovery"]
        assert evidence["withdrawal_selection_changed"] is True
        assert evidence["withdrawal_preview"]["valid"] is True
    assert scope.tracks is None and scope.search is None


def test_valid_nominal_withdrawal_preserves_exact_trajectory(monkeypatch):
    scope, request, context = _scope(monkeypatch)
    original = Plan(torch.tensor([True]), _trajectory(0))
    monkeypatch.setattr(
        recovery, "_preview_withdrawal", lambda *args, **kwargs: {"valid": True}
    )
    monkeypatch.setattr(
        recovery, "_tracks", lambda *args: pytest.fail("must not search")
    )
    result = recovery.recover(
        scope, request, context, original, lambda _: pytest.fail("must not rebuild")
    )
    assert result.joint_trajectory is original.joint_trajectory
    assert result.plan_success.tolist() == [True]


def test_nominal_preview_error_does_not_dispatch_unqualified_terminal(monkeypatch):
    scope, request, context = _scope(monkeypatch)
    original = Plan(torch.tensor([True]), _trajectory(0))

    def fail(*args):
        raise ImportError("missing geometry")

    monkeypatch.setattr(recovery, "_Search", fail)
    result = recovery.recover(
        scope, request, context, original, lambda _: pytest.fail("must not rebuild")
    )
    assert result.plan_success.tolist() == [False]
    assert original.plan_success.tolist() == [True]
    assert result.diagnostics.failure.code == "gen_sim_e6_continuation_unqualified"


@pytest.mark.parametrize("clear", [False, True])
def test_actual_withdrawal_still_checks_fresh_scene(monkeypatch, clear):
    scope, _, context = _scope(monkeypatch)
    scope.kind = "withdraw"
    scope.calls = [recovery.MotionCall([], NS(control_part="arm", sample_count=260))]
    original = Plan(torch.tensor([True]), _trajectory(0))
    seen = []

    def check(robot, simulation, positions, part):
        seen.append(True)
        assert positions.data_ptr() == original.joint_trajectory.positions.data_ptr()
        return {"valid": clear}

    monkeypatch.setattr(recovery, "check_free_motion", check)
    scope.generator.robot.get_solver = lambda _: object()
    result = recovery.recover(
        scope, NS(), context, original, lambda _: pytest.fail("unsupported solver")
    )
    assert seen == [True]
    assert result.plan_success.tolist() == [clear]


def test_short_withdrawal_preserves_orientation_and_uses_the_rail_direction():
    from embodichain.gen_sim.task_engine._task_program.articulation_slide import (
        _short_withdrawal_targets,
        _withdrawal_targets,
    )

    current = torch.eye(4)[None]
    current[:, :3, 3] = torch.tensor([0.2, -0.3, 0.8])
    nominal = _withdrawal_targets(current, torch.tensor([[0.0, -1.0, 0.0]]))
    before = nominal.clone()
    short = _short_withdrawal_targets(nominal)
    assert short.shape == (1, 2, 4, 4)
    assert torch.equal(short[:, 0], current)
    assert torch.equal(short[:, 1, :3, :3], current[:, :3, :3])
    assert torch.allclose(
        short[:, 1, :3, 3] - current[:, :3, 3], torch.tensor([[0.0, -0.02, 0.0]])
    )
    assert torch.equal(nominal, before)


@pytest.mark.parametrize("nominal_possible", [False, True])
def test_continuation_tries_short_cleanup_only_after_nominal_ik_failure(
    monkeypatch, nominal_possible
):
    scope = recovery.PlanningScope(NS(), "slide", withdrawal_samples=10)
    current = torch.eye(4)[None]
    robot = NS(
        get_joint_ids=lambda **kw: (0, 1),
        compute_fk=lambda **kw: current,
    )
    seen = []

    def path(poses, start):
        seen.append(poses.clone())
        if len(seen) == 1 and not nominal_possible:
            return None
        return torch.zeros(len(poses), 2)

    search = NS(robot=robot, part="arm", dt=0.04, to_root=lambda p: p, path=path)
    request = NS(
        goal=NS(
            target_pose=current,
            semantics=NS(affordance=NS(translation_axis=torch.tensor([1.0, 0.0, 0.0]))),
        )
    )
    monkeypatch.setattr(recovery, "resolve_pose_goal", lambda *a, **kw: current)
    monkeypatch.setattr(recovery, "_joint_velocity_limits", lambda *a: torch.ones(1, 2))
    monkeypatch.setattr(recovery, "check_robot_motion", lambda *a: {"valid": True})
    result = recovery._preview_withdrawal(
        scope, request, NS(), search, _trajectory(0), allow_short=True
    )
    assert result["valid"] is True
    assert result["route"] == ("nominal" if nominal_possible else "outward_20mm")
    assert len(seen) == (1 if nominal_possible else 2)
    if not nominal_possible:
        assert seen[-1][-1, 0, 3] == pytest.approx(-0.02)
        assert seen[-1][-1, 2, 3] == pytest.approx(0.0)


@pytest.mark.parametrize("clear", [False, True])
def test_short_actual_withdrawal_rebuilds_and_checks_the_fresh_scene(
    monkeypatch, clear
):
    from embodichain.lab.sim.atomic_actions import EndEffectorPoseGoal
    from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
    from embodichain.gen_sim.task_engine._task_program.articulation_slide import (
        _withdrawal_targets,
    )

    scope, _, context = _scope(monkeypatch)
    scope.kind = "withdraw"
    scope.generator.robot.get_solver = lambda _: object()
    nominal = _withdrawal_targets(torch.eye(4)[None], torch.tensor([[1.0, 0.0, 0.0]]))

    @dataclass
    class Request:
        goal: EndEffectorPoseGoal
        motion_policy: MotionPolicy

    request = Request(EndEffectorPoseGoal(nominal), MotionPolicy(sample_count=260))
    attempts, checks = [], []

    def build(selected):
        attempts.append(selected)
        active = recovery.motion_scope(scope.generator)
        active.calls = [
            recovery.MotionCall([], NS(control_part="arm", sample_count=260))
        ]
        return Plan(
            torch.tensor([True]), _trajectory(0), PlannerDiagnostics(backend="test")
        )

    def collision(*args):
        checks.append(True)
        return {"valid": clear}

    monkeypatch.setattr(recovery, "check_free_motion", collision)
    result = recovery.retry_short_withdrawal(
        scope,
        request,
        context,
        Plan(
            torch.tensor([False]),
            None,
            PlannerDiagnostics(backend="test", metadata={"nominal": "failed"}),
        ),
        build,
    )
    assert checks == [True] and len(attempts) == 1
    assert result.plan_success.tolist() == [clear]
    assert attempts[0].motion_policy == request.motion_policy
    assert attempts[0].goal.xpos[0, -1, 0, 3] == pytest.approx(0.02)
    assert torch.equal(request.goal.xpos, nominal)


def test_nominal_terminal_branches_win_before_a_shorter_cleanup(monkeypatch):
    scope, request, context = _scope(monkeypatch)

    def tracks(*args):
        for index in (0, 1):
            yield request, (torch.tensor([index]),), {"endpoint_seed_index": index}

    def build(selected):
        index = int(scope.tracks[0][0])
        scope.cursor = 1
        return Plan(torch.tensor([True]), _trajectory(index))

    def preview(scope, request, context, search, trajectory, *, allow_short=True):
        index = round(float(trajectory.positions[0, 0, 0]) * 100)
        return {
            "valid": index == 1 or allow_short,
            "route": "nominal" if index == 1 else "outward_20mm",
        }

    monkeypatch.setattr(recovery, "_tracks", tracks)
    monkeypatch.setattr(recovery, "_preview_withdrawal", preview)
    result = recovery.recover(
        scope, request, context, Plan(torch.tensor([False]), None), build
    )
    assert result.plan_success.tolist() == [True]
    assert (
        result.diagnostics.metadata["gen_sim_articulation_recovery"][
            "endpoint_seed_index"
        ]
        == 1
    )


@pytest.mark.parametrize("park_side,paired", [("right", True), ("left", False)])
def test_articulation_park_pair_requires_adjacent_same_resource(park_side, paired):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        bind_articulation_calls,
        bind_articulation_followups,
    )

    segments = []
    for index, (kind, side) in enumerate(
        (("slide", "right"), ("withdraw", "right"), ("park", park_side))
    ):
        call = NS(
            call_id=f"gen_sim.articulation_{kind}",
            resources={"primary": side},
            arguments={} if kind == "park" else {"object": "cabinet", "state": "open"},
        )
        segments.append(
            NS(
                name=kind,
                segment_id=kind,
                parallel_block=None,
                calls=(NS(call=call, segment_call_index=0),),
            )
        )
    program = NS(program_id="p", iter_segments=lambda: segments)
    assert ("p/park:0", "park") in bind_articulation_calls(program)
    pairs = bind_articulation_followups(program)
    assert ("p/slide:0", "p/withdraw:0") in pairs
    assert (("p/withdraw:0", "p/park:0") in pairs) is paired


@pytest.mark.parametrize(
    "installed,accepted,expected",
    [(True, True, True), (False, True, False), (True, False, False)],
)
def test_short_park_authority_comes_from_installed_withdrawal(
    installed, accepted, expected
):
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )

    attempt = NS(
        request=NS(invocation_id="withdraw"),
        planned_mask=torch.tensor([installed]),
        plan=NS(
            diagnostics=NS(
                metadata={
                    "gen_sim_withdrawal_target_recovery": {
                        "accepted": accepted,
                        "route": "outward_20mm",
                    }
                }
            )
        ),
    )
    engine = NS(
        _articulation_calls={"withdraw": "withdraw", "park": "park"},
        _articulation_sessions={"park": ("withdraw", NS(plan_attempts=[attempt]))},
    )
    assert (
        GenSimActionEngine._articulation_park_after_short_withdrawal(engine, "park")
        is expected
    )
    assert (
        GenSimActionEngine._articulation_park_after_short_withdrawal(engine, "unbound")
        is False
    )


@pytest.mark.parametrize(
    "owned,held,allowed",
    [(True, False, True), (False, False, False), (True, True, False)],
)
def test_existing_park_recovery_can_qualify_only_owned_empty_e6_park(
    owned, held, allowed
):
    from embodichain.gen_sim.task_engine._task_program import park_recovery
    from embodichain.lab.sim.atomic_actions import JointPositionGoal, MoveJoints
    from embodichain.lab.sim.atomic_actions.control import PARK_COMMAND

    endpoint = NS(
        target="motion",
        task_state_key="right",
        resource_id="right",
        commands={PARK_COMMAND: NS(equivalent_to=lambda other: True)},
    )
    request = NS(
        invocation_id="park",
        skill_id="move_joints",
        goal=JointPositionGoal(PARK_COMMAND),
        motion_policy=MotionPolicy(strategy="ik_interp", dynamic_collision_mode="off"),
        binding=NS(endpoints=(endpoint,), endpoint=lambda *args: endpoint),
    )
    engine = NS(
        _park_recovery_previews={},
        _actions={"move_joints": MoveJoints()},
        _articulation_park_after_short_withdrawal=lambda identity: owned,
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        task=NS(
            held_object_mask=lambda resource: torch.tensor([held]),
            coordinated_held_objects={},
        ),
    )
    assert park_recovery.eligible(engine, request, context) is allowed


@pytest.mark.parametrize(
    "failed_stage", ["backward_reach", "push", "release", "return"]
)
def test_push_recovery_records_the_first_failed_leg(failed_stage):
    scope = recovery.PlanningScope(NS(), "slide")
    pose = torch.eye(4)[None]
    options = NS(sample_count=10, start_qpos=torch.zeros(1, 2))
    scope.calls = [recovery.MotionCall([NS(xpos=pose)], options) for _ in range(5)]
    stages = iter(("backward_reach", "push", "release", "return"))

    def path(poses, start):
        return None if next(stages) == failed_stage else torch.zeros(len(poses), 2)

    search = NS(
        to_root=lambda p: p, endpoints=lambda *a: [(0, torch.zeros(1, 2))], path=path
    )
    assert (
        list(
            recovery._push_tracks(
                scope, NS(motion_policy=MotionPolicy(sample_count=260)), search
            )
        )
        == []
    )
    assert scope.path_rejections[-1]["stage"] == failed_stage


@pytest.mark.parametrize(
    "direction,accepted_distance,expected",
    [(1, 0.05, [0.05]), (1, 0.03, [0.05, 0.03]), (-1, None, [])],
)
def test_push_shorter_pregrasp_keeps_goal_and_bounds_attempts(
    monkeypatch, direction, accepted_distance, expected
):
    from embodichain.lab.sim.atomic_actions import SlideOptions

    scope = recovery.PlanningScope(
        NS(robot=NS(device=torch.device("cpu"))), "slide", withdrawal_samples=260
    )
    goal = NS(joint_target=object(), semantics=NS(affordance=None))

    @dataclass
    class Request:
        goal: object
        skill_options: SlideOptions
        motion_policy: MotionPolicy

    request = Request(
        goal, SlideOptions(approach_distance=0.1), MotionPolicy(sample_count=260)
    )
    scope.calls = [None] * 5
    monkeypatch.setattr(
        recovery.Slide,
        "_joint_displacement",
        lambda *a: (
            torch.tensor([direction]),
            torch.tensor([True]),
            torch.tensor([False]),
        ),
    )
    monkeypatch.setattr(
        recovery, "recover", lambda scope, selected, context, plan, build: plan
    )
    seen = []

    def build(selected):
        distance = selected.skill_options.approach_distance
        seen.append(distance)
        assert (
            selected.goal is request.goal
            and selected.motion_policy is request.motion_policy
        )
        return Plan(torch.tensor([distance == accepted_distance]), None)

    result = recovery.retry_push_pregrasp(
        scope,
        request,
        NS(robot=NS(qpos=torch.zeros(1, 2))),
        Plan(torch.tensor([False]), None),
        build,
    )
    assert seen == expected
    assert result.plan_success.tolist() == [accepted_distance is not None]
    assert request.skill_options.approach_distance == 0.1
