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

"""Lateral release recovery keeps the exact goal and projected payload frame."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import random
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import (
    place_branch as branch,
    place_transport as transport,
)
from embodichain.lab.sim.atomic_actions import JointPositionTarget, PlannerDiagnostics
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal, PlaceOptions
from embodichain.lab.task_program.semantics import (
    Pick,
    RegisteredSemanticCall,
    SceneObjectRef,
)


@pytest.mark.parametrize(
    "damage", [None, "on", "inside", "resource", "purpose", "grasp", "part"]
)
def test_only_bound_unconstrained_ordinary_lateral_pairs_get_permission(damage) -> None:
    pick = Pick(SceneObjectRef("item"), resources={"primary": "left"})
    args = {"object": "item", "reference": "anchor", "relation": "right_of"}
    resources = {"primary": "left"}
    purposes = {"pick": "ordinary"}
    if damage in {"on", "inside"}:
        args["relation"] = damage
    elif damage == "resource":
        resources["primary"] = "right"
    elif damage == "purpose":
        purposes["pick"] = "pour"
    elif damage == "grasp":
        from embodichain.lab.task_program.semantics import SceneAffordanceRef

        pick = replace(pick, grasp=SceneAffordanceRef("explicit"))
    elif damage == "part":
        args["reference_part"] = "part"
    place = RegisteredSemanticCall(
        "simulation.place_relative", args, resources=resources
    )
    route = NS(pick_id="pick", place_id="release", pick_call=pick, place_call=place)
    scene = NS(rigid_objects=[NS(entity_id=name) for name in ("item", "anchor")])
    assert branch.bind_lateral_invocations([route], purposes, scene) == (
        frozenset({"release"}) if damage is None else frozenset()
    )


@dataclass
class Plan:
    plan_success: torch.Tensor
    diagnostics: PlannerDiagnostics


@pytest.mark.parametrize("affordance", [False, True])
@pytest.mark.parametrize(
    "damage", [None, "success", "unbound", "other_obstacle", "self_collision", "pour"]
)
def test_on_recovery_only_replaces_its_bound_support_collision(
    monkeypatch, affordance, damage
) -> None:
    target = "landing_frame" if affordance else "support"
    action = NS(
        is_bound=True,
        ordinary_on_invocations=() if damage == "unbound" else {"release"},
        ordinary_lateral_invocations=(),
        motion_generator=NS(),
        support_surface_owners={"landing_frame": "support"} if affordance else {},
    )
    request = NS(
        invocation_id="release",
        goal=PlaceGoal(SceneEntityPose(target)),
        skill_options=PlaceOptions(
            preserve_current_object_orientation=True, cartesian_waypoint_count=2
        ),
        motion_policy=NS(strategy="ik_interp", dynamic_collision_mode="off"),
        binding=NS(endpoint=lambda *args: NS(task_state_key="left")),
    )
    rules = {"gen_sim.adaptive_grasp": "pour" if damage == "pour" else "ordinary"}
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        get_held_object=lambda _: NS(
            semantics=NS(affordance=NS(get_custom_config=rules.get))
        ),
    )
    acceptance = {
        "reason": (
            "robot_proxy_rejected"
            if damage == "self_collision"
            else "held_object_scene_envelope_rejected"
        ),
        "entity": "other" if damage == "other_obstacle" else "support",
        "collision": {"kind": "self"},
    }
    rejected = Plan(
        torch.tensor([damage == "success"]),
        PlannerDiagnostics(
            "test",
            metadata={"gen_sim_place_waypoint_probe": {"acceptance": acceptance}},
        ),
    )
    inspection = {"valid": True}
    candidate = Plan(
        torch.tensor([True]),
        PlannerDiagnostics(
            "test", metadata={"gen_sim_place_branch": {"inspection": inspection}}
        ),
    )
    planner = Mock(return_value=candidate)
    monkeypatch.setattr(branch, "plan", planner)
    build, original_goal = object(), request.goal
    result = branch.recover_preserved_on(action, request, context, rejected, build)
    assert request.goal is original_goal
    if damage is None:
        planner.assert_called_once_with(
            action, request, context, build, preserved_on_transport=True
        )
        assert result.plan_success.all()
    else:
        planner.assert_not_called()
        assert result is rejected


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "success",
        "unbound",
        "velocity",
        "scene",
        "pour",
        "cartesian",
        "unpreserved",
    ],
)
def test_recovery_only_replaces_a_self_colliding_free_transport(
    monkeypatch, damage
) -> None:
    rules = {"gen_sim.adaptive_grasp": "pour" if damage == "pour" else "ordinary"}
    action = NS(
        is_bound=True,
        ordinary_lateral_invocations=() if damage == "unbound" else {"release"},
        motion_generator=NS(),
    )
    request = NS(
        invocation_id="release",
        goal=PlaceGoal(SceneEntityPose("target")),
        skill_options=PlaceOptions(
            preserve_current_object_orientation=damage != "unpreserved",
            cartesian_waypoint_count=2,
        ),
        motion_policy=NS(strategy="ik_interp", dynamic_collision_mode="off"),
        binding=NS(endpoint=lambda *args: NS(task_state_key="left")),
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        get_held_object=lambda _: NS(
            semantics=NS(affordance=NS(get_custom_config=rules.get))
        ),
    )
    if damage == "cartesian":
        monkeypatch.setattr(branch, "_CARTESIAN_APPROACH", NS(get=lambda: True))
    probe = {
        "reason": (
            "velocity_rejected" if damage == "velocity" else "robot_proxy_rejected"
        ),
        "collision": {"kind": "scene" if damage == "scene" else "self"},
    }
    rejected = Plan(
        torch.tensor([damage == "success"]),
        PlannerDiagnostics(
            "test", metadata={"gen_sim_place_waypoint_probe": {"acceptance": probe}}
        ),
    )
    inspection = {"valid": True, "rotation_max_rad": 0.01, "rotation_path_rad": 0.02}
    candidate = Plan(
        torch.tensor([True]),
        PlannerDiagnostics(
            "test", metadata={"gen_sim_place_branch": {"inspection": inspection}}
        ),
    )
    planner = Mock(return_value=candidate)
    monkeypatch.setattr(branch, "plan", planner)
    goal = request.goal
    build = object()
    result = branch.recover_preserved_lateral(action, request, context, rejected, build)
    assert request.goal is goal
    if damage is None:
        planner.assert_called_once_with(
            action, request, context, build, preserved_transport=True
        )
        assert result.plan_success.tolist() == [True]
        assert (
            result.diagnostics.metadata["gen_sim_place_waypoint_recovery"] == inspection
        )
    else:
        assert result is rejected
        planner.assert_not_called()


@pytest.mark.parametrize("failure", [None, "plan", "budget"])
def test_free_transport_uses_projected_payload_and_closes_native_planner(
    monkeypatch, failure
) -> None:
    arm = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    pose = torch.eye(4)[None]
    pose[:, 0, 3] = 0.4
    held = NS(semantics=NS(entity_id="item"), object_to_eef=torch.eye(4)[None])
    held.object_to_eef[0, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    held.object_to_eef[0, :3, 3] = torch.tensor([0.1, 0.2, 0.3])
    expected = pose @ torch.linalg.inv(held.object_to_eef)
    context = NS(
        robot=NS(qpos=torch.zeros(1, 2)),
        get_held_object=lambda _: held,
        require_control_dt=lambda: 0.04,
    )
    robot = NS(uid="robot", compute_fk=lambda **kwargs: pose.clone())
    action = NS(
        robot=robot,
        recovery_scene_vertices={"item": torch.zeros(1, 4, 3)},
        motion_generator=NS(planner=NS(cfg=NS(sim_instance_id="sim"))),
    )
    request = NS(binding=NS(endpoint=lambda *args: NS(task_state_key="left")))
    scope = branch._Scope(action, request, context, preserved_transport=True)
    scope.contact = NS(qposes=torch.tensor([[[0.1, 0.2]]]))
    native = NS(get_local_pose=Mock(return_value=torch.eye(4)[None]))
    monkeypatch.setattr(
        transport,
        "SimulationManager",
        NS(get_instance=lambda _: NS(get_rigid_object=lambda _: native)),
    )
    monkeypatch.setattr(transport, "world_sources", lambda *args: {"table": object()})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kwargs: nullcontext())
    prefix = torch.tensor([[[0.0, 0.0], [0.1, 0.2]]])
    generator = NS(
        generate=Mock(
            return_value=NS(
                success=torch.tensor([failure != "plan"]),
                positions=prefix,
                dt=torch.tensor([[0.0, 0.04]]),
            )
        ),
        planner=NS(close=Mock()),
    )
    factory = Mock(return_value=generator)
    monkeypatch.setattr(transport, "_PayloadMotionGenerator", factory)
    monkeypatch.setattr(
        transport,
        "retime_to_control_grid",
        lambda positions, *args: (positions.clone(), None, None, None),
    )
    if failure == "budget":
        from embodichain.gen_sim.task_engine._task_program._search_cost import (
            PlanningBudgetExceeded,
        )

        generator.generate.side_effect = PlanningBudgetExceeded("test", 90.0, 91.0)
        with pytest.raises(PlanningBudgetExceeded):
            transport._free_prefix(scope, arm)
    elif failure == "plan":
        with pytest.raises(ValueError, match="free transport failed"):
            transport._free_prefix(scope, arm)
    else:
        assert torch.equal(transport._free_prefix(scope, arm), prefix)
    observed = factory.call_args.kwargs["observation"].obj
    torch.testing.assert_close(observed.get_local_pose(), expected)
    observed.get_local_pose().zero_()
    torch.testing.assert_close(observed.get_local_pose(), expected)
    native.get_local_pose.assert_not_called()
    generator.planner.close.assert_called_once()


@pytest.mark.parametrize("on_support", [False, True])
def test_branch_budget_exception_restores_scope_and_random_state(
    monkeypatch, on_support
) -> None:
    from embodichain.gen_sim.task_engine._task_program._search_cost import (
        PlanningBudgetExceeded,
    )

    action = NS(
        is_bound=True,
        ordinary_lateral_invocations={"release"},
        ordinary_on_invocations={"release"},
        motion_generator=NS(),
        robot=NS(device=torch.device("cpu")),
        failed_plan=Mock(),
    )
    request = NS(
        invocation_id="release",
        goal=PlaceGoal(SceneEntityPose("target")),
        skill_options=PlaceOptions(
            preserve_current_object_orientation=True, cartesian_waypoint_count=2
        ),
        motion_policy=NS(strategy="ik_interp", dynamic_collision_mode="off"),
        binding=NS(endpoint=lambda *args: NS(task_state_key="left")),
    )
    context = NS(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=NS(qpos=torch.zeros(1, 2)),
        get_held_object=lambda _: NS(
            semantics=NS(affordance=NS(get_custom_config=lambda key: None))
        ),
    )
    torch_state, python_state = torch.get_rng_state().clone(), random.getstate()

    def build(selected, observed):
        assert selected is request and observed is context
        assert branch._SCOPE.get().preserved_transport
        torch.rand(1)
        random.random()
        raise PlanningBudgetExceeded("test", 90.0, 91.0)

    with pytest.raises(PlanningBudgetExceeded):
        branch.plan(
            action,
            request,
            context,
            build,
            **(
                {"preserved_on_transport": True}
                if on_support
                else {"preserved_transport": True}
            ),
        )
    assert branch._SCOPE.get() is None
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    action.failed_plan.assert_not_called()


@pytest.mark.parametrize("accepted", [0, 2, None])
@pytest.mark.parametrize("on_support", [False, True])
def test_each_release_branch_rebuilds_and_qualifies_a_complete_plan(
    monkeypatch, accepted, on_support
) -> None:
    monkeypatch.setattr(branch, "_owns_ordinary", lambda *args, **kwargs: True)
    action = NS(
        robot=NS(device=torch.device("cpu")),
        failed_plan=Mock(
            return_value=Plan(torch.tensor([False]), PlannerDiagnostics("test"))
        ),
    )
    context = NS(robot=NS(qpos=torch.zeros(1, 2)))
    request = object()
    seen = []

    def build(selected, observed):
        assert selected is request and observed is context
        scope = branch._SCOPE.get()
        scope.release_branches = [torch.tensor([[float(i), 0.0]]) for i in range(4)]
        scope.groups = [torch.eye(4)[None, None]] * 2
        seen.append(scope.release_branch_index)
        return scope.release_branch_index

    def rebuild(scope, index):
        if index != accepted:
            raise ValueError("complete emitted contact path self-collision")
        return Plan(torch.tensor([True]), PlannerDiagnostics("test"))

    monkeypatch.setattr(branch, "_rebuild", rebuild)
    result = branch.plan(
        action,
        request,
        context,
        build,
        **(
            {"preserved_on_transport": True}
            if on_support
            else {"preserved_transport": True}
        ),
    )
    assert seen == list(range(accepted + 1 if accepted is not None else 4))
    assert result.plan_success.tolist() == [accepted is not None]
    assert branch._SCOPE.get() is None
    assert len(
        result.diagnostics.metadata["gen_sim_place_branch"]["release_branch_attempts"]
    ) == len(seen)
