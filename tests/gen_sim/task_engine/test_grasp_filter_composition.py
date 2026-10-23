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

"""E6 handle sampling must not hide an unrelated constrained rigid Pick."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program import grasp_filter as filters
from embodichain.gen_sim.task_engine._task_program.actions import _PICK_GRASP_RULE
from embodichain.gen_sim.task_engine._task_program.services import (
    _PickLowerer,
    _PickRoute,
)
from embodichain.lab.sim.atomic_actions import ObjectSemantics, PickUpOptions
from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.toolkits.graspkit.pose_generator import ParallelJawGripperModelCfg


def _case():
    vertices = torch.tensor([[-0.01, 0.0, -0.02], [0.01, 0.0, 0.02], [0.0, 0.01, 0.0]])
    triangles = torch.tensor([[0, 1, 2]])
    rule = filters.GraspRule(
        "object",
        "target",
        filters.geometry_key(vertices, triangles),
        (0.0, 0.0, 1.0),
        0.0,
        True,
        None,
        None,
        0.0,
    )
    delegate = NS(gripper_model=ParallelJawGripperModelCfg())
    task = filters.TaskGraspPoseGenerator(delegate, (rule,))
    wrapped = filters.E6ApproachGraspPoseGenerator(task, frozenset({"handle_geometry"}))
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(
            mesh_vertices=vertices, mesh_triangles=triangles
        ),
        geometry={},
        entity_id="object",
    )
    return task, wrapped, semantics


@pytest.mark.parametrize("wrapped", [False, True])
def test_constrained_pick_lowering_after_e6_keeps_the_immutable_rule(
    wrapped: bool,
) -> None:
    task, e6, semantics = _case()
    lowerer = _PickLowerer(
        (_PickRoute("object", "target", grasp_region="upper_half"),),
        (semantics,),
        {"hand": e6 if wrapped else task},
    )
    bound = NS(
        binding=NS(
            resources={
                "primary": NS(
                    endpoints={
                        "grasp": NS(runtime_target=NS(target_id="hand")),
                    }
                )
            }
        )
    )
    result = lowerer.lower(
        NS(arguments={"object": "object", "target": "target"}),
        context=None,
        bound=bound,
        option_template=PickUpOptions(),
    )
    assert result.goal.semantics.affordance.get_custom_config(_PICK_GRASP_RULE) == (
        "object",
        "target",
    )
    assert semantics.affordance.get_custom_config(_PICK_GRASP_RULE) is None
    assert task._active_rules == ()
    assert e6._delegate is task


@pytest.mark.parametrize("wrapped", [False, True])
def test_scoped_filter_still_rejects_wrong_region_and_binding(wrapped: bool) -> None:
    task, e6, semantics = _case()
    mesh = semantics.affordance
    resolved = filters.resolve_pick_filter(
        e6 if wrapped else task, mesh.mesh_vertices, mesh.mesh_triangles
    )
    assert resolved is task
    scoped = resolved.for_pick(
        "object", "target", mesh.mesh_vertices, mesh.mesh_triangles
    )
    poses = torch.eye(4)[None].repeat(2, 1, 1)
    poses[:, 2, 3] = torch.tensor([0.01, -0.01])
    assert scoped.accepts_grasp_poses(
        poses,
        torch.eye(4)[None].repeat(2, 1, 1),
        mesh.mesh_vertices,
        mesh.mesh_triangles,
    ).tolist() == [True, False]
    with pytest.raises(ValueError, match="no matching immutable"):
        resolved.for_pick(
            "object", "wrong_target", mesh.mesh_vertices, mesh.mesh_triangles
        )
    with pytest.raises(ValueError, match="no matching immutable"):
        resolved.for_pick(
            "object", "target", mesh.mesh_vertices * 2, mesh.mesh_triangles
        )
    assert task._active_rules == ()


def test_resolver_does_not_unwrap_handles_or_unknown_providers() -> None:
    task, _, semantics = _case()
    mesh = semantics.affordance
    vertices, triangles = mesh.mesh_vertices, mesh.mesh_triangles
    handle = vertices * 2
    wrapped = filters.E6ApproachGraspPoseGenerator(
        task, frozenset({filters.geometry_key(handle, triangles)})
    )
    assert filters.resolve_pick_filter(wrapped, vertices, triangles) is task
    assert filters.resolve_pick_filter(wrapped, handle, triangles) is None
    assert filters.resolve_pick_filter(NS(_delegate=task), vertices, triangles) is None
    direction = torch.tensor([1.0, 0.0, 0.0])
    assert torch.equal(
        wrapped._approach_direction(vertices, triangles, direction), direction
    )
    diagonal = wrapped._approach_direction(handle, triangles, direction)
    torch.testing.assert_close(diagonal, torch.tensor([2**-0.5, 0.0, -(2**-0.5)]))
    assert wrapped._delegate is task


def test_release_clearance_remains_active_through_the_e6_wrapper() -> None:
    task, _, semantics = _case()
    mesh = semantics.affordance
    release = torch.eye(4)
    release[2, 3] = 0.2
    rule = replace(
        task._rules[0],
        upper_half=False,
        release_pose=tuple(release.flatten().tolist()),
        plane_z=0.1,
    )
    task = filters.TaskGraspPoseGenerator(task._delegate, (rule,))
    wrapped = filters.E6ApproachGraspPoseGenerator(task, frozenset({"handle_geometry"}))
    resolved = filters.resolve_pick_filter(
        wrapped, mesh.mesh_vertices, mesh.mesh_triangles
    )
    scoped = resolved.for_pick(
        "object", "target", mesh.mesh_vertices, mesh.mesh_triangles
    )
    poses = torch.eye(4)[None].repeat(2, 1, 1)
    poses[1, 2, 3] = -0.1
    assert scoped.accepts_grasp_poses(
        poses,
        torch.eye(4)[None].repeat(2, 1, 1),
        mesh.mesh_vertices,
        mesh.mesh_triangles,
    ).tolist() == [True, False]


@pytest.mark.parametrize("empty,vertical", [(True, True), (False, True), (True, False)])
def test_direction_fallback_preserves_nonempty_pools_and_explicit_directions(
    monkeypatch,
    empty: bool,
    vertical: bool,
) -> None:
    from embodichain.gen_sim.task_engine._task_program import actions
    from embodichain.lab.sim.atomic_actions.affordance_sampling import (
        AffordancePoseCandidates,
    )
    from embodichain.lab.sim.atomic_actions.policies import MotionPolicy

    task, wrapped, semantics = _case()
    semantics.affordance.set_custom_config(_PICK_GRASP_RULE, ("object", "target"))
    options = PickUpOptions()
    request = NS(
        goal=NS(semantics=semantics, grasp_xpos=None),
        skill_options=options,
        motion_policy=MotionPolicy(strategy="ik_interp", dynamic_collision_mode="off"),
    )
    selection = actions._PickSelection(request=request)
    monkeypatch.setattr(
        AntipodalAffordance,
        "get_grasp_candidates",
        lambda *a, **kw: AffordancePoseCandidates(
            poses=torch.eye(4)[None, None],
            costs=torch.full((1, 1), torch.inf if empty else 0.0),
            valid=torch.full((1, 1), not empty, dtype=torch.bool),
        ),
    )
    pose = torch.eye(4)[None]
    pose[:, 0, 3] = 0.3
    returned = object()
    ordinary = object()
    monkeypatch.setattr(
        AntipodalAffordance, "sample_candidates", lambda *a, **kw: ordinary
    )

    def choose(
        affordance,
        generator,
        object_pose,
        start,
        arm,
        opts,
        context,
        key,
        choices,
        current,
    ):
        assert empty and vertical
        assert generator._active_rules == task._rules
        assert current.constrained and current is selection
        assert len(choices) == 1 and choices[0].axis is None
        assert float(choices[0].direction[0, 2]) == -0.5
        torch.rand(4)
        return returned

    robot = NS(device="cpu", compute_fk=lambda **kwargs: torch.eye(4)[None])
    action = NS(
        device="cpu",
        robot=robot,
        motion_generator=NS(),
        planning_services=NS(grasp_pose_generator=lambda _: wrapped),
        _select_adaptive_pick=choose,
        _select_feasible_grasp_variants=lambda poses, *a: (
            poses,
            torch.ones(poses.shape[:2], dtype=torch.bool),
        ),
    )
    before = torch.random.get_rng_state()
    token = actions._PICK_SELECTION.set(selection)
    try:
        result = actions.GenSimPickUp._resolve_grasp_pose(
            action,
            semantics,
            pose,
            torch.zeros(1, 7),
            NS(control_part="arm"),
            "hand",
            options,
            torch.tensor([0.0, 0.0, -1.0] if vertical else [1.0, 0.0, 0.0]),
            NS(batch_size=1, affordance_sampling=None, env_ids=torch.tensor([0])),
            sample_key="grasp",
        )
    finally:
        actions._PICK_SELECTION.reset(token)
    assert result is (returned if empty and vertical else ordinary)
    assert torch.equal(before, torch.random.get_rng_state())
    assert task._active_rules == ()
