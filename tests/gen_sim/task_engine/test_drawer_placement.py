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

import numpy as np
import pytest
import trimesh
import torch

from embodichain.gen_sim.task_engine._task_program.drawer_geometry import (
    placement_region,
    placement_pose,
)


@pytest.mark.parametrize("has_drawers", [False, True])
def test_factory_selects_runtime_without_shadowing_drawer_engine(
    monkeypatch: pytest.MonkeyPatch, has_drawers: bool
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock
    from embodichain.gen_sim.task_engine._task_program.assembly import _TaskFactory
    from embodichain.gen_sim.task_engine._task_program import drawer_runtime
    from embodichain.lab.task_program.integrations.simulation.environment import (
        SimulationTaskProgramFactory,
    )
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimPickUp,
        GenSimMoveHeldObject,
        GenSimPlace,
        GenSimPour,
    )

    standard, drawer = Mock(), Mock()
    parent = Mock(return_value=standard)
    constructor = Mock(return_value=drawer)
    monkeypatch.setattr(
        SimulationTaskProgramFactory, "create_atomic_action_engine", parent
    )
    monkeypatch.setattr(drawer_runtime, "DrawerPlacementEngine", constructor)
    registration = SimpleNamespace(validate_engine=Mock())
    factory = object.__new__(_TaskFactory)
    factory._drawers = (object(),) if has_drawers else ()
    factory._pour_receivers = {}
    factory._create_motion_generator = Mock(return_value=object())
    factory._grasp_pose_generators = {}
    factory._registration = registration
    profile = SimpleNamespace(action_control_profiles=lambda: {})
    result = _TaskFactory.create_atomic_action_engine(factory, profile)
    if has_drawers:
        assert result is drawer
        constructor.assert_called_once()
        parent.assert_not_called()
        drawer.register.assert_called_once()
        assert isinstance(drawer.register.call_args.args[0], GenSimPickUp)
        assert drawer.register.call_args.kwargs == {"replace": True}
    else:
        assert result is standard
        parent.assert_called_once()
        constructor.assert_not_called()
        assert [type(c.args[0]) for c in standard.register.call_args_list] == [
            GenSimPickUp,
            GenSimMoveHeldObject,
            GenSimPlace,
            GenSimPour,
        ]


def drawer_meshes() -> list[trimesh.Trimesh]:
    return [
        trimesh.creation.box(
            extents=size, transform=trimesh.transformations.translation_matrix(center)
        )
        for size, center in (
            ((0.32, 0.42, 0.02), (0, 0, 0)),
            ((0.02, 0.42, 0.14), (-0.16, 0, 0.08)),
            ((0.02, 0.42, 0.14), (0.16, 0, 0.08)),
            ((0.30, 0.02, 0.14), (0, -0.21, 0.08)),
            ((0.30, 0.02, 0.14), (0, 0.21, 0.08)),
        )
    ]


def test_placement_uses_floor_center_without_proving_a_cavity() -> None:
    low, high = placement_region(drawer_meshes()[:-1])
    np.testing.assert_allclose(low, [-0.16, -0.21, 0.01])
    np.testing.assert_allclose(high, [0.16, 0.21, 0.15])
    obstructed = [*drawer_meshes(), trimesh.creation.box(extents=[0.08] * 3)]
    assert np.isfinite(placement_region(obstructed)).all()


def test_oversize_object_still_gets_a_place_target() -> None:
    low, high = placement_region(drawer_meshes())
    vertices = trimesh.creation.box(extents=[0.5, 0.5, 0.5]).vertices
    pose = placement_pose(vertices, np.eye(3), np.eye(4), low, high)
    np.testing.assert_allclose(pose[:2, 3], [0, 0])
    assert pose[2, 3] > high[2]


def test_live_link_pose_and_object_rotation_define_target() -> None:
    low, high = placement_region(drawer_meshes())
    vertices = trimesh.creation.box(extents=[0.08, 0.04, 0.06]).vertices
    link = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    link[:3, 3] = [0.6, -0.3, 0.8]
    target = placement_pose(vertices, np.eye(3), link, low, high)
    moved = link.copy()
    moved[0, 3] += 0.12
    next_target = placement_pose(vertices, np.eye(3), moved, low, high)
    np.testing.assert_allclose(next_target[:3, 3] - target[:3, 3], [0.12, 0, 0])
    np.testing.assert_allclose(target[:3, :3], np.eye(3))


def test_transport_stages_outside_opening_along_calibrated_slide_axis() -> None:
    from types import SimpleNamespace

    from embodichain.gen_sim.task_engine._task_program.drawer_binding import (
        DrawerRoute,
    )
    from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
        DrawerObservation,
    )
    from embodichain.gen_sim.task_engine._task_program.articulation_binding import (
        PrismaticBinding,
    )

    binding = PrismaticBinding(
        object_id="cabinet",
        joint="slide",
        link="drawer",
        parent="cabinet_root",
        handle_path="/drawer/handle",
        source_sha256="0" * 64,
        scale=1.0,
        axis=(0.0, 1.0, 0.0),
        limits=(-0.1, 0.0),
        part_id="part_test",
    )
    route = DrawerRoute(
        "inside__cabinet__cube__part_test",
        "cube",
        binding,
        (-0.1, -0.1, 0.0),
        (0.1, 0.1, 0.1),
    )
    obj_pose = torch.eye(4).unsqueeze(0)
    link_pose = torch.eye(4).unsqueeze(0)
    obj = SimpleNamespace(
        get_vertices=lambda scale=True: torch.tensor([[[0.0, 0.0, 0.0]]]),
        get_local_pose=lambda to_matrix=True: obj_pose,
    )
    art = SimpleNamespace(
        get_link_pose=lambda *args, **kwargs: link_pose,
        joint_names=["slide"],
        get_qpos=lambda: torch.tensor([[-0.1]]),
    )
    obs = object.__new__(DrawerObservation)
    obs.route, obs.art, obs.obj = route, art, obj
    obs.vertices = np.zeros((1, 3), dtype=np.float32)
    staged, rim, release = obs.entry_poses()
    assert staged[0, 0, 3] == pytest.approx(0.0)
    assert staged[0, 1, 3] == pytest.approx(-0.14)
    assert staged[0, 2, 3] == pytest.approx(0.245)
    assert rim[0, 2, 3] == staged[0, 2, 3]
    assert release[0, 1, 3] == pytest.approx(-0.05)
    assert release[0, 2, 3] == pytest.approx(0.365)
    transform = torch.eye(4)
    transform[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    transform[:3, 3] = torch.tensor([0.6, -0.3, 0.8])
    link_pose[:] = transform
    obj_pose[:] = transform
    for actual, original in zip(obs.entry_poses(), (staged, rim, release), strict=True):
        torch.testing.assert_close(actual, transform @ original)


def test_part_identity_survives_chained_step_results() -> None:
    from embodichain.gen_sim.task_engine.semantic_planner import _bound_part

    steps = {
        "open": {"id": "open", "object": {"kind": "scene_ref"}},
        "close": {"id": "close", "object": {"kind": "step_result", "step_id": "open"}},
        "place": {"id": "place", "target": {"kind": "step_result", "step_id": "close"}},
    }
    bindings = {"role_bindings": {"open.object": "part_top"}}
    assert _bound_part(steps["place"], "target", bindings, steps) == "part_top"
    assert _bound_part(steps["close"], "object", bindings, steps) == "part_top"


def test_drawer_pick_does_not_require_future_place_feasibility(monkeypatch) -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        AntipodalAffordance,
        GraspGoal,
        ObjectSemantics,
        PickUpOptions,
    )
    from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
        DrawerPlacementEngine,
    )

    mesh = trimesh.creation.box(extents=[0.1] * 3)
    semantics = ObjectSemantics(
        AntipodalAffordance(
            mesh_vertices=torch.tensor(mesh.vertices),
            mesh_triangles=torch.tensor(mesh.faces),
        ),
        {},
        "cube",
    )

    @dataclass
    class Request:
        goal: object
        skill_options: object

    request = Request(
        GraspGoal(semantics),
        PickUpOptions(downstream_object_target_poses=(torch.eye(4),)),
    )
    recorded = []

    def plan(_self, prepared, context):
        recorded.append(prepared)
        return SimpleNamespace(plan_success=torch.tensor([False]))

    monkeypatch.setattr(AtomicActionEngine, "_plan_request", plan)
    engine = object.__new__(DrawerPlacementEngine)
    engine._planning_services = SimpleNamespace(robot=object())
    engine._drawer_observations = (
        SimpleNamespace(route=SimpleNamespace(object_id="cube")),
    )
    engine.plan_request(request, object())
    assert recorded[0].skill_options.downstream_object_target_poses == ()
    assert len(request.skill_options.downstream_object_target_poses) == 1
