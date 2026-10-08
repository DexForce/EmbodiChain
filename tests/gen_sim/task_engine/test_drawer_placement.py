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
    drawer_containment_report,
    placement_region,
    placement_pose,
    placement_pose_candidates,
)


@pytest.mark.parametrize("has_drawers", [False, True])
@pytest.mark.parametrize("coordinated", [False, True])
def test_factory_selects_runtime_without_shadowing_drawer_engine(
    monkeypatch: pytest.MonkeyPatch, has_drawers: bool, coordinated: bool
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock
    from embodichain.gen_sim.task_engine._task_program.assembly import _TaskFactory
    from embodichain.gen_sim.task_engine._task_program import assembly, drawer_runtime
    from embodichain.gen_sim.task_engine._task_program.coordinated_motion import (
        GenSimCoordinatedPickment,
    )
    from embodichain.lab.task_program.semantics import RobotSkillProfile
    from embodichain.gen_sim.task_engine._task_program.actions import (
        GenSimPickUp,
        GenSimHandOver,
        GenSimMoveHeldObject,
        GenSimPlace,
        GenSimPour,
    )

    standard, drawer = Mock(), Mock()
    parent = Mock(return_value=standard)
    constructor = Mock(return_value=drawer)
    monkeypatch.setattr(assembly, "GenSimActionEngine", parent)
    monkeypatch.setattr(drawer_runtime, "DrawerPlacementEngine", constructor)
    registration = SimpleNamespace(validate_engine=Mock())
    factory = object.__new__(_TaskFactory)
    factory._drawers = (object(),) if has_drawers else ()
    factory._pour_receivers = {}
    factory._adaptive_pick = False
    factory._pick_purposes = ()
    factory._coordinated_motion = coordinated
    factory._motion_samples = (("call", 260),)
    factory._cartesian_calls = ("drawer_call",)
    factory._articulation_calls = (("drawer_call", "slide"),)
    factory._robot = object()
    factory._robot_profile_binding = SimpleNamespace(profile_id="profile")
    factory._create_motion_generator = Mock(
        return_value=SimpleNamespace(robot=factory._robot)
    )
    factory._press_routes = ()
    factory._clearance_routes = ()
    factory._alignment_continuation_routes = ()
    factory._grasp_pose_generators = {}
    factory._registration = registration
    profile = Mock(spec=RobotSkillProfile, profile_id="profile")
    profile.action_control_profiles.return_value = {}
    result = _TaskFactory.create_atomic_action_engine(factory, profile)
    if has_drawers:
        assert result is drawer
        constructor.assert_called_once()
        assert constructor.call_args.kwargs["motion_samples"] == factory._motion_samples
        assert (
            constructor.call_args.kwargs["articulation_calls"]
            == factory._articulation_calls
        )
        parent.assert_not_called()
    else:
        assert result is standard
        parent.assert_called_once()
        assert parent.call_args.kwargs["motion_samples"] == factory._motion_samples
        assert (
            parent.call_args.kwargs["articulation_calls"] == factory._articulation_calls
        )
        constructor.assert_not_called()
    assert [type(c.args[0]) for c in result.register.call_args_list] == [
        GenSimPickUp,
        GenSimHandOver,
        GenSimMoveHeldObject,
        GenSimPlace,
        GenSimPour,
    ] + ([GenSimCoordinatedPickment] if coordinated else [])
    assert all(c.kwargs == {"replace": True} for c in result.register.call_args_list)


def test_close_resource_rewrite_is_scoped_to_bound_drawer() -> None:
    from embodichain.gen_sim.task_engine._task_program.drawer_binding import (
        rewrite_drawer_close_resources,
    )

    def e6_nodes(object_id: str, part: str) -> list[dict]:
        arguments = {"object": object_id, "state": "closed", "part": part}
        return [
            {
                "id": f"{object_id}_{index}",
                "task_type": "E6",
                "call": {
                    "kind": "registered",
                    "call_id": call_id,
                    "arguments": arguments,
                    "resources": {"primary": "left"},
                },
            }
            for index, call_id in enumerate(
                (
                    "gen_sim.articulation_slide",
                    "gen_sim.articulation_withdraw",
                    "gen_sim.articulation_park",
                )
            )
        ]

    drawer_nodes = e6_nodes("cabinet", "part_test")
    ordinary_nodes = e6_nodes("other_articulation", "part_other")
    graph = {
        "nodes": [
            {
                "id": "ordinary_place",
                "task_type": "E1",
                "call": {"kind": "place", "object": "cube"},
            },
            *drawer_nodes,
            *ordinary_nodes,
        ],
        "task_groups": [
            {"task_type": "E6", "node_ids": [node["id"] for node in drawer_nodes]},
            {
                "task_type": "E6",
                "node_ids": [node["id"] for node in ordinary_nodes],
            },
        ],
    }

    rewritten = rewrite_drawer_close_resources(
        graph,
        drawer_bindings=frozenset({("cabinet", "part_test")}),
    )

    assert all(
        node["call"]["resources"]["primary"] == "right"
        for node in rewritten["nodes"][1:4]
    )
    assert all(
        node["call"]["resources"]["primary"] == "left"
        for node in rewritten["nodes"][4:7]
    )


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


def test_placement_candidates_erode_payload_footprint_and_rotate() -> None:
    low = np.asarray([-0.16, -0.21, 0.01])
    high = np.asarray([0.16, 0.21, 0.15])
    vertices = trimesh.creation.box(extents=[0.08, 0.04, 0.06]).vertices
    candidates = placement_pose_candidates(vertices, np.eye(3), np.eye(4), low, high)
    assert len(candidates) >= 2
    reports = [
        drawer_containment_report(vertices, pose, np.eye(4), low, high)
        for pose in candidates
    ]
    assert all(report["accepted"] for report in reports)
    assert any(
        not np.allclose(candidate[:3, :3], candidates[0][:3, :3])
        for candidate in candidates[1:]
    )


def test_placement_candidates_keep_oversize_payload_for_runtime_attempt() -> None:
    low = np.asarray([-0.05, -0.05, 0.01])
    high = np.asarray([0.05, 0.05, 0.15])
    vertices = trimesh.creation.box(extents=[0.2, 0.2, 0.04]).vertices
    candidates = placement_pose_candidates(vertices, np.eye(3), np.eye(4), low, high)
    assert candidates
    assert any(
        not drawer_containment_report(vertices, pose, np.eye(4), low, high)["accepted"]
        for pose in candidates
    )


def test_drawer_containment_checks_all_payload_vertices() -> None:
    low = np.asarray([-0.1, -0.1, 0.0])
    high = np.asarray([0.1, 0.1, 0.1])
    vertices = trimesh.creation.box(extents=[0.05, 0.05, 0.05]).vertices
    inside_pose = np.eye(4)
    inside_pose[2, 3] = 0.025
    inside = drawer_containment_report(vertices, inside_pose, np.eye(4), low, high)
    outside_pose = inside_pose.copy()
    outside_pose[0, 3] = 0.09
    outside = drawer_containment_report(vertices, outside_pose, np.eye(4), low, high)
    assert inside["accepted"] is True
    assert outside["accepted"] is False


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


def test_candidate_rotations_follow_drawer_local_axes() -> None:
    low = np.asarray([-0.16, -0.21, 0.01])
    high = np.asarray([0.16, 0.21, 0.15])
    vertices = trimesh.creation.box(extents=[0.08, 0.04, 0.06]).vertices
    link = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    candidates = placement_pose_candidates(vertices, np.eye(3), link, low, high)
    assert len(candidates) >= 2
    local_rotations = [link[:3, :3].T @ pose[:3, :3] for pose in candidates]
    np.testing.assert_allclose(local_rotations[0][:, 2], local_rotations[1][:, 2])


def test_drawer_prefilter_warm_starts_and_caps_full_candidates() -> None:
    from types import SimpleNamespace

    from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
        _prefilter_place_candidates,
    )
    from embodichain.lab.sim.atomic_actions import PlaceGoal

    class Robot:
        qpos = torch.zeros(1, 2)

        def __init__(self) -> None:
            self.calls = []

        def compute_ik(self, *, pose, name, joint_seed):
            x = float(pose[0, 0, 3])
            self.calls.append((x, joint_seed.clone()))
            if x == 1.0:
                return torch.tensor([False]), joint_seed.clone()
            step = 0.01 if x == 2.0 else 0.2
            return torch.tensor([True]), joint_seed + step

    robot = Robot()
    # PlanningContext.robot is a state snapshot, not the live IK-capable Robot.
    context = SimpleNamespace(
        batch_size=1, robot=SimpleNamespace(qpos=robot.qpos.clone())
    )
    motion = SimpleNamespace(
        joint_ids=(0, 1), control_part="arm", require_target=lambda _: motion
    )
    request = SimpleNamespace(
        binding=SimpleNamespace(endpoint=lambda *_: motion),
    )

    candidates = []
    for x in (0.0, 1.0, 2.0):
        xpos = torch.eye(4).repeat(1, 3, 1, 1)
        xpos[:, :, 0, 3] = x
        candidates.append(SimpleNamespace(goal=PlaceGoal(xpos)))

    selected, records = _prefilter_place_candidates(
        request, context, tuple(candidates), robot
    )
    assert [index for index, _ in selected] == [2, 0]
    assert records[0]["accepted"] is True
    assert records[1]["failure_stage"] == "outside"
    assert len(robot.calls) == 7
    assert torch.equal(robot.calls[1][1], torch.full((1, 2), 0.2))


def test_drawer_velocity_rejection_keeps_typed_failed_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    from embodichain.gen_sim.task_engine._task_program import planning_probe
    from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
        _validate_drawer_plan_velocity,
    )

    @dataclass(frozen=True)
    class Diagnostics:
        failure: object
        metadata: dict

    @dataclass(frozen=True)
    class Plan:
        plan_success: torch.Tensor
        diagnostics: Diagnostics

    plan = Plan(torch.tensor([True]), Diagnostics(None, {}))
    monkeypatch.setattr(
        planning_probe,
        "_validate_final_plan_velocity",
        lambda plan, robot: {
            "valid_mask": torch.tensor([False]),
            "diagnostics": [{"speed_ratio": 2.0}],
        },
    )
    result = _validate_drawer_plan_velocity(plan, object())
    assert result is not None
    assert result.plan_success.tolist() == [False]
    assert result.diagnostics.failure.code == "drawer_place_velocity_limit"


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


@pytest.mark.parametrize("destination", ["drawer_target", "ordinary_target"])
def test_drawer_pick_does_not_require_future_place_feasibility(
    monkeypatch, destination
) -> None:
    from dataclasses import dataclass
    from types import SimpleNamespace
    from embodichain.lab.sim.atomic_actions import (
        AntipodalAffordance,
        GraspGoal,
        ObjectSemantics,
        PickUpOptions,
        SceneEntityPose,
    )
    from embodichain.gen_sim.task_engine._task_program.drawer_runtime import (
        DrawerPlacementEngine,
    )
    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
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
        PickUpOptions(downstream_object_target_poses=(SceneEntityPose(destination),)),
    )
    recorded = []

    transform = lambda request, context, plan: plan

    def plan(_self, prepared, context, *, plan_transform=None):
        assert plan_transform is transform
        recorded.append(prepared)
        return SimpleNamespace(plan_success=torch.tensor([False]))

    monkeypatch.setattr(GenSimActionEngine, "_plan_request", plan)
    engine = object.__new__(DrawerPlacementEngine)
    engine._planning_services = SimpleNamespace(robot=object())
    engine._drawer_observations = (
        SimpleNamespace(
            route=SimpleNamespace(object_id="cube", affordance="drawer_target")
        ),
    )
    engine.plan_request(request, object(), plan_transform=transform)
    assert len(recorded[0].skill_options.downstream_object_target_poses) == (
        0 if destination == "drawer_target" else 1
    )
    assert len(request.skill_options.downstream_object_target_poses) == 1
