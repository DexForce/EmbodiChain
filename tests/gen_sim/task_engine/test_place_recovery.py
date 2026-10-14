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

from dataclasses import dataclass
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import trimesh

from embodichain.gen_sim.task_engine._task_program import place_recovery as gate
from embodichain.lab.sim.atomic_actions import JointPositionTarget, PlannerDiagnostics


@dataclass(frozen=True)
class Plan:
    plan_success: torch.Tensor
    joint_trajectory: object
    segments: tuple
    diagnostics: PlannerDiagnostics


def _fixture(monkeypatch):
    vertices = torch.tensor(
        trimesh.creation.box(extents=(0.08, 0.08, 0.08)).vertices, dtype=torch.float32
    )[None]
    pose = torch.eye(4)[None]
    held = SimpleNamespace(
        semantics=SimpleNamespace(entity_id="item"), object_to_eef=pose
    )
    target = JointPositionTarget(control_part="arm", joint_ids=(0, 1))
    binding = SimpleNamespace(
        endpoint=lambda *_a: SimpleNamespace(
            task_state_key="held", require_target=lambda _t: target
        )
    )
    request = SimpleNamespace(
        binding=binding,
        motion_policy=SimpleNamespace(
            strategy="ik_interp", dynamic_collision_mode="off"
        ),
        skill_options=SimpleNamespace(
            preserve_current_object_orientation=True, max_approach_retract_z=None
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        robot=SimpleNamespace(qpos=torch.zeros(1, 2)),
        last_qpos=torch.zeros(1, 2),
        get_held_object=lambda _key: held,
        scene=SimpleNamespace(entities={"table": SimpleNamespace(pose=pose)}),
    )

    def fk(qpos, **_kwargs):
        result = torch.eye(4)[None, None].repeat(1, qpos.shape[1], 1, 1)
        result[:, :, 2, 3] = 0.3
        return result

    robot = SimpleNamespace(
        device=torch.device("cpu"),
        compute_batch_fk=fk,
        get_solver=lambda _: SimpleNamespace(
            get_iteration_params=lambda: {"rot_eps": 0.0005}
        ),
        get_qpos_limits=lambda **_kwargs: torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
    )
    action = SimpleNamespace(
        robot=robot, recovery_scene_vertices={"item": vertices, "table": vertices}
    )
    plan = Plan(
        torch.tensor([True]),
        SimpleNamespace(
            positions=torch.zeros(1, 3, 2), dt=torch.tensor([[0.0, 0.04, 0.04]])
        ),
        (SimpleNamespace(name="approach", stop=2),),
        PlannerDiagnostics(backend="test"),
    )
    monkeypatch.setattr(
        gate,
        "_validate_final_plan_velocity",
        lambda *_args: {"valid_mask": torch.tensor([True])},
    )

    def collision(_robot, qpos, _part):
        assert qpos.shape == (3, 2)
        return {"valid": True}

    monkeypatch.setattr(gate, "_inspect_pick_motion", collision)
    return action, request, context, plan


def test_waypoint_recovery_retains_request_start_and_rng_without_committing_effects(
    monkeypatch,
) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    before = gate._RandomState.capture(set())

    def build(selected, observed):
        assert selected is request and observed is context
        torch.rand(1)
        np.random.rand()
        random.random()
        return plan

    result = gate.recover(action, request, context, first, build)
    assert result.plan_success.tolist() == [True]
    assert result.joint_trajectory is plan.joint_trajectory
    assert context.last_qpos.tolist() == [[0.0, 0.0]]
    assert torch.equal(before.cpu, torch.random.get_rng_state())
    assert before.python == random.getstate()


@pytest.mark.parametrize(
    "case",
    [
        "coarse_success",
        "small_turn",
        "large_turn",
        "unsafe",
        "full_exhausted",
        "full_velocity",
        "boundary",
        "above_boundary",
        "worse_late_turn",
    ],
)
def test_early_waypoint_probe_replays_original_rng_and_retains_big_flip(
    monkeypatch, case
):
    from dataclasses import replace
    from embodichain.gen_sim.task_engine._task_program import actions, motion
    from embodichain.lab.sim.atomic_actions.primitives.place import Place

    source, request, context, good = _fixture(monkeypatch)
    action = actions.GenSimPlace()
    action._planning_services = SimpleNamespace(
        robot=source.robot, motion_generator=SimpleNamespace()
    )
    action.recovery_scene_vertices = source.recovery_scene_vertices
    monkeypatch.setattr(
        actions.GenSimStateHandoff,
        "from_request",
        lambda *a, **kw: SimpleNamespace(resource="held"),
    )
    monkeypatch.setattr(actions, "require_release_effect", lambda p, _: p)
    monkeypatch.setattr(action, "_with_contact_monitor_boundary", lambda p: p)
    failed = replace(good, plan_success=torch.tensor([False]))
    late = replace(good, diagnostics=PlannerDiagnostics(backend="late"))
    monkeypatch.setattr(action, "failed_plan", lambda *a, **kw: failed)
    routes = Mock(return_value=failed)
    monkeypatch.setattr(action, "_retry_place_routes", routes)
    if case == "full_velocity":
        monkeypatch.setattr(
            motion, "_joint_velocity_limits", lambda *a: torch.ones(1, 2)
        )
        monkeypatch.setattr(
            motion, "_velocity_validity", lambda *a: torch.tensor([False])
        )
    calls = []
    before = gate._RandomState.capture(set())

    def build(_action, selected, observed):
        assert selected is request and observed is context
        scope = motion._PLACE_IK_RECOVERY.get()
        kind = (
            "waypoint"
            if motion._WAYPOINT_KINEMATICS.get() is not None
            else ("coarse" if scope.defer_retries else "full")
        )
        if kind == "full":
            assert torch.equal(torch.random.get_rng_state(), before.cpu)
            assert random.getstate() == before.python
            assert np.array_equal(np.random.get_state()[1], before.numpy[1])
        calls.append(kind)
        torch.rand(1)
        random.random()
        np.random.rand()
        if kind == "coarse" and case != "coarse_success":
            return failed
        if kind == "full" and case == "full_velocity":
            scope.used = True
            return late
        if kind == "full" and case in {
            "large_turn",
            "full_exhausted",
            "worse_late_turn",
        }:
            return failed
        if kind == "waypoint" and case == "full_exhausted" and "full" in calls:
            return failed
        return good

    def inspect(_action, _request, _context, plan):
        valid = bool(plan.plan_success.all()) and (case != "unsafe" or "full" in calls)
        rotation = {
            "small_turn": 0.5,
            "boundary": torch.pi / 4,
            "above_boundary": torch.pi / 4 + 1e-4,
            "worse_late_turn": torch.pi if "full" in calls else 1.0,
        }.get(case, torch.pi)
        return {
            "valid": valid,
            "orientation_preserved": False,
            "rotation_max_rad": rotation,
            "rotation_path_rad": 1.0,
        }

    monkeypatch.setattr(Place, "_plan", build)
    monkeypatch.setattr(gate, "inspect", inspect)
    result = action._plan(request, context)
    assert result.plan_success.all()
    assert motion._PLACE_IK_RECOVERY.get() is None
    assert motion._WAYPOINT_KINEMATICS.get() is None
    if case == "coarse_success":
        assert calls == ["coarse"]
        assert result is good
    elif case in {"small_turn", "boundary"}:
        assert calls == ["coarse", "waypoint", "waypoint", "waypoint"]
    else:
        assert calls[:5] == ["coarse", "waypoint", "waypoint", "waypoint", "full"]
        if case == "large_turn":
            assert calls[5:] == ["waypoint", "waypoint", "waypoint"]
        elif case == "full_exhausted":
            routes.assert_called_once()
        elif case == "full_velocity":
            routes.assert_not_called()
            assert result.diagnostics.backend == good.diagnostics.backend
        elif case == "worse_late_turn":
            assert (
                result.diagnostics.metadata["gen_sim_place_waypoint_recovery"][
                    "rotation_max_rad"
                ]
                == 1.0
            )


@pytest.mark.parametrize(
    "damage",
    [
        "successful",
        "batch",
        "cartesian",
        "ceiling",
        "explicit_orientation",
        "dynamic",
        "pour",
    ],
)
def test_qualified_paths_and_other_owned_contracts_remain_unchanged(
    monkeypatch, damage
) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    if damage == "successful":
        first = plan
    elif damage == "batch":
        context.batch_size = 2
    elif damage == "ceiling":
        request.skill_options.max_approach_retract_z = 0.4
    elif damage == "explicit_orientation":
        request.skill_options.preserve_current_object_orientation = False
    elif damage == "dynamic":
        request.motion_policy.dynamic_collision_mode = "auto"
    elif damage == "pour":
        context.get_held_object("held").semantics.affordance = SimpleNamespace(
            get_custom_config=lambda _key: "pour"
        )
    token = gate._CARTESIAN_APPROACH.set(damage == "cartesian")
    try:
        build = Mock()
        assert gate.recover(action, request, context, first, build) is first
        build.assert_not_called()
    finally:
        gate._CARTESIAN_APPROACH.reset(token)


@pytest.mark.parametrize(
    "damage",
    ["velocity", "robot", "joint", "bounds", "scene", "missing_mesh", "missing_pose"],
)
def test_unsafe_recovery_is_never_accepted(monkeypatch, damage) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    if damage == "velocity":
        monkeypatch.setattr(
            gate,
            "_validate_final_plan_velocity",
            lambda *_a: {"valid_mask": torch.tensor([False])},
        )
    elif damage == "robot":
        monkeypatch.setattr(gate, "_inspect_pick_motion", lambda *_a: {"valid": False})
    elif damage == "joint":
        plan.joint_trajectory.positions[0, 1, 0] = 2.0
    elif damage == "bounds":
        action.robot.get_qpos_limits = lambda **_kw: torch.full((1, 2, 2), float("nan"))
    elif damage == "scene":
        action.robot.compute_batch_fk = lambda qpos, **_kw: torch.eye(4)[
            None, None
        ].repeat(1, qpos.shape[1], 1, 1)
    elif damage == "missing_mesh":
        action.recovery_scene_vertices.pop("item")
    else:
        context.scene.entities.clear()
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    assert gate.recover(action, request, context, first, lambda *_a: plan) is first


@pytest.mark.parametrize("smooth", [False, True])
def test_same_grasp_search_prefers_smooth_but_never_discards_safe_flip(
    monkeypatch, smooth
):
    from dataclasses import replace

    action, request, context, plan = _fixture(monkeypatch)
    first = replace(plan, plan_success=torch.tensor([False]))
    alternate = replace(plan, diagnostics=PlannerDiagnostics("alternative"))
    calls = []

    def build(selected, observed):
        assert selected is request and observed is context
        calls.append(len(calls))
        torch.rand(1)
        return plan if len(calls) == 1 else alternate

    def inspect(*_):
        if len(calls) == 1:
            return {
                "valid": True,
                "orientation_preserved": False,
                "rotation_max_rad": torch.pi,
                "rotation_path_rad": 2 * torch.pi,
            }
        if smooth:
            return {
                "valid": True,
                "orientation_preserved": True,
                "rotation_max_rad": 0.0,
                "rotation_path_rad": 0.0,
            }
        return {"valid": False, "reason": "collision"}

    monkeypatch.setattr(gate, "inspect", inspect)
    rng = torch.get_rng_state().clone()
    result = gate.recover(action, request, context, first, build)
    assert result.plan_success.all()
    assert result.diagnostics.backend == ("alternative" if smooth else "test")
    assert len(calls) == (2 if smooth else 3)
    assert torch.equal(torch.get_rng_state(), rng)


def test_large_intermediate_rotation_is_reported_not_rejected(monkeypatch):
    action, request, context, plan = _fixture(monkeypatch)
    original_fk = action.robot.compute_batch_fk

    def turning(qpos, **kwargs):
        poses = original_fk(qpos, **kwargs)
        poses[0, 1, :2, :2] = -torch.eye(2)
        return poses

    action.robot.compute_batch_fk = turning
    evidence = gate.inspect(action, request, context, plan)
    assert evidence["valid"] is True
    assert evidence["orientation_preserved"] is False
    assert evidence["rotation_max_rad"] == pytest.approx(torch.pi)


@pytest.mark.parametrize("damage", [None, "missing", "invalid", "articulation"])
def test_geometry_capture_requires_complete_supported_scene_and_owns_its_copy(
    damage,
) -> None:
    vertices = torch.ones(1, 3, 3)
    native = SimpleNamespace(get_vertices=lambda **_kw: vertices)
    simulation = SimpleNamespace(
        get_rigid_object=lambda uid: native if uid == "a" else None
    )
    bindings = SimpleNamespace(
        rigid_objects=[SimpleNamespace(entity_id="item", simulation_uid="a")],
        articulations=() if damage != "articulation" else (object(),),
    )
    if damage == "missing":
        bindings.rigid_objects.append(
            SimpleNamespace(entity_id="obstacle", simulation_uid="b")
        )
    elif damage == "invalid":
        vertices[0, 0, 0] = float("nan")
    meshes = gate.scene_vertices(simulation, bindings)
    if damage is None:
        assert torch.equal(meshes["item"], vertices)
        assert meshes["item"].data_ptr() != vertices.data_ptr()
    elif damage == "articulation":
        assert meshes.unavailable
    else:
        assert meshes == {}


def _articulated_scene(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program import drawer_geometry

    meshes = {}
    for name, center in (("left", -0.3), ("right", 0.3)):
        panel = trimesh.creation.box(extents=(0.04, 0.5, 0.5))
        panel.apply_translation((center, 0.0, 0.0))
        meshes[("cabinet", name)] = panel
    monkeypatch.setattr(drawer_geometry, "collision_meshes", lambda _cfg: meshes)
    vertices = torch.ones(1, 3, 3)
    art = SimpleNamespace(cfg=SimpleNamespace(fpath="panels.usdc", body_scale=(1,) * 3))
    simulation = SimpleNamespace(
        get_rigid_object=lambda _uid: SimpleNamespace(
            get_vertices=lambda **_kw: vertices
        ),
        get_articulation=lambda _uid: art,
    )
    bindings = SimpleNamespace(
        rigid_objects=[SimpleNamespace(entity_id="item", simulation_uid="a")],
        articulations=[SimpleNamespace(entity_id="cabinet", simulation_uid="native")],
        links=[
            SimpleNamespace(
                entity_id="cabinet_link",
                articulation_id="cabinet",
                native_link_name="cabinet",
            )
        ],
        rigidized_articulations=(),
    )
    return simulation, bindings, meshes


def test_articulation_collision_components_do_not_erase_rigid_geometry(monkeypatch):
    from embodichain.lab.task_program.semantics import SceneCollisionRole

    simulation, bindings, _ = _articulated_scene(monkeypatch)
    bindings.articulations[0].collision_role = SceneCollisionRole.NONE
    meshes = gate.scene_vertices(simulation, bindings)
    assert "item" in meshes
    assert len(meshes.components) == 2
    assert all(item.pose_entity_id == "cabinet_link" for item in meshes.components)
    assert all(item.owner_id == "cabinet" for item in meshes.components)
    assert all(item.faces.shape[1] == 3 for item in meshes.components)
    assert all(item.faces.dtype is torch.int64 for item in meshes.components)
    assert meshes.collision_roles["cabinet"] is SceneCollisionRole.NONE
    assert meshes.pose_entity_ids == ("cabinet_link",)


@pytest.mark.parametrize("damage", ["link_binding", "collision_mesh", "native"])
def test_articulation_capture_retains_explicit_unavailable_not_partial_success(
    monkeypatch, damage
):
    simulation, bindings, native_meshes = _articulated_scene(monkeypatch)
    if damage == "link_binding":
        bindings.links = ()
    elif damage == "collision_mesh":
        native_meshes.clear()
    else:
        simulation.get_articulation = lambda _uid: None
    meshes = gate.scene_vertices(simulation, bindings)
    assert meshes.unavailable
    assert "item" in meshes


def test_connected_hollow_collision_shell_is_not_replaced_by_filled_cabinet_aabb(
    monkeypatch,
):
    simulation, bindings, native_meshes = _articulated_scene(monkeypatch)
    native_meshes.clear()
    native_meshes[("cabinet", "connected_shell")] = trimesh.creation.annulus(
        r_min=0.2, r_max=0.3, height=0.5
    )
    meshes = gate.scene_vertices(simulation, bindings)
    assert "nonconvex_unavailable" in meshes.unavailable
    assert meshes.components == ()


@pytest.mark.parametrize(
    ("scale", "error"),
    [
        ((2.0, 2.0, 2.0), None),
        ((1.0, 2.0, 1.0), "uniform"),
        ((0.0, 0.0, 0.0), "positive"),
    ],
)
def test_articulation_capture_uses_scaled_collision_mesh_in_owning_link_frame(
    tmp_path, scale, error
):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    owner = UsdGeom.Xform.Define(stage, "/root/cabinet")
    owner.AddTranslateOp().Set((10.0, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(owner.GetPrim())
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    for name, enabled in (("panel", True), ("disabled", False)):
        mesh = UsdGeom.Mesh.Define(stage, f"/root/cabinet/{name}")
        mesh.CreatePointsAttr(box.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(box.faces))
        mesh.CreateFaceVertexIndicesAttr(box.faces.reshape(-1).tolist())
        mesh.AddTranslateOp().Set((0.2, 0.0, 0.0))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(
            enabled
        )
    stage.GetRootLayer().Save()
    art = SimpleNamespace(
        cfg=SimpleNamespace(fpath=tmp_path / "scene.usda", body_scale=scale)
    )
    simulation = SimpleNamespace(get_articulation=lambda _uid: art)
    bindings = SimpleNamespace(
        rigid_objects=(),
        articulations=[SimpleNamespace(entity_id="cabinet", simulation_uid="native")],
        links=[
            SimpleNamespace(
                entity_id="panel_link",
                articulation_id="cabinet",
                native_link_name="cabinet",
            )
        ],
    )
    meshes = gate.scene_vertices(simulation, bindings)
    if error is not None:
        assert error in meshes.unavailable
        return
    assert meshes.unavailable is None
    assert len(meshes.components) == 1
    points = meshes.components[0].vertices[0]
    assert points.amin(0).tolist() == pytest.approx([0.2, -0.2, -0.2])
    assert points.amax(0).tolist() == pytest.approx([0.6, 0.2, 0.2])


@pytest.mark.parametrize("ik_success", [True, False])
def test_failure_evidence_distinguishes_waypoint_ik_from_velocity_rejection(
    monkeypatch, ik_success
) -> None:
    action, request, context, plan = _fixture(monkeypatch)
    first = Plan(
        torch.tensor([False]), plan.joint_trajectory, plan.segments, plan.diagnostics
    )
    monkeypatch.setattr(
        gate, "inspect", lambda *_a: {"valid": False, "reason": "rejected"}
    )

    def build(*_a):
        gate._WAYPOINT_KINEMATICS.get().append(torch.tensor([ik_success]))
        return plan

    result = gate.recover(action, request, context, first, build)
    assert not result.plan_success.any()
    assert result.joint_trajectory is first.joint_trajectory
    assert (
        result.diagnostics.metadata["gen_sim_place_waypoint_probe"]["kinematic_valid"]
        is ik_success
    )
    assert gate._WAYPOINT_KINEMATICS.get() is None


@pytest.mark.parametrize("ik_success", [True, False])
def test_raw_waypoint_evidence_precedes_gen_sim_velocity_mask(
    monkeypatch, ik_success
) -> None:
    from embodichain.gen_sim.task_engine._task_program import motion

    generator = object.__new__(motion.CheckedMotionGenerator)
    generator.robot = object()
    raw = motion.PlanResult(
        success=torch.tensor([ik_success]),
        positions=torch.tensor([[[0.0], [2.0], [2.0]]]),
        dt=torch.tensor([[0.0, 0.04, 0.04]]),
    )
    monkeypatch.setattr(motion.MotionGenerator, "generate", Mock(return_value=raw))
    monkeypatch.setattr(motion, "_joint_velocity_limits", lambda *_a: torch.ones(1, 1))
    target = motion.PlanState(
        move_type=motion.MoveType.EEF_MOVE, xpos=torch.eye(4)[None]
    )
    options = motion.MotionGenOptions(
        strategy="ik_interp", control_part="arm", sample_count=3
    )
    evidence = []
    token = gate._WAYPOINT_KINEMATICS.set(evidence)
    try:
        result = generator._generate_checked([target], options)
    finally:
        gate._WAYPOINT_KINEMATICS.reset(token)
    assert not result.success.any()
    assert evidence[0].tolist() == [ik_success]
    assert raw.success.tolist() == [ik_success]
