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

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS
import numpy as np
import torch
import hashlib
import json
import pytest
from embodichain.gen_sim.task_engine._task_program import twist_binding
from embodichain.gen_sim.task_engine._task_program.press import PressSample
from embodichain.gen_sim.task_engine._task_program.twist_binding import discover_twist
from embodichain.gen_sim.task_engine._task_program.twist_runtime import evaluate_twist
from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import _INTENT_FIELD_DEFAULTS
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.gen_sim.task_engine.workflow import _semantic_draft
from embodichain.utils.utility import load_config, save_config


@pytest.mark.parametrize(
    ("body_type", "shape_count", "contact_offset", "error"),
    [
        ("static", 1, 0.01, None),
        ("kinematic", 1, 0.01, None),
        ("static", 2, 0.01, "requires all actual native table colliders"),
        ("static", 1, -0.01, "contact offsets are invalid"),
    ],
)
def test_native_table_reads_static_entity_and_checks_shape_and_contact(
    body_type, shape_count, contact_offset, error
):
    from embodichain.gen_sim.task_engine._task_program.twist_geometry_guard import (
        _native_table,
    )

    body = NS(get_shape_count=lambda: shape_count)
    entity = NS(
        native=lambda: NS(get_physical_body=lambda: body),
        get_physical_attr=lambda: NS(contact_offset=contact_offset, rest_offset=0.0),
    )
    shape = NS(
        local_pose=torch.eye(4),
        vertices=None,
        triangles=None,
        half_extents=torch.tensor([1.0, 2.0, 3.0]),
    )
    table = NS(
        num_instances=1,
        body_type=body_type,
        body_data=None if body_type == "static" else NS(entities=[entity]),
        _entities=[entity],
        get_collision_shapes=lambda env_id: [shape],
        get_local_pose=lambda to_matrix: torch.eye(4).unsqueeze(0),
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            _native_table(table)
        return

    bounds, contact = _native_table(table)
    np.testing.assert_array_equal(bounds, [[-1, -2, -3], [1, 2, 3]])
    assert contact == 0.01


@pytest.mark.parametrize("mode", ["safe", "collision", "missing", "nan"])
def test_table_clearance_requires_complete_finite_geometry(monkeypatch, mode):
    from embodichain.gen_sim.task_engine._task_program import (
        twist_geometry_guard as module,
    )
    from embodichain.lab.sim.atomic_actions import PlannerDiagnostics

    @dataclass(frozen=True)
    class Plan:
        plan_success: object
        joint_trajectory: object
        segments: tuple
        diagnostics: object

    plan = Plan(
        torch.tensor([True]),
        NS(positions=torch.zeros(1, 3, 1)),
        tuple(
            NS(name=name, start=i, stop=i + 1)
            for i, name in enumerate(("approach", "reach", "retract"))
        ),
        PlannerDiagnostics(backend="test"),
    )
    guard = module.E8GeometryGuard.__new__(module.E8GeometryGuard)
    table, mesh = NS(cfg=NS(shape=NS(fpath=None))), NS(path="pad/collision")
    guard._fresh = lambda *args: None
    guard.table, guard._table_path = table, None
    guard.simulation = NS(get_rigid_object=lambda name: table)
    guard._names, guard._meshes = ("pad",), {"pad": (mesh,)}
    guard.sensor = NS(finger_names=("pad",))
    guard._offsets = lambda names: {"pad": 0.002}
    native = NS(get_physical_body=lambda name: NS(get_shape_count=lambda: 1))
    guard.robot = NS(
        joint_names=["arm"],
        get_parent_joint_chain=lambda name: [NS(name="arm")],
        body_data=NS(entities=[NS(render_articulation=native)]),
    )
    monkeypatch.setattr(
        module,
        "_native_table",
        lambda table: (np.array([[0, -1, -1], [1, 1, 1]]), 0.002),
    )
    frames = (0, 1) if mode == "missing" else (0, 1, 2)
    point = float("nan") if mode == "nan" else 0.5 if mode == "collision" else 2.0
    guard._clouds = lambda *args: iter(
        (i, "pad", mesh, torch.tensor([[point, 0, 0]])) for i in frames
    )
    request = NS(
        binding=NS(
            endpoint=lambda *args: NS(require_target=lambda cls: NS(joint_ids=(0,)))
        )
    )
    action = NS(
        failed_plan=lambda *args, **kwargs: replace(
            plan, plan_success=torch.tensor([False])
        )
    )
    result = guard.filter_candidate(action, plan, request, NS())
    evidence = result.diagnostics.metadata["gen_sim_twist_table_clearance"]
    assert bool(result.plan_success.any()) is (mode == "safe")
    assert evidence["accepted"] is (mode == "safe")
    assert evidence["supported"] is (mode in {"safe", "collision"})
    if evidence["supported"]:
        assert evidence["minimum_clearance_after_skin_m"] == pytest.approx(
            0.996 if mode == "safe" else -0.004
        )


def test_robot_vertex_envelope_keeps_extreme_vertices_without_surface_claim(tmp_path):
    import trimesh
    from embodichain.gen_sim.task_engine._task_program.twist_geometry import (
        load_robot_link_meshes,
    )

    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [100, 0, 0]],
        faces=[[0, 1, 2], [0, 1, 3]],
        process=False,
    )
    mesh.export(tmp_path / "collider.stl")
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="test"><link name="arm"><collision><geometry><mesh filename="collider.stl"/></geometry></collision></link></robot>'
    )
    with pytest.raises(ValueError, match="degenerate"):
        load_robot_link_meshes(source, ("arm",))
    value = load_robot_link_meshes(source, ("arm",), vertices_only=True)["arm"][0]
    assert np.array_equal(value.vertices.min(0), [0, 0, 0])
    assert np.array_equal(value.vertices.max(0), [100, 1, 0])
    assert value.faces.shape == (0, 3)
    assert value.approximation == "urdfMeshVertexEnvelope"
    assert not value.vertices.flags.writeable


@pytest.mark.parametrize("mode", ["nan", "empty", "invalid_face"])
def test_robot_vertex_envelope_rejects_invalid_source_geometry(
    tmp_path, monkeypatch, mode
):
    import trimesh
    from embodichain.gen_sim.task_engine._task_program.twist_geometry import (
        load_robot_link_meshes,
    )

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]])
    if mode == "nan":
        vertices[0, 0] = float("nan")
    elif mode == "empty":
        vertices = np.empty((0, 3))
    else:
        faces[0, 2] = 99
    monkeypatch.setattr(
        trimesh, "load", lambda *a, **kw: NS(vertices=vertices, faces=faces)
    )
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="test"><link name="arm"><collision><geometry><mesh filename="collider.stl"/></geometry></collision></link></robot>'
    )
    with pytest.raises(ValueError, match="invalid"):
        load_robot_link_meshes(source, ("arm",), vertices_only=True)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_native_twist_boundary_roundoff_does_not_rewrite_observation(sign):
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        _bounded_twist_qpos,
    )

    bound = 2.5000001290505525
    observed = torch.tensor(sign * 2.500000476837158, dtype=torch.float32)
    original = observed.clone()
    assert _bounded_twist_qpos(observed, (-bound, bound)) == sign * bound
    assert torch.equal(observed, original)


@pytest.mark.parametrize("value", [2.5001, -2.5001, float("nan"), float("inf")])
def test_native_twist_boundary_still_rejects_overtravel_and_nonfinite(value):
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        _bounded_twist_qpos,
    )

    assert _bounded_twist_qpos(torch.tensor(value), (-2.5, 2.5)) is None


@pytest.fixture
def knob_scene(tmp_path, monkeypatch):
    import trimesh
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path = tmp_path / "knob.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Control")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in ("base", "knob"):
        p = UsdGeom.Xform.Define(stage, "/Control/" + name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        mass = UsdPhysics.MassAPI.Apply(p)
        mass.CreateMassAttr(0.02)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.02, 0.03))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0)))
    for body, name, position, size in [
        ("base", "panel", (0, 0, -0.01), (0.2, 0.2, 0.02)),
        ("knob", "grip", (0, 0, 0.02), (0.06, 0.06, 0.02)),
        ("knob", "pointer", (0, 0.02, 0.03), (0.002, 0.015, 0.001)),
        ("base", "label0", (0, 0.07, 0), (0.005, 0.005, 0.001)),
        ("base", "label90", (0.07, 0, 0), (0.005, 0.005, 0.001)),
    ]:
        geometry = trimesh.creation.box(extents=size)
        geometry.apply_translation(position)
        mesh = UsdGeom.Mesh.Define(stage, f"/Control/{body}/{name}")
        mesh.CreatePointsAttr(geometry.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(geometry.faces))
        mesh.CreateFaceVertexIndicesAttr(geometry.faces.reshape(-1).tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Control/knob_rotation")
    joint.CreateBody0Rel().SetTargets(["/Control/base"])
    joint.CreateBody1Rel().SetTargets(["/Control/knob"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(0.0)
    stage.GetRootLayer().Save()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    save_config(
        path.with_suffix(".twist.json"),
        {
            "schema": "gen_sim.twist-control/v1",
            "source_sha256": digest,
            "joint": "/Control/knob_rotation",
            "grip": "/Control/knob/grip",
            "pointer": "/Control/knob/pointer",
            "settings": {
                "0": ["/Control/base/label0"],
                "90": ["/Control/base/label90"],
            },
        },
    )
    art = {
        "uid": "control",
        "fpath": str(path),
        "fix_base": True,
        "body_scale": [1.0] * 3,
        "init_pos": [0, 0, 0.8],
        "init_rot": [0, 0, 0],
    }
    table = {
        "uid": "table",
        "shape": {"shape_type": "Cube", "size": [1, 1, 0.72]},
        "init_pos": [0, 0, 0.36],
    }
    return PreparedScene(
        source_config_path=tmp_path / "scene.json",
        scene_dir=tmp_path,
        planner_objects=(
            {**art, "runtime_uid": "control", "role": "articulation"},
            {**table, "runtime_uid": "table", "role": "background"},
        ),
        background=(table,),
        rigid_objects=(),
        articulations=(art,),
        uid_map={},
        table_top_z=0.72,
        z_rotation_degrees=0,
        body_scale_policy="preserve",
        body_scale=(1, 1, 1),
        asset_hashes={},
    )


def graph(scene):
    twist_binding.enrich_twist_inventory(
        list(scene.planner_objects), list(scene.articulations)
    )
    empty = {
        "kind": "none",
        "reference": "",
        "step_id": "",
        "quantifier": "one",
        "count": 0,
    }
    step = {
        **deepcopy(_INTENT_FIELD_DEFAULTS),
        "id": "step_01",
        "task_type": "E8",
        "object": {**empty, "kind": "scene_ref", "reference": "control"},
        "target": empty,
        "required_arm": "right_arm",
        "target_setting": 90,
        "depends_on": [],
    }
    candidate = TaskAgent(caller=lambda **kw: {"steps": [step]}).generate(
        "twist", "Turn to 90", candidate_count=1
    )["candidates"][0]
    _semantic_draft(candidate)
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "twist",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {"step_01.object": ["control"]},
        "role_bindings": {},
    }
    return SemanticTaskPlanner().plan(candidate, bindings, scene.planner_objects)


def test_twist_bundle_preflight_and_nested_publication(knob_scene, tmp_path):
    from embodichain.gen_sim.task_engine._bundle_runner import (
        _verify_integration_fingerprint,
    )
    from embodichain.gen_sim.task_engine.orchestration.artifacts import (
        ArtifactTransaction,
    )

    destination = tmp_path / "published"
    with ArtifactTransaction(destination) as outer:
        with ArtifactTransaction(outer.staging_dir / "bundle") as inner:
            _, paths = generate_task_program_bundle(
                graph(knob_scene),
                knob_scene,
                inner.staging_dir,
                robot_profile="dual_franka",
            )
            inner.commit()
        outer.commit()
    root = destination / "bundle"
    g = json.loads((root / "semantic_task_graph.json").read_text())
    fp = json.loads((root / "integration_fingerprint.json").read_text())
    deployment = _verify_integration_fingerprint(
        root, root / "task_program_deployment.yaml", g, fp
    )
    route = deployment.integration.adapter_factory.twist_routes[0]
    assert route.binding.target_setting == 90
    calls = [n["call"]["call_id"] for n in g["nodes"]]
    assert calls[0] == route.call_id("gen_sim.twist_prepare")
    assert calls[-1] == "gen_sim.articulation_park"
    assert calls[1:-1] == [route.call_id("gen_sim.twist")] * 32
    scene = load_config(root / "components/scene.yaml")
    assert scene["simulation"]["articulation"][0]["asset_physics_mode"] == "overlay"
    program = load_config(root / "task_program/program.yaml")
    program["program"]["items"][1]["post"] = []
    save_config(root / "task_program/program.yaml", program)
    with pytest.raises(ValueError, match="physical post-policies"):
        _verify_integration_fingerprint(
            root, root / "task_program_deployment.yaml", g, fp
        )


@pytest.mark.parametrize(
    "variant",
    [
        "positive",
        "no_contact",
        "wrong_direction",
        "rebound",
        "stale",
        "missing",
        "small",
    ],
)
def test_twist_evidence_ablation(knob_scene, variant):
    route = discover_twist(knob_scene.articulations[0], 90)
    target = route.binding.target_qpos
    samples = [
        PressSample(0, 0, "prepare", False),
        PressSample(0.1, 0, "twist", True),
        PressSample(0.2, target * 0.5, "twist", True),
        PressSample(0.3, target, "twist", True),
        PressSample(0.4, target, "cleanup", False),
    ]
    final = target
    if variant == "no_contact":
        samples = [replace(s, target_contact=False) for s in samples]
    elif variant == "wrong_direction":
        samples = [replace(s, qpos=-s.qpos) for s in samples]
    elif variant == "rebound":
        final = 0.0
    elif variant == "stale":
        samples[-1] = replace(samples[-1], timestamp=0.2)
    elif variant == "missing":
        samples[2] = replace(samples[2], target_contact=None)
    elif variant == "small":
        samples = [replace(s, qpos=s.qpos * 0.1) for s in samples]
        final = target * 0.1
    result = evaluate_twist(route, samples, final_qpos=final)
    assert result["accepted"] is (variant in {"positive", "rebound"})
    assert result["criterion"] == "contact_backed_coarse_turn"
    assert result["coarse_turn_reached"] is (variant in {"positive", "rebound"})
    if variant == "wrong_direction":
        assert result["reason"] == "joint_out_of_bounds"
