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
from dataclasses import replace
import hashlib
import json
import math

import pytest

from embodichain.gen_sim.task_engine._task_program import twist_binding
from embodichain.gen_sim.task_engine._task_program.press import PressSample
from embodichain.gen_sim.task_engine._task_program.twist_binding import (
    discover_twist,
    TwistRoute,
)
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    GenSimTwist,
    TWIST_CANDIDATE_FALLBACKS,
    _TwistCandidateState,
    _candidate_contact_decision,
    _candidate_table,
    evaluate_twist,
)
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


@pytest.fixture
def knob_scene(tmp_path, monkeypatch):
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics

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
        UsdPhysics.MassAPI.Apply(p).CreateMassAttr(0.02)
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
    monkeypatch.setitem(
        twist_binding._CALIBRATIONS,
        digest,
        ("knob_rotation", "grip", "pointer", {0: "label0", 90: "label90"}),
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


def test_twist_wrapper_preserves_public_skill_descriptor():
    from embodichain.lab.sim.atomic_actions import Twist

    assert GenSimTwist.descriptor() == Twist.descriptor()


def test_twist_candidate_table_is_bounded_and_distinct():
    table = _candidate_table()
    assert len(table) == 16
    assert [item[0] for item in table] == list(range(len(table)))
    assert len({(item[1], item[2]) for item in table}) == len(table)
    assert TWIST_CANDIDATE_FALLBACKS == 3


def test_twist_candidate_fallback_is_bounded_until_contact_locks_pose():
    state = _TwistCandidateState(order=(7, 2, 11, 5, 9), selected_index=7)
    decisions = [
        _candidate_contact_decision(
            state,
            target_contact=False,
            already_reached=False,
        )
        for _ in range(4)
    ]
    assert decisions == [(True, True), (True, True), (True, True), (False, False)]
    assert state.fallback_count == TWIST_CANDIDATE_FALLBACKS
    assert state.forced_index == 5
    state.locked = False
    assert _candidate_contact_decision(
        state,
        target_contact=True,
        already_reached=False,
    ) == (True, False)
    assert state.locked is True
    assert state.forced_index == state.selected_index


def test_twist_live_lowerer_types_match_their_factory_contracts(knob_scene):
    from types import SimpleNamespace
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        TwistFactory,
        TwistPrepareFactory,
    )
    from embodichain.lab.task_program.semantics import SceneArticulationRef

    config = knob_scene.articulations[0]
    route = discover_twist(config, 90)
    robot = object()
    art = SimpleNamespace(
        uid=config["uid"],
        joint_names=[route.binding.joint],
        cfg=SimpleNamespace(
            fpath=config["fpath"],
            body_scale=config["body_scale"],
            root_props=SimpleNamespace(fixed_base=True),
        ),
    )
    sim = SimpleNamespace(num_envs=1, get_articulation=lambda uid: art)
    registry = SimpleNamespace(
        lookup=lambda *a, **kw: SimpleNamespace(
            parent=SceneArticulationRef(config["uid"]), native_name=route.binding.link
        )
    )
    for factory in (TwistFactory(route), TwistPrepareFactory(route)):
        value = factory.create(
            simulation=sim,
            robot=robot,
            scene_registry=registry,
            engine=SimpleNamespace(robot=robot),
        )
        assert type(value).call_id == factory.call_id
        assert type(value).target_descriptor == factory.target_descriptor


def test_twist_calibrates_label_geometry_not_numeric_label_as_radians(knob_scene):
    route = discover_twist(knob_scene.articulations[0], 90)
    assert route.binding.target_qpos == pytest.approx(-math.pi / 2)
    assert route.binding.axis == (0.0, 0.0, -1.0)
    assert route.binding.axis_sign == -1
    assert TwistRoute.decode(json.loads(json.dumps(route.payload()))) == route


def test_twist_source_hash_and_unknown_settings_fail_closed(knob_scene):
    from pxr import Usd

    art = knob_scene.articulations[0]
    with pytest.raises(ValueError, match="not calibrated"):
        discover_twist(art, 2)
    stage = Usd.Stage.Open(art["fpath"])
    stage.GetRootLayer().comment = "changed source"
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="calibration"):
        discover_twist(art, 90)


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", 0),
        ("scale", True),
        ("target_qpos", float("nan")),
        ("axis", [0, 0, 0]),
        ("target_setting", True),
        ("unknown", 1),
    ],
)
def test_twist_decode_rejects_invalid_binding(knob_scene, field, value):
    payload = discover_twist(knob_scene.articulations[0], 90).payload()
    payload["binding"][field] = value
    with pytest.raises(ValueError):
        TwistRoute.decode(payload)


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
    assert calls[0] == "gen_sim.twist_prepare"
    assert calls[-1] == "gen_sim.articulation_park"
    assert calls[1:-1] == ["gen_sim.twist"] * 8
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
        "passive",
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
    if variant in {"no_contact", "passive"}:
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
    result = evaluate_twist(route, samples, final_qpos=final)
    assert result["accepted"] is (variant in {"positive", "rebound"})
