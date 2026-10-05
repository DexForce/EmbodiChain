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
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine._task_program import twist_binding
from embodichain.gen_sim.task_engine._task_program.press import PressSample
from embodichain.gen_sim.task_engine._task_program.twist_binding import (
    discover_twist,
    TwistRoute,
)
from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
    COARSE_TURN_THRESHOLD,
    GenSimTwist,
    QPOS_PATH_DEADBAND,
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


def test_twist_wrapper_preserves_public_skill_descriptor():
    from embodichain.lab.sim.atomic_actions import Twist

    assert GenSimTwist.descriptor() == Twist.descriptor()


@pytest.mark.parametrize(
    "angle,count",
    [(0.0, 8), (-math.pi / 2, 32), (math.pi / 4, 32), (1.3613568296, 32)],
)
def test_twist_chunk_budget_covers_target_and_fallbacks(angle, count):
    actual = twist_binding.required_chunk_count(angle)
    assert actual == count
    assert (
        actual - twist_binding.TWIST_FALLBACK_CHUNKS
    ) * twist_binding.TWIST_CHUNK_ANGLE >= abs(angle) - 1e-10


def test_twist_chunk_budget_rejects_unbounded_turn():
    with pytest.raises(ValueError, match="bounded chunk budget"):
        twist_binding.required_chunk_count(math.pi)
    with pytest.raises(ValueError, match="finite calibrated"):
        twist_binding.required_chunk_count(float("nan"))


def test_twist_candidate_table_is_bounded_and_distinct():
    table = _candidate_table()
    assert len(table) == 12
    assert [item[0] for item in table] == list(range(len(table)))
    assert len({(item[1], item[2]) for item in table}) == len(table)
    assert TWIST_CANDIDATE_FALLBACKS == 3
    assert twist_binding.TWIST_CHUNK_ANGLE == pytest.approx(math.radians(5.0))
    assert COARSE_TURN_THRESHOLD == pytest.approx(math.radians(15.0))
    assert QPOS_PATH_DEADBAND == pytest.approx(math.radians(0.5))


def test_twist_depth_candidates_are_final_bites_without_deep_offsets():
    bite = 0.00625
    table = _candidate_table(bite)
    for roll in (0.0, math.pi / 2, -math.pi / 2, math.pi):
        depths = [bite + shift for _, angle, shift in table if angle == roll]
        assert depths == pytest.approx([bite / 3, 2 * bite / 3, bite])
    assert all(shift <= 0 for _, _, shift in table)
    with pytest.raises(ValueError, match="positive.*bite"):
        _candidate_table(0.0)


def test_twist_candidate_order_prefers_shallow_depth_and_keeps_roll():
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        _candidate_order,
    )

    table = _candidate_table(0.00625)
    scores = {
        2: (1.0, 0.4, 0.02, 2.0),
        6: (0.0, 0.0, 0.0, 0.0),
        10: (0.0, 0.0, 0.0, 0.0),
        8: (0.0, 0.0, 0.0, 0.0),
    }
    assert _candidate_order(table, scores) == (2, 6, 10)


def test_twist_candidate_order_skips_unreachable_depths_without_changing_roll():
    from embodichain.gen_sim.task_engine._task_program.twist_runtime import (
        _candidate_order,
    )

    table = _candidate_table(0.00625)
    assert _candidate_order(
        table, {6: (0.0, 0.0, 0.0, 0.0), 10: (0.0, 0.0, 0.0, 0.0)}
    ) == (6, 10)
    assert _candidate_order(table, {}) == ()


def test_twist_fixed_roll_keeps_exactly_three_final_depths():
    bite, roll = 0.00625, -math.pi / 2
    table = _candidate_table(bite, fixed_roll=roll)
    assert len(table) == 3
    assert all(angle == roll for _, angle, _ in table)
    assert [bite + shift for _, _, shift in table] == pytest.approx(
        [bite / 3, 2 * bite / 3, bite]
    )


def test_twist_fixed_roll_is_a_fingerprinted_route_value(knob_scene):
    payload = discover_twist(knob_scene.articulations[0], 90).payload()
    payload["fixed_roll"] = -math.pi / 2
    route = TwistRoute.decode(payload)
    assert route.fixed_roll == -math.pi / 2
    assert TwistRoute.decode(json.loads(json.dumps(route.payload()))) == route


@pytest.mark.parametrize("roll", [True, float("nan"), math.pi + 0.01])
def test_twist_fixed_roll_rejects_invalid_values(knob_scene, roll):
    payload = discover_twist(knob_scene.articulations[0], 90).payload()
    payload["fixed_roll"] = roll
    with pytest.raises(ValueError):
        TwistRoute.decode(payload)


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
    sim = SimpleNamespace(
        num_envs=1, get_articulation=lambda uid: art, get_sensor=lambda uid: None
    )
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
    assert calls[1:-1] == ["gen_sim.twist"] * 32
    scene = load_config(root / "components/scene.yaml")
    assert scene["simulation"]["articulation"][0]["asset_physics_mode"] == "overlay"
    program = load_config(root / "task_program/program.yaml")
    program["program"]["items"][1]["post"] = []
    save_config(root / "task_program/program.yaml", program)
    with pytest.raises(ValueError, match="physical post-policies"):
        _verify_integration_fingerprint(
            root, root / "task_program_deployment.yaml", g, fp
        )


def _scaled_knob_with_distractor(knob_scene, scale=0.159):
    """Return a source-scale E8 scene with one movable overlapping distractor."""
    art = {**deepcopy(knob_scene.articulations[0]), "body_scale": [scale] * 3}
    distractor = {
        "uid": "distractor",
        "shape": {"shape_type": "Cube", "size": [0.02, 0.02, 0.02]},
        "init_pos": [0.01, 0.0, 0.8],
        "init_rot": [0.0, 0.0, 0.0],
        "body_scale": [1.0, 1.0, 1.0],
    }
    planner_distractor = {
        **distractor,
        "runtime_uid": "distractor",
        "role": "rigid_object",
    }
    return replace(
        knob_scene,
        articulations=(art,),
        rigid_objects=(distractor,),
        planner_objects=(
            {**art, "runtime_uid": "control", "role": "articulation"},
            knob_scene.planner_objects[1],
            planner_distractor,
        ),
    )


def test_e8_prepare_preserves_source_scale(knob_scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        prepare_press_scene,
    )

    scene = replace(
        knob_scene,
        articulations=(
            {**deepcopy(knob_scene.articulations[0]), "body_scale": [0.159] * 3},
        ),
    )
    route = discover_twist(scene.articulations[0], 90)
    adapted, audit = prepare_press_scene(scene, (route,), interaction="E8")
    assert adapted.articulations[0]["body_scale"] == pytest.approx([0.159] * 3)
    assert audit[0]["scale_policy"] == "preserve_source"


def test_e8_interaction_scale_requires_qualification(knob_scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        prepare_press_scene,
    )

    scene = _scaled_knob_with_distractor(knob_scene)
    route = discover_twist(scene.articulations[0], 90)
    with pytest.raises(ValueError, match="independent asset/gripper qualification"):
        prepare_press_scene(
            scene,
            (route,),
            interaction="E8",
            interaction_scale=1.0,
        )


def test_e8_qualified_scale_relayouts_only_declared_distractor(knob_scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        prepare_press_scene,
    )

    scene = _scaled_knob_with_distractor(knob_scene)
    route = discover_twist(scene.articulations[0], 90)
    adapted, audit = prepare_press_scene(
        scene,
        (route,),
        interaction="E8",
        interaction_scale=1.0,
        interaction_scale_qualified=True,
        relayout_uids={"distractor"},
    )
    assert adapted.articulations[0]["body_scale"] == [1.0] * 3
    assert adapted.rigid_objects[0]["init_pos"][2] == pytest.approx(0.8)
    assert adapted.rigid_objects[0]["init_pos"][:2] != pytest.approx([0.01, 0.0])
    assert (
        adapted.planner_objects[-1]["init_pos"] == adapted.rigid_objects[0]["init_pos"]
    )
    assert audit[0]["relayout"][0]["support_policy"] == "preserve_z_and_orientation"
    assert audit[0]["scale_policy"] == "qualified_interaction_scale"


def test_e8_overlap_without_declared_relayout_fails_closed(knob_scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        prepare_press_scene,
    )

    scene = _scaled_knob_with_distractor(knob_scene)
    route = discover_twist(scene.articulations[0], 90)
    with pytest.raises(ValueError, match="overlaps 'distractor'"):
        prepare_press_scene(scene, (route,), interaction="E8")


def test_e8_bundle_passes_unreferenced_rigid_ids_to_relayout(knob_scene, tmp_path):
    from unittest.mock import patch

    from embodichain.gen_sim.task_engine._task_program import press_binding

    with patch.object(
        press_binding,
        "prepare_press_scene",
        wraps=press_binding.prepare_press_scene,
    ) as prepare:
        generate_task_program_bundle(
            graph(knob_scene),
            knob_scene,
            output_dir=tmp_path / "bundle",
            robot_profile="dual_franka",
        )
    e8_calls = [
        call
        for call in prepare.call_args_list
        if call.kwargs.get("interaction") == "E8"
    ]
    assert len(e8_calls) == 1
    assert e8_calls[0].kwargs["relayout_uids"] == set()


def test_official_e8_bundle_adapts_only_small_knob_and_bounds_episode_steps(
    knob_scene, tmp_path
):
    scene = replace(
        knob_scene,
        articulations=(
            {**deepcopy(knob_scene.articulations[0]), "body_scale": [0.159] * 3},
        ),
    )
    _, paths = generate_task_program_bundle(
        graph(scene),
        scene,
        tmp_path / "small",
        robot_profile="dual_franka",
        max_episode_steps=1000000,
    )
    physical = load_config(paths.scene)["simulation"]["articulation"][0]
    assert physical["body_scale"] == pytest.approx([0.5] * 3, abs=1e-7)
    audit = json.loads((paths.root / "twist_adaptation.json").read_text())
    assert audit["scale_candidate"]["candidate_grip_depth"] == pytest.approx(0.010)
    assert audit["records"][0]["support_clearance"]["translation_z"] == 0.001
    assert audit["records"][0]["source_edited"] is False
    assert load_config(paths.deployment)["max_episode_steps"] == 15000
    assert scene.articulations[0]["body_scale"] == [0.159] * 3
    _, smaller = generate_task_program_bundle(
        graph(knob_scene),
        knob_scene,
        tmp_path / "normal",
        robot_profile="dual_franka",
        max_episode_steps=5000,
        twist_support_lift_m=0.004,
    )
    assert load_config(smaller.deployment)["max_episode_steps"] == 5000
    assert (
        load_config(smaller.scene)["simulation"]["articulation"][0]["body_scale"]
        == [1.0] * 3
    )
    normal_audit = json.loads((smaller.root / "twist_adaptation.json").read_text())
    assert normal_audit["scale_candidate"]["reasons"] == []
    assert normal_audit["records"][0]["support_clearance"]["translation_z"] == 0.004


def test_e8_bundle_relayout_keeps_supports_and_supported_children_fixed(
    knob_scene, tmp_path
):
    from unittest.mock import patch
    from embodichain.gen_sim.task_engine._task_program import press_binding

    objects = tuple(
        {
            "uid": uid,
            "shape": {"shape_type": "Cube", "size": [0.04, 0.04, 0.04]},
            "init_pos": [x, 0.0, 0.75],
        }
        for uid, x in (("support", -0.4), ("cargo", -0.3), ("distractor", 0.4))
    )
    scene = replace(
        knob_scene,
        rigid_objects=objects,
        planner_objects=(
            *knob_scene.planner_objects,
            *(
                {**obj, "runtime_uid": obj["uid"], "role": "rigid_object"}
                for obj in objects
            ),
        ),
    )
    source_graph = {
        "nodes": [
            {"object_id": "support", "parent_id": "table", "parent_relation": "on"},
            {"object_id": "cargo", "parent_id": "support", "parent_relation": "on"},
            {"object_id": "distractor", "parent_id": "table", "parent_relation": "on"},
        ]
    }
    scene.source_config_path.with_name("scene_graph.json").write_text(
        json.dumps(source_graph), encoding="utf-8"
    )
    with patch.object(
        press_binding, "prepare_press_scene", wraps=press_binding.prepare_press_scene
    ) as prepare:
        generate_task_program_bundle(
            graph(scene), scene, tmp_path / "protected", robot_profile="dual_franka"
        )
    e8_call = next(
        c for c in prepare.call_args_list if c.kwargs.get("interaction") == "E8"
    )
    assert e8_call.kwargs["relayout_uids"] == {"distractor"}


def test_e8_rejects_mixed_recipe_without_changing_shared_policy(knob_scene):
    value = graph(knob_scene)
    value["task_groups"].append({"task_type": "E1"})
    with pytest.raises(ValueError, match="standalone"):
        twist_binding.graph_routes(value, knob_scene)


def test_twist_coarse_contact_reports_wrong_direction_without_rejecting():
    route = SimpleNamespace(
        binding=SimpleNamespace(target_qpos=1.0, limits=(-2.0, 2.0))
    )
    samples = [
        PressSample(0.0, 0.0, "prepare", False),
        PressSample(0.1, -0.3, "twist", True),
        PressSample(0.2, -0.7, "twist", True),
        PressSample(0.3, 1.0, "cleanup", False),
    ]
    result = evaluate_twist(route, samples, final_qpos=1.0)
    assert result["accepted"]
    assert result["coarse_turn_reached"]
    assert result["reason"] == "coarse_turn_reached"
    assert result["contact_travel"] == pytest.approx(0.0)
    assert result["directional_travel_required"] == pytest.approx(0.2617993878)


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
        final = target * 0.1
    result = evaluate_twist(route, samples, final_qpos=final)
    assert result["accepted"] is (variant in {"positive", "rebound"})
    assert result["criterion"] == "contact_backed_coarse_turn"
    assert result["coarse_turn_reached"] is (variant in {"positive", "rebound"})
    if variant == "wrong_direction":
        assert result["reason"] == "joint_out_of_bounds"
