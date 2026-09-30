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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from .test_press import button
from embodichain.gen_sim.task_engine.agent import TaskAgent
from embodichain.gen_sim.task_engine.interpretation import _INTENT_FIELD_DEFAULTS
from embodichain.gen_sim.task_engine.orchestration.contracts import ROLE_BINDINGS_SCHEMA
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.semantic_planner import SemanticTaskPlanner
from embodichain.gen_sim.task_engine.task_program_bundle import (
    generate_task_program_bundle,
)
from embodichain.gen_sim.task_engine._task_program.assembly import load_deployment
from embodichain.gen_sim.task_engine._task_program.press_binding import (
    discover_press,
    PressRoute,
    PRESS_CALL,
    PREPARE_CALL,
)
from embodichain.gen_sim.task_engine._task_program.press_runtime import PressFactory
from embodichain.gen_sim.task_engine._task_program.configured import decode_task_lowerer
from embodichain.gen_sim.task_engine.workflow import _semantic_draft
from embodichain.utils.utility import load_config


@pytest.fixture
def scene(button, tmp_path):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(button["fpath"])
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())
    Sdf.CopySpec(
        stage.GetRootLayer(),
        "/Button/cap/contact",
        stage.GetRootLayer(),
        "/Button/cap/button_cap",
    )
    stage.RemovePrim("/Button/cap/contact")
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/Button/cap")).CreateMassAttr(0.02)
    stage.GetRootLayer().Save()
    art = {
        **button,
        "body_scale": [0.5] * 3,
        "init_pos": [0.1, 0, 0.8],
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
            {**art, "runtime_uid": "button", "role": "articulation"},
            {**table, "runtime_uid": "table", "role": "background"},
        ),
        background=(table,),
        rigid_objects=(),
        articulations=(art,),
        uid_map={"button": "button", "table": "table"},
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
        "task_type": "E9",
        "object": {**empty, "kind": "scene_ref", "reference": "button"},
        "target": empty,
        "required_arm": "right_arm",
        "target_state": "activated",
        "depends_on": [],
    }
    candidate = TaskAgent(caller=lambda **kw: {"steps": [step]}).generate(
        "press", "Press button", candidate_count=1
    )["candidates"][0]
    assert _semantic_draft(candidate)["steps"][0]["task_type"] == "E9"
    bindings = {
        "schema_version": ROLE_BINDINGS_SCHEMA,
        "task_id": "press",
        "candidate_id": candidate["candidate_id"],
        "reference_bindings": {"step_01.object": ["button"]},
        "role_bindings": {},
    }
    return SemanticTaskPlanner().plan(candidate, bindings, scene.planner_objects)


def test_press_complete_bundle_decodes_and_preflights(scene, tmp_path):
    result, paths = generate_task_program_bundle(
        graph(scene), scene, output_dir=tmp_path / "bundle", robot_profile="dual_franka"
    )
    calls = [n["call"]["call_id"] for n in result["nodes"]]
    assert calls[:2] == [PREPARE_CALL, PRESS_CALL]
    cfg = load_config(paths.deployment)
    deployment = load_deployment(
        task_program=cfg["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    from embodichain.lab.task_program.language import load_task_program

    program = load_task_program(
        paths.program,
        integration=deployment.selection,
        validation_context=deployment.integration.registration.catalog,
    )
    deployment.integration.registration.catalog.preflight(program)
    assert len(deployment.integration.adapter_factory.press_routes) == 1
    runtime = load_config(paths.scene)["simulation"]["articulation"][0]
    assert runtime["body_scale"] == [1.0] * 3
    assert runtime["qpos_limits"]["press_joint"] == pytest.approx([-0.005, 0])
    assert runtime["link_attrs"]["gen_sim_press_density"]["attrs"]["mass_props"][
        "mass"
    ] == pytest.approx(0.0025)


def test_press_fingerprint_survives_nested_publication_and_final_copy(scene, tmp_path):
    import json
    import shutil
    import trimesh
    from embodichain.gen_sim.task_engine.orchestration.artifacts import (
        ArtifactTransaction,
    )
    from embodichain.gen_sim.task_engine._bundle_runner import (
        _verify_integration_fingerprint,
    )

    mesh = tmp_path / "table.glb"
    trimesh.creation.box(extents=[1.0, 0.72, 1.0]).export(mesh)
    table = {**scene.background[0], "shape": {"shape_type": "Mesh", "fpath": str(mesh)}}
    scene = replace(
        scene,
        background=(table,),
        planner_objects=(
            scene.planner_objects[0],
            {**table, "runtime_uid": "table", "role": "background"},
        ),
    )

    def verify(root):
        return _verify_integration_fingerprint(
            root,
            root / "task_program_deployment.yaml",
            json.loads((root / "semantic_task_graph.json").read_text()),
            json.loads((root / "integration_fingerprint.json").read_text()),
        ).integration.integration_fingerprint

    with ArtifactTransaction(tmp_path / "run") as outer:
        attempt = outer.staging_dir / "attempts" / "bundle"
        with ArtifactTransaction(attempt) as inner:
            generate_task_program_bundle(
                graph(scene),
                scene,
                output_dir=inner.staging_dir,
                robot_profile="dual_franka",
            )
            before = verify(inner.staging_dir)
            published = inner.commit()
        assert verify(published) == before
        final = outer.staging_dir / "final" / "bundle"
        shutil.copytree(published, final)
        assert verify(final) == before
        root = outer.commit()
    assert verify(root / "attempts" / "bundle") == before
    assert verify(root / "final" / "bundle") == before

    cfg = load_config(root / "final" / "bundle" / "components" / "scene.yaml")
    local_asset = Path(cfg["simulation"]["background"][0]["shape"]["fpath"])
    local_asset.write_bytes(b"changed normalized geometry")
    with pytest.raises(ValueError, match="fingerprint drifted"):
        verify(root / "final" / "bundle")


def test_press_route_roundtrip_and_closed_decoder(scene):
    route = discover_press(scene.articulations[0])
    assert PressRoute.decode(route.payload()) == replace(
        route, binding=replace(route.binding, source_path="")
    )
    assert isinstance(
        decode_task_lowerer({"kind": "press", "route": route.payload()}, path="test"),
        PressFactory,
    )
    with pytest.raises(ValueError):
        decode_task_lowerer(
            {"kind": "press", "route": route.payload(), "anything": True}, path="test"
        )


def test_generator_sets_only_selected_press_to_unit_scale(scene, tmp_path):
    import json
    from embodichain.gen_sim.task_engine.scene.articulation_geometry import (
        read_articulation_geometry,
    )

    original = deepcopy(scene.articulations[0])
    other = {
        **deepcopy(original),
        "uid": "unselected",
        "body_scale": [0.25] * 3,
        "init_pos": [0.4, 0, 0.8],
    }
    scene = replace(
        scene,
        articulations=(*scene.articulations, other),
        planner_objects=(
            *scene.planner_objects,
            {**other, "runtime_uid": "unselected", "role": "articulation"},
        ),
    )
    source_bytes = Path(original["fpath"]).read_bytes()
    _, paths = generate_task_program_bundle(
        graph(scene), scene, output_dir=tmp_path / "unit", robot_profile="dual_franka"
    )
    art, untouched = load_config(paths.scene)["simulation"]["articulation"]
    assert art["body_scale"] == [1.0, 1.0, 1.0]
    assert untouched["body_scale"] == other["body_scale"]
    assert art["init_pos"][:2] == original["init_pos"][:2]
    geometry = read_articulation_geometry(original["fpath"])
    assert geometry.world_vertices(art)[:, 2].min() == pytest.approx(
        geometry.world_vertices(original)[:, 2].min(), abs=1e-7
    )
    assert scene.articulations[0] == original
    assert Path(original["fpath"]).read_bytes() == source_bytes
    audit = json.loads((paths.root / "press_adaptation.json").read_text())["records"][0]
    assert audit["original_body_scale"] == [0.5] * 3
    assert audit["adapted_body_scale"] == [1.0] * 3
    assert audit["source_edited"] is False
    assert art["joint_drive_props"]["stiffness"]["press_joint"] == 500.0
    assert art["link_attrs"]["gen_sim_press_density"]["attrs"]["mass_props"][
        "mass"
    ] == pytest.approx(0.0025)


def test_press_unit_scale_rejects_table_overhang_without_mutating_source(scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        prepare_press_scene,
    )

    art = {**deepcopy(scene.articulations[0]), "init_pos": [0.49, 0, 0.8]}
    original = deepcopy(art)
    selected = replace(scene, articulations=(art,))
    with pytest.raises(ValueError, match="table footprint"):
        prepare_press_scene(selected, (discover_press(art),))
    assert art == original


def test_press_unit_scale_does_not_touch_non_press_scenes(scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        prepare_press_scene,
    )

    result, audit = prepare_press_scene(scene, ())
    assert result is scene
    assert audit == []


def test_unit_scale_preserves_all_release_drive_parameters(scene):
    from pxr import Usd, UsdPhysics
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        calibrate_scene,
    )

    config = scene.articulations[0]
    stage = Usd.Stage.Open(config["fpath"])
    UsdPhysics.MassAPI(stage.GetPrimAtPath("/Button/cap")).GetMassAttr().Set(1.0)
    stage.GetRootLayer().Save()
    source = discover_press(config)
    unit = discover_press({**config, "body_scale": [1.0] * 3})
    before = {"simulation": {"articulation": [{"uid": config["uid"]}]}}
    after = deepcopy(before)
    calibrate_scene(before, (source,))
    calibrate_scene(after, (unit,), source_routes=(source,))
    before_art = before["simulation"]["articulation"][0]
    after_art = after["simulation"]["articulation"][0]
    assert after_art["joint_drive_props"] == before_art["joint_drive_props"]
    assert after_art["link_attrs"] == before_art["link_attrs"]
    assert after_art["qpos_limits"] != before_art["qpos_limits"]


def test_press_unit_scale_measures_glb_table_in_runtime_axes(scene, tmp_path):
    import trimesh
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        prepare_press_scene,
    )

    mesh = tmp_path / "long_table.glb"
    trimesh.creation.box(extents=[0.3, 0.05, 1.2]).export(mesh)
    table = {
        "uid": "table",
        "shape": {"shape_type": "Mesh", "fpath": str(mesh)},
        "init_pos": [0, 0, 0.7],
    }
    art = {**deepcopy(scene.articulations[0]), "init_pos": [0, 0.4, 0.8]}
    result, _ = prepare_press_scene(
        replace(scene, background=(table,), articulations=(art,)),
        (discover_press(art),),
    )
    assert result.articulations[0]["body_scale"] == [1.0] * 3


@pytest.mark.parametrize("extra", [0.0, 0.010])
def test_extra_press_distance_changes_command_only(scene, tmp_path, extra):
    from unittest.mock import patch
    from embodichain.gen_sim.task_engine._task_program import press_binding

    original = press_binding.graph_routes
    with patch.object(
        press_binding,
        "graph_routes",
        lambda g, s: tuple(
            replace(r, extra_press_distance=extra) for r in original(g, s)
        ),
    ):
        _, paths = generate_task_program_bundle(
            graph(scene),
            scene,
            output_dir=tmp_path / "bundle",
            robot_profile="dual_franka",
        )
    cfg = load_config(paths.deployment)
    deployment = load_deployment(
        task_program=cfg["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    route = deployment.integration.adapter_factory.press_routes[0]
    assert route.binding == replace(
        discover_press({**scene.articulations[0], "body_scale": [1.0] * 3}).binding,
        source_path="",
    )
    assert route.extra_press_distance == extra
    for preset in deployment.integration.registration.robot_profile_binding.presets:
        assert preset.action_option_templates[
            PRESS_CALL
        ].press_distance == pytest.approx(route.binding.stroke + extra)
        assert preset.recovery_policy.max_replans == 0
        assert preset.recovery_policy.max_action_retries == 0
    art = load_config(paths.scene)["simulation"]["articulation"][0]
    assert art["qpos_limits"][route.binding.joint] == list(route.binding.limits)
    assert art["joint_drive_props"]["stiffness"][route.binding.joint] == 500.0


@pytest.mark.parametrize(
    "extra", [-0.001, 0.01001, float("nan"), float("inf"), True, "0.001"]
)
def test_extra_press_distance_rejects_invalid_values(scene, extra):
    payload = discover_press(scene.articulations[0]).payload()
    payload["extra_press_distance"] = extra
    with pytest.raises(ValueError, match="extra_press_distance"):
        PressRoute.decode(payload)


def test_press_route_defaults_to_requested_ten_mm_extra_distance(scene):
    payload = discover_press(scene.articulations[0]).payload()
    payload.pop("extra_press_distance", None)
    assert PressRoute.decode(payload).extra_press_distance == 0.010


def test_stop_label_does_not_synthesize_a_latching_mechanism(scene):
    ordinary = discover_press(scene.articulations[0])
    stop = discover_press({**scene.articulations[0], "name": "emergency STOP"})
    assert stop == ordinary
    assert "mechanism" not in stop.payload()


def test_runtime_binding_rejects_source_changes(scene):
    from pxr import Usd
    from embodichain.gen_sim.task_engine._task_program.press_runtime import _bind
    from embodichain.lab.task_program.semantics import SceneArticulationRef

    config = scene.articulations[0]
    route = discover_press(config)
    stage = Usd.Stage.Open(config["fpath"])
    stage.GetRootLayer().comment = "source changed after binding"
    stage.GetRootLayer().Save()
    art = SimpleNamespace(
        uid=config["uid"],
        cfg=SimpleNamespace(
            fpath=config["fpath"],
            body_scale=config["body_scale"],
            root_props=SimpleNamespace(fixed_base=True),
        ),
    )
    sim = SimpleNamespace(num_envs=1, get_articulation=lambda uid: art)
    registry = SimpleNamespace(
        lookup=lambda *args, **kwargs: SimpleNamespace(
            parent=SceneArticulationRef(config["uid"]), native_name=route.binding.link
        )
    )
    with pytest.raises(ValueError, match="source hash"):
        _bind(route, sim, registry, None)


def test_duplicate_button_joint_is_ambiguous(scene):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(scene.articulations[0]["fpath"])
    UsdPhysics.PrismaticJoint.Define(stage, "/Button/other_press")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="unambiguous"):
        discover_press(scene.articulations[0])


@pytest.mark.parametrize("contact", [True, False])
def test_contact_sensor_never_drives_button_even_after_full_press(scene, contact):
    from embodichain.gen_sim.task_engine._task_program.press_runtime import (
        PressContactSensor,
    )
    from embodichain.gen_sim.task_engine._task_program.press import PressSample

    route = discover_press(scene.articulations[0])
    stroke = route.binding.stroke
    commands = []
    art = SimpleNamespace(
        get_qpos=lambda: torch.tensor([[-stroke]]),
        set_qpos=lambda q, **kw: commands.append((q, kw)),
    )

    data = {
        "is_valid": torch.tensor([[True]]),
        "user_ids": torch.tensor([[[1, 2 if contact else 3]]]),
        "impulse": torch.ones(1, 1),
        "position": torch.zeros(1, 1, 3),
        "normal": torch.zeros(1, 1, 3),
        "distance": torch.zeros(1, 1),
    }
    sensor = SimpleNamespace(
        route=route,
        art=art,
        target_actor=1,
        parent_actor=4,
        finger_actors={2},
        joint_index=0,
        clock=0.02,
        samples=[
            PressSample(0, 0, "prepare", False),
            PressSample(0.01, 0, "press", True),
        ],
        trace=[],
        dropped_contacts=0,
        get_data=lambda: data,
    )
    PressContactSensor.capture(sensor, "press")
    assert commands == []


@pytest.fixture
def press_acceptance(scene, monkeypatch):
    from dataclasses import asdict
    from embodichain.gen_sim.task_engine._task_program import press_runtime
    from embodichain.gen_sim.task_engine._task_program.press import PressSample

    route = discover_press(scene.articulations[0])
    route = replace(
        route,
        binding=replace(
            route.binding,
            axis=(1.0, 0.0, 0.0),
            press_position=(0.0, 0.0, 0.0),
            limits=(0.0, 0.005),
            released_position=0.0,
            pressed_position=0.005,
        ),
    )
    state = SimpleNamespace(position=[-0.1, 0.0, 0.0])

    def hand_pose(*args, **kwargs):
        pose = torch.eye(4).unsqueeze(0)
        pose[0, :3, 3] = torch.tensor(state.position)
        return pose

    robot = SimpleNamespace(
        get_qpos=lambda **kw: torch.zeros(1, 1),
        compute_fk=hand_pose,
        get_link_pose=hand_pose,
        get_link_vert_face=lambda name: (
            torch.tensor([[-0.005] * 3, [0.005] * 3]),
            None,
        ),
    )
    art = SimpleNamespace(
        link_names=[route.binding.link, route.binding.parent],
        get_qpos=lambda: torch.zeros(1, 1),
        get_link_pose=lambda *args, **kw: torch.eye(4).unsqueeze(0),
        get_link_vert_face=lambda name: (torch.tensor([[-0.02] * 3, [0.02] * 3]), None),
    )
    samples = [
        PressSample(0.0, 0.0, "prepare", False),
        PressSample(0.1, 0.0, "press", True),
        PressSample(0.2, 0.0001, "press", True),
        PressSample(0.3, 0.0, "cleanup", False),
        PressSample(0.4, 0.0, "cleanup", False),
    ]
    sensor = SimpleNamespace(
        robot=robot,
        art=art,
        samples=samples,
        trace=[{**asdict(s), "parent_contact": False} for s in samples],
        finger_names=["finger"],
        joint_index=0,
        retreat_verified=False,
        _sim=SimpleNamespace(sim_config=SimpleNamespace(physics_dt=0.1)),
    )
    monkeypatch.setattr(press_runtime, "ensure_sensor", lambda *a: sensor)
    port = press_runtime.PressAcceptancePort(None, None, robot, route, 0.1)
    policy = SimpleNamespace(
        cfg=SimpleNamespace(kind="wait_stable", preset=route.preset("pressed")),
        entity=SimpleNamespace(entity_id=route.binding.object_id),
    )

    def check(terminal=False):
        call_id = press_runtime.PARK_CALL if terminal else PRESS_CALL
        segment = SimpleNamespace(
            calls=[SimpleNamespace(call=SimpleNamespace(semantic_id=call_id))]
        )
        list(port.actions(policy, segment=segment, active_mask=torch.tensor([True])))
        return port.post_policy_metadata(policy, segment=segment)

    return state, sensor, check


def test_park_accepts_sideways_return_only_after_verified_axial_retreat(
    press_acceptance,
):
    state, sensor, check = press_acceptance
    assert check()["accepted"]
    state.position = [0.1, 0.3, 0.0]
    result = check(terminal=True)
    assert result["accepted"]
    assert result["retreat_clearance"] < 0
    assert result["terminal_clearance"] > 0.04
    assert result["retreat_verified"]


@pytest.mark.parametrize(
    "terminal,verified,position,reason",
    [
        (False, False, [0.1, 0.3, 0.0], "retreat_clearance_insufficient"),
        (True, False, [0.1, 0.3, 0.0], "retreat_not_verified"),
        (True, True, [0.0, 0.064, 0.0], "terminal_clearance_insufficient"),
        (True, True, [0.0, 0.0, 0.0], "terminal_clearance_insufficient"),
    ],
)
def test_park_relaxation_preserves_retreat_and_spatial_guards(
    press_acceptance, terminal, verified, position, reason
):
    state, sensor, check = press_acceptance
    state.position = position
    sensor.retreat_verified = verified
    result = check(terminal)
    assert not result["accepted"]
    assert result["reason"] == reason


def test_park_rejects_housing_contact_despite_no_button_contact(press_acceptance):
    state, sensor, check = press_acceptance
    assert check()["accepted"]
    state.position = [0.1, 0.3, 0.0]
    sensor.trace[-1]["parent_contact"] = True
    result = check(terminal=True)
    assert not result["accepted"]
    assert result["contact_released"]
    assert result["reason"] == "housing_contact_not_released"


def test_park_rejects_invalid_evidence_after_successful_retreat(press_acceptance):
    state, sensor, check = press_acceptance
    assert check()["accepted"]
    state.position = [0.1, 0.3, 0.0]
    sensor.samples[-1] = replace(sensor.samples[-1], valid=False)
    result = check(terminal=True)
    assert not result["accepted"]
    assert result["reason"] == "invalid_observation"


def test_park_rejects_renewed_button_contact(press_acceptance):
    state, sensor, check = press_acceptance
    assert check()["accepted"]
    state.position = [0.1, 0.3, 0.0]
    sensor.samples[-1] = replace(sensor.samples[-1], target_contact=True)
    result = check(terminal=True)
    assert not result["accepted"]
    assert result["reason"] == "contact_not_released"


def test_park_checks_other_appliance_links_not_just_button(press_acceptance):
    state, sensor, check = press_acceptance
    assert check()["accepted"]
    state.position = [0.1, 0.3, 0.0]
    sensor.art.link_names.append("lid")
    original_pose = sensor.art.get_link_pose
    sensor.art.get_link_pose = lambda name, **kw: (
        sensor.robot.get_link_pose("finger")
        if name == "lid"
        else original_pose(name, **kw)
    )
    result = check(terminal=True)
    assert not result["accepted"]
    assert result["terminal_clearance"] == 0.0


def test_park_retreat_proof_does_not_leak_between_episodes(monkeypatch):
    from embodichain.gen_sim.task_engine._task_program.press_runtime import (
        ContactSensor,
        PressContactSensor,
    )

    monkeypatch.setattr(ContactSensor, "reset", lambda *a: None)
    sensor = object.__new__(PressContactSensor)
    sensor.route = object()
    sensor.armed = True
    sensor.retreat_verified = True
    PressContactSensor.reset(sensor)
    assert not sensor.armed
    assert not sensor.retreat_verified


@pytest.mark.parametrize("bad", ["empty", "nan"])
def test_park_fails_closed_without_finite_appliance_geometry(press_acceptance, bad):
    state, sensor, check = press_acceptance
    assert check()["accepted"]
    state.position = [0.1, 0.3, 0.0]
    vertices = torch.empty(0, 3) if bad == "empty" else torch.full((2, 3), float("nan"))
    sensor.art.get_link_vert_face = lambda name: (vertices, None)
    with pytest.raises(ValueError, match="geometry"):
        check(terminal=True)


def test_e9_rejects_mixed_recipe_without_changing_shared_policy(scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import graph_routes

    value = graph(scene)
    value["task_groups"].append({"task_type": "E1"})
    with pytest.raises(ValueError, match="standalone"):
        graph_routes(value, scene)


@pytest.mark.parametrize(
    "variant",
    ["original", "limits_only", "calibrated", "approach_only"],
)
def test_qualification_variants_are_declared_and_fingerprinted(
    scene, tmp_path, variant
):
    from unittest.mock import patch
    from embodichain.gen_sim.task_engine._task_program import press_binding
    from embodichain.lab.task_program.semantics import EffectAssurance

    original = press_binding.graph_routes
    with patch.object(
        press_binding,
        "graph_routes",
        lambda g, s: tuple(replace(r, variant=variant) for r in original(g, s)),
    ):
        _, paths = generate_task_program_bundle(
            graph(scene),
            scene,
            output_dir=tmp_path / variant,
            robot_profile="dual_franka",
        )
    cfg = load_config(paths.deployment)
    deployment = load_deployment(
        task_program=cfg["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    assert deployment.integration.adapter_factory.press_routes[0].variant == variant
    preset = deployment.integration.registration.robot_profile_binding.presets[0]
    assert preset.effect_assurance is EffectAssurance.PROJECTED
    raw = load_config(paths.scene)
    art = raw["simulation"]["articulation"][0]
    assert ("qpos_limits" in art) is (variant != "original")
    assert ("joint_drive_props" in art) is (variant in {"calibrated", "approach_only"})
    tool = load_config(paths.embodiment)["simulation"]
    assert "gen_sim_press_pads" not in tool.get("link_attrs", {})
    assert all(
        "collision_props" not in group["attrs"]
        for group in art.get("link_attrs", {}).values()
    )
    from embodichain.utils.utility import save_config

    art["qpos_limits"] = {"press_joint": [-0.001, 0]}
    save_config(paths.scene, raw)
    changed = load_deployment(
        task_program=cfg["task_program"],
        skill_profile=load_config(paths.embodiment)["skill_profile"],
        base_dir=paths.root,
    )
    assert (
        changed.integration.integration_fingerprint
        != deployment.integration.integration_fingerprint
    )


def test_press_tip_is_derived_from_closed_hand_geometry_without_state_writes():
    from embodichain.gen_sim.task_engine._task_program.press_runtime import (
        _closed_finger_tip_offset,
    )

    tcp = torch.eye(4)
    tcp[2, 3] = 0.2
    hand = SimpleNamespace(joint_ids=(2,))
    motion = SimpleNamespace(joint_ids=(0, 1), control_part="right_arm")
    grasp = SimpleNamespace(
        require_target=lambda cls: hand,
        joint_positions=lambda *a, **kw: torch.tensor([[0.7]]),
    )
    endpoints = {
        "motion": SimpleNamespace(require_target=lambda cls: motion),
        "grasp": grasp,
    }
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            action_binding=SimpleNamespace(endpoint=lambda slot, name: endpoints[name])
        )
    )

    def fk(qpos, **kwargs):
        assert qpos[0, 2] == pytest.approx(0.7)
        pose = torch.eye(4).repeat(1, 2, 1, 1)
        pose[0, 1, 2, 3] = 0.235
        return pose

    robot = SimpleNamespace(
        link_names=["right_finger_pad"],
        joint_names=["a", "b", "hand"],
        compute_fk=fk,
        cfg=SimpleNamespace(
            solver_cfg={
                "right_arm": SimpleNamespace(end_link_name="tool", tcp=tcp.tolist())
            }
        ),
        get_link_vert_face=lambda name: (
            torch.tensor([[0.0, 0.0, -0.002], [0.0, 0.0, 0.002]]),
            None,
        ),
    )
    context = SimpleNamespace(
        robot=SimpleNamespace(qpos=torch.zeros(1, 3)), batch_size=1
    )
    assert _closed_finger_tip_offset(robot, context, bound) == pytest.approx(
        0.037, abs=1e-6
    )
    assert torch.equal(context.robot.qpos, torch.zeros(1, 3))


def test_press_calibration_preserves_declared_contact_properties(scene):
    from embodichain.gen_sim.task_engine._task_program.press_binding import (
        calibrate_scene,
    )

    route = discover_press(scene.articulations[0])
    authored = {
        "link_names_expr": [route.binding.link, route.binding.parent],
        "attrs": {"collision_props": {"contact_offset": 0.004, "rest_offset": 0.001}},
    }
    art = {
        "uid": route.binding.object_id,
        "link_attrs": {"authored": deepcopy(authored)},
    }
    report = calibrate_scene({"simulation": {"articulation": [art]}}, (route,))
    assert art["link_attrs"]["authored"] == authored
    assert set(art["link_attrs"]["gen_sim_press_density"]["attrs"]) == {"mass_props"}
    assert report[0]["contact_policy"] == "native_contact_envelope"


def test_removing_physical_acceptance_is_rejected(scene, tmp_path):
    from embodichain.utils.utility import save_config

    _, paths = generate_task_program_bundle(
        graph(scene), scene, output_dir=tmp_path / "bundle", robot_profile="dual_franka"
    )
    program = load_config(paths.program)
    program["program"]["items"][1].pop("post")
    save_config(paths.program, program)
    cfg = load_config(paths.deployment)
    with pytest.raises(ValueError, match="physical post-policies"):
        load_deployment(
            task_program=cfg["task_program"],
            skill_profile=load_config(paths.embodiment)["skill_profile"],
            base_dir=paths.root,
        )


@pytest.mark.parametrize("position,accepted", [(0.01, True), (1.0, False)])
def test_final_press_velocity_is_checked_on_every_plan(monkeypatch, position, accepted):
    from embodichain.gen_sim.task_engine._task_program.press_runtime import GenSimPress
    from embodichain.gen_sim.task_engine._task_program import motion
    from embodichain.lab.sim.atomic_actions import Press

    plan = SimpleNamespace(
        plan_success=torch.tensor([True]),
        joint_trajectory=SimpleNamespace(
            positions=torch.tensor([[[0.0], [position]]]),
            dt=torch.tensor([[0.0, 0.04]]),
        ),
    )
    monkeypatch.setattr(Press, "_plan", lambda *args: plan)
    monkeypatch.setattr(GenSimPress, "robot", property(lambda self: None))
    monkeypatch.setattr(
        motion, "_joint_velocity_limits", lambda *args: torch.tensor([[1.0]])
    )
    failed = object()
    monkeypatch.setattr(GenSimPress, "failed_plan", lambda *args, **kwargs: failed)
    assert (GenSimPress()._plan(None, None) is plan) is accepted


@pytest.mark.parametrize(
    "field,value",
    [
        ("scale", float("nan")),
        ("limits", [0]),
        ("axis", [True, 0, 1]),
        ("object_id", ""),
    ],
)
def test_invalid_press_binding_payload_is_rejected(scene, field, value):
    payload = discover_press(scene.articulations[0]).payload()
    payload["binding"][field] = value
    with pytest.raises(ValueError):
        PressRoute.decode(payload)
