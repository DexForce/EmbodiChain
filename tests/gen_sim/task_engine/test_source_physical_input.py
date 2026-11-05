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

"""Normal source preparation composes owners without constructing a simulator."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import pytest
import trimesh
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene
from embodichain.gen_sim.task_engine.source_physical_input import (
    prepare_source_physical_input,
)
from embodichain.gen_sim.task_engine.asset_preparation import fit_prepared_scene_assets


def fixture(tmp_path):
    path = tmp_path / "object.glb"
    trimesh.creation.box(extents=[0.06, 0.08, 0.18]).export(path)
    source = tmp_path / "source.json"
    source.write_text("{}")
    table = dict(
        uid="table",
        shape=dict(shape_type="Cube", size=[1.2, 1.2, 0.1]),
        init_pos=[0.6, 0.0, 0.68],
        init_rot=[0.0, 0.0, 0.0],
        body_scale=[1.0, 1.0, 1.0],
    )
    obj = dict(
        uid="object",
        shape=dict(shape_type="Mesh", fpath=str(path)),
        init_pos=[0.6, 0.0, 0.82],
        init_rot=[0.0, 0.0, 0.0],
        body_scale=[1.0, 1.0, 1.0],
        attrs=dict(mass_props=dict(mass=0.005)),
    )
    planner = tuple(
        dict(value, runtime_uid=value["uid"], source_uid=value["uid"], role=role)
        for value, role in ((table, "background"), (obj, "rigid_object"))
    )
    scene = PreparedScene(
        source,
        tmp_path,
        planner,
        (table,),
        (obj,),
        (),
        {},
        0.73,
        0.0,
        "preserve",
        (1.0, 1.0, 1.0),
        {"object": hashlib.sha256(path.read_bytes()).hexdigest()},
    )
    graph = dict(
        nodes=[
            dict(
                task_type="E1",
                call=dict(kind="pick", object="object", resources=dict(primary="left")),
            )
        ]
    )
    return scene, graph


@pytest.mark.parametrize(
    "profile,effort",
    [("default", 0.5), ("contact_experiment", 5.0), ("composition_experiment", 500.0)],
)
def test_source_preparation_keeps_exact_default_and_explicit_physical_selections(
    tmp_path, monkeypatch, profile, effort
):
    from embodichain.lab.sim.sim_manager import SimulationManager

    monkeypatch.setattr(
        SimulationManager,
        "__init__",
        lambda *a, **kw: pytest.fail("source preparation constructed simulation"),
    )
    scene, graph = fixture(tmp_path)
    before = deepcopy(scene)
    original = scene.rigid_objects[0]["shape"]["fpath"]
    bytes_before = Path(original).read_bytes()
    prepared = prepare_source_physical_input(
        scene,
        graph,
        robot_profile="dual_franka",
        cache_root=tmp_path / "derived",
        runtime_profile=profile,
    )
    physical = prepared.physical
    physical.assert_unchanged()
    assert physical.embodiment["simulation"]["init_pos"] == [-0.7, 0.0, 0.38]
    assert (
        physical.embodiment["simulation"]["joint_drive_props"]["max_effort"]["left_eef"]
        == effort
    )
    pads = (
        physical.embodiment["simulation"]
        .get("link_attrs", {})
        .get("gen_sim_contact_pads")
    )
    assert (pads is not None) == (profile == "contact_experiment")
    assert physical.execution_policy["tracking"]["consecutive_acceptances"] == 5
    assert physical.execution_policy["tracking"]["terminal_settle_timeout"] == 3.0
    assert physical.deployment["physics"] == "default"
    assert physical.deployment["env"]["events"]["settle_objects_on_reset"]["params"][
        "entity_cfgs"
    ] == [{"uid": "object"}]
    assert len(physical.scene.asset_provenance) == 1
    assert physical.scene.rigid_objects[0]["body_scale"] == [1.0, 1.0, 1.0]
    assert scene == before and Path(original).read_bytes() == bytes_before
    changed = deepcopy(physical)
    changed.embodiment["simulation"]["init_pos"][0] += 1
    with pytest.raises(ValueError, match="data changed"):
        changed.assert_unchanged()


def test_optional_fit_precedes_normalization_and_keeps_both_views(tmp_path):
    scene, graph = fixture(tmp_path)
    fitted, report = fit_prepared_scene_assets(
        scene, graph, openings={"left": 0.05}, contact_clearances={"object": 0.003}
    )
    assert report["status"] == "scaled"
    assert report["records"][0]["task_types"] == ["E1"]
    assert (
        fitted.rigid_objects[0]["body_scale"] == fitted.planner_objects[1]["body_scale"]
    )
    assert fitted.rigid_objects[0]["init_pos"] == fitted.planner_objects[1]["init_pos"]
    assert scene.rigid_objects[0]["body_scale"] == [1.0, 1.0, 1.0]
    later = deepcopy(graph)
    later["nodes"].append(
        dict(
            task_type="E1",
            call=dict(kind="pick", object="object", resources=dict(primary="right")),
        )
    )
    with pytest.raises(ValueError, match="stricter"):
        fit_prepared_scene_assets(scene, later, openings={"left": 0.05, "right": 0.04})


def test_source_preparation_validates_inputs_and_hashes_source_assets(tmp_path):
    scene, graph = fixture(tmp_path)
    prepared = prepare_source_physical_input(
        scene, graph, robot_profile="dual_franka", cache_root=tmp_path / "derived"
    )
    source = scene.source_config_path
    assert str(source) in prepared.physical.input_files
    source.write_text("changed")
    with pytest.raises(ValueError, match="file changed"):
        prepared.physical.assert_unchanged()
    with pytest.raises(TypeError, match="PreparedScene"):
        prepare_source_physical_input(
            prepared.physical,
            graph,
            robot_profile="dual_franka",
            cache_root=tmp_path / "other",
        )


def test_source_revolute_preparation_keeps_mass_lineage_and_control_receipts(
    tmp_path, monkeypatch
):
    from sim.scene._control_source_fixture import knob_source
    from embodichain.lab.sim.sim_manager import SimulationManager

    monkeypatch.setattr(
        SimulationManager,
        "__init__",
        lambda *a, **k: pytest.fail("source control preparation started simulation"),
    )
    source, base = fixture(tmp_path)
    art = knob_source(tmp_path)
    art["init_pos"][0] = 0.6
    source = replace(
        source,
        rigid_objects=(),
        articulations=(art,),
        planner_objects=(
            source.planner_objects[0],
            dict(art, runtime_uid="control", source_uid="control", role="articulation"),
        ),
    )
    calls = [
        dict(
            kind="registered",
            call_id=identity,
            arguments=dict(object="control", setting=90),
            resources=dict(primary="left"),
        )
        for identity in ("gen_sim.twist_prepare.operation", "gen_sim.twist.operation")
    ]
    graph = dict(
        nodes=[
            dict(id=str(i), task_type="E8", call=value) for i, value in enumerate(calls)
        ],
        task_groups=[dict(task_type="E8", node_ids=["0", "1"])],
    )
    before = deepcopy(source)
    result = prepare_source_physical_input(
        source, graph, robot_profile="dual_franka", cache_root=tmp_path / "physical"
    )
    physical = result.physical
    physical.assert_unchanged()
    assert source == before
    assert "twist_adaptation.json" in result.artifacts
    assert "twist_mass_adaptation.json" in result.artifacts
    assert any(name.endswith(".mass.json") for name in physical.input_files)
    runtime = physical.scene_simulation["articulation"][0]
    assert runtime["fpath"] != art["fpath"]
    assert "body_scale_mass_policy" not in runtime
    assert runtime["asset_physics_mode"] == "overlay"
    assert runtime["init_pos"][2] == pytest.approx(0.801)
    receipt = result.artifacts["twist_adaptation.json"]["records"][0]
    assert receipt["scale_policy"] == "preserve_source"
    assert receipt["support_clearance"]["requested_total_lift_m"] == 0.001


def test_source_press_preparation_retains_unit_scale_mass_and_release_bias(
    tmp_path, monkeypatch
):
    from sim.scene.test_control_inputs import button as button_fixture
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    source, _ = fixture(tmp_path)
    art = button_fixture.__wrapped__(tmp_path)
    stage = Usd.Stage.Open(art["fpath"])
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetDefaultPrim())
    edit = Sdf.BatchNamespaceEdit()
    edit.Add("/Button/cap/contact", "/Button/cap/cap_contact")
    assert stage.GetRootLayer().Apply(edit)
    UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/Button/cap")).CreateMassAttr(0.02)
    stage.GetRootLayer().Save()
    art.update(init_pos=[0.6, 0.0, 0.8], init_rot=[0.0, 0.0, 0.0])
    source = replace(
        source,
        rigid_objects=(),
        articulations=(art,),
        planner_objects=(
            source.planner_objects[0],
            dict(art, runtime_uid="button", source_uid="button", role="articulation"),
        ),
    )
    calls = [
        dict(
            kind="registered",
            call_id=identity,
            arguments=dict(object="button"),
            resources=dict(primary="left"),
        )
        for identity in ("gen_sim.press_prepare.operation", "gen_sim.press.operation")
    ]
    graph = dict(
        nodes=[
            dict(id=str(i), task_type="E9", call=value) for i, value in enumerate(calls)
        ],
        task_groups=[dict(id="operation", task_type="E9", node_ids=["0", "1"])],
    )
    result = prepare_source_physical_input(
        source, graph, robot_profile="dual_franka", cache_root=tmp_path / "physical"
    )
    runtime = result.physical.scene_simulation["articulation"][0]
    assert runtime["body_scale"] == [1.0, 1.0, 1.0]
    assert runtime["link_attrs"]["gen_sim_press_density"]["attrs"]["mass_props"][
        "mass"
    ] == pytest.approx(0.16, abs=1e-7)
    assert runtime["joint_drive_props"]["target_mode"] == {
        "press_joint": "position_velocity"
    }
    receipt = result.artifacts["press_calibration.json"]["routes"][0]
    assert (
        receipt["runtime_mass"] == pytest.approx(0.16, abs=1e-7)
        and receipt["binding"]["object_id"] == "button"
    )
    result.physical.assert_unchanged()


def test_source_preparation_rejects_file_drift_during_derived_asset_stage(
    tmp_path, monkeypatch
):
    from embodichain.gen_sim.task_engine import source_physical_input as module

    source, graph = fixture(tmp_path)
    original = module.normalize_prepared_scene_assets

    def drift(scene, cache):
        result = original(scene, cache)
        source.source_config_path.write_text("changed while preparing")
        return result

    monkeypatch.setattr(module, "normalize_prepared_scene_assets", drift)
    with pytest.raises(ValueError, match="changed during preparation"):
        prepare_source_physical_input(
            source, graph, robot_profile="dual_franka", cache_root=tmp_path / "cache"
        )


def test_prepared_source_asset_digest_cannot_be_resigned_after_file_change(tmp_path):
    scene, graph = fixture(tmp_path)
    path = Path(scene.rigid_objects[0]["shape"]["fpath"])
    trimesh.creation.box(extents=[0.07, 0.08, 0.18]).export(path)
    with pytest.raises(ValueError, match="asset changed since scene preparation"):
        prepare_source_physical_input(
            scene, graph, robot_profile="dual_franka", cache_root=tmp_path / "new-cache"
        )


@pytest.mark.parametrize("selector", ["tianji_marvin", "dual_tianji"])
def test_normal_source_preparation_selects_existing_native_marvin_without_other_tool_assembly(
    tmp_path, monkeypatch, selector
):
    import yaml
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.toolkits.graspkit import resolve_parallel_jaw_gripper_model

    def forbidden(*args, **kwargs):
        pytest.fail("static source preparation created a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    scene, graph = fixture(tmp_path)
    component = (
        Path(__file__).parents[3]
        / "extensions/embodichain_tasks/configs/components/embodiments/tianji_marvin.yaml"
    )
    original = yaml.safe_load(component.read_text())
    prepared = prepare_source_physical_input(
        scene,
        graph,
        robot_profile=selector,
        cache_root=tmp_path / "derived",
        fit_grasp_assets=True,
    )
    physical = prepared.physical
    physical.assert_unchanged()
    embodiment = physical.embodiment
    simulation = embodiment["simulation"]
    assert embodiment["embodiment_id"] == "tianji_marvin"
    assert (
        simulation["class_type"] == "TianjiMarvin"
        and simulation["with_gripper"] is True
    )
    assert Path(simulation["fpath"]).name == "robot_with_ee_acd.urdf"
    assert simulation["init_qpos"] == original["simulation"]["init_qpos"]
    for part in ("left_eef", "right_eef"):
        assert simulation["joint_drive_props"]["max_effort"][part] == 40.0
        generator = embodiment["skill_profile"]["runtime_services"][
            "grasp_pose_generators"
        ][part]
        assert (
            resolve_parallel_jaw_gripper_model(generator["model"]).max_opening_width
            == 0.095
        )
    assert not any("Robotiq" in name for name in physical.input_files)
    assert not (tmp_path / "derived/robot_model").exists()
    assert physical.robot_profile == "tianji_marvin"
