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

import json
from copy import deepcopy
from pathlib import Path
import pytest
import trimesh
import yaml
from embodichain.gen_sim.task_engine.physical_input import load_frozen_physical_input
from embodichain.gen_sim.task_engine.orchestration.source_scene import PreparedScene


@pytest.fixture(autouse=True)
def no_runtime_construction(monkeypatch):
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.objects import Robot

    def forbidden(*args, **kwargs):
        pytest.fail("frozen declaration import constructed a native runtime")

    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    monkeypatch.setattr(Robot, "__init__", forbidden)


def test_frozen_input_owns_selected_physics_and_rejects_data_or_file_drift(
    tmp_path, monkeypatch
):
    components = tmp_path / "components"
    components.mkdir()
    mesh = tmp_path / "object.glb"
    mesh.write_bytes(trimesh.creation.box().export(file_type="glb"))
    objects = [
        {
            "uid": "table",
            "init_pos": [0, 0, 0],
            "init_rot": [0, 0, 0],
            "shape": {"shape_type": "Cube", "size": [1, 1, 1]},
        },
        {
            "uid": "object",
            "init_pos": [1, 2, 3],
            "init_rot": [0, 0, 0],
            "shape": {"shape_type": "Mesh", "fpath": str(mesh)},
        },
    ]
    simulation = {"background": [objects[0]], "rigid_object": [objects[1]]}
    values = {
        "components/scene.yaml": {"scene_id": "scene", "simulation": simulation},
        "components/embodiment.yaml": {
            "embodiment_id": "robot",
            "simulation": {"uid": "robot"},
            "sensor": [],
        },
        "components/execution_policy.yaml": {"policy_id": "frozen-policy"},
        "task_program_deployment.yaml": {
            "id": "test",
            "physics": "default",
            "env": {},
            "scene": {"component": "components/scene.yaml"},
            "embodiment": {"component": "components/embodiment.yaml"},
            "task_program": {
                "program": "program.yaml",
                "integration": "integration.yaml",
                "execution_policy": "components/execution_policy.yaml",
            },
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(yaml.safe_dump(value))
    scene = PreparedScene(
        tmp_path / "source.json",
        tmp_path,
        tuple(
            {
                "runtime_uid": item["uid"],
                "source_uid": item["uid"],
                "role": role,
                "description": "source-label",
            }
            for item, role in zip(objects, ["background", "rigid_object"])
        ),
        (),
        (),
        (),
        {},
        0.5,
        0.0,
        "preserve",
        (1.0, 1.0, 1.0),
        {},
    )
    frozen = load_frozen_physical_input(scene, tmp_path)
    assert frozen.scene.planner_objects[1]["description"] == "source-label"
    assert frozen.scene.planner_objects[1]["init_pos"] == [1, 2, 3]
    assert frozen.scene.table_top_z == 0.5
    frozen.assert_unchanged()
    altered = deepcopy(frozen)
    altered.scene_simulation["rigid_object"][0]["init_pos"][0] = 9
    with pytest.raises(ValueError, match="data changed"):
        altered.assert_unchanged()
    assert frozen.scene_simulation["rigid_object"][0]["init_pos"] == [1, 2, 3]
    (tmp_path / "physical_input.json").write_text(json.dumps(frozen.manifest()))
    extra = tmp_path / "new-sidecar.json"
    extra.write_text("{}")
    from embodichain.gen_sim.task_engine import physical_input

    original_assets = physical_input._asset_files
    with monkeypatch.context() as patch:
        patch.setattr(
            physical_input,
            "_asset_files",
            lambda value, **kwargs: original_assets(value, **kwargs) | {extra},
        )
        with pytest.raises(ValueError, match="new asset dependencies"):
            load_frozen_physical_input(scene, tmp_path)
    mesh.write_bytes(mesh.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="file changed"):
        frozen.assert_unchanged()


def test_current_component_producer_freezes_complete_articulation_input_set(tmp_path):
    import hashlib
    from embodichain.gen_sim.task_engine.orchestration.source_scene import prepare_scene
    from embodichain.lab.sim.scene.component_declarations import (
        source_scene_component_declaration,
    )

    fixture = (
        Path(__file__).parents[2] / "sim/scene/fixtures/source_declaration_drawer.usda"
    )
    asset = tmp_path / "cabinet.usda"
    asset.write_bytes(fixture.read_bytes())
    config = {
        "background": [
            {"uid": "bench", "shape": {"shape_type": "Cube", "size": [1, 1, 0.1]}}
        ],
        "articulation": [
            {
                "uid": "cabinet",
                "fpath": "cabinet.usda",
                "fix_base": True,
                "body_scale": [2, 2, 2],
                "init_pos": [1, 2, 3],
                "init_rot": [0, 0, 90],
            }
        ],
    }
    source = tmp_path / "source.json"
    source.write_text(json.dumps(config))
    scene = prepare_scene(source, z_rotation_degrees=0)
    components = tmp_path / "components"
    components.mkdir()
    values = {
        "components/scene.yaml": source_scene_component_declaration(
            "source",
            background=scene.background,
            rigid_objects=scene.rigid_objects,
            articulations=scene.articulations,
        ),
        "components/embodiment.yaml": {
            "embodiment_id": "robot",
            "simulation": {"uid": "robot"},
            "sensor": [],
        },
        "components/execution_policy.yaml": {"policy_id": "frozen-policy"},
        "task_program_deployment.yaml": {
            "id": "test",
            "physics": "default",
            "env": {},
            "scene": {"component": "components/scene.yaml"},
            "embodiment": {"component": "components/embodiment.yaml"},
            "task_program": {
                "program": "program.yaml",
                "integration": "integration.yaml",
                "execution_policy": "components/execution_policy.yaml",
            },
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(yaml.safe_dump(value))
    frozen = load_frozen_physical_input(scene, tmp_path)
    expected = {str((tmp_path / name).resolve()) for name in values} | {
        str(source),
        str(asset),
    }
    assert set(frozen.input_files) == expected
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    assert (
        frozen.input_files[str(asset)] == frozen.scene.asset_hashes["cabinet"] == digest
    )
    assert frozen.scene.planner_objects[1]["_articulation_part_positions"]
    frozen.assert_unchanged()
    (tmp_path / "physical_input.json").write_text(json.dumps(frozen.manifest()))
    sidecar = asset.with_suffix(".twist.json")
    sidecar.write_text("{}")
    with pytest.raises(ValueError, match="new asset dependencies"):
        load_frozen_physical_input(scene, tmp_path)
    sidecar.unlink()
    asset.write_bytes(asset.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file changed"):
        frozen.assert_unchanged()
