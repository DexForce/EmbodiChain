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
from pathlib import Path
import pytest
from embodichain.gen_sim.task_engine.orchestration.source_scene import (
    prepare_scene,
    resolve_source_scene,
    resolve_gym_config_path,
    is_prompt2scene_export,
)


@pytest.mark.parametrize("ids", [["other"], ["cup", "cup"], ["cup"]])
def test_scene_export_rejects_mixed_companion_ids(
    tmp_path: Path, ids: list[str]
) -> None:
    path = tmp_path / "scene_config.json"
    path.write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "rigid_object": [{"uid": "cup"}],
            }
        )
    )
    (tmp_path / "scene.json").write_text(
        json.dumps({"objects": [{"id": uid} for uid in ids]})
    )
    if ids == ["cup"]:
        assert resolve_source_scene(path).path == path
    else:
        with pytest.raises(ValueError, match="companion"):
            resolve_source_scene(path)


@pytest.mark.parametrize("field", ["category", "name", "description", "is_articulated"])
def test_scene_export_rejects_companion_semantic_drift(
    tmp_path: Path, field: str
) -> None:
    path = tmp_path / "scene_config.json"
    path.write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "rigid_object": [{"uid": "cup", field: "runtime"}],
            }
        )
    )
    (tmp_path / "scene.json").write_text(
        json.dumps({"objects": [{"id": "cup", field: "different"}]})
    )
    with pytest.raises(ValueError, match=f"companion {field} differs"):
        resolve_source_scene(path)


def test_source_preparation_retains_semantic_view_and_shared_physical_defaults(
    tmp_path: Path,
) -> None:
    config = {
        "format": "embodichain.scene-export/v1",
        "background": [
            {"uid": "table_0", "shape": {"shape_type": "Cube", "size": [1, 1, 0.1]}}
        ],
        "rigid_object": [
            {
                "uid": "cup_0",
                "name": "red cup",
                "description": "a semantic cup",
                "shape": {"shape_type": "Cube", "size": [0.05, 0.05, 0.1]},
                "attrs": {"mass": 0.4},
            }
        ],
    }
    path = tmp_path / "scene_config.json"
    path.write_text(json.dumps(config))
    result = prepare_scene(path)
    assert result.uid_map == {"table_0": "table", "cup_0": "cup"}
    assert result.planner_objects[1]["description"] == "a semantic cup"
    assert result.planner_objects[1]["source_uid"] == "cup_0"
    assert "attrs" not in result.planner_objects[1]
    assert result.background[0]["attrs"]["mass_props"]["mass"] == 10.0
    assert result.rigid_objects[0]["attrs"]["mass_props"]["mass"] == 0.4
    assert result.rigid_objects[0]["attrs"]["collision_props"] == {
        "contact_offset": 0.003,
        "rest_offset": 0.001,
    }
    assert json.loads(path.read_text()) == config


def test_prepared_scene_preserves_frozen_mesh_input_and_detects_later_asset_drift(
    tmp_path, monkeypatch
):
    import hashlib
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

    def forbidden(*args, **kwargs):
        pytest.fail("source import constructed a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    test_root = Path(__file__).parents[3]
    fixture = test_root / "sim/scene/fixtures/source_ingress_mesh.glb"
    frozen = json.loads(
        (test_root / "sim/scene/data/source_mesh_inputs_b0.json").read_text()
    )
    asset = tmp_path / "table.glb"
    asset.write_bytes(fixture.read_bytes())
    config = {
        "format": "embodichain.scene-export/v1",
        "background": [
            {
                "uid": "table_0",
                "shape": {"shape_type": "Mesh", "fpath": "table.glb"},
                "init_pos": [1.0, 2.0, 0.7],
            }
        ],
        "rigid_object": [
            {
                "uid": "cup_0",
                "description": "semantic cup",
                "shape": {"shape_type": "Cube", "size": [0.1, 0.1, 0.2]},
                "init_pos": [0.2, 0.3, 0.6],
            }
        ],
    }
    path = tmp_path / "scene_config.json"
    path.write_text(json.dumps(config))
    original = path.read_bytes()
    scene = prepare_scene(
        path,
        z_rotation_degrees=-90.0,
        body_scale_policy="multiply",
        body_scale=[2.0, 1.0, 1.0],
    )
    assert scene.uid_map == {"table_0": "table", "cup_0": "cup"}
    assert scene.background[0]["init_pos"] == [2.0, -1.0, 0.7]
    assert scene.rigid_objects[0]["init_pos"] == [0.3, -0.2, 0.6]
    assert scene.planner_objects[1]["description"] == "semantic cup"
    assert (
        scene.asset_hashes["table"]
        == hashlib.sha256(asset.read_bytes()).hexdigest()
        == frozen["fixture_sha256"]
    )
    assert scene.table_top_z == 0.7 + frozen["cases"][0]["expected_top_z"]
    assert path.read_bytes() == original and asset.read_bytes() == fixture.read_bytes()
    scene.assert_source_assets()
    asset.write_bytes(asset.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed since scene preparation"):
        scene.assert_source_assets()


def test_source_preflight_rejects_invalid_transform_contract_before_reading_files(
    tmp_path,
):
    with pytest.raises(ValueError, match="body_scale_policy"):
        prepare_scene(tmp_path / "missing", body_scale_policy="unsupported")
    with pytest.raises(ValueError, match="body_scale values must be positive"):
        prepare_scene(tmp_path / "missing", body_scale=[1, -1, 1])


def test_prepared_scene_consumes_qualified_authored_revolute_declarations(
    tmp_path, monkeypatch
):
    import hashlib
    import math
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.scene.control_semantics import InvalidTwistSemanticsError

    def forbidden(*args, **kwargs):
        pytest.fail("source import constructed a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    test_root = Path(__file__).parents[3]
    monkeypatch.syspath_prepend(str(test_root / "sim/scene"))
    from _control_source_fixture import knob_source

    articulation = knob_source(tmp_path)
    source = Path(articulation["fpath"])
    original = source.read_bytes()
    sidecar = source.with_suffix(".twist.json")
    config = {
        "format": "embodichain.scene-export/v1",
        "world_alignment_applied": True,
        "background": [{"uid": "bench", "shape": {"shape_type": "Cube"}}],
        "articulation": [dict(articulation, uid="control_0")],
    }
    path = tmp_path / "scene_config.json"
    path.write_text(json.dumps(config))
    scene = prepare_scene(path)
    assert scene.uid_map == {"bench": "table", "control_0": "control"}
    attributes = scene.planner_objects[1]["attributes"]
    assert attributes["twist_setting_labels"] == pytest.approx(
        {"0": 0, "90": -math.pi / 2}
    )
    assert attributes["joint_settings"]["knob_rotation"] == pytest.approx(
        [0, -math.pi / 2]
    )
    assert attributes["twist_control_joint"] == "knob_rotation"
    assert attributes["twist_source_sha256"] == hashlib.sha256(original).hexdigest()
    assert scene.articulations[0]["fpath"] == str(source.resolve())
    assert source.read_bytes() == original and json.loads(path.read_text()) == config
    annotation = sidecar.read_text()
    sidecar.write_text(
        annotation.replace("/Control/knob/grip", "/Control/knob/missing")
    )
    with pytest.raises(InvalidTwistSemanticsError):
        prepare_scene(path)
    sidecar.unlink()
    assert (
        "twist_setting_labels"
        not in prepare_scene(path).planner_objects[1]["attributes"]
    )


@pytest.mark.parametrize(
    "filename,alignment,expected_prompt",
    [
        ("gym_config_merged.json", None, False),
        ("scene_config.json", False, True),
        ("scene_config.json", True, False),
    ],
)
def test_source_discovery_public_aliases_preserve_export_alignment(
    tmp_path, filename, alignment, expected_prompt
):
    export = tmp_path / "scene_export"
    export.mkdir()
    path = export / filename
    config = {"background": [{"uid": "table"}]}
    if alignment is not None:
        config.update(
            format="embodichain.scene-export/v1", world_alignment_applied=alignment
        )
    path.write_text(json.dumps(config))
    result = resolve_source_scene(tmp_path)
    assert result.path == path and result.is_prompt2scene is expected_prompt
    assert result.source_format == (
        "legacy_gym_config" if alignment is None else "embodichain.scene-export/v1"
    )
    assert resolve_gym_config_path(tmp_path) == path
    assert is_prompt2scene_export(tmp_path) is expected_prompt


def test_source_discovery_rejects_ambiguous_nested_exports(tmp_path):
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "gym_config.json").write_text(json.dumps({"background": []}))
    with pytest.raises(ValueError, match="Multiple exported scene configs"):
        resolve_source_scene(tmp_path)
    assert is_prompt2scene_export(tmp_path) is False
