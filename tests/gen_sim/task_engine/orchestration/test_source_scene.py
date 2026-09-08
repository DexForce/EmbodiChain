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

"""Tests for Task Engine source-scene normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine.orchestration.source_scene import (
    prepare_scene,
    resolve_gym_config_path,
    resolve_source_scene,
)


@pytest.fixture
def gym_export(tmp_path: Path) -> Path:
    """Create a minimal legacy Prompt2Scene export."""
    export = tmp_path / "gym_export"
    assets = export / "mesh_assets"
    assets.mkdir(parents=True)
    (assets / "table.glb").write_bytes(b"not-a-real-glb")
    (assets / "can.glb").write_bytes(b"not-a-real-glb")
    scene_state = export / "scene_state"
    scene_state.mkdir()
    (scene_state / "result.json").write_text("{}\n", encoding="utf-8")
    config = {
        "id": "Prompt2Scene-test-v0",
        "env": {"events": {}, "observations": {}, "dataset": {}},
        "robot": {},
        "sensor": [],
        "light": {},
        "background": [
            {
                "uid": "table_0",
                "description": "A white table.",
                "shape": {"shape_type": "Mesh", "fpath": "mesh_assets/table.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "rigid_object": [
            {
                "uid": "interact_can_0",
                "description": "A red soda can.",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": "mesh_assets/can.glb",
                    "acd_method": "coacd",
                    "max_convex_hull_num": 32,
                },
                "attrs": {"mass": 0.01},
                "init_pos": [1.0, 2.0, 0.7],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
                "max_convex_hull_num": 32,
            }
        ],
    }
    (export / "gym_config.json").write_text(json.dumps(config), encoding="utf-8")
    return export


@pytest.fixture
def scene_export(tmp_path: Path) -> Path:
    """Create a minimal canonical scene export."""
    export = tmp_path / "scene_export"
    assets = export / "mesh_assets"
    assets.mkdir(parents=True)
    (assets / "table.glb").write_bytes(b"not-a-real-glb")
    for uid in ("bottle_001", "bottle_002"):
        (assets / f"{uid}.glb").write_bytes(b"not-a-real-glb")
    config = {
        "format": "embodichain.scene-export/v1",
        "scene_id": "scene-export-test",
        "background": [
            {
                "uid": "table",
                "description": "A white table.",
                "shape": {"shape_type": "Mesh", "fpath": "mesh_assets/table.glb"},
                "init_pos": [0.0, 0.0, 0.0],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        ],
        "rigid_object": [
            {
                "uid": uid,
                "name": f"Bottle {index}",
                "description": f"Bottle instance {index}.",
                "shape": {
                    "shape_type": "Mesh",
                    "fpath": f"mesh_assets/{uid}.glb",
                },
                "init_pos": [float(index), float(index + 1), 0.7],
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
            for index, uid in enumerate(("bottle_001", "bottle_002"), start=1)
        ],
    }
    (export / "scene_config.json").write_text(json.dumps(config), encoding="utf-8")
    return export


def test_prepare_scene_normalizes_prompt2scene_export(gym_export: Path) -> None:
    scene = prepare_scene(gym_export)

    assert scene.uid_map == {"table_0": "table", "interact_can_0": "interact_can"}
    assert scene.z_rotation_degrees == -90.0
    assert scene.rigid_objects[0]["init_pos"] == [2.0, -1.0, 0.7]
    assert scene.rigid_objects[0]["max_convex_hull_num"] == 16
    assert scene.rigid_objects[0]["acd_method"] == "vhacd"
    assert scene.rigid_objects[0]["shape"]["acd_method"] == "vhacd"
    assert scene.rigid_objects[0]["shape"]["max_convex_hull_num"] == 16
    assert Path(scene.rigid_objects[0]["shape"]["fpath"]).is_file()
    assert scene.planner_objects[1]["source_uid"] == "interact_can_0"
    assert scene.planner_objects[1]["uid"] == "interact_can"


def test_prepare_scene_supports_scene_export_v1(scene_export: Path) -> None:
    scene = prepare_scene(scene_export.parent)

    assert scene.source_config_path == scene_export / "scene_config.json"
    assert scene.uid_map == {
        "table": "table",
        "bottle_001": "bottle_001",
        "bottle_002": "bottle_002",
    }
    assert scene.planner_objects[1]["name"] == "Bottle 1"
    assert scene.z_rotation_degrees == -90.0
    assert scene.rigid_objects[0]["init_pos"] == [2.0, -1.0, 0.7]


def test_prepare_scene_requires_exactly_one_background(gym_export: Path) -> None:
    source_path = gym_export / "gym_config.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["background"].append(
        {
            **source["background"][0],
            "uid": "floor_0",
            "description": "A floor beneath the work surface.",
        }
    )
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one background"):
        prepare_scene(gym_export)


def test_prepare_scene_does_not_treat_physics_attrs_as_semantics(
    gym_export: Path,
) -> None:
    scene = prepare_scene(gym_export)
    rigid_object = next(
        item for item in scene.planner_objects if item["role"] == "rigid_object"
    )

    assert rigid_object["attributes"] == {}


@pytest.mark.parametrize(
    "companion_relative_path",
    (Path("gym_export/scene_config.json"), Path("scene_export/scene_config.json")),
)
def test_source_resolution_prefers_gym_config_in_mixed_export(
    tmp_path: Path,
    companion_relative_path: Path,
) -> None:
    gym_export = tmp_path / "gym_export"
    gym_export.mkdir(parents=True)
    gym_config = gym_export / "gym_config.json"
    gym_config.write_text("{}", encoding="utf-8")
    companion = tmp_path / companion_relative_path
    companion.parent.mkdir(parents=True, exist_ok=True)
    companion.write_text(
        json.dumps({"format": "embodichain.scene-export/v1"}), encoding="utf-8"
    )

    resolved = resolve_source_scene(tmp_path)

    assert resolved.path == gym_config
    assert resolved.source_format == "legacy_gym_config"
    assert resolved.is_prompt2scene is True
    assert resolve_gym_config_path(tmp_path) == resolved.path


def test_explicit_scene_export_overrides_mixed_layout(
    gym_export: Path,
    scene_export: Path,
) -> None:
    resolved = resolve_source_scene(scene_export / "scene_config.json")

    assert resolved.path == scene_export / "scene_config.json"
    assert resolved.source_format == "embodichain.scene-export/v1"
    assert resolved.is_prompt2scene is True


def test_explicit_named_legacy_config_is_supported(gym_export: Path) -> None:
    config_path = gym_export / "official_task_config.json"
    config_path.write_text(
        (gym_export / "gym_config.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    resolved = resolve_source_scene(config_path)
    scene = prepare_scene(config_path)

    assert resolved.path == config_path
    assert resolved.source_format == "legacy_gym_config"
    assert resolved.is_prompt2scene is False
    assert scene.source_config_path == config_path


def test_explicit_robot_scene_is_centered_on_table(gym_export: Path) -> None:
    source = json.loads((gym_export / "gym_config.json").read_text(encoding="utf-8"))
    source["robot"] = {"uid": "source_robot"}
    source["background"][0]["init_pos"] = [1.0, 2.0, 0.0]
    source["rigid_object"][0]["init_pos"] = [1.2, 2.3, 0.7]
    config_path = gym_export / "official_task_config.json"
    config_path.write_text(json.dumps(source), encoding="utf-8")

    scene = prepare_scene(config_path)

    assert scene.source_scene_xy_translation == pytest.approx((-1.0, -2.0))
    assert scene.background[0]["init_pos"][:2] == pytest.approx([0.0, 0.0])
    assert scene.rigid_objects[0]["init_pos"][:2] == pytest.approx([0.2, 0.3])


def test_scene_export_rejects_unknown_format(scene_export: Path) -> None:
    config_path = scene_export / "scene_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["format"] = "embodichain.scene-export/v2"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported format"):
        resolve_source_scene(config_path)
