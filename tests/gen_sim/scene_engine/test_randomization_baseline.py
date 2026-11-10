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
import shutil

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from embodichain.gen_sim.scene_engine.randomization import baseline
from embodichain.gen_sim.scene_engine.randomization.cfg import SceneRandomizationCfg


def _export(tmp_path: Path) -> tuple[Path, SceneRandomizationCfg]:
    root = tmp_path / "scene_export"
    root.mkdir()
    table = trimesh.creation.box(extents=[2.0, 0.1, 1.0])
    table.apply_translation([0.0, -0.05, 0.0])
    entries = []
    for uid, pos, rot, parent in (
        ("table", [0, 0, 0], [0, 0, 0], None),
        ("tray", [0.2, 0.1, 0.03], [0, 0, 30], "table"),
        ("cup", [0.3, 0.1, 0.08], [10, 5, 20], "tray"),
        ("block", [0.3, 0.1, 0.14], [0, 0, 0], "cup"),
        ("bottle", [-0.4, 0, 0.2], [0, 0, 0], "table"),
    ):
        mesh = (
            table.copy()
            if uid == "table"
            else trimesh.creation.box(extents=[0.12, 0.1, 0.08])
        )
        if uid == "bottle":
            mesh.apply_translation([0.03, 0, 0])
        path = root / "mesh_assets" / uid / f"{uid}.glb"
        path.parent.mkdir(parents=True)
        mesh.export(path)
        entry = {
            "uid": uid,
            "is_articulated": False,
            "shape": {"shape_type": "Mesh", "fpath": path.relative_to(root).as_posix()},
            "body_type": "kinematic" if uid == "table" else "dynamic",
            "init_pos": pos,
            "init_rot": rot,
            "body_scale": [1.0, 1.0, 1.0],
        }
        if uid == "table":
            entry.update(
                support_surface_z=0.0,
                support_optimization_rect_xy=[
                    [-0.9, -0.4],
                    [0.9, -0.4],
                    [0.9, 0.4],
                    [-0.9, 0.4],
                ],
            )
        entries.append((entry, parent))
    config = {
        "format": "embodichain.scene-export/v1",
        "scene_id": "fixture",
        "background": [entries[0][0]],
        "rigid_object": [entry for entry, _ in entries[1:]],
        "articulation": [],
    }
    graph = {
        "nodes": [
            {
                "object_id": entry["uid"],
                "parent_id": parent,
                "parent_relation": "on" if parent is not None else None,
                "table_region": None,
                "pose_description": None,
            }
            for entry, parent in entries
        ],
        "relations": [],
    }
    (root / "scene_config.json").write_text(json.dumps(config))
    (root / "scene_graph.json").write_text(json.dumps(graph))
    (root / "scene.json").write_text('{"debug_snapshot": "preserve exactly"}')
    return root, SceneRandomizationCfg(
        scene_path=root, base_data_path=root / "scene_randomization_base.yaml"
    )


def _pose(value: dict[str, list[float]]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(value["quaternion_xyzw"]).as_matrix()
    matrix[:3, 3] = value["position"]
    return matrix


def test_prepare_groups_nested_members_and_preserves_sources(tmp_path: Path) -> None:
    root, cfg = _export(tmp_path)
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }
    result = baseline.prepare_scene_baseline(cfg)
    assert not result.cache_hit
    assert {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before} == before
    assert yaml.safe_load(result.path.read_text()) == result.data
    assert [group["root_id"] for group in result.data["groups"]] == ["bottle", "tray"]
    bottle, tray = result.data["groups"]
    assert bottle["members"] == []
    assert bottle["aabb_2d"]["center_xy"] != bottle["root_pose"]["position"][:2]
    assert [(m["entity_id"], m["parent_id"]) for m in tray["members"]] == [
        ("block", "cup"),
        ("cup", "tray"),
    ]
    raw = json.loads((root / "scene_config.json").read_text())
    by_uid = {entry["uid"]: entry for entry in raw["rigid_object"]}
    root_pose = _pose(tray["root_pose"])
    for member in tray["members"]:
        actual = root_pose @ _pose(member["pose_in_root"])
        expected = baseline._entry_pose(by_uid[member["entity_id"]])
        np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_cache_hit_avoids_mesh_parsing_and_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cfg = _export(tmp_path)
    first = baseline.prepare_scene_baseline(cfg)
    original_mtime = first.path.stat().st_mtime_ns

    def no_measure(*args: object) -> None:
        raise AssertionError("Cache reuse must not parse geometry")

    monkeypatch.setattr(baseline, "measure_scene_geometry", no_measure)
    second = baseline.prepare_scene_baseline(cfg)
    assert second.cache_hit and second.data == first.data
    assert second.path.stat().st_mtime_ns == original_mtime


@pytest.mark.parametrize("change", ["pose", "scale", "mesh", "graph"])
def test_source_changes_invalidate_cache(tmp_path: Path, change: str) -> None:
    root, cfg = _export(tmp_path)
    first = baseline.prepare_scene_baseline(cfg)
    config_path = root / "scene_config.json"
    config = json.loads(config_path.read_text())
    if change in {"pose", "scale"}:
        bottle = config["rigid_object"][-1]
        bottle["init_pos" if change == "pose" else "body_scale"][0] += 0.1
        config_path.write_text(json.dumps(config))
    elif change == "mesh":
        mesh = trimesh.creation.box(extents=[0.2, 0.1, 0.1])
        mesh.export(root / "mesh_assets/bottle/bottle.glb")
    else:
        path = root / "scene_graph.json"
        graph = json.loads(path.read_text())
        next(node for node in graph["nodes"] if node["object_id"] == "cup")[
            "parent_id"
        ] = "table"
        path.write_text(json.dumps(graph))
    second = baseline.prepare_scene_baseline(cfg)
    assert not second.cache_hit
    assert (
        second.data["source_scene"]["content_sha256"]
        != first.data["source_scene"]["content_sha256"]
    )


def test_relocated_package_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, cfg = _export(tmp_path)
    baseline.prepare_scene_baseline(cfg)
    moved = tmp_path / "moved" / "scene_export"
    shutil.copytree(root, moved)
    monkeypatch.setattr(
        baseline,
        "measure_scene_geometry",
        lambda *args: pytest.fail("Portable cache reparsed meshes"),
    )
    result = baseline.prepare_scene_baseline(
        SceneRandomizationCfg(
            scene_path=moved, base_data_path=moved / cfg.base_data_path.name
        )
    )
    assert result.cache_hit
    assert result.data["source_scene"]["export_dir"] == "."


@pytest.mark.parametrize(
    "corruption", ["syntax", "geometry", "unknown_field", "duplicate_key"]
)
def test_invalid_cache_is_rebuilt(tmp_path: Path, corruption: str) -> None:
    _, cfg = _export(tmp_path)
    first = baseline.prepare_scene_baseline(cfg)
    if corruption == "syntax":
        first.path.write_text("groups: [")
    elif corruption == "duplicate_key":
        first.path.write_text(first.path.read_text() + "schema_version: 1\n")
    else:
        value = first.data
        if corruption == "geometry":
            value["groups"][0]["aabb_2d"]["center_xy"][0] += 1
        else:
            value["unknown"] = True
        first.path.write_text(yaml.safe_dump(value))
    assert not baseline.prepare_scene_baseline(cfg).cache_hit


@pytest.mark.parametrize(
    "invalid",
    [
        "format",
        "duplicate_uid",
        "articulation",
        "invalid_parent_relation",
        "scale",
        "extra_background",
        "shape_type",
        "unsupported_section",
    ],
)
def test_invalid_sources_fail_before_persistence(tmp_path: Path, invalid: str) -> None:
    root, cfg = _export(tmp_path)
    path = root / "scene_config.json"
    value = json.loads(path.read_text())
    if invalid == "format":
        value["format"] = "other/v1"
    elif invalid == "duplicate_uid":
        value["rigid_object"].append(value["rigid_object"][0])
    elif invalid == "articulation":
        value["articulation"] = [{"uid": "unsupported"}]
    elif invalid == "scale":
        value["rigid_object"][0]["body_scale"] = [1, True, 1]
    elif invalid == "extra_background":
        value["background"].append(
            {"uid": "other", "shape": {"fpath": "../outside.glb"}}
        )
    elif invalid == "shape_type":
        value["rigid_object"][0]["shape"]["shape_type"] = "Box"
    elif invalid == "unsupported_section":
        value["deformable"] = []
    else:
        graph_path = root / "scene_graph.json"
        graph = json.loads(graph_path.read_text())
        graph["nodes"][1]["parent_relation"] = None
        graph_path.write_text(json.dumps(graph))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        baseline.prepare_scene_baseline(cfg)
    assert not cfg.base_data_path.exists()


def test_atomic_failure_preserves_previous_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, cfg = _export(tmp_path)
    first = baseline.prepare_scene_baseline(cfg)
    contents = first.path.read_bytes()

    def fail_replace(*args: object) -> None:
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(baseline.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted write"):
        baseline.prepare_scene_baseline(cfg, rebuild=True)
    assert first.path.read_bytes() == contents
    assert not list(first.path.parent.glob(".*.tmp"))


def test_destination_cannot_replace_asset_or_symlink(tmp_path: Path) -> None:
    root, cfg = _export(tmp_path)
    cfg.base_data_path = root / "mesh_assets/unrelated.yaml"
    with pytest.raises(ValueError, match="source asset directories"):
        baseline.prepare_scene_baseline(cfg)
    target = tmp_path / "keep.yaml"
    target.write_text("keep: true")
    cfg.base_data_path = root / "linked.yaml"
    cfg.base_data_path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        baseline.prepare_scene_baseline(cfg)
    assert target.read_text() == "keep: true"


def test_cli_prepares_workspace_baseline_then_reuses_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from embodichain.cli.main import main

    export_root, _ = _export(tmp_path)
    config_path = tmp_path / "scene_randomization.yaml"
    config_path.write_text("schema_version: 1\n")
    arguments = [
        "scene-randomization",
        "--output_root",
        str(tmp_path),
        "--config",
        str(config_path),
    ]

    main([*arguments, "--rebuild"])

    output_path = export_root / "scene_randomization_base.yaml"
    data = yaml.safe_load(output_path.read_text())
    assert {group["root_id"] for group in data["groups"]} == {"bottle", "tray"}
    assert "Groups: 2" in capsys.readouterr().out
    first_run = tmp_path / "scene_layouts/run_001"
    first_manifest = json.loads((first_run / "run_result.json").read_text())
    assert first_manifest["generated_count"] == 10
    assert len(list((first_run / "layouts").glob("candidate_*.json"))) == 10
    first_saved = (first_run / "run_result.json").read_bytes()
    original = (output_path.read_bytes(), output_path.stat().st_mtime_ns)

    main(arguments)

    assert "Cache: reused" in capsys.readouterr().out
    assert (output_path.read_bytes(), output_path.stat().st_mtime_ns) == original
    assert (first_run / "run_result.json").read_bytes() == first_saved
    assert (tmp_path / "scene_layouts/run_002/run_result.json").is_file()
