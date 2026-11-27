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

from dataclasses import replace
import json

import numpy as np
import pytest
import trimesh

from embodichain.gen_sim.scene_engine.randomization.adapter import (
    convert_layouts,
    randomize_scene_export,
    scene_layout_input_from_export,
)
from embodichain.gen_sim.scene_engine.randomization.generation import randomize_scene
from embodichain.gen_sim.scene_engine.randomization.loading import load_layouts
from embodichain.gen_sim.scene_engine.randomization.persistence import save_layouts
from .test_randomization_baseline import _export
from .test_randomization_loading import _source_poses


@pytest.mark.parametrize("floor_subtree", [False, True])
def test_export_adapter_generates_source_independent_layouts(
    tmp_path, floor_subtree: bool
) -> None:
    export, _ = _export(tmp_path, floor_subtree=floor_subtree)
    workspace = export.parent
    original = (workspace / "scene_export/scene_config.json").read_bytes()
    original_graph = (export / "scene_graph.json").read_bytes()
    source_poses = _source_poses(workspace)
    source = scene_layout_input_from_export(workspace)
    expected_uids = {"table", "bottle"} if floor_subtree else set(source_poses)
    assert (
        set(source.poses) == set(source.vertices) == set(source.faces) == expected_uids
    )
    assert set(source.parent_by_uid) == expected_uids - {"table"}
    run = randomize_scene_export(
        workspace,
        {
            "candidate_count": 3,
            "max_attempts": 200,
            "policy": {"group_targets": ["bottle"]},
        },
    )
    loaded = load_layouts(run / "layouts")
    assert len(loaded.variants) == 3
    assert (workspace / "scene_export/scene_config.json").read_bytes() == original
    assert (export / "scene_graph.json").read_bytes() == original_graph
    assert set(loaded.baseline_poses) == set(source_poses)
    for uid, pose in source_poses.items():
        np.testing.assert_allclose(loaded.baseline_poses[uid], pose, atol=1e-12)
    assert all(
        {change.entity_uid for change in variant.pose_changes} == {"bottle"}
        for variant in loaded.variants
    )


def test_randomization_rejects_a_scene_with_only_floor_groups(tmp_path) -> None:
    export, cfg = _export(tmp_path, floor_subtree=True)
    config_path = export / "scene_config.json"
    config = json.loads(config_path.read_text())
    config["rigid_object"][-1]["placement_support"] = "floor"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="requires at least one table group"):
        scene_layout_input_from_export(export)
    with pytest.raises(ValueError, match="requires at least one table group"):
        randomize_scene(cfg)


@pytest.mark.parametrize(
    "height_scale, translation",
    [(1.0, [0.0, 0.0, 0.0]), (2.0, [0.2, -0.1, 0.8])],
    ids=["canonical", "translated-and-scaled"],
)
def test_v4_uses_exported_tabletop_level_and_safe_rectangle(
    tmp_path, height_scale, translation
) -> None:
    export, _ = _export(tmp_path, floor_subtree=True)
    config_path = export / "scene_config.json"
    config = json.loads(config_path.read_text())
    table = config["background"][0]
    mesh_path = export / table["shape"]["fpath"]
    upper = trimesh.load_scene(mesh_path).to_geometry()
    lower = trimesh.creation.box(extents=[0.8, 0.04, 0.6])
    lower.apply_translation([-0.35, -0.32, 0.0])
    trimesh.util.concatenate([upper, lower]).export(mesh_path)
    shelf_z = -0.3
    dx, dy, dz = translation
    arena_shelf_z = shelf_z * height_scale + dz
    table["body_scale"][1] = height_scale
    table["init_pos"] = translation
    table["support_surface_z"] = shelf_z
    table["support_optimization_rect_xy"] = [
        [-0.74 + dx, -0.29 + dy],
        [0.04 + dx, -0.29 + dy],
        [0.04 + dx, 0.29 + dy],
        [-0.74 + dx, 0.29 + dy],
    ]
    bottle_pos = config["rigid_object"][-1]["init_pos"]
    bottle_pos[:] = [bottle_pos[0] + dx, bottle_pos[1] + dy, arena_shelf_z + 0.05]
    config_path.write_text(json.dumps(config))

    job = {"candidate_count": 8, "max_attempts": 100}
    run = randomize_scene_export(tmp_path, job)
    loaded = load_layouts(run / "layouts")
    assert len(loaded.variants) == job["candidate_count"]
    for variant in loaded.variants:
        (change,) = variant.pose_changes
        x, y, z = np.asarray(change.arena_pose)[:3, 3]
        # The fixture's bottle is 12 cm wide and offset +3 cm from its pivot.
        assert -0.74 + dx <= x - 0.03 and x + 0.09 <= 0.04 + dx
        assert -0.29 + dy <= y - 0.04 and y + 0.04 <= 0.29 + dy
        assert z == pytest.approx(arena_shelf_z + 0.05)
    source = json.loads((run / "scene_input.json").read_text())
    assert [corner[2] for corner in source["main_support_bounds"]] == pytest.approx(
        [arena_shelf_z, arena_shelf_z]
    )

    # Both shelves can contain this rectangle. The height alone must change
    # the source and cache identity even though the XY sampling bounds agree.
    table["support_surface_z"] = 0.0
    config_path.write_text(json.dumps(config))
    upper_run = randomize_scene_export(tmp_path, job)
    manifest = json.loads((run / "run_result.json").read_text())
    upper_manifest = json.loads((upper_run / "run_result.json").read_text())
    assert manifest["source_sha256"] != upper_manifest["source_sha256"]
    assert len(list((run.parent / "geometry_cache").glob("*.json"))) == 2


@pytest.mark.parametrize("backend,storage", [("slsqp", "json"), ("batched", "packed")])
def test_legacy_conversion_preserves_every_identity(
    tmp_path, monkeypatch, backend, storage
) -> None:
    export, cfg = _export(tmp_path)
    workspace = export.parent
    original = randomize_scene(replace(cfg, backend=backend), candidate_count=3)
    run = save_layouts(original, output_root=workspace, layout_format=storage)
    import embodichain.gen_sim.scene_engine.randomization.generation as generation

    monkeypatch.setattr(
        generation,
        "randomize_scene",
        lambda *args, **kwargs: pytest.fail("Conversion resampled"),
    )
    converted = convert_layouts(run / "layouts", tmp_path / "converted")
    replay = load_layouts(converted / "layouts")
    assert [variant.variant_id for variant in replay.variants] == [
        candidate.variant.variant_id for candidate in original.candidates
    ]


def test_generate_and_edit_trigger_one_post_export_randomization(
    tmp_path, monkeypatch
) -> None:
    from embodichain.gen_sim.scene_engine.cli import start
    from embodichain.gen_sim.scene_engine.pipeline import generate, edit
    import embodichain.gen_sim.scene_engine.randomization.adapter as adapter

    events = []
    image = tmp_path / "input.png"
    image.write_bytes(b"placeholder")
    monkeypatch.setattr(
        generate,
        "generate_scene_from_image",
        lambda **kwargs: events.append("generate"),
    )
    monkeypatch.setattr(edit, "edit_scene", lambda **kwargs: events.append("edit"))
    monkeypatch.setattr(
        adapter, "randomize_scene_export", lambda *args: events.append("randomize")
    )
    start.cli_scene_engine(
        image, tmp_path, edit_prompt="move", scene_randomization={"seed": 7}
    )
    assert events == ["generate", "edit", "randomize"]
