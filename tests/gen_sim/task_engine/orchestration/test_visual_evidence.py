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

import hashlib
import json
from pathlib import Path

from PIL import Image
import trimesh

from embodichain.gen_sim.task_engine.orchestration.scene_source import scene_revision_id
from embodichain.gen_sim.task_engine.orchestration.visual_evidence import (
    render_scene_visual_evidence,
)


def _scene_export(tmp_path: Path) -> Path:
    root = tmp_path / "scene_export"
    root.mkdir()
    objects = []
    for uid, extents, position in (
        ("table", [1.0, 0.05, 0.7], [0.0, 0.0, 0.65]),
        ("red_cube", [0.10, 0.10, 0.10], [-0.2, 0.0, 0.75]),
        ("blue_cube", [0.10, 0.10, 0.10], [0.2, 0.0, 0.75]),
    ):
        trimesh.creation.box(extents=extents).export(root / f"{uid}.glb")
        objects.append(
            {
                "uid": uid,
                "name": uid.replace("_", " "),
                "category": "table" if uid == "table" else "cube",
                "shape": {"shape_type": "Mesh", "fpath": f"{uid}.glb"},
                "init_pos": position,
                "init_rot": [0.0, 0.0, 0.0],
                "body_scale": [1.0, 1.0, 1.0],
            }
        )
    (root / "scene_config.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "background": [objects[0]],
                "rigid_object": objects[1:],
                "articulation": [],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_visual_evidence_renders_current_assets_with_uid_masks(tmp_path: Path) -> None:
    source = _scene_export(tmp_path)
    before = hashlib.sha256((source / "scene_config.json").read_bytes()).hexdigest()

    manifest = render_scene_visual_evidence(source, tmp_path / "evidence")

    assert manifest["schema_version"] == "gen_sim.visual-grounding-evidence/v1"
    assert manifest["source_config_sha256"] == before
    assert manifest["scene_revision_id"] == scene_revision_id(source)
    assert {item["uid"] for item in manifest["objects"]} == {
        "table",
        "red_cube",
        "blue_cube",
    }
    assert {view["name"] for view in manifest["views"]} == {"oblique", "top"}
    assert manifest["unrendered_uids"] == []
    assert manifest["articulation_state_untrusted_uids"] == []
    catalog = Path(manifest["catalog_path"])
    assert catalog.is_file()
    assert (
        hashlib.sha256(catalog.read_bytes()).hexdigest() == manifest["catalog_sha256"]
    )
    persisted = json.loads((tmp_path / "evidence/manifest.json").read_text())
    assert persisted["catalog_path"] == "uid_catalog.png"
    assert persisted["views"][0]["annotated_path"] == "oblique_labeled.png"
    for view in manifest["views"]:
        image = Path(view["image_path"])
        assert image.is_file()
        assert hashlib.sha256(image.read_bytes()).hexdigest() == view["image_sha256"]
        assert Image.open(image).convert("RGB").getbbox() is not None
    for item in manifest["objects"]:
        assert item["visible_views"]
        for path in item["mask_paths"].values():
            assert Image.open(path).convert("L").getbbox() is not None
    assert (
        hashlib.sha256((source / "scene_config.json").read_bytes()).hexdigest()
        == before
    )
