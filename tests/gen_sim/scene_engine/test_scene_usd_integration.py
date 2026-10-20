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

import numpy as np
import pytest
import trimesh
from PIL import Image
from dexsim.kit.usd import parse_usd

from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
    build_scene_usd,
    build_scene_usdz,
)
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    DenoisingCfg,
    RenderCfg,
    UsdSourceCfg,
    VisualObjectCfg,
)
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import MeshCfg

pytestmark = [pytest.mark.requires_sim, pytest.mark.renderer]


@pytest.mark.parametrize("packaged", [False, True], ids=["usda", "usdz"])
def test_exported_usd_texture_matches_native_gltf(
    tmp_path: Path, packaged: bool
) -> None:
    # An asymmetric atlas detects a vertical inversion; primary colours also
    # avoid colour-space rounding obscuring the orientation comparison.
    pixels = (
        np.array(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 0]]],
            dtype=np.uint8,
        )
        .repeat(32, axis=0)
        .repeat(32, axis=1)
    )
    export_root = tmp_path / "scene_export"
    mesh_root = export_root / "mesh_assets" / "table"
    mesh_root.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.02))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=mesh.vertices[:, :2] + 0.5,
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(pixels),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    mesh.export(mesh_root / "table.glb")
    (export_root / "scene_config.json").write_text(
        json.dumps(
            {
                "format": "embodichain.scene-export/v1",
                "background": [
                    {
                        "uid": "table",
                        "shape": {
                            "shape_type": "Mesh",
                            "fpath": "mesh_assets/table/table.glb",
                        },
                        "body_type": "static",
                        "init_pos": [0.0, 0.0, 0.0],
                        "init_rot": [0.0, 0.0, 0.0],
                        "max_convex_hull_num": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    usd_path = build_scene_usd(output_root=tmp_path)
    if packaged:
        usd_path = build_scene_usdz(scene_usd_path=usd_path)
    parsed = parse_usd(str(usd_path))
    source_prim = next(
        item.usd.prim_path for item in parsed.mesh_objects if item.name == "table"
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=1,
            scene_node_capacity=128,
            render_cfg=RenderCfg(
                renderer="hybrid",
                # Compare native texture sampling without DLSS availability or
                # temporal reconstruction affecting this small offscreen target.
                denoising=DenoisingCfg(window="off", offscreen="off"),
            ),
        )
    )
    imported = {}
    texture_files = []
    reference = exported = camera = item = None
    try:
        imported = sim.add_usd("whole_scene", str(usd_path))
        reference = sim.add_visual_object(
            VisualObjectCfg(
                uid="gltf_reference",
                shape=MeshCfg(
                    fpath=str(tmp_path / "scene_usd/assets/table/model.gltf")
                ),
            )
        )
        exported = sim.add_visual_object(
            VisualObjectCfg(
                uid="usd_export",
                source=UsdSourceCfg(
                    file_path=str(usd_path), source_prim_path=source_prim
                ),
            )
        )
        camera = sim.add_sensor(
            CameraCfg(
                uid="texture_validation",
                width=128,
                height=128,
                intrinsics=(128.0, 128.0, 64.0, 64.0),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    # The native glTF importer maps its XY face to world XZ.
                    eye=(0.0, -1.5, 0.0),
                    target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0),
                ),
            )
        )
        sim.prepare()
        assert exported.cfg.source.file_path == str(usd_path)
        for item in imported.values():
            assert item.cfg.source.file_path == str(usd_path)
            item.set_visible(False)
        texture_files = [
            Path(material.base_color_map)
            for material in sim._spawn_scene.builder.materials.values()
            if material.base_color_map is not None
            and "embodichain-usdz-textures-" in material.base_color_map
        ]
        if packaged:
            assert texture_files
            assert all(path.is_file() for path in texture_files)
        sim.set_ground_plane_visibility(False)
        images = []
        for show_reference in (True, False):
            reference.set_visible(show_reference)
            exported.set_visible(not show_reference)
            sim.sync_render_state()
            for _ in range(16):
                camera.update()
            images.append(camera.get_data()["color"][0, :, :, :3].cpu().numpy().copy())
        for label, image in zip(("reference", "exported"), images):
            Image.fromarray(image).save(tmp_path / f"{label}.png")
        expected, actual = (image.astype(np.float32) for image in images)
        foreground = np.ptp(expected, axis=2) > 50.0
        assert (
            foreground.sum() > 200
        ), "The reference must contain a visible coloured atlas."
        # Allow reconstruction noise at edges, while an inverted atlas differs
        # across entire coloured regions by tens of 8-bit levels.
        assert np.abs(actual - expected)[foreground].mean() < 4.0
    finally:
        sim.destroy(exit_process=False)
        imported.clear()
        reference = exported = camera = item = None
        SimulationManager.flush_cleanup_queue()
    assert all(not path.exists() for path in texture_files)
