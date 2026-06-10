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

import numpy as np
import trimesh
import pyrender
from PIL import Image
from pathlib import Path
from embodichain.gen_sim.simready_pipeline.core.asset import Asset
from embodichain.gen_sim.simready_pipeline.parser.base import AssetParser


class InternalParser(AssetParser):
    name = "internal"

    @staticmethod
    def _render_thumbnail(mesh: trimesh.Trimesh, output_path: Path) -> None:
        """
        Internal static function to handle the rendering logic.
        Camera is on X-axis positive, looking at the mesh's bounding box center.
        Z-axis is up.
        """
        bounds = mesh.bounds
        model_center = (bounds[0] + bounds[1]) / 2.0
        size = bounds[1] - bounds[0]

        target_frustum_size = max(size[1], size[2]) * 1.5
        yfov = np.pi / 4.0
        img_width, img_height = 512, 512
        camera_distance = (target_frustum_size / 2.0) / np.tan(yfov / 2.0)

        eye = model_center + np.array([camera_distance, 0.0, 0.0])
        target = model_center  # Look at the mesh center, not origin
        up = np.array([0.0, 0.0, 1.0])  # Z-up

        forward = eye - target
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        corrected_up = np.cross(forward, right)

        camera_pose = np.eye(4)
        camera_pose[:3, 0] = right
        camera_pose[:3, 1] = corrected_up
        camera_pose[:3, 2] = forward
        camera_pose[:3, 3] = eye

        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(pyrender_mesh)

        camera = pyrender.PerspectiveCamera(
            yfov=yfov, aspectRatio=img_width / img_height
        )
        scene.add(camera, pose=camera_pose)

        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        key_pose = np.eye(4)
        key_pose[:3, 3] = eye + np.array([0, camera_distance, camera_distance])
        scene.add(key_light, pose=key_pose)

        fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = eye + np.array([0, -camera_distance, 0.5 * camera_distance])
        scene.add(fill_light, pose=fill_pose)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=img_width, viewport_height=img_height
        )
        color, _ = renderer.render(scene)
        renderer.delete()

        Image.fromarray(color).save(output_path)

    def parse(self, asset: Asset, asset_root: Path) -> None:
        asset.internal.setdefault("thumbnail_path", "")
        asset.internal.setdefault("rendered", False)
        asset.internal.setdefault("error", None)

        mesh_path_ori = asset_root / asset.asset_data.get("path")
        mesh_path_sr = asset_root / "asset_simready" / "asset_simready.obj"
        mesh_path = None
        if mesh_path_sr.exists():
            mesh_path = mesh_path_sr
        elif mesh_path_ori.exists():
            mesh_path = mesh_path_ori
        else:
            asset.internal[
                "error"
            ] = "No mesh file found (neither simready nor original)"
            return

        try:

            mesh = trimesh.load(str(mesh_path), force="mesh")
            output_filename = f"{asset.asset_id}.png"
            output_path = asset_root / output_filename
            self._render_thumbnail(mesh, output_path)

            asset.internal.update(
                {
                    "thumbnail_path": f"{asset.asset_id}/{asset.asset_id}.png",
                    "rendered": True,
                    "error": None,
                }
            )

        except Exception as e:
            asset.internal.update({"rendered": False, "error": f"Exception: {str(e)}"})

        return
