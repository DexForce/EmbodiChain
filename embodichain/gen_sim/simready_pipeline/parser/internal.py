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
    def _look_at_pose(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Build a pose matrix whose -Z axis points from ``eye`` toward ``target``.

        This is the convention pyrender uses for both cameras (look direction is
        -Z) and directional lights (light travels along -Z), so the same matrix
        can orient a camera or aim a light at the model.
        """
        forward = eye - target
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        corrected_up = np.cross(forward, right)

        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = corrected_up
        pose[:3, 2] = forward
        pose[:3, 3] = eye
        return pose

    @staticmethod
    def _render_thumbnail(mesh: trimesh.Trimesh, output_path: Path) -> None:
        """
        Internal static function to handle the rendering logic.
        Camera looks at the mesh's bounding box center from a 3/4 front angle.
        Z-axis is up.
        """
        bounds = mesh.bounds
        model_center = (bounds[0] + bounds[1]) / 2.0
        size = bounds[1] - bounds[0]

        # Frame against the bounding sphere so the 3/4 view always fits nicely.
        radius = float(np.linalg.norm(size) / 2.0)
        yfov = np.pi / 4.0
        img_width, img_height = 512, 512
        camera_distance = (radius * 1.3) / np.tan(yfov / 2.0)

        up = np.array([0.0, 0.0, 1.0])  # Z-up

        # Classic 3/4 product shot: front (+X), slightly to the side (+Y),
        # slightly from above (+Z). A flat dead-on side view looks lifeless.
        view_dir = np.array([1.0, 0.55, 0.45])
        view_dir = view_dir / np.linalg.norm(view_dir)
        eye = model_center + view_dir * camera_distance
        camera_pose = InternalParser._look_at_pose(eye, model_center, up)

        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=[0.45, 0.45, 0.45],  # soft base fill, avoids black shadows
        )
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
        scene.add(pyrender_mesh)

        camera = pyrender.PerspectiveCamera(
            yfov=yfov, aspectRatio=img_width / img_height
        )
        scene.add(camera, pose=camera_pose)

        # Three-point lighting, all aimed at the model center so the
        # camera-facing side is properly lit (directional lights shine along
        # their pose -Z, so they must be oriented, not just positioned).
        key_pos = model_center + np.array([1.0, 0.8, 1.2]) * camera_distance
        key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        scene.add(
            key_light, pose=InternalParser._look_at_pose(key_pos, model_center, up)
        )

        fill_pos = model_center + np.array([0.6, -1.0, 0.3]) * camera_distance
        fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(
            fill_light, pose=InternalParser._look_at_pose(fill_pos, model_center, up)
        )

        rim_pos = model_center + np.array([-1.0, -0.5, 0.8]) * camera_distance
        rim_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
        scene.add(
            rim_light, pose=InternalParser._look_at_pose(rim_pos, model_center, up)
        )

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
        mesh_path_sr = asset_root / "asset_simready" / "asset_simready.glb"
        mesh_path = None
        if mesh_path_sr.exists():
            mesh_path = mesh_path_sr
        elif mesh_path_ori.exists():
            mesh_path = mesh_path_ori
        else:
            asset.internal["error"] = (
                "No mesh file found (neither simready nor original)"
            )
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
