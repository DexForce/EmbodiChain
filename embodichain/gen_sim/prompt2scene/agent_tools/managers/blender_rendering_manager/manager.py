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
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from embodichain.gen_sim.prompt2scene.agent_tools.managers.blender_rendering_manager.schemas import (
    RenderObjectScenesRequest,
    RenderObjectScenesResult,
)

__all__ = ["BlenderRenderingManager"]


class BlenderRenderingManager:
    """Render simulation scenes through Blender's background CLI."""

    def render_object_scenes(
        self,
        request: RenderObjectScenesRequest,
    ) -> RenderObjectScenesResult:
        """Render a front-oblique view of a collection of Z-up scenes."""
        output_path = request.output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="p2s_blender_render_") as tmp_dir:
            glb_paths = self._export_y_up_scenes(
                request.object_scenes,
                Path(tmp_dir),
            )
            self._render_glbs(
                glb_paths,
                output_path,
                timeout_seconds=request.timeout_seconds,
            )
        return RenderObjectScenesResult(output_path=output_path)

    @staticmethod
    def _export_y_up_scenes(
        object_scenes: list[tuple[str, object]],
        output_dir: Path,
    ) -> list[Path]:
        z_up_to_y_up = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        paths: list[Path] = []
        for object_id, scene in object_scenes:
            path = output_dir / f"{object_id}_render.glb"
            copied = scene.copy()
            copied.apply_transform(z_up_to_y_up)
            copied.export(path)
            paths.append(path)
        return paths

    @classmethod
    def _render_glbs(
        cls,
        glb_paths: list[Path],
        output_path: Path,
        *,
        timeout_seconds: int,
    ) -> None:
        script = cls._front_oblique_script(glb_paths, output_path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            encoding="utf-8",
            delete=False,
        ) as file:
            script_path = Path(file.name)
            file.write(script)
        try:
            subprocess.run(
                ["blender", "--background", "--python", str(script_path)],
                check=True,
                timeout=timeout_seconds,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr_tail = (exc.stderr or "").strip()[-4000:]
            raise RuntimeError(
                f"Blender front-oblique render failed:\n{stderr_tail}"
            ) from exc
        finally:
            script_path.unlink(missing_ok=True)
        if not output_path.is_file():
            raise FileNotFoundError(f"Blender render was not written: {output_path}")

    @staticmethod
    def _front_oblique_script(glb_paths: list[Path], output_path: Path) -> str:
        object_paths_json = json.dumps([str(path.resolve()) for path in glb_paths])
        output_path_json = json.dumps(str(output_path.resolve()))
        return f"""\
import bpy
import json
import mathutils

object_paths = json.loads({object_paths_json!r})
output_path = json.loads({output_path_json!r})
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for path in object_paths:
    bpy.ops.import_scene.gltf(filepath=path)
mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not mesh_objects:
    raise RuntimeError("No mesh objects were imported.")
min_corner = mathutils.Vector((float("inf"), float("inf"), float("inf")))
max_corner = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))
for obj in mesh_objects:
    for corner in obj.bound_box:
        world = obj.matrix_world @ mathutils.Vector(corner)
        min_corner.x = min(min_corner.x, world.x)
        min_corner.y = min(min_corner.y, world.y)
        min_corner.z = min(min_corner.z, world.z)
        max_corner.x = max(max_corner.x, world.x)
        max_corner.y = max(max_corner.y, world.y)
        max_corner.z = max(max_corner.z, world.z)
center = (min_corner + max_corner) * 0.5
span_x = max(max_corner.x - min_corner.x, 1.0e-4)
span_y = max(max_corner.y - min_corner.y, 1.0e-4)
span_z = max(max_corner.z - min_corner.z, 1.0e-4)
camera_data = bpy.data.cameras.new("front_oblique_camera")
camera = bpy.data.objects.new("front_oblique_camera", camera_data)
bpy.context.collection.objects.link(camera)
view_distance = max(span_x, span_y, span_z) * 2.4
camera.location = (center.x, center.y - view_distance, center.z + view_distance * 0.75)
camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
camera_data.type = "ORTHO"
camera_data.ortho_scale = max(span_x, span_y, span_z * 1.8) * 1.35
bpy.context.scene.camera = camera
light_data = bpy.data.lights.new("front_oblique_area_light", "AREA")
light = bpy.data.objects.new("front_oblique_area_light", light_data)
bpy.context.collection.objects.link(light)
light.location = camera.location
light_data.energy = 350.0
light_data.size = max(span_x, span_y) * 2.0
bpy.context.scene.world.color = (0.90, 0.90, 0.90)
try:
    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT"
except Exception:
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
bpy.context.scene.render.resolution_x = 768
bpy.context.scene.render.resolution_y = 768
bpy.context.scene.render.film_transparent = False
bpy.context.scene.view_settings.view_transform = "Standard"
bpy.context.scene.view_settings.look = "Medium Contrast"
bpy.context.scene.render.filepath = output_path
bpy.ops.render.render(write_still=True)
"""
