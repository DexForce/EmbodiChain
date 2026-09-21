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

"""Isolated Blender worker. Run by path so bpy never enters the host process."""

from __future__ import annotations

import colorsys
import json
import math
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _material(name: str, color: tuple, emission: bool = False):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeEmission" if emission else "ShaderNodeBsdfDiffuse")
    shader.inputs["Color"].default_value = (*color, 1)
    material.node_tree.links.new(shader.outputs[0], output.inputs["Surface"])
    return material


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _main(directory: Path, mode: str) -> None:
    data = np.load(directory / "geometry.npz", allow_pickle=False)
    manifest = json.loads((directory / "evidence.json").read_text())
    vertices, faces, patches = (
        data["normalized_vertices"],
        data["faces"],
        data["face_patch_ids"],
    )
    resolution = manifest["render_resolution"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
        0.8,
        0.8,
        0.8,
        1,
    )
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.7
    mesh = bpy.data.meshes.new("source_mesh")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("mesh", mesh)
    scene.collection.objects.link(obj)
    clay = _material("clay", (0.52, 0.58, 0.66))
    mesh.materials.append(clay)
    for patch in range(int(patches.max()) + 1):
        color = colorsys.hsv_to_rgb((patch * 0.61803398875) % 1, 0.55, 0.75)
        mesh.materials.append(_material(f"patch_{patch}", color, emission=True))
    if mode != "evidence":
        filename = "segmentation_colors.npy" if mode == "part" else "vertex_colors.npy"
        colors = np.load(directory / filename, allow_pickle=False) / 255.0
        layer = mesh.vertex_colors.new(name="affordance")
        # Convert sRGB to linear for an accurate fixed-scale emission heatmap.
        colors = np.where(
            colors <= 0.04045, colors / 12.92, ((colors + 0.055) / 1.055) ** 2.4
        )
        rgba = np.c_[colors[faces.ravel()], np.ones(faces.size)]
        layer.data.foreach_set("color", rgba.ravel())
        material = _material("scores", (1, 1, 1), emission=True)
        vertex_color = material.node_tree.nodes.new("ShaderNodeVertexColor")
        vertex_color.layer_name = "affordance"
        material.node_tree.links.new(
            vertex_color.outputs["Color"],
            material.node_tree.nodes.get("Emission").inputs["Color"],
        )
        mesh.materials.clear()
        mesh.materials.append(material)
    camera_data = bpy.data.cameras.new("Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 1.9
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    for index, position in enumerate([(2, -3, 4), (-3, -1, 2), (1, 3, -2)]):
        light_data = bpy.data.lights.new(f"light_{index}", "AREA")
        light_data.energy = 100
        light_data.size = 3
        light = bpy.data.objects.new(f"light_{index}", light_data)
        scene.collection.objects.link(light)
        light.location = position
        light.rotation_euler = (-light.location).to_track_quat("-Z", "Y").to_euler()
    directions = [
        (math.cos(angle), math.sin(angle), 0.5)
        for angle in np.linspace(0, 2 * math.pi, 6, endpoint=False)
    ] + [(0, 0, 1), (0, 0, -1)]
    centers = vertices[faces].mean(axis=1)
    views, visible_by_patch = [], {
        str(patch): [] for patch in range(int(patches.max()) + 1)
    }
    for index, direction in enumerate(directions):
        camera.location = Vector(direction).normalized() * 3
        camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        tiles = []
        for kind in (["clay", "patches"] if mode == "evidence" else [mode]):
            if mode == "evidence":
                for polygon, patch in zip(mesh.polygons, patches):
                    polygon.material_index = 0 if kind == "clay" else int(patch) + 1
                bpy.context.view_layer.update()
            path = directory / f"view_{index:02d}_{kind}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            tile = Image.open(path).convert("RGB")
            if kind == "patches":
                visible = []
                ray = camera.location.normalized()
                for face, center in enumerate(centers):
                    target = Vector(center)
                    hit, location, _, _ = obj.ray_cast(target + ray * 3, -ray)
                    if hit and (location - target).length < 1e-5:
                        visible.append(face)
                visible = np.asarray(visible, dtype=int)
                draw = ImageDraw.Draw(tile)
                for patch in np.unique(patches[visible]):
                    candidates = visible[patches[visible] == patch]
                    center = centers[candidates].mean(axis=0)
                    face = candidates[
                        np.linalg.norm(centers[candidates] - center, axis=1).argmin()
                    ]
                    point = world_to_camera_view(scene, camera, Vector(centers[face]))
                    x, y = int(point.x * resolution), int((1 - point.y) * resolution)
                    label = str(patch)
                    box = draw.textbbox((x, y), label, font=_font(15), anchor="mm")
                    draw.rectangle(
                        (box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1),
                        fill="white",
                        outline="#333333",
                    )
                    draw.text((x, y), label, fill="black", font=_font(15), anchor="mm")
                    visible_by_patch[str(patch)].append(index)
                tile.save(path)
            header = Image.new("RGB", (resolution, resolution + 32), "white")
            header.paste(tile, (0, 32))
            caption = f"view {index} | {kind} | camera ({direction[0]:.1f},{direction[1]:.1f},{direction[2]:.1f})"
            ImageDraw.Draw(header).text((8, 8), caption, fill="black", font=_font(13))
            tiles.append(header)
        row = Image.new("RGB", (resolution * len(tiles), resolution + 32), "white")
        for column, tile in enumerate(tiles):
            row.paste(tile, (column * resolution, 0))
        views.append(row)
    if mode == "evidence":
        images = []
        for offset in range(0, len(views), 2):
            sheet = Image.new("RGB", (resolution * 2, (resolution + 32) * 2), "white")
            for row, view in enumerate(views[offset : offset + 2]):
                sheet.paste(view, (0, row * (resolution + 32)))
            name = f"evidence_{offset // 2:02d}.png"
            sheet.save(directory / name)
            images.append(name)
        manifest["images"] = images
        manifest["patch_visible_views"] = visible_by_patch
        (directory / "evidence.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)
        )
    else:
        sheet = Image.new("RGB", (resolution * 4, (resolution + 32) * 2 + 50), "white")
        for index, view in enumerate(views):
            sheet.paste(
                view, ((index % 4) * resolution, (index // 4) * (resolution + 32))
            )
        draw = ImageDraw.Draw(sheet)
        caption = (
            "Target part: red = selected, gray = remainder"
            if mode == "part"
            else "Contact suitability: 0 blue / avoid     0.5 cyan-yellow / conditional     1 red / preferred"
        )
        draw.text((15, sheet.height - 35), caption, fill="black", font=_font(20))
        sheet.save(
            directory / ("segmentation.png" if mode == "part" else "heatmap.png")
        )


if __name__ == "__main__":
    _main(Path(sys.argv[1]), sys.argv[2])
