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

"""Decode robot visual assets without creating simulation objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from uuid import uuid4
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh

from .cfg import MarkerPrototypeCfg


@dataclass(frozen=True)
class RobotMarkerVisual:
    link_name: str
    local_pose: np.ndarray
    prototype: MarkerPrototypeCfg


def _numbers(text: str, count: int) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split()])
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError(f"Expected {count} finite numbers, received {text!r}.")
    return values


def _asset_path(root: Path, value: str) -> Path:
    if value.startswith("file://"):
        value = value[7:]
    if value.startswith("package://"):
        relative = Path(value[10:])
        candidates = [
            root / relative,
            root / Path(*relative.parts[1:]),
            root.parent / Path(*relative.parts[1:]),
        ]
    else:
        candidates = [root / value]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve visual asset {value!r} from {root}.")


def _image(path: Path | str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA")).copy()


def _mesh_parts(
    path: Path, texture_directory: Path | None = None
) -> list[MarkerPrototypeCfg]:
    scene = trimesh.load_scene(path, process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name]
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected triangle geometry in {path}: {geometry_name}")
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        visual = mesh.visual
        material = getattr(visual, "material", None)
        if isinstance(material, trimesh.visual.material.MultiMaterial):
            for index in np.unique(visual.face_materials):
                faces = mesh.faces[visual.face_materials == index]
                part = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=faces,
                    process=False,
                    visual=trimesh.visual.TextureVisuals(
                        uv=visual.uv, material=material.materials[index]
                    ),
                )
                parts.append(_mesh_prototype(part, texture_directory))
        else:
            parts.append(_mesh_prototype(mesh, texture_directory))
    if not parts:
        raise ValueError(f"No triangle visuals in {path}.")
    return parts


def _mesh_prototype(
    mesh: trimesh.Trimesh, texture_directory: Path | None = None
) -> MarkerPrototypeCfg:
    color_visual = mesh.visual.kind in {"vertex", "face"}
    if color_visual:
        mesh = mesh.copy()
        mesh.visual = mesh.visual.to_texture()
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    color = getattr(material, "baseColorFactor", None)
    if color is None:
        color = getattr(material, "diffuse", None)
    if color is None or color_visual:
        color = np.array([255, 255, 255, 255])
    color = np.asarray(color, dtype=float) / 255.0
    if isinstance(material, trimesh.visual.material.PBRMaterial) and not color_visual:
        color[:3] = _linear_to_srgb(color[:3])
    uv = getattr(mesh.visual, "uv", None)
    parameters = {"metallic": 0.0, "roughness": 1.0}
    if isinstance(material, trimesh.visual.material.PBRMaterial):
        parameters.update(
            color_space="linear",
            metallic=(
                material.metallicFactor if material.metallicFactor is not None else 1.0
            ),
            roughness=(
                material.roughnessFactor
                if material.roughnessFactor is not None
                else 1.0
            ),
            double_sided=material.doubleSided,
            alpha_mode=material.alphaMode,
            alpha_cutoff=(
                material.alphaCutoff if material.alphaCutoff is not None else 0.5
            ),
        )
        if material.emissiveFactor is not None:
            parameters["emissive"] = (*material.emissiveFactor.tolist(), 1.0)
        for source, target in (
            ("baseColorTexture", "base_color_map"),
            ("normalTexture", "normal_map"),
            ("emissiveTexture", "emissive_map"),
            ("occlusionTexture", "ao_map"),
            ("metallicRoughnessTexture", "metallic_roughness_map"),
        ):
            texture = getattr(material, source, None)
            if texture is not None and texture_directory is not None:
                target_path = texture_directory / f"{uuid4().hex}.png"
                texture.save(target_path)
                parameters[target] = str(target_path)
    elif material is not None:
        shininess = getattr(material, "kwargs", {}).get("Ns", 0.0)
        parameters["roughness"] = float(np.sqrt(2.0 / (float(shininess) + 2.0)))
    return MarkerPrototypeCfg(
        shape="mesh",
        vertices=np.asarray(mesh.vertices),
        normals=np.asarray(mesh.vertex_normals),
        material=parameters,
        faces=np.asarray(mesh.faces),
        color=tuple(color),
        uv_coords=uv,
        texture=None if image is None else np.asarray(image.convert("RGBA")).copy(),
    )


def _linear_to_srgb(color: np.ndarray) -> np.ndarray:
    """Encode linear RGB for the marker protocol's sRGB material contract."""
    return np.where(
        color <= 0.0031308,
        color * 12.92,
        1.055 * np.maximum(color, 0) ** (1 / 2.4) - 0.055,
    )


def _validate_usd_material(material: object) -> None:
    """Reject runtimes that cannot preserve USD color and sampler semantics."""
    if not all(
        hasattr(material, field) for field in ("color_space", "texture_settings")
    ):
        raise RuntimeError(
            "USD robot materials require a runtime with explicit color-space "
            "and texture-sampling support. Upgrade the visualization runtime "
            "before loading this USD asset."
        )


def _usd_texture(prototype: MarkerPrototypeCfg, material: object) -> None:
    """Reuse USD texture lowering and bake its coordinates into marker UVs."""
    from dexsim.spawn import MaterialTextureDesc
    from dexsim.spawn.adapters.material_texture import prepare_material_textures

    # Ghosts use base color and opacity. Other material lobes are not rendered.
    settings = {
        name: material.texture_settings.get(
            name, MaterialTextureDesc(channel="a" if name == "opacity_map" else "rgb")
        )
        for name in ("base_color_map", "opacity_map")
        if getattr(material, name) is not None
    }
    if not settings:
        return
    data, sampling = prepare_material_textures(
        replace(material, texture_settings=settings)
    )["base_color_map"]
    if prototype.uv_coords is None:
        raise ValueError(
            "Textured USD robot marker requires authored mesh UV coordinates."
        )
    uv = np.asarray(prototype.uv_coords, dtype=float).copy()
    angle = np.deg2rad(sampling.uv_rotation)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    uv = (uv * sampling.uv_scale) @ rotation.T + sampling.uv_translation
    # The native decoder returns bottom-up linear pixels. The marker protocol
    # stores top-down sRGB images; alpha remains linear coverage.
    data = data[::-1].copy()
    wraps = [sampling.wrap_s, sampling.wrap_t]
    for coordinate, axis in ((0, 1), (1, 0)):
        if wraps[coordinate] != "black":
            continue
        size = data.shape[axis]
        shape = list(data.shape)
        shape[axis] += 2
        padded = np.empty(shape, dtype=data.dtype)
        padded[:] = sampling.border_color
        interior = [slice(None)] * 3
        interior[axis] = slice(1, -1)
        padded[tuple(interior)] = data
        data = padded
        uv[:, coordinate] = (uv[:, coordinate] * size + 1) / (size + 2)
        wraps[coordinate] = "clamp"
    data[..., :3] = _linear_to_srgb(data[..., :3])
    prototype.texture = np.rint(np.clip(data, 0, 1) * 255).astype(np.uint8)
    prototype.uv_coords = uv
    prototype.texture_wrap = tuple(wraps)


def load_urdf_visuals(
    path: Path, texture_directory: Path | None = None
) -> list[RobotMarkerVisual]:
    root = ET.parse(path).getroot()
    materials = {item.attrib["name"]: item for item in root.findall("material")}
    meshes = {}
    result = []
    for link in root.findall("link"):
        name = link.attrib["name"]
        for visual in link.findall("visual"):
            pose = np.eye(4)
            origin = visual.find("origin")
            if origin is not None:
                pose[:3, 3] = _numbers(origin.get("xyz", "0 0 0"), 3)
                pose[:3, :3] = Rotation.from_euler(
                    "xyz", _numbers(origin.get("rpy", "0 0 0"), 3)
                ).as_matrix()
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) != 1:
                raise ValueError(f"Expected one geometry in visual of link {name!r}.")
            geometry = geometry[0]
            if geometry.tag == "mesh":
                mesh_path = _asset_path(path.parent, geometry.attrib["filename"])
                if mesh_path not in meshes:
                    meshes[mesh_path] = _mesh_parts(mesh_path, texture_directory)
                import copy

                prototypes = [copy.copy(part) for part in meshes[mesh_path]]
                scale = _numbers(geometry.get("scale", "1 1 1"), 3)
            elif geometry.tag == "box":
                prototypes = [MarkerPrototypeCfg(shape="box", color=(1, 1, 1, 1))]
                scale = _numbers(geometry.attrib["size"], 3)
            elif geometry.tag == "sphere":
                prototypes = [MarkerPrototypeCfg(shape="sphere", color=(1, 1, 1, 1))]
                scale = np.repeat(2 * float(geometry.attrib["radius"]), 3)
            elif geometry.tag == "cylinder":
                prototypes = [MarkerPrototypeCfg(shape="cylinder", color=(1, 1, 1, 1))]
                scale = np.array(
                    [2 * float(geometry.attrib["radius"])] * 2
                    + [float(geometry.attrib["length"])]
                )
            else:
                raise ValueError(f"Unsupported URDF visual geometry: {geometry.tag}")
            if not np.isfinite(scale).all() or np.any(scale <= 0):
                raise ValueError("URDF visual scale must be positive and finite.")
            material = visual.find("material")
            if material is not None and len(material) == 0:
                material = materials.get(material.get("name"))
            for prototype in prototypes:
                prototype.scale = tuple(scale)
                if material is not None:
                    color = material.find("color")
                    texture = material.find("texture")
                    if color is not None:
                        prototype.color = tuple(_numbers(color.attrib["rgba"], 4))
                    if texture is not None:
                        texture_path = _asset_path(
                            path.parent, texture.attrib["filename"]
                        )
                        prototype.texture = _image(texture_path)
                        prototype.material = dict(prototype.material or {})
                        prototype.material["base_color_map"] = str(texture_path)
                        settings = dict(prototype.material.get("texture_settings", {}))
                        settings.pop("base_color_map", None)
                        prototype.material["texture_settings"] = settings
                result.append(RobotMarkerVisual(name, pose.copy(), prototype))
    if not result:
        raise ValueError(f"No URDF visual geometry in {path}.")
    return result


def load_usd_visuals(
    path: Path, prim_path: str | None, texture_directory: Path | None = None
) -> list[RobotMarkerVisual]:
    from dexsim.spawn.adapters.common import generated_mesh
    from embodichain.lab.sim.spawn._usd_assets import _UsdTextureFiles, _parse_singleton

    scene, obj = _parse_singleton(
        path,
        ("articulations", "mesh_objects"),
        "robot marker articulation or visual object",
        prim_path=prim_path,
    )
    links = (
        [(link.name, link.visuals) for link in obj.links]
        if hasattr(obj, "links")
        else [(obj.name, obj.renders)]
    )
    files = _UsdTextureFiles(texture_directory)
    result = []
    try:
        materials = files.resolve(scene.materials)
        for link_name, renders in links:
            for render in renders:
                material = (
                    files.resolve({"inline": render.material})["inline"]
                    if render.material is not None
                    else materials.get(render.material_ref)
                )
                if render.file_path is not None:
                    prototypes = _mesh_parts(Path(render.file_path), texture_directory)
                    for prototype in prototypes:
                        prototype.vertices = trimesh.transform_points(
                            prototype.vertices, render.local_transform
                        )
                        normal_transform = np.linalg.inv(
                            render.local_transform[:3, :3]
                        ).T
                        prototype.normals = prototype.normals @ normal_transform.T
                        lengths = np.linalg.norm(
                            prototype.normals, axis=1, keepdims=True
                        )
                        prototype.normals /= np.where(lengths > 1e-8, lengths, 1.0)
                else:
                    vertices, faces, normals, uv = generated_mesh(render)
                    prototypes = [
                        MarkerPrototypeCfg(
                            shape="mesh",
                            vertices=vertices,
                            normals=normals if np.size(normals) else None,
                            faces=np.asarray(faces).reshape(-1, 3),
                            uv_coords=uv if np.size(uv) else None,
                            color=(1, 1, 1, 1),
                        )
                    ]
                for prototype in prototypes:
                    if material is not None:
                        _validate_usd_material(material)
                        if material.base_color is not None:
                            color = np.array(material.base_color, dtype=float)
                            if material.color_space == "linear":
                                color[:3] = _linear_to_srgb(color[:3])
                            prototype.color = tuple(color)
                        # Preserve original UVs for native PBR. The browser/transparent
                        # texture path below bakes the base-color sampler into UVs.
                        parameters = asdict(material)
                        for key in ("name", "usd", "import_warnings", "base_color"):
                            parameters.pop(key, None)
                        prototype.material = parameters
                        prototype.material_uv_coords = (
                            None
                            if prototype.uv_coords is None
                            else np.array(prototype.uv_coords, copy=True)
                        )
                        _usd_texture(prototype, material)
                    result.append(RobotMarkerVisual(link_name, np.eye(4), prototype))
    finally:
        files.close()
    if not result:
        raise ValueError(f"No USD visual geometry in {path}.")
    return result
