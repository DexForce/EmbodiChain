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
"""Validate physical robot inputs and prepare explicit derived geometry."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

__all__: list[str] = []


def required_collision_links(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Read explicit physical-link requirements without inferring link names."""
    groups = [metadata.get("required_collision_links", ())]
    calibrations = metadata.get("gripper_calibrations", ())
    if not isinstance(calibrations, (list, tuple)):
        raise ValueError("gripper_calibrations must contain calibration records")
    for calibration in calibrations:
        if not isinstance(calibration, Mapping) or not calibration.get("pad_links"):
            raise ValueError("Gripper calibration requires explicit pad_links")
        groups.append(calibration["pad_links"])
    links = []
    for group in groups:
        if not isinstance(group, (list, tuple)) or any(
            not isinstance(name, str) or not name.strip() for name in group
        ):
            raise ValueError("Required collision links must be nonempty link names")
        links.extend(group)
    return tuple(dict.fromkeys(links))


def validate_robot_collisions(
    source: Path, required_links: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Preflight authored collision geometry without starting native physics.

    Frame-only links need no geometry unless explicitly required. Mesh resources
    must contain finite, nondegenerate triangles; this check does not establish
    native import compatibility, contact quality or physical task success.
    """
    import numpy as np

    from embodichain.data_pipeline.real2sim._geometry import _robot_geometry_sha256

    source = source.resolve()
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as error:
        raise ValueError(f"Invalid physical robot URDF: {source}") from error
    if root.tag != "robot":
        raise ValueError("Physical robot URDF requires a robot root element")
    link_nodes = root.findall("link")
    links = {link.get("name"): link for link in link_nodes}
    if any(not name for name in links) or len(links) != len(link_nodes):
        raise ValueError("Physical robot URDF requires unique named links")
    if not root.findall("./link/collision"):
        raise ValueError("Physical robot URDF has no collision geometry")
    for name in required_links:
        if name not in links or not links[name].findall("collision"):
            raise ValueError(f"Required contact link lacks collision geometry: {name}")
    mesh_hashes: dict[str, str] = {}
    collision_links = []
    collision_count = 0

    def numbers(value: str | None, count: int, label: str) -> np.ndarray:
        try:
            values = np.asarray([float(item) for item in (value or "").split()])
        except ValueError as error:
            raise ValueError(f"Invalid collision {label}") from error
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid collision {label}")
        return values

    for name, link in links.items():
        for collision in link.findall("collision"):
            collision_count += 1
            if name not in collision_links:
                collision_links.append(name)
            origin = collision.find("origin")
            if origin is not None:
                for key in ("xyz", "rpy"):
                    numbers(origin.get(key, "0 0 0"), 3, f"{key} on {name}")
            geometries = collision.findall("geometry")
            if len(geometries) != 1 or len(geometries[0]) != 1:
                raise ValueError(f"Collision requires one geometry shape on {name}")
            shape = geometries[0][0]
            if shape.tag == "mesh":
                filename = shape.get("filename")
                if not filename:
                    raise ValueError(f"Collision mesh requires a filename on {name}")
                mesh_path = (source.parent / filename).resolve()
                if not mesh_path.is_file():
                    raise FileNotFoundError(f"Missing collision mesh: {mesh_path}")
                scale = numbers(shape.get("scale", "1 1 1"), 3, f"mesh scale on {name}")
                if np.any(scale == 0):
                    raise ValueError(f"Collision mesh has zero scale on {name}")
                import trimesh

                try:
                    mesh = trimesh.load(mesh_path, force="mesh", process=False)
                    mesh.apply_scale(scale)
                    usable = (
                        isinstance(mesh, trimesh.Trimesh)
                        and len(mesh.vertices) > 0
                        and len(mesh.faces) > 0
                        and np.isfinite(mesh.vertices).all()
                        and np.isfinite(mesh.area_faces).all()
                        and np.any(mesh.area_faces > 0)
                    )
                except Exception as error:
                    raise ValueError(f"Invalid collision mesh: {mesh_path}") from error
                if not usable:
                    raise ValueError(
                        f"Collision mesh has no usable triangles: {mesh_path}"
                    )
                mesh_hashes[str(mesh_path)] = hashlib.sha256(
                    mesh_path.read_bytes()
                ).hexdigest()
            elif shape.tag in ("box", "sphere", "cylinder"):
                attributes = {
                    "box": (("size", 3),),
                    "sphere": (("radius", 1),),
                    "cylinder": (("radius", 1), ("length", 1)),
                }[shape.tag]
                for key, count in attributes:
                    if np.any(numbers(shape.get(key), count, f"{key} on {name}") <= 0):
                        raise ValueError(
                            f"Collision {shape.tag} requires positive {key} on {name}"
                        )
            else:
                raise ValueError(
                    f"Unsupported collision geometry {shape.tag!r} on {name}"
                )
    return dict(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        robot_geometry_sha256=_robot_geometry_sha256(
            source, tuple(Path(path) for path in mesh_hashes)
        ),
        collision_count=collision_count,
        collision_links=collision_links,
        required_collision_links=list(required_links),
        collision_mesh_sha256=mesh_hashes,
    )


def prepare_robot_collisions(
    source: Path, output: Path, required_links: tuple[str, ...]
) -> tuple[Path, dict]:
    """Derive missing physics only for wholly visual-only models, preserving source.

    Existing collision models must already define the required contact links.
    Visual-only models use the same mesh/origin/scale as their visual geometry;
    the native URDF importer owns collision construction and approximation.
    """
    source = source.resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    if not required_links or any(name not in links for name in required_links):
        raise ValueError("Explicit existing finger contact links are required")
    visual_only = not root.findall("./link/collision")
    generated = []
    if visual_only:
        for name, link in links.items():
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is None or not len(geometry):
                    raise ValueError(f"Missing visual geometry on {name}")
                collision = ET.SubElement(link, "collision")
                origin = visual.find("origin")
                if origin is not None:
                    collision.append(deepcopy(origin))
                collision.append(deepcopy(geometry))
                generated.append(name)
    for name in required_links:
        if not links[name].findall("collision/geometry"):
            raise ValueError(f"Finger contact link lacks collision geometry: {name}")
    report = dict(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        derived_from_visuals=visual_only,
        generated_links=list(dict.fromkeys(generated)),
        required_contact_links=list(required_links),
        approximation="native URDF mesh collision construction",
    )
    if not visual_only:
        return source, report
    # Preserve model-relative meshes and their sidecar material files by linking
    # original resource directories; never edit or recopy recorded geometry.
    output.mkdir(parents=True, exist_ok=False)
    for mesh in root.findall("./link/*/geometry/mesh"):
        path = Path(mesh.get("filename", ""))
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Derived robot collisions require model-local meshes")
        if not (source.parent / path).is_file():
            raise ValueError(f"Missing robot mesh: {path}")
        top = output / path.parts[0]
        if not top.exists():
            top.symlink_to((source.parent / path.parts[0]).resolve())
    target = output / "robot_with_collisions.urdf"
    tree.write(target, encoding="unicode", xml_declaration=True)
    report["prepared_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return target, report


def prepare_robot_visual_winding(source: Path, output: Path) -> tuple[Path, dict]:
    """Repair CAD face orientation in a derived GLB, preserving visual placement.

    CAD exports commonly split vertices along normal seams. Weld a temporary
    topology to discover connected surfaces, then transfer only face reversals
    back to the original mesh so positions, UVs and material groups survive.
    This function accepts visual assets only; callers retain the original URDF
    collision geometry, joints and drives.
    """
    import numpy as np
    import trimesh

    source = source.resolve()
    if source == output.resolve():
        raise ValueError("Visual repair must not overwrite the source asset")
    scene = trimesh.load(source, force="scene", process=False)
    changes = {}
    for name, mesh in scene.geometry.items():
        topology = mesh.copy()
        topology.merge_vertices(merge_norm=True, merge_tex=True)
        before = topology.faces.copy()
        topology.fix_normals(multibody=True)
        flipped = np.any(topology.faces != before, axis=1)
        if np.any(flipped):
            faces = mesh.faces.copy()
            faces[flipped] = faces[flipped, ::-1]
            mesh.faces = faces
            changes[name] = int(flipped.sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    scene.export(output)
    return output, dict(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output=str(output.resolve()),
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        reversed_faces=changes,
        collision_geometry_unchanged=True,
    )
