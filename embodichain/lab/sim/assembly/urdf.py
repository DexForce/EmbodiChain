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
"""Bounded articulation export for the world-mounted mjfit gears."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import ArticulationCfg
    from embodichain.toolkits.assembly.model import AssemblyAsset

__all__ = ["to_articulation_configs"]


def to_articulation_configs(
    asset: AssemblyAsset,
    directory: str | Path,
    *,
    source_contact: bool = True,
    backend: str = "newton",
) -> dict[str, ArticulationCfg]:
    """Build real hinge articulations with one moving compound link each.

    URDF supports damping/friction but not rotor armature. The latter is
    supplied through the native joint-drive configuration. Motor transmission
    gain remains in ``AssemblyJoint.motor_gear`` for the execution host.

    Args:
        asset: Neutral asset retaining source body frames and hinge attachments.
        directory: Preparation directory for URDF and visual mesh files.
        source_contact: Preserve supported source contact parameters.
        backend: Explicit ``newton`` or ``default`` physics configuration.

    Returns:
        Source hinge-body names mapped to native articulation configurations.

    Raises:
        ValueError: If a hinge anchor or contact overlay cannot be represented.
        NotImplementedError: If the backend cannot retain source contacts.
    """
    from .conversion import to_rigid_config
    from embodichain.lab.sim.cfg import (
        ArticulationCfg,
        ArticulationRootPropertiesCfg,
        JointDrivePropertiesCfg,
        LinkPhysicsOverrideCfg,
    )
    from scipy.spatial.transform import Rotation

    from ._visual import _VISUAL_FORMAT_VERSION, write_visual_mesh

    output = Path(directory).resolve()
    for body in asset.bodies:
        if body.joint.kind == "hinge" and (
            not isinstance(body.name, str)
            or not body.name
            or body.name in {".", ".."}
            or any(char in body.name for char in ("/", "\\", "\0"))
        ):
            raise ValueError(f"Unsafe assembly body name {body.name!r}.")
    output.mkdir(parents=True, exist_ok=True)

    def output_path(name: str) -> Path:
        path = output / name
        if path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError(
                f"Assembly output escapes its preparation directory: {path}"
            )
        return path

    # Validate every destination before creating any artifact, including the
    # OBJ writer's companion MTL. Existing symlinks cannot redirect writes.
    for body in asset.bodies:
        if body.joint.kind != "hinge":
            continue
        output_path(f"{body.name}.urdf")
        if any(geom.visible for geom in body.geometries):
            visual_name = f"{body.name}_visual_v{_VISUAL_FORMAT_VERSION}.obj"
            output_path(visual_name)
            output_path(Path(visual_name).with_suffix(".mtl").name)
        for index, geom in enumerate(body.geometries):
            if geom.collision and geom.kind == "mesh":
                output_path(f"{body.name}_{index}.obj")

    configs = {}
    for body in asset.bodies:
        if body.joint.kind != "hinge":
            continue
        if not np.allclose(body.joint.position, 0):
            raise ValueError(
                "Direct hinge export supports the body-centered mjfit gear hinge only."
            )
        root = ET.Element("robot", name=body.name)
        ET.SubElement(root, "link", name="anchor")
        link = ET.SubElement(root, "link", name=body.name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, body.com)), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=str(body.mass))
        inertia = np.asarray(body.inertia)
        ET.SubElement(
            inertial,
            "inertia",
            **{
                key: str(inertia[i, j])
                for key, i, j in (
                    ("ixx", 0, 0),
                    ("ixy", 0, 1),
                    ("ixz", 0, 2),
                    ("iyy", 1, 1),
                    ("iyz", 1, 2),
                    ("izz", 2, 2),
                )
            },
        )
        if any(geom.visible for geom in body.geometries):
            # Bake only geom-local transforms into the appearance. The link
            # retains the body/world pose and all source collision descriptors.
            visual_target = output_path(
                f"{body.name}_visual_v{_VISUAL_FORMAT_VERSION}.obj"
            )
            output_path(visual_target.with_suffix(".mtl").name)
            visual_path = write_visual_mesh(body, visual_target)
            visual = ET.SubElement(link, "visual", name="body_visual")
            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(
                ET.SubElement(visual, "geometry"), "mesh", filename=visual_path.name
            )
        for index, geom in enumerate(body.geometries):
            if not geom.collision:
                continue
            mesh_path = output_path(f"{body.name}_{index}.obj")
            if geom.kind == "mesh":
                with mesh_path.open("w", encoding="utf-8") as stream:
                    for v in geom.vertices:
                        stream.write("v " + " ".join(map(str, v)) + "\n")
                    for f in geom.triangles:
                        stream.write("f " + " ".join(str(int(x) + 1) for x in f) + "\n")
            element = ET.SubElement(link, "collision", name=f"collision_{index}")
            ET.SubElement(
                element,
                "origin",
                xyz=" ".join(map(str, geom.position)),
                rpy=" ".join(
                    map(str, Rotation.from_quat(geom.quaternion).as_euler("xyz"))
                ),
            )
            geometry = ET.SubElement(element, "geometry")
            if geom.kind == "mesh":
                ET.SubElement(geometry, "mesh", filename=mesh_path.name)
            elif geom.kind == "cylinder":
                ET.SubElement(
                    geometry,
                    "cylinder",
                    radius=str(geom.size[0]),
                    length=str(2 * geom.size[1]),
                )
            elif geom.kind == "sphere":
                ET.SubElement(geometry, "sphere", radius=str(geom.size[0]))
            elif geom.kind == "box":
                ET.SubElement(
                    geometry,
                    "box",
                    size=" ".join(str(2 * x) for x in geom.size[:3]),
                )
        joint = ET.SubElement(root, "joint", name=body.joint.name, type="continuous")
        ET.SubElement(joint, "parent", link="anchor")
        ET.SubElement(joint, "child", link=body.name)
        ET.SubElement(joint, "axis", xyz=" ".join(map(str, body.joint.axis)))
        ET.SubElement(
            joint,
            "dynamics",
            damping=str(body.joint.damping),
            friction=str(body.joint.frictionloss),
        )
        path = output_path(f"{body.name}.urdf")
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="unicode")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(body.quaternion).as_matrix()
        transform[:3, 3] = body.position
        free_copy = body.copy()
        free_copy.joint.kind = "free"
        properties = to_rigid_config(
            free_copy, source_contact=source_contact, backend=backend
        ).attrs
        configs[body.name] = ArticulationCfg(
            uid=body.name,
            fpath=str(path),
            init_local_pose=transform,
            root_props=ArticulationRootPropertiesCfg(
                fixed_base=True, self_collision_enabled=False
            ),
            asset_physics_mode="overlay",
            link_attrs={
                "moving_part": LinkPhysicsOverrideCfg(
                    link_names_expr=[body.name], attrs=properties
                )
            },
            joint_drive_props=JointDrivePropertiesCfg(
                target_mode="effort",
                stiffness=0.0,
                damping=0.0,
                armature=body.joint.armature,
            ),
        )
    return configs
