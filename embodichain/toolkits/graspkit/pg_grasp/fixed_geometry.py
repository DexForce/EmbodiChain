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

"""URDF collision surfaces at a declared, fixed gripper opening.

This module needs asset files, not a simulator. Samples describe the complete
declared collision meshes, but remain a finite surface check, not a continuous
mesh-intersection or swept-volume certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh

__all__ = ["FixedGripperCollisionGeometry"]


def _rigid_transform(value: object, *, error_message: str) -> np.ndarray:
    """Validate one transform while retaining the caller's error context."""
    transform = np.asarray(value, dtype=np.float64)
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
    ):
        raise ValueError(error_message)
    return transform


def _origin(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4)
    if element is not None:
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
    return result


def _surface_points(triangles: np.ndarray, spacing: float) -> np.ndarray:
    """Include every vertex and a deterministic barycentric grid per triangle."""
    pieces = []
    for triangle in triangles:
        longest = np.linalg.norm(triangle - np.roll(triangle, 1, axis=0), axis=1).max()
        count = max(1, int(np.ceil(longest / spacing)))
        u, v = np.triu_indices(count + 1)
        # 0 <= u <= v <= count gives the full barycentric triangle.
        weights = np.stack((1 - v / count, (v - u) / count, u / count), axis=1)
        pieces.append(weights @ triangle)
    return np.unique(np.concatenate(pieces), axis=0)


@dataclass(frozen=True)
class FixedGripperCollisionGeometry:
    """Owned CPU surface samples in the canonical contact/grasp frame.

    A fixed opening checks the physical pre-grasp command declared by the
    caller. Antipodal widths do not reposition these surfaces. Closing contacts
    and arm/environment collisions are separate. ``grasp_from_tcp`` in the
    provenance maps runtime TCP coordinates into canonical contact coordinates.
    """

    points: torch.Tensor
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        points = self.points
        if (
            points.device.type != "cpu"
            or not points.is_floating_point()
            or points.ndim != 2
            or points.shape[1] != 3
            or not len(points)
            or not bool(torch.isfinite(points).all())
        ):
            raise ValueError(
                "Fixed gripper geometry requires finite CPU points (N, 3)."
            )
        _rigid_transform(
            self.provenance.get("grasp_from_tcp"),
            error_message="Explicit rigid grasp_from_tcp is required for collision geometry.",
        )
        object.__setattr__(self, "points", points.detach().clone().contiguous())
        object.__setattr__(self, "provenance", deepcopy(self.provenance))

    def canonical_to_tcp_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Map canonical contact poses to the declared runtime TCP exactly once.

        Args:
            poses: Canonical contact transforms with trailing shape ``(4, 4)``.

        Returns:
            Runtime TCP poses with the same batch shape, dtype and device.
        """
        return poses @ poses.new_tensor(self.provenance["grasp_from_tcp"])

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str | Path,
        *,
        root_link: str,
        tcp_link: str,
        collision_links: tuple[str, ...],
        joint_positions: dict[str, float],
        grasp_from_tcp: np.ndarray,
        point_spacing: float,
        link_from_tcp: np.ndarray | None = None,
        mesh_paths: dict[str, str | Path] | None = None,
    ) -> FixedGripperCollisionGeometry:
        """Sample mesh collisions at a declared joint state below ``root_link``.

        Args:
            urdf_path: URDF with collision mesh paths relative to its directory.
            root_link: Root of the gripper subtree; upstream arm FK is irrelevant.
            tcp_link: Actual runtime TCP in that subtree.
            collision_links: Every gripper link whose collisions must be checked.
            joint_positions: Explicit active positions; mimic joints follow their
                declared multiplier/offset and cannot contradict their leader.
            grasp_from_tcp: Transform taking runtime TCP coordinates to the
                generator's canonical contact frame. Public poses use the runtime TCP.
            point_spacing: Maximum edge spacing of each triangle's sample grid.
            link_from_tcp: Virtual TCP relative to ``tcp_link``; omission uses
                that named link as the runtime TCP.
            mesh_paths: Explicit filename-to-path mapping for archives whose
                mesh layout differs from their URDF references. No search or
                geometry fallback is performed; resolved files are hashed.

        Returns:
            Surface samples and asset/transform provenance for this opening.
        """
        if not np.isfinite(point_spacing) or point_spacing <= 0:
            raise ValueError("point_spacing must be finite and positive.")
        grasp_from_tcp = _rigid_transform(
            grasp_from_tcp,
            error_message="grasp_from_tcp must be a finite rigid transform.",
        )
        path = Path(urdf_path).resolve(strict=True)
        source = path.read_bytes()
        document = ET.fromstring(source)
        links = {link.attrib["name"]: link for link in document.findall("link")}
        joints = {
            joint.find("child").attrib["link"]: joint
            for joint in document.findall("joint")
        }
        if root_link not in links:
            raise ValueError(f"Unknown gripper root link {root_link!r}.")
        transforms = {root_link: np.eye(4)}

        named_joints = {
            joint.attrib["name"]: joint for joint in document.findall("joint")
        }
        resolved_positions: dict[str, float] = {}
        resolving: set[str] = set()

        def joint_position(joint: ET.Element) -> float:
            name = joint.attrib["name"]
            if name in resolved_positions:
                return resolved_positions[name]
            if name in resolving:
                raise ValueError(f"Cyclic mimic relation at joint {name!r}.")
            resolving.add(name)
            mimic = joint.find("mimic")
            if mimic is None:
                if name not in joint_positions:
                    raise ValueError(
                        f"Explicit opening position required for joint {name!r}."
                    )
                value = float(joint_positions[name])
            else:
                value = joint_position(named_joints[mimic.attrib["joint"]]) * float(
                    mimic.get("multiplier", "1")
                ) + float(mimic.get("offset", "0"))
                if name in joint_positions and not np.isclose(
                    joint_positions[name], value, rtol=0, atol=1e-9
                ):
                    raise ValueError(f"Fixed opening contradicts mimic joint {name!r}.")
            limits = joint.find("limit")
            if not np.isfinite(value) or (
                joint.attrib["type"] != "continuous"
                and (
                    limits is None
                    or not float(limits.attrib["lower"])
                    <= value
                    <= float(limits.attrib["upper"])
                )
            ):
                raise ValueError(f"Invalid fixed opening for joint {name!r}.")
            resolving.remove(name)
            resolved_positions[name] = value
            return value

        def root_from_link(name: str) -> np.ndarray:
            if name in transforms:
                return transforms[name]
            if name not in joints:
                raise ValueError(
                    f"Link {name!r} is not below gripper root {root_link!r}."
                )
            joint = joints[name]
            transform = _origin(joint.find("origin"))
            kind = joint.attrib["type"]
            if kind in {"prismatic", "revolute", "continuous"}:
                position = joint_position(joint)
                axis_element = joint.find("axis")
                axis = np.fromstring(
                    "1 0 0" if axis_element is None else axis_element.attrib["xyz"],
                    sep=" ",
                )
                if (
                    axis.shape != (3,)
                    or not np.isfinite(axis).all()
                    or np.linalg.norm(axis) == 0
                ):
                    raise ValueError(
                        f"Invalid joint axis for {joint.attrib['name']!r}."
                    )
                axis /= np.linalg.norm(axis)
                if kind == "prismatic":
                    transform[:3, 3] += transform[:3, :3] @ (axis * position)
                else:
                    transform[:3, :3] = (
                        transform[:3, :3]
                        @ Rotation.from_rotvec(axis * position).as_matrix()
                    )
            elif kind != "fixed":
                raise ValueError(
                    "Fixed templates require fixed/prismatic/revolute joints."
                )
            transforms[name] = (
                root_from_link(joint.find("parent").attrib["link"]) @ transform
            )
            return transforms[name]

        for name in joint_positions:
            if name not in named_joints:
                raise ValueError(f"Unknown opening joint {name!r}.")
            joint_position(named_joints[name])
        if link_from_tcp is None:
            link_from_tcp = np.eye(4)
        link_from_tcp = _rigid_transform(
            link_from_tcp,
            error_message="link_from_tcp must be a finite rigid transform.",
        )
        tcp_from_root = np.linalg.inv(root_from_link(tcp_link) @ link_from_tcp)
        files = {str(path): hashlib.sha256(source).hexdigest()}
        pieces = []
        part_records = []
        for name in collision_links:
            collisions = links[name].findall("collision")
            if not collisions:
                raise ValueError(f"Link {name!r} has no collision geometry.")
            for collision in collisions:
                mesh_node = collision.find("geometry/mesh")
                if mesh_node is None:
                    raise ValueError(
                        "Fixed collision templates require triangle meshes."
                    )
                filename = mesh_node.attrib["filename"]
                mesh_path = Path(
                    (mesh_paths or {}).get(filename, path.parent / filename)
                ).resolve(strict=True)
                files[str(mesh_path)] = hashlib.sha256(
                    mesh_path.read_bytes()
                ).hexdigest()
                mesh = trimesh.load(mesh_path, force="mesh", process=False)
                scale = np.fromstring(mesh_node.get("scale", "1 1 1"), sep=" ")
                transform = (
                    grasp_from_tcp
                    @ tcp_from_root
                    @ root_from_link(name)
                    @ _origin(collision.find("origin"))
                )
                vertices = np.asarray(mesh.vertices) * scale
                # Choose the surface grid before rigid transforms. TCP changes
                # must not change ceil(edge/spacing) or duplicate elimination.
                samples = _surface_points(
                    vertices[np.asarray(mesh.faces)], point_spacing
                )
                vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
                samples = samples @ transform[:3, :3].T + transform[:3, 3]
                pieces.append(samples)
                part_records.append(
                    {
                        "link": name,
                        "mesh_path": str(mesh_path),
                        "grasp_from_mesh": transform.tolist(),
                        "bounds_m": [
                            vertices.min(0).tolist(),
                            vertices.max(0).tolist(),
                        ],
                        "surface_point_count": len(samples),
                    }
                )
        if not pieces:
            raise ValueError("At least one gripper collision link is required.")
        point_array = np.concatenate(pieces).astype(np.float32)
        return cls(
            points=torch.from_numpy(point_array),
            provenance={
                "urdf_path": str(path),
                "files_sha256": files,
                "root_link": root_link,
                "tcp_link": tcp_link,
                "link_from_tcp": link_from_tcp.tolist(),
                "mesh_paths": {
                    key: str(Path(value).resolve(strict=True))
                    for key, value in (mesh_paths or {}).items()
                },
                "joint_positions": dict(resolved_positions),
                "grasp_from_tcp": grasp_from_tcp.tolist(),
                "point_spacing_m": point_spacing,
                "points_sha256": hashlib.sha256(point_array.tobytes()).hexdigest(),
                "parts": part_records,
                "scope": "fixed_opening_finite_collision_surface_samples",
            },
        )
