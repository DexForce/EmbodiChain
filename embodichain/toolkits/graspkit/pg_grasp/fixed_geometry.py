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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from ._urdf_geometry import _load_collision_geometry, _origin_transform

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


def _resolved_joint_positions(
    named_joints: Mapping[str, ET.Element],
    joint_positions: Mapping[str, float],
    joint_limits: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Resolve one explicit joint state, including declared mimic relations."""
    resolved: dict[str, float] = {}
    resolving: set[str] = set()

    def joint_position(joint: ET.Element) -> float:
        name = joint.attrib["name"]
        if name in resolved:
            return resolved[name]
        if name in resolving:
            raise ValueError(f"Cyclic mimic relation at joint {name!r}.")
        resolving.add(name)
        mimic = joint.find("mimic")
        if mimic is None:
            if name not in joint_positions:
                raise ValueError(f"Explicit state required for joint {name!r}.")
            value = float(joint_positions[name])
        else:
            value = joint_position(named_joints[mimic.attrib["joint"]]) * float(
                mimic.get("multiplier", "1")
            ) + float(mimic.get("offset", "0"))
            if name in joint_positions and not np.isclose(
                joint_positions[name], value, rtol=0, atol=1e-9
            ):
                raise ValueError(f"Declared state contradicts mimic joint {name!r}.")
        limit = joint.find("limit")
        if name in joint_limits:
            lower, upper = joint_limits[name]
        elif joint.attrib["type"] == "continuous":
            lower, upper = -np.inf, np.inf
        elif limit is not None:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
        else:
            lower, upper = np.nan, np.nan
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"Invalid state for joint {name!r}.")
        resolving.remove(name)
        resolved[name] = value
        return value

    for name in joint_positions:
        if name not in named_joints:
            raise ValueError(f"Unknown state joint {name!r}.")
        joint_position(named_joints[name])
    return resolved


def _link_transform_resolver(
    child_joints: Mapping[str, ET.Element],
    root_link: str,
    position_of,
):
    """Return memoized root-from-link transforms for one declared state."""
    transforms = {root_link: np.eye(4)}

    def root_from_link(name: str) -> np.ndarray:
        if name in transforms:
            return transforms[name]
        if name not in child_joints:
            raise ValueError(f"Link {name!r} is not below gripper root {root_link!r}.")
        joint = child_joints[name]
        transform = _origin_transform(joint.find("origin"))
        kind = joint.attrib["type"]
        if kind in {"prismatic", "revolute", "continuous"}:
            position = position_of(joint)
            axis_element = joint.find("axis")
            axis = np.fromstring(
                "1 0 0" if axis_element is None else axis_element.attrib["xyz"],
                sep=" ",
            )
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError(f"Invalid joint axis for {joint.attrib['name']!r}.")
            axis = axis / np.linalg.norm(axis)
            if kind == "prismatic":
                transform[:3, 3] += transform[:3, :3] @ (axis * position)
            else:
                transform[:3, :3] = (
                    transform[:3, :3]
                    @ Rotation.from_rotvec(axis * position).as_matrix()
                )
        elif kind != "fixed":
            raise ValueError("Fixed templates require fixed/prismatic/revolute joints.")
        transforms[name] = (
            root_from_link(joint.find("parent").attrib["link"]) @ transform
        )
        return transforms[name]

    return root_from_link


@dataclass(frozen=True)
class FixedGripperCollisionGeometry:
    """Owned CPU surface samples in the canonical contact/grasp frame.

    A fixed opening checks the physical pre-grasp command declared by the
    caller. Antipodal widths do not reposition these surfaces. Closing contacts
    and arm/environment collisions are separate. ``grasp_from_tcp`` in the
    provenance maps runtime TCP coordinates into canonical contact coordinates.

    Optional closure samples describe the declared finger surfaces at closing
    apertures, so a caller can qualify support clearance during closure instead
    of at the open command alone. Apertures are ascending and aligned with the
    leading axis of ``closure_points``.
    """

    points: torch.Tensor
    provenance: dict[str, Any]
    closure_apertures: torch.Tensor | None = None
    closure_points: torch.Tensor | None = None

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
        apertures = self.closure_apertures
        closure = self.closure_points
        if apertures is None or closure is None:
            object.__setattr__(self, "closure_apertures", None)
            object.__setattr__(self, "closure_points", None)
            return
        if (
            apertures.device.type != "cpu"
            or closure.device.type != "cpu"
            or apertures.dtype != points.dtype
            or closure.dtype != points.dtype
            or apertures.ndim != 1
            or closure.ndim != 3
            or closure.shape[2] != 3
            or closure.shape[0] != apertures.shape[0]
            or closure.shape[1] < 1
            or not len(apertures)
            or not bool(torch.isfinite(apertures).all())
            or not bool(torch.isfinite(closure).all())
        ):
            raise ValueError(
                "Closure samples require finite CPU apertures (S,) and points "
                "(S, N, 3) in the same dtype."
            )
        if bool((apertures < 0).any()) or bool((apertures[1:] <= apertures[:-1]).any()):
            raise ValueError("Closure apertures must be nonnegative and ascending.")
        object.__setattr__(
            self, "closure_apertures", apertures.detach().clone().contiguous()
        )
        object.__setattr__(
            self, "closure_points", closure.detach().clone().contiguous()
        )

    def canonical_to_tcp_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Map canonical contact poses to the declared runtime TCP exactly once.

        Args:
            poses: Canonical contact transforms with trailing shape ``(4, 4)``.

        Returns:
            Runtime TCP poses with the same batch shape, dtype and device.
        """
        return poses @ poses.new_tensor(self.provenance["grasp_from_tcp"])

    def surface_points_at(self, joint_positions: Mapping[str, float]) -> torch.Tensor:
        """Return this hand's complete surfaces at another declared state.

        The sampled mesh points and their collision origins are reused; only the
        declared URDF kinematics are re-evaluated. Closure states therefore add
        no mesh sampling and keep the exact geometry of this fixture.

        Args:
            joint_positions: Explicit active joint state in native joint names.

        Returns:
            Canonical-frame surface points with shape ``(N, 3)``.

        Raises:
            ValueError: The state is incomplete, invalid or contradicts a mimic.
        """
        document = ET.fromstring(Path(self.provenance["urdf_path"]).read_bytes())
        named_joints = {
            joint.attrib["name"]: joint for joint in document.findall("joint")
        }
        child_joints = {
            joint.find("child").attrib["link"]: joint
            for joint in document.findall("joint")
        }
        positions = _resolved_joint_positions(
            named_joints,
            joint_positions,
            {
                name: tuple(bounds)
                for name, bounds in self.provenance.get("joint_limits", {}).items()
            },
        )
        root_from_link = _link_transform_resolver(
            child_joints,
            self.provenance["root_link"],
            lambda joint: positions[joint.attrib["name"]],
        )
        tcp_from_root = np.linalg.inv(
            root_from_link(self.provenance["tcp_link"])
            @ np.asarray(self.provenance["link_from_tcp"], dtype=np.float64)
        )
        fixed = np.asarray(self.provenance["grasp_from_tcp"], dtype=np.float64) @ (
            tcp_from_root
        )
        open_points = self.points.numpy()
        chunks = []
        offset = 0
        for part in self.provenance["parts"]:
            count = int(part["surface_point_count"])
            part_points = open_points[offset : offset + count]
            offset += count
            open_transform = np.asarray(part["grasp_from_mesh"], dtype=np.float64)
            transform = (
                fixed
                @ root_from_link(part["link"])
                @ np.asarray(part["collision_origin"], dtype=np.float64)
            )
            local = (part_points - open_transform[:3, 3]) @ open_transform[:3, :3]
            chunks.append(local @ transform[:3, :3].T + transform[:3, 3])
        if offset != len(open_points):
            raise ValueError("Surface parts do not account for every stored point.")
        return torch.from_numpy(np.concatenate(chunks).astype(np.float32))

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
        joint_limits: Mapping[str, Sequence[float]] | None = None,
    ) -> FixedGripperCollisionGeometry:
        """Sample mesh collisions at a declared joint state below ``root_link``.

        Args:
            urdf_path: URDF with collision mesh paths relative to its directory.
            root_link: Root of the gripper subtree; upstream arm FK is irrelevant.
            tcp_link: Actual runtime TCP in that subtree.
            collision_links: Every gripper link whose collisions must be checked.
            joint_positions: Explicit active positions; mimic joints follow their
                declared multiplier/offset and cannot contradict their leader.
            joint_limits: Optional effective bounds by native joint name, replacing
                source limits for those joints. Each pair must be finite and ordered.
                Unlisted joints retain their URDF limits.
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

        Raises:
            ValueError: Joint bounds or an opening are invalid, or a supplied
                position contradicts the source mimic relation.
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
        effective_limits = {}
        for name, bounds in (joint_limits or {}).items():
            if name not in named_joints or named_joints[name].get("type") not in {
                "revolute",
                "prismatic",
                "continuous",
            }:
                raise ValueError(
                    f"Joint limits require a movable source joint: {name!r}."
                )
            values = np.asarray(bounds, dtype=float)
            if (
                values.shape != (2,)
                or not np.isfinite(values).all()
                or values[0] > values[1]
            ):
                raise ValueError(
                    f"Joint limits for {name!r} must be a finite ordered pair."
                )
            effective_limits[name] = values.tolist()
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
            if name in effective_limits:
                lower, upper = effective_limits[name]
            elif joint.attrib["type"] == "continuous":
                lower, upper = -np.inf, np.inf
            elif limits is not None:
                lower, upper = float(limits.attrib["lower"]), float(
                    limits.attrib["upper"]
                )
            else:
                lower, upper = np.nan, np.nan
            if not np.isfinite(value) or not lower <= value <= upper:
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
            transform = _origin_transform(joint.find("origin"))
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
                vertices, faces, origin, mesh_path = _load_collision_geometry(
                    collision, path.parent, mesh_paths=mesh_paths
                )
                if mesh_path is not None:
                    files[str(mesh_path)] = hashlib.sha256(
                        mesh_path.read_bytes()
                    ).hexdigest()
                transform = (
                    grasp_from_tcp @ tcp_from_root @ root_from_link(name) @ origin
                )
                # Choose the surface grid before rigid transforms. TCP changes
                # must not change ceil(edge/spacing) or duplicate elimination.
                samples = _surface_points(vertices[faces], point_spacing)
                vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
                samples = samples @ transform[:3, :3].T + transform[:3, 3]
                pieces.append(samples)
                part_records.append(
                    {
                        "link": name,
                        "mesh_path": str(mesh_path) if mesh_path is not None else None,
                        "geometry_kind": (
                            "mesh" if mesh_path is not None else "urdf_box"
                        ),
                        "grasp_from_mesh": transform.tolist(),
                        "collision_origin": origin.tolist(),
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
                "joint_limits": effective_limits,
                "grasp_from_tcp": grasp_from_tcp.tolist(),
                "point_spacing_m": point_spacing,
                "points_sha256": hashlib.sha256(point_array.tobytes()).hexdigest(),
                "parts": part_records,
                "scope": "fixed_opening_finite_collision_surface_samples",
            },
        )
