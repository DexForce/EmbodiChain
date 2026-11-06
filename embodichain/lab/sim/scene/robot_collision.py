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

"""Borrowed robot/source and native support geometry for discrete path queries.

These adapters capture facts and validate their owners. They neither plan paths
nor choose contact, recovery or task policies. Authored robot inputs remain
separate from native support bounds; neither certifies backend cooked geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import itertools
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
import torch

from embodichain.compute.geometry.pose_checks import as_rigid_matrix
from embodichain.lab.sim.cfg.source import _asset_file_path
from embodichain.lab.sim.spawn.source_geometry import LinkMesh
from embodichain.lab.sim.utility.urdf_collision import load_urdf_collision_inputs

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["RobotCollisionInputs", "SupportCollisionInputs"]


def _matrix(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4):
        raise ValueError("Collision geometry requires one finite rigid pose.")
    return as_rigid_matrix(result).copy()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_source(entity: object) -> Path | None:
    value = _asset_file_path(entity.cfg)
    return None if value is None else Path(value).resolve()


def _collision_subtree(document: ET.Element, root: str) -> tuple[str, ...]:
    links = document.findall("link")
    if (
        type(root) is not str
        or not root
        or root not in {link.get("name") for link in links}
    ):
        raise ValueError("A collision scope requires an explicit existing motion root.")
    children: dict[str, set[str]] = {}
    for joint in document.findall("joint"):
        children.setdefault(joint.find("parent").attrib["link"], set()).add(
            joint.find("child").attrib["link"]
        )
    selected, pending = {root}, [root]
    while pending:
        for child in children.get(pending.pop(), set()) - selected:
            selected.add(child)
            pending.append(child)
    return tuple(
        link.attrib["name"]
        for link in links
        if link.get("name") in selected and link.findall("collision")
    )


def _native_support(support: object) -> tuple[np.ndarray, float]:
    if support.num_instances != 1 or support.body_type not in {"static", "kinematic"}:
        raise ValueError("Support clearance requires one static/kinematic support.")
    entity = support.body_data.entities[0]
    body = entity.native().get_physical_body()
    shapes = support.get_collision_shapes(env_id=0)
    if body is None or not shapes or len(shapes) != int(body.get_shape_count()):
        raise ValueError("All actual native support colliders are required.")
    root = _matrix(support.get_local_pose(to_matrix=True)[0])
    values = []
    for shape in shapes:
        pose = root @ _matrix(shape.local_pose)
        if shape.vertices is not None:
            vertices = np.asarray(shape.vertices.detach().cpu(), dtype=float)
            faces = (
                None
                if shape.triangles is None
                else shape.triangles.detach().cpu().numpy()
            )
            if (
                faces is None
                or faces.ndim != 2
                or faces.shape[1:] != (3,)
                or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0
                or faces.max() >= len(vertices)
            ):
                raise ValueError("Native support mesh topology is unavailable.")
        elif shape.half_extents is not None:
            half = np.asarray(shape.half_extents.detach().cpu(), dtype=float)
            if half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any():
                raise ValueError("Native support box dimensions are invalid.")
            vertices = np.asarray(list(itertools.product((-1, 1), repeat=3))) * half
        else:
            raise ValueError("Unsupported native support collider.")
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Native support vertices are invalid.")
        values.append(vertices @ pose[:3, :3].T + pose[:3, 3])
    attr = entity.get_physical_attr()
    contact, rest = float(attr.contact_offset), float(attr.rest_offset)
    if not np.isfinite([contact, rest]).all() or contact < max(0.0, rest):
        raise ValueError("Native support contact offsets are invalid.")
    vertices = np.concatenate(values)
    return np.stack((vertices.min(0), vertices.max(0))), contact


class RobotCollisionInputs:
    """Own complete authored collision inputs for one borrowed Default/B1 robot."""

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        *,
        motion_part: str,
        collision_link_names: Sequence[str] | None = None,
    ) -> None:
        """Capture the selected solver-root subtree without changing robot state.

        Args:
            simulation: Exact Default backend owner of the robot.
            robot: Exact managed single-environment robot, borrowed for FK.
            motion_part: Explicit control part defining the solver root and joints.
            collision_link_names: Optional order of the complete collision subtree.
                Omission retains authored URDF order. Partial scopes are rejected.
        """
        if (
            simulation.num_envs != 1
            or robot.num_instances != 1
            or not simulation.is_default_backend
        ):
            raise ValueError(
                "Robot source qualification requires one Default environment."
            )
        if simulation.get_robot(robot.uid) is not robot:
            raise ValueError("The Robot must belong to the supplied native simulation.")
        self._simulation, self._robot = simulation, robot
        self._motion_part = motion_part
        self._robot_path = Path(robot.cfg.fpath).expanduser().resolve(strict=True)
        self._robot_scale = tuple(float(value) for value in robot.cfg.body_scale)
        self._joint_names = tuple(robot.joint_names)
        solver = robot.cfg.solver_cfg[motion_part]
        self._solver_root, self._solver_end = (
            solver.root_link_name,
            solver.end_link_name,
        )
        self._solver_tcp = _matrix(np.eye(4) if solver.tcp is None else solver.tcp)
        document = ET.parse(self._robot_path).getroot()
        required = _collision_subtree(document, self._solver_root)
        if isinstance(collision_link_names, str):
            raise TypeError(
                "collision_link_names must be an explicit sequence, not a string."
            )
        self._names = (
            required if collision_link_names is None else tuple(collision_link_names)
        )
        if set(self._names) != set(required):
            raise ValueError(
                "Collision inputs must cover the complete declared motion subtree: "
                f"missing={sorted(set(required) - set(self._names))}, "
                f"foreign={sorted(set(self._names) - set(required))}."
            )
        self._meshes = load_urdf_collision_inputs(
            self._robot_path,
            self._names,
            body_scale=self._robot_scale,
            vertices_only=True,
        )
        self._motion_joint_ids = tuple(robot.get_joint_ids(motion_part))
        self._motion_joint_names = {
            robot.joint_names[index] for index in self._motion_joint_ids
        }
        self._native_robot = robot.body_data.entities[0]
        paths = {self._robot_path: _digest(self._robot_path)}
        for link in document.findall("link"):
            if link.get("name") not in self._names:
                continue
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is not None:
                    path = (self._robot_path.parent / mesh.attrib["filename"]).resolve()
                    paths[path] = _digest(path)
        self._sources = tuple(paths.items())
        self._closed = False
        self.assert_current()

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed robot while this capture is open."""
        if self._closed:
            raise RuntimeError("Robot collision inputs are closed.")
        return self._robot

    @property
    def link_names(self) -> tuple[str, ...]:
        """Return the complete ordered source collision scope."""
        return self._names

    @property
    def meshes(self) -> Mapping[str, tuple[LinkMesh, ...]]:
        """Borrow immutable authored components without exposing the owner mapping."""
        return MappingProxyType(self._meshes)

    def assert_current(self) -> None:
        """Reject source, native owner, joint-layout or solver-frame drift."""
        if self._closed:
            raise RuntimeError("Robot collision inputs are closed.")
        if (
            self._simulation.num_envs != 1
            or self._robot.num_instances != 1
            or not self._simulation.is_default_backend
        ):
            raise ValueError(
                "Robot source qualification requires one Default environment."
            )
        if (
            self._simulation.get_robot(self._robot.uid) is not self._robot
            or self._robot.body_data.entities[0] is not self._native_robot
        ):
            raise ValueError("A robot collision native owner was replaced.")
        if (
            Path(self._robot.cfg.fpath).resolve() != self._robot_path
            or tuple(self._robot.cfg.body_scale) != self._robot_scale
        ):
            raise ValueError("Robot source path or body scale changed.")
        solver = self._robot.cfg.solver_cfg[self._motion_part]
        if (
            tuple(self._robot.joint_names) != self._joint_names
            or tuple(self._robot.get_joint_ids(self._motion_part))
            != self._motion_joint_ids
            or solver.root_link_name != self._solver_root
            or solver.end_link_name != self._solver_end
            or not np.array_equal(
                _matrix(np.eye(4) if solver.tcp is None else solver.tcp),
                self._solver_tcp,
            )
        ):
            raise ValueError(
                "Robot control layout or solver frame calibration changed."
            )
        for path, expected in self._sources:
            if _digest(path) != expected:
                raise ValueError("Robot collision source bytes changed.")

    def load_meshes(
        self, names: Sequence[str], *, vertices_only: bool = False
    ) -> dict[str, tuple[LinkMesh, ...]]:
        """Load additional source-qualified components in explicit link order.

        Args:
            names: Selected collision links within this complete capture.
            vertices_only: Whether triangle qualification may be omitted.

        Returns:
            Independently owned source components; no FK or surface resampling.
        """
        self.assert_current()
        if not set(names) <= set(self._names):
            raise ValueError(
                "Additional collision inputs must belong to the captured subtree."
            )
        return load_urdf_collision_inputs(
            self._robot_path,
            names,
            body_scale=self._robot_scale,
            vertices_only=vertices_only,
        )

    def require_support_owner(self, support: SupportCollisionInputs) -> None:
        """Reject a support capture borrowed from a different simulation owner.

        Args:
            support: Open support capture to pair with this robot.

        Raises:
            ValueError: The captures belong to different simulation instances.
        """
        if (
            not isinstance(support, SupportCollisionInputs)
            or self._simulation is not support._simulation
        ):
            raise ValueError(
                "Robot and support captures must share the exact simulation owner."
            )

    def moving_link_names(self) -> tuple[str, ...]:
        """Read moving native links after the caller's :meth:`assert_current`.

        This fact read does not hash the same source files again inside a path
        query. The query owner validates the capture before collecting facts.
        """
        robot = self.robot
        native = self._native_robot.render_articulation
        names = []
        for name in self._names:
            if self._motion_joint_names.intersection(
                joint.name for joint in robot.get_parent_joint_chain(name)
            ):
                body = native.get_physical_body(name)
                if body is None:
                    raise ValueError(
                        "A moving source link has no native physical body."
                    )
                if int(body.get_shape_count()) > 0:
                    names.append(name)
        return tuple(names)

    def close(self) -> None:
        """Release owned source buffers without destroying the borrowed robot."""
        self._closed = True
        self._meshes = {}
        self._simulation = self._robot = self._native_robot = None

    def contact_offsets(self, names: tuple[str, ...]) -> dict[str, float]:
        """Read native contact skins after the caller's :meth:`assert_current`.

        Args:
            names: Ordered links within the captured source collision subtree.

        Returns:
            Per-link contact offsets in metres, retaining the native values.
        """
        robot = self.robot
        if not set(names) <= set(self._names):
            raise ValueError("Contact offsets require captured collision links.")
        attrs = robot.get_link_physical_attr(names, env_ids=[0])
        if len(attrs) != len(names):
            raise ValueError("Native offset rows differ from the collider scope.")
        values = {}
        for name, attr in zip(names, attrs, strict=True):
            contact, rest = float(attr.contact_offset), float(attr.rest_offset)
            if not np.isfinite([contact, rest]).all() or contact < max(0.0, rest):
                raise ValueError("Native robot collision offsets are invalid.")
            values[name] = contact
        return values


class SupportCollisionInputs:
    """Borrow one exact static/kinematic support and retain its source identity."""

    def __init__(self, simulation: SimulationManager, support_uid: str) -> None:
        """Bind a support without requiring its native body until a bounds query.

        Args:
            simulation: Exact Default single-environment owner.
            support_uid: Exact configured rigid-object UID; no name is inferred.
        """
        if (
            type(support_uid) is not str
            or not support_uid
            or support_uid != support_uid.strip()
        ):
            raise ValueError("A support query requires an exact native support UID.")
        if simulation.num_envs != 1 or not simulation.is_default_backend:
            raise ValueError(
                "Support source qualification requires one Default environment."
            )
        self._simulation = simulation
        self._uid = support_uid
        self._support = simulation.get_rigid_object(support_uid)
        if self._support is None:
            raise ValueError("The exact native support must be available.")
        self._body_data = self._support.body_data
        self._native = None if self._body_data is None else self._body_data.entities[0]
        self._source = _optional_source(self._support)
        self._source_digest = None if self._source is None else _digest(self._source)
        self._closed = False
        self.assert_current()

    def assert_current(self) -> None:
        """Reject changed backend, support owner or source instead of recapturing."""
        if self._closed:
            raise RuntimeError("Support collision inputs are closed.")
        if self._simulation.num_envs != 1 or not self._simulation.is_default_backend:
            raise ValueError(
                "Support source qualification requires one Default environment."
            )
        if (
            self._simulation.get_rigid_object(self._uid) is not self._support
            or self._support.body_data is not self._body_data
            or (
                self._body_data is not None
                and self._body_data.entities[0] is not self._native
            )
        ):
            raise ValueError("A support collision native owner was replaced.")
        if _optional_source(self._support) != self._source:
            raise ValueError("The bound support source changed.")
        if self._source is not None and _digest(self._source) != self._source_digest:
            raise ValueError("Support collision source bytes changed.")

    def bounds(self) -> tuple[np.ndarray, float]:
        """Return the live collider-union AABB and native contact offset in metres.

        Missing or incomplete native shapes remain unavailable. This conservative
        box does not represent the exact support surface or a collision manifold.

        Returns:
            Owned arena-local minimum/maximum XYZ bounds and the native contact
            offset, both in metres. Source geometry is not substituted for a
            missing native collider.
        """
        self.assert_current()
        return _native_support(self._support)

    def close(self) -> None:
        """Release references without destroying the borrowed native support."""
        self._closed = True
        self._simulation = self._support = self._body_data = self._native = None
