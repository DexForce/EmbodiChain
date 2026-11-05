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

"""Scoped revolute-source geometry queries, independent of Atomic plan objects."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import math
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import QhullError
from scipy.spatial.transform import Rotation
import torch

from embodichain.compute.geometry import (
    axis_contact_pose,
    convex_projection_distance_to_origin,
)
from embodichain.lab.sim.cfg.source import _asset_file_path
from embodichain.lab.sim.scene.control_source import (
    RevoluteSourceSeal,
    qualify_revolute_grip_shell,
)
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from embodichain.lab.sim.scene.revolute_geometry import TwistGeometry
from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding
from embodichain.lab.sim.spawn.source_geometry import LinkMesh
from embodichain.lab.sim.utility.grasp_geometry import (
    RobotHandGeometryCfg,
    _owned_cfg,
    _resolve_hand_frame,
)
from embodichain.lab.sim.utility.urdf_collision import load_urdf_collision_inputs

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "RevolutePathQueryFactory",
    "RevolutePathQuery",
    "RevolutePathReport",
    "RevoluteDependencyReport",
]

_FK_BATCH = 64
_QUERY_ERRORS = (
    ValueError,
    RuntimeError,
    KeyError,
    AttributeError,
    OSError,
    IndexError,
)


def _matrix(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (4, 4)
        or not np.isfinite(result).all()
        or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(
            result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-5, rtol=0
        )
        or not np.isclose(np.linalg.det(result[:3, :3]), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError("Revolute geometry requires a finite rigid pose, not scale.")
    return result.copy()


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


def _native_table(table: object) -> tuple[np.ndarray, float]:
    if table.num_instances != 1 or table.body_type not in {"static", "kinematic"}:
        raise ValueError("Table clearance requires one static/kinematic support.")
    entity = table.body_data.entities[0]
    body = entity.native().get_physical_body()
    shapes = table.get_collision_shapes(env_id=0)
    if body is None or not shapes or len(shapes) != int(body.get_shape_count()):
        raise ValueError("All actual native table colliders are required.")
    root = _matrix(table.get_local_pose(to_matrix=True)[0])
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
                raise ValueError("Native table mesh topology is unavailable.")
        elif shape.half_extents is not None:
            half = np.asarray(shape.half_extents.detach().cpu(), dtype=float)
            if half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any():
                raise ValueError("Native table box dimensions are invalid.")
            vertices = np.asarray(list(itertools.product((-1, 1), repeat=3))) * half
        else:
            raise ValueError("Unsupported native table collider.")
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Native table vertices are invalid.")
        values.append(vertices @ pose[:3, :3].T + pose[:3, 3])
    attr = entity.get_physical_attr()
    contact, rest = float(attr.contact_offset), float(attr.rest_offset)
    if not np.isfinite([contact, rest]).all() or contact < max(0.0, rest):
        raise ValueError("Native table contact offsets are invalid.")
    vertices = np.concatenate(values)
    return np.stack((vertices.min(0), vertices.max(0))), contact


@dataclass(frozen=True, slots=True)
class RevolutePathReport:
    """Declared robot-link vertex envelopes against the actual native table."""

    supported: bool
    accepted: bool
    minimum_clearance_after_skin_m: float | None = None
    checked_frames: int = 0
    moving_links: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.supported) is not bool or type(self.accepted) is not bool:
            raise TypeError("Revolute path report decisions must be booleans.")
        links = tuple(self.moving_links)
        if self.supported:
            value = self.minimum_clearance_after_skin_m
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or self.accepted != (value > 0)
                or self.reason is not None
                or type(self.checked_frames) is not int
                or self.checked_frames <= 0
                or not links
                or len(set(links)) != len(links)
                or any(type(name) is not str or not name for name in links)
            ):
                raise ValueError(
                    "Supported path reports require complete finite evidence."
                )
        elif self.accepted or not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Unsupported path reports require rejection and a reason.")
        object.__setattr__(self, "moving_links", links)

    def metadata(self) -> dict[str, object]:
        """Return an owned source-compatible numeric report."""
        if not self.supported:
            return {"supported": False, "accepted": False, "reason": self.reason}
        return {
            "supported": True,
            "accepted": self.accepted,
            "minimum_clearance_after_skin_m": self.minimum_clearance_after_skin_m,
            "checked_frames": self.checked_frames,
            "moving_links": list(self.moving_links),
        }


@dataclass(frozen=True, slots=True)
class RevoluteDependencyReport:
    """Qualification for an invariant grasp and conservative extra-rotor sweeps.

    This is a source-input command-frame proxy. It does not qualify native
    cooking, continuous collisions or physical effects. The caller owns any
    semantic scene-dependency remapping after qualification succeeds.
    """

    qualified: bool
    source_sha256: str | None = None
    padding_m: float | None = None
    sweep_refinements: tuple[tuple[str, int], ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.qualified) is not bool:
            raise TypeError("Root dependency qualification must be a bool.")
        sweeps = tuple(tuple(row) for row in self.sweep_refinements)
        if self.qualified:
            if (
                not isinstance(self.source_sha256, str)
                or not self.source_sha256
                or type(self.padding_m) not in (int, float)
                or not math.isfinite(self.padding_m)
                or self.padding_m < 0
                or self.reason is not None
                or any(
                    len(row) != 2
                    or type(row[0]) is not str
                    or not row[0]
                    or type(row[1]) is not int
                    or row[1] < 0
                    for row in sweeps
                )
            ):
                raise ValueError(
                    "Qualified dependencies require finite source evidence."
                )
        elif not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Unqualified dependencies require a reason.")
        object.__setattr__(self, "sweep_refinements", sweeps)

    def metadata(self) -> dict[str, object]:
        """Return owned numeric evidence without semantic dependency IDs."""
        if not self.qualified:
            return {"qualified": False, "reason": self.reason}
        return {
            "qualified": True,
            "source_sha256": self.source_sha256,
            "padding_m": self.padding_m,
            "extra_sweeps": [
                {"path": path, "projection_refinements": count, "separated": True}
                for path, count in self.sweep_refinements
            ],
            "scope": "source-input command-frame proxy; not native cooking or continuous collision certification",
        }


class RevolutePathQueryFactory:
    """Own static source preparation for exact borrowed native/Robot instances.

    Construct once for a configured control operation. Authored robot meshes and
    source hashes are retained across observation queries; each query has fresh
    explicit poses and scope. Source changes are rejected, never recaptured as
    a new baseline. This factory has no skill registry or planning context.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        *,
        binding: RevoluteControlBinding,
        geometry: TwistGeometry,
        source_seal: RevoluteSourceSeal,
        hand_geometry_cfg: RobotHandGeometryCfg,
        table_uid: str,
        collision_link_names: Sequence[str] | None = None,
    ) -> None:
        """Capture declared collision links without changing native state.

        Args:
            simulation: Exact Default backend owner of all supplied runtime IDs.
            robot: Exact managed B1 robot, borrowed for FK and physical attributes.
            binding: Qualified source IDs, calibration and revolute geometry.
            geometry: Read-only authored collider inputs from the same source.
            source_seal: Qualified asset/calibration/mass lineage identities.
            hand_geometry_cfg: Explicit motion/hand/root/ordered-pad calibration.
            table_uid: Exact native rigid-object UID for the table screen.
            collision_link_names: Optional explicit ordering of the complete
                configured motion-root collision subtree. Partial/foreign scopes
                are rejected. Omission retains URDF order, without name inference.
        """
        if (
            type(table_uid) is not str
            or not table_uid
            or table_uid != table_uid.strip()
        ):
            raise ValueError("A revolute query requires an exact native table UID.")
        if (
            not isinstance(binding, RevoluteControlBinding)
            or not isinstance(geometry, TwistGeometry)
            or not isinstance(source_seal, RevoluteSourceSeal)
        ):
            raise TypeError("Revolute queries require typed qualified source inputs.")
        if (
            simulation.num_envs != 1
            or robot.num_instances != 1
            or not simulation.is_default_backend
        ):
            raise ValueError(
                "Revolute source qualification requires one Default environment."
            )
        if simulation.get_robot(robot.uid) is not robot:
            raise ValueError("The Robot must belong to the supplied native simulation.")
        if (
            geometry.source_sha256 != binding.source_sha256
            or source_seal.source_sha256 != binding.source_sha256
            or source_seal.calibration_sha256 != binding.calibration_sha256
            or source_seal.mass_lineage_sha256 != binding.mass_lineage_sha256
            or geometry.scale != binding.scale
            or source_seal.scale != binding.scale
        ):
            raise ValueError(
                "Revolute source inputs have different identities or scales."
            )
        self._simulation, self._robot = simulation, robot
        self._binding, self._geometry, self._seal = binding, geometry, source_seal
        self._cfg = _owned_cfg(hand_geometry_cfg)
        self._robot_path, _, hand_joints, solver = _resolve_hand_frame(robot, self._cfg)
        self._robot_scale = tuple(float(value) for value in robot.cfg.body_scale)
        self._joint_names = tuple(robot.joint_names)
        self._hand_joints = tuple(hand_joints)
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
        self._pad_metric_meshes = None
        self._meshes = load_urdf_collision_inputs(
            self._robot_path,
            self._names,
            body_scale=self._robot_scale,
            vertices_only=True,
        )
        if not set(self._cfg.pad_link_names) <= set(self._names):
            raise ValueError(
                "The declared robot collision scope must include both pads."
            )
        self._motion_joint_ids = tuple(robot.get_joint_ids(self._cfg.motion_part))
        self._motion_joint_names = {
            robot.joint_names[index] for index in self._motion_joint_ids
        }
        self._art = simulation.get_articulation(binding.object_id)
        self._table_uid, self._table = table_uid, simulation.get_rigid_object(table_uid)
        if self._art is None or self._table is None:
            raise ValueError(
                "The exact native articulation and table must be available."
            )
        self._native_robot = robot.body_data.entities[0]
        self._native_art = self._art.body_data.entities[0]
        self._table_body_data = self._table.body_data
        self._native_table = (
            None if self._table_body_data is None else self._table_body_data.entities[0]
        )
        self._table_source = _optional_source(self._table)
        paths = {self._robot_path: _digest(self._robot_path)}
        for link in document.findall("link"):
            if link.get("name") not in self._names:
                continue
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is not None:
                    path = (self._robot_path.parent / mesh.attrib["filename"]).resolve()
                    paths[path] = _digest(path)
        if self._table_source is not None:
            paths[self._table_source] = _digest(self._table_source)
        self._sources = tuple(paths.items())
        self._closed = False
        self._assert_current()
        try:
            self._shell = qualify_revolute_grip_shell(geometry, binding)
            self._shell_reason = None
        except (ValueError, AttributeError, IndexError, TypeError) as error:
            self._shell, self._shell_reason = None, str(error)

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed Robot, without resolving a global manager."""
        if self._closed:
            raise RuntimeError("Revolute query factory is closed.")
        return self._robot

    def _assert_current(self) -> None:
        if self._closed:
            raise RuntimeError("Revolute query factory is closed.")
        if (
            self._simulation.num_envs != 1
            or self._robot.num_instances != 1
            or not self._simulation.is_default_backend
        ):
            raise ValueError(
                "Revolute source qualification requires one Default environment."
            )
        if (
            self._simulation.get_robot(self._robot.uid) is not self._robot
            or self._simulation.get_articulation(self._binding.object_id)
            is not self._art
            or self._simulation.get_rigid_object(self._table_uid) is not self._table
            or self._robot.body_data.entities[0] is not self._native_robot
            or self._art.body_data.entities[0] is not self._native_art
            or self._table.body_data is not self._table_body_data
            or (
                self._table_body_data is not None
                and self._table_body_data.entities[0] is not self._native_table
            )
        ):
            raise ValueError("A revolute query native owner was replaced.")
        if (
            Path(self._robot.cfg.fpath).resolve() != self._robot_path
            or tuple(self._robot.cfg.body_scale) != self._robot_scale
        ):
            raise ValueError("Robot source path or body scale changed.")
        solver = self._robot.cfg.solver_cfg[self._cfg.motion_part]
        if (
            tuple(self._robot.joint_names) != self._joint_names
            or tuple(self._robot.get_joint_ids(self._cfg.motion_part))
            != self._motion_joint_ids
            or tuple(self._robot.cfg.control_parts[self._cfg.hand_part])
            != self._hand_joints
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
        if _optional_source(self._table) != self._table_source:
            raise ValueError("The bound table source changed.")
        self._seal.assert_current(
            source_path=_asset_file_path(self._art.cfg),
            body_scale=self._art.cfg.body_scale,
            body_scale_mass_policy=getattr(
                self._art.cfg, "body_scale_mass_policy", "native"
            ),
        )
        for path, expected in self._sources:
            if _digest(path) != expected:
                raise ValueError("Robot/table collision source bytes changed.")

    def prepare(
        self,
        *,
        scope: NativeQueryScope,
        robot_root_pose: torch.Tensor,
        parent_pose: torch.Tensor,
        child_pose: torch.Tensor,
        measured_joint_position: float,
    ) -> RevolutePathQuery:
        """Prepare a query from one explicit same-observation pose/joint snapshot.

        Root and child poses must be the registered observations for the bound
        native parent/child. Frame indices and hypothetical full joint paths are
        supplied later, so preview paths do not require mutating live robot state.
        The host closes the query before advancing the physical observation.

        Args:
            scope: Fresh preparation ID, authoritative observation time and
                scene version for native environment zero.
            robot_root_pose: Observed arena-local robot root, shape ``(4, 4)``.
            parent_pose: Registered observed source-parent pose, shape ``(4, 4)``.
            child_pose: Registered observed source-child pose, shape ``(4, 4)``.
            measured_joint_position: Same-observation revolute coordinate in radians.

        Returns:
            A separately owned query lifetime borrowing this factory's static inputs.
        """
        self._assert_current()
        return RevolutePathQuery(
            self,
            scope,
            robot_root_pose,
            parent_pose,
            child_pose,
            measured_joint_position,
        )

    def close(self) -> None:
        """Release owned source buffers; never destroy borrowed native entities."""
        self._closed = True
        self._meshes = {}
        self._pad_metric_meshes = None
        self._geometry = None
        self._simulation = self._robot = self._art = self._table = None
        self._native_robot = self._native_art = self._native_table = None
        self._table_body_data = None


class RevolutePathQuery:
    """One B1/env0 observation query for a prepared source geometry owner."""

    def __init__(
        self,
        factory: RevolutePathQueryFactory,
        scope: NativeQueryScope,
        robot_root_pose: torch.Tensor,
        parent_pose: torch.Tensor,
        child_pose: torch.Tensor,
        measured_joint_position: float,
    ) -> None:
        """Bind explicit observation values to a prepared geometry factory.

        Prefer :meth:`RevolutePathQueryFactory.prepare`, which checks current
        ownership before constructing this value. Poses are copied independently.
        """
        if not isinstance(factory, RevolutePathQueryFactory):
            raise TypeError("factory must be RevolutePathQueryFactory.")
        if not isinstance(scope, NativeQueryScope):
            raise TypeError("Revolute query scope must be NativeQueryScope.")
        if type(measured_joint_position) not in (int, float) or not math.isfinite(
            measured_joint_position
        ):
            raise ValueError("Revolute joint observation must be a finite scalar.")
        self._factory, self._scope = factory, scope
        self._root = _matrix(robot_root_pose)
        self._parent, self._child = _matrix(parent_pose), _matrix(child_pose)
        self._measured = float(measured_joint_position)
        self._closed = False

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed Robot for planning-service qualification."""
        if self._closed:
            raise RuntimeError("Revolute query is closed.")
        return self._factory.robot

    @property
    def scope(self) -> NativeQueryScope:
        """Return this preparation's immutable query/observation identity."""
        return self._scope

    def _fresh(self) -> None:
        if self._closed:
            raise RuntimeError("Revolute query is closed; prepare a fresh observation.")
        self._factory._assert_current()
        if not np.array_equal(
            _matrix(self.robot.get_local_pose(to_matrix=True)[0]), self._root
        ):
            raise ValueError("The observed robot root frame changed during the query.")

    def _clouds(
        self,
        positions: torch.Tensor,
        names: tuple[str, ...],
        frames: list[int],
        inverse: np.ndarray | None = None,
    ) -> Iterator[tuple[int, str, LinkMesh, torch.Tensor]]:
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[2] != len(self.robot.joint_names)
            or not torch.isfinite(positions).all()
        ):
            raise ValueError(
                "Revolute geometry requires a finite full-robot trajectory."
            )
        if (
            not frames
            or any(
                type(index) is not int or not 0 <= index < positions.shape[1]
                for index in frames
            )
            or len(set(frames)) != len(frames)
        ):
            raise ValueError(
                "Collision frame indices must be nonempty, exact and unique."
            )
        meshes = self._factory._meshes
        expected = {
            (index, name, mesh.path)
            for index in frames
            for name in names
            for mesh in meshes[name]
        }
        count = len(frames) * sum(len(meshes[name]) for name in names)
        if not expected or len(expected) != count:
            raise ValueError("Collision scope is empty or duplicated.")
        seen = set()
        for indices in torch.tensor(frames, device=positions.device).split(_FK_BATCH):
            fk = self.robot.compute_fk(
                qpos=positions[0, indices],
                link_names=list(names),
                qpos_joint_names=self.robot.joint_names,
            )
            if (
                fk.shape != (len(indices), len(names), 4, 4)
                or not torch.isfinite(fk).all()
            ):
                raise ValueError("Full FK differs from the declared collider scope.")
            poses = (
                torch.as_tensor(self._root, dtype=fk.dtype, device=fk.device)[
                    None, None
                ]
                @ fk
            )
            if inverse is not None:
                poses = (
                    torch.as_tensor(inverse, dtype=fk.dtype, device=fk.device)[
                        None, None
                    ]
                    @ poses
                )
            for column, name in enumerate(names):
                for mesh in meshes[name]:
                    vertices = positions.new_tensor(mesh.vertices.copy())
                    points = (
                        vertices[None] @ poses[:, column, :3, :3].transpose(-1, -2)
                        + poses[:, column, None, :3, 3]
                    )
                    for index, value in zip(indices.tolist(), points, strict=True):
                        key = (index, name, mesh.path)
                        if (
                            key not in expected
                            or key in seen
                            or value.ndim != 2
                            or value.shape[1:] != (3,)
                            or not len(value)
                            or not torch.isfinite(value).all()
                        ):
                            raise ValueError(
                                "Collision frame coverage or transformed vertices are invalid."
                            )
                        seen.add(key)
                        yield index, name, mesh, value
        if seen != expected:
            raise ValueError("Collision frame coverage is incomplete.")

    def _offsets(self, names: tuple[str, ...]) -> dict[str, float]:
        attrs = self.robot.get_link_physical_attr(names, env_ids=[0])
        if len(attrs) != len(names):
            raise ValueError("Native offset rows differ from the collider scope.")
        values = {}
        for name, attr in zip(names, attrs, strict=True):
            contact, rest = float(attr.contact_offset), float(attr.rest_offset)
            if not np.isfinite([contact, rest]).all() or contact < max(0.0, rest):
                raise ValueError("Native robot collision offsets are invalid.")
            values[name] = contact
        return values

    @property
    def source_axis(self) -> tuple[float, float, float]:
        """Return the original double-precision calibrated contact axis."""
        self._fresh()
        return self._factory._binding.axis

    def contact_metrics(
        self, full_qpos: torch.Tensor, start_qpos: torch.Tensor
    ) -> tuple[float, float, float]:
        """Measure pad/parent proximity at one candidate contact posture.

        Args:
            full_qpos: Full-robot close-end posture, shape ``(1, J)``.
            start_qpos: Same-observation start used for the joint-distance metric.

        Returns:
            Parent-near vertex fraction, maximum pad-to-grip distance, and joint
            displacement norm. The source 128-point grip/parent sampling and
            complete ordered pad inputs are retained. No ranking occurs here.
        """
        self._fresh()
        if (
            full_qpos.ndim != 2
            or full_qpos.dtype != torch.float32
            or full_qpos.shape[0] != 1
            or full_qpos.shape[1] != len(self.robot.joint_names)
            or start_qpos.shape != full_qpos.shape
            or not torch.isfinite(full_qpos).all()
            or not torch.isfinite(start_qpos).all()
        ):
            raise ValueError(
                "Contact metrics require native float32 finite B1 full-robot postures."
            )
        factory = self._factory
        geometry = factory._geometry
        if factory._pad_metric_meshes is None:
            # The source score uses triangle-qualified pad inputs, unlike the
            # arm vertex envelopes used by table/sweep bounds.
            factory._pad_metric_meshes = load_urdf_collision_inputs(
                factory._robot_path,
                factory._cfg.pad_link_names,
                body_scale=factory._robot_scale,
                vertices_only=False,
            )

        def sample(points: object) -> torch.Tensor:
            values = full_qpos.new_tensor(points)
            if (
                values.ndim != 2
                or values.shape[1] != 3
                or not len(values)
                or not torch.isfinite(values).all()
            ):
                raise ValueError(
                    "Contact scoring requires complete finite source point sets."
                )
            if values.shape[0] > 128:
                ids = (
                    torch.linspace(0, values.shape[0] - 1, 128, device=values.device)
                    .round()
                    .long()
                )
                values = values[ids]
            return values

        target = sample(geometry.grip.vertices.copy())
        parent = sample(
            [
                point
                for mesh in geometry.parent_collisions
                for point in mesh.vertices.tolist()
            ]
        )
        child_pose = full_qpos.new_tensor(self._child)
        parent_pose = full_qpos.new_tensor(self._parent)
        # Preserve the source's child-relative round trip, including precision.
        target_to_parent = torch.linalg.inv(child_pose) @ parent_pose
        parent_pose = child_pose @ target_to_parent
        target_world = target @ child_pose[:3, :3].T + child_pose[:3, 3]
        parent_world = parent @ parent_pose[:3, :3].T + parent_pose[:3, 3]
        names = factory._cfg.pad_link_names
        pads = self.robot.compute_fk(
            qpos=full_qpos,
            link_names=list(names),
            qpos_joint_names=self.robot.joint_names,
        )
        pads = full_qpos.new_tensor(self._root)[None, None].to(pads) @ pads
        distances, near = [], []
        for index, name in enumerate(names):
            vertices = full_qpos.new_tensor(
                [
                    point
                    for mesh in factory._pad_metric_meshes[name]
                    for point in mesh.vertices.tolist()
                ]
            )
            pose = pads[0, index]
            world = vertices.to(pose) @ pose[:3, :3].T + pose[:3, 3]
            distances.append(torch.cdist(world, target_world).amin())
            near.append(torch.cdist(world, parent_world).amin(-1))
        return (
            float(torch.cat(near).le(0.003).float().mean()),
            float(torch.stack(distances).amax()),
            float(torch.linalg.vector_norm(full_qpos - start_qpos).item()),
        )

    def inspect_free_path(
        self, positions: torch.Tensor, frame_indices: Sequence[int]
    ) -> RevolutePathReport:
        """Inspect caller-selected full command frames against the native table.

        The caller owns phase selection. This query preserves declared link and
        frame order, checks every source collider component, and retains the
        source 64-frame FK memory bound. It checks table clearance only.

        Args:
            positions: Finite full-robot joint samples, shape ``(1, T, robot.dof)``.
            frame_indices: Unique explicit rows covering the caller's free phases.

        Returns:
            Numeric clearance or explicit unavailable evidence. No scene state
            or native object is modified.

        Raises:
            RuntimeError: This query has already been closed.
        """
        if self._closed:
            raise RuntimeError("Revolute query is closed.")
        try:
            self._fresh()
            table, table_contact = _native_table(self._factory._table)
            frames = list(frame_indices)
            active = self._factory._motion_joint_names
            native = self._factory._native_robot.render_articulation
            names = []
            for name in self._factory._names:
                if active.intersection(
                    joint.name for joint in self.robot.get_parent_joint_chain(name)
                ):
                    body = native.get_physical_body(name)
                    if body is None:
                        raise ValueError(
                            "A moving source link has no native physical body."
                        )
                    if int(body.get_shape_count()) > 0:
                        names.append(name)
            names = tuple(names)
            if not names or not set(self._factory._cfg.pad_link_names) <= set(names):
                raise ValueError("Moving scope omits the declared pad colliders.")
            offsets = self._offsets(names)
            minimum = float("inf")
            for _, name, _, points in self._clouds(positions, names, frames):
                low, high = points.amin(0), points.amax(0)
                bounds = points.new_tensor(table)
                gap = torch.maximum(bounds[0] - high, low - bounds[1]).clamp_min(0)
                minimum = min(
                    minimum,
                    float(torch.linalg.vector_norm(gap))
                    - offsets[name]
                    - table_contact,
                )
            if not math.isfinite(minimum):
                raise ValueError("Table clearance lacks finite complete evidence.")
            return RevolutePathReport(True, minimum > 0, minimum, len(frames), names)
        except _QUERY_ERRORS as error:
            return RevolutePathReport(False, False, reason=str(error))

    def qualify_root_dependency(
        self,
        positions: torch.Tensor,
        *,
        grasp_point: Sequence[float],
        axis: torch.Tensor,
        roll: float,
    ) -> RevoluteDependencyReport:
        """Qualify an invariant source grasp without changing any semantic plan.

        Args:
            positions: Full-robot samples, shape ``(1, T, robot.dof)``.
            grasp_point: Finite target-local contact point used by the source grasp.
            axis: Target-local contact Z axis, shape ``(3,)``.
            roll: Declared contact-frame roll in radians.

        Returns:
            Qualification from the source shell, nine hinge-angle frame samples
            and complete extra-rotor sweep projections. Unsupported geometry
            retains an explicit reason; the caller keeps its original dependency.

        Raises:
            RuntimeError: This query has already been closed.
        """
        if self._closed:
            raise RuntimeError("Revolute query is closed.")
        try:
            self._fresh()
            factory = self._factory
            if factory._shell is None:
                raise ValueError(factory._shell_reason)
            binding, art, entity = factory._binding, factory._art, factory._native_art
            if (
                entity.get_root_link_name() != binding.parent
                or entity.render_articulation is None
            ):
                raise ValueError(
                    "The grasp parent is not the actual native articulation root."
                )
            joint = entity.render_articulation.get_joint_info(binding.joint)
            if joint is None:
                raise ValueError("Native revolute hinge information is unavailable.")
            first, second = _matrix(joint.origin_pose), _matrix(joint.target_pose)
            joint_axis = np.asarray(joint.axis, dtype=float).copy()
            if (
                joint_axis.shape != (3,)
                or not np.isfinite(joint_axis).all()
                or np.linalg.norm(joint_axis) <= 0
            ):
                raise ValueError("Native hinge axis is invalid.")
            joint_axis /= np.linalg.norm(joint_axis)
            parent = _matrix(art.get_link_pose(binding.parent, to_matrix=True)[0])
            child = _matrix(art.get_link_pose(binding.link, to_matrix=True)[0])
            if not np.allclose(
                self._parent, parent, rtol=0, atol=1e-5
            ) or not np.allclose(self._child, child, rtol=0, atol=1e-5):
                raise ValueError(
                    "Registered observations differ from native parent/child frames."
                )
            live = art.get_qpos()
            if (
                live.ndim != 2
                or live.shape[0] != 1
                or not torch.isfinite(live).all()
                or abs(
                    float(live[0, art.joint_names.index(binding.joint)])
                    - self._measured
                )
                > 1e-7
            ):
                raise ValueError("The resolved joint observation is stale or invalid.")
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(joint_axis * self._measured).as_matrix()
            expected, actual = (
                first @ turn @ np.linalg.inv(second),
                np.linalg.inv(parent) @ child,
            )
            if (
                np.linalg.norm(actual[:3, 3] - expected[:3, 3]) > 1e-5
                or Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude()
                > 1e-4
            ):
                raise ValueError(
                    "Native hinge does not predict the observed child pose."
                )
            native_axis, origin = second[:3, :3] @ joint_axis, second[:3, 3]
            native_axis /= np.linalg.norm(native_axis)
            if (
                np.linalg.norm(np.cross(np.asarray(grasp_point) - origin, native_axis))
                > 2e-6
            ):
                raise ValueError("The selected grasp is not on the native hinge axis.")
            grasp = axis_contact_pose(
                torch.as_tensor(child, dtype=torch.float32)[None],
                axis,
                grasp_point,
                roll,
            )
            for angle in np.linspace(*binding.limits, 9):
                turn[:3, :3] = Rotation.from_rotvec(joint_axis * angle).as_matrix()
                pose = parent @ first @ turn @ np.linalg.inv(second)
                if not torch.allclose(
                    grasp,
                    axis_contact_pose(
                        torch.as_tensor(pose, dtype=torch.float32)[None],
                        axis,
                        grasp_point,
                        roll,
                    ),
                    atol=2e-5,
                    rtol=0,
                ):
                    raise ValueError(
                        "The grasp point/basis is not rotationally invariant."
                    )
            reference = np.eye(3)[int(np.argmin(abs(native_axis)))]
            x = np.cross(native_axis, reference)
            x /= np.linalg.norm(x)
            basis = np.column_stack((x, np.cross(native_axis, x), native_axis))
            rotor = art.get_link_physical_attr(binding.link, env_ids=[0])[0]
            rotor_contact, rotor_rest = float(rotor.contact_offset), float(
                rotor.rest_offset
            )
            if not np.isfinite(
                [rotor_contact, rotor_rest]
            ).all() or rotor_contact < max(0.0, rotor_rest):
                raise ValueError("Native rotor collision offsets are invalid.")
            padding = max(self._offsets(factory._names).values()) + rotor_contact
            sweeps = []
            for mesh in factory._geometry.target_collisions:
                if mesh.path == factory._geometry.grip.path:
                    continue
                extra = (mesh.vertices - origin) @ basis
                sweeps.append(
                    {
                        "path": mesh.path,
                        "radius": float(np.linalg.norm(extra[:, :2], axis=1).max())
                        + padding,
                        "low": float(extra[:, 2].min()) - padding,
                        "high": float(extra[:, 2].max()) + padding,
                        "projection_refinements": 0,
                    }
                )
            clouds = (
                self._clouds(
                    positions,
                    factory._names,
                    list(range(positions.shape[1])),
                    np.linalg.inv(child),
                )
                if sweeps
                else ()
            )
            for _, _, _, points in clouds:
                local = points.detach().cpu().numpy()
                lo, hi = local.min(0), local.max(0)
                box = np.asarray(
                    [
                        [
                            lo[0] if index & 1 else hi[0],
                            lo[1] if index & 2 else hi[1],
                            lo[2] if index & 4 else hi[2],
                        ]
                        for index in range(8)
                    ]
                )
                box = (box - origin) @ basis
                minimum, maximum = box.min(0), box.max(0)
                radial = np.linalg.norm(
                    np.maximum(np.maximum(minimum[:2], -maximum[:2]), 0)
                )
                values = None
                for sweep in sweeps:
                    low, high, radius = sweep["low"], sweep["high"], sweep["radius"]
                    if maximum[2] < low or minimum[2] > high or radial > radius:
                        continue
                    sweep["projection_refinements"] += 1
                    if values is None:
                        values = (local - origin) @ basis
                    if not (
                        values[:, 2].max() < low
                        or values[:, 2].min() > high
                        or convex_projection_distance_to_origin(values[:, :2]) > radius
                    ):
                        raise ValueError(
                            "An extra rotor sweep remains unproven against a planned collider."
                        )
            return RevoluteDependencyReport(
                True,
                binding.source_sha256,
                padding,
                tuple(
                    (sweep["path"], sweep["projection_refinements"]) for sweep in sweeps
                ),
            )
        except (*_QUERY_ERRORS, QhullError, np.linalg.LinAlgError) as error:
            return RevoluteDependencyReport(False, reason=str(error))

    def close(self) -> None:
        """End this query without destroying its borrowed native resources."""
        self._closed = True
        self._factory = None

    def __enter__(self) -> RevolutePathQuery:
        self._fresh()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
