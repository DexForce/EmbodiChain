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
"""Instance-owned E8 source geometry screens and qualified dependencies."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import itertools
import math
from pathlib import Path
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
import torch

from .twist_binding import TwistRoute
from .twist_geometry import LinkMesh, load_robot_link_meshes

__all__: list[str] = []

_FREE_PHASES = ("approach", "reach", "retract")
_FK_BATCH = 64


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _matrix(value: Any) -> np.ndarray:
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
        raise ValueError("E8 geometry requires a finite rigid pose, not scale.")
    return result


def _native_table(table: Any) -> tuple[np.ndarray, float]:
    if table.num_instances != 1 or table.body_type not in {"static", "kinematic"}:
        raise ValueError("E8 table screen requires one static/kinematic support.")
    entity = table.body_data.entities[0]
    body = entity.native().get_physical_body()
    shapes = table.get_collision_shapes(env_id=0)
    if body is None or not shapes or len(shapes) != int(body.get_shape_count()):
        raise ValueError("E8 requires all actual native table colliders.")
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


def _segment_distance(start: np.ndarray, stop: np.ndarray) -> float:
    delta = stop - start
    denominator = float(delta @ delta)
    fraction = (
        float(np.clip(-(start @ delta) / denominator, 0, 1)) if denominator else 0.0
    )
    return float(np.linalg.norm(start + fraction * delta))


def _radial_distance(points: np.ndarray) -> float:
    """Conservative distance to the complete convex projection, including degeneracies."""
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if (
        points.ndim != 2
        or points.shape[1:] != (2,)
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Invalid rotor-axis projection.")
    allowance = max(1e-12, float(np.linalg.norm(points, axis=1).max()) * 1e-12)
    if len(points) == 1:
        distance = float(np.linalg.norm(points[0]))
    elif len(points) == 2:
        distance = _segment_distance(*points)
    else:
        center = points.mean(0)
        _, singular, directions = np.linalg.svd(points - center, full_matrices=False)
        if singular[1] <= max(1e-14, singular[0] * 1e-12):
            direction = directions[0]
            along = (points - center) @ direction
            residual = points - center - along[:, None] * direction
            distance = _segment_distance(
                center + along.min() * direction, center + along.max() * direction
            ) - float(np.linalg.norm(residual, axis=1).max())
        else:
            hull = ConvexHull(points)
            if np.all(hull.equations[:, 2] <= allowance):
                return 0.0
            polygon = points[hull.vertices]
            distance = min(
                _segment_distance(a, b)
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
            )
    return max(0.0, distance - allowance)


def _cylinder_shell(geometry: Any, binding: Any) -> dict[str, Any]:
    import trimesh

    if geometry.source_sha256 != binding.source_sha256:
        raise ValueError("E8 geometry differs from its calibrated source.")
    axis, origin = np.asarray(binding.axis), np.asarray(binding.origin)
    if not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6):
        raise ValueError("E8 requires a unit source hinge axis.")
    vertices = geometry.grip.vertices - origin
    axial = vertices @ axis
    radial = vertices - axial[:, None] * axis
    radius = np.linalg.norm(radial, axis=1)
    tolerance = max(1e-7, float(radius.max()) * 1e-5)
    if radius.min() <= tolerance or np.ptp(radius) > tolerance:
        raise ValueError("Grip shell is not a circular prism.")
    levels = []
    for value in sorted(axial):
        if not levels or abs(value - levels[-1]) > tolerance:
            levels.append(float(value))
    if len(levels) != 2:
        raise ValueError("Grip shell requires exactly two axial rings.")
    reference = np.eye(3)[int(np.argmin(abs(axis)))]
    x = np.cross(axis, reference)
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    for level in levels:
        ring = radial[np.abs(axial - level) <= tolerance]
        angles = np.unique(
            np.round(np.mod(np.arctan2(ring @ y, ring @ x), 2 * math.pi), 6)
        )
        if len(angles) > 1 and angles[-1] - angles[0] > 2 * math.pi - 1e-5:
            angles = angles[:-1]
        gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
        if len(angles) < 16 or np.max(np.abs(gaps - 2 * math.pi / len(angles))) > 1e-4:
            raise ValueError("Grip shell has incomplete/nonuniform angular coverage.")
    surface = trimesh.Trimesh(
        vertices=geometry.grip.vertices.copy(),
        faces=geometry.grip.faces.copy(),
        process=True,
    )
    if (
        not surface.is_watertight
        or not surface.is_winding_consistent
        or not surface.is_convex
        or surface.euler_number != 2
    ):
        raise ValueError("Grip shell lacks closed convex triangle topology.")
    for mesh in geometry.target_collisions:
        if mesh.path != geometry.grip.path:
            other = mesh.vertices - origin
            other = other - (other @ axis)[:, None] * axis
            if np.linalg.norm(other, axis=1).max() >= radius.min() - 4 * tolerance:
                raise ValueError(
                    "Extra rotor collider reaches the qualified grip shell."
                )
    return {"radius_m": float(radius.max()), "source_sha256": binding.source_sha256}


def _remap_dependency(plan: Any, child_id: str, parent_id: str) -> Any:
    if child_id not in plan.scene_dependencies:
        return plan
    dependencies = tuple(
        sorted(
            {
                parent_id if name == child_id else name
                for name in plan.scene_dependencies
            }
        )
    )
    cutoff = dict(plan.scene_dependency_monitor_until)
    child_cutoff, parent_cutoff = cutoff.pop(child_id, None), cutoff.pop(
        parent_id, None
    )
    if child_cutoff is not None and (
        parent_id not in plan.scene_dependencies or parent_cutoff is not None
    ):
        cutoff[parent_id] = max(child_cutoff, parent_cutoff or 0)
    return replace(
        plan, scene_dependencies=dependencies, scene_dependency_monitor_until=cutoff
    )


class E8GeometryGuard:
    """Borrow the E8 acceptance sensor; never step or mutate native state."""

    def __init__(
        self, route: TwistRoute, simulation: Any, robot: Any, sensor: Any
    ) -> None:
        self.route, self.simulation, self.robot, self.sensor = (
            route,
            simulation,
            robot,
            sensor,
        )
        self.binding = route.binding
        self.art = simulation.get_articulation(self.binding.object_id)
        if sensor.art is not self.art or sensor.route != route:
            raise ValueError(
                "E8 geometry guard requires the existing route-owned sensor."
            )
        self.geometry = sensor.geometry
        self._asset_path = Path(self.art.cfg.fpath).resolve()
        self._robot_path = Path(robot.cfg.fpath).resolve()
        self._robot_scale = tuple(float(value) for value in robot.cfg.body_scale)
        self._sources = {self._asset_path: self.binding.source_sha256}
        self._semantics_sidecar = self._asset_path.with_suffix(".twist.json")
        self._semantics_sidecar_digest = None
        self._mass_source_sidecar = None
        self._mass_source_sidecar_digest = None
        self._mass_manifest_path = self._asset_path.with_suffix(".mass.json")
        self._names: tuple[str, ...] = ()
        self._meshes: dict[str, tuple[LinkMesh, ...]] = {}
        self._setup_error: str | None = None
        self.table = simulation.get_rigid_object("table")
        self._table_path = None
        try:
            from .twist_semantics import resolve_twist_semantics

            semantics = resolve_twist_semantics(self._asset_path)
            if semantics.semantics_sha256 != self.binding.calibration_sha256:
                raise ValueError("E8 geometry semantics differ from the bound roles.")
            self._semantics_sidecar_digest = semantics.sidecar_sha256
            from .twist_mass_source import qualify_mass_source_copy

            lineage_digest = getattr(self.binding, "mass_lineage_sha256", "")
            if lineage_digest:
                mass_copy = qualify_mass_source_copy(
                    self._asset_path, manifest_sha256=lineage_digest
                )
                if (
                    mass_copy.scale != self.binding.scale
                    or getattr(self.art.cfg, "body_scale_mass_policy", "native")
                    != "native"
                ):
                    raise ValueError(
                        "E8 derived mass source has conflicting scale/policy."
                    )
                self._sources[mass_copy.manifest] = lineage_digest
                self._sources[mass_copy.source] = mass_copy.source_sha256
                self._mass_source_sidecar = mass_copy.source.with_suffix(".twist.json")
                self._mass_source_sidecar_digest = mass_copy.source_sidecar_sha256
            elif self._mass_manifest_path.exists():
                raise ValueError("E8 mass lineage was not bound before assembly.")
            self._sources[self._robot_path] = _digest(self._robot_path)
            document = ET.parse(self._robot_path).getroot()
            self._names = tuple(
                link.get("name")
                for link in document.findall("link")
                if link.get("name", "").startswith(route.arm + "_")
                and link.findall("collision")
            )
            for link in document.findall("link"):
                if link.get("name") in self._names:
                    for collision in link.findall("collision"):
                        mesh = collision.find("geometry/mesh")
                        if mesh is not None:
                            path = (
                                self._robot_path.parent / mesh.get("filename")
                            ).resolve()
                            self._sources[path] = _digest(path)
            self._meshes = load_robot_link_meshes(
                self._robot_path, self._names, self._robot_scale
            )
            if self.table is not None:
                source = getattr(self.table.cfg.shape, "fpath", None)
                if source:
                    self._table_path = Path(source).resolve()
                    self._sources[self._table_path] = _digest(self._table_path)
        except (ValueError, OSError, AttributeError, TypeError, ET.ParseError) as error:
            self._setup_error = str(error)
        try:
            self._shell = _cylinder_shell(self.geometry, self.binding)
            self._shell_reason = None
        except (ValueError, AttributeError, IndexError, TypeError) as error:
            self._shell, self._shell_reason = None, str(error)

    def _fresh(self, request: Any, context: Any) -> None:
        b = self.binding
        if self._setup_error is not None:
            raise ValueError(self._setup_error)
        if (
            context.batch_size != 1
            or self.robot.num_instances != 1
            or not self.simulation.is_default_backend
        ):
            raise ValueError(
                "E8 geometry qualification requires one Default environment."
            )
        if (
            self.simulation.get_articulation(b.object_id) is not self.art
            or self.sensor.art is not self.art
        ):
            raise ValueError("E8 articulation runtime instance changed.")
        if (
            Path(self.art.cfg.fpath).resolve() != self._asset_path
            or Path(self.robot.cfg.fpath).resolve() != self._robot_path
            or tuple(self.robot.cfg.body_scale) != self._robot_scale
        ):
            raise ValueError("E8 source path/robot scale changed.")
        if not np.allclose(self.art.cfg.body_scale, b.scale, rtol=1e-7, atol=0):
            raise ValueError("E8 articulation geometry scale changed.")
        for path, expected in self._sources.items():
            if _digest(path) != expected:
                raise ValueError("E8 geometry source bytes changed.")
        sidecar = getattr(self, "_semantics_sidecar", None)
        if sidecar is not None:
            actual = _digest(sidecar) if sidecar.is_file() else None
            if actual != self._semantics_sidecar_digest:
                raise ValueError("E8 geometry semantics changed after assembly.")
        manifest = getattr(self, "_mass_manifest_path", None)
        if manifest is not None:
            expected = getattr(b, "mass_lineage_sha256", "") or None
            actual = _digest(manifest) if manifest.is_file() else None
            if actual != expected:
                raise ValueError("E8 mass lineage changed after assembly.")
            if (
                expected
                and getattr(self.art.cfg, "body_scale_mass_policy", "native")
                != "native"
            ):
                raise ValueError(
                    "E8 derived mass source cannot also use Lab conversion."
                )
        source_sidecar = getattr(self, "_mass_source_sidecar", None)
        if source_sidecar is not None:
            actual = _digest(source_sidecar) if source_sidecar.is_file() else None
            if actual != self._mass_source_sidecar_digest:
                raise ValueError(
                    "E8 original mass-source semantics changed after assembly."
                )
        values = request.goal.semantics.geometry
        if (
            request.goal.semantics.entity_id != self.route.link_id
            or values.get("gen_sim_twist_articulation_id") != b.object_id
            or values.get("gen_sim_twist_joint") != b.joint
            or values.get("gen_sim_twist_arm") != self.route.arm
            or values.get("gen_sim_twist_target_qpos") != b.target_qpos
            or values.get("gen_sim_twist_axis_sign") != b.axis_sign
        ):
            raise ValueError(
                "Resolved E8 request belongs to a different geometry route."
            )

    def _clouds(
        self,
        plan: Any,
        names: tuple[str, ...],
        frames: list[int],
        inverse: np.ndarray | None = None,
    ) -> Iterator[tuple[int, str, LinkMesh, torch.Tensor]]:
        positions = plan.joint_trajectory.positions
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[2] != len(self.robot.joint_names)
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("E8 requires a finite full-robot trajectory.")
        root = self.robot.get_local_pose(to_matrix=True)[0]
        _matrix(root)
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
                raise ValueError("Full-FK result differs from E8 collider scope.")
            poses = root.to(fk)[None, None] @ fk
            if inverse is not None:
                poses = (
                    torch.as_tensor(inverse, dtype=fk.dtype, device=fk.device)[
                        None, None
                    ]
                    @ poses
                )
            for column, name in enumerate(names):
                for mesh in self._meshes[name]:
                    vertices = positions.new_tensor(mesh.vertices.copy())
                    points = (
                        vertices[None] @ poses[:, column, :3, :3].transpose(-1, -2)
                        + poses[:, column, None, :3, 3]
                    )
                    for index, value in zip(indices.tolist(), points, strict=True):
                        yield index, name, mesh, value

    def _checked_clouds(
        self,
        plan: Any,
        names: tuple[str, ...],
        frames: list[int],
        inverse: np.ndarray | None = None,
    ) -> Iterator[tuple[int, str, LinkMesh, torch.Tensor]]:
        expected = {
            (index, name, mesh.path)
            for index in frames
            for name in names
            for mesh in self._meshes[name]
        }
        count = len(frames) * sum(len(self._meshes[name]) for name in names)
        if not expected or len(expected) != count:
            raise ValueError("E8 collision scope is empty or duplicated.")
        seen = set()
        for index, name, mesh, points in self._clouds(plan, names, frames, inverse):
            key = (index, name, mesh.path)
            if key not in expected or key in seen:
                raise ValueError(
                    "E8 collision frame coverage is foreign or duplicated."
                )
            if (
                points.ndim != 2
                or points.shape[1:] != (3,)
                or not len(points)
                or not torch.isfinite(points).all()
            ):
                raise ValueError("E8 transformed collision vertices are invalid.")
            seen.add(key)
            yield index, name, mesh, points
        if seen != expected:
            raise ValueError("E8 collision frame coverage is incomplete.")

    def _offsets(self, names: tuple[str, ...]) -> dict[str, float]:
        attrs = self.robot.get_link_physical_attr(names, env_ids=[0])
        if len(attrs) != len(names):
            raise ValueError("Native offset rows differ from E8 collider scope.")
        values = {}
        for name, attr in zip(names, attrs, strict=True):
            contact, rest = float(attr.contact_offset), float(attr.rest_offset)
            if not np.isfinite([contact, rest]).all() or contact < max(0.0, rest):
                raise ValueError("Native robot collision offsets are invalid.")
            values[name] = contact
        return values

    def filter_candidate(
        self, action: Any, plan: Any, request: Any, context: Any
    ) -> Any:
        """Reject unsafe free-motion candidates before their existing ranking."""
        return self._filter_table(action, plan, request, context, _FREE_PHASES)

    def filter_continuation(
        self, action: Any, plan: Any, request: Any, context: Any
    ) -> Any:
        """Screen every command in an owned suffix, including closing/turning."""
        phases = tuple(segment.name for segment in plan.segments)
        expected = {
            "reach": ("approach", "reach", "close", "twist", "open", "retract"),
            "close": ("reach", "close", "twist", "open", "retract"),
            "grip": ("close", "twist", "open", "retract"),
        }.get(request.goal.semantics.geometry.get("gen_sim_twist_resume_phase"))
        if expected is None or phases != expected:
            return action.failed_plan(
                request,
                context,
                message="E8 continuation suffix is incomplete.",
                failure_code="e8_continuation_scope_rejected",
                retryable=False,
            )
        return self._filter_table(action, plan, request, context, phases)

    def _filter_table(
        self,
        action: Any,
        plan: Any,
        request: Any,
        context: Any,
        free_phases: tuple[str, ...],
    ) -> Any:
        if not plan.plan_success.any():
            return plan
        try:
            self._fresh(request, context)
            if (
                self.table is None
                or self.simulation.get_rigid_object("table") is not self.table
            ):
                raise ValueError("The actual bound table is unavailable.")
            source = getattr(self.table.cfg.shape, "fpath", None)
            if (Path(source).resolve() if source else None) != self._table_path:
                raise ValueError("The bound table source changed.")
            table, table_contact = _native_table(self.table)
            length = plan.joint_trajectory.positions.shape[1]
            frames = []
            for phase in free_phases:
                matches = [
                    segment for segment in plan.segments if segment.name == phase
                ]
                if (
                    len(matches) != 1
                    or not 0 <= matches[0].start < matches[0].stop <= length
                ):
                    raise ValueError(
                        "E8 table screen requires complete free-motion phases."
                    )
                frames.extend(range(matches[0].start, matches[0].stop))
            if len(set(frames)) != len(frames):
                raise ValueError("E8 free-motion phases overlap.")
            from embodichain.lab.sim.atomic_actions import JointPositionTarget

            target = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            active = {self.robot.joint_names[index] for index in target.joint_ids}
            native = self.robot.body_data.entities[0].render_articulation
            names = []
            for name in self._names:
                if active.intersection(
                    joint.name for joint in self.robot.get_parent_joint_chain(name)
                ):
                    body = native.get_physical_body(name)
                    if body is None:
                        raise ValueError(
                            "Moving source link has no native physical body."
                        )
                    if int(body.get_shape_count()) > 0:
                        names.append(name)
            names = tuple(names)
            if not names or not set(self.sensor.finger_names) <= set(names):
                raise ValueError("Moving scope omits active finger colliders.")
            offsets = self._offsets(names)
            minimum = float("inf")
            for _, name, _, points in self._checked_clouds(plan, names, frames):
                low, high = points.amin(0), points.amax(0)
                bounds = points.new_tensor(table)
                gap = torch.maximum(bounds[0] - high, low - bounds[1]).clamp_min(0)
                clearance = (
                    float(torch.linalg.vector_norm(gap)) - offsets[name] - table_contact
                )
                minimum = min(minimum, clearance)
            if not math.isfinite(minimum):
                raise ValueError("E8 table clearance lacks finite complete evidence.")
            evidence = {
                "supported": True,
                "accepted": minimum > 0,
                "minimum_clearance_after_skin_m": minimum,
                "checked_frames": len(frames),
                "moving_links": list(names),
            }
        except (
            ValueError,
            RuntimeError,
            KeyError,
            AttributeError,
            OSError,
            IndexError,
        ) as error:
            evidence = {"supported": False, "accepted": False, "reason": str(error)}
        if not evidence["accepted"]:
            plan = action.failed_plan(
                request,
                context,
                message="E8 candidate failed table free-motion clearance.",
                failure_code="e8_table_clearance_rejected",
                retryable=False,
            )
        if plan.diagnostics is not None:
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_twist_table_clearance": evidence,
                    },
                ),
            )
        return plan

    def root_dependency(
        self, plan: Any, request: Any, context: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Alias only a qualified invariant grasp; unsupported geometry retains child."""
        try:
            self._fresh(request, context)
            if self._shell is None:
                raise ValueError(self._shell_reason)
            b = self.binding
            entity = self.art.body_data.entities[0]
            if (
                entity.get_root_link_name() != b.parent
                or entity.render_articulation is None
            ):
                raise ValueError("E8 grasp parent is not the actual Default root.")
            joint = entity.render_articulation.get_joint_info(b.joint)
            if joint is None:
                raise ValueError("Native E8 hinge information is unavailable.")
            first, second = _matrix(joint.origin_pose), _matrix(joint.target_pose)
            joint_axis = np.asarray(joint.axis, dtype=float).copy()
            if (
                joint_axis.shape != (3,)
                or not np.isfinite(joint_axis).all()
                or np.linalg.norm(joint_axis) <= 0
            ):
                raise ValueError("Native E8 hinge axis is invalid.")
            joint_axis /= np.linalg.norm(joint_axis)
            parent = _matrix(self.art.get_link_pose(b.parent, to_matrix=True)[0])
            child = _matrix(self.art.get_link_pose(b.link, to_matrix=True)[0])
            if not np.allclose(
                _matrix(context.scene.entities[b.object_id].pose[0]),
                parent,
                rtol=0,
                atol=1e-5,
            ):
                raise ValueError("Registered scene root differs from native E8 parent.")
            measured = request.goal.semantics.geometry["gen_sim_twist_measured_qpos"]
            live = self.art.get_qpos()
            if (
                live.ndim != 2
                or live.shape[0] != 1
                or not torch.isfinite(live).all()
                or not math.isfinite(measured)
                or abs(float(live[0, self.art.joint_names.index(b.joint)]) - measured)
                > 1e-7
            ):
                raise ValueError("Resolved E8 joint observation is stale or invalid.")
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(joint_axis * measured).as_matrix()
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
            axis, origin = second[:3, :3] @ joint_axis, second[:3, 3]
            axis /= np.linalg.norm(axis)
            affordance = request.goal.semantics.affordance
            if (
                np.linalg.norm(
                    np.cross(np.asarray(affordance.grasp_position) - origin, axis)
                )
                > 2e-6
            ):
                raise ValueError("Selected E8 grasp is not on the native hinge axis.")
            grasp = affordance.get_grasp_pose(
                torch.as_tensor(child, dtype=torch.float32)[None]
            )
            for angle in np.linspace(*b.limits, 9):
                turn[:3, :3] = Rotation.from_rotvec(joint_axis * angle).as_matrix()
                pose = parent @ first @ turn @ np.linalg.inv(second)
                if not torch.allclose(
                    grasp,
                    affordance.get_grasp_pose(
                        torch.as_tensor(pose, dtype=torch.float32)[None]
                    ),
                    atol=2e-5,
                    rtol=0,
                ):
                    raise ValueError(
                        "E8 grasp point/basis is not rotationally invariant."
                    )
            reference = np.eye(3)[int(np.argmin(abs(axis)))]
            x = np.cross(axis, reference)
            x /= np.linalg.norm(x)
            basis = np.column_stack((x, np.cross(axis, x), axis))
            rotor = self.art.get_link_physical_attr(b.link, env_ids=[0])[0]
            rotor_contact, rotor_rest = float(rotor.contact_offset), float(
                rotor.rest_offset
            )
            if not np.isfinite(
                [rotor_contact, rotor_rest]
            ).all() or rotor_contact < max(0.0, rotor_rest):
                raise ValueError("Native rotor collision offsets are invalid.")
            padding = max(self._offsets(self._names).values()) + rotor_contact
            sweeps = []
            for mesh in self.geometry.target_collisions:
                if mesh.path == self.geometry.grip.path:
                    continue
                extra = (mesh.vertices - origin) @ basis
                radius = float(np.linalg.norm(extra[:, :2], axis=1).max()) + padding
                low, high = (
                    float(extra[:, 2].min()) - padding,
                    float(extra[:, 2].max()) + padding,
                )
                sweeps.append(
                    {
                        "path": mesh.path,
                        "radius": radius,
                        "low": low,
                        "high": high,
                        "projection_refinements": 0,
                    }
                )
            clouds = (
                self._checked_clouds(
                    plan,
                    self._names,
                    list(range(plan.joint_trajectory.positions.shape[1])),
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
                        or _radial_distance(values[:, :2]) > radius
                    ):
                        raise ValueError(
                            "Extra rotor sweep remains unproven against a planned collider."
                        )
            checks = [
                {
                    "path": sweep["path"],
                    "projection_refinements": sweep["projection_refinements"],
                    "separated": True,
                }
                for sweep in sweeps
            ]
            child_id, parent_id = request.goal.semantics.entity_id, b.object_id
            if child_id not in plan.scene_dependencies:
                return plan, {"qualified": True, "changed": False}
            updated = _remap_dependency(plan, child_id, parent_id)
            return updated, {
                "qualified": True,
                "changed": True,
                "source_sha256": b.source_sha256,
                "padding_m": padding,
                "extra_sweeps": checks,
                "scope": "source-input command-frame proxy; not native cooking or continuous collision certification",
            }
        except (
            ValueError,
            RuntimeError,
            KeyError,
            AttributeError,
            OSError,
            IndexError,
            QhullError,
            np.linalg.LinAlgError,
        ) as error:
            return plan, {"qualified": False, "changed": False, "reason": str(error)}
