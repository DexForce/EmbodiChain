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

"""Scoped revolute qualifications composed from shared source/path services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import QhullError
from scipy.spatial.transform import Rotation
import torch

from embodichain.compute.geometry import (
    axis_contact_pose,
    convex_projection_distance_to_origin,
)
from embodichain.lab.sim.cfg.source import _asset_file_path
from embodichain.lab.sim.motion.collision_query import (
    RobotPathQuery,
    SupportPathReport,
    _QUERY_ERRORS,
)
from embodichain.lab.sim.scene.control_source import (
    RevoluteSourceSeal,
    qualify_revolute_grip_shell,
)
from embodichain.lab.sim.scene.robot_collision import (
    RobotCollisionInputs,
    SupportCollisionInputs,
    _matrix,
)
from embodichain.lab.sim.scene.revolute_geometry import TwistGeometry
from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding
from embodichain.lab.sim.utility.grasp_geometry import (
    RobotHandGeometryCfg,
    _owned_cfg,
    _resolve_hand_frame,
)
from embodichain.toolkits.graspkit.pg_grasp.contact_metrics import (
    sample_contact_vertices,
    pad_proximity_metrics,
)
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "RevolutePathQueryFactory",
    "RevolutePathQuery",
    "RevolutePathReport",
    "RevoluteDependencyReport",
]

_CONTACT_SAMPLE_COUNT = 128
_PARENT_NEAR_DISTANCE_M = 0.003
_HINGE_FRAME_SAMPLES = 9


@dataclass(frozen=True, slots=True)
class RevolutePathReport(SupportPathReport):
    """Source-compatible revolute free-phase support-envelope report."""


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
    """Compose shared geometry captures with one calibrated revolute source.

    Robot/support identity and geometry belong to Scene captures. Motion owns
    generic path checks; this factory adds hand and revolute-source qualification.
    Source changes reject instead of being recaptured as a new baseline.
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
        self._simulation = simulation
        self._binding, self._geometry, self._seal = binding, geometry, source_seal
        self._cfg = _owned_cfg(hand_geometry_cfg)
        _, _, hand_joints, _ = _resolve_hand_frame(robot, self._cfg)
        self._hand_joints = tuple(hand_joints)
        self._robot_geometry = RobotCollisionInputs(
            simulation,
            robot,
            motion_part=self._cfg.motion_part,
            collision_link_names=collision_link_names,
        )
        if not set(self._cfg.pad_link_names) <= set(self._robot_geometry.link_names):
            raise ValueError(
                "The declared robot collision scope must include both pads."
            )
        self._support = SupportCollisionInputs(simulation, table_uid)
        self._pad_metric_meshes = None
        self._art = simulation.get_articulation(binding.object_id)
        if self._art is None:
            raise ValueError("The exact native articulation must be available.")
        self._native_art = self._art.body_data.entities[0]
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
        return self._robot_geometry.robot

    def _assert_current(self) -> None:
        if self._closed:
            raise RuntimeError("Revolute query factory is closed.")
        self._robot_geometry.assert_current()
        self._support.assert_current()
        self._assert_revolute_current()

    def _assert_revolute_current(self) -> None:
        if self._closed:
            raise RuntimeError("Revolute query factory is closed.")
        if (
            self._simulation.get_articulation(self._binding.object_id) is not self._art
            or self._art.body_data.entities[0] is not self._native_art
        ):
            raise ValueError("A revolute query native owner was replaced.")
        if (
            tuple(self.robot.cfg.control_parts[self._cfg.hand_part])
            != self._hand_joints
        ):
            raise ValueError("Robot hand control layout changed.")
        self._seal.assert_current(
            source_path=_asset_file_path(self._art.cfg),
            body_scale=self._art.cfg.body_scale,
            body_scale_mass_policy=getattr(
                self._art.cfg, "body_scale_mass_policy", "native"
            ),
        )

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
        """Close owned captures without destroying borrowed native entities."""
        self._closed = True
        self._robot_geometry.close()
        self._support.close()
        self._pad_metric_meshes = None
        self._geometry = None
        self._simulation = self._art = self._native_art = None


class RevolutePathQuery:
    """One B1 observation's contact metrics and specialized revolute qualifications."""

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
        self._paths = RobotPathQuery(
            factory._robot_geometry, scope=scope, robot_root_pose=robot_root_pose
        )
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
        self._factory._assert_revolute_current()
        self._factory._support.assert_current()
        self._paths.assert_current()

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
            factory._pad_metric_meshes = factory._robot_geometry.load_meshes(
                factory._cfg.pad_link_names, vertices_only=False
            )

        def sample(points: object) -> torch.Tensor:
            return sample_contact_vertices(
                full_qpos.new_tensor(points), maximum_count=_CONTACT_SAMPLE_COUNT
            )

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
        pad_vertices = []
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
            pad_vertices.append(world)
        near, distance = pad_proximity_metrics(
            pad_vertices,
            target_world,
            parent_world,
            parent_distance_m=_PARENT_NEAR_DISTANCE_M,
        )
        return (
            near,
            distance,
            float(torch.linalg.vector_norm(full_qpos - start_qpos).item()),
        )

    def inspect_free_path(
        self, positions: torch.Tensor, frame_indices: Sequence[int]
    ) -> RevolutePathReport:
        """Inspect selected free-phase frames through the shared support screen.

        Args:
            positions: Finite full-robot samples, shape ``(1, T, J)``.
            frame_indices: Unique explicit rows selected by the caller.

        Returns:
            The original numeric report. The generic source/native AABB screen
            additionally requires both declared pad colliders for this operation.
        """
        if self._closed:
            raise RuntimeError("Revolute query is closed.")
        try:
            self._factory._assert_revolute_current()
            report = self._paths.inspect_support_path(
                self._factory._support,
                positions,
                frame_indices,
                required_link_names=self._factory._cfg.pad_link_names,
            )
            return RevolutePathReport(
                report.supported,
                report.accepted,
                report.minimum_clearance_after_skin_m,
                report.checked_frames,
                report.moving_links,
                report.reason,
            )
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
            for angle in np.linspace(*binding.limits, _HINGE_FRAME_SAMPLES):
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
            padding = (
                max(
                    factory._robot_geometry.contact_offsets(
                        factory._robot_geometry.link_names
                    ).values()
                )
                + rotor_contact
            )
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
                self._paths.iter_link_vertices(
                    positions,
                    factory._robot_geometry.link_names,
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
        self._paths.close()
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
