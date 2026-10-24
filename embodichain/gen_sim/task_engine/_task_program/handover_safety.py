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

"""Discrete receiver-hand clearance against native non-target scene geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.runtime_commands import JointPositionPayload
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    translate_pose_world,
)
from .articulation_collision import _descendants
from .motion import _cartesian_samples
from .pick_contact import _native_meshes
from .staging_recovery import _pose

__all__: list[str] = []

HANDOVER_SCENE_CLEARANCE_REVISION = 4
_SCOPE = "receiver_hand_non_target_scene_discrete_native_shapes"


@dataclass(frozen=True, slots=True)
class _Plane:
    normal: np.ndarray
    point: np.ndarray


@dataclass(frozen=True, slots=True)
class _Body:
    name: str
    owner: str
    link: str | None
    meshes: tuple[Any, ...]
    pose: np.ndarray
    contact_offset: float
    rest_offset: float
    planes: tuple[_Plane, ...] = ()


class _MeshShapeView:
    """Read only the finite native shapes through the existing mesh adapter."""

    def __init__(self, source: Any, shapes: tuple[Any, ...]) -> None:
        self.source, self.shapes = source, shapes

    def is_collision_enable(self) -> bool:
        return self.source.is_collision_enable()

    def get_shape_count(self) -> int:
        return len(self.shapes)

    def get_shape_geometry(self, index: int) -> Any:
        return self.shapes[index]


def _shapes(body: Any, name: str) -> tuple[tuple[Any, ...], tuple[_Plane, ...]]:
    from dexsim.engine import BoxGeometry, ConvexMeshGeometry, PlaneGeometry

    finite, planes = [], []
    for index in range(int(body.get_shape_count())):
        try:
            shape = body.get_shape_geometry(index)
        except Exception as error:
            raise ValueError(
                f"handover_native_shape_readback_unavailable:{name}:shape={index}:{error}"
            ) from error
        if isinstance(shape, PlaneGeometry):
            normal = np.asarray(shape.normal, dtype=np.float64)
            distance = float(shape.distance)
            if (
                normal.shape != (3,)
                or not np.isfinite(normal).all()
                or not math.isfinite(distance)
                or np.linalg.norm(normal) == 0
            ):
                raise ValueError(
                    f"handover_native_plane_unavailable:{name}:shape={index}"
                )
            normal = normal / np.linalg.norm(normal)
            local = _pose(shape.local_pose)
            # SDK distance is measured from the origin along the plane normal.
            planes.append(
                _Plane(
                    local[:3, :3] @ normal,
                    local[:3, :3] @ (normal * distance) + local[:3, 3],
                )
            )
        elif isinstance(shape, (ConvexMeshGeometry, BoxGeometry)):
            finite.append(shape)
        else:
            kind = f"{type(shape).__module__}.{type(shape).__name__}"
            raise ValueError(
                f"handover_native_shape_unsupported:{name}:shape={index}:type={kind}"
            )
    try:
        meshes = _native_meshes(_MeshShapeView(body, tuple(finite))) if finite else ()
    except (ValueError, RuntimeError, TypeError, AttributeError) as error:
        raise ValueError(f"handover_native_mesh_unavailable:{name}:{error}") from error
    return meshes, tuple(planes)


def _body(
    name: str, owner: str, link: str | None, body: Any, pose: Any
) -> _Body | None:
    if body is None:
        raise ValueError(f"handover_native_body_unavailable:{name}")
    if not body.is_collision_enable():
        return None
    meshes, planes = _shapes(body, name)
    if not meshes and not planes:
        raise ValueError(f"handover_enabled_native_shapes_unavailable:{name}")
    contact, rest = float(body.get_contact_offset()), float(body.get_rest_offset())
    if (
        not math.isfinite(contact)
        or not math.isfinite(rest)
        or not 0 <= contact
        or rest > contact
    ):
        raise ValueError(f"handover_native_offsets_unavailable:{name}")
    return _Body(name, owner, link, meshes, _pose(pose).copy(), contact, rest, planes)


def _manager(body: _Body) -> trimesh.collision.CollisionManager:
    manager = trimesh.collision.CollisionManager()
    for index, mesh in enumerate(body.meshes):
        manager.add_object(str(index), mesh, transform=body.pose)
    return manager


def _bounds(body: _Body, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.vstack([mesh.vertices for mesh in body.meshes])
    # Transform the local AABB corners; this remains conservative under rotation.
    corners = trimesh.bounds.corners(np.array([vertices.min(0), vertices.max(0)]))
    world = trimesh.transform_points(corners, pose)
    return world.min(0), world.max(0)


class _EnvironmentScreen:
    def __init__(self, robot, request, context, root, parts, obstacles, inventory):
        self.robot = robot
        self.request = request
        self.context = context
        self.root = root.copy()
        self.parts = parts
        self.obstacles = obstacles
        self.inventory = inventory
        self.joint_names = tuple(robot.joint_names)
        self.hand_managers = tuple(_manager(part) for part in parts)
        self.obstacle_managers = tuple(
            _manager(part) if part.meshes else None for part in obstacles
        )
        self.obstacle_bounds = tuple(
            _bounds(part, part.pose) if part.meshes else None for part in obstacles
        )
        self.receiver_targets = tuple(
            request.binding.endpoint("destination", kind).require_target(
                JointPositionTarget
            )
            for kind in ("motion", "grasp")
        )

    def inspect(self, plan: Any) -> dict[str, Any]:
        """Inspect the supplied command grid without modifying any plan or state."""
        trajectory = plan.joint_trajectory
        if (
            plan.invocation_id != self.request.invocation_id
            or plan.invocation_revision != self.request.revision
            or plan.planned_scene_version != self.context.scene.version
            or trajectory is None
            or trajectory.positions.ndim != 3
            or trajectory.positions.shape[0] != 1
            or trajectory.positions.shape[2] != len(self.joint_names)
            or not trajectory.positions.shape[1]
            or not torch.isfinite(trajectory.positions).all()
            or plan.commands.frame_count != trajectory.positions.shape[1]
            or not torch.equal(plan.commands.env_ids, self.context.env_ids)
        ):
            raise ValueError("handover_scene_command_identity_unavailable")
        for index, frame in enumerate(plan.commands.frames):
            for target in self.receiver_targets:
                commands = [
                    command
                    for command in frame.commands
                    if command.target.address_fingerprint == target.address_fingerprint
                ]
                if (
                    len(commands) != 1
                    or not isinstance(commands[0].payload, JointPositionPayload)
                    or not torch.equal(
                        commands[0].payload.positions,
                        trajectory.positions[:, index, list(target.joint_ids)],
                    )
                ):
                    raise ValueError("handover_scene_command_payload_mismatch")
        qpos = trajectory.positions[0].detach().clone()
        fk = self.robot.compute_fk(
            qpos=qpos,
            link_names=[part.link for part in self.parts],
            qpos_joint_names=self.joint_names,
        )
        if (
            fk.shape != (len(qpos), len(self.parts), 4, 4)
            or not torch.isfinite(fk).all()
        ):
            raise ValueError("handover_scene_named_fk_unavailable")
        return self._inspect_hand_poses(
            self.root @ fk.detach().cpu().double().numpy(), self._report(len(qpos))
        )

    def candidate_mask(
        self,
        poses: torch.Tensor,
        eligible: torch.Tensor,
        *,
        resources: Any,
        approach_samples: int,
        pre_grasp_distance: float,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Screen the original receiving pool without planning or resampling.

        Args:
            poses: Unchanged original pool poses, shaped ``(1, N, 4, 4)``.
            eligible: The same pool's original boolean validity mask.
            resources: Invocation-bound shared HandOver participants.
            approach_samples: Shared HandOver's original approach allocation.
            pre_grasp_distance: The original invocation's approach distance.

        Returns:
            A legality mask and per-pool-index native-geometry diagnostics. This
            nominal open-hand TCP path is only a prefilter: the emitted complete
            command grid must still pass :meth:`inspect`.
        """
        indices = self._candidate_indices(
            poses, eligible, resources, approach_samples, pre_grasp_distance
        )
        mask = torch.zeros_like(eligible)
        report = {
            "revision": HANDOVER_SCENE_CLEARANCE_REVISION,
            "scope": "receiver_open_hand_nominal_approach_non_target_scene",
            "approach_samples": approach_samples,
            "pre_grasp_distance_m": pre_grasp_distance,
            "eligible_count": int(eligible.sum().item()),
            "requires_complete_command_guard": True,
            "candidates": [],
        }
        if not indices:
            return mask, report
        start, relative, opened = self._open_hand(resources)
        report["open_hand_joint_positions"] = opened.detach().cpu().tolist()
        for index in indices:
            tcp = self._candidate_path(
                start, poses[:, index], approach_samples, pre_grasp_distance
            )
            arena_poses = tcp[:, None] @ relative
            evidence = self._report(approach_samples)
            evidence["scope"] = report["scope"]
            evidence["not_evaluated"] += ["ik_reachability", "emitted_command_fk"]
            evidence = self._inspect_hand_poses(arena_poses, evidence)
            mask[0, index] = evidence["valid"]
            report["candidates"].append({"pool_index": index, **evidence})
        return mask, report

    def _candidate_indices(
        self, poses, eligible, resources, approach_samples, pre_grasp_distance
    ):
        if (
            poses.ndim != 4
            or poses.shape[0] != 1
            or poses.shape[2:] != (4, 4)
            or not poses.is_floating_point()
            or eligible.dtype is not torch.bool
            or eligible.shape != poses.shape[:2]
            or eligible.device != poses.device
            or type(approach_samples) is not int
            or approach_samples < 3
            or not math.isfinite(pre_grasp_distance)
            or pre_grasp_distance < 0
        ):
            raise ValueError("handover_candidate_pool_or_path_policy_unavailable")
        if (
            self.context.scene.timestamp != self.context.robot.timestamp
            or not torch.equal(self.robot.get_qpos(), self.context.robot.qpos)
        ):
            raise ValueError("handover_candidate_same_tick_snapshot_changed")
        arm, hand = resources.second.arm, resources.second.hand
        if any(
            actual.address_fingerprint != expected.address_fingerprint
            for actual, expected in zip((arm, hand), self.receiver_targets, strict=True)
        ):
            raise ValueError("handover_candidate_receiver_binding_mismatch")
        return torch.nonzero(eligible[0], as_tuple=False).flatten().tolist()

    def _open_hand(self, resources):
        arm, hand = resources.second.arm, resources.second.hand
        qpos = self.context.robot.qpos.detach().clone()
        opened = resources.second.hand_open_qpos.to(qpos)
        if opened.shape != (1, len(hand.joint_ids)) or not torch.isfinite(opened).all():
            raise ValueError("handover_candidate_bound_open_command_unavailable")
        qpos[:, list(hand.joint_ids)] = opened
        # Use the same named-arm start as the shared planner, including its TCP.
        start = self.robot.compute_fk(
            qpos=qpos[:, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        links = self.robot.compute_fk(
            qpos=qpos,
            link_names=[part.link for part in self.parts],
            qpos_joint_names=self.joint_names,
        )
        if (
            start.shape != (1, 4, 4)
            or links.shape != (1, len(self.parts), 4, 4)
            or not torch.isfinite(start).all()
            or not torch.isfinite(links).all()
        ):
            raise ValueError("handover_candidate_open_hand_named_fk_unavailable")
        start_array = _pose(start[0])
        relative = np.linalg.inv(start_array) @ (
            self.root @ links[0].detach().cpu().double().numpy()
        )
        return start, relative, opened

    @staticmethod
    def _candidate_path(start, grasp, approach_samples, pre_grasp_distance):
        _pose(grasp[0])
        pre_grasp = translate_pose_world(grasp, -grasp[:, :3, 2] * pre_grasp_distance)
        targets = build_pose_plan_states(torch.stack((pre_grasp, grasp), dim=1))
        sampled = _cartesian_samples(start, targets, approach_samples)
        tcp = torch.stack([start, *(state.xpos for state in sampled)], dim=1)
        return tcp[0].detach().cpu().double().numpy()

    def exchange_candidates(
        self,
        exchange_pose: torch.Tensor,
        poses: torch.Tensor,
        eligible: torch.Tensor,
        *,
        resources: Any,
        approach_samples: int,
        pre_grasp_distance: float,
        source_root: torch.Tensor,
        destination_root: torch.Tensor,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Derive at most three native-geometry exchange translations.

        The caller invokes this only after every original receiving candidate
        failed. Original relative grasps and the roots' dominant horizontal
        midpoint are preserved. Every proposal is rechecked from the unchanged
        receiver start against the complete captured scene, without IK.
        """
        indices = self._candidate_indices(
            poses, eligible, resources, approach_samples, pre_grasp_distance
        )
        exchange = _pose(exchange_pose)
        source, destination = _pose(source_root), _pose(destination_root)
        separation = np.abs(source[:3, 3] - destination[:3, 3])
        dominant = int(separation.argmax())
        midpoint = 0.5 * (source[dominant, 3] + destination[dominant, 3])
        if (
            dominant not in (0, 1)
            or separation[dominant] <= 0
            or not np.isclose(exchange[dominant, 3], midpoint)
        ):
            raise ValueError("handover_exchange_horizontal_midplane_unavailable")
        report = {
            "revision": HANDOVER_SCENE_CLEARANCE_REVISION,
            "scope": "native_receiver_suffix_aabb_exchange_proposals",
            "coordinate_frame": "planning_robot_arena",
            "dominant_root_axis": dominant,
            "preserved_midpoint_coordinate": float(midpoint),
            "eligible_count": len(indices),
            "requires_complete_command_guard": True,
            "not_evaluated": ["source_hand", "held_payload", "ik_reachability"],
        }
        if not indices:
            return [], report
        start, relative, _ = self._open_hand(resources)
        try:
            solver = self.robot.cfg.solver_cfg[resources.second.arm.control_part]
            if isinstance(solver.pos_eps, bool) or isinstance(solver.rot_eps, bool):
                raise ValueError("Boolean IK tolerances are unavailable.")
            position = float(solver.pos_eps)
            rotation = float(solver.rot_eps)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("handover_exchange_ik_uncertainty_unavailable") from error
        if (
            getattr(solver, "class_type", None) != "PytorchSolver"
            or getattr(solver, "is_only_position_constraint", False)
            or not math.isfinite(position)
            or not math.isfinite(rotation)
            or position <= 0
            or rotation <= 0
        ):
            raise ValueError("handover_exchange_ik_uncertainty_unavailable")
        radius = max(
            float(
                np.linalg.norm(
                    trimesh.transform_points(mesh.vertices, transform), axis=1
                ).max()
            )
            for part, transform in zip(self.parts, relative, strict=True)
            for mesh in part.meshes
        )
        uncertainty = position + 2 * radius * math.sin(min(rotation, math.pi) / 2)
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
        suffix_start = (approach_samples - 1) // 2
        for index in indices:
            tcp = self._candidate_path(
                start, poses[:, index], approach_samples, pre_grasp_distance
            )
            for frame in tcp[suffix_start:]:
                for part, transform in zip(self.parts, relative, strict=True):
                    a, b = _bounds(part, frame @ transform)
                    lower, upper = np.minimum(lower, a), np.maximum(upper, b)
        hand_offset = max(part.contact_offset for part in self.parts)
        near_meshes, near_planes = [], []
        for obstacle, bound in zip(self.obstacles, self.obstacle_bounds, strict=True):
            clearance = hand_offset + obstacle.contact_offset + uncertainty
            if bound is not None:
                gap = np.maximum(np.maximum(bound[0] - upper, lower - bound[1]), 0)
                if np.linalg.norm(gap) <= clearance:
                    near_meshes.append((obstacle, bound, clearance))
            for plane in obstacle.planes:
                normal = obstacle.pose[:3, :3] @ plane.normal
                point = obstacle.pose[:3, :3] @ plane.point + obstacle.pose[:3, 3]
                support = np.where(normal >= 0, lower, upper)
                signed = float((support - point) @ normal)
                if signed <= clearance:
                    near_planes.append((obstacle, normal, signed, clearance))
        report.update(
            suffix_indices_inclusive=[suffix_start, approach_samples - 1],
            native_swept_bounds_m=[lower.tolist(), upper.tolist()],
            ik_geometric_uncertainty={
                "position_tolerance_m": position,
                "rotation_tolerance_rad": rotation,
                "hand_radius_m": radius,
                "surface_uncertainty_m": uncertainty,
                "not_a_physical_acceptance_threshold": True,
            },
            native_hand_contact_offset_upper_m=hand_offset,
            near_native_bodies=[
                {
                    "name": body.name,
                    "contact_offset_m": body.contact_offset,
                    "arena_bounds_m": [value.tolist() for value in bound],
                    "contact_envelope_plus_ik_uncertainty_m": clearance,
                }
                for body, bound, clearance in near_meshes
            ],
            near_native_planes=[
                {
                    "name": body.name,
                    "normal_arena": normal.tolist(),
                    "minimum_signed_distance_m": signed,
                    "contact_envelope_plus_ik_uncertainty_m": clearance,
                }
                for body, normal, signed, clearance in near_planes
            ],
        )
        if not near_meshes and not near_planes:
            return [], report
        proposals = []
        horizontal = 1 - dominant
        for axis, sign in ((horizontal, 1), (horizontal, -1), (2, 1)):
            magnitude = max(
                [0.0]
                + [
                    (
                        float(bound[1][axis] + clearance - lower[axis])
                        if sign > 0
                        else float(upper[axis] - bound[0][axis] + clearance)
                    )
                    for _, bound, clearance in near_meshes
                ]
            )
            possible = True
            for _, normal, signed, clearance in near_planes:
                projection = sign * normal[axis]
                if projection <= 0:
                    possible = False
                    break
                magnitude = max(magnitude, (clearance - signed) / projection)
            if not possible or magnitude <= 0:
                continue
            value = poses.new_tensor(magnitude)
            value = torch.nextafter(value, value.new_tensor(math.inf))
            delta = torch.zeros((1, 3), dtype=poses.dtype, device=poses.device)
            delta[0, axis] = sign * value
            shifted = poses.clone()
            shifted[:, :, :3, 3] += delta[:, None]
            mask, evidence = self.candidate_mask(
                shifted,
                eligible,
                resources=resources,
                approach_samples=approach_samples,
                pre_grasp_distance=pre_grasp_distance,
            )
            proposals.append(
                {
                    "translation": delta,
                    "mask": mask & eligible,
                    "report": evidence,
                    "axis": ("+" if sign > 0 else "-") + "xyz"[axis],
                    "norm_m": float(value),
                }
            )
        proposals.sort(key=lambda proposal: proposal["norm_m"])
        report["proposal_count"] = len(proposals)
        return proposals, report

    def _report(self, frames: int) -> dict[str, Any]:
        return {
            "valid": True,
            "revision": HANDOVER_SCENE_CLEARANCE_REVISION,
            "scope": _SCOPE,
            "frames": frames,
            "inventory": self.inventory,
            "excluded_target_object": self.request.goal.semantics.entity_id,
            "not_evaluated": [
                "held_target_contact",
                "source_hand_and_robot_self",
                "payload_scene",
                "between_command_samples",
                "future_scene_motion",
            ],
            "geometry_source": "native_cooked_finite_shapes_and_analytic_planes",
            "clearance_policy": "sum_of_unchanged_native_contact_offsets",
            "planning_timestamp": self.context.robot.timestamp,
            "minimum_checked_distance_m": None,
            "fcl_queries": 0,
            "plane_queries": 0,
            "minimum_plane_signed_distance_m": None,
        }

    def _inspect_hand_poses(
        self, arena_poses: np.ndarray, report: dict[str, Any]
    ) -> dict[str, Any]:
        minimum = None
        for frame_index, row in enumerate(arena_poses):
            for part_index, (part, arena_pose) in enumerate(
                zip(self.parts, row, strict=True)
            ):
                pose = _pose(arena_pose)
                manager = self.hand_managers[part_index]
                for shape in range(len(part.meshes)):
                    manager.set_transform(str(shape), pose)
                lower, upper = _bounds(part, pose)
                for obstacle, other, bounds in zip(
                    self.obstacles,
                    self.obstacle_managers,
                    self.obstacle_bounds,
                    strict=True,
                ):
                    threshold = part.contact_offset + obstacle.contact_offset
                    for plane in obstacle.planes:
                        normal = obstacle.pose[:3, :3] @ plane.normal
                        point = (
                            obstacle.pose[:3, :3] @ plane.point + obstacle.pose[:3, 3]
                        )
                        vertices = trimesh.transform_points(
                            np.vstack([mesh.vertices for mesh in part.meshes]), pose
                        )
                        signed_distance = float(((vertices - point) @ normal).min())
                        report["plane_queries"] += 1
                        previous = report["minimum_plane_signed_distance_m"]
                        report["minimum_plane_signed_distance_m"] = (
                            signed_distance
                            if previous is None
                            else min(previous, signed_distance)
                        )
                        if signed_distance <= threshold:
                            return {
                                **report,
                                "valid": False,
                                "frame": frame_index,
                                "receiver_link": part.link,
                                "obstacle_owner": obstacle.owner,
                                "obstacle_link": obstacle.link,
                                "obstacle_name": obstacle.name,
                                "minimum_signed_distance_m": signed_distance,
                                "native_pair_contact_offset_m": threshold,
                                "plane_normal_arena": normal.tolist(),
                                "plane_point_arena": point.tolist(),
                                "reason": "receiver_hand_non_target_plane_contact_envelope",
                            }
                    if other is None:
                        continue
                    gap = np.maximum(
                        np.maximum(bounds[0] - upper, lower - bounds[1]), 0
                    )
                    if np.linalg.norm(gap) > threshold:
                        continue
                    intersecting = bool(manager.in_collision_other(other))
                    distance = (
                        0.0
                        if intersecting
                        else float(manager.min_distance_other(other))
                    )
                    if not math.isfinite(distance) or distance < 0:
                        raise ValueError("handover_native_mesh_distance_unavailable")
                    report["fcl_queries"] += 1
                    minimum = distance if minimum is None else min(minimum, distance)
                    report["minimum_checked_distance_m"] = minimum
                    if distance <= threshold:
                        return {
                            **report,
                            "valid": False,
                            "frame": frame_index,
                            "receiver_link": part.link,
                            "obstacle_owner": obstacle.owner,
                            "obstacle_link": obstacle.link,
                            "obstacle_name": obstacle.name,
                            "minimum_distance_m": distance,
                            "native_pair_contact_offset_m": threshold,
                            "reason": "receiver_hand_non_target_contact_envelope",
                        }
        return report


def _hand_links(robot, hand, arm):
    document = ET.parse(robot.cfg.fpath).getroot()
    joints = {joint.get("name"): joint for joint in document.findall("joint")}
    names = tuple(robot.joint_names)
    selected = {names[index] for index in hand.joint_ids}
    if not selected or not selected <= joints.keys():
        raise ValueError("handover_hand_joint_identity_unavailable")
    parents = {
        joint.find("child").get("link"): joint.find("parent").get("link")
        for joint in joints.values()
    }

    root = robot.cfg.solver_cfg[arm.control_part].end_link_name
    children = {}
    for child, parent in parents.items():
        children.setdefault(parent, set()).add(child)
    links = _descendants(children, root)
    moving = {
        name
        for name, joint in joints.items()
        if joint.get("type") != "fixed"
        and joint.find("child").get("link") in links
        and joint.find("parent").get("link") in links
    }
    if moving != selected:
        raise ValueError("handover_hand_requires_all_moving_joints_including_mimics")
    authored = {
        link.get("name")
        for link in document.findall("link")
        if link.findall("collision")
    }
    return tuple(sorted(links)), authored


def build_handover_environment_screen(
    robot: Any,
    simulation: Any,
    scene_binding: Any,
    request: Any,
    context: Any,
    resources: Any,
) -> Any:
    """Freeze Default B1 native scene geometry for one existing-hold handover.

    Native shapes already carry their effective scale. In particular, retained
    articulation materialization can differ from the original asset's YAML scale.

    Args:
        robot: Invocation-bound robot with complete named-joint FK.
        simulation: Prepared single-environment Default simulation, read only.
        scene_binding: Semantic-to-native physical identities.
        request: Resolved handover invocation without changed command policy.
        context: Same-tick robot and scene planning snapshot.
        resources: Shared HandOver's resolved source/destination participants.

    Returns:
        A screen for receiver-hand discrete commands against materialized
        non-target scene bodies. Target contact, payload, self collision,
        continuous collision and future obstacle motion are not certified.
    """
    if (
        simulation.num_envs != 1
        or not simulation.is_default_backend
        or context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or context.scene.timestamp != context.robot.timestamp
    ):
        raise ValueError("handover_scene_requires_same_tick_Default_B1_context")
    if not torch.equal(robot.get_qpos(), context.robot.qpos):
        raise ValueError("handover_scene_observed_qpos_changed")
    root = _pose(context.robot.root_pose)
    scene = simulation.spawn_result
    if scene is None:
        raise ValueError("handover_materialized_scene_unavailable")
    handles = [scene.get(path) for path in scene.get_paths()]
    required_uids = {
        item.simulation_uid
        for item in (
            *scene_binding.rigid_objects,
            *getattr(scene_binding, "articulations", ()),
        )
    }
    if any(
        sum(handle.get_name() == uid for handle in handles) != 1
        for uid in required_uids
    ):
        raise ValueError("handover_materialized_physical_inventory_unavailable")
    candidates = [
        handle
        for handle in handles
        if handle.get_name() == robot.uid and hasattr(handle, "render_articulation")
    ]
    if len(candidates) != 1:
        raise ValueError("handover_native_robot_identity_unavailable")
    native_robot = candidates[0]
    arena_from_world = np.linalg.inv(
        _pose(scene.get_arena(native_robot.arena_name).get_root_node().get_world_pose())
    )
    if not np.allclose(
        root, arena_from_world @ _pose(native_robot.get_world_pose()), atol=2e-4, rtol=0
    ):
        raise ValueError("handover_native_robot_root_frame_mismatch")
    target_id = request.goal.semantics.entity_id
    bindings = [
        item for item in scene_binding.rigid_objects if item.entity_id == target_id
    ]
    if len(bindings) != 1:
        raise ValueError("handover_exact_target_rigid_binding_unavailable")
    target_uid = bindings[0].simulation_uid
    excluded = [
        handle
        for handle in handles
        if handle.get_name() == target_uid and callable(getattr(handle, "native", None))
    ]
    if len(excluded) != 1:
        raise ValueError("handover_exact_target_native_identity_unavailable")
    hand = resources.second.hand
    destination = request.binding.endpoint("destination", "grasp").require_target(
        JointPositionTarget
    )
    if hand.address_fingerprint != destination.address_fingerprint:
        raise ValueError("handover_receiver_binding_mismatch")
    links, authored = _hand_links(robot, hand, resources.second.arm)
    if not set(links) <= set(native_robot.get_link_names()):
        raise ValueError("handover_native_receiver_topology_unavailable")
    observed = robot.compute_fk(
        qpos=context.robot.qpos.detach().clone(),
        link_names=list(links),
        qpos_joint_names=tuple(robot.joint_names),
    )
    if observed.shape != (1, len(links), 4, 4) or not torch.isfinite(observed).all():
        raise ValueError("handover_observed_named_hand_fk_unavailable")
    parts, obstacles, inventory = [], [], []
    for index, link in enumerate(links):
        expected = root @ observed[0, index].detach().cpu().numpy()
        native_pose = arena_from_world @ _pose(native_robot.get_link_pose(link))
        if not np.allclose(expected, native_pose, atol=2e-4, rtol=0):
            raise ValueError("handover_actual_mimic_hand_fk_frame_mismatch")
        body = native_robot.render_articulation.get_physical_body(link)
        if link not in authored and (
            body is None
            or (body.is_collision_enable() and int(body.get_shape_count()) == 0)
        ):
            inventory.append(
                {
                    "owner": robot.uid,
                    "link": link,
                    "status": "authored_collisionless_hand_frame",
                }
            )
            continue
        part = _body(f"{robot.uid}/{link}", robot.uid, link, body, np.eye(4))
        inventory.append(
            {
                "owner": robot.uid,
                "link": link,
                "status": "enabled" if part else "native_collision_disabled",
                "shapes": len(part.meshes) if part else 0,
            }
        )
        if part is not None:
            if part.planes:
                raise ValueError(
                    f"handover_receiver_infinite_shape_unsupported:{part.name}"
                )
            parts.append(part)
    if not parts:
        raise ValueError("handover_receiver_enabled_native_geometry_unavailable")
    for handle in handles:
        if handle is native_robot or handle is excluded[0]:
            continue
        uid = handle.get_name()
        if hasattr(handle, "render_articulation"):
            for link in handle.get_link_names():
                body = handle.render_articulation.get_physical_body(link)
                if body is None or (
                    body.is_collision_enable() and int(body.get_shape_count()) == 0
                ):
                    if not handle.get_link_desc(link).collisions:
                        inventory.append(
                            {
                                "owner": uid,
                                "link": link,
                                "status": "retained_collisionless_link",
                            }
                        )
                        continue
                pose = arena_from_world @ _pose(handle.get_link_pose(link))
                obstacle = _body(f"{handle.path}/{link}", uid, link, body, pose)
                inventory.append(
                    {
                        "owner": uid,
                        "link": link,
                        "status": (
                            "enabled" if obstacle else "native_collision_disabled"
                        ),
                        "shapes": (
                            len(obstacle.meshes) + len(obstacle.planes)
                            if obstacle
                            else 0
                        ),
                        "plane_shapes": len(obstacle.planes) if obstacle else 0,
                        "retained_body_scale": np.asarray(
                            handle.desc.body_scale
                        ).tolist(),
                        "geometry_scale_source": "native_shape_geometry_only",
                    }
                )
                if obstacle is not None:
                    obstacles.append(obstacle)
        elif callable(getattr(handle, "native", None)):
            body = handle.native().get_physical_body()
            obstacle = _body(
                handle.path,
                uid,
                None,
                body,
                arena_from_world @ _pose(handle.get_world_pose()),
            )
            inventory.append(
                {
                    "owner": uid,
                    "link": None,
                    "status": "enabled" if obstacle else "native_collision_disabled",
                    "shapes": (
                        len(obstacle.meshes) + len(obstacle.planes) if obstacle else 0
                    ),
                    "plane_shapes": len(obstacle.planes) if obstacle else 0,
                    "geometry_scale_source": "native_shape_geometry_only",
                }
            )
            if obstacle is not None:
                obstacles.append(obstacle)
    if not obstacles:
        raise ValueError("handover_non_target_native_scene_geometry_unavailable")
    if not torch.equal(robot.get_qpos(), context.robot.qpos):
        raise ValueError("handover_scene_snapshot_robot_changed")
    return _EnvironmentScreen(
        robot, request, context, root, tuple(parts), tuple(obstacles), inventory
    )
