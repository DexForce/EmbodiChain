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

"""Atomic command and candidate qualification over an owned native Scene query."""

from __future__ import annotations
import math
from typing import Any, TYPE_CHECKING
import numpy as np
import torch
from embodichain.compute.geometry.pose_checks import as_rigid_matrix as _pose
from embodichain.compute.geometry.mesh_bounds import (
    input_mesh_bounds,
    input_mesh_radius,
    iter_clearance_translations,
)
from embodichain.lab.sim.motion.solvers.tolerances import configured_pose_uncertainty
from embodichain.lab.sim.motion.ik import sample_cartesian_targets as _cartesian_samples
from embodichain.lab.sim.scene.native_hand_scene import NativeHandSceneQuery
from ..bindings import JointPositionTarget
from ..runtime_commands import JointPositionPayload
from ..trajectory_ops import build_pose_plan_states, translate_pose_world
from ..state import PlanningContext
from ..invocation import ResolvedActionRequest

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__: list[str] = []
HANDOVER_SCENE_CLEARANCE_REVISION = 4
_SCOPE = "receiver_hand_non_target_scene_discrete_native_shapes"


def _bounds(body, pose):
    return input_mesh_bounds(body.meshes, pose)


class HandOverEnvironmentScreen:
    """Validate Atomic identity/commands and screen its original receive pool.

    The caller owns the Scene query lifetime. Scene owns native capture, named
    link FK and native query buffers; this adapter owns command/phase and pool identity.
    """

    def __init__(
        self,
        robot: Robot,
        request: ResolvedActionRequest,
        context: PlanningContext,
        query: NativeHandSceneQuery,
    ) -> None:
        stamp = context.world_source_stamp
        if (
            type(query) is not NativeHandSceneQuery
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.timestamp != context.scene.timestamp
            or query.scope.observation_timestamp != context.robot.timestamp
            or query.scope.scene_version != context.scene.version
            or query.robot_uid != robot.uid
            or tuple(query.joint_names) != tuple(robot.joint_names)
            or stamp is None
            or stamp.source_kind != "default.contacts"
            or stamp.revision != query.native_revision
        ):
            raise ValueError(
                "HandOver native query must match the captured invocation observation."
            )
        query.assert_current()
        self.robot, self.request, self.context, self.query = (
            robot,
            request,
            context,
            query,
        )
        self.root = query.root
        self.parts, self.obstacles = query.parts, query.obstacles
        self.obstacle_bounds = query.obstacle_bounds
        self.inventory = query.inventory
        self.joint_names = tuple(robot.joint_names)
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
        return self._inspect_hand_poses(
            self.query.link_poses(self.robot, qpos), self._report(len(qpos))
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
        links = self.query.link_poses(self.robot, qpos)
        if start.shape != (1, 4, 4) or not torch.isfinite(start).all():
            raise ValueError("handover_candidate_open_hand_named_fk_unavailable")
        start_array = _pose(start[0])
        relative = np.linalg.inv(start_array) @ links[0]
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
        position, rotation = configured_pose_uncertainty(
            self.robot, resources.second.arm.control_part
        )
        radius = max(
            input_mesh_radius(part.meshes, transform)
            for part, transform in zip(self.parts, relative, strict=True)
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
        for delta, axis_label, value in iter_clearance_translations(
            lower,
            upper,
            [(bound, clearance) for _, bound, clearance in near_meshes],
            [
                (normal, signed, clearance)
                for _, normal, signed, clearance in near_planes
            ],
            horizontal=1 - dominant,
            reference=poses,
        ):
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
                    "axis": axis_label,
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
            "native_queries": 0,
            "plane_queries": 0,
            "minimum_plane_signed_distance_m": None,
        }

    def _inspect_hand_poses(
        self, arena_poses: np.ndarray, report: dict[str, Any]
    ) -> dict[str, Any]:
        return {**report, **self.query.inspect(arena_poses)}
