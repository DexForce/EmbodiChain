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

"""Task-owned grasp filtering through the baseline sampling-service interface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch

from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
from embodichain.lab.task_program.semantics import (
    GRASP_AFFORDANCE_CAPABILITY,
    SceneObjectRef,
)
from embodichain.utils import logger
from embodichain.utils.math import pose_inv

__all__: list[str] = []

GRASP_FILTER_REVISION = 2


def opening_envelope_mask(
    vertices: torch.Tensor,
    poses: torch.Tensor,
    object_pose: torch.Tensor,
    opening: float,
) -> torch.Tensor:
    """Require the full object to fit between open pads, not only a sampled chord."""
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("Opening clearance requires non-empty mesh vertices (N, 3).")
    relative = pose_inv(object_pose) @ poses
    accepted = torch.isfinite(relative).all(-1).all(-1)
    # Leave clearance at both pads beyond the millimetre-scale contact envelope.
    half_gap = opening * 0.5 - 0.002
    for start in range(0, len(poses), 64):
        chunk = relative[start : start + 64]
        axes = chunk[:, :3, 0]
        centers = (chunk[:, :3, 3] * axes).sum(-1)
        projection = vertices.to(poses) @ axes.T - centers[None]
        accepted[start : start + len(chunk)] &= torch.isfinite(projection).all(0) & (
            projection.abs().amax(0) <= half_gap
        )
    return accepted


def geometry_key(vertices: torch.Tensor, triangles: torch.Tensor) -> str:
    """Match immutable local geometry, not object world poses or mutable call state."""
    digest = hashlib.sha256()
    for value, dtype in ((vertices, torch.float32), (triangles, torch.int64)):
        data = value.detach().to(device="cpu", dtype=dtype).contiguous()
        digest.update(str(tuple(data.shape)).encode("ascii"))
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class GraspRule:
    object_id: str
    target_id: str
    geometry: str
    local_axis: tuple[float, float, float]
    midpoint: float
    upper_half: bool
    release_pose: tuple[float, ...] | None
    plane_z: float | None
    margin: float


def accepted_candidates(
    poses: torch.Tensor,
    object_pose: torch.Tensor,
    rule: GraspRule,
    model: Any,
) -> torch.Tensor:
    """Check each candidate in the object frame and at its declared release pose."""
    object_to_grasp = pose_inv(object_pose) @ poses
    accepted = torch.isfinite(poses).all(dim=-1).all(dim=-1)
    if rule.upper_half:
        projected = object_to_grasp[:, :3, 3] @ poses.new_tensor(rule.local_axis)
        accepted &= projected > rule.midpoint
    if rule.release_pose is not None:
        final_grasp = (
            poses.new_tensor(rule.release_pose).reshape(4, 4) @ object_to_grasp
        )
        opening = float(model.max_opening_width)
        corners = poses.new_tensor(
            [
                [side * opening + dx, dy, dz]
                for side in (-1.0, 1.0)
                for dx in (-0.5 * model.finger_thickness, 0.5 * model.finger_thickness)
                for dy in (-0.5 * model.finger_width, 0.5 * model.finger_width)
                for dz in (-0.5 * model.finger_length, 0.5 * model.finger_length)
            ]
        )
        world = (
            corners @ final_grasp[:, :3, :3].transpose(-1, -2)
            + final_grasp[:, None, :3, 3]
        )
        assert rule.plane_z is not None
        accepted &= world[..., 2].amin(-1) > rule.plane_z + rule.margin
    return accepted


class TaskGraspPoseGenerator(ParallelJawGraspPoseGenerator):
    """Check single-arm opening clearance and configured object-region rules.

    Rules are fixed at assembly. End-specific proposals bypass region rules but
    still require opening clearance; dual grasps retain their paired protocol.
    This provider owns no
    execution cursor, held relation, eligibility state, or recovery policy.
    """

    def __init__(
        self, delegate: ParallelJawGraspPoseGenerator, rules: tuple[GraspRule, ...]
    ) -> None:
        super().__init__(delegate.gripper_model)
        self._delegate = delegate
        self._rules = rules
        keys = {rule.geometry for rule in rules}
        self._by_geometry = {
            key: tuple(rule for rule in rules if rule.geometry == key) for key in keys
        }

    def require_rule(
        self,
        object_id: str,
        target_id: str,
        vertices: torch.Tensor,
        triangles: torch.Tensor,
    ) -> None:
        key = geometry_key(vertices, triangles)
        if not any(
            rule.object_id == object_id
            and rule.target_id == target_id
            and rule.geometry == key
            for rule in self._rules
        ):
            raise ValueError("Constrained Pick has no matching immutable grasp filter.")

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        results = self._delegate.get_valid_grasp_poses(
            mesh_vertices=mesh_vertices,
            mesh_triangles=mesh_triangles,
            obj_poses=obj_poses,
            approach_direction=approach_direction,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=is_positive_part,
        )
        rules = (
            ()
            if obj_longest_axis is not None
            else self._by_geometry.get(geometry_key(mesh_vertices, mesh_triangles), ())
        )
        if obj_poses.shape != (len(results), 4, 4):
            raise ValueError("Grasp filter rows must match the observed object poses.")
        filtered = []
        counts = []
        for row, (poses, costs) in enumerate(results):
            if (
                poses.ndim != 3
                or poses.shape[1:] != (4, 4)
                or costs.shape != poses.shape[:1]
            ):
                raise ValueError(
                    "Grasp proposals must have matching pose and cost rows."
                )
            costs = costs.to(poses.device)
            accepted = torch.isfinite(costs)
            raw_count = int(accepted.sum())
            accepted &= opening_envelope_mask(
                mesh_vertices,
                poses,
                obj_poses[row].to(poses),
                self.gripper_model.max_opening_width,
            )
            opening_count = int(accepted.sum())
            for rule in rules:
                accepted &= accepted_candidates(
                    poses, obj_poses[row].to(poses), rule, self.gripper_model
                )
            counts.append(int(accepted.sum()))
            if not counts[-1]:
                logger.log_info(
                    "GenSim grasp filter rejected all proposals: "
                    f"raw={raw_count}, opening={opening_count}, "
                    f"rules={len(rules)}."
                )
            if accepted.any():
                filtered.append((poses[accepted].clone(), costs[accepted].clone()))
            else:
                # The baseline sampler represents a failed row by infinite cost.
                # A finite padding pose is never eligible for a command or effect.
                padding = (
                    poses[:1] if poses.shape[0] else obj_poses[row : row + 1].to(poses)
                )
                filtered.append((padding.clone(), poses.new_full((1,), torch.inf)))
        logger.log_info(f"GenSim grasp filter candidate counts: {counts}.")
        return filtered

    def get_best_grasp_poses(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retain the baseline articulation-service protocol, not used by GenSim Pick."""
        return self._delegate.get_best_grasp_poses(**kwargs)

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any) -> Any:
        """Coordinated grasps have a separate, unchanged paired-contact contract."""
        return self._delegate.get_dual_arm_valid_grasp_poses(**kwargs)


def install_grasp_filters(
    registration: Any, simulation: Any, generators: dict[str, Any]
) -> dict[str, Any]:
    """Assemble immutable geometry rules through the existing public scene binding."""
    routes = tuple(
        route
        for factory in registration.registered_semantic_lowerer_factories
        if factory.call_id == "simulation.pick"
        or factory.call_id.startswith("gen_sim.pick.")
        for route in factory.routes
        if route.grasp_region is not None
        or route.release_clearance_object_pose is not None
    )
    registry = registration.scene_binding.build(simulation)
    registration.validate_scene_registry(registry)
    rules = []
    for route in routes:
        ref = registry.resolve(route.object_id, expected_type=SceneObjectRef)
        grasp = registry.resolve_affordance(ref, capability=GRASP_AFFORDANCE_CAPABILITY)
        sem = registry.object_semantics(ref, affordance=grasp)
        affordance = sem.affordance
        axis = getattr(affordance, "internal_axis", None)
        if route.grasp_region is not None and axis is None:
            raise ValueError("Upper-half filtering requires a declared local axis.")
        axis = torch.tensor([0.0, 0.0, 1.0]) if axis is None else axis.detach().cpu()
        axis = torch.nn.functional.normalize(axis, dim=-1)
        projection = affordance.mesh_vertices.detach().cpu() @ axis
        release = route.release_clearance_object_pose
        if release is not None and not callable(getattr(release, "to_matrix", None)):
            raise ValueError(
                "Grasp filtering requires an explicit absolute release pose."
            )
        rules.append(
            GraspRule(
                route.object_id,
                route.target_id,
                geometry_key(affordance.mesh_vertices, affordance.mesh_triangles),
                tuple(float(x) for x in axis),
                float((projection.amin() + projection.amax()) * 0.5),
                route.grasp_region == "upper_half",
                (
                    None
                    if release is None
                    else tuple(float(x) for x in release.to_matrix().reshape(-1))
                ),
                route.release_clearance_plane_z,
                route.release_clearance_safety_margin,
            )
        )
    for preset in registration.robot_profile_binding.presets:
        for options in preset.action_option_templates.values():
            if getattr(options, "rotate_upright", None) is not None:
                raise ValueError(
                    "Post-sampling grasp rotation is incompatible with fixed grasp filters."
                )
    for generator in generators.values():
        if not isinstance(generator, ParallelJawGraspPoseGenerator):
            raise TypeError(
                "Task grasp filtering requires calibrated parallel-jaw generators."
            )
    return {
        name: TaskGraspPoseGenerator(generator, tuple(rules))
        for name, generator in generators.items()
    }
