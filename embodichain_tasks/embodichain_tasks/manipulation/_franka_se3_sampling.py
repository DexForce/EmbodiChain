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
"""Task-independent, IK-feasible Cartesian motion primitive sampling for Franka."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import warp as wp

import newton.ik as ik
from newton._src.sim.ik.ik_common import eval_fk_batched

from ._franka_reach import (
    DEFAULT_JOINT_Q,
    FRANKA_NUM_ARM_JOINTS,
    _build_franka_model,
    _quat_angle_distance,
)
from ._waypoint_sampling import (
    DIRECTION_TURN,
    NUM_ACTIVE_GROUPS,
    NUM_DIRECTION_RELATIONS,
    MultiscaleWaypointSampler,
    WaypointJointSamples,
    balanced_ids,
)

SE3_PRIMITIVE_JOINT_BASE = 0
SE3_PRIMITIVE_TRANSLATION = 1
SE3_PRIMITIVE_ROTATION = 2
SE3_PRIMITIVE_COUPLED = 3
NUM_SE3_PRIMITIVES = 4
SE3_PRIMITIVE_NAMES = ("joint_base", "translation", "rotation", "coupled")


def _balanced_complement_ids(
    existing_ids: torch.Tensor,
    count: int,
    num_categories: int,
    device: torch.device,
) -> torch.Tensor:
    """Fill the remaining rows to make combined category counts as even as possible."""
    count = int(count)
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    existing_counts = (
        torch.bincount(existing_ids.to(torch.long), minlength=int(num_categories))
        .cpu()
        .tolist()
    )
    allocations = [0] * int(num_categories)
    tie_order = torch.randperm(int(num_categories), device=device).cpu().tolist()
    tie_rank = {category: rank for rank, category in enumerate(tie_order)}
    for _ in range(count):
        minimum = min(existing_counts)
        category = min(
            (
                category
                for category, category_count in enumerate(existing_counts)
                if category_count == minimum
            ),
            key=tie_rank.__getitem__,
        )
        existing_counts[category] += 1
        allocations[category] += 1

    ids = torch.repeat_interleave(
        torch.arange(int(num_categories), dtype=torch.long, device=device),
        torch.tensor(allocations, dtype=torch.long, device=device),
    )
    return ids[torch.randperm(count, device=device)]


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    return torch.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class StratifiedSE3EpisodeSamples:
    """A batch of recursively sampled, FK-reachable waypoint sequences."""

    initial_joint_q: torch.Tensor
    waypoint_samples: WaypointJointSamples
    primitive_type: torch.Tensor


class FrankaStratifiedSE3Sampler:
    """Mix joint-space coverage with decoupled SE(3) motion primitives.

    Half of every batch requests joint-base targets.  The other half is balanced
    between translation-only, rotation-only, and coupled full-pose targets.
    Cartesian targets are solved from the preceding waypoint with
    analytic-Jacobian IK and rejected unless they satisfy pose accuracy, joint
    limits, and the per-waypoint action budget.  Accepted Cartesian segments are
    kept, then joint-base and fallback rows complement their realized joint-space
    scales and directions so the unified batch remains globally stratified.

    The distribution uses only robot kinematics and fixed geometric ranges; it
    does not consume downstream tasks or demonstrations.
    """

    # Six balanced slots give exactly 50% joint-base and 1/6 for each Cartesian
    # primitive when the batch size is divisible by six.
    _BALANCED_SLOT_TO_PRIMITIVE = (
        SE3_PRIMITIVE_JOINT_BASE,
        SE3_PRIMITIVE_JOINT_BASE,
        SE3_PRIMITIVE_JOINT_BASE,
        SE3_PRIMITIVE_TRANSLATION,
        SE3_PRIMITIVE_ROTATION,
        SE3_PRIMITIVE_COUPLED,
    )

    def __init__(
        self,
        joint_sampler: MultiscaleWaypointSampler,
        *,
        generation_batch_size: int,
        num_waypoints: int,
        translation_range: tuple[float, float] = (0.03, 0.20),
        rotation_range: tuple[float, float] = (0.15, 1.50),
        ik_iterations: int = 24,
        max_retries: int = 10,
        ik_position_tolerance: float = 2.0e-3,
        ik_rotation_tolerance: float = 2.0e-2,
    ) -> None:
        self.joint_sampler = joint_sampler
        self.device = joint_sampler.device
        self.dtype = joint_sampler.dtype
        self.generation_batch_size = int(generation_batch_size)
        self.num_waypoints = int(num_waypoints)
        self.translation_range = tuple(float(x) for x in translation_range)
        self.rotation_range = tuple(float(x) for x in rotation_range)
        self.ik_iterations = int(ik_iterations)
        self.max_retries = int(max_retries)
        self.ik_position_tolerance = float(ik_position_tolerance)
        self.ik_rotation_tolerance = float(ik_rotation_tolerance)

        if self.generation_batch_size < 1 or self.num_waypoints < 1:
            raise ValueError("generation_batch_size and num_waypoints must be positive")
        if not 0.0 < self.translation_range[0] < self.translation_range[1]:
            raise ValueError("translation_range must be positive and increasing")
        if not 0.0 < self.rotation_range[0] < self.rotation_range[1] <= math.pi:
            raise ValueError("rotation_range must be in (0, pi] and increasing")
        if self.ik_iterations < 1 or self.max_retries < 1:
            raise ValueError("ik_iterations and max_retries must be positive")

        self._setup_ik()
        self._pool_initial = torch.empty(
            0, self._num_coords, dtype=self.dtype, device=self.device
        )
        self._pool_waypoints = torch.empty(
            0,
            self.num_waypoints,
            self._num_coords,
            dtype=self.dtype,
            device=self.device,
        )
        self._pool_primitives = torch.empty(
            0, self.num_waypoints, dtype=torch.long, device=self.device
        )
        self._pool_num_waypoints = self.num_waypoints
        self.last_generation_stats: dict[str, float] = {}

    def clear_pool(self) -> None:
        """Discard prefetched episodes after an explicit RNG reseed."""
        self._reset_pool(self.num_waypoints)

    def _reset_pool(self, num_waypoints: int) -> None:
        self._pool_num_waypoints = int(num_waypoints)
        self._pool_initial = torch.empty(
            0, self._num_coords, dtype=self.dtype, device=self.device
        )
        self._pool_waypoints = torch.empty(
            0,
            self._pool_num_waypoints,
            self._num_coords,
            dtype=self.dtype,
            device=self.device,
        )
        self._pool_primitives = torch.empty(
            0,
            self._pool_num_waypoints,
            dtype=torch.long,
            device=self.device,
        )

    def _setup_ik(self) -> None:
        self._ik_model, self._ee_index = _build_franka_model(
            1, requires_grad=False, device=str(self.device)
        )
        self._num_coords = int(self._ik_model.joint_coord_count)
        batch = self.generation_batch_size

        self._ik_joint_q_in = wp.zeros(
            (batch, self._num_coords), dtype=wp.float32, device=self._ik_model.device
        )
        self._ik_joint_q_out = wp.zeros_like(self._ik_joint_q_in)
        self._ik_joint_qd = wp.zeros(
            (batch, self._ik_model.joint_dof_count),
            dtype=wp.float32,
            device=self._ik_model.device,
        )
        self._ik_body_q = wp.zeros(
            (batch, self._ik_model.body_count),
            dtype=wp.transform,
            device=self._ik_model.device,
        )
        self._ik_body_qd = wp.zeros(
            (batch, self._ik_model.body_count),
            dtype=wp.spatial_vector,
            device=self._ik_model.device,
        )

        target_positions = wp.zeros(batch, dtype=wp.vec3, device=self._ik_model.device)
        target_rotations = wp.zeros(batch, dtype=wp.vec4, device=self._ik_model.device)
        self._position_objective = ik.IKObjectivePosition(
            self._ee_index, wp.vec3(0.0, 0.0, 0.0), target_positions
        )
        self._rotation_objective = ik.IKObjectiveRotation(
            self._ee_index, wp.quat_identity(), target_rotations
        )

        safe_lower = wp.clone(self._ik_model.joint_limit_lower)
        safe_upper = wp.clone(self._ik_model.joint_limit_upper)
        safe_lower_t = wp.to_torch(safe_lower)
        safe_upper_t = wp.to_torch(safe_upper)
        safe_lower_t[:FRANKA_NUM_ARM_JOINTS] = self.joint_sampler.safe_lower
        safe_upper_t[:FRANKA_NUM_ARM_JOINTS] = self.joint_sampler.safe_upper
        joint_limit_objective = ik.IKObjectiveJointLimit(
            safe_lower, safe_upper, weight=10.0
        )
        self._ik_solver = ik.IKSolver(
            self._ik_model,
            batch,
            [
                self._position_objective,
                self._rotation_objective,
                joint_limit_objective,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _copy_torch_to_warp(self, dst: wp.array, src: torch.Tensor, dtype) -> None:
        wp.copy(dst, wp.from_torch(src.detach().contiguous(), dtype=dtype))

    def _fk(self, joint_q: torch.Tensor) -> torch.Tensor:
        self._copy_torch_to_warp(self._ik_joint_q_in, joint_q, wp.float32)
        eval_fk_batched(
            self._ik_model,
            self._ik_joint_q_in,
            self._ik_joint_qd,
            self._ik_body_q,
            self._ik_body_qd,
        )
        return wp.to_torch(self._ik_body_q)[:, self._ee_index].clone()

    def _sample_log_uniform(
        self, count: int, bounds: tuple[float, float]
    ) -> torch.Tensor:
        unit = torch.rand(count, dtype=self.dtype, device=self.device)
        lower, upper = bounds
        return torch.exp(math.log(lower) + unit * math.log(upper / lower))

    def _sample_pose_targets(
        self, current_pose: torch.Tensor, primitive: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = current_pose.shape[0]
        target_pos = current_pose[:, :3].clone()
        target_quat = current_pose[:, 3:7].clone()

        direction = torch.randn(count, 3, dtype=self.dtype, device=self.device)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        distance = self._sample_log_uniform(count, self.translation_range)
        translate = (primitive == SE3_PRIMITIVE_TRANSLATION) | (
            primitive == SE3_PRIMITIVE_COUPLED
        )
        target_pos[translate] += direction[translate] * distance[translate, None]

        axis = torch.randn(count, 3, dtype=self.dtype, device=self.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        angle = self._sample_log_uniform(count, self.rotation_range)
        half = 0.5 * angle
        delta_quat = torch.cat(
            [axis * torch.sin(half).unsqueeze(-1), torch.cos(half).unsqueeze(-1)],
            dim=-1,
        )
        rotate = (primitive == SE3_PRIMITIVE_ROTATION) | (
            primitive == SE3_PRIMITIVE_COUPLED
        )
        rotated = _quat_mul(delta_quat, target_quat)
        target_quat[rotate] = rotated[rotate]
        target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        return target_pos, target_quat

    def _solve_ik(
        self,
        current_joint_q: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._copy_torch_to_warp(self._ik_joint_q_in, current_joint_q, wp.float32)
        self._copy_torch_to_warp(
            self._position_objective.target_positions, target_pos, wp.vec3
        )
        self._copy_torch_to_warp(
            self._rotation_objective.target_rotations, target_quat, wp.vec4
        )
        self._ik_solver.step(
            self._ik_joint_q_in,
            self._ik_joint_q_out,
            iterations=self.ik_iterations,
            step_size=1.0,
        )
        solved_q = wp.to_torch(self._ik_joint_q_out).clone()
        solved_pose = self._fk(solved_q)
        return solved_q, solved_pose

    def _sample_cartesian_candidates(
        self,
        current_q: torch.Tensor,
        current_pose: torch.Tensor,
        primitive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cartesian = primitive != SE3_PRIMITIVE_JOINT_BASE
        accepted = ~cartesian
        output_q = current_q.clone()
        safe_lower = self.joint_sampler.safe_lower
        safe_upper = self.joint_sampler.safe_upper

        for _ in range(self.max_retries):
            if bool(accepted.all()):
                break
            target_pos, target_quat = self._sample_pose_targets(current_pose, primitive)
            # Rows already accepted are harmless identity IK problems. This
            # keeps one fixed-size solver and makes partial environment resets
            # draw from a pre-generated task pool rather than rebuilding IK.
            target_pos[accepted] = current_pose[accepted, :3]
            target_quat[accepted] = current_pose[accepted, 3:7]
            solved_q, solved_pose = self._solve_ik(current_q, target_pos, target_quat)
            position_error = (solved_pose[:, :3] - target_pos).norm(dim=-1)
            rotation_error = _quat_angle_distance(solved_pose[:, 3:7], target_quat)
            arm_q = solved_q[:, :FRANKA_NUM_ARM_JOINTS]
            h = (
                (arm_q - current_q[:, :FRANKA_NUM_ARM_JOINTS]).abs()
                / self.joint_sampler.action_scale
            ).amax(dim=-1)
            within_limits = (
                (arm_q >= safe_lower - 1.0e-5) & (arm_q <= safe_upper + 1.0e-5)
            ).all(dim=-1)
            valid = (
                (~accepted)
                & torch.isfinite(solved_q).all(dim=-1)
                & (position_error <= self.ik_position_tolerance)
                & (rotation_error <= self.ik_rotation_tolerance)
                & (h >= float(self.joint_sampler.bucket_lowers[0]) - 1.0e-6)
                & (h <= self.joint_sampler.max_h + 1.0e-6)
                & within_limits
            )
            output_q[valid] = solved_q[valid]
            accepted |= valid

        return output_q, accepted & cartesian

    def _generate_batch(
        self, num_waypoints: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        count = self.generation_batch_size
        num_waypoints = (
            self.num_waypoints if num_waypoints is None else int(num_waypoints)
        )
        if not 1 <= num_waypoints <= self.num_waypoints:
            raise ValueError("num_waypoints must be in [1, sampler.num_waypoints]")
        initial_q = (
            torch.tensor(DEFAULT_JOINT_Q, dtype=self.dtype, device=self.device)
            .unsqueeze(0)
            .expand(count, -1)
            .clone()
        )
        initial_q[:, :FRANKA_NUM_ARM_JOINTS] = self.joint_sampler.sample_start_arm_q(
            count
        )
        current_q = initial_q.clone()
        waypoint_qs = []
        primitive_types = []
        requested_cartesian = 0
        accepted_cartesian = 0
        previous_direction = None

        slot_mapping = torch.tensor(
            self._BALANCED_SLOT_TO_PRIMITIVE,
            dtype=torch.long,
            device=self.device,
        )
        for waypoint_idx in range(num_waypoints):
            slots = balanced_ids(
                count, len(self._BALANCED_SLOT_TO_PRIMITIVE), self.device
            )
            primitive = slot_mapping[slots]
            current_pose = self._fk(current_q)
            cartesian_q, accepted = self._sample_cartesian_candidates(
                current_q, current_pose, primitive
            )
            actual_primitive = torch.where(
                (primitive != SE3_PRIMITIVE_JOINT_BASE) & ~accepted,
                torch.full_like(primitive, SE3_PRIMITIVE_JOINT_BASE),
                primitive,
            )
            requested_cartesian += int(
                (primitive != SE3_PRIMITIVE_JOINT_BASE).sum().item()
            )
            accepted_cartesian += int(accepted.sum().item())

            cartesian = actual_primitive != SE3_PRIMITIVE_JOINT_BASE
            joint_base = ~cartesian
            joint_base_count = int(joint_base.sum())
            next_q = cartesian_q.clone()

            cartesian_delta = (
                cartesian_q[:, :FRANKA_NUM_ARM_JOINTS]
                - current_q[:, :FRANKA_NUM_ARM_JOINTS]
            ) / self.joint_sampler.action_scale
            cartesian_h = cartesian_delta.abs().amax(dim=-1)
            cartesian_bucket = torch.bucketize(
                cartesian_h[cartesian], self.joint_sampler.bucket_lowers[1:]
            )
            base_bucket_ids = _balanced_complement_ids(
                cartesian_bucket,
                joint_base_count,
                self.joint_sampler.num_distance_buckets,
                self.device,
            )
            active_group_ids = balanced_ids(
                joint_base_count, NUM_ACTIVE_GROUPS, self.device
            )

            if previous_direction is None:
                base_relation_ids = torch.full(
                    (joint_base_count,),
                    DIRECTION_TURN,
                    dtype=torch.long,
                    device=self.device,
                )
                base_previous_direction = None
            else:
                cartesian_direction = cartesian_delta[cartesian] / cartesian_h[
                    cartesian
                ].unsqueeze(-1).clamp_min(1.0e-8)
                cartesian_relation = self.joint_sampler.classify_direction_relation(
                    previous_direction[cartesian], cartesian_direction
                )
                base_relation_ids = _balanced_complement_ids(
                    cartesian_relation,
                    joint_base_count,
                    NUM_DIRECTION_RELATIONS,
                    self.device,
                )
                base_previous_direction = previous_direction[joint_base]

            base_segment = self.joint_sampler.sample_stratified_segment(
                current_q[joint_base, :FRANKA_NUM_ARM_JOINTS],
                previous_direction=base_previous_direction,
                bucket_ids=base_bucket_ids,
                active_group_ids=active_group_ids,
                relation_ids=base_relation_ids,
            )
            next_q[joint_base, :FRANKA_NUM_ARM_JOINTS] = base_segment.joint_q

            actual_delta = (
                next_q[:, :FRANKA_NUM_ARM_JOINTS] - current_q[:, :FRANKA_NUM_ARM_JOINTS]
            ) / self.joint_sampler.action_scale
            actual_h = actual_delta.abs().amax(dim=-1)
            previous_direction = actual_delta / actual_h.unsqueeze(-1).clamp_min(1.0e-8)
            current_q = next_q
            waypoint_qs.append(current_q.clone())
            primitive_types.append(actual_primitive)

        waypoint_qs_t = torch.stack(waypoint_qs, dim=1)
        primitive_t = torch.stack(primitive_types, dim=1)
        self.last_generation_stats = {
            "requested_cartesian": float(requested_cartesian),
            "accepted_cartesian": float(accepted_cartesian),
            "cartesian_acceptance_rate": accepted_cartesian
            / max(requested_cartesian, 1),
        }
        return initial_q, waypoint_qs_t, primitive_t

    def sample(
        self, count: int, *, num_waypoints: int | None = None
    ) -> StratifiedSE3EpisodeSamples:
        """Draw episodes from a fixed-size generation pool for cheap partial resets."""
        count = int(count)
        if not 1 <= count <= self.generation_batch_size:
            raise ValueError("count must be in [1, generation_batch_size]")
        num_waypoints = (
            self.num_waypoints if num_waypoints is None else int(num_waypoints)
        )
        if not 1 <= num_waypoints <= self.num_waypoints:
            raise ValueError("num_waypoints must be in [1, sampler.num_waypoints]")
        if num_waypoints != self._pool_num_waypoints:
            # Stratified APG resets a full generation batch at one K, so this
            # normally discards no samples.  Keeping a single pool also avoids
            # retaining K-specific GPU buffers that are never used together.
            self._reset_pool(num_waypoints)
        if self._pool_initial.shape[0] < count:
            initial, waypoints, primitive = self._generate_batch(num_waypoints)
            self._pool_initial = torch.cat([self._pool_initial, initial], dim=0)
            self._pool_waypoints = torch.cat([self._pool_waypoints, waypoints], dim=0)
            self._pool_primitives = torch.cat([self._pool_primitives, primitive], dim=0)

        initial = self._pool_initial[:count].clone()
        waypoint_qs = self._pool_waypoints[:count].clone()
        primitive = self._pool_primitives[:count].clone()
        self._pool_initial = self._pool_initial[count:]
        self._pool_waypoints = self._pool_waypoints[count:]
        self._pool_primitives = self._pool_primitives[count:]
        metadata = self.joint_sampler.describe_waypoints(
            initial[:, :FRANKA_NUM_ARM_JOINTS],
            waypoint_qs[:, :, :FRANKA_NUM_ARM_JOINTS],
        )
        return StratifiedSE3EpisodeSamples(initial, metadata, primitive)
