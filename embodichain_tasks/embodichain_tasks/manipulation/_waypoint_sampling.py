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
"""Task-agnostic joint-space sampling for ordered waypoint constraints."""

from __future__ import annotations

from dataclasses import dataclass

import torch

ACTIVE_GROUP_SINGLE = 0
ACTIVE_GROUP_SPARSE = 1
ACTIVE_GROUP_DENSE = 2
NUM_ACTIVE_GROUPS = 3

DIRECTION_CONTINUE = 0
DIRECTION_TURN = 1
DIRECTION_REVERSE = 2
NUM_DIRECTION_RELATIONS = 3


@dataclass(frozen=True)
class WaypointJointSamples:
    """Sampled joint targets and the coverage strata that generated them."""

    joint_qs: torch.Tensor
    scale_h: torch.Tensor
    distance_bucket: torch.Tensor
    active_joint_count: torch.Tensor
    direction_relation: torch.Tensor


@dataclass(frozen=True)
class WaypointJointSegmentSamples:
    """One recursively sampled joint-space segment for a batch of paths."""

    joint_q: torch.Tensor
    direction: torch.Tensor
    scale_h: torch.Tensor
    active_joint_count: torch.Tensor
    direction_relation: torch.Tensor


def balanced_ids(count: int, num_categories: int, device: torch.device) -> torch.Tensor:
    """Return randomly ordered category ids with counts differing by at most one."""
    category_order = torch.randperm(int(num_categories), device=device)
    ids = category_order[torch.arange(count, device=device) % int(num_categories)]
    return ids[torch.randperm(count, device=device)]


class MultiscaleWaypointSampler:
    """Sample recursive, feasible waypoint segments in normalized control scale.

    Segment difficulty is
    ``h = max_j(abs(delta_q_j) / action_scale_j)``. Distance buckets and active
    joint groups are balanced independently within every sampled waypoint.
    """

    def __init__(
        self,
        joint_lower: torch.Tensor,
        joint_upper: torch.Tensor,
        action_scale: float | torch.Tensor,
        *,
        distance_bucket_lowers: tuple[float, ...],
        max_h: float,
        joint_limit_margin: float,
        max_retries: int = 64,
        sobol_seed: int = 1,
    ) -> None:
        self.device = joint_lower.device
        self.dtype = joint_lower.dtype
        self.joint_lower = joint_lower.detach().clone()
        self.joint_upper = joint_upper.detach().clone()
        self.num_joints = int(self.joint_lower.numel())

        margin = torch.as_tensor(
            joint_limit_margin, dtype=self.dtype, device=self.device
        )
        self.safe_lower = self.joint_lower + margin
        self.safe_upper = self.joint_upper - margin
        if bool((self.safe_lower >= self.safe_upper).any()):
            raise ValueError("joint_limit_margin leaves an empty safe joint range")

        scale = torch.as_tensor(action_scale, dtype=self.dtype, device=self.device)
        if scale.numel() == 1:
            scale = scale.expand(self.num_joints)
        if scale.shape != (self.num_joints,) or bool((scale <= 0.0).any()):
            raise ValueError("action_scale must be positive and scalar or per-joint")
        self.action_scale = scale

        lowers = torch.as_tensor(
            distance_bucket_lowers, dtype=self.dtype, device=self.device
        )
        if lowers.ndim != 1 or lowers.numel() < 1:
            raise ValueError("distance_bucket_lowers must contain at least one value")
        if bool((lowers <= 0.0).any()) or bool((lowers[1:] <= lowers[:-1]).any()):
            raise ValueError("distance_bucket_lowers must be positive and increasing")
        self.max_h = float(max_h)
        if self.max_h < float(lowers[-1]):
            raise ValueError("max_h must be >= the final distance-bucket lower bound")
        self.bucket_lowers = lowers
        self.bucket_uppers = torch.cat([lowers[1:], lowers.new_tensor([self.max_h])])
        self.num_distance_buckets = int(lowers.numel())
        self.max_retries = int(max_retries)
        if self.max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        self.reset_sobol(sobol_seed)

    def reset_sobol(self, seed: int) -> None:
        """Restart the scrambled Sobol start-state stream."""
        self._sobol = torch.quasirandom.SobolEngine(
            dimension=self.num_joints,
            scramble=True,
            seed=int(seed) % (2**31 - 1),
        )

    def sample_start_arm_q(self, count: int) -> torch.Tensor:
        """Draw low-discrepancy starts inside the safe joint range."""
        unit = self._sobol.draw(int(count)).to(device=self.device, dtype=self.dtype)
        return self.safe_lower + unit * (self.safe_upper - self.safe_lower)

    def _sample_active_counts(self, group_ids: torch.Tensor) -> torch.Tensor:
        counts = torch.ones_like(group_ids)
        sparse = group_ids == ACTIVE_GROUP_SPARSE
        dense = group_ids == ACTIVE_GROUP_DENSE
        counts[sparse] = 2 + torch.randint(
            0, 2, (int(sparse.sum()),), device=self.device
        )
        counts[dense] = 4 + torch.randint(
            0, self.num_joints - 3, (int(dense.sum()),), device=self.device
        )
        return counts

    def _candidate_direction(
        self,
        active_counts: torch.Tensor,
        relation_ids: torch.Tensor,
        previous_direction: torch.Tensor | None,
    ) -> torch.Tensor:
        count = int(active_counts.shape[0])
        ranks = (
            torch.rand(count, self.num_joints, device=self.device)
            .argsort(dim=-1)
            .argsort(dim=-1)
        )
        active = ranks < active_counts.unsqueeze(-1)
        random_direction = torch.randn(
            count, self.num_joints, dtype=self.dtype, device=self.device
        )

        direction = random_direction
        if previous_direction is not None:
            continued = previous_direction + 0.15 * random_direction
            reversed_direction = -previous_direction + 0.15 * random_direction
            direction = torch.where(
                (relation_ids == DIRECTION_CONTINUE).unsqueeze(-1),
                continued,
                direction,
            )
            direction = torch.where(
                (relation_ids == DIRECTION_REVERSE).unsqueeze(-1),
                reversed_direction,
                direction,
            )

        direction = direction * active.to(self.dtype)
        norm = direction.abs().amax(dim=-1, keepdim=True)
        fallback = random_direction * active.to(self.dtype)
        direction = torch.where(norm > 1.0e-8, direction, fallback)
        direction = direction / direction.abs().amax(dim=-1, keepdim=True).clamp_min(
            1.0e-8
        )
        return self._make_primary_dominant(direction)

    @staticmethod
    def _make_primary_dominant(direction: torch.Tensor) -> torch.Tensor:
        """Keep one unit component and cap cooperative components at 0.35."""
        primary_idx = direction.abs().argmax(dim=-1, keepdim=True)
        primary = direction.gather(dim=-1, index=primary_idx)
        return direction.clamp(min=-0.35, max=0.35).scatter(
            dim=-1, index=primary_idx, src=primary
        )

    def _direction_h_max(
        self, current: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        delta_per_h = direction * self.action_scale
        positive = delta_per_h > 1.0e-8
        negative = delta_per_h < -1.0e-8
        capacity = torch.full_like(delta_per_h, float("inf"))
        capacity = torch.where(
            positive, (self.safe_upper - current) / delta_per_h, capacity
        )
        capacity = torch.where(
            negative, (self.safe_lower - current) / delta_per_h, capacity
        )
        return capacity.amin(dim=-1)

    @staticmethod
    def classify_direction_relation(
        previous_direction: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        cosine = torch.nn.functional.cosine_similarity(
            direction, previous_direction, dim=-1
        )
        relation = torch.full_like(cosine, DIRECTION_TURN, dtype=torch.long)
        relation = torch.where(
            cosine > 0.5,
            torch.full_like(relation, DIRECTION_CONTINUE),
            relation,
        )
        return torch.where(
            cosine < -0.5,
            torch.full_like(relation, DIRECTION_REVERSE),
            relation,
        )

    def _feasible_fallback_direction(
        self,
        current: torch.Tensor,
        active_counts: torch.Tensor,
        lower_h: torch.Tensor,
    ) -> torch.Tensor:
        """Choose active joints by available range and point into free space."""
        positive_capacity = (self.safe_upper - current) / self.action_scale
        negative_capacity = (current - self.safe_lower) / self.action_scale
        capacity = torch.maximum(positive_capacity, negative_capacity)
        sign = torch.where(
            positive_capacity >= negative_capacity,
            torch.ones_like(capacity),
            -torch.ones_like(capacity),
        )
        ranks = capacity.argsort(dim=-1, descending=True).argsort(dim=-1)
        active = ranks < active_counts.unsqueeze(-1)
        # The best joint defines h. Other selected joints use the largest
        # magnitude that cannot make them the limiting factor.
        magnitude = torch.minimum(
            torch.ones_like(capacity),
            0.5 * capacity / lower_h.unsqueeze(-1).clamp_min(1.0e-8),
        )
        magnitude = torch.where(ranks == 0, torch.ones_like(magnitude), magnitude)
        direction = sign * magnitude * active.to(self.dtype)
        return self._make_primary_dominant(direction)

    def _sample_segment(
        self,
        current: torch.Tensor,
        previous_direction: torch.Tensor | None,
        bucket_ids: torch.Tensor,
        active_group_ids: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = int(current.shape[0])
        active_counts = self._sample_active_counts(active_group_ids)
        accepted = torch.zeros(count, dtype=torch.bool, device=self.device)
        directions = torch.zeros_like(current)
        scales = torch.zeros(count, dtype=self.dtype, device=self.device)
        feasible_relations = relation_ids.clone()

        lower_h = self.bucket_lowers[bucket_ids]
        bucket_upper_h = self.bucket_uppers[bucket_ids]
        for retry in range(self.max_retries):
            # A long continuation can become impossible near a joint boundary.
            # Preserve it for most retries, then fall back to an unconstrained
            # turn rather than clipping the target or emitting an invalid edge.
            if retry == (3 * self.max_retries) // 4:
                feasible_relations = torch.where(
                    accepted,
                    feasible_relations,
                    torch.full_like(feasible_relations, DIRECTION_TURN),
                )
            candidate = self._candidate_direction(
                active_counts, feasible_relations, previous_direction
            )
            feasible_upper = torch.minimum(
                bucket_upper_h, self._direction_h_max(current, candidate)
            )
            if previous_direction is None:
                relation_ok = torch.ones_like(accepted)
            else:
                candidate_relation = self.classify_direction_relation(
                    previous_direction, candidate
                )
                relation_ok = candidate_relation == feasible_relations
            newly_accepted = (
                (~accepted) & (feasible_upper >= lower_h - 1.0e-6) & relation_ok
            )
            if bool(newly_accepted.any()):
                ratio = (feasible_upper / lower_h).clamp_min(1.0)
                unit = torch.rand(count, dtype=self.dtype, device=self.device)
                sampled_h = lower_h * torch.exp(unit * torch.log(ratio))
                directions[newly_accepted] = candidate[newly_accepted]
                scales[newly_accepted] = sampled_h[newly_accepted]
                accepted = accepted | newly_accepted
            if bool(accepted.all()):
                break

        if not bool(accepted.all()):
            fallback = self._feasible_fallback_direction(
                current, active_counts, lower_h
            )
            feasible_upper = torch.minimum(
                bucket_upper_h, self._direction_h_max(current, fallback)
            )
            newly_accepted = (~accepted) & (feasible_upper >= lower_h - 1.0e-6)
            if bool(newly_accepted.any()):
                ratio = (feasible_upper / lower_h).clamp_min(1.0)
                unit = torch.rand(count, dtype=self.dtype, device=self.device)
                sampled_h = lower_h * torch.exp(unit * torch.log(ratio))
                directions[newly_accepted] = fallback[newly_accepted]
                scales[newly_accepted] = sampled_h[newly_accepted]
                if previous_direction is not None:
                    fallback_relation = self.classify_direction_relation(
                        previous_direction, fallback
                    )
                    feasible_relations[newly_accepted] = fallback_relation[
                        newly_accepted
                    ]
                accepted = accepted | newly_accepted

        if not bool(accepted.all()):
            failed = int((~accepted).sum().item())
            raise RuntimeError(
                f"Could not sample {failed} feasible waypoint segments after "
                f"{self.max_retries} attempts"
            )

        next_q = current + scales.unsqueeze(-1) * self.action_scale * directions
        if bool(
            (
                (next_q < self.safe_lower - 1.0e-5)
                | (next_q > self.safe_upper + 1.0e-5)
            ).any()
        ):
            raise RuntimeError("multiscale sampler produced an out-of-range waypoint")
        return next_q, directions, scales, active_counts, feasible_relations

    def _rebalance_direction_relations(
        self,
        current_before: torch.Tensor,
        previous_direction: torch.Tensor,
        bucket_ids: torch.Tensor,
        active_group_ids: torch.Tensor,
        requested_relations: torch.Tensor,
        next_q: torch.Tensor,
        directions: torch.Tensor,
        scales: torch.Tensor,
        active_counts: torch.Tensor,
        actual_relations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recover balanced relation marginals using feasible batch-level swaps."""
        target_counts = torch.bincount(
            requested_relations, minlength=NUM_DIRECTION_RELATIONS
        )
        for _ in range(4):
            made_progress = False
            for desired in range(NUM_DIRECTION_RELATIONS):
                actual_counts_by_relation = torch.bincount(
                    actual_relations, minlength=NUM_DIRECTION_RELATIONS
                )
                deficit = int(
                    (target_counts[desired] - actual_counts_by_relation[desired])
                    .clamp_min(0)
                    .item()
                )
                if deficit == 0:
                    continue
                surplus_relation = actual_counts_by_relation > target_counts
                candidates = (
                    surplus_relation[actual_relations]
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                if candidates.numel() == 0:
                    continue
                desired_relations = torch.full(
                    (candidates.numel(),),
                    desired,
                    dtype=torch.long,
                    device=self.device,
                )
                proposal = self._sample_segment(
                    current_before[candidates],
                    previous_direction[candidates],
                    bucket_ids[candidates],
                    active_group_ids[candidates],
                    desired_relations,
                )
                feasible = (proposal[4] == desired).nonzero(as_tuple=False).squeeze(-1)
                if feasible.numel() == 0:
                    continue
                selected_local = feasible[:deficit]
                selected = candidates[selected_local]
                next_q[selected] = proposal[0][selected_local]
                directions[selected] = proposal[1][selected_local]
                scales[selected] = proposal[2][selected_local]
                active_counts[selected] = proposal[3][selected_local]
                actual_relations[selected] = proposal[4][selected_local]
                made_progress = True
            if (
                torch.equal(
                    torch.bincount(actual_relations, minlength=NUM_DIRECTION_RELATIONS),
                    target_counts,
                )
                or not made_progress
            ):
                break
        return next_q, directions, scales, active_counts, actual_relations

    def sample_stratified_segment(
        self,
        current_arm_q: torch.Tensor,
        *,
        bucket_ids: torch.Tensor,
        active_group_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        previous_direction: torch.Tensor | None = None,
    ) -> WaypointJointSegmentSamples:
        """Sample one segment with caller-selected coverage strata.

        This entry point lets a higher-level sampler compose accepted Cartesian
        segments with joint-space segments while preserving one global set of
        distance and direction marginals.
        """
        current_before = current_arm_q
        next_q, direction, h, active_counts, actual_relations = self._sample_segment(
            current_before,
            previous_direction,
            bucket_ids,
            active_group_ids,
            relation_ids,
        )
        if previous_direction is not None:
            next_q, direction, h, active_counts, actual_relations = (
                self._rebalance_direction_relations(
                    current_before,
                    previous_direction,
                    bucket_ids,
                    active_group_ids,
                    relation_ids,
                    next_q,
                    direction,
                    h,
                    active_counts,
                    actual_relations,
                )
            )
        return WaypointJointSegmentSamples(
            joint_q=next_q,
            direction=direction,
            scale_h=h,
            active_joint_count=active_counts,
            direction_relation=actual_relations,
        )

    def sample_waypoints(
        self, initial_arm_q: torch.Tensor, num_waypoints: int
    ) -> WaypointJointSamples:
        """Generate recursive waypoints with balanced scale and sparsity strata."""
        count = int(initial_arm_q.shape[0])
        current = initial_arm_q.clone()
        previous_direction = None
        waypoint_qs = []
        scale_h = []
        bucket_ids_all = []
        active_counts_all = []
        relations_all = []

        for waypoint_idx in range(int(num_waypoints)):
            bucket_ids = balanced_ids(count, self.num_distance_buckets, self.device)
            active_group_ids = balanced_ids(count, NUM_ACTIVE_GROUPS, self.device)
            if waypoint_idx == 0:
                relation_ids = torch.full(
                    (count,), DIRECTION_TURN, dtype=torch.long, device=self.device
                )
                stored_relations = torch.full_like(relation_ids, -1)
            else:
                relation_ids = balanced_ids(count, NUM_DIRECTION_RELATIONS, self.device)
                stored_relations = relation_ids

            segment = self.sample_stratified_segment(
                current,
                previous_direction=previous_direction,
                bucket_ids=bucket_ids,
                active_group_ids=active_group_ids,
                relation_ids=relation_ids,
            )
            current = segment.joint_q
            direction = segment.direction
            h = segment.scale_h
            active_counts = segment.active_joint_count
            feasible_relations = segment.direction_relation
            waypoint_qs.append(current.clone())
            scale_h.append(h)
            bucket_ids_all.append(bucket_ids)
            active_counts_all.append(active_counts)
            relations_all.append(
                stored_relations if waypoint_idx == 0 else feasible_relations
            )
            previous_direction = direction

        return WaypointJointSamples(
            joint_qs=torch.stack(waypoint_qs, dim=1),
            scale_h=torch.stack(scale_h, dim=1),
            distance_bucket=torch.stack(bucket_ids_all, dim=1),
            active_joint_count=torch.stack(active_counts_all, dim=1),
            direction_relation=torch.stack(relations_all, dim=1),
        )

    def describe_waypoints(
        self, initial_arm_q: torch.Tensor, waypoint_arm_qs: torch.Tensor
    ) -> WaypointJointSamples:
        """Derive strata metadata for an externally supplied joint path."""
        path = torch.cat([initial_arm_q.unsqueeze(1), waypoint_arm_qs], dim=1)
        delta = path[:, 1:] - path[:, :-1]
        normalized_delta = delta / self.action_scale
        h = normalized_delta.abs().amax(dim=-1)
        bucket = torch.bucketize(h, self.bucket_lowers[1:], right=False)
        active_count = (delta.abs() > 1.0e-6).sum(dim=-1)
        relations = torch.full_like(active_count, -1)
        if waypoint_arm_qs.shape[1] > 1:
            directions = normalized_delta / h.unsqueeze(-1).clamp_min(1.0e-8)
            relation = self.classify_direction_relation(
                directions[:, :-1], directions[:, 1:]
            )
            relations[:, 1:] = relation
        return WaypointJointSamples(
            joint_qs=waypoint_arm_qs,
            scale_h=h,
            distance_bucket=bucket,
            active_joint_count=active_count,
            direction_relation=relations,
        )
