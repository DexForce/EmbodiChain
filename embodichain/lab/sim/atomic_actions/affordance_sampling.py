# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Reproducible pose sampling within Affordance-owned geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math

import torch

__all__ = [
    "AffordancePoseCandidates",
    "AffordanceSample",
    "AffordanceSamplingContext",
]


@dataclass(frozen=True, slots=True)
class AffordanceSamplingContext:
    """Immutable identity for reproducible parallel Affordance sampling.

    Args:
        count: Number of logical branches in each sampling group.
        seed: Host-selected base seed.
        episode_id: Host-selected episode or batch identity.
        attempt_id: Explicit resampling attempt identity.
    """

    count: int = 1
    seed: int = 0
    episode_id: int = 0
    attempt_id: int = 0

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 1:
            raise ValueError("count must be a positive integer.")
        for name in ("seed", "episode_id", "attempt_id"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    @property
    def enabled(self) -> bool:
        """Whether more than the nominal branch was requested."""
        return self.count > 1

    def generator(self, key: str, *, group: int = 0) -> torch.Generator:
        """Create a private CPU generator for a stable named stream.

        Args:
            key: Non-empty sampling operation identity.
            group: Non-negative logical branch-group identity.

        Returns:
            A generator independent of PyTorch's global random state.
        """
        if type(key) is not str or not key:
            raise ValueError("key must be a non-empty string.")
        if type(group) is not int or group < 0:
            raise ValueError("group must be a non-negative integer.")
        payload = repr((self.seed, self.episode_id, self.attempt_id, key, group))
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
        return torch.Generator(device="cpu").manual_seed(seed % (2**63))

    def sample_range(
        self,
        nominal: float | torch.Tensor,
        bounds: tuple[float, float] | None,
        *,
        env_ids: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        """Stratify a legal scalar interval while retaining nominal branches.

        Args:
            nominal: Scalar or one value per environment row.
            bounds: Inclusive lower and upper bounds, or ``None`` for no freedom.
            env_ids: Stable non-negative environment identities, shape ``(B,)``.
            key: Sampling operation identity.

        Returns:
            Sampled values with shape ``(B,)`` on the environment device.
        """
        _validate_env_ids(env_ids)
        values = torch.as_tensor(nominal, dtype=torch.float32, device=env_ids.device)
        if values.ndim == 0 or values.shape == (1,):
            values = values.reshape(1).expand(env_ids.numel()).clone()
        elif values.shape == env_ids.shape:
            values = values.clone()
        else:
            raise ValueError("nominal must be scalar or match env_ids.")
        if not torch.isfinite(values).all():
            raise ValueError("nominal must contain only finite values.")
        if bounds is None or not self.enabled:
            return values
        lower, upper = _validate_range(bounds, name=key)
        if ((values < lower) | (values > upper)).any():
            raise ValueError("nominal must lie inside the sampling bounds.")

        groups: dict[int, torch.Tensor] = {}
        for row, env_id in enumerate(env_ids.cpu().tolist()):
            group, branch = divmod(env_id, self.count)
            if branch == 0:
                continue
            if group not in groups:
                generator = self.generator(key, group=group)
                strata = torch.randperm(self.count - 1, generator=generator)
                jitter = torch.rand(self.count - 1, generator=generator)
                groups[group] = (strata + jitter) / (self.count - 1)
            values[row] = lower + (upper - lower) * float(groups[group][branch - 1])
        return values

    def metadata(self) -> dict[str, int]:
        """Return JSON-compatible sampling provenance."""
        return {
            "count": self.count,
            "seed": self.seed,
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
        }


@dataclass(frozen=True, slots=True)
class AffordancePoseCandidates:
    """Batched pose candidates with cost and explicit validity.

    Args:
        poses: Homogeneous poses with shape ``(B, K, 4, 4)``.
        costs: Lower-is-better candidate costs with shape ``(B, K)``.
        valid: Real-candidate mask with shape ``(B, K)``.
    """

    poses: torch.Tensor
    costs: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.poses, torch.Tensor):
            raise TypeError("poses must be a torch.Tensor.")
        if self.poses.ndim != 4 or self.poses.shape[-2:] != (4, 4):
            raise ValueError("poses must have shape (B, K, 4, 4).")
        shape = self.poses.shape[:2]
        if not all(shape):
            raise ValueError(
                "candidate batch and candidate dimensions must be non-empty."
            )
        if not isinstance(self.costs, torch.Tensor) or self.costs.shape != shape:
            raise ValueError("costs must match candidate shape (B, K).")
        if not isinstance(self.valid, torch.Tensor) or self.valid.shape != shape:
            raise ValueError("valid must match candidate shape (B, K).")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must have dtype torch.bool.")
        if not self.poses.is_floating_point() or not self.costs.is_floating_point():
            raise TypeError("poses and costs must use floating-point dtypes.")
        if (
            self.costs.device != self.poses.device
            or self.valid.device != self.poses.device
        ):
            raise ValueError("candidate tensors must share a device.")
        object.__setattr__(self, "poses", self.poses.clone())
        object.__setattr__(self, "costs", self.costs.clone())
        object.__setattr__(self, "valid", self.valid.clone())

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        device: torch.device | str,
    ) -> AffordancePoseCandidates:
        """Pad variable candidate rows without making padding selectable.

        Args:
            rows: Per-row ``(poses, costs)`` values; rows may be empty.
            device: Output device.

        Returns:
            A rectangular candidate batch with explicit validity.
        """
        if not rows:
            raise ValueError("rows must be non-empty.")
        width = max(1, max(int(poses.shape[0]) for poses, _ in rows))
        output_device = torch.device(device)
        poses = torch.eye(4, dtype=torch.float32, device=output_device).repeat(
            len(rows), width, 1, 1
        )
        costs = torch.full(
            (len(rows), width), torch.inf, dtype=torch.float32, device=output_device
        )
        valid = torch.zeros((len(rows), width), dtype=torch.bool, device=output_device)
        for row, (row_poses, row_costs) in enumerate(rows):
            if not isinstance(row_poses, torch.Tensor) or not isinstance(
                row_costs, torch.Tensor
            ):
                raise TypeError("candidate rows must contain tensors.")
            count = int(row_poses.shape[0])
            if row_poses.shape != (count, 4, 4) or row_costs.shape != (count,):
                raise ValueError(
                    "candidate rows require poses (K, 4, 4) and costs (K,)."
                )
            poses[row, :count] = row_poses.to(device=output_device, dtype=torch.float32)
            costs[row, :count] = row_costs.to(device=output_device, dtype=torch.float32)
            valid[row, :count] = torch.isfinite(costs[row, :count]) & torch.isfinite(
                poses[row, :count]
            ).all(dim=(-1, -2))
        poses[~valid] = torch.eye(4, dtype=poses.dtype, device=output_device)
        return cls(poses=poses, costs=costs, valid=valid)


@dataclass(frozen=True, slots=True)
class AffordanceSample:
    """Owned result of one batched Affordance sampling operation.

    Args:
        success: Per-row legal-sample mask with shape ``(B,)``.
        poses: Selected poses with shape ``(B, 4, 4)``.
        metadata: Sampling provenance and selection diagnostics.
    """

    success: torch.Tensor
    poses: torch.Tensor
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.success, torch.Tensor):
            raise TypeError("success must be a torch.Tensor.")
        if self.success.dtype != torch.bool or self.success.ndim != 1:
            raise TypeError("success must be a one-dimensional boolean tensor.")
        if self.success.numel() == 0:
            raise ValueError("success must be non-empty.")
        if not isinstance(self.poses, torch.Tensor):
            raise TypeError("poses must be a torch.Tensor.")
        if self.poses.shape != (self.success.numel(), 4, 4):
            raise ValueError("poses must have shape (B, 4, 4) matching success.")
        if not self.poses.is_floating_point():
            raise TypeError("poses must use a floating-point dtype.")
        if self.poses.device != self.success.device:
            raise ValueError("success and poses must share a device.")
        if not torch.isfinite(self.poses).all():
            raise ValueError("poses must contain only finite values.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "success", self.success.clone())
        object.__setattr__(self, "poses", self.poses.clone())
        object.__setattr__(self, "metadata", dict(self.metadata))


def _sample_pose_candidates(
    candidates: AffordancePoseCandidates,
    sampling: AffordanceSamplingContext | None,
    *,
    env_ids: torch.Tensor,
    key: str,
    reference_poses: torch.Tensor | None = None,
) -> AffordanceSample:
    """Select quality-ranked, geometrically distinct candidates by branch."""
    _validate_env_ids(env_ids)
    if type(key) is not str or not key:
        raise ValueError("key must be a non-empty string.")
    if env_ids.shape != candidates.poses.shape[:1]:
        raise ValueError("env_ids must match the candidate batch.")
    if env_ids.device != candidates.poses.device:
        raise ValueError("env_ids and candidates must share a device.")
    if reference_poses is not None:
        if reference_poses.shape != (len(env_ids), 4, 4):
            raise ValueError("reference_poses must have shape (B, 4, 4).")
        if reference_poses.device != candidates.poses.device:
            raise ValueError("reference_poses and candidates must share a device.")

    eligible = (
        candidates.valid
        & torch.isfinite(candidates.costs)
        & torch.isfinite(candidates.poses).all(dim=(-1, -2))
    )
    masked_costs = candidates.costs.masked_fill(~eligible, torch.inf)
    selected = masked_costs.argmin(dim=1)
    success = eligible.any(dim=1)
    unique_counts: list[int] = []
    reused: list[bool] = []

    for row, env_id in enumerate(env_ids.cpu().tolist()):
        indices = torch.argsort(masked_costs[row], stable=True)
        indices = indices[eligible[row, indices]]
        if not len(indices):
            unique_counts.append(0)
            reused.append(False)
            continue
        if sampling is None or not sampling.enabled:
            unique_counts.append(len(indices))
            reused.append(False)
            continue

        indices = indices[: max(32, 4 * sampling.count)]
        geometry = candidates.poses[row, indices].clone()
        if reference_poses is not None:
            geometry = torch.linalg.inv(reference_poses[row]) @ geometry
        position = geometry[:, :3, 3]
        rotation = geometry[:, :3, :3].flatten(1)
        scale = torch.linalg.vector_norm(position.amax(0) - position.amin(0)).clamp_min(
            0.01
        )
        generator = sampling.generator(key, group=env_id // sampling.count)
        jitter = torch.rand(len(indices), generator=generator).to(
            candidates.poses.device
        )
        order = [0]
        available = torch.ones(
            len(indices), dtype=torch.bool, device=candidates.poses.device
        )
        nearest = torch.full((len(indices),), torch.inf, device=candidates.poses.device)
        while True:
            last = order[-1]
            position_distance = torch.linalg.vector_norm(
                position - position[last], dim=1
            )
            rotation_distance = torch.linalg.vector_norm(
                rotation - rotation[last], dim=1
            )
            available &= ~(
                (position_distance <= 1.0e-4) & (rotation_distance <= 1.0e-3)
            )
            nearest = torch.minimum(
                nearest, position_distance / scale + rotation_distance
            )
            if not available.any():
                break
            priority = nearest * (0.75 + 0.25 * jitter)
            order.append(int(priority.masked_fill(~available, -torch.inf).argmax()))
        branch = env_id % sampling.count
        selected[row] = indices[order[branch % len(order)]]
        unique_counts.append(len(order))
        reused.append(branch >= len(order))

    output = candidates.poses[
        torch.arange(len(env_ids), device=candidates.poses.device), selected
    ].clone()
    output[~success] = torch.eye(
        4, dtype=candidates.poses.dtype, device=candidates.poses.device
    )
    return AffordanceSample(
        success=success,
        poses=output,
        metadata={
            "key": key,
            "sampling": None if sampling is None else sampling.metadata(),
            "candidate_ids": torch.where(success, selected, -1).cpu().tolist(),
            "valid_candidate_counts": eligible.sum(1).cpu().tolist(),
            "unique_candidate_counts": unique_counts,
            "reused": reused,
        },
    )


def _roll_poses(poses: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate local x/y axes while preserving contact positions and z axes."""
    if poses.ndim != 3 or poses.shape[-2:] != (4, 4):
        raise ValueError("poses must have shape (B, 4, 4).")
    if angles.shape != poses.shape[:1] or angles.device != poses.device:
        raise ValueError("angles must match the pose batch and device.")
    result = poses.clone()
    cosine = angles.cos()[:, None]
    sine = angles.sin()[:, None]
    x_axis = poses[:, :3, 0]
    y_axis = poses[:, :3, 1]
    result[:, :3, 0] = cosine * x_axis + sine * y_axis
    result[:, :3, 1] = -sine * x_axis + cosine * y_axis
    return result


def _validate_env_ids(env_ids: torch.Tensor) -> None:
    if not isinstance(env_ids, torch.Tensor):
        raise TypeError("env_ids must be a torch.Tensor.")
    if env_ids.dtype != torch.long or env_ids.ndim != 1 or env_ids.numel() == 0:
        raise ValueError("env_ids must be a non-empty int64 vector.")
    if (env_ids < 0).any() or torch.unique(env_ids).numel() != env_ids.numel():
        raise ValueError("env_ids must be non-negative and unique.")


def _validate_range(bounds: tuple[float, float], *, name: str) -> tuple[float, float]:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{name} must contain lower and upper bounds.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in bounds
    ):
        raise ValueError(f"{name} must contain finite real bounds.")
    lower, upper = map(float, bounds)
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed upper bound.")
    return lower, upper
