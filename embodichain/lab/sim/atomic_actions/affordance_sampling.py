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

"""Reproducible sampling within task-owned affordance constraints.

The candidate axis is independent of the simulation batch. Sampling never
reads live simulation state, changes shared geometry, or steps an environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from collections.abc import Sequence

import torch

from embodichain.utils import configclass

__all__ = [
    "AffordanceExpansionCfg",
    "AffordanceSamplingContext",
    "AffordancePoseCandidates",
]


@configclass
class AffordanceExpansionCfg:
    """Environment-owned expansion settings.

    Args:
        count: Total parallel branches; one preserves deterministic selection.
    """

    count: int = 1

    def __post_init__(self) -> None:
        if type(self.count) is not int or self.count < 1:
            raise ValueError("affordance expansion count must be a positive integer.")


@dataclass(frozen=True, slots=True)
class AffordanceSamplingContext:
    """Immutable random-stream identity, independent of observation ordering.

    Args:
        count: Total branches, including the nominal branch at index zero.
        seed: Effective environment seed.
        episode_id: Host-owned episode or collection batch identifier.
        attempt_id: Explicit resampling attempt, never an observation counter.
    """

    count: int = 1
    seed: int = 0
    episode_id: int = 0
    attempt_id: int = 0

    def __post_init__(self) -> None:
        AffordanceExpansionCfg(count=self.count)
        for name in ("seed", "episode_id", "attempt_id"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

    @property
    def enabled(self) -> bool:
        """Whether this context requests affordance expansion."""
        return self.count > 1

    def generator(self, key: str, *, group: int = 0) -> torch.Generator:
        """Create a private CPU generator for a stable named stream.

        Args:
            key: Invocation and parameter identity.
            group: Optional independent group of parallel branches.

        Returns:
            Generator independent of Python and Torch global random state.
        """
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
        """Stratify a legal scalar interval, retaining each group's nominal row.

        Args:
            nominal: Scalar or per-environment reference value.
            bounds: Explicit task-accepted interval; None fixes the value.
            env_ids: Stable int64 environment identifiers, shape (B,).
            key: Stable invocation and parameter identity.

        Returns:
            Values of shape (B,) on the environment device.
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
            raise ValueError("nominal must be finite.")
        if bounds is None:
            return values
        lower, upper = _validate_range(bounds, name=key)
        if not self.enabled:
            return values
        if ((values < lower) | (values > upper)).any():
            raise ValueError("nominal must lie inside the accepted sampling range.")
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
        """Return JSON-compatible provenance for an action plan."""
        return {
            name: getattr(self, name)
            for name in ("count", "seed", "episode_id", "attempt_id")
        }


@dataclass(frozen=True, slots=True)
class AffordancePoseCandidates:
    """Explicit candidate poses, costs and validity, including padded rows.

    Args:
        poses: Homogeneous poses with shape (B, K, 4, 4).
        costs: Lower-is-better scores, shape (B, K).
        valid: Boolean real-candidate mask, shape (B, K).
    """

    poses: torch.Tensor
    costs: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.poses.ndim != 4 or self.poses.shape[-2:] != (4, 4):
            raise ValueError("candidate poses must have shape (B, K, 4, 4).")
        shape = self.poses.shape[:2]
        if not all(shape) or self.costs.shape != shape or self.valid.shape != shape:
            raise ValueError(
                "candidate costs and validity must match non-empty (B, K)."
            )
        if self.valid.dtype != torch.bool:
            raise TypeError("candidate validity must be boolean.")
        if not self.poses.is_floating_point() or not self.costs.is_floating_point():
            raise TypeError("candidate poses and costs must be floating point.")
        if (
            self.costs.device != self.poses.device
            or self.valid.device != self.poses.device
        ):
            raise ValueError("candidate tensors must share a device.")

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        device: torch.device | str,
    ) -> AffordancePoseCandidates:
        """Pad variable candidate sets without promoting padding to valid poses.

        Args:
            rows: Per-environment (poses, costs) pairs, including empty rows.
            device: Output device.

        Returns:
            At least one identity placeholder per row, always masked if empty.
        """
        if not rows:
            raise ValueError("candidate rows must be non-empty.")
        width = max(1, max(poses.shape[0] for poses, _ in rows))
        poses = torch.eye(4, device=device).repeat(len(rows), width, 1, 1)
        costs = torch.full((len(rows), width), torch.inf, device=device)
        valid = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
        for row, (row_poses, row_costs) in enumerate(rows):
            count = row_poses.shape[0]
            if row_poses.shape != (count, 4, 4) or row_costs.shape != (count,):
                raise ValueError(
                    "candidate rows require (K, 4, 4) poses and (K,) costs."
                )
            poses[row, :count] = row_poses.to(device=device, dtype=torch.float32)
            costs[row, :count] = row_costs.to(device=device, dtype=torch.float32)
            valid[row, :count] = torch.isfinite(costs[row, :count]) & torch.isfinite(
                poses[row, :count]
            ).all(dim=(-1, -2))
        poses[~valid] = torch.eye(4, device=device)
        return cls(poses, costs, valid)

    def select(
        self,
        sampling: AffordanceSamplingContext | None,
        *,
        env_ids: torch.Tensor,
        key: str,
        reference_poses: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Select quality-ranked, geometrically distinct candidates by branch.

        Args:
            sampling: Expansion stream or None for minimum-cost selection.
            env_ids: Stable environment identifiers, one per candidate row.
            key: Invocation and sample identity.
            reference_poses: Optional object poses for local-frame diversity.

        Returns:
            Row success, selected poses and JSON-compatible selection metadata.
            Duplicate candidates are reused only after unique choices run out.
        """
        _validate_env_ids(env_ids)
        if env_ids.shape != self.poses.shape[:1] or env_ids.device != self.poses.device:
            raise ValueError("env_ids must match candidate batch and device.")
        if reference_poses is not None and reference_poses.shape != (
            len(env_ids),
            4,
            4,
        ):
            raise ValueError("reference_poses must have shape (B, 4, 4).")
        eligible = (
            self.valid
            & torch.isfinite(self.costs)
            & torch.isfinite(self.poses).all(dim=(-1, -2))
        )
        masked = self.costs.masked_fill(~eligible, torch.inf)
        selected = masked.argmin(dim=1)
        success = eligible.any(dim=1)
        unique_counts: list[int] = []
        reused: list[bool] = []
        for row, env_id in enumerate(env_ids.cpu().tolist()):
            indices = torch.argsort(masked[row], stable=True)
            indices = indices[eligible[row, indices]]
            if not len(indices):
                unique_counts.append(0)
                reused.append(False)
                continue
            if sampling is None or not sampling.enabled:
                unique_counts.append(len(indices))
                reused.append(False)
                continue
            # Restrict diversification to a bounded quality-ranked pool. The
            # generator/skill has already rejected physically invalid samples.
            indices = indices[: max(32, 4 * sampling.count)]
            geometry = self.poses[row, indices].clone()
            if reference_poses is not None:
                geometry = torch.linalg.inv(reference_poses[row]) @ geometry
            # Normalize translation by pool extent; rotations use their actual
            # matrices, avoiding quaternion-sign and Euler-wrap discontinuities.
            position = geometry[:, :3, 3]
            rotation = geometry[:, :3, :3].flatten(1)
            scale = torch.linalg.vector_norm(
                position.amax(0) - position.amin(0)
            ).clamp_min(0.01)
            generator = sampling.generator(key, group=env_id // sampling.count)
            jitter = torch.rand(len(indices), generator=generator).to(self.poses.device)
            order = [0]
            available = torch.ones(
                len(indices), dtype=torch.bool, device=self.poses.device
            )
            nearest = torch.full((len(indices),), torch.inf, device=self.poses.device)
            while True:
                last = order[-1]
                dp = torch.linalg.vector_norm(position - position[last], dim=1)
                dr = torch.linalg.vector_norm(rotation - rotation[last], dim=1)
                available &= ~((dp <= 1.0e-4) & (dr <= 1.0e-3))
                nearest = torch.minimum(nearest, dp / scale + dr)
                if not available.any():
                    break
                priority = nearest * (0.75 + 0.25 * jitter)
                order.append(int(priority.masked_fill(~available, -torch.inf).argmax()))
            branch = env_id % sampling.count
            selected[row] = indices[order[branch % len(order)]]
            unique_counts.append(len(order))
            reused.append(branch >= len(order))
        output = self.poses[
            torch.arange(len(env_ids), device=self.poses.device), selected
        ].clone()
        output[~success] = torch.eye(
            4, device=self.poses.device, dtype=self.poses.dtype
        )
        metadata = {
            "candidate_ids": torch.where(success, selected, -1).cpu().tolist(),
            "valid_candidate_counts": eligible.sum(1).cpu().tolist(),
            "unique_candidate_counts": unique_counts,
            "reused": reused,
        }
        return success, output, metadata


def _validate_env_ids(env_ids: torch.Tensor) -> None:
    if env_ids.dtype != torch.long or env_ids.ndim != 1 or env_ids.numel() == 0:
        raise ValueError("env_ids must be a non-empty int64 vector.")
    if (env_ids < 0).any() or torch.unique(env_ids).numel() != env_ids.numel():
        raise ValueError("env_ids must be non-negative and unique.")


def _validate_range(bounds: tuple[float, float], *, name: str) -> tuple[float, float]:
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{name} must contain lower and upper bounds.")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in bounds
    ):
        raise ValueError(f"{name} must contain finite real bounds.")
    lower, upper = map(float, bounds)
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed upper bound.")
    return lower, upper


def _roll_poses(poses: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate local x/y axes while fixing each contact position and z axis."""
    result = poses.clone()
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
    x, y = poses[:, :3, 0], poses[:, :3, 1]
    result[:, :3, 0] = cosine * x + sine * y
    result[:, :3, 1] = -sine * x + cosine * y
    return result
