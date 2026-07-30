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

"""Runtime loading and sampling of cached robot workspaces."""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

__all__ = ["RobotWorkspace", "WorkspaceSample"]


@dataclass
class WorkspaceSample:
    """A batch of cached, reachable robot configurations and FK poses."""

    eef_pose: torch.Tensor
    """End-effector poses in the local arena frame, shape ``(B, K, 4, 4)``."""

    qpos: torch.Tensor
    """Control-part joint configurations, shape ``(B, K, D)``."""

    indices: torch.Tensor
    """Workspace cache indices, shape ``(B, K)``; invalid entries are ``-1``."""

    valid: torch.Tensor
    """Whether each returned sample satisfies the runtime filters, shape ``(B, K)``."""

    score: torch.Tensor | None = None
    """Optional cached reachability score, shape ``(B, K)``."""


class RobotWorkspace:
    """Reachable Cartesian samples backed by aligned joint configurations.

    The cached Cartesian points are used to define the sampling distribution.
    Runtime callers should recompute end-effector poses from :attr:`qpos` so the
    result uses the robot base pose of the target environment.
    """

    SUPPORTED_STRATEGIES = ("point_uniform", "voxel_uniform")

    def __init__(
        self,
        positions: torch.Tensor,
        qpos: torch.Tensor,
        *,
        scores: torch.Tensor | None = None,
        voxel_size: float = 0.03,
        metadata: dict | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        """Initialize a runtime workspace.

        Args:
            positions: Cached Cartesian positions, shape ``(N, 3)``.
            qpos: Joint configurations aligned with ``positions``, shape
                ``(N, D)``.
            scores: Optional score aligned with ``positions``, shape ``(N,)``.
            voxel_size: Cartesian voxel edge length in meters.
            metadata: Optional cache metadata.
            source_path: Optional source cache path.

        Raises:
            ValueError: If tensors are empty, have incompatible shapes, or
                ``voxel_size`` is not positive.
        """
        positions = torch.as_tensor(positions, dtype=torch.float32)
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=positions.device)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must have shape (N, 3); got {tuple(positions.shape)}."
            )
        if qpos.ndim != 2:
            raise ValueError(f"qpos must have shape (N, D); got {tuple(qpos.shape)}.")
        if len(positions) == 0:
            raise ValueError("Workspace cache contains no reachable samples.")
        if len(positions) != len(qpos):
            raise ValueError(
                "Workspace positions and joint configurations must be aligned; "
                f"got {len(positions)} positions and {len(qpos)} configurations."
            )
        if voxel_size <= 0:
            raise ValueError(f"voxel_size must be positive; got {voxel_size}.")

        if scores is not None:
            scores = torch.as_tensor(
                scores, dtype=torch.float32, device=positions.device
            ).reshape(-1)
            if len(scores) != len(positions):
                raise ValueError(
                    "Workspace scores and positions must be aligned; "
                    f"got {len(scores)} scores and {len(positions)} positions."
                )

        self.positions = positions
        self.qpos = qpos
        self.scores = scores
        self.voxel_size = float(voxel_size)
        self.metadata = metadata or {}
        self.source_path = Path(source_path) if source_path is not None else None
        self._voxel_index_cache: dict[
            float | None, tuple[torch.Tensor, torch.Tensor, int]
        ] = {}

    @property
    def device(self) -> torch.device:
        """Return the device holding workspace tensors."""
        return self.positions.device

    @property
    def num_samples(self) -> int:
        """Return the number of cached reachable samples."""
        return len(self.positions)

    def to(self, device: torch.device | str) -> RobotWorkspace:
        """Move workspace tensors to a device in-place.

        Args:
            device: Target torch device.

        Returns:
            This workspace instance.
        """
        self.positions = self.positions.to(device)
        self.qpos = self.qpos.to(device)
        if self.scores is not None:
            self.scores = self.scores.to(device)
        self._voxel_index_cache.clear()
        return self

    @classmethod
    def from_cache(
        cls,
        cache_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        voxel_size: float = 0.03,
    ) -> RobotWorkspace:
        """Load an analyzer results cache for runtime sampling.

        Args:
            cache_path: Cache entry directory or direct ``results.npz`` path.
            device: Device on which runtime tensors are stored.
            voxel_size: Cartesian voxel edge length in meters.

        Returns:
            Loaded runtime workspace.

        Raises:
            FileNotFoundError: If the cache archive does not exist.
            ValueError: If no point set aligns with ``joint_configurations``.
        """
        source = Path(cache_path).expanduser()
        npz_path = source / "results.npz" if source.is_dir() else source
        if not npz_path.is_file():
            raise FileNotFoundError(f"Workspace cache archive not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files}

        if "joint_configurations" not in arrays:
            raise ValueError(
                f"Workspace cache {npz_path} has no joint_configurations array."
            )
        qpos = arrays["joint_configurations"]

        positions = None
        for field in ("reachable_points", "workspace_points"):
            candidate = arrays.get(field)
            if candidate is not None and len(candidate) == len(qpos):
                positions = candidate
                break
        if positions is None:
            raise ValueError(
                "Workspace cache has no Cartesian point array aligned with "
                f"{len(qpos)} joint configurations."
            )

        scores = arrays.get("success_rates")
        if scores is not None and len(scores) != len(positions):
            mask = arrays.get("reachability_mask")
            if (
                mask is not None
                and len(mask) == len(scores)
                and int(np.asarray(mask, dtype=bool).sum()) == len(positions)
            ):
                scores = scores[np.asarray(mask, dtype=bool)]
            else:
                scores = None

        metadata: dict = {}
        meta_path = npz_path.with_name("meta.json")
        if meta_path.is_file():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}

        return cls(
            torch.as_tensor(positions, device=device),
            torch.as_tensor(qpos, device=device),
            scores=(
                torch.as_tensor(scores, device=device) if scores is not None else None
            ),
            voxel_size=voxel_size,
            metadata=metadata,
            source_path=npz_path,
        )

    def sample_indices(
        self,
        count: int,
        *,
        strategy: Literal["point_uniform", "voxel_uniform"] = "voxel_uniform",
        min_score: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample cache indices.

        Args:
            count: Number of indices to return.
            strategy: Point-uniform or Cartesian-voxel-uniform sampling.
            min_score: Optional minimum cached score.
            generator: Optional random number generator.

        Returns:
            Index tensor with shape ``(count,)``.

        Raises:
            ValueError: If arguments are invalid or filters reject every point.
        """
        if count <= 0:
            raise ValueError(f"count must be positive; got {count}.")
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported workspace strategy {strategy!r}; "
                f"expected one of {self.SUPPORTED_STRATEGIES}."
            )

        candidates = torch.arange(self.num_samples, device=self.device)
        if min_score is not None:
            if self.scores is None:
                raise ValueError(
                    "min_score was requested but this workspace cache has no scores."
                )
            candidates = candidates[self.scores >= min_score]
        if len(candidates) == 0:
            raise ValueError("No workspace samples satisfy the score filter.")

        if strategy == "point_uniform":
            selected = torch.randint(
                len(candidates),
                (count,),
                device=self.device,
                generator=generator,
            )
            return candidates[selected]

        cache_key = float(min_score) if min_score is not None else None
        voxel_index = self._voxel_index_cache.get(cache_key)
        if voxel_index is None:
            voxel_coords = torch.floor(self.positions[candidates] / self.voxel_size).to(
                torch.int64
            )
            _, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
            num_voxels = int(inverse.max().item()) + 1
            voxel_index = (candidates, inverse, num_voxels)
            self._voxel_index_cache[cache_key] = voxel_index
        candidates, inverse, num_voxels = voxel_index
        selected_voxels = torch.randint(
            num_voxels,
            (count,),
            device=self.device,
            generator=generator,
        )

        sampled = torch.empty(count, dtype=torch.long, device=self.device)
        for voxel_id in torch.unique(selected_voxels):
            output_mask = selected_voxels == voxel_id
            members = candidates[inverse == voxel_id]
            member_indices = torch.randint(
                len(members),
                (int(output_mask.sum().item()),),
                device=self.device,
                generator=generator,
            )
            sampled[output_mask] = members[member_indices]
        return sampled
