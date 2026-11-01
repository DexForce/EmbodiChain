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
"""Closed configuration for bounded joint scene/deployment search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import math
import torch

from embodichain.utils import configclass
from ._values import count

__all__ = ["SceneRobotFeasibilityCfg"]


@configclass
class SceneRobotFeasibilityCfg:
    """Search, branch, region and independent validation budgets.

    All calls share ``max_wall_time_s`` including preparation. Hosts must honor
    the supplied deadline at their planning/control boundaries. Numerical batch
    size does not enable batched physical execution. Region widths are fractions
    of the declared parameter ranges; fixed parameters keep zero width.
    """

    seed: int = 0
    max_wall_time_s: float = 600.0
    search_fraction: float = 0.6
    max_candidates: int = 4096
    batch_size: int = 256
    global_fraction: float = 0.2
    local_radius: float = 0.1
    min_region_width: float = 0.01
    archive_bins: int = 8
    archive_per_bin: int = 2
    max_regions: int = 4
    validation_samples: int = 8
    max_ik_calls: int = 16384
    max_planning_calls: int = 1024
    max_rollouts: int = 128
    recording_max_bytes: int = 134217728
    beam_width: int = 8
    ik_seeds: int = 4
    joint_resolution_rad: float = 0.05
    max_motion_samples: int = 4096
    planning_timeout_s: float = 5.0
    max_rrt_nodes: int = 256
    device: str = "cpu"

    def __post_init__(self) -> None:
        count(self.seed, "seed", 0)
        if self.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        for name in (
            "max_candidates",
            "batch_size",
            "archive_bins",
            "archive_per_bin",
            "max_regions",
            "validation_samples",
            "max_ik_calls",
            "max_planning_calls",
            "max_rollouts",
            "recording_max_bytes",
            "beam_width",
            "ik_seeds",
            "max_motion_samples",
            "max_rrt_nodes",
        ):
            count(getattr(self, name), name)
        for name in ("max_wall_time_s", "joint_resolution_rad", "planning_timeout_s"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "search_fraction",
            "local_radius",
            "min_region_width",
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 < value < 1
            ):
                raise ValueError(f"{name} must lie strictly between zero and one")
        if (
            type(self.global_fraction) not in (int, float)
            or not math.isfinite(self.global_fraction)
            or not 0 < self.global_fraction <= 1
        ):
            raise ValueError("global_fraction must lie in (0,1]")
        if not isinstance(self.device, str) or not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must name cpu or a CUDA device")
        try:
            target = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError("device must be a valid Torch CPU/CUDA device") from error
        if target.type not in ("cpu", "cuda"):
            raise ValueError("device must name cpu or a CUDA device")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> SceneRobotFeasibilityCfg:
        """Decode settings without importing or executing configuration values.

        Args:
            values: Closed field mapping.
        Returns:
            Validated configuration.
        """
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown feasibility settings: {sorted(unknown)}")
        return cls(**values)
