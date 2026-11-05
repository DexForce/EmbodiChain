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

"""Owned numeric Pour projection operands with optional source trace identity."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from numbers import Real
import numpy as np
import torch
from embodichain.compute.geometry.pouring import PourGraspProjection

__all__ = ["PourGraspProjectionInputs"]


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class PourGraspProjectionInputs:
    """Retain raw operand dtypes and exact occurrence identities for Pick audit.

    Args:
        pick_id: Current Pick occurrence identity.
        stage_id: Declared following transport occurrence.
        pour_id: Declared Pour occurrence.
        object_id: Exact source rigid identity.
        receiver_id: Exact receiving rigid identity.
        resource: Declared shared resource identity.
        source_vertices: Source-local mesh points, preserving acquisition dtype.
        receiver_vertices: Receiver-local mesh points, preserving acquisition dtype.
        staging: Resolved world staging transform.
        receiver_pose: Same-observation receiver transform.
        return_pose: Original float tensor return target, shape ``(4,4)``.
        pad_line_tcp: Ordered jaw vector in TCP coordinates, shape ``(3,)``.
        gravity_world: Actual gravity vector used by the projection.
        clearance_m: Original authored staging clearance, retained for trace.
        angle_rad: Original future Pour angle.
        sample_count: Original future Pour sample budget.
        proxy_position_tolerance_m: Optional declared region budget for trace.

    This value contains no compiler, program or future-state callbacks. Compute's
    numeric projection remains the calculation owner; these owned raw values
    retain the bounded source IK diagnostic record without changing its math.
    """

    pick_id: str
    stage_id: str
    pour_id: str
    object_id: str
    receiver_id: str
    resource: str
    source_vertices: np.ndarray
    receiver_vertices: np.ndarray
    staging: np.ndarray
    receiver_pose: np.ndarray
    return_pose: torch.Tensor
    pad_line_tcp: torch.Tensor
    gravity_world: tuple[float, float, float]
    clearance_m: float
    angle_rad: float
    sample_count: int
    proxy_position_tolerance_m: float | None = None
    projection: PourGraspProjection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "pick_id",
            "stage_id",
            "pour_id",
            "object_id",
            "receiver_id",
            "resource",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact nonempty identity.")
        for name in ("clearance_m", "proxy_position_tolerance_m"):
            value = getattr(self, name)
            if value is None and name == "proxy_position_tolerance_m":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative.")
        if (
            not isinstance(self.return_pose, torch.Tensor)
            or self.return_pose.shape != (4, 4)
            or not self.return_pose.is_floating_point()
        ):
            raise ValueError("Pour trace requires the original tensor return pose.")
        for name in (
            "source_vertices",
            "receiver_vertices",
            "staging",
            "receiver_pose",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "return_pose", self.return_pose.detach().cpu().clone())
        if not isinstance(self.pad_line_tcp, torch.Tensor):
            raise ValueError("Pour trace requires the original tensor jaw vector.")
        object.__setattr__(
            self, "pad_line_tcp", self.pad_line_tcp.detach().cpu().clone()
        )
        object.__setattr__(self, "gravity_world", tuple(self.gravity_world))
        object.__setattr__(
            self,
            "projection",
            PourGraspProjection(
                source_vertices=self.source_vertices,
                receiver_vertices=self.receiver_vertices,
                staging=self.staging,
                receiver_pose=self.receiver_pose,
                return_pose=self.return_pose.numpy(),
                pad_line_tcp=self.pad_line_tcp,
                gravity_world=self.gravity_world,
                angle_rad=self.angle_rad,
                sample_count=self.sample_count,
            ),
        )

    def recipe(self) -> dict[str, str | torch.Tensor | float | int | None]:
        """Return source recipe operands separately from numeric projection data.

        Returns:
            Owned declarative trace fields; no evidence or planning callbacks.
        """
        return {
            name: (
                getattr(self, name).clone()
                if name == "return_pose"
                else getattr(self, name)
            )
            for name in (
                "pick_id",
                "stage_id",
                "pour_id",
                "object_id",
                "receiver_id",
                "resource",
                "return_pose",
                "clearance_m",
                "angle_rad",
                "sample_count",
                "proxy_position_tolerance_m",
            )
        }
