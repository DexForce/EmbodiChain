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


"""Neutral declared release target shared by source and standard grasp services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
import numpy as np
import torch
from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    NativeQueryScope,
    _rigid_pose,
    _neutral_identity,
)

__all__ = ["GraspOpeningTarget"]


@dataclass(frozen=True, slots=True, eq=False)
class GraspOpeningTarget:
    """Already-grounded support and object poses from the closing observation.

    Args:
        support_uid: Exact native rigid support UID selected by the caller.
        reference_pose: Support-to-world pose, shape (4, 4), copied to CPU float64.
        object_target_pose: Final placed-object-to-world pose, shape (4, 4), copied
            to CPU float64. Tensor inputs are detached without changing the input.
        scope: Must equal the closing query's fresh preparation scope.
        identity: Optional neutral diagnostic IDs; never runtime or program objects.
    """

    support_uid: str
    reference_pose: np.ndarray | torch.Tensor
    object_target_pose: np.ndarray | torch.Tensor
    scope: NativeQueryScope
    identity: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            type(self.support_uid) is not str
            or not self.support_uid
            or self.support_uid != self.support_uid.strip()
        ):
            raise ValueError("Opening requires an explicit native support UID.")
        if not isinstance(self.scope, NativeQueryScope):
            raise TypeError("Opening scope must be NativeQueryScope.")
        for name in ("reference_pose", "object_target_pose"):
            pose = _rigid_pose(getattr(self, name))
            pose.setflags(write=False)
            object.__setattr__(self, name, pose)
        object.__setattr__(
            self, "identity", MappingProxyType(_neutral_identity(self.identity))
        )
