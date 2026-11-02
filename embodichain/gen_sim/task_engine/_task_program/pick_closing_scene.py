# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Read-only scene clearance during a source-qualified, fixed-arm closure."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

__all__: list[str] = []


@dataclass(frozen=True)
class _ClosingWorld:
    world: Any
    source_uid: str

    def get_rigid_object_uid_list(self) -> list[str]:
        return [
            uid
            for uid in self.world.get_rigid_object_uid_list()
            if uid != self.source_uid
        ]

    def get_rigid_object(self, uid: str) -> Any:
        return self.world.get_rigid_object(uid)

    def get_articulation_uid_list(self) -> list[str]:
        return self.world.get_articulation_uid_list()

    def get_articulation(self, uid: str) -> Any:
        return self.world.get_articulation(uid)


def check_closing_scene(
    robot: Any,
    simulation: Any,
    positions: torch.Tensor,
    control_part: str,
    arm_joint_ids: tuple[int, ...],
    source_uid: str,
    *,
    contact_fraction: float,
    grid_samples: int,
) -> dict[str, Any]:
    """Check the environment while native closing geometry owns source contact.

    The open-hand endpoint still includes the source. During closure its arm
    geometry cannot change, and the independently checked native closing sweep
    owns all hand/source pairs. Only nominal pre-contact FK is screened: after
    acquisition the object can stop actual closure before the command endpoint.
    This view never changes physical collisions.
    """
    from .articulation_collision import check_free_motion

    if positions.ndim != 2 or not len(positions) or not torch.isfinite(positions).all():
        raise ValueError("Closing scene samples must be finite nonempty (T, D).")
    arm = positions[:, list(arm_joint_ids)]
    if not torch.allclose(arm, arm[:1].expand_as(arm), atol=1e-7, rtol=0):
        raise ValueError("Closing-only source contact requires a fixed arm.")
    if source_uid not in simulation.get_rigid_object_uid_list():
        raise ValueError("Closing scene requires the bound native rigid source.")
    if (
        not math.isfinite(contact_fraction)
        or not 0 <= contact_fraction <= 1
        or grid_samples < 2
    ):
        raise ValueError("Closing scene requires the native first-contact grid.")
    count = round(contact_fraction * (grid_samples - 1)) + 1
    fractions = torch.linspace(
        0, contact_fraction, count, device=positions.device, dtype=positions.dtype
    )
    samples = torch.lerp(positions[:1], positions[-1:], fractions[:, None])
    return check_free_motion(
        robot, _ClosingWorld(simulation, source_uid), samples, control_part
    )
