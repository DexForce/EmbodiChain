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

"""Read explicit driver-joint opening calibration without task semantics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import torch

__all__ = ["DriverOpeningBinding", "create_joint_opening_reader"]

_RELEASE_SCOPE = (
    "experimental_open_hand_without_bilateral_grip_not_contact_free_separation"
)


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact identifier.")


@dataclass(frozen=True, slots=True)
class DriverOpeningBinding:
    """Explicit joint-order calibration for the measured open-hand observation.

    The declaring robot/profile owns driver selection; this layer does not infer
    mimic joints or hand geometry from a model name or a control-part suffix.
    """

    control_part: str
    joint_names: tuple[str, ...]
    driver_indices: tuple[int, ...]
    open_qpos: tuple[float, ...]
    tolerance: float

    def __post_init__(self) -> None:
        _identifier(self.control_part, "control_part")
        if (
            type(self.joint_names) is not tuple
            or not self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
        ):
            raise ValueError("joint_names must be a nonempty unique tuple.")
        for name in self.joint_names:
            _identifier(name, "joint_names")
        if (
            type(self.driver_indices) is not tuple
            or not self.driver_indices
            or len(set(self.driver_indices)) != len(self.driver_indices)
            or any(
                type(i) is not int or not 0 <= i < len(self.joint_names)
                for i in self.driver_indices
            )
        ):
            raise ValueError("driver_indices must select unique declared joints.")
        if (
            type(self.open_qpos) is not tuple
            or len(self.open_qpos) != len(self.joint_names)
            or not all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                for value in self.open_qpos
            )
        ):
            raise ValueError("open_qpos must match the complete finite joint order.")
        if (
            isinstance(self.tolerance, bool)
            or not math.isfinite(self.tolerance)
            or self.tolerance <= 0
        ):
            raise ValueError("Opening tolerance must be finite and positive.")


def create_joint_opening_reader(
    robot: object, bindings: tuple[DriverOpeningBinding, ...]
) -> Callable[[str, int], Mapping[str, object]]:
    """Read declared drivers while requiring the complete hand row to be finite.

    Args:
        robot: Selected robot exposing joint names, control indices and qpos.
        bindings: Explicit calibration for each observed control part.

    Returns:
        Per-part/per-environment reader with the original strict open threshold.
    """
    declared = {item.control_part: item for item in bindings}
    if len(declared) != len(bindings):
        raise ValueError("Opening bindings require unique control parts.")
    for part, binding in declared.items():
        names = tuple(
            robot.joint_names[int(index)] for index in robot.get_joint_ids(name=part)
        )
        if names != binding.joint_names:
            raise ValueError(
                "Release opening requires the exact configured joint order."
            )

    def observe(part: str, env_id: int) -> Mapping[str, object]:
        binding = declared[part]
        qpos = robot.get_qpos(name=part)
        if (
            qpos.ndim != 2
            or qpos.shape[1] != len(binding.joint_names)
            or not 0 <= env_id < qpos.shape[0]
        ):
            raise ValueError(
                "Release joint observation has an invalid shape or environment."
            )
        indices = list(binding.driver_indices)
        measured = qpos[env_id, indices]
        target = measured.new_tensor(binding.open_qpos)[indices]
        finite = bool(torch.isfinite(qpos[env_id]).all())
        return {
            "finite": finite,
            "open": finite
            and bool(((measured - target).abs() < binding.tolerance).all()),
            "driving_joint_names": [binding.joint_names[i] for i in indices],
            "measured_qpos": measured.detach().cpu().tolist(),
            "open_qpos": target.detach().cpu().tolist(),
            "tolerance": binding.tolerance,
            "scope": _RELEASE_SCOPE,
        }

    return observe
