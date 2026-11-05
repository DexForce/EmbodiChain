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


"""Read calibrated driver openings using the canonical Atomic binding type."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

__all__ = ["create_joint_opening_reader"]


class _OpeningBinding(Protocol):
    """Read-only structural input; the declaring owner retains the real binding."""

    @property
    def control_part(self) -> str: ...

    def bind(self, robot: object) -> Callable[[int], Mapping[str, object]]: ...


def create_joint_opening_reader(
    robot: object, bindings: tuple[_OpeningBinding, ...]
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
    readers = {part: binding.bind(robot) for part, binding in declared.items()}

    def observe(part: str, env_id: int) -> Mapping[str, object]:
        return readers[part](env_id)

    return observe
