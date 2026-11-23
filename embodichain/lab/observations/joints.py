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

"""Detached joint measurements in configured control-part order."""

from __future__ import annotations

from typing_extensions import TypedDict

__all__ = ["ControlPartJointState", "SessionJointState"]


class ControlPartJointState(TypedDict):
    """Measured scalar positions and velocities aligned with joint_names.

    All lists have one entry per selected joint, in configured order.
    Units describe position (rad or m); velocity uses the same unit per second.
    """

    control_part: str
    joint_names: list[str]
    units: list[str]
    qpos: list[float]
    qvel: list[float]


class SessionJointState(ControlPartJointState):
    """A control-part measurement with the native object, layout and clock.

    Each DOF index belongs to this scene topology. Clock fields are null
    when the backend does not expose a physical clock.
    """

    session_id: str
    object_ref: str
    topology_revision: int
    dof_ids: list[int]
    step_index: int | None
    simulation_time_s: float | None
