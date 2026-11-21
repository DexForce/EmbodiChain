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

"""Detached measurements of robot control parts."""

from __future__ import annotations

from typing_extensions import TypedDict

__all__ = ["RobotState"]


class RobotState(TypedDict):
    """One detached measurement ordered by the configured control part."""

    session_id: str
    object_ref: str
    topology_revision: int
    control_part: str
    joint_names: list[str]
    dof_ids: list[int]
    qpos: list[float]
    units: list[str]
    step_index: int | None
    simulation_time_s: float | None
