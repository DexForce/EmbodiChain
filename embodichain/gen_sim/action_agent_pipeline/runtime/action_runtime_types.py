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

"""Typed records shared across runtime orchestration modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import WorldState

__all__ = [
    "ExecutedAtomicAction",
    "CoordinatedPayloadRuntimeState",
    "CoordinatedGraspPair",
    "_ExecutedAtomicAction",
    "_CoordinatedPayloadRuntimeState",
    "_CoordinatedGraspPair",
]


@dataclass(frozen=True)
class ExecutedAtomicAction:
    action: np.ndarray
    next_state: WorldState | None
    robot_name: str | None
    control: str | None
    failed_env_mask: torch.Tensor | None = None
    atomic_action_class: str | None = None
    resolved_target_pose: torch.Tensor | None = None
    resolved_object_target_pose: torch.Tensor | None = None
    resolved_eef_target_pose: torch.Tensor | None = None
    resolved_left_eef_target_pose: torch.Tensor | None = None
    resolved_right_eef_target_pose: torch.Tensor | None = None


@dataclass(frozen=True)
class CoordinatedPayloadRuntimeState:
    carrier_uid: str
    payload_uids: tuple[str, ...]
    initial_carrier_pose: torch.Tensor
    carrier_to_payload: tuple[torch.Tensor, ...]
    support_half_extents: tuple[float, float]
    max_payload_drift: float = 0.04
    max_carrier_tilt: float = float(np.deg2rad(10.0))


@dataclass(frozen=True)
class CoordinatedGraspPair:
    left_object_to_eef: torch.Tensor
    right_object_to_eef: torch.Tensor
    priority: int
    score: float
    axis_kind: str


# Retain identity aliases so imports written before these DTOs became public
# keep their isinstance and pickle behavior.
_ExecutedAtomicAction = ExecutedAtomicAction
_CoordinatedPayloadRuntimeState = CoordinatedPayloadRuntimeState
_CoordinatedGraspPair = CoordinatedGraspPair
