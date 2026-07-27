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

"""Structural protocols shared by deterministic builders and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

__all__ = [
    "RelativePlacementLike",
    "RelativeSpecLike",
    "ArrangementStepLike",
    "ArrangementSpecLike",
    "StackingStepLike",
    "StackingSpecLike",
    "_RelativePlacementLike",
    "_RelativeSpecLike",
    "_ArrangementStepLike",
    "_ArrangementSpecLike",
    "_StackingStepLike",
    "_StackingSpecLike",
]


class RelativePlacementLike(Protocol):
    intent: str
    active_side: str
    moved_runtime_uid: str
    moved_source_uid: str
    reference_runtime_uid: str
    reference_source_uid: str
    relation: str
    high_offset: tuple[float, float, float]
    release_offset: tuple[float, float, float]
    reference_is_initial_pose: bool
    high_position: Sequence[float] | None
    release_position: Sequence[float] | None
    orientation_goal: str
    orientation_axis: str
    orientation_align_to_runtime_uid: str | None
    hover_height: float
    upright_in_place: bool
    pickup_upright_direction: Sequence[float] | None
    pickup_rotate_upright: float | None
    surface_clearance: float


class RelativeSpecLike(RelativePlacementLike, Protocol):
    placements: Sequence[RelativePlacementLike]
    task_prompt_summary: str
    task_description: str
    action_sketch: Sequence[str]
    basic_background_notes: str
    coordinated_direction: str | None
    coordinated_terminal_behavior: str | None


class ArrangementStepLike(Protocol):
    source_uid: str
    runtime_uid: str
    slot_index: int
    active_side: str
    target_xy: Sequence[float]
    release_position: Sequence[float]
    high_position: Sequence[float]
    size_score: float | None
    color: str | None
    orientation_goal: str
    orientation_axis: str


class ArrangementSpecLike(Protocol):
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    order_by: str
    order_direction: str
    axis: str
    anchor: str
    line_origin_xy: Sequence[float]
    spacing: float
    layout_clearance: float
    steps: Sequence[ArrangementStepLike]


class StackingStepLike(Protocol):
    source_uid: str
    runtime_uid: str
    layer_index: int
    active_side: str
    target_position: Sequence[float]
    high_position: Sequence[float]
    support_runtime_uid: str | None
    size_score: float | None
    color: str | None
    orientation_goal: str
    orientation_axis: str


class StackingSpecLike(Protocol):
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    stack_mode: str
    order_by: str
    anchor: str
    anchor_xy: Sequence[float]
    anchor_source_uid: str | None
    anchor_runtime_uid: str | None
    steps: Sequence[StackingStepLike]


# Keep the original internal spellings as identity aliases for downstream
# modules that imported these protocols before they became public contracts.
_RelativePlacementLike = RelativePlacementLike
_RelativeSpecLike = RelativeSpecLike
_ArrangementStepLike = ArrangementStepLike
_ArrangementSpecLike = ArrangementSpecLike
_StackingStepLike = StackingStepLike
_StackingSpecLike = StackingSpecLike
