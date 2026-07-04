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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "_ArrangementLineSpec",
    "_ArrangementLineStepSpec",
    "_ObjectManipulationSpec",
    "_ObjectManipulationStepSpec",
    "_StackingSpec",
    "_StackingStepSpec",
    "GeneratedActionAgentConfigPaths",
    "TargetReplacementSpec",
    "_BasketTaskRoles",
    "_RelativePlacementSpec",
    "_RelativePlacementStepSpec",
    "_ResolvedTargetReplacement",
    "_SceneObject",
]


@dataclass(frozen=True)
class GeneratedActionAgentConfigPaths:
    """Paths written by the action-agent config generator."""

    output_dir: Path
    gym_config: Path
    agent_config: Path
    task_prompt: Path
    basic_background: Path
    atom_actions: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class TargetReplacementSpec:
    """Prompt-to-geometry replacement for one source target object."""

    source_uid: str
    prompt: str
    output_dir_name: str


@dataclass(frozen=True)
class _SceneObject:
    source_uid: str
    source_role: str
    config: dict[str, Any]


@dataclass(frozen=True)
class _BasketTaskRoles:
    table_source_uid: str
    container_source_uid: str
    left_target_source_uid: str
    right_target_source_uid: str
    container_runtime_uid: str
    left_target_runtime_uid: str
    right_target_runtime_uid: str
    target_noun: str
    left_target_noun: str
    right_target_noun: str
    container_noun: str


@dataclass(frozen=True)
class _ResolvedTargetReplacement:
    source_uid: str
    prompt: str
    output_dir_name: str
    mesh_path: Path
    runtime_noun: str
    reused: bool = False


@dataclass(frozen=True)
class _RelativePlacementStepSpec:
    intent: str
    moved_source_uid: str
    reference_source_uid: str
    moved_runtime_uid: str
    reference_runtime_uid: str
    relation: str
    active_side: str
    release_offset: list[float]
    high_offset: list[float]
    reference_is_initial_pose: bool = False
    release_position: list[float] | None = None
    high_position: list[float] | None = None
    orientation_goal: str = "preserve"
    orientation_axis: str = "none"
    orientation_align_to_runtime_uid: str | None = None
    hover_height: float = 0.10
    upright_in_place: bool = False
    pickup_upright_direction: list[float] | None = None
    pickup_rotate_upright: float | None = None


@dataclass(frozen=True)
class _RelativePlacementSpec:
    intent: str
    table_source_uid: str
    moved_source_uid: str
    reference_source_uid: str
    moved_runtime_uid: str
    reference_runtime_uid: str
    relation: str
    active_side: str
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    action_sketch: list[str]
    release_offset: list[float]
    high_offset: list[float]
    placements: tuple[_RelativePlacementStepSpec, ...]
    reference_is_initial_pose: bool = False
    release_position: list[float] | None = None
    high_position: list[float] | None = None
    orientation_goal: str = "preserve"
    orientation_axis: str = "none"
    orientation_align_to_runtime_uid: str | None = None
    hover_height: float = 0.10
    upright_in_place: bool = False
    pickup_upright_direction: list[float] | None = None
    pickup_rotate_upright: float | None = None


_ObjectManipulationStepSpec = _RelativePlacementStepSpec
_ObjectManipulationSpec = _RelativePlacementSpec


@dataclass(frozen=True)
class _ArrangementLineStepSpec:
    source_uid: str
    runtime_uid: str
    slot_index: int
    active_side: str
    target_xy: list[float]
    release_position: list[float]
    high_position: list[float]
    size_score: float | None = None
    color: str | None = None
    orientation_goal: str = "axis_align"
    orientation_axis: str = "y"


@dataclass(frozen=True)
class _ArrangementLineSpec:
    table_source_uid: str
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    order_by: str
    order_direction: str
    axis: str
    anchor: str
    steps: tuple[_ArrangementLineStepSpec, ...]
    line_origin_xy: list[float]
    spacing: float
    layout_clearance: float


@dataclass(frozen=True)
class _StackingStepSpec:
    source_uid: str
    runtime_uid: str
    layer_index: int
    active_side: str
    target_position: list[float]
    high_position: list[float]
    support_runtime_uid: str | None = None
    size_score: float | None = None
    color: str | None = None
    orientation_goal: str = "preserve"
    orientation_axis: str = "none"


@dataclass(frozen=True)
class _StackingSpec:
    table_source_uid: str
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    stack_mode: str
    order_by: str
    anchor: str
    anchor_xy: list[float]
    steps: tuple[_StackingStepSpec, ...]
