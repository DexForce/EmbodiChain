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

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
)

__all__ = [
    "ArrangementLineSpec",
    "ArrangementLineStepSpec",
    "RelativePlacementSpec",
    "RelativePlacementStepSpec",
    "SceneObject",
    "StackingSpec",
    "StackingStepSpec",
    "_ArrangementLineSpec",
    "_ArrangementLineStepSpec",
    "_StackingSpec",
    "_StackingStepSpec",
    "GeneratedActionAgentConfigPaths",
    "_RelativePlacementSpec",
    "_RelativePlacementStepSpec",
    "_SceneObject",
]


@dataclass(frozen=True)
class GeneratedActionAgentConfigPaths:
    """Paths written by the action-agent config generator.

    ``gym_config``, ``agent_config``, and ``task_graph`` are runtime inputs.
    ``task_prompt``, ``basic_background``, and ``atom_actions`` are diagnostic
    records retained for human review and compatibility with existing tooling.
    """

    output_dir: Path
    gym_config: Path
    agent_config: Path
    task_prompt: Path
    task_graph: Path
    basic_background: Path
    atom_actions: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class SceneObject:
    """One source scene entity used during deterministic config generation."""

    source_uid: str
    source_role: str
    config: dict[str, Any]


@dataclass(frozen=True)
class RelativePlacementStepSpec:
    """One normalized object placement in a relative-manipulation task."""

    intent: str
    moved_source_uid: str
    reference_source_uid: str
    moved_runtime_uid: str
    reference_runtime_uid: str
    relation: str
    active_side: str
    release_offset: list[float]
    high_offset: list[float]
    arm_request: str = "auto"
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
    surface_clearance: float = DEFAULT_SURFACE_RELEASE_CLEARANCE


@dataclass(frozen=True)
class RelativePlacementSpec:
    """Normalized semantic and geometric plan for relative manipulation."""

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
    placements: tuple[RelativePlacementStepSpec, ...]
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
    surface_clearance: float = DEFAULT_SURFACE_RELEASE_CLEARANCE
    coordinated_direction: str | None = None
    coordinated_terminal_behavior: str | None = None


@dataclass(frozen=True)
class ArrangementLineStepSpec:
    """One object's resolved slot and execution metadata in a line layout."""

    source_uid: str
    runtime_uid: str
    slot_index: int
    active_side: str
    target_xy: list[float]
    release_position: list[float]
    high_position: list[float]
    size_score: float | None = None
    color: str | None = None
    orientation_goal: str = "preserve"
    orientation_axis: str = "none"
    category: str = "object"
    cross_side: bool = False
    execution_index: int = 0
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArrangementLineSpec:
    """Resolved multi-object line-arrangement plan."""

    table_source_uid: str
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    order_by: str
    order_direction: str
    axis: str
    anchor: str
    steps: tuple[ArrangementLineStepSpec, ...]
    line_origin_xy: list[float]
    spacing: float
    layout_clearance: float
    category_order: tuple[str, ...] = ()
    spatial_direction: str = "ascending"


@dataclass(frozen=True)
class StackingStepSpec:
    """One resolved layer in a deterministic stacking plan."""

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
class StackingSpec:
    """Resolved bottom-to-top stacking plan."""

    table_source_uid: str
    task_description: str
    task_prompt_summary: str
    basic_background_notes: str
    stack_mode: str
    order_by: str
    anchor: str
    anchor_xy: list[float]
    steps: tuple[StackingStepSpec, ...]
    anchor_source_uid: str | None = None
    anchor_runtime_uid: str | None = None


# Historical private names remain aliases while package-internal consumers
# migrate. Alias identity matters because callers use these classes in
# ``dataclasses.replace`` and occasional ``isinstance`` checks.
_SceneObject = SceneObject
_RelativePlacementStepSpec = RelativePlacementStepSpec
_RelativePlacementSpec = RelativePlacementSpec
_ArrangementLineStepSpec = ArrangementLineStepSpec
_ArrangementLineSpec = ArrangementLineSpec
_StackingStepSpec = StackingStepSpec
_StackingSpec = StackingSpec
