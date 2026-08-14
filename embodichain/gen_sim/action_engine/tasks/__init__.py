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

"""Task-first generation and scene hand-off for Action Engine v2."""

from __future__ import annotations

from .factory import BatchGenerationResult, TaskFactory, task_capability_catalog
from .interpretation import (
    GroundingCaller,
    INSTRUCTION_INTENT_SCHEMA,
    InstructionDraftResult,
    InstructionCaller,
    InstructionIntent,
    ground_instruction_draft,
    interpret_instruction_draft,
    interpret_and_ground_task_spec,
    validate_instruction_intent,
)
from .planning import GroundedTaskSpec, plan_grounded_task_spec
from .recipes import instantiate_seed_graph
from .scene import SceneHandoff, validate_scene_handoff

__all__ = [
    "BatchGenerationResult",
    "GroundedTaskSpec",
    "GroundingCaller",
    "INSTRUCTION_INTENT_SCHEMA",
    "InstructionDraftResult",
    "InstructionCaller",
    "InstructionIntent",
    "SceneHandoff",
    "TaskFactory",
    "ground_instruction_draft",
    "instantiate_seed_graph",
    "interpret_instruction_draft",
    "interpret_and_ground_task_spec",
    "plan_grounded_task_spec",
    "task_capability_catalog",
    "validate_instruction_intent",
    "validate_scene_handoff",
]
