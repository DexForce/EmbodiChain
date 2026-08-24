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

"""Stable public contracts for Action Engine programs."""

from __future__ import annotations

from .motion import (
    MOTION_MODIFIER_MODES,
    MOTION_POLICY_VERSION,
    motion_policy,
    validate_motion_policy,
)
from .programs import (
    EXECUTION_PROGRAM_SCHEMA,
    TASK_AGENT_SCHEMA,
    execution_program_hash,
    validate_execution_program,
    validate_task_agent,
)
from .task_contracts import (
    PLACEMENT_RELATIONS,
    RELATIONS,
    TASK_CONTRACTS,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    TaskContract,
    normalize_placement_relation,
    task_contract,
    task_success_type,
)
from .v2 import (
    REASONING_TYPES,
    TASK_LEVELS,
    TASK_TYPES,
    public_task_spec,
    seed_graph_hash,
    validate_public_task_spec,
    validate_scene_requirements,
    validate_seed_graph,
    validate_task_spec,
)
from .visual_contracts import (
    OCCLUSION_RELATION,
    VISUAL_RELATION_PARTICIPANTS,
    requested_visual_task_predicates,
)

__all__ = [
    "EXECUTION_PROGRAM_SCHEMA",
    "MOTION_POLICY_VERSION",
    "MOTION_MODIFIER_MODES",
    "OCCLUSION_RELATION",
    "REASONING_TYPES",
    "RELATIONS",
    "PLACEMENT_RELATIONS",
    "TASK_CONTRACTS",
    "TASK_LEVELS",
    "TASK_TYPES",
    "TASK_AGENT_SCHEMA",
    "TERMINAL_BEHAVIORS",
    "TRANSPORT_DIRECTIONS",
    "VISUAL_RELATION_PARTICIPANTS",
    "TaskContract",
    "execution_program_hash",
    "motion_policy",
    "normalize_placement_relation",
    "public_task_spec",
    "requested_visual_task_predicates",
    "seed_graph_hash",
    "task_contract",
    "task_success_type",
    "validate_public_task_spec",
    "validate_scene_requirements",
    "validate_seed_graph",
    "validate_task_spec",
    "validate_execution_program",
    "validate_motion_policy",
    "validate_task_agent",
]
