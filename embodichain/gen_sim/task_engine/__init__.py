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

"""Scene-independent task interpretation and protocol ownership."""

from __future__ import annotations

from .agent import (
    TaskAgent,
    TaskGenerationError,
    derive_scene_request,
    derive_success_spec,
)
from .contracts import (
    SCENE_REQUEST_SCHEMA,
    SUCCESS_SPEC_SCHEMA,
    TASK_CANDIDATE_SET_SCHEMA,
    TASK_DRAFT_SCHEMA,
    SceneRequest,
    SuccessSpec,
    TaskCandidate,
    TaskCandidateSet,
    TaskDraft,
    canonical_hash,
    validate_scene_request,
    validate_success_spec,
    validate_task_candidate,
    validate_task_candidate_set,
    validate_task_draft,
)
from .interpretation import (
    INSTRUCTION_INTENT_SCHEMA,
    InstructionCaller,
    InstructionDraftResult,
    InstructionIntent,
    interpret_instruction_draft,
    validate_instruction_intent,
)
from .ontology import (
    RELATIONS,
    TASK_CONTRACTS,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    TaskContract,
    task_contract,
    task_success_type,
)

__all__ = [
    "INSTRUCTION_INTENT_SCHEMA",
    "InstructionCaller",
    "InstructionDraftResult",
    "InstructionIntent",
    "RELATIONS",
    "SCENE_REQUEST_SCHEMA",
    "SUCCESS_SPEC_SCHEMA",
    "SceneRequest",
    "SuccessSpec",
    "TASK_CANDIDATE_SET_SCHEMA",
    "TASK_CONTRACTS",
    "TASK_DRAFT_SCHEMA",
    "TERMINAL_BEHAVIORS",
    "TRANSPORT_DIRECTIONS",
    "TaskAgent",
    "TaskCandidate",
    "TaskCandidateSet",
    "TaskContract",
    "TaskDraft",
    "TaskGenerationError",
    "canonical_hash",
    "derive_scene_request",
    "derive_success_spec",
    "interpret_instruction_draft",
    "task_contract",
    "task_success_type",
    "validate_instruction_intent",
    "validate_scene_request",
    "validate_success_spec",
    "validate_task_candidate",
    "validate_task_candidate_set",
    "validate_task_draft",
]
