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

"""Task understanding, normative declarations and semantic graph contracts."""

from __future__ import annotations

from time import perf_counter
from typing import Any

_CLI_STARTED_AT = perf_counter()

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

from .task_spec import (
    TaskSpecGenerator,
    generate_task_spec,
    task_spec_from_candidate,
    task_template_from_candidate,
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

from .state_machine import (
    StageStatus,
    TaskEngineState,
    WorkflowStage,
    complete_stage,
    fail_stage,
    initial_state,
    replay_events,
    skip_stage,
    start_stage,
)

from .workflow_contracts import (
    TASK_RUN_REQUEST_SCHEMA,
    SceneInputKind,
    TaskRunRequest,
    scene_input_kind,
    validate_scene_history_root,
    validate_scene_output_separation,
    validate_task_run_request,
)

__all__ = [
    "TASK_ENGINE_DEFAULTS_SCHEMA",
    "TaskEngineExecutionCfg",
    "TaskEnginePlanningCfg",
    "TaskEngineWorkflowCfg",
    "TaskEngineWorkflow",
    "TaskEngineRunResult",
    "SubprocessActionExecutor",
    "TASK_ENGINE_RUN_MANIFEST_SCHEMA",
    "SceneAnalysis",
    "SceneEngineBackend",
    "SceneRevision",
    "load_task_engine_config",
    "SemanticTaskPlanner",
    "SemanticTaskPlan",
    "UnsupportedSemanticCapabilityError",
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
    "TaskSpecGenerator",
    "TaskContract",
    "TaskDraft",
    "TaskGenerationError",
    "TASK_RUN_REQUEST_SCHEMA",
    "SceneInputKind",
    "StageStatus",
    "TaskEngineState",
    "TaskRunRequest",
    "WorkflowStage",
    "canonical_hash",
    "derive_scene_request",
    "derive_success_spec",
    "generate_task_spec",
    "task_spec_from_candidate",
    "complete_stage",
    "fail_stage",
    "initial_state",
    "interpret_instruction_draft",
    "replay_events",
    "task_contract",
    "task_template_from_candidate",
    "task_success_type",
    "scene_input_kind",
    "validate_scene_history_root",
    "skip_stage",
    "start_stage",
    "validate_instruction_intent",
    "validate_scene_request",
    "validate_scene_output_separation",
    "validate_success_spec",
    "validate_task_candidate",
    "validate_task_candidate_set",
    "validate_task_draft",
    "validate_task_run_request",
]


def __getattr__(name: str) -> Any:
    """Load planning and orchestration only when their public APIs are requested."""
    if name in {"SceneAnalysis", "SceneEngineBackend", "SceneRevision"}:
        from . import scene_backend

        return getattr(scene_backend, name)
    if name in {
        "TaskEngineWorkflow",
        "TaskEngineRunResult",
        "SubprocessActionExecutor",
        "TASK_ENGINE_RUN_MANIFEST_SCHEMA",
    }:
        from . import workflow

        return getattr(workflow, name)
    if name in {
        "TASK_ENGINE_DEFAULTS_SCHEMA",
        "TaskEngineExecutionCfg",
        "TaskEnginePlanningCfg",
        "TaskEngineWorkflowCfg",
        "load_task_engine_config",
    }:
        from . import config

        return getattr(config, name)
    if name in {
        "SemanticTaskPlanner",
        "SemanticTaskPlan",
        "UnsupportedSemanticCapabilityError",
    }:
        from . import semantic_planner

        return getattr(semantic_planner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
