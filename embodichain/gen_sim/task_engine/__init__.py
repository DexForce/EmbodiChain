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

"""Task-level semantic intent and planning contracts."""

from __future__ import annotations

from typing import Any

from .contracts import (
    FORBIDDEN_SEMANTIC_GRAPH_FIELDS,
    PLANNER_ROUTES,
    REASONING_TYPES,
    SEMANTIC_TASK_GRAPH_FILENAME,
    SEMANTIC_TASK_GRAPH_SCHEMA,
    TASK_LEVELS,
    TASK_SPEC_FILENAME,
    TASK_SPEC_SCHEMA,
    FailurePolicy,
    PlannerProvenance,
    SemanticTaskGraph,
    SemanticTaskNode,
    SuccessSpec,
    TaskGroupSpec,
    TaskInstanceSpec,
    TaskSpec,
    decode_semantic_task_graph,
    decode_task_spec,
    semantic_task_graph_hash,
    task_spec_hash,
)
from .frontend import (
    SCENE_REQUIREMENTS_SCHEMA,
    TASK_DRAFT_OUTPUT_SCHEMA,
    TASK_DRAFT_SCHEMA,
    BoundTaskDraft,
    RoleRequirement,
    SceneRequirements,
    SemanticCallCandidate,
    TaskDraft,
    TaskDraftCaller,
    TaskInterpretationError,
    TaskInterpretationResult,
    bind_task_draft,
    decode_scene_requirements,
    decode_task_draft,
    interpret_task_candidates,
    validate_planner_projection,
)
from .ontology import (
    RELATIONS,
    TASK_CONTRACTS,
    TASK_TYPES,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    TaskContract,
    task_contract,
    task_success_type,
)
from .config import (
    TASK_ENGINE_DEFAULTS_SCHEMA,
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    TaskEngineWorkflowCfg,
    load_task_engine_config,
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
    "FORBIDDEN_SEMANTIC_GRAPH_FIELDS",
    "PLANNER_ROUTES",
    "REASONING_TYPES",
    "RELATIONS",
    "SCENE_REQUIREMENTS_SCHEMA",
    "SEMANTIC_TASK_GRAPH_FILENAME",
    "SEMANTIC_TASK_GRAPH_SCHEMA",
    "TASK_CONTRACTS",
    "TASK_DRAFT_OUTPUT_SCHEMA",
    "TASK_DRAFT_SCHEMA",
    "TASK_LEVELS",
    "TASK_SPEC_FILENAME",
    "TASK_SPEC_SCHEMA",
    "TASK_TYPES",
    "TERMINAL_BEHAVIORS",
    "TRANSPORT_DIRECTIONS",
    "FailurePolicy",
    "BoundTaskDraft",
    "PlannerProvenance",
    "SemanticTaskGraph",
    "SemanticTaskNode",
    "SuccessSpec",
    "RoleRequirement",
    "SceneRequirements",
    "SemanticCallCandidate",
    "TaskContract",
    "TaskGroupSpec",
    "TaskInstanceSpec",
    "TaskSpec",
    "TaskDraft",
    "TaskDraftCaller",
    "TaskInterpretationError",
    "TaskInterpretationResult",
    "bind_task_draft",
    "decode_scene_requirements",
    "decode_semantic_task_graph",
    "decode_task_draft",
    "decode_task_spec",
    "semantic_task_graph_hash",
    "interpret_task_candidates",
    "task_contract",
    "task_spec_hash",
    "task_success_type",
    "validate_planner_projection",
    "TASK_RUN_REQUEST_SCHEMA",
    "TASK_ENGINE_DEFAULTS_SCHEMA",
    "SceneInputKind",
    "SceneAnalysis",
    "SceneEngineBackend",
    "SceneRevision",
    "StageStatus",
    "TaskEngineState",
    "TaskEngineExecutionCfg",
    "TaskEnginePlanningCfg",
    "TaskEngineWorkflowCfg",
    "TaskEngineRunResult",
    "TaskEngineWorkflow",
    "TaskRunRequest",
    "WorkflowStage",
    "TASK_ENGINE_RUN_MANIFEST_SCHEMA",
    "SubprocessActionExecutor",
    "complete_stage",
    "fail_stage",
    "initial_state",
    "load_task_engine_config",
    "replay_events",
    "task_contract",
    "task_spec_hash",
    "task_success_type",
    "scene_input_kind",
    "validate_scene_history_root",
    "scene_blueprint_objects",
    "skip_stage",
    "start_stage",
    "validate_scene_output_separation",
    "validate_task_run_request",
]

_SCENE_BACKEND_EXPORTS = {
    "SceneAnalysis",
    "SceneEngineBackend",
    "SceneRevision",
    "scene_blueprint_objects",
}
_WORKFLOW_EXPORTS = {
    "TASK_ENGINE_RUN_MANIFEST_SCHEMA",
    "SubprocessActionExecutor",
    "TaskEngineRunResult",
    "TaskEngineWorkflow",
}


def __getattr__(name: str) -> Any:
    """Load orchestration entry points lazily to avoid engine import cycles."""
    if name in _SCENE_BACKEND_EXPORTS:
        from . import scene_backend

        return getattr(scene_backend, name)
    if name in _WORKFLOW_EXPORTS:
        from . import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
