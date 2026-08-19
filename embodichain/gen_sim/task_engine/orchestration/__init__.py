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

"""Task-owned orchestration across task, scene, and action engines."""

from __future__ import annotations

from embodichain.gen_sim.action_engine.agent import ActionAgent, ActionGraph
from embodichain.gen_sim.action_engine.runtime import ExecutionReport

from .artifacts import (
    ArtifactTransaction,
    CONSERVATIVE_SCENE_GRAPH_FILENAME,
    TaskEngineArtifactPaths,
    FEASIBILITY_REPORT_FILENAME,
    PREPARATION_FAILURE_FILENAME,
    STATIC_SCENE_MANIFEST_FILENAME,
    task_engine_artifact_paths,
    write_execution_report,
    write_preparation_failure,
)
from .contracts import (
    BINDING_REPORT_SCHEMA,
    EXECUTION_REPORT_SCHEMA,
    GROUNDED_TASK_PLAN_SCHEMA,
    ROLE_BINDINGS_SCHEMA,
    SCENE_MANIFEST_SCHEMA,
    BindingReport,
    GroundedTaskPlan,
    RoleBindings,
    SceneManifest,
)
from .coordinator import (
    TaskEngineCoordinator,
    PreparationResult,
    build_grounded_task_plan,
    lower_task_candidate,
)
from .scene_adapter import SceneAdaptation, SceneAdapter, SceneAdapterProtocolError
from .scene_source import (
    SceneSourceFingerprint,
    SceneSourceRef,
    fingerprint_scene_source,
    verify_scene_source_fingerprint,
)

__all__ = [
    "ActionAgent",
    "ActionGraph",
    "ArtifactTransaction",
    "CONSERVATIVE_SCENE_GRAPH_FILENAME",
    "BINDING_REPORT_SCHEMA",
    "BindingReport",
    "TaskEngineArtifactPaths",
    "TaskEngineCoordinator",
    "EXECUTION_REPORT_SCHEMA",
    "ExecutionReport",
    "FEASIBILITY_REPORT_FILENAME",
    "GROUNDED_TASK_PLAN_SCHEMA",
    "GroundedTaskPlan",
    "PREPARATION_FAILURE_FILENAME",
    "PreparationResult",
    "ROLE_BINDINGS_SCHEMA",
    "RoleBindings",
    "SCENE_MANIFEST_SCHEMA",
    "STATIC_SCENE_MANIFEST_FILENAME",
    "SceneAdaptation",
    "SceneAdapter",
    "SceneAdapterProtocolError",
    "SceneManifest",
    "SceneSourceFingerprint",
    "SceneSourceRef",
    "build_grounded_task_plan",
    "task_engine_artifact_paths",
    "fingerprint_scene_source",
    "verify_scene_source_fingerprint",
    "lower_task_candidate",
    "write_execution_report",
    "write_preparation_failure",
]
