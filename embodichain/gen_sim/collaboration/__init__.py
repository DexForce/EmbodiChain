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

"""Cross-engine orchestration for task, scene, and action owners."""

from __future__ import annotations

from embodichain.gen_sim.action_engine.agent import ActionAgent, ActionGraph
from embodichain.gen_sim.action_engine.runtime import ExecutionReport
from embodichain.gen_sim.task_engine import TaskAgent, TaskGenerationError

from .artifacts import (
    ArtifactTransaction,
    CollaborationArtifactPaths,
    collaboration_artifact_paths,
    write_execution_report,
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
    CollaborationCoordinator,
    Coordinator,
    PreparationResult,
    build_grounded_task_plan,
    lower_task_candidate,
)
from .scene_adapter import SceneAdaptation, SceneAdapter, SceneAdapterProtocolError
from .scene_store import (
    ScenePackageCorruptError,
    ScenePackageNotFoundError,
    ScenePackageRef,
    ScenePackageStore,
    SceneSourceRef,
)

__all__ = [
    "ActionAgent",
    "ActionGraph",
    "ArtifactTransaction",
    "BINDING_REPORT_SCHEMA",
    "BindingReport",
    "CollaborationArtifactPaths",
    "CollaborationCoordinator",
    "Coordinator",
    "EXECUTION_REPORT_SCHEMA",
    "ExecutionReport",
    "GROUNDED_TASK_PLAN_SCHEMA",
    "GroundedTaskPlan",
    "PreparationResult",
    "ROLE_BINDINGS_SCHEMA",
    "RoleBindings",
    "SCENE_MANIFEST_SCHEMA",
    "SceneAdaptation",
    "SceneAdapter",
    "SceneAdapterProtocolError",
    "SceneManifest",
    "ScenePackageCorruptError",
    "ScenePackageNotFoundError",
    "ScenePackageRef",
    "ScenePackageStore",
    "SceneSourceRef",
    "TaskAgent",
    "TaskGenerationError",
    "build_grounded_task_plan",
    "collaboration_artifact_paths",
    "lower_task_candidate",
    "write_execution_report",
]
