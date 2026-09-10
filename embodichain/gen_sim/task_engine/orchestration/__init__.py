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

"""Task-owned orchestration across task, scene, and Semantic Skill engines."""

from __future__ import annotations

from typing import Any

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
    ROLE_BINDINGS_SCHEMA,
    SCENE_MANIFEST_SCHEMA,
    BindingReport,
    RoleBindings,
    SceneManifest,
)
from .scene_adapter import (
    CandidateSelection,
    SceneAdaptation,
    SceneAdapter,
    SceneAdapterProtocolError,
)
from .scene_source import (
    SceneSourceFingerprint,
    SceneSourceRef,
    fingerprint_scene_source,
    verify_scene_source_fingerprint,
)
from .legacy_scene import (
    LEGACY_SCENE_CONVERSION_SCHEMA,
    LegacySceneRevision,
    convert_legacy_gym_project,
    restore_locked_scene_entities,
)

__all__ = [
    "ArtifactTransaction",
    "CONSERVATIVE_SCENE_GRAPH_FILENAME",
    "BINDING_REPORT_SCHEMA",
    "BindingReport",
    "TaskEngineArtifactPaths",
    "TaskEngineCoordinator",
    "FEASIBILITY_REPORT_FILENAME",
    "PREPARATION_FAILURE_FILENAME",
    "PreparationResult",
    "ROLE_BINDINGS_SCHEMA",
    "RoleBindings",
    "SCENE_MANIFEST_SCHEMA",
    "STATIC_SCENE_MANIFEST_FILENAME",
    "SceneAdaptation",
    "CandidateSelection",
    "SceneAdapter",
    "SceneAdapterProtocolError",
    "SceneManifest",
    "SceneSourceFingerprint",
    "SceneSourceRef",
    "task_engine_artifact_paths",
    "fingerprint_scene_source",
    "LEGACY_SCENE_CONVERSION_SCHEMA",
    "LegacySceneRevision",
    "convert_legacy_gym_project",
    "restore_locked_scene_entities",
    "verify_scene_source_fingerprint",
    "write_execution_report",
    "write_preparation_failure",
]


def __getattr__(name: str) -> Any:
    """Load coordinator entry points lazily to avoid graph-contract cycles."""
    if name in {"PreparationResult", "TaskEngineCoordinator"}:
        from . import coordinator

        return getattr(coordinator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
