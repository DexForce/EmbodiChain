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

"""Task Engine ownership of scene adaptation and static feasibility."""

from __future__ import annotations

from .contracts import (
    ASSESSMENT_STATUSES,
    FEASIBILITY_REPORT_SCHEMA,
    STATIC_SCENE_MANIFEST_SCHEMA,
    FeasibilityReport,
    StaticSceneManifest,
    validate_feasibility_report,
    validate_static_scene_manifest,
)
from .feasibility import FeasibilityBroker
from .scene_engine_v1 import SceneEngineV1Adapter
from .conservative_graph import (
    CONSERVATIVE_SCENE_GRAPH_SCHEMA,
    ConservativeSceneGraph,
    build_conservative_scene_graph,
    validate_conservative_scene_graph,
)

__all__ = [
    "ASSESSMENT_STATUSES",
    "CONSERVATIVE_SCENE_GRAPH_SCHEMA",
    "FEASIBILITY_REPORT_SCHEMA",
    "STATIC_SCENE_MANIFEST_SCHEMA",
    "FeasibilityBroker",
    "FeasibilityReport",
    "SceneEngineV1Adapter",
    "StaticSceneManifest",
    "ConservativeSceneGraph",
    "build_conservative_scene_graph",
    "validate_feasibility_report",
    "validate_static_scene_manifest",
    "validate_conservative_scene_graph",
]
