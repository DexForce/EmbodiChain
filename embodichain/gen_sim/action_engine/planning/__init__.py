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

"""Stable route-free task planning API."""

from __future__ import annotations

from .online import plan_online_seed_graph
from .dual import CandidatePair, plan_candidates_parallel
from .linker import (
    CONTRACT_LINKER_VERSION,
    link_seed_graph,
    link_task_dependencies,
    validate_persisted_contracts,
)
from .planner import plan_task
from .selection import (
    CandidateEvaluation,
    evaluate_candidate,
    fuse_seed_graphs,
    select_seed_graph,
)
from .vision import (
    CameraObservation,
    SceneObservation,
    analyze_visual_scene,
    collect_scene_observation,
    validate_visual_facts,
)

__all__ = [
    "CONTRACT_LINKER_VERSION",
    "CameraObservation",
    "CandidatePair",
    "CandidateEvaluation",
    "SceneObservation",
    "analyze_visual_scene",
    "collect_scene_observation",
    "evaluate_candidate",
    "fuse_seed_graphs",
    "link_seed_graph",
    "link_task_dependencies",
    "plan_online_seed_graph",
    "plan_candidates_parallel",
    "plan_task",
    "select_seed_graph",
    "validate_visual_facts",
    "validate_persisted_contracts",
]
