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

"""Host-neutral expansion orchestration with lazy simulation dependencies."""

from __future__ import annotations
from importlib import import_module

__all__ = [
    "ExpansionEngine",
    "ExpansionJobCfg",
    "ExpansionAssignment",
    "ExpansionHost",
    "EpisodeSink",
    "ExpansionSession",
    "ExpansionSessionCfg",
    "ExpansionCoverage",
    "EpisodeCoverage",
    "ExpansionRunCfg",
    "load_expansion_job",
    "preflight_expansion_job",
    "run_expansion_job",
    "inspect_expansion_job",
    "resume_expansion_export",
    "ArtifactEpisodeSink",
    "EpisodeCandidate",
    "ExpansionEpisode",
    "ExpertSource",
    "ExpansionOperator",
    "ExpansionContext",
    "OperatorCapability",
    "PinnedExpertSource",
    "UniformRangeOperator",
    "JointSceneSource",
    "ExpansionRecipe",
    "RecipeOperation",
    "BatchCompatibility",
    "PreparedEpisodeWorkItem",
    "ExpansionWorkItem",
    "ExpansionRecipeSource",
    "PreparingExpansionHost",
    "GymExpansionHost",
    "ExpansionJournal",
    "RecipeAttempt",
    "ArtifactReference",
    "EpisodeArtifactManifest",
    "load_expansion_episode",
    "resolve_expansion_job",
    "build_expansion_report",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    owner = {
        "ExpansionRunCfg": "configured",
        "ExpansionSession": "session",
        "ExpansionSessionCfg": "session_cfg",
        "ExpansionCoverage": "coverage",
        "EpisodeCoverage": "coverage",
        "load_expansion_job": "configured",
        "preflight_expansion_job": "configured",
        "run_expansion_job": "workflow",
        "inspect_expansion_job": "workflow",
        "resume_expansion_export": "workflow",
        "ArtifactEpisodeSink": "persistence",
        "EpisodeCandidate": "contracts",
        "ExpansionEpisode": "contracts",
        "ExpertSource": "recipes",
        "ExpansionOperator": "recipes",
        "ExpansionContext": "recipes",
        "OperatorCapability": "recipes",
        "PinnedExpertSource": "recipes",
        "UniformRangeOperator": "recipes",
        "JointSceneSource": "scene",
        "ExpansionRecipe": "recipes",
        "RecipeOperation": "recipes",
        "BatchCompatibility": "recipes",
        "GymExpansionHost": "host",
        "ExpansionJournal": "journal",
        "RecipeAttempt": "journal",
        "ArtifactReference": "artifacts",
        "EpisodeArtifactManifest": "artifacts",
        "load_expansion_episode": "persistence",
        "resolve_expansion_job": "configured",
        "build_expansion_report": "reporting",
    }.get(name, "engine")
    module = f"{__name__}.{owner}"
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
