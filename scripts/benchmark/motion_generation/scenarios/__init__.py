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

"""Scenario providers for motion-generation benchmarks."""

from __future__ import annotations

from .atomic_objects import (
    AtomicObjectHandle,
    atomic_object_kind_names,
    create_atomic_object,
    register_atomic_object_kind,
)
from .atomic_task import (
    AtomicSkillCaseProvider,
    AtomicTaskScenario,
    atomic_skill_provider_names,
    create_atomic_skill_provider,
    register_atomic_skill_provider,
)
from .base import ScenarioEvaluation, ScenarioProvider
from .free_space import FreeSpaceScenario

__all__ = [
    "AtomicObjectHandle",
    "AtomicSkillCaseProvider",
    "AtomicTaskScenario",
    "atomic_object_kind_names",
    "atomic_skill_provider_names",
    "create_atomic_object",
    "create_atomic_skill_provider",
    "FreeSpaceScenario",
    "register_atomic_object_kind",
    "register_atomic_skill_provider",
    "ScenarioEvaluation",
    "ScenarioProvider",
]
