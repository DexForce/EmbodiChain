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

"""Capability-driven planning and live execution for generated simulations.

Action Engine deliberately exposes a small public surface. Natural-language
goals become a typed TaskSpec, deterministic planning lowers them into a
coordinate-free SeedGraph, and the runtime grounds that graph only against live
simulator state.
"""

from __future__ import annotations

from .unbound import (
    UNBOUND_ACTION_PLAN_SCHEMA,
    UnboundActionPlan,
    build_unbound_action_plan,
    validate_unbound_action_plan,
)

from .protocol import (
    ACTION_ENGINE_CONFIG_SCHEMA,
    ACTION_ENGINE_ENV_ID,
    EXECUTION_PROGRAM_SCHEMA,
    TASK_AGENT_SCHEMA,
)

__all__ = [
    "ACTION_ENGINE_CONFIG_SCHEMA",
    "ACTION_ENGINE_ENV_ID",
    "EXECUTION_PROGRAM_SCHEMA",
    "TASK_AGENT_SCHEMA",
    "UNBOUND_ACTION_PLAN_SCHEMA",
    "UnboundActionPlan",
    "build_unbound_action_plan",
    "validate_unbound_action_plan",
]
