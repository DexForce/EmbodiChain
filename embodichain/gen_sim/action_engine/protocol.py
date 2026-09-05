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

"""Cross-layer identifiers owned by Action Engine.

These values are serialized into generated artifacts, so changing one is a
protocol migration rather than a local rename.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ACTION_ENGINE_CONFIG_SCHEMA",
    "ACTION_ENGINE_ENV_ID",
    "AGENT_CONFIG_FILENAME",
    "COMPARISON_FILENAME",
    "EXECUTION_PROGRAM_FILENAME",
    "EXECUTION_PROGRAM_SCHEMA",
    "FAST_GYM_CONFIG_FILENAME",
    "SCENE_REQUIREMENTS_FILENAME",
    "SCENE_REQUIREMENTS_SCHEMA",
    "SEED_TASK_GRAPH_PNG_FILENAME",
    "SEED_GRAPH_SCHEMA",
    "TASK_SPEC_FILENAME",
    "TASK_SPEC_SCHEMA",
    "TASK_AGENT_FILENAME",
    "TASK_AGENT_SCHEMA",
]

ACTION_ENGINE_ENV_ID: Final = "ActionEngine-v1"
ACTION_ENGINE_CONFIG_SCHEMA: Final = "action_engine_config_v2"
TASK_AGENT_SCHEMA: Final = "action_engine_task_agent_v1"
EXECUTION_PROGRAM_SCHEMA: Final = "action_engine_execution_graph_v1"
SEED_GRAPH_SCHEMA: Final = "action_engine_seed_graph_v3"
TASK_SPEC_SCHEMA: Final = "action_engine_task_spec_v2"
SCENE_REQUIREMENTS_SCHEMA: Final = "action_engine_scene_requirements_v2"

FAST_GYM_CONFIG_FILENAME: Final = "fast_gym_config.json"
AGENT_CONFIG_FILENAME: Final = "agent_config.json"
TASK_AGENT_FILENAME: Final = "task_agent.json"
EXECUTION_PROGRAM_FILENAME: Final = "seed_task_graph.json"
SEED_TASK_GRAPH_PNG_FILENAME: Final = "seed_task_graph.png"
TASK_SPEC_FILENAME: Final = "task_spec.json"
SCENE_REQUIREMENTS_FILENAME: Final = "scene_requirements.json"
COMPARISON_FILENAME: Final = "comparison.json"
