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

"""Stable artifact and integration identifiers for the action-agent pipeline."""

from __future__ import annotations

from typing import Final

__all__ = [
    "ACTION_AGENT_ENV_ID",
    "AGENT_CONFIG_FILENAME",
    "ATOM_ACTIONS_FILENAME",
    "BASIC_BACKGROUND_FILENAME",
    "COMPILED_GRAPH_FILENAME",
    "DEFAULT_VIEWER_CAMERA_UID",
    "FAST_GYM_CONFIG_FILENAME",
    "TASK_GRAPH_FILENAME",
    "TASK_PROMPT_FILENAME",
]

# Artifact names are serialized into generated bundles and referenced by
# runtime loaders. Renaming one is therefore a protocol migration, not a local
# filesystem cleanup.
FAST_GYM_CONFIG_FILENAME: Final = "fast_gym_config.json"
AGENT_CONFIG_FILENAME: Final = "agent_config.json"
TASK_PROMPT_FILENAME: Final = "task_prompt.txt"
SEED_TASK_GRAPH_FILENAME: Final = "seed_task_graph.json"
TASK_GRAPH_FILENAME: Final = "task_graph.json"
BASIC_BACKGROUND_FILENAME: Final = "basic_background.txt"
ATOM_ACTIONS_FILENAME: Final = "atom_actions.txt"

# This derived cache is colocated with generated configs but may be deleted and
# rebuilt independently from the required generation artifacts above.
COMPILED_GRAPH_FILENAME: Final = "agent_compiled_graph.json"

# These identifiers cross Gym registration, generated sensor configuration,
# and runtime observation lookup, so all stages must use the same values.
ACTION_AGENT_ENV_ID: Final = "AtomicActionsAgent-v3"
DEFAULT_VIEWER_CAMERA_UID: Final = "cam_high"
