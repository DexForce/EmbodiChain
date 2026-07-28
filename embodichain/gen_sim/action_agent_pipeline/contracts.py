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

"""Compatibility facade for the action-agent serialized protocol.

New production modules must import from the owning ``protocol`` submodule or
from ``config.defaults`` for numeric fallback policy. This explicit facade
keeps historical external imports stable without retaining duplicate owners.
"""

from __future__ import annotations

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    ROBOTIQ_ARG2F_140_CLOSE_QPOS,
    ROBOTIQ_ARG2F_140_OPEN_QPOS,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.actions import (
    ARM_ACTION_KEYS,
    ATOMIC_ACTION_CLASSES,
    CONTROL_ARM,
    CONTROL_HAND,
    DUAL_ARM_NAME,
    LEFT_ARM_ACTION_KEY,
    LEFT_ARM_NAME,
    MAX_COORDINATED_PAYLOADS,
    OBJECT_ORIENTATION_AXES,
    OBJECT_ORIENTATION_GOALS,
    POSE_REFERENCES,
    RIGHT_ARM_ACTION_KEY,
    RIGHT_ARM_NAME,
    SUPPORTED_CONTROLS,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.artifacts import (
    ACTION_AGENT_ENV_ID,
    AGENT_CONFIG_FILENAME,
    ATOM_ACTIONS_FILENAME,
    BASIC_BACKGROUND_FILENAME,
    COMPILED_GRAPH_FILENAME,
    DEFAULT_VIEWER_CAMERA_UID,
    FAST_GYM_CONFIG_FILENAME,
    TASK_GRAPH_FILENAME,
    TASK_PROMPT_FILENAME,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.success import (
    SUCCESS_TERM_ALIASES,
    SUCCESS_TERM_TYPES,
    SuccessTerm,
)
from embodichain.gen_sim.action_agent_pipeline.protocol.tasks import (
    MANIPULATION_INTENTS,
    RELATIVE_RELATIONS,
    SIDE_RELATIONS,
    TASK_ROUTE_ARRANGEMENT_LINE,
    TASK_ROUTE_OBJECT_MANIPULATION,
    TASK_ROUTE_STACKING,
    TASK_ROUTE_UNSUPPORTED,
    TASK_ROUTES,
)

__all__ = [
    "ACTION_AGENT_ENV_ID",
    "AGENT_CONFIG_FILENAME",
    "ARM_ACTION_KEYS",
    "ATOM_ACTIONS_FILENAME",
    "ATOMIC_ACTION_CLASSES",
    "BASIC_BACKGROUND_FILENAME",
    "COMPILED_GRAPH_FILENAME",
    "CONTROL_ARM",
    "CONTROL_HAND",
    "DEFAULT_VIEWER_CAMERA_UID",
    "DUAL_ARM_NAME",
    "FAST_GYM_CONFIG_FILENAME",
    "LEFT_ARM_NAME",
    "LEFT_ARM_ACTION_KEY",
    "MANIPULATION_INTENTS",
    "MAX_COORDINATED_PAYLOADS",
    "OBJECT_ORIENTATION_AXES",
    "OBJECT_ORIENTATION_GOALS",
    "POSE_REFERENCES",
    "RELATIVE_RELATIONS",
    "RIGHT_ARM_NAME",
    "RIGHT_ARM_ACTION_KEY",
    "ROBOTIQ_ARG2F_140_CLOSE_QPOS",
    "ROBOTIQ_ARG2F_140_OPEN_QPOS",
    "SIDE_RELATIONS",
    "SuccessTerm",
    "SUCCESS_TERM_ALIASES",
    "SUCCESS_TERM_TYPES",
    "SUPPORTED_CONTROLS",
    "TASK_GRAPH_FILENAME",
    "TASK_PROMPT_FILENAME",
    "TASK_ROUTE_ARRANGEMENT_LINE",
    "TASK_ROUTE_OBJECT_MANIPULATION",
    "TASK_ROUTE_STACKING",
    "TASK_ROUTE_UNSUPPORTED",
    "TASK_ROUTES",
]
