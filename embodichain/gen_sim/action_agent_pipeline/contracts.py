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

"""Stable cross-stage contracts for the action-agent pipeline.

This module intentionally contains protocol values rather than tuning knobs.
Generation, prompt construction, graph compilation, and runtime execution all
exchange these exact strings, so keeping one authoritative definition prevents
otherwise silent drift between stages.
"""

from __future__ import annotations

from typing import Final

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
    "TASK_GRAPH_CACHE_FILENAME",
    "TASK_GRAPH_FILENAME",
    "TASK_PROMPT_FILENAME",
    "TASK_ROUTE_ARRANGEMENT_LINE",
    "TASK_ROUTE_OBJECT_MANIPULATION",
    "TASK_ROUTE_STACKING",
    "TASK_ROUTE_UNSUPPORTED",
    "TASK_ROUTES",
]

# Artifact names are a public contract: generated agent_config.json refers to
# these files by relative path and the runtime resolves them from that directory.
FAST_GYM_CONFIG_FILENAME: Final = "fast_gym_config.json"
AGENT_CONFIG_FILENAME: Final = "agent_config.json"
TASK_PROMPT_FILENAME: Final = "task_prompt.txt"
TASK_GRAPH_FILENAME: Final = "task_graph.json"
BASIC_BACKGROUND_FILENAME: Final = "basic_background.txt"
ATOM_ACTIONS_FILENAME: Final = "atom_actions.txt"
TASK_GRAPH_CACHE_FILENAME: Final = "agent_task_graph.json"
COMPILED_GRAPH_FILENAME: Final = "agent_compiled_graph.json"

# These identifiers cross Gym registration, generated configs, sensor templates,
# graph generation, and runtime dispatch. They must change as one protocol.
ACTION_AGENT_ENV_ID: Final = "AtomicActionsAgent-v3"
DEFAULT_VIEWER_CAMERA_UID: Final = "cam_high"

LEFT_ARM_NAME: Final = "left_arm"
RIGHT_ARM_NAME: Final = "right_arm"
DUAL_ARM_NAME: Final = "dual_arm"
LEFT_ARM_ACTION_KEY: Final = "left_arm_action"
RIGHT_ARM_ACTION_KEY: Final = "right_arm_action"
ARM_ACTION_KEYS: Final = frozenset({LEFT_ARM_ACTION_KEY, RIGHT_ARM_ACTION_KEY})

# Keep the Robotiq fallback in one place. Robot profiles remain authoritative;
# the environment uses these values only for legacy configs without profile data.
ROBOTIQ_ARG2F_140_OPEN_QPOS: Final = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
ROBOTIQ_ARG2F_140_CLOSE_QPOS: Final = (0.7, -0.7, 0.7, -0.7, -0.7, 0.7)

TASK_ROUTE_STACKING: Final = "stacking"
TASK_ROUTE_ARRANGEMENT_LINE: Final = "arrangement_line"
TASK_ROUTE_OBJECT_MANIPULATION: Final = "object_manipulation"
TASK_ROUTE_UNSUPPORTED: Final = "unsupported"
TASK_ROUTES: Final = frozenset(
    {
        TASK_ROUTE_STACKING,
        TASK_ROUTE_ARRANGEMENT_LINE,
        TASK_ROUTE_OBJECT_MANIPULATION,
        TASK_ROUTE_UNSUPPORTED,
    }
)

RELATIVE_RELATIONS: Final = frozenset(
    {
        "inside",
        "on",
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "front_left_of",
        "back_left_of",
        "front_right_of",
        "back_right_of",
    }
)
SIDE_RELATIONS: Final = RELATIVE_RELATIONS - {"inside", "on"}
MANIPULATION_INTENTS: Final = frozenset(
    {"place_relative", "hold_hover", "coordinated_pickment"}
)
OBJECT_ORIENTATION_GOALS: Final = frozenset(
    {"preserve", "upright", "lay_flat", "axis_align"}
)
OBJECT_ORIENTATION_AXES: Final = frozenset(
    {"none", "x", "y", "long_axis", "short_axis"}
)
POSE_REFERENCES: Final = frozenset({"object", "absolute", "relative"})
MAX_COORDINATED_PAYLOADS: Final = 4

ATOMIC_ACTION_CLASSES: Final = frozenset(
    {
        "CoordinatedPickment",
        "PickUp",
        "MoveEndEffector",
        "MoveJoints",
        "MoveHeldObject",
        "Place",
    }
)
CONTROL_ARM: Final = "arm"
CONTROL_HAND: Final = "hand"
SUPPORTED_CONTROLS: Final = frozenset({CONTROL_ARM, CONTROL_HAND})


class SuccessTerm:
    """Canonical success predicate names serialized into generated configs."""

    OBJECT_POSITION_NEAR: Final = "object_position_near"
    OBJECT_XY_NEAR: Final = "object_xy_near"
    OBJECT_IN_CONTAINER: Final = "object_in_container"
    OBJECT_ON_OBJECT: Final = "object_on_object"
    OBJECT_NOT_FALLEN: Final = "object_not_fallen"
    OBJECT_AXIS_OFFSET_NEAR: Final = "object_axis_offset_near"
    OBJECT_AXIS_NEAR: Final = "object_axis_near"
    OBJECTS_COLLINEAR: Final = "objects_collinear"
    OBJECTS_ORDERED: Final = "objects_ordered"
    OBJECT_LIFTED: Final = "object_lifted"
    OBJECT_HELD_BY_GRIPPER: Final = "object_held_by_gripper"
    OBJECT_HELD_BY_BOTH_GRIPPERS: Final = "object_held_by_both_grippers"
    BOTH_GRIPPERS_OPEN: Final = "both_grippers_open"
    GRIPPERS_CLEAR_OF_OBJECT: Final = "grippers_clear_of_object"
    BOTH_ARMS_AT_INITIAL_QPOS: Final = "both_arms_at_initial_qpos"


SUCCESS_TERM_TYPES: Final = frozenset(
    value
    for name, value in vars(SuccessTerm).items()
    if name.isupper() and isinstance(value, str)
)

# Aliases are accepted only for backward compatibility. New generated configs
# always emit the canonical value on the right-hand side.
SUCCESS_TERM_ALIASES: Final = {
    "object_near_position": SuccessTerm.OBJECT_POSITION_NEAR,
    "object_near_xy": SuccessTerm.OBJECT_XY_NEAR,
    "object_on": SuccessTerm.OBJECT_ON_OBJECT,
    "on_object": SuccessTerm.OBJECT_ON_OBJECT,
    "not_fallen": SuccessTerm.OBJECT_NOT_FALLEN,
    "object_relative_axis_near": SuccessTerm.OBJECT_AXIS_OFFSET_NEAR,
    "object_coordinate_near": SuccessTerm.OBJECT_AXIS_NEAR,
    "object_height_above_initial": SuccessTerm.OBJECT_LIFTED,
    "object_gripper_near": SuccessTerm.OBJECT_HELD_BY_GRIPPER,
}
