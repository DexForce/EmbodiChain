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

from enum import Enum, IntEnum


class SemanticMask(IntEnum):
    """
    SemanticMask is an enumeration representing different semantic regions in an image or scene.

    Attributes:
        BACKGROUND (int): Represents the background region (value: 0).
        FOREGROUND (int): Represents the foreground objects (value: 1).
        ROBOT_LEFT (int): Represents the left robot region (value: 2).
        ROBOT_RIGHT (int): Represents the right robot region (value: 3).
    """

    BACKGROUND = 0
    FOREGROUND = 1
    ROBOT_LEFT = 2
    ROBOT_RIGHT = 3


class EndEffector(Enum):
    GRIPPER = "gripper"
    DEXTROUSHAND = "hand"


class EefExecute(Enum):
    OPEN = "execute_open"
    CLOSE = "execute_close"


class ControlParts(Enum):
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_EEF = "left_eef"
    RIGHT_EEF = "right_eef"
    HEAD = "head"
    WAIST = "waist"


class Hints(Enum):
    EEF = (
        ControlParts.LEFT_EEF.value,
        ControlParts.RIGHT_EEF.value,
        EndEffector.GRIPPER.value,
        EndEffector.DEXTROUSHAND.value,
    )
    ARM = (ControlParts.LEFT_ARM.value, ControlParts.RIGHT_ARM.value)


class JointType(Enum):
    QPOS = "qpos"


class EefType(Enum):
    POSE = "eef_pose"


class ActionMode(Enum):
    ABSOLUTE = ""
    RELATIVE = "delta_"  # This indicates the action is relative change with respect to last state.


class LeRobotKey(Enum):
    """LeRobot standard field keys - official LeRobot dataset format."""

    OBS_STR = "observation"
    OBS_PREFIX = "observation."
    OBS_ENV_STATE = "observation.environment_state"
    OBS_STATE = "observation.state"
    OBS_QVEL = "observation.qvel"
    OBS_QF = "observation.qf"
    OBS_IMAGE = "observation.image"
    OBS_IMAGES = "observation.images"
    OBS_LANGUAGE = "observation.language"
    OBS_LANGUAGE_TOKENS = "observation.language.tokens"
    OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
    OBS_LANGUAGE_SUBTASK = "observation.subtask"
    OBS_LANGUAGE_SUBTASK_TOKENS = "observation.subtask.tokens"
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK = "observation.subtask.attention_mask"
    ACTION = "action"
    ACTION_PREFIX = "action."
    ACTION_TOKENS = "action.tokens"
    ACTION_TOKEN_MASK = "action.token_mask"
    REWARD = "next.reward"
    TRUNCATED = "next.truncated"
    DONE = "next.done"
    INFO = "info"
