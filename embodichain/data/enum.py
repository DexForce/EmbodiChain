# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from enum import Enum


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
