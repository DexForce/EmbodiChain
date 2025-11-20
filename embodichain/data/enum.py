# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import torch
import numpy as np

from typing import List, Tuple, Union, Dict
from enum import Enum, IntEnum
from itertools import product
from aenum import Enum as AEnum
from aenum import NoAlias

from embodichain.utils.utility import get_right_name


class ModalInput:
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray] = None,
        mask: Union[torch.Tensor, np.ndarray] = None,
        name: str = "",
    ):
        self.data = data
        self.mask = mask  # indicator mask for the data, e.g., which part is valid.
        self.name = name


class Privilege:
    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray] = None,
        mask: Union[torch.Tensor, np.ndarray] = None,
        name: str = "",
    ):
        self.data = data
        self.mask = mask  # indicator mask for the data, e.g., which part is valid.
        self.name = name


class Mask(Privilege):
    pass


class Exteroception(Privilege):
    pass


class State(Privilege):
    pass


class Proprioception(ModalInput):
    pass


class Image(ModalInput):
    pass


class GeoMap(ModalInput):
    pass


class Lang(ModalInput):
    pass


class Modality(Enum):
    STATES = "states"
    STATE_INDICATOR = "state_indicator"
    ACTIONS = "actions"
    ACTION_INDICATOR = "action_indicator"
    IMAGES = "images"
    LANG = "lang"
    LANG_INDICATOR = "lang_indicator"
    GEOMAP = "geomap"  # e.g., depth, point cloud, etc.
    VISION_LANGUAGE = "vision_language"  # e.g., image + lang


class JointType(Enum):
    QPOS = "qpos"


class EefType(Enum):
    POSE = "eef_pose"


class ActionMode(Enum):
    ABSOLUTE = ""
    RELATIVE = "delta_"  # This indicates the action is relative change with respect to last state.


class EndEffector(Enum):
    GRIPPER = "gripper"
    DEXTROUSHAND = "hand"


class EefExecute(Enum):
    OPEN = "execute_open"
    CLOSE = "execute_close"


class CameraName(Enum):
    HEAD = "cam_high"
    HEAD_RIGHT = get_right_name("cam_high")
    RIGHT_WRIST = "cam_right_wrist"
    LEFT_WRIST = "cam_left_wrist"


class ControlParts(Enum):
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_EEF = "left_eef"
    RIGHT_EEF = "right_eef"
    HEAD = "head"
    WAIST = "waist"


class TeleoperationData(Enum):
    """Enum for teleoperation data conversion script specific string constants"""

    # Camera types
    HEAD_CAMERA = "head"
    HAND_CAMERA = "hand"

    # Camera positions
    LEFT_PLACE = "left"
    RIGHT_PLACE = "right"

    # Camera name prefixes
    CAM_HIGH_PREFIX = "cam_high"
    CAM_HAND_PREFIX = "cam_hand"

    # File names and patterns
    METADATA_FILE = "metadata.jsonl"
    QPOS_PATTERN = "pose_record_*.json"
    IMAGE_PATH_KEY = "image_path"
    TIMESTAMP_KEY = "timestamp"
    CAMERA_TYPE_KEY = "camera_type"

    # Data structure keys
    OBSERVATIONS = "observations"
    IMAGES = "images"
    QPOS = "qpos"
    ACTION = "action"
    FRAMES = "frames"
    DATA = "data"

    # Joint keys (common ones)
    LEFT_GRIPPER = "LEFT_GRIPPER"
    RIGHT_GRIPPER = "RIGHT_GRIPPER"
    LEFT_HAND_PREFIX = "LEFT_HAND"
    RIGHT_HAND_PREFIX = "RIGHT_HAND"
    # Joint index mapping for real robot data
    LEFT_ARM_QPOS_INDICES = [6, 7, 8, 9, 10, 11, 12]
    RIGHT_ARM_QPOS_INDICES = [14, 15, 16, 17, 18, 19, 20]
    LEFT_EEF_DEXTROUSHAND_INDICES = [22, 23, 24, 25, 26, 27]
    RIGHT_EEF_DEXTROUSHAND_INDICES = [28, 29, 30, 31, 32, 33]
    WAIST_QPOS_INDICES = [
        3,
    ]
    HEAD_QPOS_INDICES = [4, 5]


class Hints(Enum):
    EEF = (
        ControlParts.LEFT_EEF.value,
        ControlParts.RIGHT_EEF.value,
        EndEffector.GRIPPER.value,
        EndEffector.DEXTROUSHAND.value,
    )
    ARM = (ControlParts.LEFT_ARM.value, ControlParts.RIGHT_ARM.value)


class CameraLoc(AEnum):
    # The difference between CameraLoc and CameraName is that CameraLoc allows duplicate values.
    # And the value is used to indicate the sub-network ids, e.g. LEFT_WRIST and RIGHT_WRIST share the same sub-network feature extraction.
    _settings_ = NoAlias
    HEAD = 0
    RIGHT_WRIST = 1
    LEFT_WRIST = 1


class CameraOrder(IntEnum):
    # This is used to indicate the order of camera inputs, for both simulation and real deployment, training and inference.
    # For dual system, the order is HEAD, RIGHT_WRIST, LEFT_WRIST.
    HEAD = 0
    RIGHT_WRIST = 1
    LEFT_WRIST = 2


DEFAULT_CAMERA_ORDER = {tmp.value: CameraName[tmp.name].value for tmp in CameraOrder}
DEFAULT_CAMERA_LOC = {CameraName[tmp.name].value: tmp.value for tmp in CameraLoc}


def link_type(*args) -> str:
    l = len(args)
    if l == 0:
        return ""
    elif l == 1:
        return args[0]
    elif l >= 2:
        ret_str = "[{}]".format(args[0])
        for i in range(1, l):
            ret_str += "_[{}]".format(args[i])
        return ret_str


combined_members = {
    link_type(a.name + b.name, c.name + d.name, e.name): link_type(
        a.value + b.value, c.value + d.value, e.value
    )
    for a, b, c, d, e in product(
        ActionMode, JointType, ActionMode, EefType, EndEffector
    )
}
ActionType = Enum("ActionType", combined_members)
combined_proprio_members = {
    link_type(a.name, b.name, c.name): link_type(a.value, b.value, c.value)
    for a, b, c in product(JointType, EefType, EndEffector)
}
ProprioType = Enum("ProprioType", combined_proprio_members)


def parse_action_type(action_type: str) -> Tuple[str, str, str]:
    splits = action_type.split("[")
    assert len(splits) == 3, "{} must contain 3-[].".format(action_type)
    proprio_type = splits[0].split("]")[0]
    eef_type = splits[1].split("]")[0]
    end_effector = splits[2].split("]")[0]
    return proprio_type, eef_type, end_effector


def parse_proprio_type(proprio_type: str) -> Tuple[str, str, str]:
    return parse_action_type(proprio_type)


class PrivilegeType(Enum):
    EXTEROCEPTION = "exteroception"
    MASK = "mask"
    STATE = "state"
    PROGRESS = "progress"


SUPPORTED_PROPRIO_TYPES = [
    ControlParts.LEFT_ARM.value + EefType.POSE.value,
    ControlParts.RIGHT_ARM.value + EefType.POSE.value,
    ControlParts.LEFT_ARM.value + JointType.QPOS.value,
    ControlParts.RIGHT_ARM.value + JointType.QPOS.value,
    ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
    ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
    ControlParts.LEFT_EEF.value + EndEffector.GRIPPER.value,
    ControlParts.RIGHT_EEF.value + EndEffector.GRIPPER.value,
]
SUPPORTED_ACTION_TYPES = SUPPORTED_PROPRIO_TYPES + [
    ControlParts.LEFT_ARM.value + ActionMode.RELATIVE.value + JointType.QPOS.value,
    ControlParts.RIGHT_ARM.value + ActionMode.RELATIVE.value + JointType.QPOS.value,
]
SUPPORTED_EXTRA_VISION_TYPES = [
    Modality.GEOMAP.value,
    PrivilegeType.EXTEROCEPTION.value,
    PrivilegeType.MASK.value,
]


def search_sub_str(sub_str: str, refs: List[str]):
    ret = [False for _ in refs]
    for i, ref in enumerate(refs):
        ret[i] = sub_str in ref
    return ret


class ArmEnum(IntEnum):
    LEFT_ARM_ONLY = 1
    RIGHT_ARM_ONLY = 2
    DUAL_ARM = 3


class ArmName(Enum):
    LEFT_ARM_ONLY = "left_arm"
    RIGHT_ARM_ONLY = "right_arm"


class SemanticMask(IntEnum):
    BACKGROUND = 0
    FOREGROUND = 1
    ROBOT = 2


def get_all_cond(suffix: str = "_cond") -> Dict[str, str]:
    cond_dict = {}
    for modality in Modality:
        cond_dict[modality.value] = modality.value + suffix
    for privilege in PrivilegeType:
        cond_dict[privilege.value] = privilege.value + suffix
    return cond_dict


def is_dual_arms(dofs: int) -> bool:
    return dofs > 10


from collections import deque


class HistoryChunks:
    def __init__(self, history_len: int = 2) -> None:
        self.deque = deque(maxlen=history_len)
        self.history_len = history_len

    def inqueue(self, data: ModalInput) -> None:
        self.deque.append(data)

    def __getitem__(
        self,
        index: int,
    ) -> ModalInput:
        return self.deque[index]

    def __len__(
        self,
    ) -> int:
        return len(self.deque)

    def isfull(
        self,
    ) -> bool:
        return len(self) == self.history_len

    def get_list(
        self,
    ) -> List[ModalInput]:
        return list(self.deque)

    def clean(self) -> None:
        self.deque.clear()
