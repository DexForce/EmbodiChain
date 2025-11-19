from copy import deepcopy
from typing import Dict


class ConfigTemplate:
    @staticmethod
    def get_scope() -> Dict:
        return deepcopy(
            {
                "type": "DiGraph",
                "dim": [7],
                "init": {},
                "dtype": "float32",
            }
        )

    @staticmethod
    def get_action_config() -> Dict:
        return deepcopy(
            {
                "scope": {},
                "node": {},
                "edge": {},
                "sync": {},
                "misc": {},
                "trajectory_path": "",
            }
        )

    @staticmethod
    def get_record_config() -> Dict:
        return deepcopy(
            {
                "camera": [
                    {
                        "name": "cam1",
                        "resolution": [640, 480],
                        "position": [2, 0, 2],
                        "look_at": [0.5, 0, 1],
                        "video_fps": 25,
                        "type": "sync",
                    }
                ]
            }
        )

    @staticmethod
    def get_domain_randomization_config() -> Dict:
        return deepcopy(
            {
                "enable": True,
                "light": {
                    "light1": {
                        "enable": True,
                        "intensity_range": [5000000, 15000000],
                        "position_range": [[-0.5, -0.5, 2], [0.5, 0.5, 2]],
                        "color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
                    }
                },
                "material": {"enable": True},
            }
        )

    @staticmethod
    def get_success_params_config() -> Dict:
        return deepcopy({"strict": False})

    @staticmethod
    def get_dataset_config() -> Dict:
        from embodichain.data.enum import (
            PrivilegeType,
            CameraName,
            JointType,
            Modality,
            EndEffector,
            ControlParts,
            EefType,
        )

        return deepcopy(
            {
                "instruction": {"lang": "Pour water from the bottle into the mug."},
                "robot_meta": {
                    "arm_dofs": 14,
                    "control_freq": 25,
                    "observation": {
                        "vision": {
                            "cam_high": ["mask"],
                            "cam_right_wrist": ["mask"],
                            "cam_left_wrist": ["mask"],
                        },
                        Modality.STATES.value: [
                            ControlParts.LEFT_ARM.value + JointType.QPOS.value,
                            ControlParts.RIGHT_ARM.value + JointType.QPOS.value,
                            ControlParts.LEFT_EEF.value
                            + EndEffector.DEXTROUSHAND.value,
                            ControlParts.RIGHT_EEF.value
                            + EndEffector.DEXTROUSHAND.value,
                        ],
                        # PrivilegeType.EXTEROCEPTION.value: {
                        #     "cameras": [
                        #         CameraName.HEAD.value,
                        #         CameraName.RIGHT_WRIST.value,
                        #         CameraName.LEFT_WRIST.value,
                        #     ],
                        #     "kpnts_number": 2,
                        #     "interval": [0.02, 0.05],
                        #     "groups": 6,
                        # },
                    },
                    Modality.ACTIONS.value: [
                        ControlParts.LEFT_ARM.value + JointType.QPOS.value,
                        ControlParts.RIGHT_ARM.value + JointType.QPOS.value,
                        ControlParts.LEFT_EEF.value + EndEffector.DEXTROUSHAND.value,
                        ControlParts.RIGHT_EEF.value + EndEffector.DEXTROUSHAND.value,
                    ],
                    "min_len_steps": 30,
                },
            }
        )

    @staticmethod
    def get_node_config() -> Dict:
        return deepcopy(
            {
                "name": "xxx",
                "master": "",
                "slaver": "",
                "timestep": 0,
                "trajectory": {"anayltic": True, "kwargs": {}},
            }
        )
