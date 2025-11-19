# ----------------------------------------------------------------------------
# Copyright (c) 2021-2024 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
import os
import sys
import tempfile
from pathlib import Path

import unittest

sys.path.append(str(Path(__file__).parent.parent))
from common import UnittestMetaclass


from embodichain.utils.utility import dict2args, change_nested_dict
from embodichain.lab.gym.envs import NoFailWrapper
from embodichain.data.enum import Modality, PrivilegeType
from embodichain.utils.logger import log_error
from embodichain.data.data_engine.data_dict_extractor import DataDictExtractor


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

FILE_SERVER = "http://192.168.3.120"


class TestDataDictExtractor(unittest.TestCase, metaclass=UnittestMetaclass):
    datacenter_backup = Path("/tmp/datacenter_test")
    base_url = "http://192.168.3.120/MixedAI/"

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    # TODO: delete it
    def derandomize_gym_config(self, gym_config):
        # TODO: automatically got the randomized variables
        derand_config_updates = {
            ("robot_action", "max_random_shift"): [
                0.0,
                0.0,
                0.0,
                0.0,
            ],  # list is not hashable
            ("robot_action", "table_z_shift"): 0.0,
        }
        derand_config_deletes = [
            ("robot_action", "init_xpos_range"),
            ("robot_action", "camera_intrinsic_range"),
            ("robot_action", "camera_xpos_range"),
        ]
        for keys, value in derand_config_updates.items():
            gym_config = change_nested_dict(gym_config, keys, "update", value)
        for keys in derand_config_deletes:
            gym_config = change_nested_dict(gym_config, keys, "delete")
        return gym_config

    def derandomize_objs(self, objs):
        # TODO: maybe move to env?
        # TODO: maybe use validation_from_name
        """Traverse all the objs and groups them by name, save only the selected objs in list.

        Args:
            objs (list): List of the objs

        Returns:
            objs (list): List of selected objs.
        """
        # TODO: automatically got the randomized variables
        selected_objs = []
        select_criterion = {
            "mesh_file": lambda mesh_file: (
                ("standard_fork_scale" in mesh_file)
                or ("standard_spone6_translate" in mesh_file)
                or ("3_center" in mesh_file)
            )
        }
        for obj in objs:
            ret = False
            for attr, criteria in select_criterion.items():
                ret = criteria(getattr(obj, str(attr), None))
                if ret == False:
                    break
            if ret == True:
                selected_objs.append(obj)
        return selected_objs

    def change_vision_config(self, gym_config, vision_config):
        changed_gym_config = change_nested_dict(
            gym_config,
            ("dataset", "robot_meta", "observation", "vision"),
            "update",
            vision_config,
        )
        changed_gym_config = change_nested_dict(
            changed_gym_config, ("dataset", "robot_meta", "min_len_steps"), "update", 0
        )
        return changed_gym_config

    def simplify_action_config(self, action_config):
        """Simplify the action config, as testing the DataDictExtractor doesn't need the full trajectory, save only the first 'edge' of each 'scope', and delete all 'sync'

        Args:
            action_config (Dict): The action config.

        Returns:
            action_config (Dict): The simplified action config
        """
        for scope, edge_list in action_config["edge"].items():
            action_config["edge"][scope] = edge_list[:1]
        action_config["sync"] = {}
        return action_config

    def test_datadictextractor(self, local: bool = False):
        from embodichain import embodichain_dir
        from embodichain.lab.scripts.run_env import (
            make_env,
            generate_and_execute_action_list,
            load_gym_config,
            load_objs,
        )
        import os

        if local:
            save_dir = os.path.join(embodichain_dir, "outputs")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            save_dir = tempfile.mkdtemp()

        obj_conf_path = os.path.join(
            "configs",
            "gym",
            "rearrangement",
            "CobotMagic",
            "object_config.json",
        )
        gym_conf_path = os.path.join(
            "configs",
            "gym",
            "rearrangement",
            "CobotMagic",
            "gym_config.json",
        )
        action_conf_path = os.path.join(
            "configs",
            "gym",
            "rearrangement",
            "CobotMagic",
            "action_config.json",
        )
        input_dict = {
            "render_backend": "egl",
            "debug_mode": False,
            "obj_config": obj_conf_path,
            "gym_config": gym_conf_path,
            "action_config": action_conf_path,
            # "headless": not local,
            "headless": True,
            "enable_rt": False,
            "task_type": "rearrange_tableware_dex_aloha",
            "robot_name": "CobotMagic",
            "save_path": save_dir,
            "save_video": False,
            "online_config": "",
            "gpu_id": 0,
            "warehouse_dir": None,
            "local_dir": "",
        }
        args = dict2args(input_dict)

        gym_config = load_gym_config(args)
        gym_config["dataset"]["save_path"] = save_dir
        gym_config["max_episodes"] = 1
        gym_config["action_config"] = self.simplify_action_config(
            gym_config["action_config"]
        )
        gym_config = self.derandomize_gym_config(gym_config)

        objs = load_objs(args)
        selected_objs = self.derandomize_objs(objs)

        test_vision_configs = [
            {
                "cam_high": [
                    PrivilegeType.MASK.value,
                    Modality.GEOMAP.value,
                    PrivilegeType.EXTEROCEPTION.value,
                ],
                "cam_right_wrist": [
                    PrivilegeType.MASK.value,
                    PrivilegeType.EXTEROCEPTION.value,
                ],
                "cam_left_wrist": [
                    PrivilegeType.MASK.value,
                    PrivilegeType.EXTEROCEPTION.value,
                ],
            },  # full
            {"cam_high": [], "cam_right_wrist": [], "cam_left_wrist": []},  # null
            {
                "cam_high": [PrivilegeType.MASK.value],
                "cam_right_wrist": [PrivilegeType.MASK.value],
                "cam_left_wrist": [PrivilegeType.MASK.value],
            },  # only mask
            {
                "cam_high": [PrivilegeType.EXTEROCEPTION.value],
                "cam_right_wrist": [PrivilegeType.EXTEROCEPTION.value],
                "cam_left_wrist": [PrivilegeType.EXTEROCEPTION.value],
            },  # only exteroception
            {
                "cam_high": [Modality.GEOMAP.value],
                "cam_right_wrist": [],
                "cam_left_wrist": [],
            },  # only geomap
            {
                "cam_high": [Modality.GEOMAP.value, PrivilegeType.EXTEROCEPTION.value],
                "cam_right_wrist": [PrivilegeType.MASK.value],
                "cam_left_wrist": [],
            },  # random
        ]
        for test_vision_config in test_vision_configs:
            gym_config = self.change_vision_config(gym_config, test_vision_config)
            env = make_env(args, selected_objs, gym_config)
            env = NoFailWrapper(env)

            first_obs, info = env.reset(options={"apply_sampler": True})
            (
                action_list,
                obs_list,
                first_obs,
                valid,
            ) = generate_and_execute_action_list(env, 0, first_obs, False)

            ret = DataDictExtractor(env.unwrapped).extract(
                obs_list, action_list, save=False
            )
            env.scene.close_window()
            env.scene.destroy()


if __name__ == "__main__":
    # `unittest.main()` is the standard usage to start testing, here we use a customed
    # TestLoader to keep executing order of functions the same as their writing order

    # unittest.main(testLoader=OrderedTestLoader())

    test = TestDataDictExtractor()
    test.test_datadictextractor(local=True)
