# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
import os
import tempfile
from pathlib import Path

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from common import UnittestMetaclass

from embodichain.utils.utility import dict2args, change_nested_dict
from embodichain.lab.gym.envs import AlignDataWrapper


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

FILE_SERVER = "http://192.168.3.120"


class TestRearrangmentAlignMaster(unittest.TestCase, metaclass=UnittestMetaclass):
    datacenter_backup = Path("/tmp/datacenter_test")
    base_url = "http://192.168.3.120/MixedAI/"

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

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
            ("robot_action", "rotation_range"): [[0, 0], [0, 0], [0, 0]],
            ("robot_action", "exchange_prob"): 0.0,
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
                ("3_center" in mesh_file)
                or ("standard_spone6_translate" in mesh_file)
                or ("standard_fork_scale" in mesh_file)
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

    def test_offline_run_env(self, local: bool):
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
            embodichain_dir,
            "lab",
            "configs",
            "rearrangement",
            "CobotMagic",
            "object_config.json",
        )
        gym_conf_path = os.path.join(
            embodichain_dir,
            "lab",
            "configs",
            "rearrangement",
            "CobotMagic",
            "gym_config.json",
        )
        action_conf_path = os.path.join(
            embodichain_dir,
            "lab",
            "configs",
            "rearrangement",
            "CobotMagic",
            "action_config.json",
        )
        online_conf_path = os.path.join(
            embodichain_dir,
            "lab",
            "data_engine",
            "online",
            "online_config.json",
        )
        input_dict = {
            "render_backend": "egl",
            "debug_mode": False,
            "obj_config": obj_conf_path,
            "gym_config": gym_conf_path,
            "action_config": action_conf_path,
            "headless": True,
            "enable_rt": False,
            "task_type": "",
            "robot_name": "",
            "save_path": save_dir,
            "save_video": False,
            "online_config": online_conf_path,
        }
        args = dict2args(input_dict)
        gym_config = load_gym_config(args)
        gym_config["dataset"]["save_path"] = save_dir
        gym_config["max_episodes"] = 1
        gym_config = self.derandomize_gym_config(gym_config)

        objs = load_objs(args)
        selected_objs = self.derandomize_objs(objs)

        env = make_env(args, selected_objs, gym_config)
        align_criteria = {"obs": {"name": "none"}, "action": {"name": "none"}}
        env = AlignDataWrapper(env, align_criteria, vis=local)

        first_obs, info = env.reset(options={"apply_sampler": True})
        action_list, obs_list, first_obs, valid = generate_and_execute_action_list(
            env, 0, first_obs, False
        )

        assert valid, f"action_list or obs_list generation fails for env {env.spec.id}."

        for align_task in ["align_master", "align_reproduce"]:
            align_valid_dict = env.align_data(obs_list, action_list, align_task)
            assert all(
                align_valid_dict.values()
            ), f"{align_task} comparison fails for env {env.spec.id}."


if __name__ == "__main__":
    test = TestRearrangmentAlignMaster()
    test.test_offline_run_env(local=True)
