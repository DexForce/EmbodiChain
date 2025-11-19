# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from common import UnittestMetaclass, OrderedTestLoader
from embodichain.utils.utility import dict2args
import tempfile
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

FILE_SERVER = "http://192.168.3.120"


class TestOfflineRunEnv(unittest.TestCase, metaclass=UnittestMetaclass):
    datacenter_backup = Path("/tmp/datacenter_test")
    base_url = "http://192.168.3.120/MixedAI/"

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_offline_run_env(self):
        from embodichain import embodichain_dir
        from embodichain.lab.scripts.run_env import (
            make_env,
            main,
            load_gym_config,
            load_objs,
        )
        import os

        with tempfile.TemporaryDirectory(prefix=self.__class__.__name__) as temp_dir:
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
                "gpu_id": 0,
                "debug_mode": False,
                "obj_config": obj_conf_path,
                "gym_config": gym_conf_path,
                "action_config": action_conf_path,
                "headless": True,
                "enable_rt": False,
                "task_type": "",
                "robot_name": "",
                "save_path": temp_dir,
                "save_video": False,
                "online_config": "",
                "warehouse_dir": None,
                "local_dir": "",
            }
            args = dict2args(input_dict)
            gym_config = load_gym_config(args)
            gym_config["dataset"]["save_path"] = temp_dir
            gym_config["max_episodes"] = 1
            env = make_env(args, load_objs(args), gym_config)
            main(args, env, gym_config)


if __name__ == "__main__":
    # `unittest.main()` is the standard usage to start testing, here we use a customed
    # TestLoader to keep executing order of functions the same as their writing order

    unittest.main(testLoader=OrderedTestLoader())
