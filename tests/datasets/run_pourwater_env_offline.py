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

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from common import UnittestMetaclass, OrderedTestLoader

import os
import tempfile
import gymnasium
from pathlib import Path

from embodichain.utils.utility import dict2args
from embodichain.utils.utility import load_json
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import config_to_cfg


class TestPourWaterOfflineRunEnv(unittest.TestCase, metaclass=UnittestMetaclass):
    datacenter_backup = Path("/tmp/datacenter_test")

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_offline_run_env(self):
        from embodichain.lab.scripts.run_env import main
        import os

        with tempfile.TemporaryDirectory(prefix=self.__class__.__name__) as temp_dir:
            gym_conf_path = os.path.join(
                "configs",
                "gym",
                "pour_water",
                "gym_config.json",
            )
            action_conf_path = os.path.join(
                "configs",
                "gym",
                "pour_water",
                "action_config.json",
            )
            input_dict = {
                "num_envs": 1,  # TODO: change it to >1 as v3 supports it. but now CobotMagic use cpu-OPWSolver. Wait @Chenjian for gpu version.
                "device": "cpu",  # TODO: test both cpu and cuda device
                "headless": True,
                "enable_rt": False,
                "gpu_id": 0,
                "save_video": False,
                "save_path": temp_dir,
                "debug_mode": False,
                "filter_visual_rand": False,
                "online_config": "",
                "gym_config": gym_conf_path,
                "action_config": action_conf_path,
            }
            args = dict2args(input_dict)
            gym_config = load_json(args.gym_config)
            gym_config["env"]["dataset"]["save_path"] = temp_dir
            gym_config["max_episodes"] = 1

            cfg: EmbodiedEnvCfg = config_to_cfg(gym_config)
            cfg.filter_visual_rand = args.filter_visual_rand

            action_config = {}
            if args.action_config is not None:
                action_config = load_json(args.action_config)
                action_config["action_config"] = action_config

            cfg.num_envs = args.num_envs
            cfg.sim_cfg = SimulationManagerCfg(
                headless=args.headless,
                sim_device=args.device,
                enable_rt=args.enable_rt,
                gpu_id=args.gpu_id,
            )

            env = gymnasium.make(id=gym_config["id"], cfg=cfg, **action_config)
            main(args, env, gym_config)


if __name__ == "__main__":
    # `unittest.main()` is the standard usage to start testing, here we use a customed
    # TestLoader to keep executing order of functions the same as their writing order

    unittest.main(testLoader=OrderedTestLoader())
