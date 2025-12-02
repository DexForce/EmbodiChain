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

from embodichain.lab.gym.envs.action_bank.configurable_action import (
    ActionBank,
    tag_node,
    tag_edge,
    get_func_tag,
)
import numpy as np
import os
from typing import Dict, Tuple, Union, List, Callable
import unittest
from embodichain.utils.utility import load_json
import inspect

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from common import UnittestMetaclass, OrderedTestLoader


class FakePourwaterEnv:
    def __init__(self) -> None:
        pass


class FakePourwaterActionBank(ActionBank):
    @staticmethod
    @tag_node
    def A(env: FakePourwaterEnv):
        env.A = "A"
        return True

    @staticmethod
    @tag_node
    def B(env: FakePourwaterEnv):
        env.B = env.A
        return True

    @staticmethod
    @tag_node
    def C(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_node
    def D(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_node
    def a(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_node
    def b(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_node
    def aa(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_node
    def bb(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_node
    def cc(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_node
    def dd(env: FakePourwaterEnv):
        return True

    @staticmethod
    @tag_edge
    def init_to_pre1(env: FakePourwaterEnv, **kwargs):
        return np.random.rand(6, 1)

    @staticmethod
    @tag_edge
    def grasp_to_move(env: FakePourwaterEnv, **kwargs):
        return np.random.rand(6, 2)

    @staticmethod
    @tag_edge
    def move_to_rotation(env: FakePourwaterEnv, **kwargs):
        env.move_to_rotation = np.random.rand(6, 3)
        return env.move_to_rotation

    @staticmethod
    @tag_edge
    def rotation_back_to_move(env: FakePourwaterEnv, **kwargs):
        return np.random.rand(6, 4)

    @staticmethod
    @tag_edge
    def init_to_monitor(env: FakePourwaterEnv, **kwargs):
        return np.random.rand(6, 1)

    @staticmethod
    @tag_edge
    def left_arm_go_back(env: FakePourwaterEnv, **kwargs):
        return np.random.rand(6, 2)

    @staticmethod
    @tag_edge
    def lopen(env: FakePourwaterEnv, **kwargs):
        return np.random.rand(1, 10)

    @staticmethod
    @tag_edge
    def ropen(env: FakePourwaterEnv, **kwargs) -> np.ndarray:
        return np.random.rand(1, 10)


class TestActionBank(unittest.TestCase, metaclass=UnittestMetaclass):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_simple(self):
        class FakeFunctions:
            def __init__(
                self,
            ) -> None:
                self.dummy_function = lambda: 1

            def get_functions(self, names: List[str]) -> Dict[str, Callable]:
                """
                Returns a dictionary of dummy functions for the given names.
                """
                return {name: self.dummy_function for name in names}

        funcs = FakeFunctions()
        conf = load_json(os.path.join("configs", "gym", "action_bank", "conf.json"))
        action_bank = ActionBank(conf)
        action_bank.parse_network(
            funcs.get_functions(
                ["A", "B", "C", "D", "a", "b", "aa", "bb", "cc", "dd"],
            ),
            funcs.get_functions(
                [
                    "init_to_pre1",
                    "grasp_to_move",
                    "move_to_rotation",
                    "rotation_back_to_move",
                    "move_back_to_grasp",
                    "grasp_back_to_pre1",
                    "init_to_monitor",
                    "left_arm_go_back",
                    "lopen",
                    "ropen",
                ],
            ),
            vis_graph=False,
        )

    def test_hook_and_gantt(self):
        conf = load_json(os.path.join("configs", "gym", "action_bank", "conf.json"))
        action_bank = FakePourwaterActionBank(conf)
        print(get_func_tag("node").functions[action_bank.__class__.__name__])
        _, jobs_data, jobkey2index = action_bank.parse_network(
            get_func_tag("node").functions[action_bank.__class__.__name__],
            get_func_tag("edge").functions[action_bank.__class__.__name__],
            vis_graph=False,
        )

        action_bank.gantt(jobs_data, jobkey2index, vis=False)

    def test_create_action_list(self):
        np.random.seed(0)
        conf = load_json(os.path.join("configs", "gym", "action_bank", "conf.json"))
        action_bank = FakePourwaterActionBank(conf)
        graph_compose, jobs_data, jobkey2index = action_bank.parse_network(
            get_func_tag("node").functions[action_bank.__class__.__name__],
            get_func_tag("edge").functions[action_bank.__class__.__name__],
            vis_graph=False,
        )
        env = FakePourwaterEnv()
        packages = action_bank.gantt(jobs_data, jobkey2index, vis=False)
        ret = action_bank.create_action_list(env, graph_compose, packages)

        assert (
            np.linalg.norm(ret["left_arm"][:, 3:10] - ret["left_arm"][:, 3:4]) <= 1e-6
        )  # padding.
        assert (
            np.linalg.norm(ret["right_arm"][:, 3:6] - env.move_to_rotation) <= 1e-6
        )  # rotation_back_to_move

    def test_bad_conf(self):
        np.random.seed(0)
        conf = load_json(os.path.join("configs", "gym", "action_bank", "conf.json"))
        conf["node"]["right_arm"] = [
            {
                "init_to_pre1": {
                    "src": "home_qpos",
                    "sink": "bottle_pre1_pose",
                    "duration": 1,
                    "kwargs": {},
                },
                "grasp_to_move": {
                    "src": "bottle_pre1_pose",
                    "sink": "bottle_grasp",
                    "duration": 2,
                    "kwargs": {},
                },
            }
        ]
        action_bank = FakePourwaterActionBank(conf)
        self.assertRaises(
            ValueError,
            action_bank.parse_network,
            get_func_tag("node").functions[action_bank.__class__.__name__],
            get_func_tag("edge").functions[action_bank.__class__.__name__],
            vis_graph=False,
        )


if __name__ == "__main__":
    # `unittest.main()` is the standard usage to start testing, here we use a customed
    # TestLoader to keep executing order of functions the same as their writing order

    unittest.main(testLoader=OrderedTestLoader())
