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
import time
import torch
import pytest
import numpy as np
from copy import deepcopy
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import CobotMagicCfg

from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.planners.motion_generator import MotionGenerator


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]
        return tensor.numpy()
    return np.array(tensor)


class BaseTestMotionGenerator(object):
    @classmethod
    def setup_class(cls):
        cls.config = SimulationManagerCfg(headless=True, sim_device="cpu")
        cls.robot_sim = SimulationManager(cls.config)
        cls.robot_sim.set_manual_update(False)

        cfg_dict = {
            "uid": "CobotMagic",
            "init_pos": [0.0, 0.0, 0.7775],
            "init_qpos": [
                -0.3,
                0.3,
                1.0,
                1.0,
                -1.2,
                -1.2,
                0.0,
                0.0,
                0.6,
                0.6,
                0.0,
                0.0,
                0.05,
                0.05,
                0.05,
                0.05,
            ],
            "solver_cfg": {
                "left_arm": {
                    "class_type": "OPWSolver",
                    "end_link_name": "left_link6",
                    "root_link_name": "left_arm_base",
                    "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
                },
                "right_arm": {
                    "class_type": "OPWSolver",
                    "end_link_name": "right_link6",
                    "root_link_name": "right_arm_base",
                    "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
                },
            },
        }

        cls.robot: Robot = cls.robot_sim.add_robot(
            cfg=CobotMagicCfg.from_dict(cfg_dict)
        )

        cls.arm_name = "left_arm"

        cls.motion_gen = MotionGenerator(
            robot=cls.robot,
            uid=cls.arm_name,
            planner_type="toppra",
            default_velocity=0.2,
            default_acceleration=0.5,
        )

        # Test data for trajectory generation
        qpos_fk = torch.tensor(
            [[0.0, np.pi / 4, -np.pi / 4, 0.0, np.pi / 4, 0.0]], dtype=torch.float32
        )
        xpos_begin = cls.robot.compute_fk(
            name=cls.arm_name, qpos=qpos_fk, to_matrix=True
        )
        xpos_mid = deepcopy(xpos_begin)
        xpos_mid[0, 2, 3] -= 0.1  # Move down by 0.1m in Z direction
        xpos_final = deepcopy(xpos_mid)
        xpos_final[0, 0, 3] += 0.2  # Move forward by 0.2m in X direction

        qpos_begin = cls.robot.compute_ik(pose=xpos_begin, name=cls.arm_name)[1][0]
        qpos_mid = cls.robot.compute_ik(pose=xpos_mid, name=cls.arm_name)[1][0]
        qpos_final = cls.robot.compute_ik(pose=xpos_final, name=cls.arm_name)[1][0]

        cls.qpos_list = [qpos_begin, qpos_mid, qpos_final]
        cls.xpos_list = [
            xpos_begin[0].numpy(),
            xpos_mid[0].numpy(),
            xpos_final[0].numpy(),
        ]

        cls.sample_num = 20

    def get_joint_ids(self):
        return self.robot.get_joint_ids(self.arm_name)

    def get_current_qpos(self):
        qpos_tensor = self.robot.get_qpos()
        if qpos_tensor.ndim == 2 and qpos_tensor.shape[0] == 1:
            qpos_tensor = qpos_tensor[0]
        return qpos_tensor[self.get_joint_ids()].cpu()

    def verify_final_xpos(self, expected_xpos, decimal=5e-3):
        final_xpos = self.robot.compute_fk(
            qpos=self.get_current_qpos(), name=self.arm_name, to_matrix=True
        )
        np.testing.assert_array_almost_equal(
            to_numpy(final_xpos)[:3, 3],
            to_numpy(expected_xpos)[:3, 3],
            decimal=decimal,
            err_msg=f"Expected: {to_numpy(expected_xpos)[:3, 3]}, Got: {to_numpy(final_xpos)[:3, 3]}",
        )

    def _execute_trajectory(self, qpos_list, forward=True, delay=0.01):
        indices = (
            range(len(qpos_list)) if forward else range(len(qpos_list) - 1, -1, -1)
        )
        for i in indices:
            self.robot.set_qpos(qpos=qpos_list[i], joint_ids=self.get_joint_ids())
            time.sleep(delay)
        time.sleep(delay * 2)

    @classmethod
    def teardown_class(cls):
        try:
            cls.robot_sim.destroy()
            print("robot_sim destroyed successfully")
        except Exception as e:
            print(f"Error during robot_sim.destroy(): {e}")

    def _execute_forward_trajectory(self, robot, qpos_list, delay=0.1):
        """Helper method to execute trajectory"""
        # Forward
        for q in qpos_list:
            robot.set_qpos(qpos=q, joint_ids=self.robot.get_joint_ids(self.arm_name))
            time.sleep(delay)
        time.sleep(delay * 5)

    def _execute_backward_trajectory(self, robot, qpos_list, delay=0.1):
        """Helper method to execute trajectory"""
        # Backward
        for q in qpos_list[::-1]:
            robot.set_qpos(qpos=q, joint_ids=self.robot.get_joint_ids(self.arm_name))
            time.sleep(delay)
        time.sleep(delay * 5)


class TestMotionGenerator(BaseTestMotionGenerator):
    """Test suite for MotionGenerator trajectory generation"""

    @pytest.mark.parametrize("is_linear", [True, False])
    def test_create_trajectory_with_xpos(self, is_linear):
        """Test trajectory generation with cartesian positions"""
        self.robot.set_qpos(qpos=self.qpos_list[0], joint_ids=self.get_joint_ids())
        time.sleep(0.2)
        out_qpos_list, out_xpos_list = self.motion_gen.create_discrete_trajectory(
            xpos_list=self.xpos_list,
            is_use_current_qpos=True,
            sample_num=self.sample_num,
            is_linear=is_linear,
            sample_method=TrajectorySampleMethod.QUANTITY,
            qpos_seed=self.qpos_list[0],
        )
        out_qpos_list = to_numpy(out_qpos_list)
        assert (
            len(out_qpos_list) == self.sample_num
        ), f"Sample number mismatch: {len(out_qpos_list)} != {self.sample_num}"
        np.testing.assert_array_almost_equal(
            out_xpos_list[-1], self.xpos_list[-1], decimal=3
        )
        self._execute_trajectory(out_qpos_list, forward=True)
        self.verify_final_xpos(self.xpos_list[-1])
        self._execute_trajectory(out_qpos_list, forward=False)
        self.verify_final_xpos(self.xpos_list[0])

    @pytest.mark.parametrize("is_linear", [True, False])
    def test_create_trajectory_with_qpos(self, is_linear):
        """Test trajectory generation with joint positions"""
        self.robot.set_qpos(qpos=self.qpos_list[0], joint_ids=self.get_joint_ids())
        time.sleep(0.05)
        qpos_list_in = [qpos.to("cpu").numpy() for qpos in self.qpos_list]
        out_qpos_list, out_xpos_list = self.motion_gen.create_discrete_trajectory(
            qpos_list=qpos_list_in,
            sample_num=self.sample_num,
            is_linear=False,
            sample_method=TrajectorySampleMethod.QUANTITY,
            qpos_seed=self.qpos_list[0],
        )
        out_qpos_list = to_numpy(out_qpos_list)
        assert (
            len(out_qpos_list) == self.sample_num
        ), f"Sample number mismatch: {len(out_qpos_list)} != {self.sample_num}"
        np.testing.assert_array_almost_equal(
            out_qpos_list[-1], self.qpos_list[-1], decimal=3
        )
        self._execute_trajectory(out_qpos_list, forward=True)
        self.verify_final_xpos(self.xpos_list[-1])
        self._execute_trajectory(out_qpos_list, forward=False)
        self.verify_final_xpos(self.xpos_list[0])

    @pytest.mark.parametrize("xpos_or_qpos", ["xpos", "qpos"])
    def test_estimate_trajectory_sample_count(self, xpos_or_qpos: str):
        """Test estimation of trajectory sample count"""
        if xpos_or_qpos == "xpos":
            estimated_num = self.motion_gen.estimate_trajectory_sample_count(
                xpos_list=self.xpos_list,
                step_size=0.01,
                angle_step=np.pi / 90,
            )
        else:
            estimated_num = self.motion_gen.estimate_trajectory_sample_count(
                qpos_list=self.qpos_list,
                step_size=0.01,
                angle_step=np.pi / 90,
            )
        assert (estimated_num - 30) < 2, "Estimated sample count failed"


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    pytest_args = ["-v", "-s", __file__]
    pytest.main(pytest_args)
