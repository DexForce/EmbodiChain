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

import os
import torch
import pytest
import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.data import get_data_path

from embodichain.lab.sim.solvers.srs_solver import SRSSolver, SRSSolverCfg
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1ArmKind,
    DexforceW1Version,
)
from embodichain.lab.sim.robots.dexforce_w1.params import (
    W1ArmKineParams,
)


class BaseSolverTest:
    solver = {}

    def get_arm_config(self):
        return [
            (DexforceW1ArmSide.LEFT, DexforceW1ArmKind.ANTHROPOMORPHIC, "left_arm"),
            (DexforceW1ArmSide.RIGHT, DexforceW1ArmKind.ANTHROPOMORPHIC, "right_arm"),
            (DexforceW1ArmSide.LEFT, DexforceW1ArmKind.INDUSTRIAL, "left_arm"),
            (DexforceW1ArmSide.RIGHT, DexforceW1ArmKind.INDUSTRIAL, "right_arm"),
        ]

    def setup_solver(self, solver_type: str, device: str = "cpu"):
        for arm_side, arm_kind, arm_name in self.get_arm_config():
            arm_params = W1ArmKineParams(
                arm_side=arm_side,
                arm_kind=arm_kind,
                version=DexforceW1Version.V021,
            )
            if arm_kind == DexforceW1ArmKind.ANTHROPOMORPHIC:
                urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
            else:
                urdf = get_data_path("DexforceW1V021/DexforceW1_v02_2.urdf")

            cfg = SRSSolverCfg()
            cfg.joint_names = [
                f"{'LEFT' if arm_side == DexforceW1ArmSide.LEFT else 'RIGHT'}_J{i+1}"
                for i in range(7)
            ]
            cfg.end_link_name = (
                "left_ee" if arm_side == DexforceW1ArmSide.LEFT else "right_ee"
            )
            cfg.root_link_name = (
                "left_arm_base"
                if arm_side == DexforceW1ArmSide.LEFT
                else "right_arm_base"
            )
            cfg.urdf_path = urdf
            cfg.dh_params = arm_params.dh_params
            cfg.qpos_limits = arm_params.qpos_limits
            cfg.T_e_oe = arm_params.T_e_oe
            cfg.T_b_ob = arm_params.T_b_ob
            cfg.link_lengths = arm_params.link_lengths
            cfg.rotation_directions = arm_params.rotation_directions

            solver_key = f"{arm_name}_{arm_kind.name}"
            self.solver[solver_key] = SRSSolver(cfg=cfg, num_envs=1, device=device)

    @pytest.mark.parametrize(
        "arm_side, arm_kind, arm_name",
        [
            (DexforceW1ArmSide.LEFT, DexforceW1ArmKind.ANTHROPOMORPHIC, "left_arm"),
            (DexforceW1ArmSide.RIGHT, DexforceW1ArmKind.ANTHROPOMORPHIC, "right_arm"),
            (DexforceW1ArmSide.LEFT, DexforceW1ArmKind.INDUSTRIAL, "left_arm"),
            (DexforceW1ArmSide.RIGHT, DexforceW1ArmKind.INDUSTRIAL, "right_arm"),
        ],
    )
    def test_ik(
        self, arm_side: DexforceW1ArmSide, arm_kind: DexforceW1ArmKind, arm_name: str
    ):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed
        solver_key = f"{arm_name}_{arm_kind.name}"
        device = self.solver[solver_key].device

        qpos_fk = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 4, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )

        fk_xpos = self.solver[solver_key].get_fk(qpos=qpos_fk)

        _, ik_qpos = self.solver[solver_key].get_ik(fk_xpos, return_all_solutions=False)

        ik_xpos = self.solver[solver_key].get_fk(qpos=ik_qpos[:, 0, :])

        assert torch.allclose(
            fk_xpos, ik_xpos, atol=1e-3, rtol=1e-3
        ), f"FK and IK results do not match for {solver_key}"

    @classmethod
    def teardown_class(cls):
        if cls.solver is not None:
            try:
                del cls.solver
                print("solver destroyed successfully")
            except Exception as e:
                print(f"Error during solver destruction: {e}")


# Base test class for CPU and CUDA
class BaseRobotSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, solver_type: str, device: str = "cpu"):
        # Set up simulation with specified device (CPU or CUDA)
        config = SimulationManagerCfg(headless=True, sim_device=device)
        self.sim = SimulationManager(config)

        # Load robot URDF file
        urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")
        assert os.path.isfile(urdf)

        w1_left_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.LEFT,
            arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
            version=DexforceW1Version.V021,
        )
        w1_right_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.RIGHT,
            arm_kind=DexforceW1ArmKind.ANTHROPOMORPHIC,
            version=DexforceW1Version.V021,
        )

        # Robot configuration dictionary
        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "left_arm": [f"LEFT_J{i+1}" for i in range(7)],
                "right_arm": [f"RIGHT_J{i+1}" for i in range(7)],
                "torso": ["ANKLE", "KNEE", "BUTTOCK", "WAIST"],
                "head": [f"NECK{i+1}" for i in range(2)],
            },
            "drive_pros": {
                "stiffness": {
                    "LEFT_J[1-7]": 1e4,
                    "RIGHT_J[1-7]": 1e4,
                    "ANKLE": 1e7,
                    "KNEE": 1e7,
                    "BUTTOCK": 1e7,
                    "WAIST": 1e7,
                },
                "damping": {
                    "LEFT_J[1-7]": 1e3,
                    "RIGHT_J[1-7]": 1e3,
                    "ANKLE": 1e4,
                    "KNEE": 1e4,
                    "BUTTOCK": 1e4,
                    "WAIST": 1e4,
                },
                "max_effort": {
                    "LEFT_J[1-7]": 1e5,
                    "RIGHT_J[1-7]": 1e5,
                    "ANKLE": 1e10,
                    "KNEE": 1e10,
                    "BUTTOCK": 1e10,
                    "WAIST": 1e10,
                },
            },
            "attrs": {
                "mass": 1e-1,
                "static_friction": 0.95,
                "dynamic_friction": 0.9,
                "linear_damping": 0.7,
                "angular_damping": 0.7,
                "max_depenetration_velocity": 10.0,
                "min_position_iters": 32,
                "min_velocity_iters": 8,
            },
            "solver_cfg": {
                "left_arm": {
                    "class_type": solver_type,
                    "end_link_name": "left_ee",
                    "root_link_name": "left_arm_base",
                    "dh_params": w1_left_arm_params.dh_params,
                    "qpos_limits": w1_left_arm_params.qpos_limits,
                    "T_b_ob": w1_right_arm_params.T_b_ob,
                    "T_e_oe": w1_left_arm_params.T_e_oe,
                    "link_lengths": w1_left_arm_params.link_lengths,
                    "rotation_directions": w1_left_arm_params.rotation_directions,
                },
                "right_arm": {
                    "class_type": solver_type,
                    "end_link_name": "right_ee",
                    "root_link_name": "right_arm_base",
                    "dh_params": w1_right_arm_params.dh_params,
                    "qpos_limits": w1_right_arm_params.qpos_limits,
                    "T_b_ob": w1_right_arm_params.T_b_ob,
                    "T_e_oe": w1_right_arm_params.T_e_oe,
                    "link_lengths": w1_right_arm_params.link_lengths,
                    "rotation_directions": w1_right_arm_params.rotation_directions,
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

        # Wait for robot to stabilize.
        self.sim.update(step=100)

    @pytest.mark.parametrize("arm_name", ["left_arm", "right_arm"])
    def test_robot_ik(self, arm_name: str):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed

        qpos_fk = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 4, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=self.robot.device,
        )

        fk_xpos = self.robot.compute_fk(qpos=qpos_fk, name=arm_name, to_matrix=True)

        res, ik_qpos = self.robot.compute_ik(pose=fk_xpos, name=arm_name)

        if ik_qpos.dim() == 3:
            ik_xpos = self.robot.compute_fk(
                qpos=ik_qpos[0][0], name=arm_name, to_matrix=True
            )
        else:
            ik_xpos = self.robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)

        assert torch.allclose(
            fk_xpos, ik_xpos, atol=1e-4, rtol=1e-4
        ), f"FK and IK results do not match for {arm_name}"

        # test for failed xpos
        invalid_pose = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 10.0],
                    [0.0, 1.0, 0.0, 10.0],
                    [0.0, 0.0, 1.0, 10.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
            device=self.robot.device,
        )
        res, ik_qpos = self.robot.compute_ik(
            pose=invalid_pose, joint_seed=ik_qpos, name=arm_name
        )
        dof = ik_qpos.shape[-1]
        assert res[0] == False
        assert ik_qpos.shape == (1, dof)

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestSRSCPUSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_solver(solver_type="SRSSolver", device="cpu")


class TestSRSCUDASolver(BaseSolverTest):
    def setup_method(self):
        self.setup_solver(solver_type="SRSSolver", device="cuda")


class TestSRSCPURobotSolver(BaseRobotSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="SRSSolver", device="cpu")


@pytest.mark.skip(reason="Skipping CUDA tests temporarily")
class TestSRSCUDARobotSolver(BaseRobotSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="SRSSolver", device="cuda")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", __file__]
    pytest.main(pytest_args)
