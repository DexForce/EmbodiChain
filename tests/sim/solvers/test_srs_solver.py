# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

from __future__ import annotations

import os

import torch

torch._dynamo.config.cache_size_limit = 128  # recompile_limit
import numpy as np
import pytest

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots.dexforce_w1.params import (
    W1ArmKineParams,
)
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmSide,
    DexforceW1Version,
)
from embodichain.lab.sim.solvers.srs_solver import SRSSolver, SRSSolverCfg


class BaseSolverTest:
    def get_arm_config(self):
        return [
            (DexforceW1ArmSide.LEFT, "left_arm"),
            (DexforceW1ArmSide.RIGHT, "right_arm"),
        ]

    def setup_solver(self, solver_type: str, device: str = "cpu"):
        self.solver = {}
        for arm_side, arm_name in self.get_arm_config():
            arm_params = W1ArmKineParams(
                arm_side=arm_side,
                version=DexforceW1Version.V021,
            )
            urdf = get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")

            cfg = SRSSolverCfg()
            cfg.joint_names = [
                f"{'LEFT' if arm_side == DexforceW1ArmSide.LEFT else 'RIGHT'}_J{i + 1}"
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
            cfg.user_qpos_limits = arm_params.qpos_limits
            cfg.T_e_oe = arm_params.T_e_oe
            cfg.T_b_ob = arm_params.T_b_ob
            cfg.link_lengths = arm_params.link_lengths
            cfg.rotation_directions = arm_params.rotation_directions
            cfg.ik_nearest_weight = np.array([2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0])

            self.solver[arm_name] = SRSSolver(cfg=cfg, num_envs=1, device=device)

    def teardown_method(self):
        """Release per-test solver instances and their backend allocations."""
        self.solver.clear()

    @pytest.mark.parametrize(
        "arm_side, arm_name",
        [
            (DexforceW1ArmSide.LEFT, "left_arm"),
            (DexforceW1ArmSide.RIGHT, "right_arm"),
        ],
    )
    def test_ik(self, arm_side: DexforceW1ArmSide, arm_name: str):
        # Test inverse kinematics (IK) with a 1x4x4 homogeneous matrix pose and a joint_seed
        device = self.solver[arm_name].device

        qpos_fk = torch.tensor(
            [[0.0, 0.0, 0.0, -np.pi / 4, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )

        fk_xpos = self.solver[arm_name].get_fk(qpos=qpos_fk)

        _, ik_qpos = self.solver[arm_name].get_ik(fk_xpos, return_all_solutions=False)

        ik_xpos = self.solver[arm_name].get_fk(qpos=ik_qpos[:, 0, :])

        assert torch.allclose(
            fk_xpos, ik_xpos, atol=1e-3, rtol=1e-3
        ), f"FK and IK results do not match for {arm_name}"

    @pytest.mark.parametrize(
        "return_all_solutions, expected_shape",
        [(False, (2, 1, 7)), (True, (2, 0, 7))],
    )
    def test_no_solution_preserves_output_rank(
        self, return_all_solutions: bool, expected_shape: tuple[int, int, int]
    ):
        """Test an entirely failed batch still follows the IK output contract."""
        solver = self.solver[next(iter(self.solver))]
        target_xpos = torch.eye(4, dtype=torch.float32, device=solver.device).repeat(
            2, 1, 1
        )
        target_xpos[:, :3, 3] = torch.tensor(
            [100.0, 100.0, 100.0], dtype=torch.float32, device=solver.device
        )
        qpos_seed = torch.zeros((2, 7), dtype=torch.float32, device=solver.device)

        success, solutions = solver.get_ik(
            target_xpos,
            qpos_seed=qpos_seed,
            return_all_solutions=return_all_solutions,
        )

        assert success.shape == (2,)
        assert not success.any()
        assert solutions.shape == expected_shape

    def test_update_with_robot_limit_intersects_existing_solver_limits(self):
        """Test robot limit sync only tightens solver limits and never widens them."""
        solver_key = next(iter(self.solver))
        solver = self.solver[solver_key]
        solver_limits = solver.get_qpos_limits()

        configured_lower = torch.tensor(
            solver_limits["lower_qpos_limits"],
            dtype=torch.float32,
            device=solver.device,
        )
        configured_upper = torch.tensor(
            solver_limits["upper_qpos_limits"],
            dtype=torch.float32,
            device=solver.device,
        )

        looser_robot_limits = torch.stack(
            (configured_lower - 0.1, configured_upper + 0.1), dim=-1
        )
        solver.update_with_robot_limit(looser_robot_limits)
        looser_sync_limits = solver.get_qpos_limits()
        assert torch.allclose(
            torch.tensor(
                looser_sync_limits["lower_qpos_limits"],
                dtype=torch.float32,
                device=solver.device,
            ),
            configured_lower,
            atol=1e-5,
        ), "FAIL: robot sync widened solver lower_qpos_limits"
        assert torch.allclose(
            torch.tensor(
                looser_sync_limits["upper_qpos_limits"],
                dtype=torch.float32,
                device=solver.device,
            ),
            configured_upper,
            atol=1e-5,
        ), "FAIL: robot sync widened solver upper_qpos_limits"

        margin = torch.minimum(
            torch.full_like(configured_lower, 0.05),
            0.25 * (configured_upper - configured_lower),
        )
        tighter_robot_limits = torch.stack(
            (configured_lower + margin, configured_upper - margin), dim=-1
        )
        solver.update_with_robot_limit(tighter_robot_limits)
        tighter_sync_limits = solver.get_qpos_limits()
        assert torch.allclose(
            torch.tensor(
                tighter_sync_limits["lower_qpos_limits"],
                dtype=torch.float32,
                device=solver.device,
            ),
            tighter_robot_limits[:, 0],
            atol=1e-5,
        ), "FAIL: robot sync did not tighten solver lower_qpos_limits"
        assert torch.allclose(
            torch.tensor(
                tighter_sync_limits["upper_qpos_limits"],
                dtype=torch.float32,
                device=solver.device,
            ),
            tighter_robot_limits[:, 1],
            atol=1e-5,
        ), "FAIL: robot sync did not tighten solver upper_qpos_limits"

    def test_seeded_redundancy_sampling_expands_around_geometric_arm_angle(self):
        """Test redundancy samples expand around the seed's geometric arm angle."""
        solver = self.solver[next(iter(self.solver))]
        solver.cfg.num_samples = 5
        seed = torch.tensor(
            [[0.15, -0.35, 0.40, -0.70, 0.20, 0.30, -0.15]],
            dtype=torch.float32,
            device=solver.device,
        )

        seed_arm_angle = solver.impl._get_seed_arm_angles(seed)
        angles = solver.impl._sample_elbow_angles(seed)

        step = solver.cfg.redundancy_step
        expected = torch.stack(
            (
                seed_arm_angle,
                seed_arm_angle + step,
                seed_arm_angle - step,
            ),
            dim=1,
        )
        expected = torch.remainder(expected + torch.pi, 2.0 * torch.pi) - torch.pi
        assert torch.allclose(angles[:, :3], expected, atol=1e-6)
        assert angles.shape[1] == solver.cfg.num_samples
        assert torch.all(angles >= -torch.pi)
        assert torch.all(angles < torch.pi)
        wrapped_delta = torch.atan2(
            torch.sin(angles[:, :, None] - angles[:, None, :]),
            torch.cos(angles[:, :, None] - angles[:, None, :]),
        ).abs()
        diagonal = torch.eye(
            angles.shape[1], dtype=torch.bool, device=solver.device
        ).unsqueeze(0)
        assert torch.all(wrapped_delta.masked_fill(diagonal, torch.inf) > 1e-6)

    def test_underfilled_seeded_sampling_uses_complete_uniform_grid(self):
        """Test an exhausted radial sequence becomes a gap-free circular grid."""
        solver = self.solver[next(iter(self.solver))]
        solver.cfg.num_samples = 100
        seed = torch.tensor(
            [[0.15, -0.35, 0.40, -0.70, 0.20, 0.30, -0.15]],
            dtype=torch.float32,
            device=solver.device,
        )

        seed_arm_angle = solver.impl._get_seed_arm_angles(seed)
        angles = solver.impl._sample_elbow_angles(seed)
        offsets = (
            torch.remainder(
                angles - seed_arm_angle.unsqueeze(1) + torch.pi,
                2.0 * torch.pi,
            )
            - torch.pi
        )
        sorted_offsets = offsets.sort(dim=1).values
        circular_gaps = torch.cat(
            (
                sorted_offsets[:, 1:] - sorted_offsets[:, :-1],
                sorted_offsets[:, :1] + 2.0 * torch.pi - sorted_offsets[:, -1:],
            ),
            dim=1,
        )

        assert angles.shape == (1, solver.cfg.num_samples)
        assert torch.allclose(angles[:, 0], seed_arm_angle, atol=1e-6)
        assert torch.allclose(
            circular_gaps,
            torch.full_like(circular_gaps, 2.0 * torch.pi / solver.cfg.num_samples),
            atol=1e-6,
        )

    def test_horizontal_shoulder_wrist_seed_recovers_its_fk_pose(self):
        """Test horizontal shoulder-wrist geometry retains its shoulder azimuth."""
        seed = torch.tensor(
            [[0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )

        for solver in self.solver.values():
            device_seed = seed.to(solver.device)
            target = solver.get_fk(device_seed)
            arm_angle = solver.impl._get_seed_arm_angles(device_seed)
            success, solution = solver.get_ik(target, device_seed)

            assert torch.all(torch.isfinite(arm_angle))
            assert torch.all(success)
            assert torch.allclose(
                solver.get_fk(solution[:, 0]), target, atol=1e-4, rtol=1e-4
            )

    def test_redundancy_step_larger_than_pi_is_rejected(self):
        """Test invalid seeded-search steps fail instead of silently using one sample."""
        solver = self.solver[next(iter(self.solver))]
        cfg = solver.cfg.copy()
        cfg.redundancy_step = np.pi + 1e-3

        with pytest.raises(ValueError, match=r"range \(0, pi\]"):
            cfg.init_solver(num_envs=1, device=solver.device)

    def test_periodic_joint_values_wrap_inside_limits_near_seed(self):
        """Test equivalent revolute values are retained instead of rejected."""
        solver = self.solver[next(iter(self.solver))]
        joints = np.array([-3.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        limits = np.array([[3.0, 3.2]] + [[-1.0, 1.0]] * 6)
        seed = np.array([3.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        wrapped = solver.impl._wrap_to_limits(joints, limits, seed)

        assert wrapped is not None
        assert np.isclose(wrapped[0], -3.2 + 2.0 * np.pi)
        assert np.all(wrapped >= limits[:, 0])
        assert np.all(wrapped <= limits[:, 1])

    def test_all_solution_deduplication_is_periodic_and_order_preserving(self):
        """Test deduplication retains the first periodic representative."""
        solver = self.solver[next(iter(self.solver))]
        first = torch.tensor(
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], device=solver.device
        )
        second = torch.tensor(
            [-0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4], device=solver.device
        )
        solutions = torch.stack(
            (first, first + 2.0 * torch.pi, second, second - 2.0 * torch.pi)
        )

        unique = solver.impl._deduplicate_solutions(solutions)

        assert unique.shape == (2, 7)
        assert torch.allclose(unique[0], first)
        assert torch.allclose(unique[1], second)

    def test_all_solution_deduplication_compares_retained_representatives(self):
        """Test a discarded chain neighbor cannot remove a later unique row."""
        solver = self.solver[next(iter(self.solver))]
        tolerance = 1e-5
        solutions = torch.zeros((3, 7), device=solver.device)
        solutions[:, 0] = torch.tensor(
            [0.0, 0.9 * tolerance, 1.8 * tolerance],
            device=solver.device,
        )

        unique = solver.impl._deduplicate_solutions(solutions, tolerance=tolerance)

        assert unique.shape == (2, 7)
        assert torch.equal(unique[0], solutions[0])
        assert torch.equal(unique[1], solutions[2])

    def test_runtime_tcp_and_weight_updates_reach_backend(self):
        """Test mutable solver settings synchronize analytical backend caches."""
        solver = self.solver[next(iter(self.solver))]
        tcp = np.eye(4)
        tcp[:3, 3] = np.array([0.01, -0.02, 0.03])
        weights = np.arange(1.0, 8.0)

        solver.set_tcp(tcp)
        assert np.allclose(solver.impl.tcp_inv_np, np.linalg.inv(tcp))

        assert solver.set_ik_nearest_weight(weights)
        assert np.allclose(solver.cfg.ik_nearest_weight, weights)
        if solver.device.type == "cpu":
            assert torch.allclose(
                solver.impl.ik_nearest_weight_tensor.cpu(),
                torch.from_numpy(weights).float(),
            )


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
            version=DexforceW1Version.V021,
        )
        w1_right_arm_params = W1ArmKineParams(
            arm_side=DexforceW1ArmSide.RIGHT,
            version=DexforceW1Version.V021,
        )

        # Robot configuration dictionary
        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "left_arm": [f"LEFT_J{i + 1}" for i in range(7)],
                "right_arm": [f"RIGHT_J{i + 1}" for i in range(7)],
                "torso": ["ANKLE", "KNEE", "BUTTOCK", "WAIST"],
                "head": [f"NECK{i + 1}" for i in range(2)],
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
        SimulationManager.flush_cleanup_queue()


class TestSRSCPUSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_solver(solver_type="SRSSolver", device="cpu")


class TestSRSCUDASolver(BaseSolverTest):
    def setup_method(self):
        self.setup_solver(solver_type="SRSSolver", device="cuda")

    def test_cpu_cuda_backend_parity(self):
        """Test CPU and CUDA select equivalent solutions for identical inputs."""
        sample_qpos = torch.tensor(
            [
                [0.15, -0.35, 0.25, -0.70, 0.20, 0.30, -0.15],
                [-0.20, 0.25, -0.30, -0.55, 0.35, -0.20, 0.10],
                [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0, 0.0],
                [np.pi / 6, 0.0, 0.0, -np.pi / 2, 0.4, 0.0, np.pi / 6],
            ],
            dtype=torch.float32,
        )

        for cuda_solver in self.solver.values():
            cpu_solver = cuda_solver.cfg.init_solver(
                num_envs=sample_qpos.shape[0], device=torch.device("cpu")
            )
            cuda_seed = sample_qpos.to(cuda_solver.device)
            target = cuda_solver.get_fk(cuda_seed)

            cuda_arm_angle = cuda_solver.impl._get_seed_arm_angles(cuda_seed)
            cpu_arm_angle = cpu_solver.impl._get_seed_arm_angles(sample_qpos)
            arm_angle_delta = torch.atan2(
                torch.sin(cuda_arm_angle.cpu() - cpu_arm_angle.cpu()),
                torch.cos(cuda_arm_angle.cpu() - cpu_arm_angle.cpu()),
            )
            assert torch.allclose(
                arm_angle_delta,
                torch.zeros_like(arm_angle_delta),
                atol=1e-4,
                rtol=1e-4,
            )

            cuda_success, cuda_qpos = cuda_solver.get_ik(target, cuda_seed)
            cpu_success, cpu_qpos = cpu_solver.get_ik(target.cpu(), sample_qpos)

            assert torch.equal(cuda_success.cpu(), cpu_success.cpu())
            assert torch.all(cuda_success)

            cuda_solution = cuda_qpos[:, 0].cpu()
            cpu_solution = cpu_qpos[:, 0].cpu()
            wrapped_delta = torch.atan2(
                torch.sin(cuda_solution - cpu_solution),
                torch.cos(cuda_solution - cpu_solution),
            )
            dh_params = np.asarray(cuda_solver.cfg.dh_params)
            directions = np.asarray(cuda_solver.cfg.rotation_directions)
            model_q2 = sample_qpos[:, 1].numpy() * directions[1] + dh_params[1, 3]
            model_q6 = sample_qpos[:, 5].numpy() * directions[5] + dh_params[5, 3]
            singular = torch.from_numpy(
                (np.abs(np.sin(model_q2)) <= 1e-6) | (np.abs(np.sin(model_q6)) <= 1e-6)
            )
            assert torch.allclose(
                wrapped_delta[~singular],
                torch.zeros_like(wrapped_delta[~singular]),
                atol=1e-4,
                rtol=1e-4,
            )

            # At an Euler singularity, coupled joints can have substantially
            # different parameterizations for the same pose. The redundancy
            # step bounds arm-angle sampling, not individual joint deltas, so
            # singular solutions are compared through FK below.
            cuda_reconstructed = cuda_solver.get_fk(cuda_qpos[:, 0]).cpu()
            cpu_reconstructed = cpu_solver.get_fk(cpu_qpos[:, 0]).cpu()
            assert torch.allclose(
                cuda_reconstructed, target.cpu(), atol=1e-4, rtol=1e-4
            )
            assert torch.allclose(cpu_reconstructed, target.cpu(), atol=1e-4, rtol=1e-4)
            assert torch.allclose(
                cuda_reconstructed, cpu_reconstructed, atol=1e-4, rtol=1e-4
            )


class TestSRSCPURobotSolver(BaseRobotSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="SRSSolver", device="cpu")


class TestSRSCUDARobotSolver(BaseRobotSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="SRSSolver", device="cuda")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", __file__]
    pytest.main(pytest_args)
