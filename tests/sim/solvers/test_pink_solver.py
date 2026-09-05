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
from pathlib import Path

import numpy as np
import pytest
import torch

pin = pytest.importorskip("pinocchio")
pink = pytest.importorskip("pink")

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RobotCfg, RenderCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.solvers.null_space_posture_task import NullSpacePostureTask
from embodichain.lab.sim.solvers.pink_solver import PinkSolverCfg

PLANAR_URDF = """<?xml version="1.0"?>
<robot name="pink_test">
  <link name="base"/><link name="link1"/><link name="link2"/><link name="tool"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/><child link="link1"/><origin xyz="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/><origin xyz="0.5 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="2"/>
  </joint>
  <joint name="tool_fixed" type="fixed">
    <parent link="link2"/><child link="tool"/><origin xyz="0.5 0 0"/>
  </joint>
</robot>
"""

OFFSET_PLANAR_URDF = PLANAR_URDF.replace(
    '<link name="base"/>',
    '<link name="world"/><link name="base"/>'
    '<joint name="base_fixed" type="fixed">'
    '<parent link="world"/><child link="base"/>'
    '<origin xyz="0.2 -0.3 0.4" rpy="0 0 0.5"/>'
    "</joint>",
)


# Base test class for differential solver
class BaseSolverTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self, solver_type: str):
        # Set up simulation with specified device (CPU or CUDA)
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)
        self.sim.set_manual_update(False)

        # Load robot URDF file
        urdf = get_data_path("Rokae/SR5/SR5.urdf")

        assert os.path.isfile(urdf)

        cfg_dict = {
            "fpath": urdf,
            "control_parts": {
                "main_arm": [
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                ],
            },
            "solver_cfg": {
                "main_arm": {
                    "class_type": "PinkSolver",
                    "end_link_name": "ee_link",
                    "root_link_name": "base_link",
                },
            },
        }

        self.robot: Robot = self.sim.add_robot(cfg=RobotCfg.from_dict(cfg_dict))

    def test_differential_solver(self):
        # Test differential solver with a 1x4x4 homogeneous matrix pose and a joint_seed
        arm_name = "main_arm"

        qpos_fk = torch.tensor(
            [[0.0, 0.0, np.pi / 2, 0.0, np.pi / 2, 0.0]], dtype=torch.float32
        )

        fk_xpos = self.robot.compute_fk(qpos=qpos_fk, name=arm_name, to_matrix=True)

        # Define start and end poses
        start_pose = fk_xpos.clone()[0]
        end_pose = fk_xpos.clone()[0]
        end_pose[:3, 3] += torch.tensor([0.0, 0.4, 0.0], dtype=torch.float32)

        # Interpolate poses
        num_steps = 100
        interpolated_poses = [
            torch.lerp(start_pose, end_pose, t) for t in np.linspace(0, 1, num_steps)
        ]

        ik_qpos = qpos_fk

        for i, pose in enumerate(interpolated_poses):
            res, ik_qpos = self.robot.compute_ik(
                pose=pose, joint_seed=ik_qpos, name=arm_name
            )
            assert res, f"IK failed for step {i} with pose:\n{pose}"

            # Verify forward kinematics matches the target pose
            ik_xpos = self.robot.compute_fk(qpos=ik_qpos, name=arm_name, to_matrix=True)
            assert torch.allclose(
                pose, ik_xpos, atol=1e-3, rtol=1e-3
            ), f"FK result does not match target pose at step {i}."

            # Set robot joint positions
            self.robot.set_qpos(
                qpos=ik_qpos, joint_ids=self.robot.get_joint_ids(arm_name)
            )

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


class TestPinkSolverUnit:
    """Exercise PinkSolver without a live simulation."""

    @staticmethod
    def _make_solver(urdf_path: Path, *, max_iterations: int = 200):
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            max_iterations=max_iterations,
            pos_eps=1e-5,
            rot_eps=1e-5,
            show_ik_warnings=False,
        )
        tcp = np.eye(4)
        tcp[0, 3] = 0.1
        cfg.tcp = tcp
        return cfg.init_solver(num_envs=2, device=torch.device("cpu"))

    def test_batch_ik_reconstructs_tcp_targets(self, tmp_path: Path):
        """Test batch solving and removal of TCP from controlled-frame targets."""
        solver = self._make_solver(tmp_path / "planar.urdf")
        truth = torch.tensor([[0.4, -0.2], [-0.5, 0.3]], dtype=torch.float32)
        targets = solver.get_fk(truth)

        success, solutions = solver.get_ik(targets, torch.zeros_like(truth))

        assert torch.all(success)
        assert solutions.shape == (2, 1, 2)
        reconstructed = solver.get_fk(solutions[:, 0])
        assert torch.allclose(reconstructed, targets, atol=1e-4, rtol=1e-4)

    def test_init_solver_accepts_positional_device(self, tmp_path: Path):
        """Test the SolverCfg positional-device interface remains supported."""
        urdf_path = tmp_path / "positional_device.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            show_ik_warnings=False,
        )

        solver = cfg.init_solver(torch.device("cpu"), num_envs=1)

        assert solver.device == torch.device("cpu")

    def test_non_convergence_returns_per_target_seed(self, tmp_path: Path):
        """Test failed targets are reported and preserve their individual seeds."""
        solver = self._make_solver(
            tmp_path / "planar_unreachable.urdf", max_iterations=1
        )
        seeds = torch.tensor([[0.1, -0.1], [-0.2, 0.2]], dtype=torch.float32)
        targets = torch.eye(4).repeat(2, 1, 1)
        targets[:, :3, 3] = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])

        success, solutions = solver.get_ik(targets, seeds)

        assert not torch.any(success)
        assert solutions.shape == (2, 1, 2)
        assert torch.allclose(solutions[:, 0], seeds)

    def test_zero_configured_damping_uses_positive_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Test every QP solve receives positive damping when cfg.damp is zero."""
        solver = self._make_solver(tmp_path / "zero_damping.urdf")
        solver.cfg.damp = 0.0
        target = solver.get_fk(torch.tensor([[0.4, -0.2]], dtype=torch.float32))
        observed_damping: list[float] = []
        original_solve_ik = solver.pink.solve_ik

        def recording_solve_ik(**kwargs):
            observed_damping.append(float(kwargs["damping"]))
            return original_solve_ik(**kwargs)

        monkeypatch.setattr(solver.pink, "solve_ik", recording_solve_ik)

        solver.get_ik(target, torch.zeros((1, 2), dtype=torch.float32))

        assert observed_damping
        assert min(observed_damping) >= np.sqrt(np.finfo(float).eps)

    def test_single_batched_seed_is_broadcast(self, tmp_path: Path):
        """Test a ``(1, dof)`` seed can initialize a target batch."""
        solver = self._make_solver(tmp_path / "planar_seed_broadcast.urdf")
        truth = torch.tensor([[0.4, -0.2], [-0.5, 0.3]], dtype=torch.float32)
        targets = solver.get_fk(truth)

        success, solutions = solver.get_ik(targets, torch.zeros(1, 2))

        assert torch.all(success)
        assert solutions.shape == (2, 1, 2)

    def test_zero_cost_frame_axes_do_not_block_convergence(self, tmp_path: Path):
        """Test unconstrained task axes are excluded from residual checks."""
        urdf_path = tmp_path / "planar_axis_costs.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        frame_task = pink.tasks.FrameTask(
            frame="tool",
            position_cost=[1.0, 0.0, 1.0],
            orientation_cost=0.0,
        )
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            variable_input_tasks=[frame_task],
            is_only_position_constraint=True,
            show_ik_warnings=False,
        )
        solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))
        target = solver.get_fk(torch.zeros(1, 2))
        target[:, 1, 3] += 10.0

        success, solution = solver.get_ik(target, torch.zeros(1, 2))

        assert torch.all(success)
        assert torch.allclose(solution[:, 0], torch.zeros(1, 2))

    def test_multiple_variable_frame_tasks_are_rejected(self, tmp_path: Path):
        """Test the single-pose IK API rejects ambiguous frame targets."""
        urdf_path = tmp_path / "planar_multiple_targets.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        frame_tasks = [
            pink.tasks.FrameTask(
                frame="tool",
                position_cost=1.0,
                orientation_cost=1.0,
            )
            for _ in range(2)
        ]
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            variable_input_tasks=frame_tasks,
            show_ik_warnings=False,
        )

        with pytest.raises(ValueError, match="exactly one FrameTask"):
            cfg.init_solver(num_envs=1, device=torch.device("cpu"))

    def test_invalid_tcp_update_is_transactional(self, tmp_path: Path):
        """Test a singular TCP is rejected without replacing the active TCP."""
        solver = self._make_solver(tmp_path / "planar_tcp.urdf")
        previous_tcp = solver.get_tcp().copy()

        with pytest.raises(np.linalg.LinAlgError):
            solver.set_tcp(np.zeros((4, 4)))

        assert np.array_equal(solver.get_tcp(), previous_tcp)

    def test_ik_targets_are_relative_to_configured_root(self, tmp_path: Path):
        """Test an upstream URDF offset does not leak into root-relative IK."""
        urdf_path = tmp_path / "offset_planar.urdf"
        urdf_path.write_text(OFFSET_PLANAR_URDF, encoding="utf-8")
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            show_ik_warnings=False,
        )
        solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))
        truth = torch.tensor([[0.35, -0.25]], dtype=torch.float32)
        target = solver.get_fk(truth)

        success, solution = solver.get_ik(target, torch.zeros_like(truth))

        assert torch.all(success)
        assert torch.allclose(
            solver.get_fk(solution[:, 0]), target, atol=1e-4, rtol=1e-4
        )
        assert torch.allclose(solver._get_fk(truth[0]), target[0], atol=1e-5, rtol=1e-5)

    def test_effective_limits_are_synchronized_with_pink(self, tmp_path: Path):
        """Test user and runtime robot limits tighten the Pinocchio model."""
        urdf_path = tmp_path / "planar_limits.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint2", "joint1"],
            root_link_name="base",
            end_link_name="tool",
            user_qpos_limits=[[-0.4, -0.3], [0.5, 0.6]],
            show_ik_warnings=False,
        )
        solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))

        assert np.allclose(solver.robot.model.lowerPositionLimit, [-0.3, -0.4])
        assert np.allclose(solver.robot.model.upperPositionLimit, [0.6, 0.5])

        solver.update_with_robot_limit(
            torch.tensor([[-0.2, 0.25], [-0.1, 0.3]], dtype=torch.float32)
        )

        assert np.allclose(solver.robot.model.lowerPositionLimit, [-0.1, -0.2])
        assert np.allclose(solver.robot.model.upperPositionLimit, [0.3, 0.25])

        assert solver.set_qpos_limits([-0.05, -0.15], [0.2, 0.2])
        assert np.allclose(solver.robot.model.lowerPositionLimit, [-0.1, -0.05])
        assert np.allclose(solver.robot.model.upperPositionLimit, [0.2, 0.2])
        assert torch.allclose(solver.lower_qpos_limits, torch.tensor([-0.05, -0.1]))
        assert torch.allclose(solver.upper_qpos_limits, torch.tensor([0.2, 0.2]))

        with pytest.raises(ValueError, match="shape"):
            solver.update_with_robot_limit(torch.zeros(2, 3))

        setter_first_solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))
        assert setter_first_solver.set_qpos_limits([-0.05, -0.15], [0.2, 0.2])
        setter_first_solver.update_with_robot_limit(
            torch.tensor([[-0.2, 0.25], [-0.1, 0.3]], dtype=torch.float32)
        )
        assert np.allclose(
            setter_first_solver.robot.model.lowerPositionLimit, [-0.1, -0.05]
        )
        assert np.allclose(
            setter_first_solver.robot.model.upperPositionLimit, [0.2, 0.2]
        )

    def test_invalid_limit_setters_are_transactional(self, tmp_path: Path):
        """Test rejected limits preserve configured, effective, and model state."""
        solver = self._make_solver(tmp_path / "planar_transactional_limits.urdf")
        solver.update_with_robot_limit(
            torch.tensor([[-0.2, 0.25], [-0.1, 0.3]], dtype=torch.float32)
        )
        configured_lower = solver._configured_lower_limits.clone()
        configured_upper = solver._configured_upper_limits.clone()
        effective_lower = solver.lower_qpos_limits.clone()
        effective_upper = solver.upper_qpos_limits.clone()
        model_lower = solver.robot.model.lowerPositionLimit.copy()
        model_upper = solver.robot.model.upperPositionLimit.copy()
        invalid_limits = [
            ([0.0], [1.0]),
            ([np.nan, 0.0], [1.0, 1.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([0.5, 0.5], [0.6, 0.6]),
        ]

        for lower, upper in invalid_limits:
            with pytest.raises(ValueError):
                solver.set_qpos_limits(lower, upper)
            assert torch.equal(solver._configured_lower_limits, configured_lower)
            assert torch.equal(solver._configured_upper_limits, configured_upper)
            assert torch.equal(solver.lower_qpos_limits, effective_lower)
            assert torch.equal(solver.upper_qpos_limits, effective_upper)
            assert np.array_equal(solver.robot.model.lowerPositionLimit, model_lower)
            assert np.array_equal(solver.robot.model.upperPositionLimit, model_upper)

    def test_initial_posture_is_projected_into_effective_limits(self, tmp_path: Path):
        """Test default seeds and posture targets cannot start outside user limits."""
        urdf_path = tmp_path / "planar_initial_limits.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        posture_task = NullSpacePostureTask(cost=1e-3)
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            user_qpos_limits=[[0.2, 0.3], [0.5, 0.6]],
            fixed_input_tasks=[posture_task],
            show_ik_warnings=False,
        )
        solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))

        assert np.allclose(solver.init_qpos, [0.2, 0.3])
        assert np.allclose(solver.fixed_input_tasks[0].target_q, [0.2, 0.3])
        _, seeds = solver._normalize_inputs(torch.eye(4), None)
        assert np.allclose(seeds, [[0.2, 0.3]])

    def test_invalid_adaptive_controls_are_rejected(self, tmp_path: Path):
        """Test invalid convergence controls fail during solver initialization."""
        urdf_path = tmp_path / "planar_invalid.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            damping_decay=1.1,
        )

        with pytest.raises(ValueError, match="damping_decay"):
            cfg.init_solver(num_envs=1, device=torch.device("cpu"))

    def test_null_space_posture_task_is_integrated(self, tmp_path: Path):
        """Test posture targets, all-joint defaults, and QP integration."""
        urdf_path = tmp_path / "planar_posture.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        posture_task = NullSpacePostureTask(cost=1e-3)
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            fixed_input_tasks=[posture_task],
            show_ik_warnings=False,
        )
        solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))
        solver_posture_task = solver.fixed_input_tasks[0]

        solver.update_null_space_joint_targets(np.array([0.3, -0.2]))
        solver.pink_cfg.update(np.zeros(2))

        assert np.allclose(
            solver_posture_task.compute_error(solver.pink_cfg), [-0.3, 0.2]
        )
        assert np.allclose(
            solver_posture_task.compute_jacobian(solver.pink_cfg), np.eye(2)
        )
        hessian, gradient = solver_posture_task.compute_qp_objective(solver.pink_cfg)
        assert hessian.shape == (2, 2)
        assert gradient.shape == (2,)
        assert np.all(np.isfinite(hessian))
        assert np.all(np.isfinite(gradient))

        selected_task = NullSpacePostureTask(cost=1.0, controlled_joints=["joint2"])
        selected_task.set_target(np.array([0.3, -0.2]))
        assert np.allclose(selected_task.compute_error(solver.pink_cfg), [0.0, 0.2])
        assert np.allclose(
            selected_task.compute_jacobian(solver.pink_cfg), np.diag([0.0, 1.0])
        )

        projected_task = NullSpacePostureTask(cost=1.0, controlled_frames=["tool"])
        projected_task.set_target(np.array([0.3, -0.2]))
        projector = projected_task.compute_jacobian(solver.pink_cfg)
        frame_jacobian = solver.pink_cfg.get_frame_jacobian("tool")
        assert np.linalg.norm(frame_jacobian @ projector) < 1e-10

    def test_full_rank_primary_task_is_not_vetoed_by_posture_merit(
        self, tmp_path: Path
    ):
        """Test a projected-out posture error cannot block frame convergence."""
        urdf_path = tmp_path / "planar_full_rank_posture.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        posture_task = NullSpacePostureTask(
            cost=10.0,
            controlled_frames=["tool"],
        )
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            fixed_input_tasks=[posture_task],
            pos_eps=1e-5,
            rot_eps=1e-5,
            show_ik_warnings=False,
        )
        solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))
        truth = torch.tensor([[0.4, -0.2]], dtype=torch.float32)
        target = solver.get_fk(truth)

        projector = posture_task.compute_jacobian(solver.pink_cfg)
        success, solution = solver.get_ik(target, torch.zeros_like(truth))

        assert np.linalg.norm(projector) < 1e-10
        assert torch.all(success)
        assert torch.allclose(
            solver.get_fk(solution[:, 0]), target, atol=1e-4, rtol=1e-4
        )

    def test_posture_secondary_merit_uses_cost_once(self, tmp_path: Path):
        """Test posture cost follows Pink's cost-squared objective scaling."""
        urdf_path = tmp_path / "planar_posture_merit_cost.urdf"
        urdf_path.write_text(PLANAR_URDF, encoding="utf-8")
        posture_task = NullSpacePostureTask(cost=1.0)
        cfg = PinkSolverCfg(
            urdf_path=str(urdf_path),
            joint_names=["joint1", "joint2"],
            root_link_name="base",
            end_link_name="tool",
            fixed_input_tasks=[posture_task],
            show_ik_warnings=False,
        )
        solver = cfg.init_solver(num_envs=1, device=torch.device("cpu"))
        posture_task.set_target(np.array([0.3, -0.2]))
        solver.pink_cfg.update(np.zeros(2))

        _, unit_cost_merit, _, _ = solver._task_metrics()
        posture_task.cost = 2.0
        _, doubled_cost_merit, _, _ = solver._task_metrics()

        assert doubled_cost_merit == pytest.approx(4.0 * unit_cost_merit)

    def test_null_space_posture_task_rejects_unknown_entities(self, tmp_path: Path):
        """Test invalid joint and frame selectors fail with useful errors."""
        solver = self._make_solver(tmp_path / "planar_unknown.urdf")
        joint_task = NullSpacePostureTask(cost=1.0, controlled_joints=["missing"])
        joint_task.set_target(np.zeros(2))
        with pytest.raises(ValueError, match="Unknown controlled joints"):
            joint_task.compute_error(solver.pink_cfg)

        frame_task = NullSpacePostureTask(cost=1.0, controlled_frames=["missing"])
        frame_task.set_target(np.zeros(2))
        with pytest.raises(ValueError, match="Unknown controlled frames"):
            frame_task.compute_jacobian(solver.pink_cfg)

    def test_null_space_posture_default_excludes_floating_base(self):
        """Test the default posture mask contains only actuated coordinates."""
        model = pin.Model()
        root_id = model.addJoint(
            0,
            pin.JointModelFreeFlyer(),
            pin.SE3.Identity(),
            "root_joint",
        )
        model.appendBodyToJoint(root_id, pin.Inertia.Random(), pin.SE3.Identity())
        joint_id = model.addJoint(
            root_id,
            pin.JointModelRZ(),
            pin.SE3.Identity(),
            "joint1",
        )
        model.appendBodyToJoint(joint_id, pin.Inertia.Random(), pin.SE3.Identity())
        neutral = pin.neutral(model)
        configuration = pink.configuration.Configuration(
            model,
            model.createData(),
            pin.integrate(model, neutral, np.full(model.nv, 0.1)),
        )

        for controlled_joints in (None, ["joint1"]):
            task = NullSpacePostureTask(
                cost=1.0,
                controlled_joints=controlled_joints,
            )
            task.set_target(neutral)

            assert np.allclose(task.compute_error(configuration)[:6], 0.0)
            assert not np.isclose(task.compute_error(configuration)[6], 0.0)
            assert np.allclose(
                task.compute_jacobian(configuration),
                np.diag([0.0] * 6 + [1.0]),
            )

        frame_id = model.addJointFrame(joint_id)
        frame_name = model.frames[frame_id].name

        class CoupledJacobianConfiguration:
            """Expose a primary Jacobian coupling root and actuated velocity."""

            def __init__(self):
                self.model = model

            @staticmethod
            def get_frame_jacobian(name: str) -> np.ndarray:
                assert name == frame_name
                jacobian = np.zeros((1, model.nv))
                jacobian[0, 0] = 1.0
                jacobian[0, -1] = 1.0
                return jacobian

        projected_task = NullSpacePostureTask(
            cost=1.0,
            controlled_frames=[frame_name],
        )
        projector = projected_task.compute_jacobian(CoupledJacobianConfiguration())

        assert np.allclose(projector[:6, :], 0.0)
        assert np.allclose(projector[:, :6], 0.0)
        assert projector[6, 6] == pytest.approx(0.5)


@pytest.mark.skip(reason="Skipping Pink tests temporarily")
class TestPinkSolver(BaseSolverTest):
    def setup_method(self):
        self.setup_simulation(solver_type="PinkSolver")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", __file__]
    pytest.main(pytest_args)
