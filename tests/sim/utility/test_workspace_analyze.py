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

import torch
import pytest
import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import CobotMagicCfg
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    SamplingConfig,
    VisualizationConfig,
    DimensionConstraint,
)


# Base test class for workspace analyzer
class BaseWorkspaceAnalyzeTest:
    sim = None  # Define as a class attribute

    def setup_simulation(self):
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)
        self.sim.set_manual_update(False)

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

        self.robot: Robot = self.sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()


class TestJointWorkspaceAnalyzeTest(BaseWorkspaceAnalyzeTest):
    def setup_method(self):
        self.setup_simulation()

    def test_joint_workspace_analyze(self):
        from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
            WorkspaceAnalyzer,
            WorkspaceAnalyzerConfig,
            AnalysisMode,
        )

        joint_config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.JOINT_SPACE,
            sampling=SamplingConfig(num_samples=100),
            control_part_name="left_arm",
        )
        wa_joint = WorkspaceAnalyzer(
            robot=self.robot,
            config=joint_config,
            sim_manager=self.sim,
        )
        results = wa_joint.analyze(num_samples=100)

        assert "workspace_points" in results
        assert results["workspace_points"].shape[0] > 0
        assert results["mode"] == "joint_space"
        print(
            f"Joint space results: {results['num_valid']}/{results['num_samples']} valid points"
        )


class TestCartesianWorkspaceAnalyzeTest(BaseWorkspaceAnalyzeTest):
    def setup_method(self):
        self.setup_simulation()

    def test_cartesian_workspace_analyze(self):
        from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
            WorkspaceAnalyzer,
            WorkspaceAnalyzerConfig,
            AnalysisMode,
        )

        # Get reference pose for IK orientation
        left_qpos = torch.tensor([-0.3, 0.3, 1.0, 1.0, -1.2, -1.2])
        reference_pose = self.robot.compute_fk(
            qpos=left_qpos.unsqueeze(0),
            name="left_arm",
            to_matrix=True,
        )

        cartesian_config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.CARTESIAN_SPACE,
            sampling=SamplingConfig(
                num_samples=50
            ),  # Smaller sample for faster testing
            constraint=DimensionConstraint(
                min_bounds=[-0.5, -0.5, 0.8],
                max_bounds=[0.5, 0.5, 1.5],
            ),
            reference_pose=reference_pose[0].cpu().numpy(),
            ik_samples_per_point=1,  # Single IK attempt per point for speed
            control_part_name="left_arm",
        )

        wa_cartesian = WorkspaceAnalyzer(
            robot=self.robot,
            config=cartesian_config,
            sim_manager=self.sim,
        )
        results = wa_cartesian.analyze(num_samples=50)

        assert "all_points" in results
        assert "reachable_points" in results
        assert results["all_points"].shape[0] > 0
        assert results["mode"] == "cartesian_space"
        print(
            f"Cartesian space results: {results['num_reachable']}/{results['num_samples']} reachable points"
        )


class TestPlaneWorkspaceAnalyzeTest(BaseWorkspaceAnalyzeTest):
    def setup_method(self):
        self.setup_simulation()

    def test_plane_workspace_analyze(self):
        from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
            WorkspaceAnalyzer,
            WorkspaceAnalyzerConfig,
            AnalysisMode,
        )

        # Get reference pose for IK orientation
        left_qpos = torch.tensor([-0.3, 0.3, 1.0, 1.0, -1.2, -1.2])
        reference_pose = self.robot.compute_fk(
            qpos=left_qpos.unsqueeze(0),
            name="left_arm",
            to_matrix=True,
        )

        plane_config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.PLANE_SAMPLING,
            sampling=SamplingConfig(num_samples=30),  # Small sample for testing
            plane_normal=torch.tensor([0.0, 0.0, 1.0]),  # XY plane
            plane_point=torch.tensor([0.0, 0.0, 1.2]),  # At height 1.2m
            reference_pose=reference_pose[0].cpu().numpy(),
            ik_samples_per_point=1,  # Single IK attempt per point for speed
            visualization=VisualizationConfig(
                enabled=False,  # Disable visualization for testing
            ),
            control_part_name="left_arm",
        )

        wa_plane = WorkspaceAnalyzer(
            robot=self.robot,
            config=plane_config,
            sim_manager=self.sim,
        )
        results = wa_plane.analyze(num_samples=30)

        assert "all_points" in results
        assert "reachable_points" in results
        assert results["all_points"].shape[0] > 0
        assert results["mode"] == "plane_sampling"
        assert "plane_sampling_config" in results
        print(
            f"Plane sampling results: {results['num_reachable']}/{results['num_samples']} reachable points"
        )
        print(
            f"Plane configuration: Normal={results['plane_sampling_config']['plane_normal']}, Point={results['plane_sampling_config']['plane_point']}"
        )


class TestWorkspaceAnalyzerComprehensive(BaseWorkspaceAnalyzeTest):
    """Comprehensive test class for testing consistency between different modes"""

    def setup_method(self):
        self.setup_simulation()

    def test_all_modes_consistency(self):
        """Test basic functionality and result consistency of all three modes"""
        from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
            WorkspaceAnalyzer,
            WorkspaceAnalyzerConfig,
            AnalysisMode,
        )

        # Get reference pose
        left_qpos = torch.tensor([-0.3, 0.3, 1.0, 1.0, -1.2, -1.2])
        reference_pose = self.robot.compute_fk(
            qpos=left_qpos.unsqueeze(0),
            name="left_arm",
            to_matrix=True,
        )

        num_samples = 20  # Small sample for fast testing
        results = {}

        # Test joint space mode
        joint_config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.JOINT_SPACE,
            sampling=SamplingConfig(num_samples=100),
            control_part_name="left_arm",
        )
        wa_joint = WorkspaceAnalyzer(
            robot=self.robot,
            config=joint_config,
            sim_manager=self.sim,
        )
        results["joint"] = wa_joint.analyze(num_samples=num_samples)

        # Test Cartesian space mode
        cartesian_config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.CARTESIAN_SPACE,
            sampling=SamplingConfig(num_samples=num_samples),
            constraint=DimensionConstraint(
                min_bounds=[-0.5, -0.5, 0.8],
                max_bounds=[0.5, 0.5, 1.5],
            ),
            reference_pose=reference_pose[0].cpu().numpy(),
            ik_samples_per_point=1,
            control_part_name="left_arm",
        )
        wa_cartesian = WorkspaceAnalyzer(
            robot=self.robot,
            config=cartesian_config,
            sim_manager=self.sim,
        )
        results["cartesian"] = wa_cartesian.analyze(num_samples=num_samples)

        # Test plane sampling mode
        plane_config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.PLANE_SAMPLING,
            sampling=SamplingConfig(num_samples=num_samples),
            plane_normal=torch.tensor([0.0, 0.0, 1.0]),
            plane_point=torch.tensor([0.0, 0.0, 1.2]),
            reference_pose=reference_pose[0].cpu().numpy(),
            ik_samples_per_point=1,
            control_part_name="left_arm",
        )
        wa_plane = WorkspaceAnalyzer(
            robot=self.robot,
            config=plane_config,
            sim_manager=self.sim,
        )
        results["plane"] = wa_plane.analyze(num_samples=num_samples)

        # Verify all modes return valid results
        for mode_name, result in results.items():
            assert "analysis_time" in result
            assert result["analysis_time"] > 0
            assert "mode" in result
            print(
                f"{mode_name.capitalize()} mode: {result['mode']}, Time: {result['analysis_time']:.2f}s"
            )

        # Verify joint space mode
        assert results["joint"]["mode"] == "joint_space"
        assert results["joint"]["num_valid"] > 0

        # Verify Cartesian space mode
        assert results["cartesian"]["mode"] == "cartesian_space"
        assert "reachable_points" in results["cartesian"]

        # Verify plane sampling mode
        assert results["plane"]["mode"] == "plane_sampling"
        assert "plane_sampling_config" in results["plane"]

        print("\n=== Comprehensive Test Summary ===")
        print(
            f"Joint space: {results['joint']['num_valid']}/{results['joint']['num_samples']} valid points"
        )
        print(
            f"Cartesian space: {results['cartesian']['num_reachable']}/{results['cartesian']['num_samples']} reachable points"
        )
        print(
            f"Plane sampling: {results['plane']['num_reachable']}/{results['plane']['num_samples']} reachable points"
        )


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    pytest_args = ["-v", "-s", __file__]
    pytest.main(pytest_args)
