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

"""
This script demonstrates the creation and simulation of dexforce w1 robot,
and performs a grasp cup to coffee machine task in a simulated environment.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from tqdm import tqdm

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.demo_base import DemoBase
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.robots.dexforce_w1.cfg import DexforceW1Cfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    maybe_init_gpu_physics,
    maybe_open_window,
    resolve_demo_steps,
    run_simulation_loop,
    setup_print_options,
)
from embodichain.utils import logger


class GraspCupToCaffeDemo(DemoBase):
    """Grasp a cup with the dexforce_w1 right arm and place it on the caffe."""

    def setup(self) -> None:
        """Create the simulation, robot, table, caffe and cup, then settle."""
        self.sim = create_default_sim(
            self.args,
            arena_space=2.5,
            num_envs=self.args.num_envs,
            add_default_light=False,
        )
        self.robot = self._create_robot()
        self.table = self._create_table()
        self.caffe = self._create_caffe()
        self.cup = self._create_cup()
        self.sim.update(step=1)

        # Apply random perturbation.
        self.apply_random_xy_perturbation(self.cup, max_perturbation=0.05)
        self.apply_random_xy_perturbation(self.caffe, max_perturbation=0.05)

        maybe_open_window(self.sim, self.args)
        maybe_init_gpu_physics(self.sim)

    def run(self) -> None:
        """Execute the grasp-and-place trajectory and keep the sim live."""
        with DemoRecording(self.sim, self.args, prefix="grasp_cup_to_caffe"):
            self._run_simulation()
            logger.log_info("\n Press Ctrl+C to exit simulation loop.")
            run_simulation_loop(
                self.sim,
                max_steps=resolve_demo_steps(self.args),
                steps_per_update=10,
            )

    def _create_robot(self) -> Robot:
        """Create and configure the dexforce_w1 robot.

        Returns:
            The configured robot instance added to the simulation.
        """
        cfg = DexforceW1Cfg.from_dict(
            {
                "uid": "dexforce_w1",
                "init_pos": [0.4, -0.5, 0.0],
            }
        )
        cfg.solver_cfg["left_arm"].tcp = np.array(
            [
                [1.0, 0.0, 0.0, 0.012],
                [0.0, 1.0, 0.0, 0.04],
                [0.0, 0.0, 1.0, 0.11],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        cfg.solver_cfg["right_arm"].tcp = np.array(
            [
                [1.0, 0.0, 0.0, 0.012],
                [0.0, 1.0, 0.0, -0.04],
                [0.0, 0.0, 1.0, 0.11],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        cfg.init_qpos = [
            1.0000e00,
            -2.0000e00,
            1.0000e00,
            0.0000e00,
            -2.6921e-05,
            -2.6514e-03,
            -1.5708e00,
            1.4575e00,
            -7.8540e-01,
            1.2834e-01,
            1.5708e00,
            -2.2310e00,
            -7.8540e-01,
            1.4461e00,
            -1.5708e00,
            1.6716e00,
            7.8540e-01,
            7.6745e-01,
            0.0000e00,
            3.8108e-01,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            1.5000e00,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            0.0000e00,
            1.5000e00,
            6.9974e-02,
            7.3950e-02,
            6.6574e-02,
            6.0923e-02,
            0.0000e00,
            6.7342e-02,
            7.0862e-02,
            6.3684e-02,
            5.7822e-02,
            0.0000e00,
        ]
        return self.sim.add_robot(cfg=cfg)

    def _create_table(self) -> RigidObject:
        """Create the table rigid object.

        Returns:
            The table object added to the simulation.
        """
        scoop_cfg = RigidObjectCfg(
            uid="table",
            shape=MeshCfg(
                fpath=get_data_path("MultiW1Data/table_a.obj"),
            ),
            attrs=RigidBodyAttributesCfg(
                mass=0.5,
            ),
            max_convex_hull_num=8,
            body_type="kinematic",
            init_pos=[1.1, -0.5, 0.08],
            init_rot=[0.0, 0.0, 0.0],
        )
        scoop = self.sim.add_rigid_object(cfg=scoop_cfg)
        return scoop

    def _create_caffe(self) -> Robot:
        """Create the caffe (container) articulated object.

        Returns:
            The caffe object added to the simulation.
        """
        container_cfg = ArticulationCfg(
            uid="caffe",
            fpath=get_data_path("MultiW1Data/cafe/cafe.urdf"),
            init_pos=[1.05, -0.5, 0.79],
            init_rot=[0, 0, -30],
            attrs=RigidBodyAttributesCfg(
                mass=1.0,
            ),
            drive_pros=JointDrivePropertiesCfg(
                stiffness=1.0, damping=0.1, max_effort=100.0, drive_type="force"
            ),
        )
        container = self.sim.add_articulation(cfg=container_cfg)
        return container

    def _create_cup(self) -> RigidObject:
        """Create the cup rigid object.

        Returns:
            The cup object added to the simulation.
        """
        scoop_cfg = RigidObjectCfg(
            uid="cup",
            shape=MeshCfg(
                fpath=get_data_path("MultiW1Data/paper_cup_2.obj"),
            ),
            attrs=RigidBodyAttributesCfg(
                mass=0.3,
            ),
            max_convex_hull_num=1,
            body_type="dynamic",
            init_pos=[0.86, -0.76, 0.841],
            init_rot=[0.0, 0.0, 0.0],
        )
        scoop = self.sim.add_rigid_object(cfg=scoop_cfg)
        return scoop

    def _create_trajectory(self) -> torch.Tensor:
        """Generate the right-arm trajectory to grasp the cup and place it.

        Returns:
            Interpolated trajectory of shape ``[n_envs, n_waypoint, dof]``.
        """
        robot = self.robot
        cup = self.cup
        caffe = self.caffe
        right_arm_ids = robot.get_joint_ids("right_arm")
        hand_open_qpos = torch.tensor(
            [0.0, 1.5, 0.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.sim.device,
        )
        hand_close_qpos = torch.tensor(
            [0.1, 1.5, 0.3, 0.2, 0.3, 0.3],
            dtype=torch.float32,
            device=self.sim.device,
        )

        cup_position = cup.get_local_pose(to_matrix=True)[:, :3, 3]

        # Grasp cup waypoint generation.
        rest_right_qpos = robot.get_qpos()[:, right_arm_ids]  # [n_envs, dof]
        right_arm_xpos = robot.compute_fk(
            qpos=rest_right_qpos, name="right_arm", to_matrix=True
        )
        approach_cup_relative_position = torch.tensor(
            [-0.05, -0.06, 0.025], dtype=torch.float32, device=self.sim.device
        )
        pick_cup_relative_position = torch.tensor(
            [-0.03, -0.028, 0.021], dtype=torch.float32, device=self.sim.device
        )

        approach_xpos = right_arm_xpos.clone()
        approach_xpos[:, :3, 3] = cup_position + approach_cup_relative_position

        pick_xpos = right_arm_xpos.clone()
        pick_xpos[:, :3, 3] = cup_position + pick_cup_relative_position

        lift_xpos = pick_xpos.clone()
        lift_xpos[:, 2, 3] += 0.07

        # Place cup to caffe waypoint generation.
        caffe_position = caffe.get_local_pose(to_matrix=True)[:, :3, 3]
        place_cup_up_relative_position = torch.tensor(
            [-0.14, -0.18, 0.13], dtype=torch.float32, device=self.sim.device
        )
        place_cup_down_relative_position = torch.tensor(
            [-0.14, -0.18, 0.09], dtype=torch.float32, device=self.sim.device
        )

        place_cup_up_pose = lift_xpos.clone()
        place_cup_up_pose[:, :3, 3] = caffe_position + place_cup_up_relative_position
        place_down_pose = lift_xpos.clone()
        place_down_pose[:, :3, 3] = caffe_position + place_cup_down_relative_position
        # Compute ik for each waypoint.
        is_success, approach_qpos = robot.compute_ik(
            pose=approach_xpos, joint_seed=rest_right_qpos, name="right_arm"
        )
        is_success, pick_qpos = robot.compute_ik(
            pose=pick_xpos, joint_seed=approach_qpos, name="right_arm"
        )
        is_success, lift_qpos = robot.compute_ik(
            pose=lift_xpos, joint_seed=pick_qpos, name="right_arm"
        )
        is_success, place_up_qpos = robot.compute_ik(
            pose=place_cup_up_pose, joint_seed=lift_qpos, name="right_arm"
        )
        is_success, place_down_qpos = robot.compute_ik(
            pose=place_down_pose, joint_seed=place_up_qpos, name="right_arm"
        )

        n_envs = self.sim.num_envs

        # Combine hand and arm trajectory.
        arm_trajectory = torch.cat(
            [
                rest_right_qpos[:, None, :],
                approach_qpos[:, None, :],
                pick_qpos[:, None, :],
                pick_qpos[:, None, :],
                lift_qpos[:, None, :],
                place_up_qpos[:, None, :],
                place_down_qpos[:, None, :],
                place_down_qpos[:, None, :],
                lift_qpos[:, None, :],
                rest_right_qpos[:, None, :],
            ],
            dim=1,
        )
        hand_trajectory = torch.cat(
            [
                hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_close_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
                hand_open_qpos[None, None, :].repeat(n_envs, 1, 1),
            ],
            dim=1,
        )
        all_trajectory = torch.cat([arm_trajectory, hand_trajectory], dim=-1)
        # Trajectory with shape [n_envs, n_waypoint, dof].
        interp_trajectory = interpolate_with_distance(
            trajectory=all_trajectory, interp_num=150, device=self.sim.device
        )
        return interp_trajectory

    def _run_simulation(self) -> None:
        """Execute the generated trajectory to grasp the cup and place it."""
        # [n_envs, n_waypoint, dof]
        interp_trajectory = self._create_trajectory()

        right_arm_ids = self.robot.get_joint_ids("right_arm")
        right_hand_ids = self.robot.get_joint_ids("right_eef")
        combine_ids = np.concatenate([right_arm_ids, right_hand_ids])
        n_waypoints = interp_trajectory.shape[1]
        logger.log_info("Executing trajectory...")
        for i in tqdm(range(n_waypoints)):
            self.robot.set_qpos(interp_trajectory[:, i, :], joint_ids=combine_ids)
            self.sim.update(step=10)

    @staticmethod
    def apply_random_xy_perturbation(
        item: RigidObject | Robot, max_perturbation: float = 0.02
    ) -> None:
        """Apply random perturbation to the object's XY position.

        Args:
            item: The object to perturb.
            max_perturbation: Maximum perturbation magnitude.
        """
        item_pose = item.get_local_pose(to_matrix=True)
        item_xy = item_pose[:, :2, 3].to("cpu").numpy()
        perturbation = np.random.uniform(
            low=-max_perturbation, high=max_perturbation, size=item_xy.shape
        )
        new_xy = item_xy + perturbation
        item_pose[:, :2, 3] = torch.tensor(
            new_xy, dtype=torch.float32, device=item_pose.device
        )
        item.set_local_pose(item_pose)


def main() -> None:
    """Entry point for the grasp-cup-to-caffe demo."""
    setup_print_options()
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    parser = add_demo_args(parser)
    args = parser.parse_args()
    GraspCupToCaffeDemo(args).main()


if __name__ == "__main__":
    main()
