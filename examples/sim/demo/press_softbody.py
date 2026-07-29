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
This script demonstrates the creation and simulation of a robot with a soft object,
and performs a pressing task in a simulated environment.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from dexsim.utility.path import get_resources_data_path

from embodichain.lab.sim.cfg import (
    SoftObjectCfg,
    SoftbodyPhysicalAttributesCfg,
    SoftbodyVoxelAttributesCfg,
)
from embodichain.lab.sim.demo_base import DemoBase
from embodichain.lab.sim.objects import Robot, SoftObject
from embodichain.lab.sim.robots import URRobotCfg
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


class PressSoftbodyDemo(DemoBase):
    """Press a soft cow with the end link of a UR10 robot."""

    def setup(self) -> None:
        """Create the simulation, robot and soft object, then open the viewer."""
        self.sim = create_default_sim(
            self.args,
            arena_space=5.0,
            num_envs=self.args.num_envs,
            add_default_light=False,
        )
        self.robot = self._create_robot()
        self.soft_cow = self._create_soft_cow()
        maybe_init_gpu_physics(self.sim)
        maybe_open_window(self.sim, self.args)

    def run(self) -> None:
        """Press the cow and keep the simulation live until interrupted."""
        with DemoRecording(self.sim, self.args, prefix="press_softbody"):
            self._press_cow()
            logger.log_info("\n Press Ctrl+C to exit simulation loop.")
            run_simulation_loop(
                self.sim,
                max_steps=resolve_demo_steps(self.args),
                steps_per_update=10,
            )

    def _create_robot(self) -> Robot:
        """Create and configure a UR10 robot in the simulation.

        Returns:
            The configured robot instance added to the simulation.
        """
        cfg = URRobotCfg.from_dict(
            {
                "robot_type": "ur10",
                "uid": "UR10",
                "solver_cfg": {"arm": {"tcp": np.eye(4)}},
                "init_qpos": [
                    0.0,
                    -np.pi / 2,
                    -np.pi / 2,
                    np.pi / 2,
                    -np.pi / 2,
                    0.0,
                ],
            }
        )
        return self.sim.add_robot(cfg=cfg)

    def _create_soft_cow(self) -> SoftObject:
        """Create the soft cow object in the simulation.

        Returns:
            The soft cow object.
        """
        cow: SoftObject = self.sim.add_soft_object(
            cfg=SoftObjectCfg(
                uid="cow",
                shape=MeshCfg(
                    fpath=get_resources_data_path("Model", "cow", "cow2.obj"),
                ),
                init_rot=[0, 90, 0],
                init_pos=[0.45, -0.1, 0.12],
                voxel_attr=SoftbodyVoxelAttributesCfg(
                    simulation_mesh_resolution=8,
                    maximal_edge_length=0.5,
                ),
                physical_attr=SoftbodyPhysicalAttributesCfg(
                    youngs=5e3,
                    poissons=0.45,
                    density=100,
                    dynamic_friction=0.1,
                ),
            ),
        )
        return cow

    def _press_cow(self) -> None:
        """Drive the robot end link to press the soft cow."""
        start_qpos = self.robot.get_qpos()
        arm_ids = self.robot.get_joint_ids("arm")
        arm_start_qpos = start_qpos[:, arm_ids]

        arm_start_xpos = self.robot.compute_fk(
            arm_start_qpos, name="arm", to_matrix=True
        )
        press_xpos = arm_start_xpos.clone()
        press_xpos[:, :3, 3] = torch.tensor(
            [0.5, -0.1, 0.005], device=press_xpos.device
        )

        approach_xpos = press_xpos.clone()
        approach_xpos[:, 2, 3] += 0.05

        is_success, approach_qpos = self.robot.compute_ik(
            approach_xpos, joint_seed=arm_start_qpos, name="arm"
        )

        arm_trajectory = torch.concatenate([arm_start_qpos, approach_qpos])
        interp_trajectory = interpolate_with_distance(
            trajectory=arm_trajectory[None, :, :], interp_num=50, device=self.sim.device
        )
        interp_trajectory = interp_trajectory[0]
        for qpos in interp_trajectory:
            self.robot.set_qpos(
                qpos.unsqueeze(0).repeat(self.sim.num_envs, 1), joint_ids=arm_ids
            )
            self.sim.update(step=5)


def main() -> None:
    """Entry point for the press-softbody demo."""
    setup_print_options()
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    parser = add_demo_args(parser)
    # Soft-body simulation requires GPU physics; default to CUDA.
    parser.set_defaults(device="cuda")
    args = parser.parse_args()
    PressSoftbodyDemo(args).main()


if __name__ == "__main__":
    main()
