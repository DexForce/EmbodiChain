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
Gizmo-Robot Example: Test Gizmo class on a robot (UR10)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    create_default_sim,
    maybe_open_window,
    resolve_demo_steps,
    shutdown_sim,
)
from embodichain.utils import logger
from embodichain.lab.sim.robots.dexforce_w1.cfg import DexforceW1Cfg


def main() -> None:
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_demo_args(parser)
    args = parser.parse_args()

    sim = create_default_sim(
        args,
        width=1920,
        height=1080,
        physics_dt=1.0 / 100.0,
        add_default_light=False,
    )
    sim.set_manual_update(False)

    cfg = DexforceW1Cfg.from_dict(
        {
            "uid": "w1_gizmo_test",
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
    robot = sim.add_robot(cfg=cfg)

    # Set initial joint positions for both arms
    # Left arm: 8 joints (WAIST + 7 LEFT_J), Right arm: 8 joints (WAIST + 7 RIGHT_J)
    left_arm_qpos = torch.tensor(
        [
            [0, 0, -np.pi / 4, np.pi / 4, -np.pi / 2, 0.0, np.pi / 4, 0.0]
        ],  # WAIST + LEFT_J[1-7]
        dtype=torch.float32,
        device=robot.device,
    )
    right_arm_qpos = torch.tensor(
        [
            [0, 0, np.pi / 4, -np.pi / 4, np.pi / 2, 0.0, -np.pi / 4, 0.0]
        ],  # WAIST + RIGHT_J[1-7]
        dtype=torch.float32,
        device=robot.device,
    )

    left_joint_ids = robot.get_joint_ids("left_arm")
    right_joint_ids = robot.get_joint_ids("right_arm")

    robot.set_qpos(qpos=left_arm_qpos, joint_ids=left_joint_ids)
    robot.set_qpos(qpos=right_arm_qpos, joint_ids=right_joint_ids)

    time.sleep(0.2)  # Wait for a moment to ensure everything is set up

    # Enable gizmo for both arms using the new API
    if not args.headless:
        sim.enable_gizmo(uid="w1_gizmo_test", control_part="left_arm")
        sim.enable_gizmo(uid="w1_gizmo_test", control_part="right_arm")
        for arm_name in ("left_arm", "right_arm"):
            if not sim.has_gizmo("w1_gizmo_test", control_part=arm_name):
                shutdown_sim(sim)
                raise RuntimeError(f"Failed to enable {arm_name} gizmo.")

    maybe_open_window(sim, args)

    logger.log_info("Gizmo-DexForce W1 example started!")
    logger.log_info("Use the gizmos to drag both robot arms' end-effectors")
    logger.log_info("Press Ctrl+C to stop the simulation")

    run_simulation(sim, max_steps=resolve_demo_steps(args))


def run_simulation(sim: SimulationManager, max_steps: int | None = None) -> None:
    step_count = 0
    try:
        last_time = time.time()
        last_step = 0
        while max_steps is None or step_count < max_steps:
            time.sleep(0.033)  # 30Hz
            # Update all gizmos managed by sim
            sim.update_gizmos()
            step_count += 1

            if step_count % 100 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )
                logger.log_info(f"Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count
    except KeyboardInterrupt:
        logger.log_info("\nStopping simulation...")
    finally:
        shutdown_sim(sim)
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
