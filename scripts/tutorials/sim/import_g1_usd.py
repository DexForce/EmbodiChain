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

"""Import and drive the original AGIBot G1 grippers on CPU with no renderer.

Run with a DexSim wheel that supports USD closed-loop articulations:

    python scripts/tutorials/sim/import_g1_usd.py /path/to/G1_120s/robot.usda

Keep the entire original G1_120s directory from the GenieSimAssets bundle
described in EmbodiChain #904. The entry layer references other files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    """Load the complete USD scene and command both grippers through Robot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usd", type=Path, help="Original G1_120s/robot.usda")
    usd = parser.parse_args().usd
    if not usd.is_file():
        parser.error(f"USD file does not exist: {usd}")

    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RenderCfg, RobotCfg

    control_parts = {
        "left_eef": ["idx41_gripper_l_outer_joint1"],
        "right_eef": ["idx81_gripper_r_outer_joint1"],
    }
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=1,
            physics_cfg=DefaultPhysicsCfg(device="cpu", physics_dt=0.001),
            render_cfg=RenderCfg(renderer="no-render"),
        )
    )
    robot = None
    try:
        # add_usd keeps the scene-owned closure joints. robot_cfgs exposes the
        # articulation as a Robot facade for sending joint targets.
        robot = sim.add_usd(
            "G1",
            str(usd.resolve()),
            robot_cfgs={"/G1": RobotCfg(uid="G1", control_parts=control_parts)},
        )["/G1"]
        sim.prepare()
        sim.get_world().enable_physics(True)
        for target in (0.0, 0.4, 0.0):
            for part in control_parts:
                robot.set_qpos(
                    torch.full((1, 1), target, device=robot.device), name=part
                )
            sim.update(step=1000)
            angles = {part: robot.get_qpos(name=part).item() for part in control_parts}
            print(f"target {target:.2f} rad, measured joint angles (rad): {angles}")
    finally:
        robot = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
