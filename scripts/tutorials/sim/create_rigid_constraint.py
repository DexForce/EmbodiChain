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
This script demonstrates how to attach two rigid objects via a fixed constraint,
observe the constraint holding their relative pose, and then remove it.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.cfg import (
    RigidObjectCfg,
    RigidConstraintCfg,
    RigidBodyAttributesCfg,
    RenderCfg,
)
from embodichain.lab.sim.shapes import CubeCfg

# Number of physics sub-steps per update call.
STEPS_PER_UPDATE = 1
# Print the relative pose every N update calls.
PRINT_EVERY = 20
# How long to simulate while attached / detached (in update calls).
PHASE_STEPS = 120


def main():
    """Main function to create and run the constraint tutorial scene."""

    # Parse command line arguments (adds --headless, --num_envs, --device, ...).
    parser = argparse.ArgumentParser(
        description="Attach and detach two cubes via a fixed rigid constraint"
    )
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()

    # The simulation teardown (``SimulationManager.destroy``) calls ``os._exit``,
    # which skips flushing Python's stdout buffer. Line-buffer stdout so every
    # ``print`` below is visible even when the script is piped to a file.
    sys.stdout.reconfigure(line_buffering=True)

    # Configure the simulation.
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        num_envs=args.num_envs,
        arena_space=3.0,
    )

    sim = SimulationManager(sim_cfg)

    # Shared physics attributes for the two cubes.
    physics_attrs = RigidBodyAttributesCfg(
        mass=0.2,
        dynamic_friction=0.5,
        static_friction=0.5,
        restitution=0.1,
    )

    # Add two dynamic cubes to the scene. cube_a starts higher than cube_b so
    # that, once detached, the lower cube lands first and the relative pose
    # visibly changes (while welded, the constraint holds it constant).
    cube_a = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube_a",
            shape=CubeCfg(size=[0.16, 0.16, 0.16]),
            attrs=physics_attrs,
            init_pos=[0.0, 0.0, 1.40],
        )
    )
    cube_b = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube_b",
            shape=CubeCfg(size=[0.16, 0.16, 0.16]),
            attrs=physics_attrs,
            init_pos=[0.0, 0.0, 1.20],
        )
    )

    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    print("[INFO]: Scene setup complete with two cubes (cube_a, cube_b).")

    # --- Phase 1: attach the two cubes with a fixed constraint ---------------
    # local_frame_a / local_frame_b default to identity, so the constraint
    # welds the cubes at their *current* relative pose.
    constraint = sim.create_rigid_constraint(
        cfg=RigidConstraintCfg(
            name="cube_weld",
            rigid_object_a_uid="cube_a",
            rigid_object_b_uid="cube_b",
        )
    )
    print("[INFO]: Created constraint 'cube_weld' between cube_a and cube_b.")

    # Open the viewer (unless --headless) so the welded motion is visible.
    if not args.headless:
        sim.open_window()

    print("[INFO]: Stepping physics while ATTACHED (relative pose held):")
    _run_phase(sim, cube_a, cube_b, attached=True)

    # --- Phase 2: remove the constraint ------------------------------------
    sim.remove_rigid_constraint("cube_weld")
    assert sim.get_rigid_constraint("cube_weld") is None
    print("\n[INFO]: Removed constraint 'cube_weld'. cube_a and cube_b are now free.")

    print("[INFO]: Stepping physics while DETACHED (relative pose may drift):")
    _run_phase(sim, cube_a, cube_b, attached=False)

    print("\n[INFO]: Tutorial complete.")
    sim.destroy()


def _relative_z(cube_a, cube_b) -> float:
    """Return the z-component of cube_b's pose relative to cube_a (env 0).

    This reads the two bodies' world poses directly, so it works both while the
    constraint is active (the value stays constant) and after removal (the
    value drifts as the cubes move independently).

    Args:
        cube_a: The first :class:`RigidObject`.
        cube_b: The second :class:`RigidObject`.

    Returns:
        The relative z (cube_b.z - cube_a.z) in meters.
    """
    pose_a = cube_a.get_local_pose(to_matrix=True)
    pose_b = cube_b.get_local_pose(to_matrix=True)
    return float(pose_b[0, 2, 3] - pose_a[0, 2, 3])


def _run_phase(sim, cube_a, cube_b, attached: bool) -> None:
    """Step the simulation for one phase and print the bodies' relative z.

    Args:
        sim: The :class:`SimulationManager`.
        cube_a: The first :class:`RigidObject`.
        cube_b: The second :class:`RigidObject`.
        attached: True while the constraint is active, False after removal.
    """
    rel_z = _relative_z(cube_a, cube_b)
    print(f"  step {0:4d}: relative z (cube_b - cube_a) = {rel_z:.4f} m")
    for step in range(1, PHASE_STEPS + 1):
        sim.update(step=STEPS_PER_UPDATE)
        if step % PRINT_EVERY == 0:
            rel_z = _relative_z(cube_a, cube_b)
            print(f"  step {step:4d}: relative z (cube_b - cube_a) = {rel_z:.4f} m")


if __name__ == "__main__":
    main()
