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

"""Load a passive articulation and open or close it with joint forces."""

from __future__ import annotations

import argparse

import torch

from dexsim.types import DriveType

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    RenderCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.visualization import visualization_cfg_from_args

DRAWER_ASSET = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"
DRAWER_USER_QPOS_LIMITS = {"slide_rails": [0.0, 0.18]}
DRAWER_JOINT_FORCE = 1.0
JOINT_LIMIT_TOLERANCE = 1.0e-3


def create_articulation(sim: SimulationManager) -> Articulation:
    """Load a drawer articulation with the passive default drive.

    Args:
        sim: Simulation manager that owns the scene.

    Returns:
        The loaded drawer articulation.

    Raises:
        RuntimeError: If the constructed backend joints are not passive.
    """
    # Resolve the drawer URDF and configure its initial pose. ``drive_pros`` is
    # intentionally omitted: ArticulationCfg defaults to drive_type="none".
    articulation_cfg = ArticulationCfg(
        uid="drawer",
        fpath=get_data_path(DRAWER_ASSET),
        init_pos=(0.0, 0.0, 0.05),
        fix_base=True,
        # The asset limit is [0.0, 0.2]; keep 90% of its travel range.
        qpos_limits=DRAWER_USER_QPOS_LIMITS,
    )

    # Load one articulation instance into every simulation environment.
    articulation: Articulation = sim.add_articulation(cfg=articulation_cfg)
    sim.prepare()

    # Query the constructed DexSim entities, not only the config object.
    backend_drive_types = articulation.get_joint_drive_type()
    expected_drive_types = [
        [DriveType.NONE] * articulation.dof for _ in range(sim.num_envs)
    ]
    if backend_drive_types != expected_drive_types:
        raise RuntimeError(
            "Expected every articulation joint drive to be DriveType.NONE, "
            f"but received {backend_drive_types!r}."
        )

    print(f"[INFO]: Loaded articulation with {articulation.dof} joint(s)", flush=True)
    print(f"[INFO]: Joint names: {articulation.joint_names}", flush=True)
    print(
        f"[INFO]: Config drive type: {articulation.cfg.drive_pros.drive_type}",
        flush=True,
    )
    print(f"[INFO]: Backend drive types: {backend_drive_types}", flush=True)
    print(
        f"[INFO]: Effective qpos limits: {articulation.get_qpos_limits()}", flush=True
    )
    return articulation


def apply_drawer_force(articulation: Articulation, opening: bool) -> None:
    """Apply a joint force that opens or closes the drawer.

    Args:
        articulation: Drawer articulation receiving the force.
        opening: If True, apply positive force; otherwise apply negative force.
    """
    force = DRAWER_JOINT_FORCE if opening else -DRAWER_JOINT_FORCE
    joint_forces = torch.full_like(articulation.get_qpos(), force)
    articulation.set_qf(joint_forces)


def run_simulation(
    sim: SimulationManager,
    articulation: Articulation,
    max_steps: int | None = None,
) -> None:
    """Open and close the drawer by reversing force at its joint limits.

    Args:
        sim: Simulation manager to advance.
        articulation: Drawer articulation whose joints are updated.
        max_steps: Optional number of steps to run before returning.
    """
    qpos_limits = articulation.get_qpos_limits()
    closed_qpos = qpos_limits[..., 0]
    open_qpos = qpos_limits[..., 1]
    opening = True
    step_count = 0
    print(
        f"[INFO]: Applying +{DRAWER_JOINT_FORCE:.1f} N to open the drawer",
        flush=True,
    )
    try:
        while max_steps is None or step_count < max_steps:
            qpos = articulation.get_qpos()
            if opening and torch.all(qpos >= open_qpos - JOINT_LIMIT_TOLERANCE).item():
                print(f"[INFO]: Drawer reached open limit: {qpos}", flush=True)
                opening = False
                print(
                    f"[INFO]: Applying -{DRAWER_JOINT_FORCE:.1f} N to close the drawer",
                    flush=True,
                )
            elif (
                not opening
                and torch.all(qpos <= closed_qpos + JOINT_LIMIT_TOLERANCE).item()
            ):
                print(f"[INFO]: Drawer reached closed limit: {qpos}", flush=True)
                opening = True
                print(
                    f"[INFO]: Applying +{DRAWER_JOINT_FORCE:.1f} N to open the drawer",
                    flush=True,
                )

            apply_drawer_force(articulation, opening=opening)
            sim.update(step=1)
            step_count += 1
    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        articulation.set_qf(torch.zeros_like(articulation.get_qpos()))


def main() -> None:
    """Load and simulate a passive drawer articulation."""
    parser = argparse.ArgumentParser(
        description="Load an articulation with its default passive joint drive"
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional number of simulation steps before exiting.",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    # Configure the simulation. Window creation is deferred until the asset is loaded.
    sim_cfg = SimulationManagerCfg(
        headless=args.headless,
        sim_device=args.device,
        num_envs=args.num_envs,
        arena_space=2.0,
        physics_dt=1.0 / 100.0,
        physics_cfg=physics_cfg_for_backend(args.physics),
        render_cfg=RenderCfg(renderer=args.renderer),
        visualization=visualization_cfg_from_args(args),
    )
    sim = SimulationManager(sim_cfg)

    try:
        articulation = create_articulation(sim)
        print(f"[INFO]: Initial joint positions: {articulation.get_qpos()}", flush=True)

        if not args.headless and not args.viser:
            sim.open_window()

        print("[INFO]: Running simulation. Press Ctrl+C to stop.", flush=True)
        run_simulation(sim, articulation, max_steps=args.max_steps)
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
