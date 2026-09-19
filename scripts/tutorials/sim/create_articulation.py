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

"""Open and close a passive drawer with joint forces.

The default demonstrates a URDF drawer. Pass ``--asset usd`` to load
``DrawerUSD/drawer_001.usdc`` and move only its top drawer.
"""

from __future__ import annotations

import argparse
from typing import Literal
from embodichain.cli.sim import add_sim_args_to_parser


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options without initializing simulation resources."""
    parser = argparse.ArgumentParser(
        description="Load an articulation with its default passive joint drive"
    )
    add_sim_args_to_parser(parser)
    parser.add_argument(
        "--asset",
        choices=("urdf", "usd"),
        default="urdf",
        help="Drawer asset to demonstrate; usd moves the top drawer of DrawerUSD.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional number of simulation steps before exiting.",
    )
    return parser


if __name__ == "__main__":
    # Parse before importing optional simulation/planning dependencies.
    _cli_args = build_parser().parse_args()


import torch

from dexsim.types import DriveType

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    DefaultRigidBodyPropertiesCfg,
    JointDrivePropertiesCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.visualization import visualization_cfg_from_args

DRAWER_ASSET = "SlidingBoxDrawer/SlidingBoxDrawer.urdf"
USD_DRAWER_ASSET = "DrawerUSD/drawer_001.usdc"
DRAWER_USER_QPOS_LIMITS = {"slide_rails": [0.0, 0.18]}
DRAWER_JOINT_FORCE_LIMIT = 1.0
DRAWER_POSITION_GAIN = 20.0
DRAWER_VELOCITY_GAIN = 4.0
JOINT_POSITION_TOLERANCE = 1.0e-3
JOINT_VELOCITY_TOLERANCE = 1.0e-2


def create_articulation(
    sim: SimulationManager, asset: Literal["urdf", "usd"] = "urdf"
) -> Articulation:
    """Load a drawer articulation with the passive default drive.

    Args:
        sim: Simulation manager that owns the scene.
        asset: URDF drawer or the USD cabinet with three drawers.

    Returns:
        The loaded drawer articulation.

    Raises:
        ValueError: If the asset selector is unknown.
        RuntimeError: If the constructed backend joints are not passive.
    """
    if asset not in ("urdf", "usd"):
        raise ValueError(f"Unknown drawer asset: {asset!r}.")
    is_usd = asset == "usd"
    # Resolve the selected asset and explicitly request the passive drive used by
    # this tutorial while retaining all unconfigured asset properties.
    articulation_cfg = ArticulationCfg(
        uid="drawer",
        fpath=get_data_path(USD_DRAWER_ASSET if is_usd else DRAWER_ASSET),
        asset_physics_mode="overlay",
        # The USD cabinet extends 0.31 m below its root; lift it clear of the ground.
        init_pos=(0.0, 0.0, 0.8 if is_usd else 0.05),
        root_props=ArticulationRootPropertiesCfg(fixed_base=True),
        joint_drive_props=JointDrivePropertiesCfg(drive_type="none"),
        # Keep 90% of the URDF travel; retain the USD's named per-drawer limits.
        qpos_limits=None if is_usd else DRAWER_USER_QPOS_LIMITS,
        # Newton currently has no body-level damping setting. Remove the
        # Default backend's damping so both passive models use zero damping.
        attrs=RigidBodyPhysicsCfg(
            rigid_props=DefaultRigidBodyPropertiesCfg(
                linear_damping=0.0,
                angular_damping=0.0,
            )
        ),
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
        f"[INFO]: Config drive type: {articulation.cfg.joint_drive_props.drive_type}",
        flush=True,
    )
    print(f"[INFO]: Backend drive types: {backend_drive_types}", flush=True)
    print(
        f"[INFO]: Effective qpos limits: {articulation.get_qpos_limits()}", flush=True
    )
    return articulation


def apply_drawer_force(
    articulation: Articulation,
    target_qpos: torch.Tensor,
    joint_ids: list[int] | None = None,
) -> None:
    """Apply effort-limited PD control toward a drawer position.

    Args:
        articulation: Drawer articulation receiving the force.
        target_qpos: Target positions for every environment and selected joint.
        joint_ids: Joint indices to control; None selects all joints.
    """
    qpos = articulation.get_qpos()
    qvel = articulation.get_qvel()
    if joint_ids is not None:
        qpos, qvel = qpos[:, joint_ids], qvel[:, joint_ids]
    position_error = target_qpos - qpos
    joint_forces = DRAWER_POSITION_GAIN * position_error - DRAWER_VELOCITY_GAIN * qvel
    joint_forces = torch.clamp(
        joint_forces,
        min=-DRAWER_JOINT_FORCE_LIMIT,
        max=DRAWER_JOINT_FORCE_LIMIT,
    )
    articulation.set_qf(joint_forces, joint_ids=joint_ids)


def run_simulation(
    sim: SimulationManager,
    articulation: Articulation,
    max_steps: int | None = None,
    *,
    joint_name: str = "slide_rails",
    open_at_lower_limit: bool = False,
) -> None:
    """Open and close the drawer with effort-limited position tracking.

    Args:
        sim: Simulation manager to advance.
        articulation: Drawer articulation whose joints are updated.
        max_steps: Optional number of steps to run before returning.
        joint_name: Drawer joint to move; all other joints remain passive.
        open_at_lower_limit: Whether opening moves toward the lower joint limit.
    """
    joint_ids = [articulation.joint_names.index(joint_name)]
    qpos_limits = articulation.get_qpos_limits(joint_ids=joint_ids)
    closed_qpos = qpos_limits[..., 1 if open_at_lower_limit else 0]
    open_qpos = qpos_limits[..., 0 if open_at_lower_limit else 1]
    opening = True
    target_qpos = open_qpos
    step_count = 0
    print(
        f"[INFO]: Tracking {joint_name!r} toward the open position with effort limited to "
        f"+/-{DRAWER_JOINT_FORCE_LIMIT:.1f} N",
        flush=True,
    )
    try:
        while max_steps is None or step_count < max_steps:
            qpos = articulation.get_qpos()[:, joint_ids]
            qvel = articulation.get_qvel()[:, joint_ids]
            settled = torch.all(
                (torch.abs(qpos - target_qpos) <= JOINT_POSITION_TOLERANCE)
                & (torch.abs(qvel) <= JOINT_VELOCITY_TOLERANCE)
            ).item()
            if settled:
                reached_position = "open" if opening else "closed"
                print(
                    f"[INFO]: Drawer settled at {reached_position} position: "
                    f"qpos={qpos}, qvel={qvel}",
                    flush=True,
                )
                opening = not opening
                target_qpos = open_qpos if opening else closed_qpos
                target_position = "open" if opening else "closed"
                print(
                    f"[INFO]: Tracking the {target_position} position",
                    flush=True,
                )

            apply_drawer_force(
                articulation, target_qpos=target_qpos, joint_ids=joint_ids
            )
            sim.update(step=1)
            step_count += 1
    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        articulation.set_qf(
            torch.zeros_like(articulation.get_qpos()[:, joint_ids]),
            joint_ids=joint_ids,
        )


def main(args: argparse.Namespace | None = None) -> None:
    """Load and simulate a passive drawer articulation."""
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1")

    open_native_window = not args.headless and not args.viser

    # Construct the World without a window so Spawn can finish first. The
    # requested native window is opened explicitly after create_articulation().
    sim_cfg = SimulationManagerCfg(
        headless=True,
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
        articulation = create_articulation(sim, asset=args.asset)
        print(f"[INFO]: Initial joint positions: {articulation.get_qpos()}", flush=True)

        if open_native_window:
            sim.open_window()

        print("[INFO]: Running simulation. Press Ctrl+C to stop.", flush=True)
        run_simulation(
            sim,
            articulation,
            max_steps=args.max_steps,
            joint_name="top_slide" if args.asset == "usd" else "slide_rails",
            open_at_lower_limit=args.asset == "usd",
        )
    finally:
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    try:
        main(_cli_args)
    finally:
        SimulationManager.flush_cleanup_queue()
