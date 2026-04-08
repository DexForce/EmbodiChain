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

"""Standalone script to preview a USD or mesh asset in the simulation.

Usage examples::

    # Preview a rigid object from USD
    python -m embodichain.lab.scripts.preview_asset \\
        --asset_path /path/to/sugar_box.usda \\
        --asset_type rigid \\
        --preview

    # Preview an articulation from USD
    python -m embodichain.lab.scripts.preview_asset \\
        --asset_path /path/to/robot.usd \\
        --asset_type articulation \\
        --preview

    # Headless check (no render window)
    python -m embodichain.lab.scripts.preview_asset \\
        --asset_path /path/to/asset.usda \\
        --headless
"""

from __future__ import annotations

import argparse
import os

from typing import TYPE_CHECKING

from embodichain.utils.logger import log_info, log_warning, log_error

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager


def build_sim_cfg(args: argparse.Namespace):
    """Build a SimulationManagerCfg from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        SimulationManagerCfg: Simulation configuration.
    """
    from embodichain.lab.sim.cfg import RenderCfg
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg

    return SimulationManagerCfg(
        headless=args.headless,
        sim_device=args.sim_device,
        render_cfg=RenderCfg(renderer=args.renderer),
    )


def load_asset(sim: SimulationManager, args: argparse.Namespace):
    """Load the asset into the simulation.

    If ``--asset_type`` is not specified and the file is USD, the script will
    inspect the USD stage for articulation roots to decide between articulation
    and rigid object.  For non-USD files the default is always ``rigid``.

    Args:
        sim: The simulation manager instance.
        args: Parsed CLI arguments.

    Returns:
        The loaded asset object (RigidObject or Articulation).
    """
    from embodichain.lab.sim.cfg import (
        ArticulationCfg,
        LightCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import MeshCfg

    # --- light -----------------------------------------------------------
    sim.set_emission_light(intensity=150)

    asset_path = args.asset_path
    asset_type = args.asset_type
    # check suffix for asset, if is urdf, then treat as articulation, otherwise treat as rigid object
    if os.path.splitext(asset_path)[1].lower() == ".urdf":
        log_info(
            "URDF file detected. Setting asset type to 'articulation' automatically.",
            color="green",
        )
        asset_type = "articulation"

    uid = args.uid
    init_pos = tuple(args.init_pos)
    init_rot = tuple(args.init_rot)

    # --- load the asset --------------------------------------------------
    if asset_type == "articulation":
        log_info("Loading asset as articulation ...", color="green")
        cfg = ArticulationCfg(
            uid=uid,
            fpath=asset_path,
            init_pos=init_pos,
            init_rot=init_rot,
            fix_base=args.fix_base,
            use_usd_properties=args.use_usd_properties,
        )
        return sim.add_articulation(cfg)
    else:
        log_info("Loading asset as rigid object ...", color="green")
        cfg = RigidObjectCfg(
            uid=uid,
            shape=MeshCfg(fpath=asset_path),
            init_pos=init_pos,
            init_rot=init_rot,
            body_type=args.body_type,
            use_usd_properties=args.use_usd_properties,
        )
        return sim.add_rigid_object(cfg)


def preview(sim: SimulationManager, asset) -> None:
    """Enter interactive preview mode.

    Provides a simple REPL:

    * ``p`` — enter an IPython embed session with ``sim`` and ``asset`` in scope.
    * ``s <N>`` — step the simulation *N* times (default 10).
    * ``q`` — quit.

    Args:
        sim: The simulation manager instance.
        asset: The loaded asset (RigidObject or Articulation).
    """
    print("Press `p` to enter embed mode to interact with the asset.")
    print("Press `s <N>` to step the simulation N times (default 10).")
    print("Press `q` to quit the simulation.")

    while True:
        txt = input().strip()

        if txt == "q":
            break
        elif txt == "p":
            try:
                from IPython import embed
            except ImportError:
                log_error(
                    "IPython is not installed. Preview mode requires IPython to be "
                    "available. Please install it with `pip install ipython` and try again."
                )
                continue

            embed()
        elif txt.startswith("s"):
            parts = txt.split()
            n = int(parts[1]) if len(parts) > 1 else 10
            log_info(f"Stepping simulation {n} times ...")
            sim.update(step=n)
        else:
            log_warning(f"Unknown command: {txt!r}")


def main(args: argparse.Namespace) -> None:
    """Orchestrate: create simulation, load asset, optionally preview, destroy.

    Args:
        args: Parsed CLI arguments.
    """
    from embodichain.lab.sim.sim_manager import SimulationManager

    sim_cfg = build_sim_cfg(args)
    log_info("Creating simulation manager ...", color="green")
    sim = SimulationManager(sim_cfg)

    try:
        asset = load_asset(sim, args)
        log_info(f"Asset loaded successfully: {type(asset).__name__}", color="green")

        if args.preview:
            preview(sim, asset)
        elif not args.headless:
            # Keep window open when not headless and not in preview mode
            log_info("Window open. Press Ctrl+C to exit.", color="green")
            try:
                while True:
                    sim.update(step=1)
            except KeyboardInterrupt:
                pass
    finally:
        log_info("Destroying simulation ...", color="green")
        sim.destroy()


def cli():
    """Command-line interface for asset preview.

    Parses CLI arguments and launches the preview workflow.
    """
    parser = argparse.ArgumentParser(
        description="Preview a USD or mesh asset in the EmbodiChain simulation."
    )

    parser.add_argument(
        "--asset_path",
        type=str,
        required=True,
        help="Path to the asset file (.usd/.usda/.usdc/.obj/.stl/.glb).",
    )
    parser.add_argument(
        "--asset_type",
        type=str,
        choices=["rigid", "articulation"],
        default="rigid",
        help="Asset type. Auto-detected for USD files if not specified (default: rigid).",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default=None,
        help="Unique identifier for the asset in the scene. Derived from filename if not specified.",
    )
    parser.add_argument(
        "--init_pos",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.5],
        metavar=("X", "Y", "Z"),
        help="Initial position (default: 0 0 0.5).",
    )
    parser.add_argument(
        "--init_rot",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Initial rotation in degrees (default: 0 0 0).",
    )
    parser.add_argument(
        "--body_type",
        type=str,
        choices=["dynamic", "kinematic", "static"],
        default="dynamic",
        help="Body type for rigid objects (default: kinematic).",
    )
    parser.add_argument(
        "--use_usd_properties",
        action="store_true",
        default=False,
        help="Use physical properties from the USD file instead of defaults.",
    )
    parser.add_argument(
        "--fix_base",
        action="store_true",
        default=True,
        help="Fix the base of articulations (default: True).",
    )
    parser.add_argument(
        "--sim_device",
        type=str,
        default="cpu",
        help="Simulation device (default: cpu).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without rendering window.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["legacy", "hybrid", "fast-rt"],
        default="hybrid",
        help="Renderer backend (default: hybrid).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Enter interactive embed mode after loading.",
    )

    args = parser.parse_args()

    # Derive uid from filename if not specified
    if args.uid is None:
        args.uid = os.path.splitext(os.path.basename(args.asset_path))[0]

    main(args)


if __name__ == "__main__":
    cli()
