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

"""Preview a USD or mesh asset in the simulation.

Usage examples::

    # Preview a rigid object from USD
    embodichain preview-asset \\
        --asset_path /path/to/sugar_box.usda \\
        --asset_type rigid \\
        --preview

    # Preview an articulation from USD
    embodichain preview-asset \\
        --asset_path /path/to/robot.usd \\
        --asset_type articulation \\
        --preview

    # Headless check (no render window)
    embodichain preview-asset \\
        --asset_path /path/to/asset.usda \\
        --headless

    # Preview in a browser with Viser
    embodichain preview-asset \\
        --asset_path /path/to/asset.usda \\
        --viser

    # Preview with a built-in environment map
    embodichain preview-asset \\
        --asset_path /path/to/sugar_box.usda \\
        --env_map "Studio"

    # Preview with a custom HDR environment map
    embodichain preview-asset \\
        --asset_path /path/to/sugar_box.usda \\
        --env_map /path/to/environment.hdr
"""

from __future__ import annotations

import argparse
import os

from collections.abc import Sequence
from typing import TYPE_CHECKING

from embodichain.utils.logger import log_info, log_warning, log_error

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Articulation, RigidObject
    from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg
    from embodichain.lab.scripts.preview_joint_control import (
        ArticulationPreviewController,
    )


def build_sim_cfg(args: argparse.Namespace) -> SimulationManagerCfg:
    """Build a SimulationManagerCfg from CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        SimulationManagerCfg: Simulation configuration.
    """
    from embodichain.lab.sim.cfg import RenderCfg
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg
    from embodichain.lab.visualization import visualization_cfg_from_args

    return SimulationManagerCfg(
        headless=args.headless,
        sim_device=args.sim_device,
        render_cfg=RenderCfg(renderer=args.renderer),
        visualization=visualization_cfg_from_args(args),
    )


def load_assets(
    sim: SimulationManager,
    args: argparse.Namespace,
) -> list[RigidObject | Articulation]:
    """Load one or more assets into the simulation.

    URDF files are always loaded as articulations. Other file types use the
    value of ``--asset_type``, which defaults to ``rigid``.

    Args:
        sim: The simulation manager instance.
        args: Parsed CLI arguments.

    Returns:
        list: Loaded asset objects (RigidObject or Articulation).
    """
    from embodichain.lab.sim.cfg import (
        ArticulationCfg,
        LightCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import MeshCfg

    asset_paths = args.asset_path
    init_pos = tuple(args.init_pos)
    init_rot = tuple(args.init_rot)
    spacing = float(args.asset_spacing)

    loaded_assets = []
    for idx, asset_path in enumerate(asset_paths):
        asset_suffix = os.path.splitext(asset_path)[1].lower()
        asset_type = args.asset_type
        # URDF is always loaded as articulation.
        if asset_suffix == ".urdf":
            log_info(
                f"URDF file detected for {asset_path}. "
                "Setting asset type to 'articulation' automatically.",
                color="green",
            )
            asset_type = "articulation"

        if args.uid is None:
            base_uid = os.path.splitext(os.path.basename(asset_path))[0]
        else:
            base_uid = args.uid
        uid = base_uid if len(asset_paths) == 1 else f"{base_uid}_{idx}"

        asset_init_pos = (
            init_pos[0] + idx * spacing,
            init_pos[1],
            init_pos[2],
        )

        # --- load the asset --------------------------------------------------
        if asset_type == "articulation":
            log_info(
                f"Loading asset as articulation: {asset_path} "
                f"(uid={uid}, pos={asset_init_pos}) ...",
                color="green",
            )
            cfg = ArticulationCfg(
                uid=uid,
                fpath=asset_path,
                init_pos=asset_init_pos,
                init_rot=init_rot,
                fix_base=args.fix_base,
                use_usd_properties=args.use_usd_properties,
                # The auxiliary pytorch-kinematics chain only accepts URDF XML.
                build_pk_chain=asset_suffix not in {".usd", ".usda", ".usdc"},
            )
            loaded_assets.append(sim.add_articulation(cfg))
        else:
            log_info(
                f"Loading asset as rigid object: {asset_path} "
                f"(uid={uid}, pos={asset_init_pos}) ...",
                color="green",
            )
            cfg = RigidObjectCfg(
                uid=uid,
                shape=MeshCfg(fpath=asset_path),
                init_pos=asset_init_pos,
                init_rot=init_rot,
                body_type=args.body_type,
                use_usd_properties=args.use_usd_properties,
            )
            loaded_assets.append(sim.add_rigid_object(cfg))

    return loaded_assets


def preview(
    sim: SimulationManager,
    assets: list[RigidObject | Articulation],
    joint_controller: ArticulationPreviewController | None = None,
) -> None:
    """Enter interactive preview mode.

    Provides a simple REPL:

    * ``p`` — enter an IPython embed session with ``sim`` and ``assets`` in scope.
    * ``s <N>`` — step the simulation *N* times (default 10).
    * ``q`` — quit.

    Args:
        sim: The simulation manager instance.
        assets: Loaded assets (list of RigidObject/Articulation).
        joint_controller: Optional Viser articulation preview controller.
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
            _step_preview_simulation(sim, joint_controller, step=n)
        else:
            log_warning(f"Unknown command: {txt!r}")


def _publish_loaded_assets(sim: SimulationManager, args: argparse.Namespace) -> None:
    """Immediately publish assets loaded after Viser startup.

    Args:
        sim: Simulation manager containing the newly loaded assets.
        args: Parsed CLI arguments.
    """
    if getattr(args, "viser", False):
        sim.capture_visualization_safely(force=True)


def _setup_viser_joint_control(
    sim: SimulationManager,
    assets: list[RigidObject | Articulation],
    args: argparse.Namespace,
) -> ArticulationPreviewController | None:
    """Register articulation joint controls with the active Viser runtime."""
    if not getattr(args, "viser", False) or not getattr(
        args,
        "joint_control",
        True,
    ):
        return None

    from embodichain.lab.scripts.preview_joint_control import (
        ArticulationPreviewController,
    )
    from embodichain.lab.sim.objects import Articulation

    articulations = [asset for asset in assets if isinstance(asset, Articulation)]
    if not articulations:
        return None
    runtime = sim.visualization_runtime
    if runtime is None:
        log_warning("Viser is enabled but its visualization runtime is unavailable.")
        return None

    controller = ArticulationPreviewController(articulations, runtime)
    if not controller.has_controls:
        log_warning("No supported independent articulation joints were found.")
        return None
    controller.update()
    runtime.set_joint_control_provider(controller)
    log_info(
        "Viser articulation joint controls enabled.",
        color="green",
    )
    return controller


def _step_preview_simulation(
    sim: SimulationManager,
    joint_controller: ArticulationPreviewController | None,
    *,
    step: int = 1,
) -> None:
    """Apply pending preview controls before every physics step."""
    for _ in range(step):
        if joint_controller is not None:
            joint_controller.update()
        sim.update(step=1)


def _run_preview_mode(
    sim: SimulationManager,
    assets: list[RigidObject | Articulation],
    args: argparse.Namespace,
    joint_controller: ArticulationPreviewController | None = None,
) -> None:
    """Run the interactive REPL or keep the selected visualizer alive.

    Args:
        sim: Active simulation manager.
        assets: Loaded assets exposed to the interactive REPL.
        args: Parsed CLI arguments.
        joint_controller: Optional Viser articulation preview controller.
    """
    if args.preview:
        preview(sim, assets, joint_controller)
        return

    viser_enabled = bool(getattr(args, "viser", False))
    if args.headless and not viser_enabled:
        return

    target = "Viser browser preview" if viser_enabled else "Simulation window"
    log_info(f"{target} open. Press Ctrl+C to exit.", color="green")
    try:
        while True:
            _step_preview_simulation(sim, joint_controller)
    except KeyboardInterrupt:
        pass


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
        if args.env_map:
            log_info(f"Setting environment map: {args.env_map} ...", color="green")
            sim.set_indirect_lighting(args.env_map)

        assets = load_assets(sim, args)
        log_info(f"Loaded {len(assets)} asset(s) successfully.", color="green")
        joint_controller = _setup_viser_joint_control(sim, assets, args)
        _publish_loaded_assets(sim, args)
        _run_preview_mode(sim, assets, args, joint_controller)
    finally:
        log_info("Destroying simulation ...", color="green")
        sim.destroy()


def _create_parser() -> argparse.ArgumentParser:
    """Create the complete asset-preview argument parser."""
    parser = argparse.ArgumentParser(
        prog="embodichain preview-asset",
        description="Preview a USD or mesh asset in the EmbodiChain simulation.",
    )

    parser.add_argument(
        "--asset_path",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to asset file(s) (.usd/.usda/.usdc/.obj/.stl/.glb/.urdf).",
    )
    parser.add_argument(
        "--asset_type",
        type=str,
        choices=["rigid", "articulation"],
        default="rigid",
        help="Asset type for non-URDF files (default: rigid).",
    )
    parser.add_argument(
        "--uid",
        type=str,
        default=None,
        help=(
            "Base unique identifier for assets in the scene. If multiple assets are "
            "provided, suffix '_<index>' is appended automatically."
        ),
    )
    parser.add_argument(
        "--asset_spacing",
        type=float,
        default=1.0,
        help="Relative spacing in meters between assets along +X axis (default: 1.0).",
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
        default="kinematic",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fix or unfix the base of articulations (default: fixed).",
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
        choices=["hybrid", "fast-rt", "offline-rt"],
        default="hybrid",
        help="Renderer backend (default: hybrid).",
    )
    parser.add_argument(
        "--env_map",
        type=str,
        default=None,
        help=(
            "Environment map for indirect lighting. Accepts a built-in IBL resource "
            "name (e.g. 'Studio') or an absolute file path (.hdr/.png/.exr)."
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=False,
        help="Enter interactive embed mode after loading.",
    )
    parser.add_argument(
        "--joint-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expose supported articulation joints in Viser (default: enabled; "
            "use --no-joint-control to disable)."
        ),
    )

    from embodichain.lab.visualization import add_viser_args_to_parser

    add_viser_args_to_parser(parser)
    return parser


def cli(argv: Sequence[str] | None = None) -> None:
    """Command-line interface for asset preview.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    args = _create_parser().parse_args(argv)

    main(args)


if __name__ == "__main__":
    cli()


__all__ = ["build_sim_cfg", "cli", "load_assets", "main", "preview"]
