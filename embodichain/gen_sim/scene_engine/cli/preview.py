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


from __future__ import annotations

import argparse
from pathlib import Path
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import (
    VisualizationCfg,
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)
from embodichain.gen_sim.scene_engine.pipeline.utils.scene_usd import (
    _add_articulations,
    load_scene_export_into_sim,
    load_scene_usd_into_sim,
)

if TYPE_CHECKING:
    from embodichain.lab.scripts.preview_joint_control import (
        ArticulationPreviewController,
    )
    from embodichain.lab.sim.objects import Articulation


def preview_scene_export(
    *,
    output_root: str | Path,
    device: str = "cpu",
    headless: bool = False,
    visualization: VisualizationCfg | None = None,
    joint_control: bool = True,
    use_usd: bool = False,
) -> None:
    """Preview a Scene Engine export or its materialized whole-scene USD.

    Args:
        output_root: Scene Engine output root containing ``scene_export/``.
        device: Simulation device, for example ``"cpu"`` or ``"cuda"``.
        headless: Load and validate the scene without an interactive preview.
        visualization: Optional live-visualization configuration.
        joint_control: Expose supported articulation joints in Viser.
        use_usd: Load ``scene_usd/scene.usda`` through its manifest instead of
            assembling the source GLB/USDC assets.
    """
    resolved_output_root = Path(output_root).expanduser().resolve()
    source_path = (
        resolved_output_root / "scene_usd" / "scene.usda"
        if use_usd
        else resolved_output_root / "scene_export" / "scene_config.json"
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Scene preview source not found: {source_path}")

    sim = SimulationManager(
        SimulationManagerCfg(
            width=1920,
            height=1080,
            headless=headless,
            physics_dt=1.0 / 100.0,
            sim_device=device,
            visualization=(
                VisualizationCfg() if visualization is None else visualization
            ),
        )
    )
    try:
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        articulations = (
            load_scene_usd_into_sim(sim=sim, output_root=resolved_output_root)
            if use_usd
            else load_scene_export_into_sim(
                sim=sim,
                output_root=resolved_output_root,
                force_static_rigids=True,
            )
        )

        is_viser = sim.sim_config.visualization.backend == "viser"
        joint_controller = _setup_viser_joint_control(
            sim=sim,
            articulations=articulations,
            enabled=is_viser and joint_control,
        )
        if headless and not is_viser:
            sim.update(step=1)
            print(f"Loaded scene preview headlessly: {source_path}")
            return

        if is_viser:
            print(f"Previewing in Viser: {source_path}")
        else:
            print(f"Previewing: {source_path}")
            # Native DexSim windows do not advance the manually-updated world
            # for us.  In particular, whole-scene USD import restores each
            # rigid's pose after constructing its wrapper; without an update,
            # the window renders the pre-restore mesh-node state while Viser
            # (which updates below) looks correct.
            sim.update(step=1)
            sim.open_window()
        print("Close with Ctrl-C.")
        while True:
            if is_viser:
                # Browser commands are applied on the simulation thread before capture.
                if joint_controller is not None:
                    joint_controller.update()
            sim.update(step=1)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping preview.")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _setup_viser_joint_control(
    *,
    sim: SimulationManager,
    articulations: list[Articulation],
    enabled: bool,
) -> ArticulationPreviewController | None:
    """Expose supported exported-articulation joints through the Viser runtime."""
    if not enabled or not articulations:
        return None
    runtime = sim.visualization_runtime
    if runtime is None:
        raise RuntimeError(
            "Viser joint control requires an active visualization runtime."
        )
    from embodichain.lab.scripts.preview_joint_control import (
        ArticulationPreviewController,
    )

    controller = ArticulationPreviewController(articulations, runtime)
    if not controller.has_controls:
        return None
    controller.update()
    runtime.set_joint_control_provider(controller)
    print("Viser articulation joint controls enabled.")
    return controller


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="embodichain preview-scene",
        description="Preview a Scene Engine scene export in EmbodiChain simulation.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Scene Engine output root containing scene_export/.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Simulation device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Load and validate the exported scene without opening a window.",
    )
    parser.add_argument(
        "--usd",
        action="store_true",
        help="Load scene_usd/scene.usda through its manifest.",
    )
    parser.add_argument(
        "--joint-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Expose supported articulation joints in Viser (default: enabled).",
    )
    add_viser_args_to_parser(parser)
    args = parser.parse_args(argv)
    preview_scene_export(
        output_root=args.output_root,
        device=args.device,
        headless=args.headless,
        visualization=visualization_cfg_from_args(args),
        joint_control=args.joint_control,
        use_usd=args.usd,
    )


if __name__ == "__main__":
    main()
