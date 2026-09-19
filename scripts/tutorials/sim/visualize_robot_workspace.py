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

"""Show a UR5 and its manipulability-colored workspace in the same scene.

Open the native DexSim viewer::

    python scripts/tutorials/sim/visualize_robot_workspace.py

Show the simulated robot and workspace in the EmbodiChain Viser viewer::

    python scripts/tutorials/sim/visualize_robot_workspace.py --backend viser

Render a native offscreen image for verification::

    python scripts/tutorials/sim/visualize_robot_workspace.py --headless

Joint-space sampling uses FK and allows different orientations. Cartesian
sampling (``--mode cartesian_space``) checks IK at the robot's initial tool
orientation and also draws rejected samples in gray. Neither mode establishes
collision-free motion. Colors describe the stored configurations; the cloud is
static and must be recomputed if the robot base is moved.

The scene uses EmbodiChain/DexSim rendering. Matplotlib is used only to look up
the existing colormap; no Matplotlib figure or window is created. The detailed
velocity-ellipsoid plots remain in workspace_manipulability_visualization.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.motion.workspace import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.motion.workspace.configs import (
    CacheConfig,
    MetricConfig,
    MetricType,
    SamplingConfig,
    VisualizationConfig,
)
from embodichain.lab.sim.motion.workspace.visualizers import (
    ManipulabilityColorCfg,
    ManipulabilityVisualizer,
    align_manipulability_scores,
)
from embodichain.lab.sim.robots import URRobotCfg
from embodichain.lab.visualization import VisualizationCfg, ViserServerCfg
from embodichain.utils import logger
from embodichain.utils.math import look_at_to_pose


def build_parser() -> argparse.ArgumentParser:
    """Return options for the single-robot workspace demonstration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("dexsim", "viser"), default="dexsim")
    parser.add_argument(
        "--mode", choices=("joint_space", "cartesian_space"), default="joint_space"
    )
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument("--base-position", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument(
        "--headless", action="store_true", help="Save one DexSim offscreen image."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/robot_workspace.png")
    )
    parser.add_argument(
        "--max-steps", type=int, help="Stop the viewer after this many physics steps."
    )
    return parser


def publish_workspace(
    sim: SimulationManager,
    results: dict[str, Any],
    *,
    backend: str,
    log_scale: bool = False,
) -> None:
    """Publish aligned workspace colors in the robot's simulation scene.

    Args:
        sim: Prepared simulation containing the analyzed robot in environment 0.
        results: Fresh analysis results for that robot and environment.
        backend: ``dexsim`` for the native viewer, or ``viser`` for the browser.
        log_scale: Use logarithmic color normalization.
    """
    point_set = align_manipulability_scores(results)
    # Robot.compute_batch_fk already includes the robot/chain-root transform.
    # Both overlay backends consume world coordinates: add only the arena offset.
    offset = sim.arena_offsets[0].detach().cpu().numpy()
    world_points = point_set.points + offset
    visualizer = ManipulabilityVisualizer(
        backend="sim_manager" if backend == "dexsim" else "viser",
        sim_manager=sim,
        control_part_name="arm",
        color_cfg=ManipulabilityColorCfg(
            log_scale=log_scale,
            # Native points use screen pixels; Viser points use scene units.
            point_size=3.0 if backend == "dexsim" else 0.006,
        ),
    )
    visualizer.visualize(
        world_points,
        scores=point_set.scores,
        reachable_mask=point_set.reachable_mask,
    )
    mapping = visualizer.map_colors(point_set.scores, point_set.reachable_mask)
    logger.log_info(
        f"Workspace: {point_set.num_reachable} reachable, "
        f"{point_set.num_unreachable} rejected; "
        f"raw w range={mapping.raw_range}, color range={mapping.clip_range}. "
        "Viridis: purple=low, yellow=high; gray=rejected."
    )


def save_scene_image(
    sim: SimulationManager, output: Path, eye: np.ndarray, target: np.ndarray
) -> None:
    """Render the robot and workspace through a native DexSim camera.

    Args:
        sim: Simulation with its point cloud already created.
        output: Destination PNG path.
        eye: World-frame camera position.
        target: World-frame look-at position.
    """
    pose = look_at_to_pose(eye, target, (0.0, 0.0, 1.0))[0].cpu().numpy()
    pose[:3, 1:3] *= -1  # OpenGL camera axes used by DexSim.
    camera = sim.get_env().create_camera("workspace_overview", 1280, 960)
    if hasattr(camera, "is_open") and not camera.is_open():
        camera.open_camera()
    camera.set_world_pose(np.asarray(pose, dtype=np.float32))
    camera.render()
    frame = np.ascontiguousarray(np.asarray(camera.get_rgb_map())[..., :3])
    if frame.size == 0:
        raise RuntimeError("The native camera returned an empty image.")
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output)
    logger.log_info(f"Saved robot and workspace to {output.resolve()}")


def main(argv: list[str] | None = None) -> None:
    """Create, prepare, analyze and display a real simulated UR5.

    Args:
        argv: Optional CLI argument list; defaults to process arguments.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.headless and args.backend != "dexsim":
        parser.error("--headless captures DexSim; use --backend dexsim")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=args.device,
            num_envs=1,
            width=1280,
            height=960,
            visualization=VisualizationCfg(
                backend="viser" if args.backend == "viser" else "none",
                viser_server=ViserServerCfg(port=args.port),
            ),
        )
    )
    try:
        robot = sim.add_robot(
            cfg=URRobotCfg.from_dict(
                {
                    "robot_type": "ur5",
                    "init_pos": tuple(args.base_position),
                    "init_qpos": [0.0, -1.2, 1.6, -1.2, -1.57, 0.0],
                }
            )
        )
        # add_robot declares the asset; prepare materializes robot state/solvers.
        sim.prepare()
        analyzer = WorkspaceAnalyzer(
            robot,
            WorkspaceAnalyzerConfig(
                mode=AnalysisMode(args.mode),
                control_part_name="arm",
                sampling=SamplingConfig(num_samples=args.num_samples, seed=args.seed),
                cache=CacheConfig(enabled=False),
                metric=MetricConfig(enabled_metrics=[MetricType.MANIPULABILITY]),
                visualization=VisualizationConfig(enabled=False),
                retain_diagnostics=True,
                ik_samples_per_point=4,
            ),
            sim_manager=sim,
        )
        results = analyzer.analyze(visualize=False)
        publish_workspace(sim, results, backend=args.backend, log_scale=args.log_scale)

        origin = np.asarray(args.base_position) + sim.arena_offsets[0].cpu().numpy()
        eye = (origin + (1.9, -2.3, 1.7)).astype(np.float32)
        target = (origin + (0.0, 0.0, 0.45)).astype(np.float32)
        if args.headless:
            save_scene_image(sim, args.output, eye, target)
            return
        if args.backend == "dexsim":
            if not sim.open_window():
                raise RuntimeError("Unable to open the native DexSim viewer.")
            sim.get_world().get_windows().set_look_at(
                eye=eye, look_at=target, up=np.array((0, 0, 1), dtype=np.float32)
            )
        else:
            logger.log_info(f"Workspace viewer: {sim.visualization_health.endpoint}")
        logger.log_info("Robot and workspace are ready. Press Ctrl+C to exit.")
        step = 0
        while args.max_steps is None or step < args.max_steps:
            sim.update(step=1)
            step += 1
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        logger.log_info("Stopping workspace viewer.")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
