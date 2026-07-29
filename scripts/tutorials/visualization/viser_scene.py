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

"""Run a headless EmbodiChain scene in the browser with Viser.

Example:

.. code-block:: bash

    python scripts/tutorials/visualization/viser_scene.py --port 8080

Then open ``http://127.0.0.1:8080``. Press Ctrl+C to stop the example.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ArticulationCfg, RenderCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import (
    FrameOverlay,
    PointCloudOverlay,
    SceneExporter,
    SceneOverlays,
    TargetOverlay,
    TrajectoryOverlay,
    VisualizationCfg,
    VisualizationRuntime,
    ViserServerCfg,
)


def parse_arguments() -> argparse.Namespace:
    """Parse tutorial command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize a headless EmbodiChain scene through Viser."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Viser bind host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser bind port.")
    parser.add_argument("--scene-fps", type=float, default=15.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--renderer", default="hybrid")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after this many steps; zero runs until Ctrl+C.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace physics to wall time so browser motion is easy to inspect.",
    )
    parser.add_argument(
        "--robot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the SR5 articulation in addition to the rigid cube.",
    )
    return parser.parse_args()


def create_overlays() -> SceneOverlays:
    """Create frame, trajectory, target, and point-cloud V0 overlays."""
    angles = np.linspace(0.0, 2.0 * np.pi, 80, dtype=np.float32)
    trajectory = np.stack(
        (
            0.55 + 0.25 * np.cos(angles),
            0.25 * np.sin(angles),
            np.full_like(angles, 0.65),
        ),
        axis=1,
    )
    grid = np.linspace(-0.25, 0.25, 18, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid)
    points = np.stack(
        (xx.ravel(), yy.ravel(), np.full(xx.size, 0.015, dtype=np.float32)),
        axis=1,
    )
    return SceneOverlays(
        frames=(
            FrameOverlay(
                overlay_id="world_debug",
                position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        ),
        trajectories=(TrajectoryOverlay(overlay_id="demo_path", points=trajectory),),
        targets=(
            TargetOverlay(
                overlay_id="goal_pose",
                position=np.array([0.8, 0.0, 0.65], dtype=np.float32),
                wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ),
        ),
        point_clouds=(
            PointCloudOverlay(
                overlay_id="sampled_workspace",
                points=points,
                colors=(120, 150, 255),
                point_size=0.008,
            ),
        ),
    )


def main() -> None:
    """Build the scene, start Viser, and run the physics loop."""
    args = parse_arguments()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be greater than zero.")
    physics_dt = 1.0 / 100.0
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device=args.device,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_dt=physics_dt,
            num_envs=args.num_envs,
            arena_space=2.0,
        )
    )
    sim.add_rigid_object(
        RigidObjectCfg(
            uid="viser_cube",
            shape=CubeCfg(size=[0.16, 0.16, 0.16]),
            body_type="kinematic",
            init_pos=[0.45, 0.0, 0.5],
        )
    )
    articulation = None
    if args.robot:
        articulation = sim.add_articulation(
            ArticulationCfg(
                uid="sr5",
                fpath=get_data_path("Rokae/SR5/SR5.urdf"),
                build_pk_chain=False,
            )
        )
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    visualization_cfg = VisualizationCfg(
        backend="viser",
        scene_fps=args.scene_fps,
        env_ids=list(range(min(args.num_envs, 4))),
        max_visible_envs=4,
        viser_server=ViserServerCfg(host=args.host, port=args.port),
    )
    exporter = SceneExporter(sim, visualization_cfg)
    runtime = VisualizationRuntime(exporter, visualization_cfg)
    overlays = create_overlays()

    try:
        runtime.start()
        print(f"[INFO] Viser scene ready at {runtime.endpoint}")
        print("[INFO] Press Ctrl+C to stop.")
        step = 0
        while args.steps == 0 or step < args.steps:
            loop_started = time.perf_counter()
            sim_time = step * physics_dt
            if articulation is not None and articulation.dof > 0:
                qpos = articulation.get_qpos().clone()
                qpos[:, 0] = 0.55 * math.sin(sim_time * 1.2)
                articulation.set_qpos(qpos)
            cube = sim.get_rigid_object("viser_cube")
            cube_pose = cube.get_local_pose()
            cube_pose[:, 2] = 0.5 + 0.12 * math.sin(sim_time * 2.0)
            cube.set_local_pose(cube_pose)
            sim.update(step=1)
            runtime.capture(
                sim_step=step,
                sim_time=sim_time,
                overlays=overlays,
                force=not args.realtime,
            )
            step += 1
            if args.realtime:
                remaining = physics_dt - (time.perf_counter() - loop_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping Viser tutorial.")
    finally:
        runtime.stop()
        stats = runtime.stats
        print(
            "[INFO] Visualization stats: "
            f"captured={stats.captured_frames}, published={stats.published_frames}, "
            f"dropped={stats.dropped_frames}, bytes={stats.frame_bytes}"
        )
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
