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

"""Visualize a deterministic RGB point cloud with DexSim.

Run with::

    python scripts/tutorials/sim/visualize_point_cloud.py

Render one frame with an offscreen camera, without opening a native window::

    python scripts/tutorials/sim/visualize_point_cloud.py --headless

The viewer should show red X, green Y, and blue Z point axes. The script uses
``uint8`` per-point colors to exercise the color normalization in
``SimulationManager.visualize_point_cloud``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import VisualizationCfg
from embodichain.utils import logger
from embodichain.utils.math import look_at_to_pose

CAMERA_EYE = (2.0, -2.0, 1.5)
CAMERA_TARGET = (0.0, 0.0, 0.35)
CAMERA_UP = (0.0, 0.0, 1.0)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DEFAULT_OUTPUT_PATH = Path("outputs/point_cloud_visualization.png")


def build_demo_point_cloud(
    num_points_per_axis: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a color-coded three-axis point cloud for visual inspection.

    Args:
        num_points_per_axis: Number of points rendered for each axis.

    Returns:
        Point positions with shape ``(3 * N, 3)`` and ``uint8`` RGB colors
        with the same leading dimension.
    """
    horizontal = np.linspace(-0.75, 0.75, num_points_per_axis, dtype=np.float32)
    vertical = np.linspace(0.05, 1.25, num_points_per_axis, dtype=np.float32)
    ground_height = np.full_like(horizontal, 0.05)
    zeros = np.zeros_like(horizontal)

    x_axis = np.column_stack((horizontal, zeros, ground_height))
    y_axis = np.column_stack((zeros, horizontal, ground_height))
    z_axis = np.column_stack((zeros, zeros, vertical))
    points = np.concatenate((x_axis, y_axis, z_axis), axis=0)

    red = np.full((num_points_per_axis, 3), (255, 0, 0), dtype=np.uint8)
    green = np.full((num_points_per_axis, 3), (0, 255, 0), dtype=np.uint8)
    blue = np.full((num_points_per_axis, 3), (0, 0, 255), dtype=np.uint8)
    colors = np.concatenate((red, green, blue), axis=0)
    return points, colors


def build_camera_pose() -> np.ndarray:
    """Build the offscreen camera pose for the point-cloud overview."""
    pose = look_at_to_pose(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP)[0].cpu().numpy()
    # DexSim cameras use the OpenGL camera-axis convention.
    pose[:3, 1] = -pose[:3, 1]
    pose[:3, 2] = -pose[:3, 2]
    return np.asarray(pose, dtype=np.float32)


def render_headless_frame(sim: SimulationManager, output_path: Path) -> None:
    """Render the point cloud once with an offscreen camera and save a PNG.

    Args:
        sim: Simulation containing the point cloud to render.
        output_path: PNG destination. Its parent directory is created if needed.
    """
    camera = sim.get_env().create_camera(
        "point_cloud_tutorial_camera", FRAME_WIDTH, FRAME_HEIGHT
    )
    if hasattr(camera, "is_open") and camera.is_open() is False:
        camera.open_camera()

    camera.set_world_pose(build_camera_pose())
    camera.render()
    frame = np.ascontiguousarray(np.asarray(camera.get_rgb_map())[..., :3])
    if frame.size == 0:
        raise RuntimeError("The offscreen camera returned an empty RGB frame.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)
    logger.log_info(f"Saved offscreen point-cloud frame to {output_path}.")


def parse_args() -> argparse.Namespace:
    """Parse the tutorial's optional offscreen-rendering arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Render one PNG with an offscreen camera instead of opening the viewer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "PNG path used with --headless "
            f"(default: {DEFAULT_OUTPUT_PATH.as_posix()})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Create the RGB point cloud and display or render it once."""
    args = parse_args()
    sim = SimulationManager(
        SimulationManagerCfg(
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            headless=True,
            visualization=VisualizationCfg(),
        )
    )
    try:
        points, colors = build_demo_point_cloud()
        sim.visualize_point_cloud(
            points=points,
            colors=colors,
            point_size=8.0,
            name="rgb_point_cloud_axes",
        )
        if args.headless:
            sim.update(step=1)
            render_headless_frame(sim, args.output)
            return

        if not sim.open_window():
            raise RuntimeError("Unable to open the native DexSim viewer.")

        sim.get_world().get_windows().set_look_at(
            eye=np.array(CAMERA_EYE, dtype=np.float32),
            look_at=np.array(CAMERA_TARGET, dtype=np.float32),
            up=np.array(CAMERA_UP, dtype=np.float32),
        )
        logger.log_info(
            "Point-cloud viewer open: red=X, green=Y, blue=Z. Press Ctrl+C to exit."
        )
        while True:
            sim.update(step=1)
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        logger.log_info("Stopping point-cloud viewer.")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
