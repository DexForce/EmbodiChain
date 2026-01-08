# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import numpy as np
from typing import List, Tuple
from embodichain.utils import logger
from embodichain.lab.sim.objects import Robot


def sample_circular_plane_reachability(
    robot: Robot,
    control_part: str,
    ref_xpos: np.ndarray,
    center_xy: Tuple[float, float],
    z_height: float,
    radius: float,
    resolution: float = 0.01,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Sample a circular plane at a given z height and return all IK-reachable points.

    Args:
        robot: Robot object
        control_part: Name of the control part (e.g., "left_arm")
        ref_xpos: Reference end-effector pose matrix (4x4), defines orientation for sampled points
        center_xy: Center (x, y) of the circle
        z_height: z coordinate of the sampling plane
        radius: Radius of the sampling circle (meters)
        resolution: Sampling resolution (meters), default 0.05

    Returns:
        reachable_positions: List of reachable xyz positions
        reachable_poses: List of reachable full pose matrices (4x4)
    """

    center_x, center_y = center_xy

    logger.log_info(f"Start sampling in circular plane...")
    logger.log_info(f"- Center: ({center_x:.3f}, {center_y:.3f}, {z_height:.3f}) m")
    logger.log_info(f"- Radius: {radius:.3f} m")
    logger.log_info(f"- Resolution: {resolution:.3f} m")

    # 1. Generate sample points in the circular plane
    sample_points = []

    # Sample in a square grid, then filter points inside the circle
    num_samples = int(np.ceil(2 * radius / resolution))
    x_range = np.linspace(-radius, radius, num_samples)
    y_range = np.linspace(-radius, radius, num_samples)

    for dx in x_range:
        for dy in y_range:
            if dx**2 + dy**2 <= radius**2:
                sample_points.append(np.array([center_x + dx, center_y + dy, z_height]))

    logger.log_info(f"- Total sample points: {len(sample_points)}")

    # 2. IK test (batch)
    num_points = len(sample_points)
    reachable_positions = []
    reachable_poses = []

    logger.log_info(f"Start IK validation for {num_points} sample points...")

    # Pre-build all target poses
    target_poses_list = []
    for point in sample_points:
        target_pose = ref_xpos.copy()
        target_pose[:3, 3] = point
        target_poses_list.append(target_pose)

    import time

    t0 = time.time()
    ret, _ = robot.compute_batch_ik(
        pose=[target_poses_list], joint_seed=None, name=control_part
    )
    t1 = time.time()
    logger.log_info(f"compute_batch_ik time: {t1-t0:.3f} seconds")

    for j, is_valid in enumerate(ret[0]):
        if is_valid:
            reachable_positions.append(sample_points[j])
            reachable_poses.append(target_poses_list[j])

    logger.log_info(f"- Reachable points: {len(reachable_positions)}")
    if len(sample_points) > 0:
        logger.log_info(
            f"- Reachability: {len(reachable_positions)/len(sample_points)*100:.1f}%"
        )

    return reachable_positions, reachable_poses
