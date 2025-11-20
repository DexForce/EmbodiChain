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

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import UnivariateSpline
from typing import Tuple, List


def quadratic_interp(
    t: np.ndarray, positions: np.ndarray, k: int = 2
) -> Tuple[UnivariateSpline, UnivariateSpline, UnivariateSpline]:
    """Performs quadratic interpolation on 3D position data.

    This function uses `UnivariateSpline` to interpolate the x, y, and z
    components of 3D position data over a given time array.

    Args:
        t (np.ndarray): A 1D array of time values.
        positions (np.ndarray): A 2D array of shape (n, 3) representing 3D positions,
            where each row corresponds to a position [x, y, z].
        k (int, optional): The degree of the spline. Defaults to 2.

    Returns:
        Tuple[UnivariateSpline, UnivariateSpline, UnivariateSpline]: A tuple containing
        the splines for the x, y, and z components of the positions.
    """
    spl_x = UnivariateSpline(t, positions[:, 0], k=k)
    spl_y = UnivariateSpline(t, positions[:, 1], k=k)
    spl_z = UnivariateSpline(t, positions[:, 2], k=k)
    return spl_x, spl_y, spl_z


def interpolate_with_curve(
    single_pose: np.ndarray,
    traj_a: List[np.ndarray],
    n: int = 1,
    num_interp_points: int = 20,
    lookback: int = 5,
) -> List[np.ndarray]:
    """Performs curve interpolation to maintain the original trajectory shape features.

    This function interpolates a trajectory segment using both position and rotation
    data, ensuring the interpolated segment preserves the shape characteristics of
    the original trajectory.

    Args:
        single_pose (np.ndarray): A single 4x4 homogeneous transformation matrix.
        traj_a (List[np.ndarray]): The original trajectory, represented as a list of
            4x4 homogeneous transformation matrices.
        n (int, optional): The fixed number of reserved ending poses. Must be between
            1 and the length of `traj_a`. Defaults to 1.
        num_interp_points (int, optional): The number of interpolation points to generate.
            Defaults to 20.
        lookback (int, optional): The number of prefix points used for shape reference.
            Defaults to 5.

    Returns:
        List[np.ndarray]: A list of 4x4 homogeneous transformation matrices representing
        the interpolated trajectory segment.

    Raises:
        ValueError: If `n` is not between 1 and the length of `traj_a`.
    """
    # check
    if n <= 0 or n > len(traj_a):
        raise ValueError("n must be between 1 and length of traj_a")
    lookback = min(lookback, len(traj_a) - n)

    # find segment to interpolate
    ref_start = max(0, len(traj_a) - n - lookback)
    reference_segment = traj_a[ref_start : len(traj_a) - n] + traj_a[-n:]
    all_poses = [single_pose] + reference_segment
    positions = np.array([pose[:3, 3] for pose in all_poses])
    rotations = [R.from_matrix(pose[:3, :3]) for pose in all_poses]

    # positions
    diffs = np.diff(positions, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dists = np.cumsum(dists)
    cum_dists = np.insert(cum_dists, 0, 0)
    t_normalized = cum_dists / cum_dists[-1]
    cs_x, cs_y, cs_z = quadratic_interp(t_normalized, positions)
    # rotations
    key_rots = R.from_matrix(np.array([r.as_matrix() for r in rotations]))
    slerp = Slerp(t_normalized, key_rots)

    # generate interpolated segment
    interp_range = t_normalized[1]
    interp_t = np.linspace(0, interp_range, num_interp_points)
    interp_positions = np.vstack([cs_x(interp_t), cs_y(interp_t), cs_z(interp_t)]).T
    interp_rotations = slerp(interp_t)
    interp_segment = []
    for pos, rot in zip(interp_positions, interp_rotations):
        mat = np.eye(4)
        mat[:3, :3] = rot.as_matrix()
        mat[:3, 3] = pos
        interp_segment.append(mat)

    return interp_segment


def interpolate_poses(
    pose1: np.ndarray, pose2: np.ndarray, num: int = 10
) -> np.ndarray:
    """Interpolates between two 4x4 transformation matrices.

    This function performs interpolation between two poses represented as 4x4
    homogeneous transformation matrices. It interpolates both the translation
    and rotation components, generating a specified number of intermediate poses.

    Args:
        pose1 (np.ndarray): The starting pose, a 4x4 homogeneous transformation matrix.
        pose2 (np.ndarray): The ending pose, a 4x4 homogeneous transformation matrix.
        num (int, optional): The number of interpolated poses to generate. Defaults to 10.

    Returns:
        np.ndarray: An array of shape (num, 4, 4) containing the interpolated poses.
    """
    # extract
    t1, t2 = pose1[:3, 3], pose2[:3, 3]
    r1, r2 = pose1[:3, :3], pose2[:3, :3]

    # interolation
    t_interp = np.array([np.linspace(t1[i], t2[i], num) for i in range(3)]).T
    from scipy.spatial.transform import Slerp, Rotation

    rotations = Rotation.from_matrix([r1, r2])
    slerp = Slerp([0, 1], rotations)
    r_interp = slerp(np.linspace(0, 1, num)).as_matrix()

    # generate poses
    poses = np.zeros((num, 4, 4))
    for i in range(num):
        poses[i, :3, :3] = r_interp[i]
        poses[i, :3, 3] = t_interp[i]
        poses[i, 3, 3] = 1

    return poses
