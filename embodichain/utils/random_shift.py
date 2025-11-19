# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
import random
from typing import Union
from typing import Union, List, Optional


def generate_random_perturbation(
    # max_angle: Union[float, int] = 0.1, max_trans: Union[float, int] = 0.1
    max_angle: Union[float, int] = 0.1,
    max_trans: Union[float, int] = 0.1,
    *,
    xy_range: Optional[Union[np.ndarray, List[float]]] = None,
    z_range: Optional[Union[np.ndarray, List[float]]] = None,
    z_angle_range_deg: Optional[Union[np.ndarray, List[float]]] = None,
) -> np.ndarray:
    """Generates a random small perturbation transformation matrix (4x4).

    This function creates a random 4x4 transformation matrix that includes a small
    rotation and translation. The rotation is generated using Rodrigues' formula
    for a random angle around the Z-axis, and the translation is a random vector
    within the specified range.

    Args:
        max_angle (Union[float, int]): Deprecated when using config ranges. Maximum rotation angle in radians
            (legacy fallback) used if z_angle_range_deg is None.
        max_trans (Union[float, int]): Deprecated when using config ranges. Maximum XY translation magnitude
            (legacy fallback) used if xy_range is None.
        xy_range (np.ndarray):  [x_min, x_max, y_min, y_max] in meters. If provided, overrides max_trans.
        z_range (np.ndarray): [z_min, z_max] in meters. Optional. Default 0 if not provided.
        z_angle_range_deg (np.ndarray): [deg_min, deg_max] in degrees. If provided, overrides max_angle.

    Returns:
        np.ndarray: A 4x4 transformation matrix representing the random perturbation.
    """
    # Generate random rotation (represented by axis angle)
    np.random.seed(None)
    random.seed(None)
    axis = np.array([0, 0, 1])
    if z_angle_range_deg is not None:
        deg_min, deg_max = float(z_angle_range_deg[0]), float(z_angle_range_deg[1])
        angle = np.deg2rad(random.uniform(deg_min, deg_max))
    else:
        angle = random.uniform(-max_angle, max_angle)
    # Axis Angular Rotation Matrix (using Rodrigues' formula)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    # Generate random translations
    t = np.random.uniform(-max_trans, max_trans, size=(3, 1))
    t[2, 0] = 0

    if xy_range is not None:
        x_min, x_max, y_min, y_max = [float(v) for v in xy_range]
        tx = np.random.uniform(x_min, x_max)
        ty = np.random.uniform(y_min, y_max)
    else:
        tx, ty = np.random.uniform(-max_trans, max_trans, size=(2,))

    if z_range is not None:
        z_min, z_max = [float(v) for v in z_range]
        tz = np.random.uniform(z_min, z_max)
    else:
        tz = 0.0

    t = np.array([[tx], [ty], [tz]], dtype=float)

    # combine rotation and translation into a 4x4 transformation matrix
    T_perturb = np.eye(4)
    T_perturb[:3, :3] = R
    T_perturb[:3, 3:] = t
    return T_perturb
