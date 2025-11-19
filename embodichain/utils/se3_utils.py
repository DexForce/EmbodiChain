# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np

from scipy.spatial.transform import Rotation as R


def normalize_vector(v):
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def cross_product(u, v):
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = np.stack((i, j, k), axis=1)
    return out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]

    x = normalize_vector(x_raw)
    z = cross_product(x, y_raw)
    z = normalize_vector(z)
    y = cross_product(z, x)

    x = x.reshape(-1, 3, 1)
    y = y.reshape(-1, 3, 1)
    z = z.reshape(-1, 3, 1)
    matrix = np.concatenate((x, y, z), axis=2)
    return matrix


def compute_4x4_transform_matrix_from_ortho6dxyz(ortho6dxyz):
    matrix = compute_rotation_matrix_from_ortho6d(ortho6dxyz[..., 3:9])

    transform_matrix = np.eye(4)
    transform_matrix = np.repeat(
        transform_matrix[np.newaxis, :, :], len(ortho6dxyz), axis=0
    )
    transform_matrix[:, :3, :3] = matrix
    transform_matrix[:, :3, 3] = ortho6dxyz[:, :3]
    return transform_matrix


def convert_rotation_matrix_to_euler(rotmat):
    """
    Convert rotation matrix (3x3) to Euler angles (rpy).
    """
    r = R.from_matrix(rotmat)
    euler = r.as_euler("xyz", degrees=False)

    return euler
