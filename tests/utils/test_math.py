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

import torch

from embodichain.utils.math import (
    convert_quat,
    default_orientation,
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
    trans_matrix_to_xyz_quat,
    xyz_quat_to_4x4_matrix,
)


def _distinct_xyzw() -> torch.Tensor:
    """Return a normalized quaternion whose components expose order mistakes."""
    quaternion = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)


def test_quaternion_matrix_round_trip_uses_xyzw() -> None:
    quaternion = _distinct_xyzw()

    rotation = matrix_from_quat(quaternion)
    restored = quat_from_matrix(rotation)

    torch.testing.assert_close(restored, quaternion, atol=1.0e-6, rtol=1.0e-6)


def test_quaternion_product_and_conjugate_return_xyzw_identity() -> None:
    quaternion = _distinct_xyzw()

    product = quat_mul(quaternion, quat_conjugate(quaternion))

    torch.testing.assert_close(
        product,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_quaternion_application_reads_scalar_from_last_component() -> None:
    half_sqrt_two = 2.0**-0.5
    z_quarter_turn_xyzw = torch.tensor(
        [[0.0, 0.0, half_sqrt_two, half_sqrt_two]], dtype=torch.float32
    )

    rotated = quat_apply(z_quarter_turn_xyzw, torch.tensor([[1.0, 0.0, 0.0]]))

    torch.testing.assert_close(
        rotated,
        torch.tensor([[0.0, 1.0, 0.0]]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_pose_vector_round_trip_uses_xyz_plus_xyzw() -> None:
    pose = torch.cat((torch.tensor([[0.25, -0.5, 0.75]]), _distinct_xyzw()), dim=-1)

    restored = trans_matrix_to_xyz_quat(xyz_quat_to_4x4_matrix(pose))

    torch.testing.assert_close(restored, pose, atol=1.0e-6, rtol=1.0e-6)


def test_identity_and_boundary_conversion_orders_are_explicit() -> None:
    xyzw = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    torch.testing.assert_close(
        default_orientation(1, "cpu"), torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    )
    torch.testing.assert_close(
        convert_quat(xyzw, to="wxyz"), torch.tensor([[4.0, 1.0, 2.0, 3.0]])
    )
