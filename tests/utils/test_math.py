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

import numpy as np
import pytest
import torch

from embodichain.utils.math import (
    convert_quat,
    default_orientation,
    inv_transform,
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    trans_matrix_to_xyz_quat,
    xyz_quat_to_4x4_matrix,
)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("kind", ["identity", "translation", "rotation_translation"])
def test_inv_transform_preserves_backend_dtype_and_input(backend, dtype, kind) -> None:
    transform = np.eye(4, dtype=dtype)
    if kind != "identity":
        transform[:3, 3] = [0.123456789, -0.987654321, 0.13579]
    if kind == "rotation_translation":
        # +90 degrees about Y exposes overlapping transpose assignments.
        transform[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    expected = np.linalg.inv(transform)
    if backend == "torch":
        transform = torch.from_numpy(transform)
        original = transform.clone()
    else:
        original = transform.copy()

    actual = inv_transform(transform)

    assert type(actual) is type(transform)
    assert actual.dtype == transform.dtype
    if backend == "torch":
        assert actual.device == transform.device
        torch.testing.assert_close(actual, torch.from_numpy(expected))
        torch.testing.assert_close(actual @ transform, torch.eye(4, dtype=actual.dtype))
        torch.testing.assert_close(transform, original, atol=0, rtol=0)
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_allclose(actual @ transform, np.eye(4), atol=1e-7)
        np.testing.assert_array_equal(transform, original)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_inv_transform_accepts_noncontiguous_input(backend) -> None:
    storage = np.zeros((8, 8), dtype=np.float64)
    storage[::2, ::2] = [
        [0, -1, 0, 0.2],
        [1, 0, 0, -0.4],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1],
    ]
    expected = np.linalg.inv(storage[::2, ::2])
    if backend == "torch":
        transform = torch.from_numpy(storage)[::2, ::2]
        assert not transform.is_contiguous()
        torch.testing.assert_close(inv_transform(transform), torch.from_numpy(expected))
    else:
        transform = storage[::2, ::2]
        assert not transform.flags.c_contiguous
        np.testing.assert_allclose(inv_transform(transform), expected)


def test_inv_transform_preserves_autograd() -> None:
    # A differentiable rigid pose: one rotation angle and three translations.
    parameters = torch.tensor(
        [0.4, 0.2, -0.3, 0.5], dtype=torch.float64, requires_grad=True
    )

    def inverse_from_parameters(values):
        angle, x, y, z = values.unbind()
        zero, one = values.new_zeros(()), values.new_ones(())
        c, s = angle.cos(), angle.sin()
        transform = torch.stack(
            (
                torch.stack((c, -s, zero, x)),
                torch.stack((s, c, zero, y)),
                torch.stack((zero, zero, one, z)),
                torch.stack((zero, zero, zero, one)),
            )
        )
        return inv_transform(transform)

    assert torch.autograd.gradcheck(inverse_from_parameters, (parameters,))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_inv_transform_keeps_cuda_device() -> None:
    transform = torch.eye(4, dtype=torch.float64, device="cuda")
    transform[:3, 3] = transform.new_tensor([0.1, -0.2, 0.3])

    inverse = inv_transform(transform)

    assert inverse.device == transform.device
    torch.testing.assert_close(inverse, torch.linalg.inv(transform))


@pytest.mark.parametrize("backend", [np.zeros, torch.zeros])
@pytest.mark.parametrize("shape", [(3, 3), (4, 5), (1, 4, 4)])
def test_inv_transform_rejects_non_single_pose_shapes(backend, shape) -> None:
    with pytest.raises(ValueError, match="single.*4, 4"):
        inv_transform(backend(shape))


def test_inv_transform_rejects_unsupported_input_type() -> None:
    with pytest.raises(TypeError, match="NumPy array or Torch tensor"):
        inv_transform(np.eye(4).tolist())


def test_inv_transform_legacy_import_is_compatible() -> None:
    from embodichain.utils.utility import inv_transform as legacy_inv_transform

    assert legacy_inv_transform is inv_transform


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


def test_named_quaternion_boundary_helpers_make_direction_explicit() -> None:
    xyzw = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    wxyz = torch.tensor([[4.0, 1.0, 2.0, 3.0]])

    torch.testing.assert_close(quat_xyzw_to_wxyz(xyzw), wxyz)
    torch.testing.assert_close(quat_wxyz_to_xyzw(wxyz), xyzw)


def test_named_quaternion_boundary_helpers_preserve_numpy_backend() -> None:
    xyzw = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    wxyz = np.array([[4.0, 1.0, 2.0, 3.0]], dtype=np.float32)

    np.testing.assert_array_equal(quat_xyzw_to_wxyz(xyzw), wxyz)
    np.testing.assert_array_equal(quat_wxyz_to_xyzw(wxyz), xyzw)
