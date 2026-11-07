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

import inspect
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from embodichain.utils import math as math_utils
from embodichain.utils.math import (
    convert_quat,
    default_orientation,
    inv_transform,
    interpolate_poses,
    interpolate_rotations,
    look_at_to_pose,
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


def test_math_wildcard_import_exposes_public_math_functions() -> None:
    expected = {
        name
        for name, value in vars(math_utils).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or isinstance(value, torch.jit.ScriptFunction))
    }
    namespace: dict[str, object] = {}
    exec("from embodichain.utils.math import *", namespace)

    assert {name for name in namespace if not name.startswith("_")} == expected


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("custom_up", [False, True])
def test_look_at_keeps_degenerate_rows_independent(
    device: str, custom_up: bool
) -> None:
    eye = torch.tensor(
        [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device
    )
    target = torch.zeros_like(eye)
    up = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], device=device
    )

    pose = (
        look_at_to_pose(eye, target, up) if custom_up else look_at_to_pose(eye, target)
    )
    isolated = torch.cat(
        [
            (
                look_at_to_pose(eye[row], target[row], up[row])
                if custom_up
                else look_at_to_pose(eye[row], target[row])
            )
            for row in range(len(eye))
        ]
    )

    assert pose.device == eye.device
    assert torch.isfinite(pose).all()
    torch.testing.assert_close(pose, isolated)
    torch.testing.assert_close(pose[:, :3, 3], eye)
    root_half = 2.0**-0.5  # The first camera looks diagonally down the X/Z plane.
    torch.testing.assert_close(
        pose[0, :3, :3],
        eye.new_tensor(
            [
                [0.0, root_half, -root_half],
                [1.0, 0.0, 0.0],
                [0.0, -root_half, -root_half],
            ]
        ),
    )
    # Keep the existing default-up convention for the vertical camera.
    torch.testing.assert_close(
        pose[1, :3, :3], torch.diag(eye.new_tensor([-1.0, 1.0, -1.0]))
    )
    rotation = pose[:, :3, :3]
    torch.testing.assert_close(
        rotation.transpose(1, 2) @ rotation,
        torch.eye(3, device=device).expand(len(eye), 3, 3),
    )


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_look_at_promotes_integer_eye_before_converting_float_coordinates(
    device: str,
) -> None:
    eye = torch.tensor([[0, 0, 1]], device=device)
    target = [[0.25, 0.5, 0.0]]
    up = [[0.0, 0.25, 1.0]]

    pose = look_at_to_pose(eye, target, up)

    assert pose.device == eye.device
    assert torch.isfinite(pose).all()
    torch.testing.assert_close(pose, look_at_to_pose(eye.float(), target, up))


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


@pytest.mark.parametrize("backend", [np.asarray, torch.as_tensor])
@pytest.mark.parametrize(
    "kind",
    ["zero", "singular", "scale", "shear", "reflection", "last_row", "nan", "inf"],
)
def test_inv_transform_rejects_invalid_rigid_transforms(backend, kind) -> None:
    transform = np.eye(4)
    if kind == "zero":
        transform[:] = 0
    elif kind == "singular":
        transform[2, 2] = 0
    elif kind == "scale":
        transform[0, 0] = 1.1
    elif kind == "shear":
        transform[0, 1] = 0.2
    elif kind == "reflection":
        transform[0, 0] = -1
    elif kind == "last_row":
        transform[3, 0] = 0.1
    else:
        transform[0, 3] = float(kind)
    error = np.linalg.LinAlgError if kind in ("zero", "singular") else ValueError

    with pytest.raises(error):
        inv_transform(backend(transform))


@pytest.mark.parametrize("backend", [np.asarray, torch.as_tensor])
def test_inv_transform_accepts_rounded_float32_rotations(backend) -> None:
    # A 45-degree rotation rounded to six decimal places, as in robot configs.
    transform = np.array(
        [
            [0.707107, -0.707107, 0, 0.13],
            [0.707107, 0.707107, 0, -0.02],
            [0, 0, 1, 0.03],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    inverse = inv_transform(backend(transform))
    np.testing.assert_allclose(np.asarray(inverse) @ transform, np.eye(4), atol=1e-6)


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


@pytest.mark.parametrize(
    ("angle", "device"),
    [
        (0.0, "cpu"),
        (1e-6, "cpu"),
        (0.02, "cpu"),
        (0.05, "cpu"),
        (0.6, "cpu"),
        (np.pi / 2 - 0.01, "cpu"),
        (np.pi / 2 + 0.01, "cpu"),
        (np.pi - 0.01, "cpu"),
        pytest.param(0.02, "cuda:0", marks=pytest.mark.gpu),
    ],
)
def test_rotation_interpolation_retains_small_motion_and_endpoints(
    angle: float, device: str
) -> None:
    start = Rotation.from_euler("x", 0.3).as_matrix()
    expected = (
        Rotation.from_euler("z", np.linspace(0, angle, 5)[:, None]).as_matrix() @ start
    )
    start_tensor = torch.tensor(start, dtype=torch.float64, device=device)
    end_tensor = torch.tensor(expected[-1], dtype=torch.float64, device=device)
    original_start, original_end = start_tensor.clone(), end_tensor.clone()

    result = interpolate_rotations(start_tensor, end_tensor, num_steps=4)

    torch.testing.assert_close(
        result, torch.tensor(expected, dtype=torch.float64, device=device)
    )
    torch.testing.assert_close(start_tensor, original_start, rtol=0, atol=0)
    torch.testing.assert_close(end_tensor, original_end, rtol=0, atol=0)


def test_pose_interpolation_keeps_small_rotation_aligned_with_translation() -> None:
    start = torch.eye(4, dtype=torch.float64)
    end = torch.eye(4, dtype=torch.float64)
    end[:3, :3] = torch.from_numpy(Rotation.from_euler("z", 0.02).as_matrix())
    end[:3, 3] = torch.tensor([1.0, 0.5, -0.25])

    poses, interior_count = interpolate_poses(start, end, num_steps=3)

    expected = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    expected[:, :3, :3] = torch.from_numpy(
        Rotation.from_euler("z", np.linspace(0, 0.02, 5)[:, None]).as_matrix()
    )
    expected[:, :3, 3] = (
        torch.linspace(0, 1, 5, dtype=torch.float64)[:, None] * end[:3, 3]
    )
    assert interior_count == 3
    torch.testing.assert_close(poses, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("angle", "device"),
    [
        (0.02, "cpu"),
        (np.pi / 2 - 0.01, "cpu"),
        (np.pi / 2 + 0.01, "cpu"),
        (np.pi - 0.01, "cpu"),
        pytest.param(0.02, "cuda:0", marks=pytest.mark.gpu),
        pytest.param(np.pi - 0.01, "cuda:0", marks=pytest.mark.gpu),
    ],
)
def test_rotation_interpolation_preserves_low_precision(angle, device, dtype) -> None:
    # A 90-degree X rotation is nonidentity and exactly representable in half types.
    start_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    start = torch.tensor(start_matrix, dtype=dtype, device=device)
    end = torch.tensor(
        Rotation.from_euler("z", angle).as_matrix() @ start_matrix,
        dtype=dtype,
        device=device,
    )
    actual = interpolate_rotations(start, end, 4)
    expected = torch.tensor(
        Rotation.from_euler("z", np.linspace(0, angle, 5)[:, None]).as_matrix()
        @ start_matrix,
        dtype=dtype,
        device=device,
    )
    assert actual.dtype == dtype and actual.device == start.device
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[0], start, atol=0, rtol=0)
    torch.testing.assert_close(actual[-1], end, atol=0, rtol=0)


@pytest.mark.parametrize(
    "angle,device",
    [
        (0.0, "cpu"),
        (1e-6, "cpu"),
        (0.02, "cpu"),
        pytest.param(0.02, "cuda:0", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_rotation_interpolation_has_finite_analytic_gradient(
    angle, device, dtype
) -> None:
    theta = torch.tensor(angle, dtype=dtype, device=device, requires_grad=True)
    c, s = theta.cos(), theta.sin()
    zero, one = theta.new_zeros(()), theta.new_ones(())
    end = torch.stack([c, -s, zero, s, c, zero, zero, zero, one]).reshape(3, 3)
    output = interpolate_rotations(torch.eye(3, dtype=dtype, device=device), end, 4)
    (gradient,) = torch.autograd.grad(output[1, 0, 1], theta)
    expected_value = -torch.sin(theta.detach() / 4)
    expected_gradient = -torch.cos(theta.detach() / 4) / 4
    torch.testing.assert_close(output[1, 0, 1], expected_value)
    torch.testing.assert_close(gradient, expected_gradient)


def test_rotation_interpolation_requires_a_positive_interval_count() -> None:
    with pytest.raises(ValueError, match="num_steps must be positive"):
        interpolate_rotations(torch.eye(3), torch.eye(3), 0)


@pytest.mark.parametrize(
    ("interior_count", "device"),
    [(0, "cpu"), (2, "cpu"), pytest.param(0, "cuda:0", marks=pytest.mark.gpu)],
)
def test_pose_interpolation_has_a_uniform_return_contract(
    interior_count: int, device: str
) -> None:
    start = torch.eye(4, dtype=torch.float64, device=device)
    start[:3, 3] = start.new_tensor([0.1, 0.2, 0.3])
    end = torch.eye(4, dtype=torch.float64, device=device)
    end[:3, :3] = end.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    end[:3, 3] = end.new_tensor([1.0, 2.0, 3.0])
    original_start, original_end = start.clone(), end.clone()

    poses, count = interpolate_poses(start, end, num_steps=interior_count)

    assert count == interior_count
    assert poses.shape == (interior_count + 2, 4, 4)
    torch.testing.assert_close(poses[0], start)
    torch.testing.assert_close(poses[-1], end)
    torch.testing.assert_close(start, original_start, rtol=0, atol=0)
    torch.testing.assert_close(end, original_end, rtol=0, atol=0)
