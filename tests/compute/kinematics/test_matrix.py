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

"""Compiled matrix geometry agrees with Torch products, including gradients."""

from __future__ import annotations

import pytest
import torch

from embodichain.compute.kinematics.matrix import matrix_product, matrix_vector_product


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("size", [3, 4])
def test_broadcast_noncontiguous_products_and_gradients(compiled, size):
    torch.manual_seed(71)
    left = (
        torch.randn(2, 1, size, size, dtype=torch.float64)
        .transpose(-1, -2)
        .requires_grad_()
    )
    right = (
        torch.randn(1, 4, size, size, dtype=torch.float64)
        .transpose(-1, -2)
        .requires_grad_()
    )
    vector = torch.randn(1, 4, size, dtype=torch.float64, requires_grad=True)

    def evaluate(left, right, vector):
        return matrix_product(left, right), matrix_vector_product(left, vector)

    fn = (
        torch.compile(evaluate, backend="aot_eager", fullgraph=True)
        if compiled
        else evaluate
    )
    actual = fn(left, right, vector)
    expected = (left @ right, (left @ vector[..., None]).squeeze(-1))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(
        sum(value.square().sum() for value in actual), (left, right, vector)
    )
    reference_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in expected), (left, right, vector)
    )
    for value, reference in zip(gradients, reference_gradients):
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("compiled", [False, True])
def test_autocast_keeps_torch_product_precision(compiled):
    matrix = torch.randn(2, 3, 3)
    vector = torch.randn(2, 3)

    def evaluate(matrix, vector):
        return matrix_product(matrix, matrix), matrix_vector_product(matrix, vector)

    fn = (
        torch.compile(evaluate, backend="aot_eager", fullgraph=True)
        if compiled
        else evaluate
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = fn(matrix, vector)
        expected = (matrix @ matrix, (matrix @ vector[..., None]).squeeze(-1))
    for value, reference in zip(actual, expected):
        assert value.dtype == torch.bfloat16
        torch.testing.assert_close(value, reference)


def test_compiled_integer_products_preserve_dtype():
    matrix = torch.arange(18, dtype=torch.int32).reshape(2, 3, 3)
    vector = torch.arange(6, dtype=torch.int32).reshape(2, 3)

    def evaluate(matrix, vector):
        return matrix_product(matrix, matrix), matrix_vector_product(matrix, vector)

    actual = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(
        matrix, vector
    )
    expected = (matrix @ matrix, (matrix @ vector[..., None]).squeeze(-1))
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
