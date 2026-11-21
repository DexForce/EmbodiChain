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

"""Batched matrix products with fused small-matrix evaluation under compilation."""

from __future__ import annotations

import torch

__all__ = ["matrix_product", "matrix_vector_product"]


def matrix_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply matrices with broadcast batch axes and preserve gradients.

    Args:
        left: Matrices with shape ``(..., M, K)``.
        right: Matrices with shape ``(..., K, N)`` and broadcastable batch axes.

    Returns:
        Matrix products with shape ``(..., M, N)``. Compiled FP32/FP64 3-by-3 products
        use pointwise reductions that fuse with adjacent geometry. Other
        sizes, eager calls and autocast regions use Torch matrix multiplication.
    """
    if (
        torch.compiler.is_compiling()
        and left.shape[-2:] == (3, 3)
        and right.shape[-2:] == (3, 3)
        and left.dtype == right.dtype
        and left.dtype in (torch.float32, torch.float64)
        and not torch.is_autocast_enabled(left.device.type)
    ):
        return (left.unsqueeze(-1) * right.unsqueeze(-3)).sum(-2)
    return left @ right


def matrix_vector_product(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply matrices to vectors with broadcast batch axes and preserve gradients.

    Args:
        matrix: Matrices with shape ``(..., M, K)``.
        vector: Vectors with shape ``(..., K)`` and broadcastable batch axes.

    Returns:
        Vectors with shape ``(..., M)``. Compiled FP32/FP64 3-by-3 products fuse their
        pointwise reduction; other sizes, eager calls and autocast regions
        use Torch matrix multiplication.
    """
    if (
        torch.compiler.is_compiling()
        and matrix.shape[-2:] == (3, 3)
        and vector.shape[-1] == 3
        and matrix.dtype == vector.dtype
        and matrix.dtype in (torch.float32, torch.float64)
        and not torch.is_autocast_enabled(matrix.device.type)
    ):
        return (matrix * vector.unsqueeze(-2)).sum(-1)
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)
