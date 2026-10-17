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

"""Private float32/float64 anchored interpolation kernels."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import warp as wp

__all__ = []

# Keep float64 timing arithmetic comparable to the NumPy/Torch reference.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.kernel
def _interpolate_f32(
    q: wp.array3d(dtype=wp.float32),
    left: wp.array2d(dtype=wp.int64),
    right: wp.array2d(dtype=wp.int64),
    weights: wp.array2d(dtype=wp.float64),
    out: wp.array3d(dtype=wp.float32),
):
    b, t, d = wp.tid()
    lo = left[b, t]
    hi = right[b, t]
    if lo == hi:
        out[b, t, d] = q[b, lo, d]
    else:
        difference = q[b, hi, d] - q[b, lo, d]
        out[b, t, d] = wp.float32(
            wp.float64(q[b, lo, d]) + weights[b, t] * wp.float64(difference)
        )


@wp.kernel
def _interpolate_f64(
    q: wp.array3d(dtype=wp.float64),
    left: wp.array2d(dtype=wp.int64),
    right: wp.array2d(dtype=wp.int64),
    weights: wp.array2d(dtype=wp.float64),
    out: wp.array3d(dtype=wp.float64),
):
    b, t, d = wp.tid()
    lo = left[b, t]
    hi = right[b, t]
    if lo == hi:
        out[b, t, d] = q[b, lo, d]
    else:
        out[b, t, d] = q[b, lo, d] + weights[b, t] * (q[b, hi, d] - q[b, lo, d])


def evaluate(
    positions: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Launch on the caller's Torch stream; explicit backend errors propagate."""
    wp.init()
    q = positions.contiguous()
    out = positions.new_empty((*left.shape, positions.shape[-1]))
    scope = (
        wp.ScopedStream(wp.stream_from_torch(q.device)) if q.is_cuda else nullcontext()
    )
    with scope:
        wp.launch(
            kernel=_interpolate_f32 if q.dtype == torch.float32 else _interpolate_f64,
            dim=out.shape,
            inputs=[
                wp.from_torch(q),
                wp.from_torch(left),
                wp.from_torch(right),
                wp.from_torch(weights),
            ],
            outputs=[wp.from_torch(out)],
            device=str(q.device),
        )
    return out
