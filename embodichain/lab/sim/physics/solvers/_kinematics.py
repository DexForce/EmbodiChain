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

"""First-order Torch autograd boundary for native Newton FK."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.autograd.function import once_differentiable
import warp as wp

__all__: list[str] = []


def _stream_scope(device: torch.device):
    if device.type == "cuda":
        return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))
    return nullcontext()


class _NewtonFK(torch.autograd.Function):
    """Record each call into independently owned Warp state and adjoints."""

    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, qd: torch.Tensor, model: Any):
        import newton

        if q.dtype != torch.float32 or qd.dtype != torch.float32:
            raise TypeError("Newton FK requires float32 physical state tensors.")
        if q.device != qd.device or q.device != torch.device(
            wp.device_to_torch(model.device)
        ):
            raise ValueError("Newton model and FK inputs must share a device.")
        if q.shape != (model.joint_coord_count,) or qd.shape != (
            model.joint_dof_count,
        ):
            raise ValueError(
                "Newton FK requires flattened model-order joint coordinates."
            )
        ctx.device = q.device
        with _stream_scope(q.device):
            # Never retain aliases to caller/native buffers: a later publication
            # or reset may overwrite them before this transition's backward.
            q_owned = q.detach().contiguous().clone()
            qd_owned = qd.detach().contiguous().clone()
            ctx.q_wp = wp.from_torch(q_owned, requires_grad=True)
            ctx.qd_wp = wp.from_torch(qd_owned, requires_grad=True)
            ctx.state = model.state(requires_grad=True)
            ctx.tape = wp.Tape()
            with ctx.tape:
                newton.eval_fk(model, ctx.q_wp, ctx.qd_wp, ctx.state)
            # Clones also keep the public outputs independent of Warp scratch.
            body_q = wp.to_torch(ctx.state.body_q).clone()
            body_qd = wp.to_torch(ctx.state.body_qd).clone()
        return body_q, body_qd

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any, body_q_grad: torch.Tensor | None, body_qd_grad: torch.Tensor | None
    ):
        with _stream_scope(ctx.device):
            ctx.tape.zero()
            grads = {}
            if body_q_grad is not None:
                grads[ctx.state.body_q] = wp.from_torch(
                    body_q_grad.contiguous(), dtype=wp.transform
                )
            if body_qd_grad is not None:
                grads[ctx.state.body_qd] = wp.from_torch(
                    body_qd_grad.contiguous(), dtype=wp.spatial_vector
                )
            ctx.tape.backward(grads=grads)
            q_grad = wp.to_torch(ctx.q_wp.grad).clone()
            qd_grad = wp.to_torch(ctx.qd_wp.grad).clone()
        return q_grad, qd_grad, None


def differentiable_fk(
    model: Any, q: torch.Tensor, qd: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body poses and COM velocities with first-order input gradients.

    Args:
        model: Finalized Newton model; its topology and physical parameters
            must remain unchanged while a returned graph is alive.
        q: Flattened float32 Newton joint coordinates.
        qd: Flattened float32 Newton joint velocities.

    Returns:
        Body transforms ``(N, 7)`` and COM linear/angular velocities ``(N, 6)``.
        Storage belongs to this call, independently of live simulation state.
    """
    return _NewtonFK.apply(q, qd, model)
