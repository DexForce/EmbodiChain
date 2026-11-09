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

"""Explicit SDPA and optional Flash-Attention backends for author attention."""

from __future__ import annotations
import torch
from torch.nn import functional as F


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    backend: str = "sdpa",
) -> torch.Tensor:
    if backend == "flash":
        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:
            raise ImportError(
                "The flash backend requires a Flash-Attention build matching the installed Torch/CUDA stack."
            ) from exc
        return flash_attn_func(
            q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale
        )
    if backend != "sdpa":
        raise ValueError(f"Unsupported attention backend: {backend!r}")
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        scale=softmax_scale,
    ).transpose(1, 2)
