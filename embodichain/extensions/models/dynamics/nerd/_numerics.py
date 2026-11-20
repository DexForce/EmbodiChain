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

"""Explicit numerical execution for reference-aligned dynamics training.

This module depends only on Torch and the standard library so an isolated
historical training worker can load the same policy without importing the SDK.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext

import torch
from torch import Tensor

__all__: list[str] = []


def _validate_numerical_profile(profile: str) -> str:
    if not isinstance(profile, str) or profile not in {"default", "reference"}:
        raise ValueError("numerical_profile must be 'default' or 'reference'")
    return profile


def _checkpoint_numerical_profile(state: dict) -> str:
    """Resolve legacy default checkpoints and reject conflicting annotations."""
    configured = _validate_numerical_profile(
        state["train_cfg"].get("numerical_profile", "default")
    )
    declared = _validate_numerical_profile(state.get("numerical_profile", configured))
    if configured != declared:
        raise ValueError("Checkpoint numerical_profile metadata mismatch")
    backend = state["model_cfg"].get("attention_backend", "default")
    if state.get("attention_backend", backend) != backend:
        raise ValueError("Checkpoint attention_backend metadata mismatch")
    if declared == "reference" and backend != "math":
        raise ValueError("Reference checkpoint requires math attention_backend")
    return declared


def _attention_context(profile: str) -> AbstractContextManager:
    """Select math SDPA explicitly, restoring all backend flags on exit."""
    _validate_numerical_profile(profile)
    if profile == "default":
        return nullcontext()
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        return torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    )


@torch.jit.script
def _ordered_sum(values: Tensor) -> Tensor:
    # Do not replace this left fold with a parallel/tree sum. The squared
    # inputs are computed separately, preserving multiply/add rounding order.
    total = values.new_zeros(())
    for index in range(values.numel()):
        total = total + values[index]
    return total


@torch.no_grad()
def _clip_grad_norm_(
    parameters: Iterable[Tensor],
    max_norm: float,
    *,
    profile: str,
    error_if_nonfinite: bool = True,
) -> Tensor:
    """Clip FP32 single-device gradients with a fixed reference total norm.

    Each tensor retains its FP32 L2 norm. Only the final ordered square/sum/sqrt
    uses FP64, matching the validated CPU scalar policy. CUDA scalars stay on
    device; callers can combine their finite check with step logging by setting
    ``error_if_nonfinite=False`` and checking the returned norm before Adam.
    """
    _validate_numerical_profile(profile)
    if profile == "default":
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, error_if_nonfinite=error_if_nonfinite
        )
    parameters = list(parameters)
    grads = [
        parameter.grad.detach()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grads:
        return parameters[0].new_zeros(()) if parameters else torch.zeros(())
    device = grads[0].device
    if device.type not in {"cpu", "cuda"} or any(
        grad.device != device or grad.dtype != torch.float32 or grad.is_sparse
        for grad in grads
    ):
        raise ValueError(
            "reference numerical_profile requires dense FP32 gradients on one CPU/CUDA device"
        )
    norms = torch.stack(torch._foreach_norm(grads, 2.0))
    squared = norms.to(dtype=torch.float64).square()
    total = _ordered_sum(squared).sqrt().to(dtype=torch.float32)
    if error_if_nonfinite and not bool(torch.isfinite(total)):
        raise FloatingPointError("Non-finite reference gradient norm")
    coefficient = torch.clamp(float(max_norm) / (total + 1e-6), max=1.0)
    torch._foreach_mul_(grads, coefficient)
    return total
