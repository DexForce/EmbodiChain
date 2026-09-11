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


"""Stateless batched measurements used by task policies and final evaluation."""

from __future__ import annotations
import torch

__all__ = ["axis_tilt"]


def axis_tilt(pose: torch.Tensor, local_axis: torch.Tensor) -> torch.Tensor:
    """Measure tilt with atan2, preserving small angles in float32 observations.

    Args:
        pose: Batched rigid transforms, shaped ``(N, 4, 4)``.
        local_axis: Unit local axis, shaped ``(3,)``.

    Returns:
        Tilt in radians as float64, shaped ``(N,)``. Invalid numbers propagate.
    """
    if pose.ndim != 3 or pose.shape[-2:] != (4, 4) or local_axis.shape != (3,):
        raise ValueError("Expected poses (N, 4, 4) and one local axis (3,).")
    axis = pose[:, :3, :3].double() @ local_axis.double()
    return torch.atan2(torch.linalg.vector_norm(axis[:, :2], dim=-1), axis[:, 2])
