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

"""Internal owned identifiers and proper rigid-transform validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from embodichain.lab._scene_config.values import (
    _identifier,
    _nonnegative_integer,
    _rigid_pose,
)

if TYPE_CHECKING:
    import torch

__all__: list[str] = []


def _owned_batch_pose(value: torch.Tensor, name: str) -> torch.Tensor:
    """Own nonempty batched rigid tensor poses using the scene/query tolerance contract."""
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[0] < 1
        or value.shape[1:] != (4, 4)
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} requires finite floating poses (B, 4, 4).")
    owned = value.detach().clone()
    # Validate small metadata matrices in one CPU transfer, avoiding a
    # cuSOLVER initialization merely to check a 3x3 determinant on the sim GPU.
    matrix = owned.to(device="cpu", dtype=torch.float64)
    rotation = matrix[:, :3, :3]
    if (
        not torch.isfinite(matrix).all()
        or not torch.allclose(
            matrix[:, 3],
            matrix.new_tensor([0, 0, 0, 1]).expand(len(matrix), -1),
            atol=1e-6,
            rtol=0,
        )
        or not torch.allclose(
            rotation.transpose(-1, -2) @ rotation,
            torch.eye(3, device=matrix.device, dtype=matrix.dtype).expand(
                len(matrix), -1, -1
            ),
            atol=1e-5,
            rtol=0,
        )
        or not torch.isclose(
            torch.linalg.det(rotation), matrix.new_tensor(1.0), atol=1e-5, rtol=0
        ).all()
    ):
        raise ValueError(f"{name} must be a rigid transform.")
    return owned


def _owned_b1_pose(value: torch.Tensor, name: str) -> torch.Tensor:
    """Keep the B=1 scene provider boundary unchanged."""
    import torch

    if not isinstance(value, torch.Tensor) or value.shape != (1, 4, 4):
        raise ValueError(f"{name} requires one finite floating pose (1, 4, 4).")
    return _owned_batch_pose(value, name)
