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

"""Initial-state distribution shared by the tasks and policy benchmark."""

from __future__ import annotations
import torch
from embodichain_models.dynamics.nerd.codec import quat_rotate
from ._scene_assets import reference_profile

__all__: list[str] = []


def sample_task_reset(
    robot: str,
    count: int,
    device: torch.device | str,
    generator: torch.Generator,
    random_reset: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample task states using an explicit generator without touching policy RNG."""
    profile = reference_profile(robot)
    if robot == "cartpole":
        q = torch.tensor(profile["default_q"], device=device).expand(count, -1).clone()
        qd = torch.zeros(count, profile["qd_dim"], device=device)
        if random_reset:
            q = (
                2 * torch.rand(count, 2, device=device, generator=generator) - 1
            ) * q.new_tensor((1.0, torch.pi))
            qd = 2 * torch.rand(count, 2, device=device, generator=generator) - 1
    elif robot == "ant":
        q = torch.tensor(profile["default_q"], device=device).expand(count, -1).clone()
        qd = torch.zeros(count, profile["qd_dim"], device=device)
        if random_reset:
            q[:, :3] += (
                2 * torch.rand(count, 3, device=device, generator=generator) - 1
            ) * q.new_tensor((0.2, 0.2, 0.1))
            axis = 2 * torch.rand(count, 3, device=device, generator=generator) - 1
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            angle = (
                2 * torch.rand(count, 1, device=device, generator=generator) - 1
            ) * (torch.pi / 6)
            q[:, 3:7] = torch.cat(
                (axis * torch.sin(angle / 2), torch.cos(angle / 2)), -1
            )
            q[:, 7:] += 0.2 * (
                2 * torch.rand(count, 8, device=device, generator=generator) - 1
            )
            qd[:, :3] = quat_rotate(
                q[:, 3:7],
                0.1
                * (2 * torch.rand(count, 3, device=device, generator=generator) - 1),
            )
            qd[:, 3:6] = quat_rotate(
                q[:, 3:7],
                0.25
                * (2 * torch.rand(count, 3, device=device, generator=generator) - 1),
            )
            qd[:, 6:] = 0.25 * (
                2 * torch.rand(count, 8, device=device, generator=generator) - 1
            )
    else:
        raise ValueError(f"Unsupported task: {robot}")
    return q, qd
