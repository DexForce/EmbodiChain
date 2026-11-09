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

"""Pure reference task observations and rewards for learned robot dynamics.

Costs follow the Cartpole and forward-Ant definitions in NVIDIA NeRD. Inputs use
Newton world COM linear/angular velocities; legacy twist conversion belongs to
the state codec. These functions do not implement a simulator or reset policy.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations
import torch
from torch import Tensor
from embodichain_models.dynamics.nerd.codec import quat_rotate

__all__ = ["reference_task_values"]


def reference_task_values(
    robot: str, q: Tensor, qd: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return graph-connected observations/reward and a discrete failure mask.

    Args:
        robot: ``cartpole`` or ``ant``.
        q: Coordinates with trailing width 2 or 15, arbitrary batch/time axes.
        qd: Native velocities with trailing width 2 or 14 and matching axes.

    Returns:
        Observation (width 4 for Cartpole, 27 for Ant), scalar reward per row,
        and boolean task/numerical failure per row. Cartpole observes wrapped
        pole angle; Ant omits root x/y and observes COM velocity. Ant rewards
        forward motion along +y plus upright/heading bonuses. Non-finite states
        and invalid zero quaternions fail; reward values remain unsanitized so
        evaluators can account explicitly for numerical failures.
    """
    if robot not in {"cartpole", "ant"}:
        raise ValueError(f"Unsupported reference robot {robot!r}")
    q_dim, qd_dim = (2, 2) if robot == "cartpole" else (15, 14)
    if q.shape[-1] != q_dim or qd.shape[-1] != qd_dim or q.shape[:-1] != qd.shape[:-1]:
        raise ValueError("Reference state dimensions or batch axes do not match")
    if robot == "cartpole":
        angle = torch.remainder(q[..., 1] + torch.pi, 2 * torch.pi) - torch.pi
        observation = torch.cat((q[..., :1], angle[..., None], qd), -1)
        reward = -(
            angle.square() + 0.05 * q[..., 0].square() + 0.1 * qd.square().sum(-1)
        )
        failed = (q[..., 0].abs() > 4) | (qd.abs() > 10).any(-1)
    else:
        observation = torch.cat((q[..., 2:], qd), -1)
        up = quat_rotate(q[..., 3:7], q.new_tensor((0.0, 0.0, 1.0)))
        heading = quat_rotate(q[..., 3:7], q.new_tensor((0.0, 1.0, 0.0)))
        reward = qd[..., 1] + 0.1 * up[..., 2] + heading[..., 1]
        failed = (q[..., 2] < 0.3) | (q[..., 3:7].norm(dim=-1) < 1e-6)
    failed = failed | ~torch.isfinite(q).all(-1) | ~torch.isfinite(qd).all(-1)
    return observation, reward, failed
