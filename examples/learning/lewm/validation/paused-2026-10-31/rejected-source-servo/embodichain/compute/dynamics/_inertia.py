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
"""Generalized inertia and velocity integration without simulator dependencies."""

from __future__ import annotations

import torch

__all__ = []


def _generalized_mass_matrix(
    linear_jacobian: torch.Tensor,
    angular_jacobian: torch.Tensor,
    mass: torch.Tensor,
    world_inertia: torch.Tensor,
    armature: torch.Tensor,
) -> torch.Tensor:
    """Assemble inertia from COM Jacobians and world-frame body tensors.

    Jacobians have shape ``(N, L, D, 3)``, mass ``(L,)``, world inertia
    ``(N, L, 3, 3)`` and armature ``(D,)``. Inputs share dtype/device. The
    returned ``(N, D, D)`` tensor owns its storage. Kinematics and topology
    selection belong to the caller.
    """
    linear = torch.einsum("nlia,l,nlja->nij", linear_jacobian, mass, linear_jacobian)
    angular = torch.einsum(
        "nlia,nlab,nljb->nij", angular_jacobian, world_inertia, angular_jacobian
    )
    return linear + angular + torch.diag(armature)[None]


def _implicit_velocity_step(
    q_before: torch.Tensor,
    v_before: torch.Tensor,
    v_physics: torch.Tensor,
    mass: torch.Tensor,
    damping: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a smooth velocity derivative after an explicit-force physics step.

    ``damping`` is the negative velocity derivative of smooth generalized
    forces, projected into the integrator's structural pattern when needed.
    States have shape ``(N, D)`` and matrices ``(N, D, D)``. The caller owns
    contact/constraint solving and state publication. ``mass + dt*damping``
    must be nonsingular; solve failures propagate to the caller.
    """
    rhs = mass @ (v_physics - v_before)[..., None]
    velocity = v_before + torch.linalg.solve(mass + dt * damping, rhs)[..., 0]
    return q_before + dt * velocity, velocity
