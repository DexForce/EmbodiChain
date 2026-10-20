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

"""Differentiable NeRD state and contact conventions, independent of simulation.

Coordinate/target conventions follow NVlabs/neural-robot-dynamics (Apache-2.0).
The native-COM adapter and functional history ownership are EmbodiChain additions.
"""

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations
from collections.abc import Mapping, Sequence
import torch
from torch import Tensor

__all__ = [
    "NeRDStateCodec",
    "quat_multiply",
    "quat_rotate",
    "quat_inverse",
    "plane_contact_features",
    "append_history",
    "reset_history",
]


def quat_inverse(q: Tensor) -> Tensor:
    """Return the inverse of unit xyzw quaternions."""
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def quat_multiply(a: Tensor, b: Tensor) -> Tensor:
    """Multiply broadcastable xyzw quaternions."""
    av, bv = torch.broadcast_tensors(a[..., :3], b[..., :3])
    return torch.cat(
        (
            a[..., 3:] * bv + b[..., 3:] * av + torch.linalg.cross(av, bv),
            a[..., 3:] * b[..., 3:] - (av * bv).sum(-1, keepdim=True),
        ),
        -1,
    )


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors by broadcastable unit xyzw quaternions."""
    xyz, v = torch.broadcast_tensors(q[..., :3], v)
    uv = torch.linalg.cross(xyz, v)
    return v + 2 * (q[..., 3:] * uv + torch.linalg.cross(xyz, uv))


def _canonical(q: Tensor) -> Tensor:
    return torch.where(q[..., 3:] < 0, -q, q)


def _normalize(q: Tensor) -> Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-9)


def _wrap(x: Tensor) -> Tensor:
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


class NeRDStateCodec:
    """Encode Newton child-COM velocities as the official NeRD spatial twist.

    Args:
        robot: ``cartpole`` or ``ant``; fixes q/qd layout to 2/2 or 15/14.
        base_com: Root body's COM offset in its local body frame.
        continuous_q_indices: Coordinates wrapped to [-pi, pi). Defaults to the
            Cartpole pole coordinate and no Ant coordinates.

    State tensors may have any leading batch/time dimensions. Ant q uses xyz,
    xyzw, eight joint coordinates. Native qd uses world COM linear velocity,
    world angular velocity, eight joint speeds. The legacy spatial twist uses
    angular velocity then linear velocity referenced to the world origin.
    """

    def __init__(
        self,
        robot: str = "cartpole",
        base_com: Sequence[float] = (0.0, 0.0, 0.0),
        continuous_q_indices: Sequence[int] | None = None,
    ) -> None:
        self.robot = robot.lower()
        if self.robot not in {"cartpole", "ant"}:
            raise ValueError(f"Unsupported NeRD robot: {robot}")
        self.q_dim, self.qd_dim = (2, 2) if self.robot == "cartpole" else (15, 14)
        if len(base_com) != 3:
            raise ValueError("base_com must contain three coordinates")
        self.base_com = tuple(base_com)
        self.continuous_q_indices = tuple(
            continuous_q_indices
            if continuous_q_indices is not None
            else ((1,) if self.robot == "cartpole" else ())
        )
        if any(
            i < 0 or i >= self.q_dim or (self.robot == "ant" and i < 7)
            for i in self.continuous_q_indices
        ):
            raise ValueError("continuous coordinates must be scalar joint coordinates")

    def _check(self, q: Tensor, qd: Tensor) -> None:
        if (
            q.shape[-1] != self.q_dim
            or qd.shape[-1] != self.qd_dim
            or q.shape[:-1] != qd.shape[:-1]
        ):
            raise ValueError(
                f"Expected matching q/qd batch shapes with dimensions {self.q_dim}/{self.qd_dim}"
            )

    def _wrap_q(self, q: Tensor) -> Tensor:
        if not self.continuous_q_indices:
            return q
        return torch.stack(
            [
                _wrap(q[..., i]) if i in self.continuous_q_indices else q[..., i]
                for i in range(self.q_dim)
            ],
            -1,
        )

    def native_to_legacy(self, q: Tensor, qd: Tensor) -> Tensor:
        """Convert native qd to angular-first world-origin spatial velocity."""
        self._check(q, qd)
        if self.robot == "cartpole":
            return qd
        com = q[..., :3] + quat_rotate(q[..., 3:7], q.new_tensor(self.base_com))
        omega = qd[..., 3:6]
        return torch.cat(
            (omega, qd[..., :3] + torch.linalg.cross(com, omega), qd[..., 6:]), -1
        )

    def legacy_to_native(self, q: Tensor, velocity: Tensor) -> Tensor:
        """Convert angular-first world-origin velocity to native COM velocity."""
        self._check(q, velocity)
        if self.robot == "cartpole":
            return velocity
        com = q[..., :3] + quat_rotate(q[..., 3:7], q.new_tensor(self.base_com))
        omega = velocity[..., :3]
        return torch.cat(
            (
                velocity[..., 3:6] - torch.linalg.cross(com, omega),
                omega,
                velocity[..., 6:],
            ),
            -1,
        )

    def _body(
        self, anchor: Tensor, q: Tensor, velocity: Tensor
    ) -> tuple[Tensor, Tensor]:
        inv = quat_inverse(anchor[..., 3:7])
        pos = quat_rotate(inv, q[..., :3] - anchor[..., :3])
        rot = quat_multiply(inv, q[..., 3:7])
        omega = quat_rotate(inv, velocity[..., :3])
        nu = quat_rotate(
            inv,
            velocity[..., 3:6] - torch.linalg.cross(anchor[..., :3], velocity[..., :3]),
        )
        return torch.cat((pos, rot, q[..., 7:]), -1), torch.cat(
            (omega, nu, velocity[..., 6:]), -1
        )

    def encode(
        self,
        q: Tensor,
        qd: Tensor,
        effort: Tensor,
        contacts: Mapping[str, Tensor] | None = None,
        gravity_dir: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode inputs at t; contact points/normals are world-frame flattened slots.

        Returns:
            Named unnormalized model features, retaining their input graphs.
        """
        velocity = self.native_to_legacy(q, qd)
        if self.robot == "cartpole":
            if effort.shape != qd.shape:
                raise ValueError(
                    "Cartpole effort must have two generalized coordinates"
                )
            return {
                "states_embedding": torch.cat((self._wrap_q(q), velocity), -1),
                "joint_f": effort,
            }
        bq, bv = self._body(q, q, velocity)
        bq = torch.cat((bq[..., :3], _canonical(bq[..., 3:7]), bq[..., 7:]), -1)
        if effort.shape[-1] not in (8, 14) or effort.shape[:-1] != q.shape[:-1]:
            raise ValueError(
                "Ant effort must have 8 actuated or 14 generalized coordinates"
            )
        if contacts is None:
            raise ValueError("Ant requires explicitly ordered fixed contact slots")
        depth = contacts["contact_depths"]
        slots = depth.shape[-1]
        if depth.shape[:-1] != q.shape[:-1]:
            raise ValueError("Contact batch/time shape must match state")
        inv = quat_inverse(q[..., 3:7]).unsqueeze(-2)
        points = contacts["contact_points_1"].reshape(*q.shape[:-1], slots, 3)
        normals = contacts["contact_normals"].reshape_as(points)
        points = quat_rotate(inv, points - q[..., None, :3])
        normals = quat_rotate(inv, normals)
        if "contact_masks" in contacts:
            mask = contacts["contact_masks"].bool()
        else:
            t0 = contacts.get("contact_thicknesses_0", torch.zeros_like(depth))
            t1 = contacts.get("contact_thicknesses_1", torch.zeros_like(depth))
            mask = depth < (4 * (t0 + t1)).clamp_min(0.12)
        if gravity_dir is None:
            gravity_dir = q.new_tensor((0.0, 0.0, -1.0)).expand(*q.shape[:-1], 3)
        return {
            "states_embedding": torch.cat((self._wrap_q(bq), bv), -1),
            "contact_normals": torch.where(mask[..., None], normals, 0).flatten(-2),
            "contact_points_1": torch.where(mask[..., None], points, 0).flatten(-2),
            "contact_depths": torch.where(mask, depth, 0),
            "joint_f": effort[..., -8:],
            "gravity_dir": quat_rotate(inv.squeeze(-2), gravity_dir),
        }

    def targets(self, q: Tensor, qd: Tensor, next_q: Tensor, next_qd: Tensor) -> Tensor:
        """Encode relative next state in the frame at t, including quaternion delta."""
        velocity = self.native_to_legacy(q, qd)
        next_velocity = self.native_to_legacy(next_q, next_qd)
        if self.robot == "ant":
            anchor = q
            q, velocity = self._body(anchor, q, velocity)
            next_q, next_velocity = self._body(anchor, next_q, next_velocity)
            delta = _canonical(
                quat_multiply(next_q[..., 3:7], quat_inverse(q[..., 3:7]))
            )
            diff = self._wrap_q(next_q - q)
            return torch.cat(
                (diff[..., :3], delta, diff[..., 7:], next_velocity - velocity), -1
            )
        return torch.cat((self._wrap_q(next_q - q), next_velocity - velocity), -1)

    def decode(
        self, q: Tensor, qd: Tensor, prediction: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Decode a relative prediction into native Newton q/qd without detaching."""
        if prediction.shape != (*q.shape[:-1], self.q_dim + self.qd_dim):
            raise ValueError("Prediction shape does not match state")
        velocity = self.native_to_legacy(q, qd)
        if self.robot == "cartpole":
            return self._wrap_q(q + prediction[..., :2]), qd + prediction[..., 2:]
        bq, bv = self._body(q, q, velocity)
        predicted_rot = _normalize(
            quat_multiply(_normalize(prediction[..., 3:7]), bq[..., 3:7])
        )
        local_q = torch.cat(
            (
                bq[..., :3] + prediction[..., :3],
                predicted_rot,
                bq[..., 7:] + prediction[..., 7 : self.q_dim],
            ),
            -1,
        )
        local_v = bv + prediction[..., self.q_dim :]
        rot = q[..., 3:7]
        pos = q[..., :3] + quat_rotate(rot, local_q[..., :3])
        next_rot = quat_multiply(rot, local_q[..., 3:7])
        omega = quat_rotate(rot, local_v[..., :3])
        nu = torch.linalg.cross(q[..., :3], omega) + quat_rotate(rot, local_v[..., 3:6])
        next_q = self._wrap_q(torch.cat((pos, next_rot, local_q[..., 7:]), -1))
        next_velocity = torch.cat((omega, nu, local_v[..., 6:]), -1)
        return next_q, self.legacy_to_native(next_q, next_velocity)


def plane_contact_features(
    points_world: Tensor,
    thickness0: Tensor,
    plane_normal: Tensor,
    plane_offset: Tensor | float = 0.0,
    thickness1: Tensor | float = 0.0,
) -> dict[str, Tensor]:
    """Project fixed slots onto n dot x = offset; normals must be unit length.

    Args:
        points_world: Candidate center points, shape (..., slots, 3).
        thickness0: Shape radius plus margin, shape (..., slots).
        plane_normal: Unit normal, broadcastable to candidate points.
        plane_offset: Plane displacement along the normal.
        thickness1: Plane margin.

    Returns:
        Official contact fields; depth is center-to-plane distance, before thickness.
    """
    normals = torch.broadcast_to(plane_normal, points_world.shape)
    depth = (points_world * normals).sum(-1) - plane_offset
    projected = points_world - depth[..., None] * normals
    t0 = torch.broadcast_to(thickness0, depth.shape)
    t1 = torch.zeros_like(depth) + thickness1
    return {
        "contact_points_1": projected.flatten(-2),
        "contact_normals": normals.flatten(-2),
        "contact_depths": depth,
        "contact_thicknesses_0": t0,
        "contact_thicknesses_1": t1,
        "contact_masks": depth < (4 * (t0 + t1)).clamp_min(0.12),
    }


def append_history(
    history: Mapping[str, Tensor], lengths: Tensor, features: Mapping[str, Tensor]
) -> tuple[dict[str, Tensor], Tensor]:
    """Append one token to left-aligned fixed capacity histories without mutation.

    Full rows slide left. Short rows retain position indices starting at zero.
    Storage is O(batch * capacity * feature_width), with no host synchronization.
    """
    if not history or set(history) != set(features):
        raise ValueError("History and feature keys must be identical and nonempty")
    capacity = next(iter(history.values())).shape[1]
    pos = torch.arange(capacity, device=lengths.device)[None, :]
    full = lengths >= capacity
    result = {}
    for name, values in history.items():
        shifted = torch.cat((values[:, 1:], torch.zeros_like(values[:, :1])), 1)
        base = torch.where(full[:, None, None], shifted, values)
        at = pos == lengths.clamp_max(capacity - 1)[:, None]
        result[name] = torch.where(at[..., None], features[name][:, None, :], base)
    return result, (lengths + 1).clamp_max(capacity)


def reset_history(
    history: Mapping[str, Tensor], lengths: Tensor, mask: Tensor
) -> tuple[dict[str, Tensor], Tensor]:
    """Functionally clear selected rows while retaining other rows' gradients."""
    return {
        k: torch.where(mask[:, None, None], torch.zeros_like(v), v)
        for k, v in history.items()
    }, torch.where(mask, 0, lengths)
