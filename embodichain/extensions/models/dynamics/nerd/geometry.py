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

"""Tensor-only NeRD contact geometry with an injected differentiable FK backend."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from .codec import plane_contact_features, quat_rotate

__all__ = ["NeRDGeometry"]


class NeRDGeometry:
    """Evaluate fixed contact slots from the current graph-connected state.

    Args:
        gravity: Fixed world gravity, shape ``(batch, 3)``.
        fk: Optional tensor callable returning body poses and velocities from
            canonical ``q, qd``. Poses use flattened body order and xyz/xyzw.
            The callable must preserve input gradients and own each result's
            storage. It must never read live simulation state.
        contact_bodies: Body indices, shape ``(batch, slots)``. Omission denotes
            contact-free dynamics and does not require an FK backend.
        local_points: Fixed body-local contact centers, ``(batch, slots, 3)``.
        thickness: Fixed radius plus shape margin, ``(batch, slots)``.
        plane_normal: Unit world-plane normal, shape ``(3,)``.
        plane_offset: Plane displacement along its normal.
        plane_margin: Fixed plane margin.

    Static tensors are detached, cloned and cached once. FK and contact results
    are never cached across calls. Boolean contact masks retain the codec's
    piecewise gradient contract; contact-switch events are not differentiated.
    """

    def __init__(
        self,
        gravity: Tensor,
        *,
        fk: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]] | None = None,
        contact_bodies: Tensor | None = None,
        local_points: Tensor | None = None,
        thickness: Tensor | None = None,
        plane_normal: Tensor | None = None,
        plane_offset: Tensor | float = 0.0,
        plane_margin: Tensor | float = 0.0,
    ) -> None:
        if (
            gravity.ndim != 2
            or gravity.shape[0] < 1
            or gravity.shape[1] != 3
            or not gravity.is_floating_point()
            or not bool(torch.isfinite(gravity).all())
        ):
            raise ValueError("gravity must be finite floating (batch, 3) values")
        self.gravity = gravity.detach().clone()
        self.gravity_dir = self.gravity / self.gravity.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        self.fk = fk
        batch = gravity.shape[0]
        if contact_bodies is None:
            if local_points is not None or thickness is not None:
                raise ValueError("Contact points and thickness require body indices")
            contact_bodies = torch.empty(
                (batch, 0), dtype=torch.long, device=gravity.device
            )
            local_points = gravity.new_empty((batch, 0, 3))
            thickness = gravity.new_empty((batch, 0))
        if (
            contact_bodies.ndim != 2
            or contact_bodies.shape[0] != batch
            or contact_bodies.dtype != torch.long
            or contact_bodies.device != gravity.device
            or bool((contact_bodies < 0).any())
        ):
            raise ValueError("contact_bodies must be nonnegative int64 (batch, slots)")
        if (
            local_points is None
            or local_points.shape != (*contact_bodies.shape, 3)
            or thickness is None
            or thickness.shape != contact_bodies.shape
        ):
            raise ValueError(
                "Contact points/thickness must match the fixed slot layout"
            )
        for value in (local_points, thickness):
            if (
                value.dtype != gravity.dtype
                or value.device != gravity.device
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    "Fixed geometry must share finite gravity dtype/device"
                )
        if bool((thickness < 0).any()):
            raise ValueError("Contact thickness cannot be negative")
        if contact_bodies.shape[1] and fk is None:
            raise ValueError("Contact dynamics requires an FK callable")
        self.contact_bodies = contact_bodies.detach().clone()
        self.local_points = local_points.detach().clone()
        self.thickness = thickness.detach().clone()
        normal = (
            gravity.new_tensor((0.0, 0.0, 1.0))
            if plane_normal is None
            else plane_normal
        )
        if (
            normal.shape != (3,)
            or normal.dtype != gravity.dtype
            or normal.device != gravity.device
            or not bool(torch.isfinite(normal).all())
            or not bool(torch.isclose(normal.norm(), normal.new_tensor(1.0)))
        ):
            raise ValueError("plane_normal must be a finite unit geometry vector")
        self.plane_normal = normal.detach().clone()
        self.plane_offset = (
            torch.as_tensor(plane_offset, dtype=gravity.dtype, device=gravity.device)
            .detach()
            .clone()
        )
        self.plane_margin = (
            torch.as_tensor(plane_margin, dtype=gravity.dtype, device=gravity.device)
            .detach()
            .clone()
        )
        if (
            self.plane_offset.ndim
            or self.plane_margin.ndim
            or not bool(torch.isfinite(self.plane_offset))
            or not bool(torch.isfinite(self.plane_margin))
            or bool(self.plane_margin < 0)
        ):
            raise ValueError("Plane offset/margin must be finite scalars; margin >= 0")

    def contacts(self, q: Tensor, qd: Tensor) -> dict[str, Tensor]:
        """Compute contacts from this step's canonical state, preserving its graph.

        Args:
            q: Batched canonical joint coordinates.
            qd: Batched canonical joint velocities.

        Returns:
            The codec's ordered contact fields. Contact-free models skip FK.
        """
        if (
            q.ndim != 2
            or qd.ndim != 2
            or q.shape[0] != self.gravity.shape[0]
            or qd.shape[0] != q.shape[0]
            or q.dtype != self.gravity.dtype
            or qd.dtype != q.dtype
            or q.device != self.gravity.device
            or qd.device != q.device
        ):
            raise ValueError(
                "State and geometry must have matching batch axes and dtype/device"
            )
        if self.contact_bodies.shape[1]:
            poses, _ = self.fk(q, qd)
            return self.contacts_from_poses(poses)
        return self._empty_contacts(q)

    def contacts_from_poses(self, poses: Tensor) -> dict[str, Tensor]:
        """Project body-local slots using caller-owned xyz/xyzw body poses.

        Args:
            poses: Flattened body poses, shape ``(bodies, 7)``; independent of
                live native buffers when retained for backward.

        Returns:
            Contact points, normals, distances, thicknesses and boolean masks.
            Static local points are a read-only view of the fixed geometry.
        """
        if (
            poses.ndim != 2
            or poses.shape[1] != 7
            or poses.dtype != self.gravity.dtype
            or poses.device != self.gravity.device
        ):
            raise ValueError(
                "Body poses must be (bodies, 7) on the geometry dtype/device"
            )
        if not self.contact_bodies.shape[1]:
            return self._empty_contacts(poses)
        selected = poses[self.contact_bodies]
        world_points = selected[..., :3] + quat_rotate(
            selected[..., 3:], self.local_points
        )
        result = plane_contact_features(
            world_points,
            self.thickness,
            self.plane_normal,
            self.plane_offset,
            self.plane_margin,
        )
        result["contact_points_0"] = self.local_points.flatten(1)
        return result

    def _empty_contacts(self, like: Tensor) -> dict[str, Tensor]:
        empty = like.new_empty((self.gravity.shape[0], 0))
        return {
            name: empty
            for name in (
                "contact_points_0",
                "contact_points_1",
                "contact_normals",
                "contact_depths",
                "contact_thicknesses_0",
                "contact_thicknesses_1",
            )
        }
