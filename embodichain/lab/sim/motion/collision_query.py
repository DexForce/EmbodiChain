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

"""Discrete source-envelope path checks, independent of manipulation skills.

The support screen uses authored robot vertex AABBs and a native support union
AABB. This preserves the source conservative decision rule; it does not invoke
cuRobo/Newton, replan a trajectory, or certify continuous collision freedom.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
from types import TracebackType
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.lab.sim.scene.robot_collision import (
    RobotCollisionInputs,
    SupportCollisionInputs,
    _matrix,
)
from embodichain.lab.sim.spawn.source_geometry import LinkMesh
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["RobotPathQuery", "SupportPathReport"]

_FK_BATCH = 64
_QUERY_ERRORS = (
    ValueError,
    RuntimeError,
    KeyError,
    AttributeError,
    OSError,
    IndexError,
)


@dataclass(frozen=True, slots=True)
class SupportPathReport:
    """Discrete robot-link vertex envelopes against a native support union AABB.

    Args:
        supported: Whether complete inputs support the requested screen.
        accepted: Whether the supported clearance is strictly positive.
        minimum_clearance_after_skin_m: Minimum box gap after both contact skins,
            in metres; absent when the screen is unavailable.
        checked_frames: Number of caller-selected command frames inspected.
        moving_links: Complete ordered moving-link scope used by the screen.
        reason: Explanation for unavailable inputs; absent for supported results.
    """

    supported: bool
    accepted: bool
    minimum_clearance_after_skin_m: float | None = None
    checked_frames: int = 0
    moving_links: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.supported) is not bool or type(self.accepted) is not bool:
            raise TypeError("Support path report decisions must be booleans.")
        links = tuple(self.moving_links)
        if self.supported:
            value = self.minimum_clearance_after_skin_m
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or self.accepted != (value > 0)
                or self.reason is not None
                or type(self.checked_frames) is not int
                or self.checked_frames <= 0
                or not links
                or len(set(links)) != len(links)
                or any(type(name) is not str or not name for name in links)
            ):
                raise ValueError(
                    "Supported path reports require complete finite evidence."
                )
        elif self.accepted or not isinstance(self.reason, str) or not self.reason:
            raise ValueError("Unsupported path reports require rejection and a reason.")
        object.__setattr__(self, "moving_links", links)

    def metadata(self) -> dict[str, object]:
        """Return an owned source-compatible numeric report."""
        if not self.supported:
            return {"supported": False, "accepted": False, "reason": self.reason}
        return {
            "supported": True,
            "accepted": self.accepted,
            "minimum_clearance_after_skin_m": self.minimum_clearance_after_skin_m,
            "checked_frames": self.checked_frames,
            "moving_links": list(self.moving_links),
        }


class RobotPathQuery:
    """Bind one explicit root observation to borrowed robot source geometry."""

    def __init__(
        self,
        geometry: RobotCollisionInputs,
        *,
        scope: NativeQueryScope,
        robot_root_pose: torch.Tensor,
    ) -> None:
        """Prepare a B1 query without requiring a hand or articulated target.

        Args:
            geometry: Open source capture retained by the caller.
            scope: Host-owned observation identity; the host closes the query
                before advancing the observation.
            robot_root_pose: Observed arena-local robot root, shape ``(4, 4)``.
        """
        if not isinstance(geometry, RobotCollisionInputs):
            raise TypeError("geometry must be RobotCollisionInputs.")
        if not isinstance(scope, NativeQueryScope):
            raise TypeError("scope must be NativeQueryScope.")
        self._geometry, self._scope = geometry, scope
        self._root = _matrix(robot_root_pose)
        self._closed = False
        self.assert_current()

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed robot for FK while this query is open."""
        if self._closed:
            raise RuntimeError("Robot path query is closed.")
        return self._geometry.robot

    @property
    def scope(self) -> NativeQueryScope:
        """Return the immutable host-owned observation identity."""
        return self._scope

    def assert_current(self) -> None:
        """Reject a closed query, changed source or changed observed robot root."""
        if self._closed:
            raise RuntimeError(
                "Robot path query is closed; prepare a fresh observation."
            )
        self._geometry.assert_current()
        if not np.array_equal(
            _matrix(self.robot.get_local_pose(to_matrix=True)[0]), self._root
        ):
            raise ValueError("The observed robot root frame changed during the query.")

    def iter_link_vertices(
        self,
        positions: torch.Tensor,
        names: tuple[str, ...],
        frames: list[int],
        inverse: np.ndarray | None = None,
    ) -> Iterator[tuple[int, str, LinkMesh, torch.Tensor]]:
        """Yield every selected frame/component after source-frame FK.

        Args:
            positions: Finite full-robot samples, shape ``(1, T, J)``.
            names: Unique captured collision links in explicit order.
            frames: Nonempty unique sample indices in explicit order.
            inverse: Optional rigid transform applied after the observed root.

        Yields:
            Frame index, link name, source component and transformed vertices.
            FK retains the original 64-frame bound and source component order.
        """
        self.assert_current()
        return self._iter_link_vertices(positions, names, frames, inverse)

    def _iter_link_vertices(
        self,
        positions: torch.Tensor,
        names: tuple[str, ...],
        frames: list[int],
        inverse: np.ndarray | None = None,
    ) -> Iterator[tuple[int, str, LinkMesh, torch.Tensor]]:
        if len(set(names)) != len(names) or not set(names) <= set(
            self._geometry.link_names
        ):
            raise ValueError("Path links must be unique captured collision links.")
        if inverse is not None:
            inverse = _matrix(inverse)
        if (
            positions.ndim != 3
            or positions.shape[0] != 1
            or positions.shape[2] != len(self.robot.joint_names)
            or not torch.isfinite(positions).all()
        ):
            raise ValueError("Path geometry requires a finite full-robot trajectory.")
        if (
            not frames
            or any(
                type(index) is not int or not 0 <= index < positions.shape[1]
                for index in frames
            )
            or len(set(frames)) != len(frames)
        ):
            raise ValueError(
                "Collision frame indices must be nonempty, exact and unique."
            )
        meshes = self._geometry.meshes
        expected = {
            (index, name, mesh.path)
            for index in frames
            for name in names
            for mesh in meshes[name]
        }
        count = len(frames) * sum(len(meshes[name]) for name in names)
        if not expected or len(expected) != count:
            raise ValueError("Collision scope is empty or duplicated.")
        seen = set()
        for indices in torch.tensor(frames, device=positions.device).split(_FK_BATCH):
            fk = self.robot.compute_fk(
                qpos=positions[0, indices],
                link_names=list(names),
                qpos_joint_names=self.robot.joint_names,
            )
            if (
                fk.shape != (len(indices), len(names), 4, 4)
                or not torch.isfinite(fk).all()
            ):
                raise ValueError("Full FK differs from the declared collider scope.")
            poses = (
                torch.as_tensor(self._root, dtype=fk.dtype, device=fk.device)[
                    None, None
                ]
                @ fk
            )
            if inverse is not None:
                poses = (
                    torch.as_tensor(inverse, dtype=fk.dtype, device=fk.device)[
                        None, None
                    ]
                    @ poses
                )
            for column, name in enumerate(names):
                for mesh in meshes[name]:
                    vertices = positions.new_tensor(mesh.vertices.copy())
                    points = (
                        vertices[None] @ poses[:, column, :3, :3].transpose(-1, -2)
                        + poses[:, column, None, :3, 3]
                    )
                    for index, value in zip(indices.tolist(), points, strict=True):
                        key = (index, name, mesh.path)
                        if (
                            key not in expected
                            or key in seen
                            or value.ndim != 2
                            or value.shape[1:] != (3,)
                            or not len(value)
                            or not torch.isfinite(value).all()
                        ):
                            raise ValueError(
                                "Collision frame coverage or transformed vertices are invalid."
                            )
                        seen.add(key)
                        yield index, name, mesh, value
        if seen != expected:
            raise ValueError("Collision frame coverage is incomplete.")

    def inspect_support_path(
        self,
        support: SupportCollisionInputs,
        positions: torch.Tensor,
        frame_indices: Sequence[int],
        *,
        required_link_names: Sequence[str] = (),
    ) -> SupportPathReport:
        """Inspect selected command frames using source/native AABB separation.

        Args:
            support: Exact borrowed static/kinematic support capture.
            positions: Finite full-robot samples, shape ``(1, T, J)``.
            frame_indices: Nonempty unique rows selected by the caller.
            required_link_names: Explicit colliders that must be in the moving
                native scope. No hand, pad or phase names are inferred.

        Returns:
            Supported acceptance/rejection or unavailable evidence with a reason.
            Clearance subtracts each native contact skin; only positive margins
            pass. Box overlap does not establish actual mesh collision.

        Raises:
            RuntimeError: The query has already been closed.
        """
        if self._closed:
            raise RuntimeError("Robot path query is closed.")
        try:
            self.assert_current()
            self._geometry.require_support_owner(support)
            bounds, support_contact = support.bounds()
            frames = list(frame_indices)
            names = self._geometry.moving_link_names()
            if not names or not set(required_link_names) <= set(names):
                raise ValueError("Moving scope omits required collision links.")
            offsets = self._geometry.contact_offsets(names)
            minimum = float("inf")
            for _, name, _, points in self._iter_link_vertices(
                positions, names, frames
            ):
                low, high = points.amin(0), points.amax(0)
                support_bounds = points.new_tensor(bounds)
                gap = torch.maximum(
                    support_bounds[0] - high, low - support_bounds[1]
                ).clamp_min(0)
                minimum = min(
                    minimum,
                    float(torch.linalg.vector_norm(gap))
                    - offsets[name]
                    - support_contact,
                )
            if not math.isfinite(minimum):
                raise ValueError("Support clearance lacks finite complete evidence.")
            return SupportPathReport(True, minimum > 0, minimum, len(frames), names)
        except _QUERY_ERRORS as error:
            return SupportPathReport(False, False, reason=str(error))

    def close(self) -> None:
        """End this query without closing its caller-owned geometry captures."""
        self._closed = True
        self._geometry = None

    def __enter__(self) -> RobotPathQuery:
        self.assert_current()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
