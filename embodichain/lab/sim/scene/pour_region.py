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

"""Owned numeric receiver geometry for source-compatible Pour staging."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable
import numpy as np
from embodichain.compute.geometry.pouring import _vertices
from embodichain.compute.geometry.pour_region import (
    _faces,
    _position_budget,
    region_staging_pose,
)

__all__ = ["PourRegionGeometry", "PourGeometryProvider"]


def _immutable(value: np.ndarray) -> np.ndarray:
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class PourRegionGeometry:
    """Owned source/receiver arrays; observed poses remain fresh call inputs.

    Args:
        vertices: Exact local arrays keyed by explicit source/receiver identities.
        faces: Original receiver triangle indices under matching identities.
        gravity_world: Explicit configured gravity, never a guessed direction.
        proxy_position_tolerance_m: Declared positive inset budget in metres.
    """

    vertices: Mapping[str, np.ndarray]
    faces: Mapping[str, np.ndarray]
    gravity_world: tuple[float, float, float]
    proxy_position_tolerance_m: float

    def __post_init__(self) -> None:
        budget = _position_budget(self.proxy_position_tolerance_m)
        vertices = {
            key: _immutable(_vertices(value)) for key, value in self.vertices.items()
        }
        faces = {
            key: _immutable(_faces(value, len(vertices[key])))
            for key, value in self.faces.items()
        }
        gravity = np.asarray(self.gravity_world)
        if (
            gravity.shape != (3,)
            or np.iscomplexobj(gravity)
            or not np.isfinite(gravity).all()
        ):
            raise ValueError("Pour region requires finite configured gravity.")
        object.__setattr__(self, "vertices", MappingProxyType(vertices))
        object.__setattr__(self, "faces", MappingProxyType(faces))
        object.__setattr__(self, "gravity_world", tuple(float(v) for v in gravity))
        object.__setattr__(self, "proxy_position_tolerance_m", budget)

    def staging_pose(
        self,
        object_id: str,
        receiver_id: str,
        source_pose: np.ndarray,
        receiver_pose: np.ndarray,
        arm_origin_world: np.ndarray,
        *,
        clearance_m: float,
    ) -> np.ndarray:
        """Apply the shared pure rule to current participants and the bound arm.

        Args:
            object_id: Exact source key in the owned vertices mapping.
            receiver_id: Exact receiver key in both mappings.
            source_pose: Current observed world source transform.
            receiver_pose: Current observed world receiver transform.
            arm_origin_world: Current observed active-arm root position.
            clearance_m: Explicit vertical staging reserve.

        Returns:
            Owned numeric staging target, not a native path or effect certificate.
        """
        return region_staging_pose(
            self.vertices[object_id],
            self.vertices[receiver_id],
            self.faces[receiver_id],
            source_pose,
            receiver_pose,
            arm_origin_world,
            clearance_m=clearance_m,
            gravity_world=self.gravity_world,
            proxy_position_tolerance_m=self.proxy_position_tolerance_m,
        )


@runtime_checkable
class PourGeometryProvider(Protocol):
    """Read-only shared cached inputs; pose observation remains with each consumer."""

    @property
    def scene_vertices(self) -> Mapping[str, np.ndarray]:
        """Return owned source/receiver vertex snapshots keyed by canonical identity."""
        ...

    @property
    def gravity_world(self) -> tuple[float, float, float]:
        """Return actual configured world gravity retained with these inputs."""
        ...

    def region_geometry(
        self, object_id: str, receiver_id: str, proxy_position_tolerance_m: float
    ) -> PourRegionGeometry | None:
        """Return the exact cache, None for an absent pair, or reject a conflicting budget."""
        ...
