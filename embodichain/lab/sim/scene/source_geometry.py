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
"""Neutral owned link-local collider inputs shared by static and robot readers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["LinkMesh"]


@dataclass(frozen=True, slots=True)
class LinkMesh:
    """Owned link-local input geometry, preserving source vertex and face order.

    Args:
        path: Authored collider identity, within its source asset.
        owner_path: Owning link identity; vertices use this link's frame.
        vertices: Finite real ``(V, 3)`` coordinates with source dtype preserved.
        faces: Integer ``(F, 3)`` vertex indices. Empty faces explicitly mean
            vertex-only input and cannot establish a triangle surface.
        approximation: Authored collision approximation declaration.

    Arrays are copied and exposed read-only. This value does not contain native
    cooked shapes, runtime transforms, support evidence or collision results.
    """

    path: str
    owner_path: str
    vertices: np.ndarray
    faces: np.ndarray
    approximation: str

    def __post_init__(self) -> None:
        vertices = np.array(self.vertices, copy=True)
        faces = np.array(self.faces, copy=True)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or vertices.dtype.kind not in "fiu"
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Link mesh vertices must be finite (V, 3) inputs.")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or faces.dtype.kind not in "iu"
            or len(faces)
            and (faces.min() < 0 or faces.max() >= len(vertices))
        ):
            raise ValueError("Link mesh faces must index the captured input vertices.")
        vertices.setflags(write=False)
        faces.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    def __deepcopy__(self, memo: dict[int, Any]) -> LinkMesh:
        copied = LinkMesh(
            self.path, self.owner_path, self.vertices, self.faces, self.approximation
        )
        memo[id(self)] = copied
        return copied
