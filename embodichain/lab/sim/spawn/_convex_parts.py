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

"""Read precomputed object-local convex pieces without decomposition."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = []


def load_convex_parts(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Require non-empty, finite, three-dimensional indexed convex-piece inputs."""
    from dexsim.kit.meshproc.convex_cache import load_obj_as_convex_parts

    source = Path(path).expanduser().resolve(strict=True)
    parts = load_obj_as_convex_parts(str(source))
    if not parts:
        raise ValueError(
            f"Precomputed decomposition contains no convex parts: {source}"
        )
    for vertices, triangles in parts:
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or len(vertices) < 4
            or not np.isfinite(vertices).all()
            or triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or len(triangles) < 4
            or triangles.min() < 0
            or triangles.max() >= len(vertices)
            or np.linalg.matrix_rank(vertices.astype(np.float64) - vertices[0]) != 3
        ):
            raise ValueError(f"Invalid precomputed convex part in {source}")
    return parts
