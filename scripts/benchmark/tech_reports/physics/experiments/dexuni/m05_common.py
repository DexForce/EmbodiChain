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
"""Shared M05 inflatable-bag geometry, phases and load metrics.

The bag asset is from
``demo_dexuni_w1_plastic_inflatable_bag_pick_release.py`` (Newton chip bag).
The right W1 arm squeezes the bag across its width and places it on the
worktop; after an unloaded recovery window, the left arm sets a heavy rigid block
on its centre.
"""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import numpy as np

OUTPUT = (
    Path(__file__).resolve().parents[3].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/m05_inflatable_pouch_load"
)
TABLE_Z = 0.86
BAG_START = np.array((0.43, -0.22))
# Plan suggested (0.52, 0); an IK-only scan showed it is unreachable at table
# height for either arm. The current target (0.435, 0) also admits a low,
# slightly tilted left-hand release with the tighter physical contact offsets.
BAG_TARGET = np.array((0.435, 0.0))
BLOCK_START = np.array((0.40, 0.22))
BLOCK_HALF = 0.015  # 30 mm cube; prescribed mass independent of dimensions
BLOCK_MASS = 4.0
# Example bag shape (source OBJ units). Example scale 0.36 (130 x 96 x 52 mm);
# 0.30 (108 x 80 x 43 mm) leaves room in the 98.6 mm open jaw.
BAG_SCALE = 0.30
BAG_REST_BULGE = 0.030
BAG_SOURCE_HALF_WIDTH = 0.133928
BAG_SOURCE_HALF_LENGTH = 0.180
BAG_SOURCE_FACE_HALF_THICKNESS = 0.052
TOP_GROUP_HALF = 0.02  # top-centre node group: within 20 mm (X and Y) of the centre
TIMING = {
    "home": 0.5,
    "bag approach": 2.0,
    "bag descend": 3.0,
    "bag close": 4.0,
    "bag settle": 4.4,
    "bag lift": 6.0,
    "bag carry": 8.0,
    "bag lower": 9.0,
    "bag release": 9.6,
    "right retreat": 10.6,
    "unloaded reference": 14.6,
    "block approach": 15.8,
    "block descend": 17.0,
    "block close": 17.8,
    "block settle": 18.2,
    "block lift": 20.6,
    "block carry": 24.0,
    "block lower": 27.0,
    "block load hold": 27.6,
    "block release": 28.6,
    "left retreat": 30.0,
    "observe": 35.0,
}
PHASES = tuple((end, name) for name, end in TIMING.items())
DURATION = TIMING["observe"]
__all__ = [
    "BAG_START",
    "BAG_TARGET",
    "BAG_SCALE",
    "BAG_REST_BULGE",
    "BLOCK_HALF",
    "BLOCK_MASS",
    "BLOCK_START",
    "OUTPUT",
    "PHASES",
    "TABLE_Z",
    "TIMING",
    "DURATION",
    "chip_bag_mesh",
    "load_metrics",
    "phase",
    "top_centre_group",
    "supports",
]


def supports() -> list[tuple[str, tuple[float, ...], tuple[float, ...]]]:
    """Common physical worktop and legs; target markings are render-only."""
    items = [("table", (0.46, 0.0, TABLE_Z - 0.025), (0.6, 1.64, 0.05))]
    for i, x in enumerate((0.21, 0.71)):
        for j, y in enumerate((-0.74, 0.74)):
            items.append(
                (
                    f"leg_{i}_{j}",
                    (x, y, (TABLE_Z - 0.05) / 2),
                    (0.05, 0.05, TABLE_Z - 0.05),
                )
            )
    return items


def phase(time: float) -> str:
    """Name of the scheduled phase at ``time``."""
    for end, name in PHASES:
        if time <= end + 1e-9:
            return name
    return "observe"


def chip_bag_mesh(
    path: Path, scale: float = BAG_SCALE
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load the example chip bag, bulge, scale and lay it flat at the origin.

    Same steps as ``load_chip_bag_mesh`` of the inflatable-bag example: closed
    two-manifold check, rest bulge, scale (example 0.36) and the example
    rotation, which puts the length along world X, the width along Y and the
    thickness along Z (130 x 96 x 52 mm at 0.36). The result is centred at the origin.

    Returns:
        Vertex positions (float64), triangles and the rest volume (m^3).
    """
    vertices, faces = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        elif fields[0] == "f":
            faces.append(
                [int(value.split("/", maxsplit=1)[0]) - 1 for value in fields[1:]]
            )
    triangles = [(f[0], f[i], f[i + 1]) for f in faces for i in range(1, len(f) - 1)]
    edges = Counter(
        tuple(sorted(e))
        for t in triangles
        for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))
    )
    if any(count != 2 for count in edges.values()):
        raise ValueError(f"Bag OBJ is not a closed two-manifold surface: {path}")
    positions = np.asarray(vertices, dtype=np.float64)
    for vertex in positions:
        x, y, z = vertex
        if abs(y) < 1.0e-8:
            continue
        width = math.cos(0.5 * math.pi * min(1.0, abs(x) / BAG_SOURCE_HALF_WIDTH)) ** 2
        length = (
            math.cos(0.5 * math.pi * min(1.0, abs(z) / BAG_SOURCE_HALF_LENGTH)) ** 2
        )
        face = min(1.0, abs(y) / BAG_SOURCE_FACE_HALF_THICKNESS)
        vertex[1] += math.copysign(1.0, y) * BAG_REST_BULGE * width * length * face
    triangle_array = np.asarray(triangles, dtype=np.int32)
    scaled = positions * scale
    a, b, c = (scaled[triangle_array[:, i]] for i in range(3))
    volume = abs(
        float(np.einsum("ij,ij->i", a - a[0], np.cross(b - a[0], c - a[0])).sum() / 6.0)
    )
    # Example rotation Rz(90) * Rx(90): source (x, y, z) -> world (z, x, y).
    world = np.c_[scaled[:, 2], scaled[:, 0], scaled[:, 1]]
    world -= 0.5 * (world.min(axis=0) + world.max(axis=0))
    return world, triangle_array, volume


def top_centre_group(vertices: np.ndarray) -> np.ndarray:
    """Upper-face nodes within TOP_GROUP_HALF of the centre (material IDs)."""
    centre = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    ids = np.flatnonzero(
        (np.abs(vertices[:, 0] - centre[0]) <= TOP_GROUP_HALF)
        & (np.abs(vertices[:, 1] - centre[1]) <= TOP_GROUP_HALF)
        & (vertices[:, 2] > centre[2])
    )
    if ids.size == 0:
        raise ValueError("Empty top-centre node group")
    return ids


def load_metrics(bag: np.ndarray, block_pose: np.ndarray, group: np.ndarray) -> dict:
    """Bag envelope/centre heights and block position (metres).

    Args:
        bag: Bag node positions, shape (N, 3).
        block_pose: Block world pose as a 4 x 4 matrix.
        group: Top-centre node IDs from :func:`top_centre_group`.
    """
    corners = np.array(
        [(x, y, z, 1.0) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    )
    corners[:, :3] *= BLOCK_HALF
    block_corners = (block_pose @ corners.T).T[:, :3]
    centre = bag.mean(axis=0)
    return {
        "bag_centre": centre,
        "bag_height_m": float(bag[:, 2].max() - bag[:, 2].min()),
        "bag_lowest_z_m": float(bag[:, 2].min()),
        "bag_top_centre_z_m": float(bag[group, 2].mean()),
        "bag_target_xy_error_m": float(np.linalg.norm(centre[:2] - BAG_TARGET)),
        "block_centre": block_pose[:3, 3].copy(),
        "block_bottom_z_m": float(block_corners[:, 2].min()),
    }
