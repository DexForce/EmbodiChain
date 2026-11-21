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
"""Shared C01 layout, goods, phases and packing metrics.

C01 follows ``demo_dexuni_supermarket_packing.py``: the same vest-style shopping
bag hangs, unpinned, from its handles on the example's floor-standing rail rack.
The bag and rack are turned -90 deg about the vertical as one rigid layout (bag
length along world Y, rack uprights behind the bag) and stand in a notch cut
into the back of the W1 worktop, so the bag body hangs below the worktop. An IK
scan put the plan's mouth centre (0.52, 0) and pouch input (0.60, -0.23) out of
reach for a downward gripper near worktop height; x <= ~0.48 is reachable. The
left arm packs the example's tetrahedral bread, returns home, and then the right
arm packs the pneumatic milk pouch at a separate drop site. Extra rigid cargo
is retired.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OUTPUT = (
    Path(__file__).resolve().parents[3].parents[2]
    / "outputs/benchmarks/tech_reports/dexuni/c01_supermarket_packing"
)
TABLE_Z = 0.86
# Shopping-bag origin (authored bottom centre). The example's rails sit 0.329 m
# above it; they are put 0.10 m above the worktop, so the bag mouth (0.27 m)
# lies ~0.04 m above the worktop and the bag body hangs into the notch.
RAIL_ABOVE_TABLE = 0.10
RAIL_LOCAL_Z = 0.329
BAG_POSITION = np.array((0.46, 0.0, TABLE_Z + RAIL_ABOVE_TABLE - RAIL_LOCAL_Z))
BAG_YAW_DEG = -90.0
BAG_MOUTH_LOCAL_Z = 0.27
# Worktop notch (open at the back edge) around the turned bag (world x
# 0.369-0.551, y +-0.15) and rack (uprights at x 0.61, y +-0.122), plus >= 20 mm.
# Include the loaded film's swing envelope: at x=0.345 the pouch impact drives
# the front film into the worktop edge. Input goods remain on the side slabs.
NOTCH_X0 = 0.30
NOTCH_HALF_Y = 0.18
WORKTOP_X = (0.16, 0.76)
WORKTOP_HALF_Y = 0.82
# Goods (input centres, XY).
BLOCK_START = np.array((0.40, 0.23))
BLOCK_HALF = 0.025  # 50 mm cube
BLOCK_MASS = 0.25
BREAD_START = np.array((0.36, 0.22))
BREAD_YAW_DEG = 90.0  # 55 mm side across the fingers (world Y)
POUCH_START = np.array((0.48, -0.26))
# Reference two-item spacing, 70 mm along the bag: left bread at +Y and right
# milk at -Y. The retired cube constants remain inactive legacy metadata.
BLOCK_DROP = BAG_POSITION[:2] + np.array((0.0, 0.035))
BREAD_DROP = BAG_POSITION[:2] + np.array((0.0, 0.035))
POUCH_DROP = BAG_POSITION[:2] + np.array((0.0, -0.035))
GOODS = ("bread", "pouch")
GOOD_ARMS = {"bread": "left", "pouch": "right"}

# Phase timetable (end time in s). One arm at the bag at a time.
_ITEM_PHASES = (
    ("approach", 1.5),
    ("descend", 1.0),
    ("close", 0.8),
    ("settle", 0.4),
    ("lift", 1.5),
    ("carry", 2.0),
    ("lower", 1.2),
    ("release", 0.8),
    ("retreat", 1.0),
)


def _timetable() -> tuple[tuple[float, str], ...]:
    rows, t = [(0.5, "home")], 0.5
    for item in GOODS:
        for name, duration in _ITEM_PHASES:
            if item == "pouch" and name == "release":
                duration = 2.0
            t += duration
            rows.append((round(t, 6), f"{item} {name}"))
        if GOOD_ARMS[item] == "left":
            t += 1.0
            rows.append((round(t, 6), "left home"))
    t += 1.0
    rows.append((round(t, 6), "right home"))
    t += 4.0
    rows.append((round(t, 6), "settle"))
    return tuple(rows)


PHASES = _timetable()
DURATION = PHASES[-1][0]
__all__ = [
    "BAG_MOUTH_LOCAL_Z",
    "BAG_POSITION",
    "BAG_YAW_DEG",
    "BLOCK_DROP",
    "BLOCK_HALF",
    "BLOCK_MASS",
    "BLOCK_START",
    "BREAD_DROP",
    "BREAD_START",
    "BREAD_YAW_DEG",
    "DURATION",
    "GOODS",
    "GOOD_ARMS",
    "NOTCH_HALF_Y",
    "NOTCH_X0",
    "OUTPUT",
    "PHASES",
    "POUCH_DROP",
    "POUCH_START",
    "RAIL_LOCAL_Z",
    "TABLE_Z",
    "end_of",
    "bag_groups",
    "bag_world",
    "packing_metrics",
    "pouch_mesh",
    "phase",
    "rack_foot",
    "rack_parts",
    "worktop_pieces",
]


def phase(time: float) -> str:
    """Name of the scheduled phase at ``time``."""
    for end, name in PHASES:
        if time <= end + 1e-9:
            return name
    return PHASES[-1][1]


def end_of(name: str) -> float:
    """End time of a named phase."""
    return next(end for end, label in PHASES if label == name)


def worktop_pieces() -> list[tuple[str, tuple[float, ...], tuple[float, ...]]]:
    """(name, centre, size) of the notched worktop and its legs."""
    x0, x1 = WORKTOP_X
    top = TABLE_Z - 0.025
    side_y = 0.5 * (NOTCH_HALF_Y + WORKTOP_HALF_Y)
    side_w = WORKTOP_HALF_Y - NOTCH_HALF_Y
    items = [
        (
            "table_front",
            (0.5 * (x0 + NOTCH_X0), 0.0, top),
            (NOTCH_X0 - x0, 2 * WORKTOP_HALF_Y, 0.05),
        ),
        (
            "table_left",
            (0.5 * (NOTCH_X0 + x1), side_y, top),
            (x1 - NOTCH_X0, side_w, 0.05),
        ),
        (
            "table_right",
            (0.5 * (NOTCH_X0 + x1), -side_y, top),
            (x1 - NOTCH_X0, side_w, 0.05),
        ),
    ]
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


def _turn(xy) -> np.ndarray:
    a = np.deg2rad(BAG_YAW_DEG)
    c, s_ = np.cos(a), np.sin(a)
    xy = np.asarray(xy, dtype=float)
    return np.stack(
        (c * xy[..., 0] - s_ * xy[..., 1], s_ * xy[..., 0] + c * xy[..., 1]), axis=-1
    )


def bag_world(local: np.ndarray) -> np.ndarray:
    """Authored bag vertices (bottom-centre origin) to world, turned and placed."""
    out = np.array(local, dtype=np.float64)
    out[:, :2] = _turn(out[:, :2])
    return out + BAG_POSITION


def bag_groups(local: np.ndarray) -> dict[str, np.ndarray]:
    """Material node groups from the authored bag: mouth rim and bottom."""
    z = np.asarray(local)[:, 2]
    groups = {
        "mouth": np.flatnonzero(np.abs(z - BAG_MOUTH_LOCAL_Z) <= 0.01),
        "bottom": np.flatnonzero(z <= 0.002),
    }
    if any(ids.size == 0 for ids in groups.values()):
        raise ValueError("Empty bag marker group")
    return groups


def pouch_mesh() -> tuple[np.ndarray, np.ndarray, float]:
    """Build the supermarket demo's unscaled sealed milk pouch and rest volume."""
    import math

    vertices, faces = [], []
    nx, ny = 8, 10
    for side in (-1, 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                u, v = i / nx, j / ny
                z = side * (
                    0.003 + 0.014 * math.sin(math.pi * u) * math.sin(math.pi * v)
                )
                vertices.append((0.066 * (u - 0.5), 0.092 * (v - 0.5), z))
    layer = (nx + 1) * (ny + 1)
    for side in range(2):
        for j in range(ny):
            for i in range(nx):
                a = side * layer + j * (nx + 1) + i
                triangles = [(a, a + 1, a + nx + 2), (a, a + nx + 2, a + nx + 1)]
                faces.extend(t if side else tuple(reversed(t)) for t in triangles)
    perimeter = list(range(nx + 1)) + [j * (nx + 1) + nx for j in range(1, ny + 1)]
    perimeter += [ny * (nx + 1) + i for i in range(nx - 1, -1, -1)]
    perimeter += [j * (nx + 1) for j in range(ny - 1, 0, -1)]
    for a, b in zip(perimeter, perimeter[1:] + perimeter[:1], strict=True):
        faces.extend(((a, b, b + layer), (a, b + layer, a + layer)))
    points = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(faces, dtype=np.int32)
    xyz = points.astype(np.float64)[triangles]
    volume = float(
        abs(np.einsum("ij,ij->i", xyz[:, 0], np.cross(xyz[:, 1], xyz[:, 2])).sum()) / 6
    )
    return points, triangles, volume


def rack_parts(rail_centres_local) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Example rack tubes (name, end a, end b, radius), turned with the bag.

    Source layout per rail (bag frame): upright at local y +0.15 from 0.04 m
    above the floor to the rail, rail from y -0.13 to +0.15, end stop 25 mm.
    """
    parts = []
    for i, (cx, _, cz) in enumerate(rail_centres_local):
        z = BAG_POSITION[2] + cz

        def at(y, height):
            return np.array((*(_turn((cx, y)) + BAG_POSITION[:2]), height))

        parts += [
            (f"rack_upright_{i}", at(0.15, 0.04), at(0.15, z), 0.008),
            (f"rack_rail_{i}", at(-0.13, z), at(0.15, z), 0.005),
            (f"rack_end_{i}", at(-0.13, z), at(-0.13, z + 0.025), 0.005),
        ]
    return parts


def rack_foot() -> tuple[np.ndarray, np.ndarray]:
    """Example rack foot box (centre, size), turned with the bag."""
    centre = np.array((*(_turn((0.0, 0.15)) + BAG_POSITION[:2]), 0.02))
    return centre, np.array((0.16, 0.46, 0.04))


def packing_metrics(bag: np.ndarray, centres: dict[str, np.ndarray], groups) -> dict:
    """Goods centres relative to the bag mouth and bottom marker groups (m).

    A good counts as in the bag when its centre lies in the XY box of the mouth
    group, below the mouth group mean and no more than 5 mm below the lowest bag
    node (the example's PackingChecks bound).
    """
    mouth = bag[groups["mouth"]]
    lo, hi = mouth[:, :2].min(axis=0), mouth[:, :2].max(axis=0)
    mouth_z = float(mouth[:, 2].mean())
    bottom_z = float(bag[groups["bottom"], 2].mean())
    out = {
        "bag_lowest_z_m": float(bag[:, 2].min()),
        "bag_bottom_z_m": bottom_z,
        "bag_mouth_z_m": mouth_z,
        "mouth_xy_box": np.r_[lo, hi],
    }
    for name, c in centres.items():
        inside_xy = bool(np.all(c[:2] >= lo) and np.all(c[:2] <= hi))
        out[f"{name}_centre"] = np.asarray(c)
        out[f"{name}_in_bag"] = bool(
            inside_xy and out["bag_lowest_z_m"] - 0.005 <= c[2] <= mouth_z
        )
    return out
