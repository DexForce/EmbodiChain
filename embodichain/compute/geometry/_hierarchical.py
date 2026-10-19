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

"""Pure CPU/CUDA tensor proposals for support-tree XY/yaw layouts.

Inputs contain numeric geometry and integer node indices. Asset loading,
polygon policy, UID adaptation and persistent identities belong to callers.
"""

from __future__ import annotations

import torch

__all__: list[str] = []


def _counter_units(
    *, key: int, start: int, count: int, width: int, device: torch.device
) -> torch.Tensor:
    counter = torch.arange(
        start * width, (start + count) * width, dtype=torch.int64, device=device
    )
    word = (counter & 0xFFFFFFFF) ^ (counter >> 32) ^ key
    for _ in range(2):
        word = ((word ^ (word >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return (((word ^ (word >> 16)).to(torch.float64) + 0.5) / 2**32).reshape(
        count, width
    )


def _matrix_product(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Fixed reduction order avoids batch-size-dependent BLAS dispatch rounding."""
    result = first[..., :, 0, None] * second[..., None, 0, :]
    for index in range(1, first.shape[-1]):
        result = result + first[..., :, index, None] * second[..., None, index, :]
    return result


def _rotate_points(points: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    result = points[..., 0, None] * rotation[..., None, :, 0]
    result = result + points[..., 1, None] * rotation[..., None, :, 1]
    return result + points[..., 2, None] * rotation[..., None, :, 2]


def _subtree_points(
    node: int,
    local: list[torch.Tensor],
    corners: torch.Tensor,
    parents: tuple[int, ...],
) -> torch.Tensor:
    """Conservative complete-body/subtree corners in the node frame."""
    transforms = {
        node: torch.eye(4, dtype=corners.dtype, device=corners.device).expand(
            len(local[0]), 4, 4
        )
    }
    chunks = [corners[node].expand(len(local[0]), 8, 3)]
    for child in range(node + 1, len(parents)):
        if parents[child] in transforms:
            transform = _matrix_product(transforms[parents[child]], local[child])
            transforms[child] = transform
            chunks.append(
                _rotate_points(corners[child], transform[:, :3, :3])
                + transform[:, None, :3, 3]
            )
    return torch.cat(chunks, dim=1)


def _coverage(
    bounds: torch.Tensor,
    integral: torch.Tensor,
    origin: torch.Tensor,
    resolution: float,
) -> torch.Tensor:
    """Check every positive-area intersecting cell using a summed-area table."""
    coordinates = (bounds - origin) / resolution
    lower = torch.floor(coordinates[:, 0] + 1e-10).to(torch.int64)
    upper = torch.ceil(coordinates[:, 1] - 1e-10).to(torch.int64)
    size = torch.tensor(integral.shape, device=bounds.device) - 1
    valid = (
        torch.isfinite(bounds).all(dim=(-1, -2))
        & (lower >= 0).all(-1)
        & (upper <= size).all(-1)
        & (upper > lower).all(-1)
    )
    lo = torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), size)
    hi = torch.minimum(torch.maximum(upper, torch.zeros_like(upper)), size)
    bad = (
        integral[hi[:, 0], hi[:, 1]]
        - integral[lo[:, 0], hi[:, 1]]
        - integral[hi[:, 0], lo[:, 1]]
        + integral[lo[:, 0], lo[:, 1]]
    )
    return valid & (bad == 0)


def _fixed_prefix_sum(weights: torch.Tensor) -> torch.Tensor:
    """Use a fixed scan tree independent of CUDA's single/multi-row dispatch."""
    cumulative = weights
    offset = 1
    while offset < weights.shape[1]:
        cumulative = torch.cat(
            (cumulative[:, :offset], cumulative[:, offset:] + cumulative[:, :-offset]),
            dim=1,
        )
        offset *= 2
    return cumulative


def _sample_free_aabb_region(
    region: torch.Tensor,
    half: torch.Tensor,
    obstacles: torch.Tensor,
    unit: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample once from the free center rectangles of already placed AABBs.

    Obstacle edges partition each bounded center region into disjoint cells.
    Selecting a free cell by area avoids collision retries. Zero-width axes
    retain their lower-dimensional feasible measure, including exact-fit points.
    """
    low, high = region[:, 0], region[:, 1]
    forbidden_low = obstacles[:, :, 0] - half[:, None, :] - margin
    forbidden_high = obstacles[:, :, 1] + half[:, None, :] + margin
    edges = torch.cat(
        (low[:, None, :], high[:, None, :], forbidden_low, forbidden_high), dim=1
    )
    edges = (
        torch.minimum(torch.maximum(edges, low[:, None, :]), high[:, None, :])
        .sort(dim=1)
        .values
    )
    width = edges[:, 1:] - edges[:, :-1]
    middle = edges[:, :-1] + width / 2
    blocked = torch.zeros(
        (len(region), width.shape[1], width.shape[1]),
        dtype=torch.bool,
        device=region.device,
    )
    # Keep obstacle reduction memory quadratic in edge count. Materializing
    # every obstacle simultaneously would add an unbounded cubic temporary.
    for start in range(0, obstacles.shape[1], 8):
        stop = min(start + 8, obstacles.shape[1])
        inside_x = (
            middle[:, :, None, 0] > forbidden_low[:, None, start:stop, 0] + 1e-10
        ) & (middle[:, :, None, 0] < forbidden_high[:, None, start:stop, 0] - 1e-10)
        inside_y = (
            middle[:, :, None, 1] > forbidden_low[:, None, start:stop, 1] + 1e-10
        ) & (middle[:, :, None, 1] < forbidden_high[:, None, start:stop, 1] - 1e-10)
        blocked |= (inside_x[:, :, None, :] & inside_y[:, None, :, :]).any(dim=-1)
    # A collapsed axis has exactly one representative cell, rather than many
    # duplicate zero-width cells at repeated clipping boundaries.
    collapsed = high == low
    first = torch.arange(width.shape[1], device=width.device)[None, :] == 0
    wx = torch.where(
        collapsed[:, None, 0] & first, torch.ones_like(width[:, :, 0]), width[:, :, 0]
    )
    wy = torch.where(
        collapsed[:, None, 1] & first, torch.ones_like(width[:, :, 1]), width[:, :, 1]
    )
    weights = torch.where(blocked, 0.0, wx[:, :, None] * wy[:, None, :]).flatten(1)
    cumulative = _fixed_prefix_sum(weights)
    total = cumulative[:, -1]
    index = (
        torch.searchsorted(
            cumulative, (unit[:, 0] * total)[:, None].contiguous(), right=True
        )
        .flatten()
        .clamp_max(weights.shape[1] - 1)
    )
    ix, iy = index // width.shape[1], index % width.shape[1]
    rows = torch.arange(len(region), device=region.device)
    xy = torch.stack(
        (
            edges[rows, ix, 0] + unit[:, 1] * width[rows, ix, 0],
            edges[rows, iy, 1] + unit[:, 2] * width[rows, iy, 1],
        ),
        dim=1,
    )
    valid = (high >= low).all(dim=-1) & (total > 0) & torch.isfinite(xy).all(dim=-1)
    return xy, valid


def _sample_hierarchy(
    relative: torch.Tensor,
    corners: torch.Tensor,
    parents: tuple[int, ...],
    supports: dict[int, dict],
    targets: dict[int, dict],
    root_keys: dict[int, int],
    table: torch.Tensor,
    *,
    start: int,
    count: int,
    randomize_groups: bool,
    sibling_margin: float,
    group_margin: float,
    support_gap: float,
    movable_roots: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One proposal per ordinal, with no per-member retry loops."""
    local = [matrix.expand(count, 4, 4).clone() for matrix in relative]
    valid = torch.ones(count, dtype=torch.bool, device=relative.device)
    sampled = {}
    for node in reversed(range(len(parents))):
        if node in targets:
            spec = targets[node]
            support = supports[parents[node]]
            frame = support["frame"]
            original = _matrix_product(torch.linalg.inv(frame), relative[node])
            unit = _counter_units(
                key=spec["key"],
                start=start,
                count=count,
                width=3,
                device=relative.device,
            )
            yaw = spec["yaw"][0] + unit[:, 2] * (spec["yaw"][1] - spec["yaw"][0])
            angle = yaw * (torch.pi / 180)
            rotation = torch.zeros(
                (count, 3, 3), dtype=relative.dtype, device=relative.device
            )
            rotation[:, 0, 0] = rotation[:, 1, 1] = torch.cos(angle)
            rotation[:, 1, 0] = torch.sin(angle)
            rotation[:, 0, 1] = -torch.sin(angle)
            rotation[:, 2, 2] = 1
            rotation = _matrix_product(rotation, original[:3, :3])
            points = _rotate_points(
                _subtree_points(node, local, corners, parents), rotation
            )
            minimum, maximum = points[:, :, :2].amin(1), points[:, :, :2].amax(1)
            size = torch.tensor(support["integral"].shape, device=relative.device) - 1
            lower = support["origin"] - minimum
            upper = support["origin"] + size * support["resolution"] - maximum
            if spec["xy"] is not None:
                lower = torch.maximum(lower, original[:2, 3] + spec["xy"][0])
                upper = torch.minimum(upper, original[:2, 3] + spec["xy"][1])
            valid &= (upper >= lower).all(-1)
            xy = lower + unit[:, :2] * (upper - lower)
            pose = (
                torch.eye(4, dtype=relative.dtype, device=relative.device)
                .expand(count, 4, 4)
                .clone()
            )
            pose[:, :3, :3] = rotation
            pose[:, :2, 3] = xy
            pose[:, 2, 3] = (
                support_gap + support.get("height_ceiling", 0.0) - spec["min_z"]
            )
            local[node] = _matrix_product(frame, pose)
            sampled[node] = torch.cat((xy, yaw[:, None]), dim=1)
        if node in supports:
            support = supports[node]
            inverse = torch.linalg.inv(support["frame"])
            boxes = []
            for child in range(node + 1, len(parents)):
                if parents[child] != node:
                    continue
                transform = _matrix_product(inverse, local[child])
                points = (
                    _rotate_points(
                        _subtree_points(child, local, corners, parents),
                        transform[:, :3, :3],
                    )
                    + transform[:, None, :3, 3]
                )
                bounds = torch.stack(
                    (points[:, :, :2].amin(1), points[:, :, :2].amax(1)), dim=1
                )
                valid &= _coverage(
                    bounds,
                    support["integral"],
                    support["origin"],
                    support["resolution"],
                )
                for other in boxes:
                    gap = torch.maximum(
                        bounds[:, 0] - other[:, 1], other[:, 0] - bounds[:, 1]
                    ).amax(-1)
                    valid &= gap >= sibling_margin
                boxes.append(bounds)
    world = []
    for node, parent in enumerate(parents):
        world.append(
            local[node] if parent < 0 else _matrix_product(world[parent], local[node])
        )
    root_boxes, deltas = [], []
    fixed_boxes = {}
    if randomize_groups and movable_roots is not None:
        for root in root_keys:
            if root in movable_roots:
                continue
            points = (
                _rotate_points(
                    _subtree_points(root, local, corners, parents),
                    world[root][:, :3, :3],
                )
                + world[root][:, None, :3, 3]
            )
            box = torch.stack(
                (points[:, :, :2].amin(1), points[:, :, :2].amax(1)), dim=1
            )
            valid &= (box[:, 0] >= table[0] - 1e-10).all(-1) & (
                box[:, 1] <= table[1] + 1e-10
            ).all(-1)
            for other in root_boxes:
                valid &= (
                    torch.maximum(
                        box[:, 0] - other[:, 1], other[:, 0] - box[:, 1]
                    ).amax(-1)
                    >= group_margin - 1e-10
                )
            fixed_boxes[root] = box
            root_boxes.append(box)
    for root, key in root_keys.items():
        points = (
            _rotate_points(
                _subtree_points(root, local, corners, parents), world[root][:, :3, :3]
            )
            + world[root][:, None, :3, 3]
        )
        box = torch.stack((points[:, :, :2].amin(1), points[:, :, :2].amax(1)), dim=1)
        if root in fixed_boxes:
            deltas.append(
                torch.zeros((count, 2), dtype=relative.dtype, device=relative.device)
            )
            continue
        if randomize_groups:
            unit = _counter_units(
                key=key, start=start, count=count, width=3, device=relative.device
            )
            half = (box[:, 1] - box[:, 0]) / 2
            center_region = torch.stack((table[0] + half, table[1] - half), dim=1)
            obstacles = (
                torch.stack(root_boxes, dim=1)
                if root_boxes
                else box.new_empty((count, 0, 2, 2))
            )
            center, placed = _sample_free_aabb_region(
                center_region, half, obstacles, unit, group_margin
            )
            valid &= placed
            delta = center - box.mean(dim=1)
        else:
            delta = torch.zeros(
                (count, 2), dtype=relative.dtype, device=relative.device
            )
        box = box + delta[:, None, :]
        valid &= (box[:, 0] >= table[0] - 1e-10).all(-1) & (
            box[:, 1] <= table[1] + 1e-10
        ).all(-1)
        for other in root_boxes:
            valid &= (
                torch.maximum(box[:, 0] - other[:, 1], other[:, 0] - box[:, 1]).amax(-1)
                >= group_margin - 1e-10
            )
        root_boxes.append(box)
        deltas.append(delta)
    return (
        torch.stack(deltas, dim=1),
        (
            torch.stack([sampled[node] for node in sorted(targets)], dim=1)
            if targets
            else relative.new_empty((count, 0, 3))
        ),
        valid,
    )
