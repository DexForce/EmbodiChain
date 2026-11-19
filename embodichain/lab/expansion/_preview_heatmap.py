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

"""Measured object-position coverage for the expansion replay."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._preview_data import PreviewData

__all__: list[str] = []

_BINS = 32


def _grid_edges(bounds: np.ndarray) -> np.ndarray:
    """Build equal-length XY axes with a margin and a useful scale for still objects."""
    desired = max(float(np.max(bounds[1] - bounds[0])) / (_BINS - 2), 0.001)
    magnitude = 10.0 ** math.floor(math.log10(desired))
    step = next(n * magnitude for n in (1, 2, 2.5, 5, 10) if n * magnitude >= desired)
    center = np.rint(bounds.mean(axis=0) / step)
    return (center[:, None] + np.arange(_BINS + 1) - _BINS // 2) * step


def object_heatmaps(data: PreviewData) -> dict[str, Any]:
    """Count the distinct recordings with a measured object origin in each XY cell."""
    clips = data.all_clips
    specs = data.object_specs or {}
    uids = sorted(
        data.solid_clip.measured.object_poses,
        key=lambda uid: (
            specs.get(uid, {}).get("body_type") in {"static", "kinematic"},
            uid,
        ),
    )
    objects = []
    for uid in uids:
        paths = [clip.measured.object_poses[uid][:, :2] for clip in clips]
        bounds = {
            "all": np.array(
                [
                    np.min([p.min(axis=0) for p in paths], axis=0),
                    np.max([p.max(axis=0) for p in paths], axis=0),
                ],
                dtype=float,
            )
        }
        for phase, index in (("start", 0), ("end", -1)):
            xy = np.stack([p[index] for p in paths])
            bounds[phase] = np.stack([xy.min(axis=0), xy.max(axis=0)])

        x_edges, y_edges = _grid_edges(bounds["all"])
        counts = {phase: np.zeros((_BINS, _BINS), dtype=np.int32) for phase in bounds}
        for xy in paths:
            column = x_edges.searchsorted(xy[:, 0], side="right") - 1
            row = y_edges.searchsorted(xy[:, 1], side="right") - 1
            cells = np.ravel_multi_index((row, column), (_BINS, _BINS))
            visited = np.bincount(cells, minlength=_BINS**2) > 0
            counts["all"] += visited.reshape(_BINS, _BINS)
            counts["start"].flat[cells[0]] += 1
            counts["end"].flat[cells[-1]] += 1
        objects.append(
            {
                "uid": uid,
                "x_edges_m": x_edges.tolist(),
                "y_edges_m": y_edges.tolist(),
                "views": {
                    phase: {"counts": grid.tolist(), "bounds_m": bounds[phase].tolist()}
                    for phase, grid in counts.items()
                },
            }
        )
    return {
        "episodes": len(clips),
        "explicit_reference": data.reference is not None,
        "objects": objects,
    }
