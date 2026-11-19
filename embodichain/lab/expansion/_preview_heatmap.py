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

"""Measured starting positions of object_pose layout targets in the expansion replay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._preview_data import PreviewData

__all__: list[str] = []

_BINS = 32


def _grid_edges(bounds: np.ndarray) -> np.ndarray:
    """Fit each axis with half-cell margins or a 1 mm window for a constant coordinate."""
    low, high = bounds
    span = high - low
    step = np.where(span > 0, span / (_BINS - 1), 0.001 / _BINS)
    # Keep adjacent edges distinguishable at the measured world coordinates.
    step = np.maximum(step, 2 * np.spacing(np.maximum(np.abs(low), np.abs(high))))
    center = low + span / 2
    return center[:, None] + (np.arange(_BINS + 1) - _BINS / 2) * step[:, None]


def object_heatmaps(data: PreviewData) -> dict[str, Any]:
    """Bin one measured start per targeted rigid object and object_pose input episode."""
    starts: dict[str, list[np.ndarray]] = {}
    episodes = 0
    for clip in data.clips:
        if clip.object_pose_uids:
            episodes += 1
        for uid in clip.object_pose_uids:
            starts.setdefault(uid, []).append(clip.measured.object_poses[uid][0, :2])

    specs = data.object_specs or {}
    uids = sorted(
        starts,
        key=lambda uid: (
            specs.get(uid, {}).get("body_type") in {"static", "kinematic"},
            uid,
        ),
    )
    objects = []
    for uid in uids:
        xy = np.asarray(starts[uid], dtype=float)
        bounds = np.stack([xy.min(axis=0), xy.max(axis=0)])
        x_edges, y_edges = _grid_edges(bounds)
        # Canvas rows are Y and columns are X.
        counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(y_edges, x_edges))
        objects.append(
            {
                "uid": uid,
                "episodes": len(xy),
                "x_edges_m": x_edges.tolist(),
                "y_edges_m": y_edges.tolist(),
                "views": {
                    "start": {
                        "counts": counts.astype(int).tolist(),
                        "bounds_m": bounds.tolist(),
                    }
                },
            }
        )
    return {
        "episodes": episodes,
        "explicit_reference": data.reference is not None,
        "objects": objects,
    }
