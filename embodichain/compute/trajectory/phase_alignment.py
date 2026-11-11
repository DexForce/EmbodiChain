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

"""Bounded monotone clock fitting to an independently generated joint path."""

from __future__ import annotations

import numpy as np

__all__ = ["align_joint_phase"]


def align_joint_phase(
    native: np.ndarray, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project observations onto native edges with monotone progress.

    The output always lies on the native joint polyline. The observations
    determine only the phase clock. Coarse-to-fine searches operate in bounded
    chunks instead of allocating a full source-by-native distance matrix.
    Call separately between every required skill anchor.

    Args:
        native: Independently generated native path, shaped (N,D).
        observations: Source phase on its unchanged clock, shaped (T,D).

    Returns:
        Native-path samples on the observed clock and normalized native index
        progress. Endpoint and direction mismatches remain visible as errors.
    """
    path, source = (np.asarray(v, dtype=np.float64) for v in (native, observations))
    if (
        path.ndim != 2
        or source.ndim != 2
        or not len(path)
        or not len(source)
        or path.shape[1] != source.shape[1]
        or not np.isfinite(path).all()
        or not np.isfinite(source).all()
        or max(len(path), len(source)) > 200_000
    ):
        raise ValueError("Phase alignment requires bounded finite joint paths.")
    if len(path) == 1 or np.max(np.abs(path - path[0])) < 1e-12:
        return np.repeat(path[:1], len(source), axis=0), np.linspace(0, 1, len(source))
    coarse = np.unique(np.linspace(0, len(path) - 1, min(128, len(path))).astype(int))
    radius = int(np.ceil((len(path) - 1) / max(1, len(coarse) - 1))) + 2
    offsets = np.arange(-radius, radius + 1)
    indices = np.empty(len(source))
    for start in range(0, len(source), 64):
        values = source[start : start + 64]
        distances = np.sum((values[:, None] - path[coarse][None]) ** 2, axis=-1)
        centers = coarse[np.argmin(distances, axis=1)]
        edges = np.clip(centers[:, None] + offsets, 0, len(path) - 2)
        left, delta = path[edges], path[edges + 1] - path[edges]
        squared = np.sum(delta**2, axis=-1)
        fraction = np.clip(
            np.sum((values[:, None] - left) * delta, axis=-1)
            / np.maximum(squared, 1e-30),
            0,
            1,
        )
        error = np.sum(
            (values[:, None] - left - fraction[..., None] * delta) ** 2, axis=-1
        )
        best = np.argmin(error, axis=1)
        indices[start : start + len(values)] = (
            edges[np.arange(len(values)), best] + fraction[np.arange(len(values)), best]
        )
    indices = np.maximum.accumulate(indices)
    indices[0], indices[-1] = 0.0, len(path) - 1.0
    if len(source) == 1:
        indices[0] = len(path) - 1.0
    result = np.column_stack(
        [
            np.interp(indices, np.arange(len(path)), path[:, joint])
            for joint in range(path.shape[1])
        ]
    )
    return result, indices / (len(path) - 1)
