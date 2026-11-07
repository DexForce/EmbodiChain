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

"""Shared input validation for the Warp solver and offline numerical reference."""

from __future__ import annotations

from numbers import Integral

import numpy as np

__all__ = []


def _limits(value: object, dofs: int) -> np.ndarray:
    """Canonicalize symmetric magnitudes or signed lower/upper bounds."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(dofs, float(array))
    if array.shape == (dofs,):
        array = np.stack((-array, array), axis=-1)
    if array.shape != (dofs, 2) or not np.isfinite(array).all():
        raise ValueError("limits must be finite scalar, (DOF,), or (DOF, 2)")
    if np.any(array[:, 0] > 0) or np.any(array[:, 1] < 0):
        raise ValueError("rest-to-rest limits must contain zero")
    return array


def _validate_sampling(sample_count: int | None, sample_dt: float | None) -> None:
    if (sample_count is None) == (sample_dt is None):
        raise ValueError("specify exactly one of sample_count and sample_dt")
    if sample_count is not None and (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, Integral)
        or sample_count < 2
    ):
        raise ValueError("sample_count must be an integer >= 2")
    if sample_dt is not None and (not np.isfinite(sample_dt) or sample_dt <= 0):
        raise ValueError("sample_dt must be finite and positive")


def _validate_grid_size(grid_size: int) -> None:
    if (
        isinstance(grid_size, bool)
        or not isinstance(grid_size, Integral)
        or grid_size < 3
    ):
        raise ValueError("grid_size must be an integer >= 3")
