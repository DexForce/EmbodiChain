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

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["to_numpy_array"]


def to_numpy_array(
    value: object,
    dtype: npt.DTypeLike,
    *,
    copy: bool = True,
) -> np.ndarray:
    """Convert tensor-like data to a contiguous CPU NumPy array.

    Args:
        value: Tensor-like or array-like source value.
        dtype: Requested NumPy data type.
        copy: Whether the returned array must own an independent copy.

    Returns:
        A contiguous NumPy array on the CPU.
    """
    detached = value.detach() if hasattr(value, "detach") else value
    cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
    numpy_value = cpu_value.numpy() if hasattr(cpu_value, "numpy") else cpu_value
    array = np.ascontiguousarray(np.asarray(numpy_value, dtype=dtype))
    return array.copy() if copy else array
