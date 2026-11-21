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

"""Prepared pinned SSIK reference for source model a526344e32701b7e5c4eec409d11329227407a6f1cc0f70b4bbad67cc5cedb79.

The joint prefix matches ssik.prebuilt.flexiv.rizon4_ik within
5e-07 m/rad representation tolerance. Only the end frame
is adapted. EmbodiChain verifies every result against the original PK TCP.
"""

from __future__ import annotations

import importlib
import numpy as np
from embodichain.lab.sim.motion.solvers._ssik_reference import (
    _prepare_cached_jointlock,
    _solve_cached_jointlock,
)

_PREBUILT = importlib.import_module("ssik.prebuilt.flexiv.rizon4_ik")
_SOURCE_TO_PREBUILT = np.array(
    [
        [
            0.9999999999999961,
            8.783298343050495e-08,
            2.0508053872740786e-10,
            -1.965292173444766e-08,
        ],
        [
            -8.78329833764626e-08,
            0.9999999999999613,
            -2.6349895055732497e-07,
            -2.5506186537849196e-08,
        ],
        [
            -2.051035942356772e-10,
            2.6349895021009313e-07,
            0.999999999999965,
            -0.043000023461067696,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
SOLVER_NAME = _PREBUILT.SOLVER_NAME
_CACHED_GRID = _prepare_cached_jointlock(_PREBUILT)


def solve(target, **kwargs):
    """Return sampled reference branches without implicit rescue/refinement."""
    kwargs.setdefault("allow_refinement", False)
    kwargs.setdefault("allow_rescue", False)
    kwargs.setdefault("native", False)
    transformed = np.asarray(target, dtype=np.float64) @ _SOURCE_TO_PREBUILT
    return _solve_cached_jointlock(_PREBUILT, _CACHED_GRID, transformed, **kwargs)
