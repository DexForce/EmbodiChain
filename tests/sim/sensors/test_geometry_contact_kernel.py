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

"""Geometry force reduction filters exact shapes and counterpart environments."""

from __future__ import annotations

import numpy as np
import warp as wp

from embodichain.lab.sim.sensors.contact._warp.geometry_contact import (
    _reduce_geometry_netforce,
)


def test_geometry_netforce_combines_vectors_and_filters_counterparts() -> None:
    count = wp.array([5], dtype=wp.int32, device="cpu")
    first = wp.array([1, 1, 5, 3, 1], dtype=wp.int32, device="cpu")
    second = wp.array([2, 2, 4, 2, 5], dtype=wp.int32, device="cpu")
    forces = wp.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, -3, 0, 0, 0, 0],
            [20, 0, 0, 0, 0, 0],
            [40, 0, 0, 0, 0, 0],
        ],
        dtype=wp.spatial_vector,
        device="cpu",
    )
    targets = wp.array([-1, 0, -1, -1, 1, -1], dtype=wp.int32, device="cpu")
    objects = wp.array([-1, -1, 0, -1, -1, 1], dtype=wp.int32, device="cpu")
    output = wp.zeros(2, dtype=wp.vec3, device="cpu")
    wp.launch(
        _reduce_geometry_netforce,
        dim=5,
        inputs=[count, first, second, forces, targets, objects, 1],
        outputs=[output],
        device="cpu",
    )
    # Metal-shape and cross-environment forces do not contribute. Summing
    # vectors before taking their norm differs from summing force magnitudes.
    np.testing.assert_allclose(output.numpy(), [[1, 2, 0], [0, 3, 0]])
    output.zero_()
    count.assign(np.array([0], dtype=np.int32))
    wp.launch(
        _reduce_geometry_netforce,
        dim=5,
        inputs=[count, first, second, forces, targets, objects, 1],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_allclose(output.numpy(), 0)
