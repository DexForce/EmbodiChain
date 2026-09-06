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
import warp as wp

from embodichain.lab.sim.sensors._warp.contact import scatter_contact_data
from embodichain.utils.warp.kernels import scatter_contact_data as legacy_scatter


def test_contact_kernel_compatibility_and_per_environment_capacity() -> None:
    assert legacy_scatter is scatter_contact_data
    wp.init()
    # Three contacts in two environments; the second row in env 0 overflows.
    contacts = wp.array(np.arange(33, dtype=np.float32).reshape(3, 11), device="cpu")
    users = wp.array([[10, 11], [12, 13], [20, 21]], dtype=wp.int32, device="cpu")
    env_ids = wp.array([0, 0, 1], dtype=wp.int32, device="cpu")
    counts = wp.zeros(2, dtype=wp.int32, device="cpu")
    vectors = [wp.zeros((2, 1, 3), dtype=wp.float32, device="cpu") for _ in range(3)]
    scalars = [wp.zeros((2, 1), dtype=wp.float32, device="cpu") for _ in range(2)]
    out_users = wp.zeros((2, 1, 2), dtype=wp.int32, device="cpu")
    valid = wp.zeros((2, 1), dtype=wp.bool, device="cpu")
    wp.launch(
        scatter_contact_data,
        dim=3,
        inputs=[
            contacts,
            users,
            env_ids,
            counts,
            1,
            *vectors,
            *scalars,
            out_users,
            valid,
        ],
        device="cpu",
    )
    np.testing.assert_array_equal(counts.numpy(), [1, 1])
    np.testing.assert_array_equal(valid.numpy(), [[True], [True]])
    np.testing.assert_array_equal(vectors[0].numpy()[1, 0], [22.0, 23.0, 24.0])
    np.testing.assert_array_equal(out_users.numpy()[1, 0], [20, 21])
