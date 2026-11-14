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
import pytest
from embodichain_models.dynamics.worldparticle.data import load_sample


def test_initial_state_and_reference_follow_config_slice(tmp_path):
    positions = np.arange(24 * 3 * 3, dtype=np.float32).reshape(24, 3, 3)
    velocities = -positions - 1
    path = tmp_path / "sample.npz"
    np.savez(
        path,
        position=positions,
        velocity=velocities,
        feats=np.ones((3, 3)),
        box=np.zeros((1, 3)),
        box_feats=np.ones((1, 3)),
    )
    profile = dict(checkpoint="fancy_fluid", stride=1, reference=True)
    q, qd, *_, reference = load_sample(path, profile, frame_start=10, frame_end=20)
    np.testing.assert_array_equal(q, positions[10])
    np.testing.assert_array_equal(qd, velocities[10])
    np.testing.assert_array_equal(reference, positions[10:20])
    profile["stride"] = 5
    q, qd, *_, reference = load_sample(path, profile, frame_start=1, frame_end=4)
    np.testing.assert_array_equal(q, positions[5])
    np.testing.assert_array_equal(qd, velocities[5])
    np.testing.assert_array_equal(reference, positions[::5][1:4])
    with pytest.raises(ValueError, match="nonempty"):
        load_sample(path, profile, frame_start=10)
