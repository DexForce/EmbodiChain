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

import pytest

from embodichain.compute import trajectory
from embodichain.lab.sim.utility import action_utils

# The compatibility surface preserves historical names. New compute APIs are
# consumed through compute and do not automatically enlarge this legacy shim.
LEGACY_TRAJECTORY_EXPORTS = (
    "differentiate_positions",
    "interpolate_with_distance",
    "interpolate_with_nums",
    "resample_in_time",
    "resample_with_distance",
    "retime_to_control_grid",
    "sort_and_padding_key_frame",
    "warp_trajectory_qpos",
)


@pytest.mark.parametrize("name", LEGACY_TRAJECTORY_EXPORTS)
def test_legacy_trajectory_exports_are_compute_functions(name: str) -> None:
    assert getattr(action_utils, name) is getattr(trajectory, name)
