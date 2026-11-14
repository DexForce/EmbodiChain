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
from embodichain.lab.expansion import MeasuredTrajectory
from embodichain.lab.visualization.markers import (
    MarkerGroup,
    MarkerGroupCfg,
    MarkerPrototypeCfg,
)


def test_measured_comparison_owns_boundaries_and_preserves_marker_identity():
    times = np.array([0.0, 0.1, 0.3])
    poses = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (3, 1))
    clip = MeasuredTrajectory(
        "original",
        "robot",
        ("joint",),
        times,
        np.array([[1.0], [2.0], [3.0]]),
        poses,
        {"object": poses},
    )
    times[1] = 99
    poses[0, 0] = 99
    assert [clip.frame_at(t) for t in (-1.0, 0.0, 0.1, 0.2, 0.3, 5.0)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert clip.root_poses[0, 0] == clip.object_poses["object"][0, 0] == 0
    with pytest.raises(ValueError, match="ordered"):
        MeasuredTrajectory(
            "bad",
            "robot",
            ("joint",),
            np.array([0.0, 0.0]),
            np.zeros((2, 1)),
            np.zeros((2, 7)),
            {},
        )
    group = MarkerGroup(
        MarkerGroupCfg(
            name="ghost", prototypes={"box": MarkerPrototypeCfg(shape="box")}
        ),
        num_envs=1,
        origins=np.zeros((1, 3)),
    )
    group.update(translations=np.zeros((3, 3)), visible=np.array([False, True, False]))
    complete = group.snapshot()
    selected = group.snapshot(visible_only=True)
    assert len(complete) == 3 and len(selected) == 1
    assert selected[0].overlay_id == complete[1].overlay_id
    group.update(visible=np.array([False, False, True]))
    assert group.snapshot(visible_only=True)[0].overlay_id == complete[2].overlay_id
    group.remove()
