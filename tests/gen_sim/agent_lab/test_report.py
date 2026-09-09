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

from embodichain.gen_sim.agent_lab.report import motion_summary


def test_real_motion_summary_handles_extra_keyframes_at_the_same_time() -> None:
    samples = [
        {"time": 0, "qpos": [[0]], "objects": {"cup": [[0, 0, 0, 1, 0, 0, 0]]}},
        {"time": 1, "qpos": [[0.5]], "objects": {"cup": [[0, 0, 0.2, 1, 0, 0, 0]]}},
        {"time": 1, "qpos": [[0.5]], "objects": {"cup": [[0, 0, 0.2, 1, 0, 0, 0]]}},
    ]
    result = motion_summary(samples)
    assert result["sampled_max_joint_speed_rad_s"] == pytest.approx(0.5)
    assert result["objects"]["cup"]["max_z_increase_m"] == pytest.approx(0.2)
    assert result["task_success"] is None


def test_empty_recording_does_not_establish_task_success() -> None:
    assert motion_summary([]) == {"samples": 0, "task_success": None}


def test_articulation_displacement_is_reported_separately_from_robot_motion() -> None:
    samples = [
        {
            "time": 0,
            "qpos": [[0]],
            "objects": {},
            "articulations": {"button": {"qpos": [[0]]}},
        },
        {
            "time": 1,
            "qpos": [[0]],
            "objects": {},
            "articulations": {"button": {"qpos": [[-0.003]]}},
        },
    ]
    result = motion_summary(samples)
    assert result["articulations"]["button"]["max_abs_qpos_change"] == pytest.approx(
        0.003
    )
    assert result["task_success"] is None
