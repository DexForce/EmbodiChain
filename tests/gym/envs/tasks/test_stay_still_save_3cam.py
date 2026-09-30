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

from types import SimpleNamespace

import torch

from embodichain_tasks.special.stay_still_save_3cam import (
    StayStillSave3CamEnv,
)


def test_stay_still_three_camera_demo_has_300_steps() -> None:
    """The benchmark task emits one deterministic segment of 300 actions."""

    env = StayStillSave3CamEnv.__new__(StayStillSave3CamEnv)
    env.robot = SimpleNamespace(get_qpos=lambda: torch.zeros(1, 6))

    (segment,) = tuple(env.create_demo_segments())
    actions = list(segment.actions)

    assert segment.name == "hold_still_300_steps"
    assert segment.progress_total_steps == 300
    assert len(actions) == 300
    assert all(action.shape == (1, 6) for action in actions)
