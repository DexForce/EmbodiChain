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

from embodichain.lab.gym.envs.managers.record import record_camera_data


def test_record_camera_uses_stable_configured_video_name() -> None:
    recorder = record_camera_data.__new__(record_camera_data)
    recorder._name = "audience"
    recorder._video_name = "task0_1_audience"
    recorder._current_episode = 0

    assert recorder._resolve_video_name() == "task0_1_audience"

    recorder._current_episode = 2
    assert recorder._resolve_video_name() == "task0_1_audience_episode_2"


def test_record_camera_keeps_legacy_fallback_name() -> None:
    recorder = record_camera_data.__new__(record_camera_data)
    recorder._name = "validation"
    recorder._video_name = None
    recorder._current_episode = 3

    assert recorder._resolve_video_name() == "episode_3_validation"
