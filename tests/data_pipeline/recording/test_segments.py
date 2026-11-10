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

from embodichain.data_pipeline.recording.segments import segments_for_frames


def test_half_open_ranges_preserve_boundaries_and_gaps() -> None:
    first = {"start_step": 0, "end_step": 2, "instruction": "Pick"}
    second = {"start_step": 3, "end_step": 4, "instruction": "Place"}
    empty = {"start_step": 4, "end_step": 4}

    assert segments_for_frames({"segments": [second, empty, first]}, 5) == [
        first,
        first,
        None,
        second,
        None,
    ]


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 5), (3, 2)])
def test_ranges_outside_episode_are_rejected(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="outside episode"):
        segments_for_frames({"segments": [{"start_step": start, "end_step": end}]}, 4)


def test_overlapping_segments_fail_before_persistence() -> None:
    with pytest.raises(ValueError, match="overlap at frame 1"):
        segments_for_frames(
            {
                "segments": [
                    {"start_step": 0, "end_step": 2},
                    {"start_step": 1, "end_step": 3},
                ]
            },
            4,
        )


def test_fractional_boundaries_are_not_silently_truncated() -> None:
    with pytest.raises(TypeError, match="integers"):
        segments_for_frames({"segments": [{"start_step": 0.5, "end_step": 2}]}, 4)
