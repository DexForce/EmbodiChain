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

"""Logical randomness is invariant to batching and unrelated operator fields."""

from __future__ import annotations
import pytest
from embodichain.lab.expansion import PinnedExpertSource, UniformRangeOperator


@pytest.mark.parametrize("batch", [1, 8, 32])
def test_random_streams_are_slot_and_completion_order_independent(batch):
    source = PinnedExpertSource("task_program", "source-v1")
    op = UniformRangeOperator("mass_scale", {"cube": (0.5, 2.0)})
    expected = [dict(op.sample(source.context(7, i))) for i in range(67)]
    actual = {}
    for start in range(0, 67, batch):
        for i in reversed(range(start, min(start + batch, 67))):
            actual[i] = dict(op.sample(source.context(7, i)))
    assert [actual[i] for i in range(67)] == expected
    assert len({r["cube"] for r in expected}) == 67
    other = UniformRangeOperator(
        "mass_scale", {"other": (0.0, 1.0), "cube": (0.5, 2.0)}
    )
    assert other.sample(source.context(7, 3))["cube"] == expected[3]["cube"]
    assert (
        op.sample(PinnedExpertSource("task_program", "source-v2").context(7, 3))["cube"]
        != expected[3]["cube"]
    )
    with pytest.raises(TypeError):
        op.sample(source.context(7, 0))["cube"] = 0
