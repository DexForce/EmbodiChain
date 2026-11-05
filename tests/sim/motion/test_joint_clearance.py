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

"""Bounded offset selection preserves source candidates, sweeps and random flow."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
from embodichain.lab.sim.motion.joint_clearance import select_clearance_joint_offset
from _joint_clearance_fixture import clearance_fixture, record_clearance

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/idle_clearance_source.json").read_text()
)


@pytest.mark.parametrize("case", _REFERENCE["cases"], ids=lambda case: case["mode"])
def test_idle_offset_source_records(tmp_path, case):
    robot, qpos, query, calls = clearance_fixture(tmp_path, case["mode"])
    assert (
        record_clearance(
            lambda: select_clearance_joint_offset(
                robot,
                qpos,
                active_control_part="active",
                joint_name="offset_joint",
                maximum_offset=0.6,
                protected_link_name="protected",
                query=query,
            ),
            calls,
        )
        == case["expected"]
    )


def test_idle_clearance_rejects_multiple_rows_before_query(tmp_path):
    import torch

    robot, qpos, query, calls = clearance_fixture(
        tmp_path, _REFERENCE["cases"][0]["mode"]
    )
    with pytest.raises(ValueError, match="B1"):
        select_clearance_joint_offset(
            robot,
            qpos.expand(2, -1),
            active_control_part="active",
            joint_name="offset_joint",
            maximum_offset=0.6,
            protected_link_name="protected",
            query=query,
        )
    assert calls == []
