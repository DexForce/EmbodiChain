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

"""Source complete-request aperture recovery with actual sampled grasp inputs."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json, random
import numpy as np
import torch
import pytest
from ._preshape_plan_fixture import preshape_plan_fixture, preshape_plan_record

_DATA = json.loads(
    (Path(__file__).parent / "data/preshape_recovery_b0.json").read_text()
)


@pytest.mark.parametrize("case", _DATA["cases"], ids=lambda case: case["mode"])
def test_complete_slide_preshape_keeps_source_commands_measurements_and_random_flow(
    tmp_path, case
):
    fixture = preshape_plan_fixture(tmp_path, case["mode"])
    random.seed(901)
    np.random.seed(901)
    torch.manual_seed(901)
    actual = preshape_plan_record(
        fixture, fixture.engine.plan(fixture.invocation, fixture.context)
    )
    assert actual == case["expected"]
    if case["mode"] == "recovered":
        assert actual["success"] == [True] and actual["measurements"] == [
            0.84,
            0.83,
            0.82,
            0.81,
            0.8,
            0.79,
            0.78,
        ]
    if case["mode"] == "aperture_exhausted":
        assert len(actual["measurements"]) == 10 and actual["motion_calls"] == []
    if case["mode"] == "table_only":
        assert actual["success"] == [False] and actual["measurements"] == []


def test_preshape_recovery_never_silently_enables_filter_or_external_nominal(tmp_path):
    fixture = preshape_plan_fixture(tmp_path, "nominal")
    invocation = replace(
        fixture.invocation,
        skill_options=replace(
            fixture.invocation.skill_options, qualify_hand_clearance=False
        ),
    )
    with pytest.raises(ValueError, match="qualified hand/table"):
        fixture.engine.plan(invocation, fixture.context)
    action = (
        fixture.engine.actions["slide"]
        if hasattr(fixture.engine, "actions")
        else fixture.engine._actions["slide"]
    )
    request = fixture.engine._resolve(fixture.invocation)
    with pytest.raises(ValueError, match="no external nominal"):
        action.plan_complete_request(
            request,
            fixture.context,
            build=lambda req: pytest.fail("unexpected build"),
            external_nominal=True,
        )
