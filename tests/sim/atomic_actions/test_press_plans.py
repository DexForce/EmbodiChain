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

"""Public Press plans retain complete frozen command, budget and RNG behavior."""

from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from ._press_plan_fixture import press_fixture, press_record

_SOURCE = json.loads(
    (Path(__file__).parent / "data/press_plans_source_reference.json").read_text()
)


@pytest.mark.parametrize("case", _SOURCE["cases"], ids=lambda case: case["mode"])
def test_public_press_plan_preserves_frozen_source_commands_and_recovery(
    tmp_path, case
):
    engine, context, invocation, calls, solves = press_fixture(tmp_path, case["mode"])
    random.seed(913)
    np.random.seed(913)
    torch.manual_seed(913)
    result = press_record(engine.plan(invocation, context), calls, solves)
    assert json.loads(json.dumps(result, allow_nan=False)) == case["expected"]


def test_public_bounded_press_rejects_batch_before_motion_queries(tmp_path):
    engine, context, invocation, calls, solves = press_fixture(
        tmp_path,
        "nominal",
        batch_size=2,
    )
    with pytest.raises(ValueError, match="exactly one environment"):
        engine.plan(invocation, context)
    assert calls == [] and solves == []
