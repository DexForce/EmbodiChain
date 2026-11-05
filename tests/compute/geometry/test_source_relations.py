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

"""Static geometry proposals preserve independent B0 classifications and tie order."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from embodichain.compute.geometry.source_relations import (
    source_orientation_label,
    source_support_proposal,
)

CASES = json.loads(
    (Path(__file__).parent / "data/source_relations_b0.json").read_text()
)


@pytest.mark.parametrize("case", CASES["orientation"])
def test_source_orientation_classification(case):
    assert source_orientation_label(case["input"]) == case["expected"]


@pytest.mark.parametrize("case", CASES["support"])
def test_source_support_candidate_and_tie_order(case):
    assert (
        source_support_proposal(
            case["uid"],
            np.asarray(case["child"]),
            case["objects"],
            tolerance=case["tolerance"],
        )
        == case["expected"]
    )
