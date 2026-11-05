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
"""Frozen source inset proposals preserve emitted transforms and rejection."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from embodichain.compute.geometry.pour_region import region_staging_pose

_CASES = json.loads((Path(__file__).parent / "data/pour_region_b0.json").read_text())[
    "cases"
]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case["name"])
def test_inset_staging_matches_frozen_source(case):
    values = {
        name: np.asarray(value) if isinstance(value, list) else value
        for name, value in case["inputs"].items()
    }
    if values["proxy_position_tolerance_m"] == "infinity":
        values["proxy_position_tolerance_m"] = float("inf")
    before = {
        name: value.copy()
        for name, value in values.items()
        if isinstance(value, np.ndarray)
    }
    expected = case["expected"]
    if "error" in expected:
        with pytest.raises(ValueError) as error:
            region_staging_pose(**values)
        assert str(error.value) == expected["error"]
    else:
        actual = region_staging_pose(**values)
        np.testing.assert_array_equal(actual, np.asarray(expected["target"]))
        assert actual.dtype == np.float64
    for name, original in before.items():
        np.testing.assert_array_equal(values[name], original)
