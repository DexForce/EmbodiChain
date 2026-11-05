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


"""Carrier-envelope rows against the frozen source calculation."""

from __future__ import annotations
from pathlib import Path
import json
import torch
from embodichain.compute.geometry.cargo import inside_carrier_envelope


def test_carrier_envelope_matches_frozen_source_rows():
    case = json.loads(
        (Path(__file__).parent / "data/cargo_envelope_reference.json").read_text()
    )
    actual = inside_carrier_envelope(
        *(
            torch.tensor(case[key], dtype=torch.float64)
            for key in ("vertices", "object_pose", "carrier_pose", "bounds")
        )
    )
    assert actual.tolist() == case["result"]
