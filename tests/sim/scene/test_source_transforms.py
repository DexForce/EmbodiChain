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

"""Physical source transforms retain the frozen source numeric/error contract."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import pytest
from embodichain.lab.sim.scene.source_transforms import (
    normalize_source_pose,
    apply_source_body_scale,
    rotate_source_world_z,
)

_BASELINE = json.loads(
    (Path(__file__).parent / "data/source_transforms_b0.json").read_text()
)


@pytest.mark.parametrize("case", _BASELINE["cases"], ids=lambda row: row["id"])
def test_public_transform_keeps_source_values_order_rounding_and_errors(case):
    selected = deepcopy(case["config"])
    functions = {
        "pose": normalize_source_pose,
        "scale": apply_source_body_scale,
        "rotate": rotate_source_world_z,
    }

    def apply():
        for operation, arguments in case["steps"]:
            functions[operation](selected, **arguments)

    expected = case["expected"]
    if "error_type" in expected:
        with pytest.raises(Exception) as raised:
            apply()
        assert type(raised.value).__name__ == expected["error_type"]
        assert str(raised.value) == expected["error"]
    else:
        apply()
        assert (
            json.dumps(
                selected, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
            == expected["serialized"]
        )


def test_unknown_scale_policy_keeps_existing_public_source_preflight_rejection():
    # B0 prepare_scene validates this enum before reading its source. Its private
    # helper fallthrough is preserved only as historical evidence, not a mode.
    with pytest.raises(
        ValueError, match="policy must be preserve, multiply, or absolute"
    ):
        apply_source_body_scale(
            {"body_scale": [1, 2, 3]}, policy="unsupported", requested=[1, 1, 1]
        )


@pytest.mark.parametrize("requested", [[1, 2], [1, -1, 2]])
def test_standalone_scale_ingress_rejects_malformed_vectors_before_mutation(requested):
    source = {"body_scale": [1, 2, 3]}
    before = deepcopy(source)
    with pytest.raises(ValueError, match="vector"):
        apply_source_body_scale(source, policy="multiply", requested=requested)
    assert source == before
