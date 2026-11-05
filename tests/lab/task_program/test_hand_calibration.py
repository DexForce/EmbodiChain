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

"""Physical hand alias selection retains the declarative profile boundary."""

from __future__ import annotations
from copy import deepcopy
import pytest
from embodichain.lab.task_program.integrations.hand_calibration import (
    physical_hand_resources,
)


def profile():
    resources = [
        dict(
            resource_id=side,
            endpoints=[
                dict(endpoint_id="motion", control_part=f"{side}_arm"),
                dict(
                    endpoint_id="grasp",
                    control_part=f"{side}_eef",
                    command_preset=f"{side}_commands",
                ),
            ],
        )
        for side in ("left", "right")
    ]
    resources.append(
        dict(
            resource_id="primary",
            members=["right"],
            endpoints=deepcopy(resources[1]["endpoints"]),
        )
    )
    return dict(resources=resources)


def test_single_member_alias_is_one_physical_hand_and_returns_owned_bindings():
    source = profile()
    before = deepcopy(source)
    selected = physical_hand_resources(source)
    assert [row["resource_id"] for row in selected] == ["left", "right"]
    selected[0]["endpoints"][0]["control_part"] = "changed"
    assert source == before


@pytest.mark.parametrize("damage", ["conflict", "cycle", "duplicate"])
def test_ambiguous_hand_aliases_cannot_create_another_physical_owner(damage):
    source = profile()
    if damage == "conflict":
        source["resources"][-1]["endpoints"][1]["control_part"] = "left_eef"
    elif damage == "cycle":
        source["resources"][-1]["members"] = ["primary"]
    else:
        source["resources"].append(
            dict(
                resource_id="another",
                endpoints=deepcopy(source["resources"][0]["endpoints"]),
            )
        )
    with pytest.raises(ValueError, match="alias|cyclic|Duplicate"):
        physical_hand_resources(source)
