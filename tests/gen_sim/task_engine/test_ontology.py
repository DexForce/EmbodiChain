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

from embodichain.gen_sim.task_engine import (
    RELATIONS,
    TASK_CONTRACTS,
    TASK_TYPES,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    task_contract,
    task_success_type,
)


def test_task_contract_catalog_covers_e1_through_e9_without_robot_routes() -> None:
    assert set(TASK_CONTRACTS) == set(TASK_TYPES)
    assert all(
        "required_arm" not in contract.applicable_intent_fields
        for contract in TASK_CONTRACTS.values()
    )
    assert {contract.source_structure for contract in TASK_CONTRACTS.values()} == {
        "articulation",
        "rigid_object",
    }
    assert task_contract("E2").success_type == "object_upright"


def test_e5_success_is_semantic_rather_than_gripper_specific() -> None:
    assert task_success_type("E5", {"terminal_behavior": "hold"}) == (
        "object_cooperatively_held"
    )
    assert task_success_type("E5", {"terminal_behavior": "place"}) == "semantic_goal"
    with pytest.raises(ValueError, match="terminal_behavior"):
        task_success_type("E5", {"terminal_behavior": "none"})


def test_symbolic_transport_values_are_language_neutral_protocol_enums() -> None:
    assert {"on", "inside", "behind", "left_of"} <= RELATIONS
    assert {"none", "up", "left", "world_y"} <= TRANSPORT_DIRECTIONS
    assert TERMINAL_BEHAVIORS == {"none", "hold", "place"}
