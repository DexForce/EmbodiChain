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

from embodichain.gen_sim.action_engine.domain import (
    RELATIONS,
    TASK_CONTRACTS,
    TASK_TYPES,
    TERMINAL_BEHAVIORS,
    TRANSPORT_DIRECTIONS,
    normalize_placement_relation,
    task_contract,
    task_success_type,
)


def test_task_contract_catalog_covers_the_canonical_protocol() -> None:
    assert set(TASK_CONTRACTS) == set(TASK_TYPES)
    assert all(contract.core_actions for contract in TASK_CONTRACTS.values())
    assert {contract.source_structure for contract in TASK_CONTRACTS.values()} == {
        "articulation",
        "rigid_object",
    }
    assert task_contract("E2").success_type == "object_upright"
    assert task_contract("E5").scene_affordances == {
        "dual_graspable",
        "rigid",
    }


def test_task_contracts_declare_carrier_flow_and_resource_semantics() -> None:
    e1 = task_contract("E1")
    e3 = task_contract("E3")
    e5 = task_contract("E5")

    assert e1.direct_payload_relations == {"on", "inside"}
    assert e1.accepts_direct_payloads
    assert e3.primary_role_field == "source_role"
    assert e5.accepts_direct_payloads
    assert e5.moves_primary_object
    assert e5.resource_mode == "coordinated"
    assert not task_contract("E2").accepts_direct_payloads


def test_e5_success_depends_only_on_terminal_behavior() -> None:
    assert task_success_type("E5", {"terminal_behavior": "hold"}) == (
        "held_by_both_grippers"
    )
    assert task_success_type("E5", {"terminal_behavior": "place"}) == "semantic_goal"
    with pytest.raises(ValueError, match="terminal_behavior"):
        task_success_type("E5", {"terminal_behavior": "none"})


def test_symbolic_transport_values_are_language_neutral_protocol_enums() -> None:
    assert {"on", "inside", "behind", "left_of"} <= RELATIONS
    assert {"none", "up", "left", "world_y"} <= TRANSPORT_DIRECTIONS
    assert TERMINAL_BEHAVIORS == {"none", "hold", "place"}


@pytest.mark.parametrize("relation", ["above", "on_top", "on_top_of"])
def test_released_hover_and_legacy_support_relations_normalize_to_on(
    relation: str,
) -> None:
    assert normalize_placement_relation(relation) == "on"


def test_placement_relation_normalization_rejects_non_spatial_semantics() -> None:
    with pytest.raises(ValueError, match="Unsupported placement relation"):
        normalize_placement_relation("visual_slot")
