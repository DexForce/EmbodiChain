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

"""All model-description facade branches have real publicly installed owners."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import pytest
import yaml


@pytest.mark.parametrize(
    "identity",
    ["dual_franka_robotiq_arg2f_140", "dual_ur10_robotiq_arg2f_140", "tianji_marvin"],
)
def test_model_owner_dispatch_has_complete_topology_and_owned_declarations(
    identity, monkeypatch
):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain_robots.task_program import (
        declare_source_task_embodiment,
        resolve_task_hand_calibrations,
        declare_task_contact_pads,
        declare_task_robot_mount,
        require_task_robot_assets,
        task_robot_lateral_positions,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("model metadata created a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    root = Path(__file__).parents[3]
    original = yaml.safe_load(
        (
            root
            / f"extensions/embodichain_tasks/configs/components/embodiments/{identity}.yaml"
        ).read_text()
    )
    before = deepcopy(original)
    declared = declare_source_task_embodiment(original)
    hands = resolve_task_hand_calibrations(declared)
    assert len(hands) == 2 and {hand.resource_id for hand in hands} == {"left", "right"}
    assert original == before
    expected_model = (
        "tianji_marvin_gripper" if identity == "tianji_marvin" else "robotiq_arg2f_140"
    )
    assert {hand.nominal_envelope_declaration()["model_id"] for hand in hands} == {
        expected_model
    }
    assert all(
        hand.opening_driver_indices(
            tuple(declared["simulation"]["control_parts"][hand.hand_part])
        )
        for hand in hands
    )
    overlay = declare_task_contact_pads(declared)
    assert "gen_sim_contact_pads" in overlay["simulation"]["link_attrs"]
    assert "gen_sim_contact_pads" not in declared["simulation"].get("link_attrs", {})
    mounted = declare_task_robot_mount(
        declared, table_top_z=0.73, table_bounds=[[0.0, -0.5, 0.7], [0.8, 0.5, 0.73]]
    )
    assert mounted["embodiment_id"] == identity
    hints = task_robot_lateral_positions(identity, mounted)
    assert (
        hints is None
        if identity.startswith("dual_franka")
        else set(hints) == {"left", "right"}
    )
    require_task_robot_assets(identity)
