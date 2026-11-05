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
"""Native Marvin task descriptions have no alternate-model or frontend dependency."""

from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import pytest
import yaml


def test_standard_marvin_task_description_keeps_native_component_and_two_physical_hands(
    monkeypatch,
):
    import random
    import numpy as np
    import torch
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.task_program.integrations.hand_calibration import (
        physical_hand_resources,
    )
    from embodichain_robots.tianji.marvin.task_program import (
        declare_tianji_task_program_embodiment as declare_source_task_embodiment,
        tianji_marvin_task_hand_calibrations as resolve_task_hand_calibrations,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("declaration created a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    root = Path(__file__).parents[3]
    component = (
        root
        / "extensions/embodichain_tasks/configs/components/embodiments/tianji_marvin.yaml"
    )
    original_bytes = component.read_bytes()
    original = yaml.safe_load(original_bytes)
    before = deepcopy(original)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    declared = declare_source_task_embodiment(original)
    hands = resolve_task_hand_calibrations(declared)
    assert original == before and component.read_bytes() == original_bytes
    assert random.getstate() == python_state and torch.equal(
        torch.get_rng_state(), torch_state
    )
    actual_numpy = np.random.get_state()
    assert (
        actual_numpy[0] == numpy_state[0]
        and np.array_equal(actual_numpy[1], numpy_state[1])
        and actual_numpy[2:] == numpy_state[2:]
    )
    simulation = declared["simulation"]
    assert (
        simulation["class_type"] == "TianjiMarvin"
        and simulation["with_gripper"] is True
    )
    assert Path(simulation["fpath"]).name == "robot_with_ee_acd.urdf"
    assert len(declared["skill_profile"]["resources"]) == 4
    assert [
        row["resource_id"] for row in physical_hand_resources(declared["skill_profile"])
    ] == ["left", "right"]
    assert len(hands) == 2 and len(simulation["sensors"]) == 2
    presets = {
        row["preset_id"]: row for row in declared["skill_profile"]["command_presets"]
    }
    for hand in hands:
        expected = [
            f"{hand.side.upper()}_HAND_FINGER_1",
            f"{hand.side.upper()}_HAND_FINGER_2",
        ]
        assert simulation["control_parts"][hand.hand_part] == expected
        assert hand.opening_driver_indices(tuple(expected)) == (0,)
        assert (
            hand.nominal_envelope_declaration()["model_id"] == "tianji_marvin_gripper"
        )
        assert hand.revolute_grip_calibration() is None
        assert presets[hand.command_preset]["commands"] == {
            "open": [0.0, 0.0],
            "grasp": [-0.05, -0.05],
        }
        solver = simulation["solver_cfg"][hand.motion_part]
        assert (
            solver["class_type"] == "SSIKSolver"
            and solver["tcp"]
            == before["simulation"]["solver_cfg"][hand.motion_part]["tcp"]
        )
        assert (
            declared["skill_profile"]["runtime_services"]["grasp_pose_generators"][
                hand.hand_part
            ]
            == before["skill_profile"]["runtime_services"]["grasp_pose_generators"][
                hand.hand_part
            ]
        )
    assert declare_source_task_embodiment(declared) == declared


@pytest.mark.parametrize("damage", ["model", "commands", "alias", "tcp"])
def test_standard_marvin_cannot_borrow_other_hand_models_or_conflicting_aliases(damage):
    from embodichain_robots.tianji.marvin.task_program import (
        declare_tianji_task_program_embodiment as declare_source_task_embodiment,
    )

    root = Path(__file__).parents[3]
    original = yaml.safe_load(
        (
            root
            / "extensions/embodichain_tasks/configs/components/embodiments/tianji_marvin.yaml"
        ).read_text()
    )
    profile = original["skill_profile"]
    if damage == "model":
        profile["runtime_services"]["grasp_pose_generators"]["left_eef"][
            "model"
        ] = "robotiq_arg2f_140"
    elif damage == "commands":
        profile["command_presets"][0]["commands"]["grasp"] = [
            0.7,
            -0.7,
            0.7,
            -0.7,
            -0.7,
            0.7,
        ]
    elif damage == "alias":
        profile["resources"][2]["endpoints"][1]["control_part"] = "left_eef"
    else:
        original["simulation"]["solver_cfg"]["left_arm"]["tcp"][0][3] = 0.2
    with pytest.raises(ValueError, match="native model|commands|alias|grasp TCP"):
        declare_source_task_embodiment(original)


def test_native_description_composes_existing_hand_over_through_public_loader(
    monkeypatch,
):
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.task_program.integrations import (
        load_configured_task_program_deployment,
    )
    from embodichain_robots.tianji.marvin.task_program import (
        declare_tianji_task_program_embodiment,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("public declaration composition created a native runtime")

    monkeypatch.setattr(Robot, "__init__", forbidden)
    monkeypatch.setattr(SimulationManager, "__init__", forbidden)
    root = Path(__file__).parents[3]
    component = (
        root
        / "extensions/embodichain_tasks/configs/components/embodiments/tianji_marvin.yaml"
    )
    task = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/hand_over/task.tianji_marvin.yaml"
    )
    declared = declare_tianji_task_program_embodiment(
        yaml.safe_load(component.read_text())
    )
    config = yaml.safe_load(task.read_text())
    deployment = load_configured_task_program_deployment(
        task_program=config["task_program"],
        skill_profile=declared["skill_profile"],
        base_dir=task.parent,
    )
    assert deployment.selection.robot_profile == "tianji_marvin"
    assert deployment.program_path.is_file()
    assert (
        deployment.integration.registration.robot_profile_binding.profile_id
        == "tianji_marvin"
    )
