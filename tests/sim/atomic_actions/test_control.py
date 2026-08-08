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

"""Tests for control-part semantic command profiles and invocation overrides."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionControlOverrides,
    ActionPlanningServices,
    ControlPartCommandProfile,
    JointPositionCommand,
)


def _services() -> ActionPlanningServices:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.get_joint_ids.side_effect = lambda name: (
        [0, 1, 2] if name == "arm" else [3, 4]
    )
    generator = Mock()
    generator.robot = robot
    generator.device = torch.device("cpu")
    generator.planner.cfg.planner_type = "stub"
    return ActionPlanningServices(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(2),
                grasp=torch.ones(2),
            )
        },
    )


def test_joint_position_command_broadcasts_owned_batch() -> None:
    source = torch.tensor([0.1, 0.2])
    command = JointPositionCommand(source)
    source.fill_(9.0)

    resolved = command.resolve(n_envs=3, control_dof=2, device="cpu")
    resolved[0].fill_(7.0)

    assert torch.allclose(resolved[1:], torch.tensor([[0.1, 0.2], [0.1, 0.2]]))
    assert torch.allclose(command.positions, torch.tensor([0.1, 0.2]))


def test_joint_position_command_rejects_incompatible_control_part() -> None:
    command = JointPositionCommand(torch.zeros(2))

    with pytest.raises(ValueError, match="resolved control part has 3"):
        command.resolve(n_envs=1, control_dof=3, device="cpu")


def test_control_profile_is_resolved_from_robot_control_part() -> None:
    resolved = _services().resolve_binding(
        ActionBinding(
            manipulators={"primary": "arm"},
            end_effectors={"primary": "hand"},
        )
    )

    grasp = resolved.end_effector().joint_positions(
        "grasp",
        n_envs=2,
        device="cpu",
    )

    assert grasp.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    with pytest.raises(KeyError, match="Available commands"):
        resolved.end_effector().joint_positions(
            "pinch",
            n_envs=2,
            device="cpu",
        )


def test_invocation_override_replaces_only_resolved_role_snapshot() -> None:
    services = _services()
    override_source = torch.full((2,), 0.4)
    overrides = ActionControlOverrides(
        end_effectors={
            "primary": {"grasp": JointPositionCommand(override_source)},
        }
    )
    override_source.fill_(8.0)
    binding = ActionBinding(
        manipulators={"primary": "arm"},
        end_effectors={"primary": "hand"},
    )

    overridden = services.resolve_binding(binding, overrides)
    base = services.resolve_binding(binding)
    overrides.end_effectors["primary"]["grasp"].positions.fill_(6.0)  # type: ignore[attr-defined]

    assert torch.allclose(
        overridden.end_effector().joint_positions("grasp", n_envs=1, device="cpu"),
        torch.full((1, 2), 0.4),
    )
    assert torch.equal(
        base.end_effector().joint_positions("grasp", n_envs=1, device="cpu"),
        torch.ones(1, 2),
    )


def test_override_rejects_role_not_present_in_binding() -> None:
    services = _services()
    binding = ActionBinding(end_effectors={"primary": "hand"})
    overrides = ActionControlOverrides(
        end_effectors={
            "destination": {"open": JointPositionCommand(torch.zeros(2))},
        }
    )

    with pytest.raises(KeyError, match="unbound end effector roles"):
        services.resolve_binding(binding, overrides)
