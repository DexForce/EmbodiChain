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
    ActionControlOverrides,
    ActionPlanningServices,
    ControlCommand,
    ControlPartCommandProfile,
    DisjointSlotEndpoints,
    JointPositionCommand,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)


class _BrokenSnapshotCommand(ControlCommand):
    """Command double whose snapshot violates the public command contract."""

    def snapshot(self) -> ControlCommand:
        """Return an invalid snapshot for validation coverage."""
        return "invalid"  # type: ignore[return-value]

    def equivalent_to(self, other: ControlCommand) -> bool:
        """Return whether another command has this test-only type."""
        return isinstance(other, _BrokenSnapshotCommand)


class _SelfSnapshotCommand(ControlCommand):
    """Command double that leaks its source instance as the snapshot."""

    def snapshot(self) -> ControlCommand:
        """Return this instance in violation of ownership isolation."""
        return self

    def equivalent_to(self, other: ControlCommand) -> bool:
        """Return whether another command has this test-only type."""
        return isinstance(other, _SelfSnapshotCommand)


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


def _contract() -> SkillBindingContract:
    """Return the endpoint contract used by the direct-binding tests."""
    return SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(endpoint_id="motion"),
                    SkillEndpointRequirement(
                        endpoint_id="grasp",
                        required_commands={"grasp": JointPositionCommand},
                    ),
                ),
                constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
            ),
        ),
    )


def _binding(services: ActionPlanningServices):
    """Bind the test contract to concrete robot control parts."""
    return services.bind_control_parts(
        _contract(),
        {"primary": {"motion": "arm", "grasp": "hand"}},
    )


def test_joint_position_command_broadcasts_owned_batch() -> None:
    source = torch.tensor([0.1, 0.2])
    command = JointPositionCommand(source)
    source.fill_(9.0)

    resolved = command.resolve(num_envs=3, control_dof=2, device="cpu")
    resolved[0].fill_(7.0)

    assert torch.allclose(resolved[1:], torch.tensor([[0.1, 0.2], [0.1, 0.2]]))
    assert torch.allclose(command.positions, torch.tensor([0.1, 0.2]))


def test_joint_position_command_rejects_incompatible_control_part() -> None:
    command = JointPositionCommand(torch.zeros(2))

    with pytest.raises(ValueError, match="resolved control part has 3"):
        command.resolve(num_envs=1, control_dof=3, device="cpu")


def test_control_profile_rejects_invalid_command_snapshot_type() -> None:
    with pytest.raises(TypeError, match="snapshot.*ControlCommand"):
        ControlPartCommandProfile(commands={"stop": _BrokenSnapshotCommand()})


def test_control_profile_rejects_command_snapshot_alias() -> None:
    with pytest.raises(TypeError, match="independently owned"):
        ControlPartCommandProfile(commands={"stop": _SelfSnapshotCommand()})


def test_control_profile_rejects_command_name_outer_whitespace() -> None:
    with pytest.raises(ValueError, match="outer whitespace"):
        ControlPartCommandProfile(
            commands={" stop ": JointPositionCommand(torch.zeros(1))}
        )


def test_resource_free_contract_does_not_require_robot_control_parts() -> None:
    robot = object()
    generator = Mock(robot=robot, device=torch.device("cpu"))
    services = ActionPlanningServices(generator)

    binding = services.bind_control_parts(SkillBindingContract(), {})

    assert binding.owner_id == services.binding_owner_id
    assert binding.endpoints == ()


def test_control_profile_is_resolved_from_robot_control_part() -> None:
    resolved = _binding(_services())

    grasp = resolved.endpoint("primary", "grasp").joint_positions(
        "grasp",
        num_envs=2,
        device="cpu",
    )

    assert grasp.tolist() == [[1.0, 1.0], [1.0, 1.0]]
    with pytest.raises(KeyError, match="available commands"):
        resolved.endpoint("primary", "grasp").joint_positions(
            "pinch",
            num_envs=2,
            device="cpu",
        )


def test_invocation_override_replaces_only_resolved_endpoint_snapshot() -> None:
    services = _services()
    override_source = torch.full((2,), 0.4)
    overrides = ActionControlOverrides(
        endpoints={
            "primary": {
                "grasp": {"grasp": JointPositionCommand(override_source)},
            },
        }
    )
    override_source.fill_(8.0)
    binding = _binding(services)

    overridden = services.apply_command_overrides(binding, overrides)
    base = services.apply_command_overrides(binding, ActionControlOverrides())
    overrides.endpoints["primary"]["grasp"]["grasp"].positions.fill_(6.0)  # type: ignore[attr-defined]

    assert torch.allclose(
        overridden.endpoint("primary", "grasp").joint_positions(
            "grasp", num_envs=1, device="cpu"
        ),
        torch.full((1, 2), 0.4),
    )
    assert torch.equal(
        base.endpoint("primary", "grasp").joint_positions(
            "grasp", num_envs=1, device="cpu"
        ),
        torch.ones(1, 2),
    )


def test_override_rejects_endpoint_not_present_in_binding() -> None:
    services = _services()
    binding = _binding(services)
    overrides = ActionControlOverrides(
        endpoints={
            "destination": {
                "grasp": {"open": JointPositionCommand(torch.zeros(2))},
            },
        }
    )

    with pytest.raises(KeyError, match="unbound endpoints"):
        services.apply_command_overrides(binding, overrides)
