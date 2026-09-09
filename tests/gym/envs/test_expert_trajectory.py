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

"""Tests for source-neutral expert joint trajectory contracts."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_trajectory import (
    ExpertTrajectoryCfg,
    build_expert_action_spec,
    encode_expert_action,
    prepare_expert_joint_trajectory,
)
from embodichain.lab.sim.motion.planners import PlanResult


def test_expert_command_mode_defaults_to_position() -> None:
    cfg = ExpertTrajectoryCfg()

    assert cfg.joint_command_mode == "position"


def test_expert_command_mode_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="joint_command_mode"):
        ExpertTrajectoryCfg(joint_command_mode="velocity")


def test_embodied_environment_owns_expert_trajectory_configuration() -> None:
    cfg = EmbodiedEnvCfg()

    assert isinstance(cfg.expert_trajectory, ExpertTrajectoryCfg)
    assert cfg.expert_trajectory.joint_command_mode == "position"


def test_timed_expert_trajectory_is_retimed_and_velocity_is_derived() -> None:
    source = PlanResult(
        success=torch.ones(1, dtype=torch.bool),
        positions=torch.tensor([[[0.0], [1.0]]]),
        velocities=torch.full((1, 2, 1), 99.0),
        dt=torch.tensor([[0.0, 0.75]]),
    )

    prepared = prepare_expert_joint_trajectory(
        source,
        control_dt=0.5,
        joint_command_mode="position_velocity",
    )

    torch.testing.assert_close(
        prepared.positions[0, :, 0], torch.tensor([0.0, 0.5, 1.0])
    )
    torch.testing.assert_close(
        prepared.velocities[0, :, 0], torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(prepared.dt, torch.tensor([[0.0, 0.5, 0.5]]))


def test_plan_result_is_the_canonical_expert_trajectory_input() -> None:
    result = PlanResult(
        success=torch.ones(1, dtype=torch.bool),
        positions=torch.tensor([[[0.0], [1.0]]]),
        dt=torch.tensor([[0.0, 0.75]]),
    )

    prepared = prepare_expert_joint_trajectory(
        result,
        control_dt=0.5,
        joint_command_mode="position_velocity",
    )

    assert isinstance(prepared, PlanResult)
    torch.testing.assert_close(
        prepared.positions[0, :, 0], torch.tensor([0.0, 0.5, 1.0])
    )
    torch.testing.assert_close(
        prepared.velocities[0, :, 0], torch.tensor([0.0, 1.0, 0.0])
    )


def test_untimed_position_velocity_trajectory_requires_velocity() -> None:
    source = PlanResult(
        success=torch.ones(1, dtype=torch.bool), positions=torch.zeros(1, 2, 1)
    )

    with pytest.raises(ValueError, match="velocities"):
        prepare_expert_joint_trajectory(
            source,
            control_dt=0.1,
            joint_command_mode="position_velocity",
        )


def test_position_mode_preserves_untimed_positions_without_velocity() -> None:
    positions = torch.tensor([[[0.0], [1.0]]])

    prepared = prepare_expert_joint_trajectory(
        PlanResult(success=torch.ones(1, dtype=torch.bool), positions=positions),
        control_dt=0.1,
        joint_command_mode="position",
    )

    torch.testing.assert_close(prepared.positions, positions)
    assert prepared.velocities is None
    torch.testing.assert_close(prepared.dt, torch.tensor([[0.0, 0.1]]))


def test_position_velocity_action_spec_has_documented_layout() -> None:
    spec = build_expert_action_spec(
        joint_names=["shoulder", "elbow"],
        joint_command_mode="position_velocity",
    )

    assert spec.width == 4
    assert spec.qpos_slice == (0, 2)
    assert spec.qvel_slice == (2, 4)
    assert spec.feature_names == (
        "shoulder.position",
        "elbow.position",
        "shoulder.velocity",
        "elbow.velocity",
    )


def test_position_action_encoding_keeps_existing_qpos_layout() -> None:
    spec = build_expert_action_spec(
        joint_names=["joint_1", "joint_3"],
        joint_command_mode="position",
    )

    encoded = encode_expert_action(
        torch.tensor([[10.0, 20.0]]),
        spec=spec,
        active_joint_ids=[1, 3],
    )

    torch.testing.assert_close(encoded, torch.tensor([[10.0, 20.0]]))


def test_position_velocity_action_encoding_selects_active_full_robot_targets() -> None:
    spec = build_expert_action_spec(
        joint_names=["joint_1", "joint_3"],
        joint_command_mode="position_velocity",
    )
    action = TensorDict(
        {
            "qpos": torch.tensor([[5.0, 10.0, 7.0, 20.0]]),
            "qvel": torch.tensor([[0.0, 1.0, 0.0, 2.0]]),
        },
        batch_size=[1],
    )

    encoded = encode_expert_action(
        action,
        spec=spec,
        active_joint_ids=[1, 3],
    )

    torch.testing.assert_close(encoded, torch.tensor([[10.0, 20.0, 1.0, 2.0]]))


def test_position_velocity_action_encoding_rejects_missing_qvel() -> None:
    spec = build_expert_action_spec(
        joint_names=["joint_0"],
        joint_command_mode="position_velocity",
    )

    with pytest.raises(ValueError, match="qvel"):
        encode_expert_action(
            TensorDict({"qpos": torch.zeros(1, 1)}, batch_size=[1]),
            spec=spec,
            active_joint_ids=[0],
        )


def test_expert_action_encoding_rejects_nonfinite_targets() -> None:
    spec = build_expert_action_spec(
        joint_names=["joint_0"],
        joint_command_mode="position_velocity",
    )
    action = TensorDict(
        {
            "qpos": torch.zeros(1, 1),
            "qvel": torch.tensor([[float("nan")]]),
        },
        batch_size=[1],
    )

    with pytest.raises(ValueError, match="finite"):
        encode_expert_action(action, spec=spec, active_joint_ids=[0])
