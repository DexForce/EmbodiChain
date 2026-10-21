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

"""Tests for built-in joint action terms."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from embodichain.lab.gym.envs.managers import ActionManager, ActionTermCfg
from embodichain.lab.gym.envs.managers.actions import (
    EefPoseAction,
    JointEffortAction,
    JointPositionAction,
    JointPositionToLimitsAction,
    JointVelocityAction,
    ParallelGripperAction,
    RelativeEefPoseAction,
    RelativeJointPositionAction,
    resolve_action_contract,
)

from action_test_utils import make_action_env, make_cfg


def test_joint_position_applies_selected_policy_order() -> None:
    """Non-contiguous part order is preserved at the controller boundary."""
    env = make_action_env(
        num_envs=1,
        joint_names=("joint_0", "joint_1", "joint_2", "joint_3"),
        parts={"arm": (3, 1)},
    )
    term = JointPositionAction(
        make_cfg(JointPositionAction, part_name="arm", preserve_order=True), env
    )

    term.process_actions(torch.tensor([[0.3, 0.1]]))
    term.apply_actions()

    kwargs = env.robot.set_qpos.call_args.kwargs
    assert kwargs["joint_ids"] == [3, 1]
    torch.testing.assert_close(kwargs["qpos"], torch.tensor([[0.3, 0.1]]))
    assert term.controlled_joint_ids == (3, 1)
    assert term.descriptor.joint_names == ("joint_3", "joint_1")


def test_joint_position_action_space_uses_selected_qpos_limits() -> None:
    """Absolute joint actions expose the selected joints' physical limits."""
    env = make_action_env(
        num_envs=1,
        joint_names=("joint_0", "joint_1"),
        parts={"arm": (1, 0)},
    )
    env.robot.body_data.qpos_limits = torch.tensor([[[-0.5, 0.7], [-1.2, 1.4]]])
    term = JointPositionAction(
        make_cfg(JointPositionAction, part_name="arm", preserve_order=True), env
    )

    np.testing.assert_allclose(term.action_space.low, [-1.2, -0.5])
    np.testing.assert_allclose(term.action_space.high, [1.4, 0.7])


def test_relative_joint_position_action_space_is_unbounded_without_clip() -> None:
    """Relative actions remain unbounded unless an explicit clip is configured."""
    env = make_action_env(num_envs=1, joint_names=("joint_0", "joint_1"))
    term = RelativeJointPositionAction(
        make_cfg(RelativeJointPositionAction, part_name="arm"), env
    )

    assert np.isneginf(term.action_space.low).all()
    assert np.isposinf(term.action_space.high).all()


def test_joint_position_to_limits_maps_each_dimension() -> None:
    """Normalized actions use each selected joint's own limits."""
    env = make_action_env(joint_names=("joint_0", "joint_1"))
    env.robot.body_data.qpos_limits = torch.tensor([[[-2.0, 2.0], [0.0, 4.0]]])
    term = JointPositionToLimitsAction(
        make_cfg(JointPositionToLimitsAction, part_name="arm"), env
    )

    term.process_actions(torch.tensor([[-1.0, 1.0], [0.0, 0.0]]))

    torch.testing.assert_close(
        term.processed_actions,
        torch.tensor([[-2.0, 4.0], [0.0, 2.0]]),
    )
    np.testing.assert_array_equal(term.action_space.low, np.full(2, -1.0))
    np.testing.assert_array_equal(term.action_space.high, np.full(2, 1.0))


def test_relative_joint_position_uses_selected_current_qpos() -> None:
    """Relative commands add scaled actions to current selected joints."""
    env = make_action_env(
        num_envs=1,
        joint_names=("joint_0", "joint_1", "joint_2"),
        parts={"arm": (2, 0)},
    )
    env.robot.get_qpos.return_value = torch.tensor([[0.1, 5.0, 0.3]])
    term = RelativeJointPositionAction(
        make_cfg(
            RelativeJointPositionAction,
            part_name="arm",
            preserve_order=True,
            scale=0.5,
            clip=1.0,
        ),
        env,
    )

    term.process_actions(torch.tensor([[2.0, -2.0]]))

    torch.testing.assert_close(term.raw_actions, torch.tensor([[1.0, -1.0]]))
    torch.testing.assert_close(term.processed_actions, torch.tensor([[0.8, -0.4]]))


def test_joint_velocity_applies_selected_limits_and_ids() -> None:
    """Velocity actions use selected velocity bounds and setter."""
    env = make_action_env(num_envs=1, joint_names=("joint_0", "joint_1"))
    env.robot.body_data.qvel_limits = torch.tensor([[2.0, 3.0]])
    term = JointVelocityAction(
        make_cfg(JointVelocityAction, part_name="arm", scale=2.0), env
    )

    term.process_actions(torch.tensor([[0.5, -0.5]]))
    term.apply_actions()

    torch.testing.assert_close(term.processed_actions, torch.tensor([[1.0, -1.0]]))
    np.testing.assert_array_equal(term.action_space.low, np.array([-2.0, -3.0]))
    assert env.robot.set_qvel.call_args.kwargs["joint_ids"] == [0, 1]


def test_joint_effort_applies_selected_limits_and_ids() -> None:
    """Effort actions use selected effort bounds and setter."""
    env = make_action_env(num_envs=1, joint_names=("joint_0", "joint_1"))
    env.robot.body_data.qf_limits = torch.tensor([[4.0, 5.0]])
    term = JointEffortAction(
        make_cfg(JointEffortAction, part_name="arm", scale=2.0), env
    )

    term.process_actions(torch.tensor([[0.5, -0.5]]))
    term.apply_actions()

    torch.testing.assert_close(term.processed_actions, torch.tensor([[1.0, -1.0]]))
    np.testing.assert_array_equal(term.action_space.high, np.array([4.0, 5.0]))
    assert env.robot.set_qf.call_args.kwargs["joint_ids"] == [0, 1]


def test_joint_action_reset_preserves_unselected_rows() -> None:
    """Term reset clears raw and processed state only for selected rows."""
    env = make_action_env(num_envs=2, joint_names=("joint_0", "joint_1"))
    term = JointPositionAction(make_cfg(JointPositionAction, part_name="arm"), env)
    action = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    term.process_actions(action)

    term.reset(torch.tensor([0]))

    torch.testing.assert_close(term.raw_actions[0], torch.zeros(2))
    torch.testing.assert_close(term.raw_actions[1], action[1])
    torch.testing.assert_close(term.processed_actions[0], torch.zeros(2))
    torch.testing.assert_close(term.processed_actions[1], action[1])


def test_manager_masks_inactive_position_rows_with_measured_joint_hold() -> None:
    """Completed vector-demo rows hold measured qpos before term application."""
    env = make_action_env(num_envs=2, joint_names=("joint_0", "joint_1"))
    env.robot.get_qpos.return_value = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    manager = ActionManager(
        {
            "arm_action": ActionTermCfg(
                func=JointPositionAction,
                params={"part_name": "arm"},
            )
        },
        env,
    )
    manager.process_action(torch.tensor([[9.0, 9.0], [8.0, 8.0]]))

    manager.mask_inactive(torch.tensor([False, True]))
    manager.apply_action()

    torch.testing.assert_close(
        env.robot.set_qpos.call_args.kwargs["qpos"],
        torch.tensor([[0.1, 0.2], [8.0, 8.0]]),
    )


def test_eef_action_applies_ik_only_to_selected_arm() -> None:
    """EEF IK applies only the selected arm joints in configured order."""
    env = make_action_env(
        num_envs=1,
        joint_names=("joint_0", "joint_1", "joint_2"),
        parts={"arm": (0, 2)},
    )
    env.robot.get_qpos.return_value = torch.tensor([[0.1, 9.0, 0.2]])
    env.robot.compute_ik.return_value = (
        torch.tensor([True]),
        torch.tensor([[0.4, 0.5]]),
    )
    term = EefPoseAction(
        make_cfg(EefPoseAction, part_name="arm", pose_representation="xyz_rpy"),
        env,
    )

    term.process_actions(torch.zeros(1, 6))
    term.apply_actions()

    kwargs = env.robot.set_qpos.call_args.kwargs
    assert kwargs["joint_ids"] == [0, 2]
    torch.testing.assert_close(kwargs["qpos"], torch.tensor([[0.4, 0.5]]))
    assert term.action_dim == 6


def test_eef_action_holds_failed_ik_row() -> None:
    """Failed IK rows retain the current selected arm configuration."""
    env = make_action_env(
        num_envs=2,
        joint_names=("joint_0", "joint_1"),
        parts={"arm": (0, 1)},
    )
    env.robot.get_qpos.return_value = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    env.robot.compute_ik.return_value = (
        torch.tensor([True, False]),
        torch.tensor([[0.5, 0.6], [0.8, 0.9]]),
    )
    term = EefPoseAction(
        make_cfg(EefPoseAction, part_name="arm", pose_representation="xyz_rpy"),
        env,
    )

    term.process_actions(torch.zeros(2, 6))

    torch.testing.assert_close(term.processed_actions[0], torch.tensor([0.5, 0.6]))
    torch.testing.assert_close(term.processed_actions[1], torch.tensor([0.3, 0.4]))
    assert term.ik_success.tolist() == [True, False]


def test_continuous_parallel_gripper_maps_opposing_commands() -> None:
    """One scalar interpolates explicit per-joint endpoint commands."""
    env = make_action_env(
        num_envs=2,
        joint_names=("j0", "j1", "j2", "j3", "j4", "left", "j6", "right"),
        parts={"hand": (5, 7)},
    )
    term = ParallelGripperAction(
        make_cfg(
            ParallelGripperAction,
            part_name="hand",
            command_mode="continuous",
            lower_command={"left": 0.0, "right": 0.04},
            upper_command={"left": 0.04, "right": 0.0},
        ),
        env,
    )

    term.process_actions(torch.tensor([[-1.0], [1.0]]))

    torch.testing.assert_close(
        term.processed_actions,
        torch.tensor([[0.0, 0.04], [0.04, 0.0]]),
    )


def test_binary_parallel_gripper_has_one_policy_dimension() -> None:
    """Binary grippers select complete close/open joint configurations."""
    env = make_action_env(
        num_envs=2,
        joint_names=("left", "right"),
        parts={"hand": (0, 1)},
    )
    term = ParallelGripperAction(
        make_cfg(
            ParallelGripperAction,
            part_name="hand",
            command_mode="binary",
            open_command={"left": 0.04, "right": 0.04},
            close_command={"left": 0.0, "right": 0.0},
        ),
        env,
    )

    term.process_actions(torch.tensor([[-0.1], [0.1]]))

    assert term.action_dim == 1
    torch.testing.assert_close(term.processed_actions[0], term.close_command)
    torch.testing.assert_close(term.processed_actions[1], term.open_command)


@pytest.mark.parametrize("part_name", ["eef", "hand", "custom_claw"])
def test_arm_plus_gripper_dimensions_follow_resolved_arm_width(part_name: str) -> None:
    """Default and custom part names derive action widths from resolved joints."""
    for arm_width, expected in ((6, 7), (7, 8)):
        joint_names = tuple(f"arm_{index}" for index in range(arm_width)) + ("finger",)
        env = make_action_env(
            num_envs=1,
            joint_names=joint_names,
            parts={"arm": tuple(range(arm_width)), part_name: (arm_width,)},
        )
        manager = ActionManager(
            {
                "arm_action": ActionTermCfg(
                    func=JointPositionAction,
                    params={"part_name": "arm"},
                ),
                "gripper_action": ActionTermCfg(
                    func=ParallelGripperAction,
                    params={
                        "part_name": part_name,
                        "command_mode": "continuous",
                        "use_joint_limits": True,
                    },
                ),
            },
            env,
        )
        assert manager.total_action_dim == expected
        assert manager.single_action_space.shape == (expected,)


def test_parallel_gripper_never_controls_mimic_follower() -> None:
    """Resolved hand IDs contain only the independent gripper leader."""
    env = make_action_env(
        num_envs=1,
        joint_names=("arm", "finger_leader", "finger_follower"),
        parts={"hand": (1,)},
    )
    term = ParallelGripperAction(
        make_cfg(
            ParallelGripperAction,
            part_name="hand",
            command_mode="continuous",
            use_joint_limits=True,
        ),
        env,
    )

    term.process_actions(torch.zeros(1, 1))
    term.apply_actions()

    assert term.controlled_joint_ids == (1,)
    assert env.robot.set_qpos.call_args.kwargs["joint_ids"] == [1]


def test_action_manager_trace_separates_requested_and_processed_commands() -> None:
    """Action traces preserve policy requests beside term commands."""
    env = make_action_env(
        num_envs=1,
        joint_names=("joint_0", "joint_1"),
        parts={"arm": (0, 1)},
    )
    manager = ActionManager(
        {
            "arm_action": ActionTermCfg(
                func=JointPositionAction,
                params={"part_name": "arm", "scale": 2.0},
            )
        },
        env,
    )
    requested = torch.tensor([[0.1, 0.2]])
    manager.process_action(requested)

    trace = manager.action_trace()

    torch.testing.assert_close(trace.requested, requested)
    torch.testing.assert_close(trace.processed["arm_action"], requested * 2.0)
    torch.testing.assert_close(trace.executed["arm_action"], requested * 2.0)
    assert trace.command_types == ("qpos",)


@pytest.fixture
def relative_eef_env() -> SimpleNamespace:
    """Supply measured arm joints, arena FK, and selected-arm IK without simulation."""
    env = make_action_env(
        num_envs=2,
        joint_names=("joint_0", "finger", "joint_2"),
        parts={"arm": (2, 0), "eef": (1,)},
    )
    env.robot.get_qpos.return_value = torch.tensor([[0.1, 5.0, 0.3], [0.4, 6.0, 0.6]])
    measured = torch.eye(4).repeat(2, 1, 1)
    measured[:, :3, 3] = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    env.robot.compute_fk = MagicMock(return_value=measured)
    env.robot.compute_ik.return_value = (
        torch.tensor([True, True]),
        torch.tensor([[0.8, 0.9], [1.0, 1.1]]),
    )
    names = ("joint_2", "joint_0")
    env.solver = SimpleNamespace(
        joint_names=names,
        pk_serial_chain=SimpleNamespace(get_joint_parameter_names=lambda: names),
    )
    env.robot.get_solver = MagicMock(return_value=env.solver)
    return env


def test_relative_eef_pose_uses_measured_pose_each_step(
    relative_eef_env: SimpleNamespace,
) -> None:
    """Repeated requests reference the new measured state rather than an old target."""
    env = relative_eef_env
    term = RelativeEefPoseAction(make_cfg(RelativeEefPoseAction, part_name="arm"), env)
    request = torch.tensor([[0.5, -1.0, 0.25, 0.0, 0.0, 0.0]]).repeat(2, 1)

    term.process_actions(request)
    torch.testing.assert_close(
        term.target_pose[:, :3, 3],
        torch.tensor([[0.21, 0.28, 0.405], [0.51, 0.58, 0.705]]),
    )
    torch.testing.assert_close(
        env.robot.compute_ik.call_args.kwargs["joint_seed"],
        torch.tensor([[0.3, 0.1], [0.6, 0.4]]),
    )
    env.robot.compute_fk.return_value[:, :3, 3] += 0.1
    env.robot.get_qpos.return_value += 0.2

    term.process_actions(request)
    term.apply_actions()

    torch.testing.assert_close(
        term.target_pose[:, :3, 3],
        torch.tensor([[0.31, 0.38, 0.505], [0.61, 0.68, 0.805]]),
    )
    torch.testing.assert_close(
        env.robot.compute_fk.call_args.kwargs["qpos"],
        torch.tensor([[0.5, 0.3], [0.8, 0.6]]),
    )
    assert env.robot.compute_fk.call_args.kwargs["name"] == "arm"
    assert env.robot.set_qpos.call_args.kwargs["joint_ids"] == [2, 0]
    torch.testing.assert_close(term.previous_raw_actions, request)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_relative_eef_pose_rotation_left_composes_in_arena(
    relative_eef_env: SimpleNamespace,
    dtype: torch.dtype,
) -> None:
    """An arena X rotation acts on the left of a measured quarter-turn about Z."""
    env = relative_eef_env
    env.robot.compute_fk.return_value[:, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    term = RelativeEefPoseAction(make_cfg(RelativeEefPoseAction, part_name="arm"), env)
    request = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=dtype).repeat(2, 1)

    term.process_actions(request)

    cosine, sine = np.cos(0.10), np.sin(0.10)
    expected = torch.tensor(
        [[0.0, -1.0, 0.0], [cosine, 0.0, -sine], [sine, 0.0, cosine]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(term.target_pose[:, :3, :3], expected.repeat(2, 1, 1))
    torch.testing.assert_close(term.current_pose, env.robot.compute_fk.return_value)
    torch.testing.assert_close(
        env.robot.compute_ik.call_args.kwargs["pose"], term.target_pose
    )


@pytest.mark.parametrize("nonfinite_solution", [False, True])
def test_relative_eef_pose_holds_failed_ik(
    relative_eef_env: SimpleNamespace, nonfinite_solution: bool
) -> None:
    """Failed IK and invalid successful solutions both hold measured selected joints."""
    env = relative_eef_env
    env.robot.compute_ik.return_value = (
        torch.tensor([True, nonfinite_solution]),
        torch.tensor([[0.8, 0.9], [float("nan") if nonfinite_solution else 8.0, 9.0]]),
    )
    term = RelativeEefPoseAction(make_cfg(RelativeEefPoseAction, part_name="arm"), env)

    term.process_actions(torch.zeros(2, 6))
    term.apply_actions()

    torch.testing.assert_close(
        term.processed_actions, torch.tensor([[0.8, 0.9], [0.6, 0.4]])
    )
    torch.testing.assert_close(
        env.robot.set_qpos.call_args.kwargs["qpos"], term.processed_actions
    )
    assert term.ik_success.tolist() == [True, False]


def test_relative_eef_pose_clamps_requests_and_exposes_semantics(
    relative_eef_env: SimpleNamespace,
) -> None:
    """Descriptor metadata and clipping describe the actual normalized controller."""
    term = RelativeEefPoseAction(
        make_cfg(
            RelativeEefPoseAction,
            part_name="arm",
            translation_scale=0.04,
            rotation_scale=0.2,
        ),
        relative_eef_env,
    )
    request = torch.tensor([[2.0, -2.0, 0.5, 0.0, 0.0, 3.0]]).repeat(2, 1)

    term.process_actions(request)

    np.testing.assert_array_equal(term.action_space.low, np.full(6, -1.0))
    np.testing.assert_array_equal(term.action_space.high, np.ones(6))
    torch.testing.assert_close(term.raw_actions, request)
    torch.testing.assert_close(
        term.target_pose[:, :3, 3] - term.current_pose[:, :3, 3],
        torch.tensor([[0.04, -0.04, 0.02]]).repeat(2, 1),
    )
    assert term.clipped_mask.tolist() == [[True, True, False, False, False, True]] * 2
    cosine, sine = np.cos(0.2), np.sin(0.2)
    expected_rotation = torch.tensor(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        term.target_pose[:, :3, :3], expected_rotation.repeat(2, 1, 1)
    )
    descriptor = term.descriptor.to_dict()
    assert descriptor["contract"] == "eef_pose.relative@1"
    assert descriptor["units"] == ["normalized"] * 6
    assert descriptor["normalization"] == "minus_one_to_one"
    assert descriptor["metadata"] == {
        "frame": "arena",
        "rotation": "rotation_vector",
        "rotation_composition": "left",
        "target_reference": "measured_pose",
        "translation_scale": 0.04,
        "rotation_scale": 0.2,
        "translation_units": "m",
        "rotation_units": "rad",
        "clip": [-1.0, 1.0],
        "ik_failure": "measured_joint_hold",
    }
    assert resolve_action_contract("eef_pose.relative@1") is RelativeEefPoseAction


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_relative_eef_pose_rejects_nonfinite_before_state_changes(
    relative_eef_env: SimpleNamespace, value: float
) -> None:
    """Nonfinite requests cannot replace history, diagnostics, or pending commands."""
    term = RelativeEefPoseAction(
        make_cfg(RelativeEefPoseAction, part_name="arm"), relative_eef_env
    )
    term.process_actions(torch.ones(2, 6))
    buffer_names = (
        "raw_actions",
        "previous_raw_actions",
        "processed_actions",
        "current_pose",
        "target_pose",
        "clipped_mask",
        "ik_success",
    )
    snapshots = {name: getattr(term, name).clone() for name in buffer_names}
    invalid = torch.zeros(2, 6)
    invalid[1, 3] = value

    with pytest.raises(ValueError, match="finite"):
        term.process_actions(invalid)

    for name, snapshot in snapshots.items():
        torch.testing.assert_close(getattr(term, name), snapshot)
    assert relative_eef_env.robot.compute_ik.call_count == 1


@pytest.mark.parametrize("env_ids", [[0], torch.tensor([0]), None])
def test_relative_eef_pose_reset_preserves_other_rows(
    relative_eef_env: SimpleNamespace, env_ids: list[int] | torch.Tensor | None
) -> None:
    """Partial resets clear all selected history and diagnostics and retain others."""
    term = RelativeEefPoseAction(
        make_cfg(RelativeEefPoseAction, part_name="arm"), relative_eef_env
    )
    term.process_actions(torch.ones(2, 6))
    term.process_actions(torch.full((2, 6), 2.0))
    buffer_names = (
        "raw_actions",
        "previous_raw_actions",
        "processed_actions",
        "current_pose",
        "target_pose",
        "clipped_mask",
        "ik_success",
    )
    previous_rows = {name: getattr(term, name)[1].clone() for name in buffer_names}

    term.reset(env_ids)

    for row in ([0, 1] if env_ids is None else [0]):
        for name in buffer_names:
            expected = (
                torch.eye(4)
                if name in {"current_pose", "target_pose"}
                else torch.zeros_like(getattr(term, name)[row])
            )
            torch.testing.assert_close(getattr(term, name)[row], expected)
    if env_ids is not None:
        for name, previous in previous_rows.items():
            torch.testing.assert_close(getattr(term, name)[1], previous)


@pytest.mark.parametrize(
    "name", ["translation_scale", "rotation_scale", "dls_damping", "max_joint_delta"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_relative_eef_pose_rejects_invalid_scales(
    relative_eef_env: SimpleNamespace, name: str, value: float
) -> None:
    """Controller scales must describe finite nonzero motion bounds."""
    with pytest.raises(ValueError, match=name):
        RelativeEefPoseAction(
            make_cfg(RelativeEefPoseAction, part_name="arm", **{name: value}),
            relative_eef_env,
        )


def test_relative_eef_and_gripper_compose_seven_policy_dimensions(
    relative_eef_env: SimpleNamespace,
) -> None:
    """The relative pose contract combines with the existing one-scalar gripper."""
    manager = ActionManager(
        {
            "arm_action": ActionTermCfg(
                func=resolve_action_contract("eef_pose.relative@1"),
                contract="eef_pose.relative@1",
                params={"part_name": "arm"},
            ),
            "gripper_action": make_cfg(
                ParallelGripperAction,
                part_name="eef",
                command_mode="continuous",
                lower_command={"finger": 0.0},
                upper_command={"finger": 0.04},
            ),
        },
        relative_eef_env,
    )

    manager.process_action(torch.zeros(2, 7))
    manager.apply_action()

    assert manager.total_action_dim == 7
    assert [(item.start, item.stop) for item in manager.descriptors] == [(0, 6), (6, 7)]
    torch.testing.assert_close(
        manager.get_term("gripper_action").processed_actions, torch.full((2, 1), 0.02)
    )
    assert [
        call.kwargs["joint_ids"]
        for call in relative_eef_env.robot.set_qpos.call_args_list
    ] == [[2, 0], [1]]


def test_absolute_eef_quaternion_remains_absolute(
    relative_eef_env: SimpleNamespace,
) -> None:
    """Absolute quaternion requests keep their seven-component arena pose semantics."""
    term = EefPoseAction(
        make_cfg(EefPoseAction, part_name="arm", pose_representation="xyz_quat_xyzw"),
        relative_eef_env,
    )
    absolute_pose = torch.tensor([[0.9, 0.8, 0.7, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)

    term.process_actions(absolute_pose)

    target = relative_eef_env.robot.compute_ik.call_args.kwargs["pose"]
    torch.testing.assert_close(target[:, :3, 3], absolute_pose[:, :3])
    torch.testing.assert_close(target[:, :3, :3], torch.eye(3).repeat(2, 1, 1))
    relative_eef_env.robot.compute_fk.assert_not_called()


@pytest.fixture
def dls_eef_env(relative_eef_env: SimpleNamespace) -> SimpleNamespace:
    """Supply a TCP Jacobian with independent linear and angular X columns."""
    env = relative_eef_env
    names = ("joint_2", "joint_0")
    jacobian = torch.zeros(2, 6, 2)
    jacobian[:, 0, 0] = 1
    jacobian[:, 3, 1] = 1
    env.solver = SimpleNamespace(
        joint_names=names,
        root_link_name="solver_root",
        pk_serial_chain=SimpleNamespace(get_joint_parameter_names=lambda: names),
        tcp_xpos=np.eye(4),
        get_jacobian=MagicMock(return_value=jacobian),
    )
    env.robot.get_solver = MagicMock(return_value=env.solver)
    env.robot.get_link_pose = MagicMock(return_value=torch.eye(4).repeat(2, 1, 1))
    env.robot.get_control_part_base_pose = MagicMock(
        return_value=torch.eye(4).repeat(2, 1, 1)
    )
    return env


def test_dls_rotates_linear_and_angular_jacobian_into_arena(
    dls_eef_env: SimpleNamespace,
) -> None:
    """A root rotated by pi about Z reverses both arena-X Jacobian blocks."""
    env = dls_eef_env
    env.robot.get_link_pose.return_value[:, :3, :3] = torch.diag(
        torch.tensor([-1.0, -1.0, 1.0])
    )
    term = RelativeEefPoseAction(
        make_cfg(RelativeEefPoseAction, part_name="arm", ik_method="dls"), env
    )
    term.process_actions(torch.tensor([[1.0, 0, 0, 1.0, 0, 0]]).repeat(2, 1))

    expected = torch.tensor([[-0.02, -0.10]]).repeat(2, 1) / (1 + 0.05**2)
    torch.testing.assert_close(term.joint_delta, expected)
    env.robot.compute_ik.assert_not_called()
    torch.testing.assert_close(
        env.solver.get_jacobian.call_args.args[0],
        torch.tensor([[0.3, 0.1], [0.6, 0.4]]),
    )
    assert env.solver.get_jacobian.call_args.kwargs == {}
    env.robot.get_link_pose.assert_called_once_with(
        link_name="solver_root", to_matrix=True
    )
    env.robot.get_control_part_base_pose.assert_not_called()
    assert term.ik_success.all()


def test_dls_shifts_nonzero_tcp_jacobian_without_pk_locations(
    dls_eef_env: SimpleNamespace,
) -> None:
    """TCP rotation is removed before shifting the arena Jacobian to its offset."""
    env = dls_eef_env
    env.solver.get_jacobian.return_value.zero_()
    env.solver.get_jacobian.return_value[:, 5, 0] = 1.0
    env.solver.tcp_xpos[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    env.solver.tcp_xpos[:3, 3] = [0.2, 0.0, 0.0]
    env.robot.compute_fk.return_value[:, :3, :3] = torch.diag(
        torch.tensor([-1.0, -1.0, 1.0])
    )
    term = RelativeEefPoseAction(
        make_cfg(RelativeEefPoseAction, part_name="arm", ik_method="dls"), env
    )

    term.process_actions(torch.tensor([[1.0, 0, 0, 0, 0, 0]]).repeat(2, 1))

    # End-link Rz(pi/2) puts its +X offset along arena +Y, so angular +Z
    # induces linear -X. The scalar DLS denominator includes both Jacobian rows.
    expected = torch.tensor([[-0.2 * 0.02 / (1 + 0.2**2 + 0.05**2), 0.0]])
    torch.testing.assert_close(term.joint_delta, expected.repeat(2, 1))
    assert env.solver.get_jacobian.call_args.kwargs == {}
    assert term.ik_success.all()


def test_dls_rescales_joint_step_and_clips_joint_limits(
    dls_eef_env: SimpleNamespace,
) -> None:
    """The entire increment is rescaled before individual position limits apply."""
    env = dls_eef_env
    env.solver.get_jacobian.return_value[:, 0, 0] = 0.01
    env.robot.get_qpos.return_value[0, 2] = 0.98
    term = RelativeEefPoseAction(
        make_cfg(
            RelativeEefPoseAction, part_name="arm", ik_method="dls", dls_damping=0.001
        ),
        env,
    )
    term.process_actions(torch.tensor([[1.0, 0, 0, 1.0, 0, 0]]).repeat(2, 1))

    assert term.joint_step_limited.tolist() == [True, True]
    assert term.joint_limit_clipped_mask.tolist() == [[True, False], [False, False]]
    assert term.joint_delta.abs().max() <= 0.100001
    torch.testing.assert_close(term.processed_actions[0, 0], torch.tensor(1.0))
    torch.testing.assert_close(term.joint_delta[1, 0], torch.tensor(0.1))
    expected_other_delta = (
        0.1 / (1 + 0.001**2) * 0.1 / (0.02 * 0.01 / (0.01**2 + 0.001**2))
    )
    torch.testing.assert_close(
        term.joint_delta[:, 1], torch.full((2,), expected_other_delta)
    )


def test_dls_damping_handles_singular_jacobian_as_a_finite_local_step(
    dls_eef_env: SimpleNamespace,
) -> None:
    """A rank-zero Jacobian yields a finite zero step without claiming pose convergence."""
    env = dls_eef_env
    env.solver.get_jacobian.return_value.zero_()
    term = RelativeEefPoseAction(
        make_cfg(RelativeEefPoseAction, part_name="arm", ik_method="dls"), env
    )
    term.process_actions(torch.ones(2, 6))

    torch.testing.assert_close(term.joint_delta, torch.zeros(2, 2))
    assert term.ik_success.tolist() == [True, True]
    metadata = term.descriptor.metadata
    assert metadata["ik_method"] == "dls"
    assert metadata["dls_damping"] == 0.05
    assert metadata["max_joint_delta"] == 0.1
    assert metadata["controller_step"] == "bounded_local_jacobian_approximation"
    assert (
        metadata["ik_success_semantics"]
        == "finite_controller_step_not_pose_convergence"
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_dls_nonfinite_jacobian_holds_row_and_partial_reset_preserves_others(
    dls_eef_env: SimpleNamespace,
    invalid: float,
) -> None:
    """Nonfinite rows hold measured joints, and all DLS diagnostics reset by row."""
    env = dls_eef_env
    env.solver.get_jacobian.return_value[0, 0, 0] = invalid
    term = RelativeEefPoseAction(
        make_cfg(
            RelativeEefPoseAction,
            part_name="arm",
            ik_method="dls",
            max_joint_delta=0.01,
        ),
        env,
    )
    term.process_actions(torch.ones(2, 6))
    term.process_actions(torch.full((2, 6), 2.0))
    assert term.ik_success.tolist() == [False, True]
    torch.testing.assert_close(term.processed_actions[0], torch.tensor([0.3, 0.1]))
    torch.testing.assert_close(term.joint_delta[0], torch.zeros(2))
    buffers = (
        "raw_actions",
        "previous_raw_actions",
        "processed_actions",
        "ik_success",
        "joint_delta",
        "joint_step_limited",
        "joint_limit_clipped_mask",
        "clipped_mask",
    )
    untouched = {name: getattr(term, name)[1].clone() for name in buffers}

    term.reset([0])

    for name in buffers:
        torch.testing.assert_close(
            getattr(term, name)[0], torch.zeros_like(getattr(term, name)[0])
        )
        torch.testing.assert_close(getattr(term, name)[1], untouched[name])


def test_dls_maps_joint_order_between_selected_joints_and_solver_chain(
    dls_eef_env: SimpleNamespace,
) -> None:
    """A permuted configured selection cannot change Jacobian column semantics."""
    env = dls_eef_env
    names = ("joint_0", "joint_2")
    env.solver.joint_names = names
    env.solver.pk_serial_chain.get_joint_parameter_names = lambda: names
    term = RelativeEefPoseAction(
        make_cfg(RelativeEefPoseAction, part_name="arm", ik_method="dls"), env
    )
    term.process_actions(torch.tensor([[1.0, 0, 0, 0, 0, 0]]).repeat(2, 1))
    term.apply_actions()

    expected = torch.tensor([[0.0, 0.02]]).repeat(2, 1) / (1 + 0.05**2)
    torch.testing.assert_close(term.joint_delta, expected)
    torch.testing.assert_close(
        env.robot.compute_fk.call_args.kwargs["qpos"],
        torch.tensor([[0.1, 0.3], [0.4, 0.6]]),
    )
    torch.testing.assert_close(
        env.solver.get_jacobian.call_args.args[0],
        env.robot.compute_fk.call_args.kwargs["qpos"],
    )
    assert env.robot.set_qpos.call_args.kwargs["joint_ids"] == [2, 0]


@pytest.mark.parametrize(
    "names", [None, ("joint_0", "unknown"), ("joint_0", "joint_0")]
)
@pytest.mark.parametrize("ik_method", ["solver", "dls"])
def test_dls_rejects_unknown_or_ambiguous_solver_joint_order(
    dls_eef_env: SimpleNamespace,
    names: tuple[str, ...] | None,
    ik_method: str,
) -> None:
    dls_eef_env.solver.joint_names = names
    with pytest.raises(ValueError, match="joint_names"):
        RelativeEefPoseAction(
            make_cfg(RelativeEefPoseAction, part_name="arm", ik_method=ik_method),
            dls_eef_env,
        )


@pytest.mark.parametrize("ik_method", ["solver", "dls"])
def test_dls_rejects_configured_names_that_disagree_with_chain_order(
    dls_eef_env: SimpleNamespace,
    ik_method: str,
) -> None:
    dls_eef_env.solver.pk_serial_chain.get_joint_parameter_names = lambda: (
        "joint_0",
        "joint_2",
    )
    with pytest.raises(ValueError, match="kinematic chain order"):
        RelativeEefPoseAction(
            make_cfg(RelativeEefPoseAction, part_name="arm", ik_method=ik_method),
            dls_eef_env,
        )


def test_relative_eef_solver_default_keeps_original_backend(
    dls_eef_env: SimpleNamespace,
) -> None:
    """Omitting ik_method retains global solver behavior and descriptor compatibility."""
    env = dls_eef_env
    term = RelativeEefPoseAction(make_cfg(RelativeEefPoseAction, part_name="arm"), env)
    term.process_actions(torch.ones(2, 6))
    env.robot.compute_ik.assert_called_once()
    env.robot.get_solver.assert_called_once_with("arm")
    env.solver.get_jacobian.assert_not_called()
    assert "ik_method" not in term.descriptor.metadata


def test_dls_reads_runtime_joint_limit_replacements(
    dls_eef_env: SimpleNamespace,
) -> None:
    """Limits replaced after term construction constrain the next controller step."""
    env = dls_eef_env
    term = RelativeEefPoseAction(
        make_cfg(RelativeEefPoseAction, part_name="arm", ik_method="dls"), env
    )
    limits = env.robot.body_data.qpos_limits.repeat(2, 1, 1)
    limits[0, 2, 1], limits[1, 2, 1] = 0.305, 0.605
    env.robot.body_data.qpos_limits = limits

    term.process_actions(torch.tensor([[1.0, 0, 0, 0, 0, 0]]).repeat(2, 1))

    torch.testing.assert_close(
        term.joint_delta, torch.tensor([[0.005, 0.0]]).repeat(2, 1)
    )
    assert term.joint_limit_clipped_mask.tolist() == [[True, False], [True, False]]
    assert term.ik_success.all()


@pytest.mark.parametrize("failed_row", [False, True])
def test_relative_eef_solver_maps_fk_seed_and_returned_joint_order(
    dls_eef_env: SimpleNamespace,
    failed_row: bool,
) -> None:
    """Exact IK preserves selected command order and measured holds after permutation."""
    env = dls_eef_env
    names = ("joint_0", "joint_2")
    env.solver.joint_names = names
    env.solver.pk_serial_chain.get_joint_parameter_names = lambda: names
    env.robot.compute_ik.return_value = (
        torch.tensor([True, not failed_row]),
        torch.tensor([[0.8, 0.9], [1.0, 1.1]]),
    )
    term = RelativeEefPoseAction(make_cfg(RelativeEefPoseAction, part_name="arm"), env)

    term.process_actions(torch.zeros(2, 6))
    term.apply_actions()

    expected_seeds = torch.tensor([[0.1, 0.3], [0.4, 0.6]])
    torch.testing.assert_close(
        env.robot.compute_fk.call_args.kwargs["qpos"], expected_seeds
    )
    torch.testing.assert_close(
        env.robot.compute_ik.call_args.kwargs["joint_seed"], expected_seeds
    )
    expected_commands = torch.tensor(
        [[0.9, 0.8], [0.6, 0.4] if failed_row else [1.1, 1.0]]
    )
    torch.testing.assert_close(term.processed_actions, expected_commands)
    assert env.robot.set_qpos.call_args.kwargs["joint_ids"] == [2, 0]
    assert term.descriptor.joint_names == ("joint_2", "joint_0")
