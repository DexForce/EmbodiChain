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

import numpy as np
import torch

from embodichain.lab.gym.envs.managers import ActionManager, ActionTermCfg
from embodichain.lab.gym.envs.managers.actions import (
    EefPoseAction,
    JointEffortAction,
    JointPositionAction,
    JointPositionToLimitsAction,
    JointVelocityAction,
    ParallelGripperAction,
    RelativeJointPositionAction,
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


def test_arm_plus_gripper_dimensions_follow_resolved_arm_width() -> None:
    """UR and Franka layouts derive 6+1 and 7+1 widths without literals."""
    for arm_width, expected in ((6, 7), (7, 8)):
        joint_names = tuple(f"arm_{index}" for index in range(arm_width)) + ("finger",)
        env = make_action_env(
            num_envs=1,
            joint_names=joint_names,
            parts={"arm": tuple(range(arm_width)), "hand": (arm_width,)},
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
                        "part_name": "hand",
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
