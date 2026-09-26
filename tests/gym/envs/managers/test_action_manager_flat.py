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

"""Tests for flat ActionManager orchestration."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from action_test_utils import make_manager


def test_manager_slices_flat_action_in_config_order() -> None:
    """Each term receives its ordered slice and descriptor binding."""
    manager = make_manager(
        (
            "arm",
            {
                "action_dim": 2,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0, 1),
            },
        ),
        (
            "gripper",
            {
                "action_dim": 1,
                "low": 0.0,
                "high": 1.0,
                "joint_ids": (2,),
            },
        ),
        num_envs=2,
    )
    first = manager.get_term("arm")
    second = manager.get_term("gripper")
    action = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    result = manager.process_action(action)

    assert result is None
    torch.testing.assert_close(first.raw_actions, action[:, :2])
    torch.testing.assert_close(second.raw_actions, action[:, 2:3])
    assert [(item.name, item.start, item.stop) for item in manager.descriptors] == [
        ("arm", 0, 2),
        ("gripper", 2, 3),
    ]


def test_manager_tracks_previous_flat_action() -> None:
    """The manager owns current and previous complete policy actions."""
    manager = make_manager(
        (
            "arm",
            {
                "action_dim": 2,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0, 1),
            },
        ),
        num_envs=1,
    )
    manager.process_action(torch.tensor([[0.1, 0.2]]))

    manager.process_action(torch.tensor([[0.3, 0.4]]))

    torch.testing.assert_close(manager.action, torch.tensor([[0.3, 0.4]]))
    torch.testing.assert_close(manager.previous_action, torch.tensor([[0.1, 0.2]]))


def test_manager_concatenates_term_spaces() -> None:
    """The public policy space follows descriptor order and bounds."""
    manager = make_manager(
        (
            "arm",
            {
                "action_dim": 2,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0, 1),
            },
        ),
        (
            "gripper",
            {
                "action_dim": 1,
                "low": 0.0,
                "high": 1.0,
                "joint_ids": (2,),
            },
        ),
    )

    space = manager.single_action_space

    assert isinstance(space, gym.spaces.Box)
    np.testing.assert_array_equal(space.low, np.array([-1.0, -1.0, 0.0]))
    np.testing.assert_array_equal(space.high, np.ones(3))


def test_manager_applies_every_term_once() -> None:
    """Application is independent from processing and preserves term order."""
    manager = make_manager(
        (
            "first",
            {
                "action_dim": 1,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0,),
            },
        ),
        (
            "second",
            {
                "action_dim": 1,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (1,),
            },
        ),
    )

    manager.apply_action()

    assert manager.get_term("first").apply_count == 1
    assert manager.get_term("second").apply_count == 1


@pytest.mark.parametrize("shape", [(2,), (2, 1), (2, 4)])
def test_manager_rejects_wrong_flat_shape(shape: tuple[int, ...]) -> None:
    """Invalid shapes fail before any raw state changes."""
    manager = make_manager(
        (
            "arm",
            {
                "action_dim": 3,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0, 1, 2),
            },
        ),
        num_envs=2,
    )

    with pytest.raises(ValueError, match=r"Expected action shape \(2, 3\)"):
        manager.process_action(torch.zeros(shape))

    torch.testing.assert_close(manager.action, torch.zeros(2, 3))


def test_manager_rejects_dict_action() -> None:
    """Policy callers cannot select the removed Dict layout."""
    manager = make_manager(
        (
            "arm",
            {
                "action_dim": 2,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0, 1),
            },
        ),
        num_envs=1,
    )

    with pytest.raises(TypeError, match="flat torch.Tensor"):
        manager.process_action({"arm": torch.zeros(1, 2)})


def test_manager_rejects_overlapping_qpos_terms() -> None:
    """Two qpos terms cannot own the same joint."""
    with pytest.raises(ValueError, match="first.*second.*joint_1"):
        make_manager(
            (
                "first",
                {
                    "action_dim": 2,
                    "low": -1.0,
                    "high": 1.0,
                    "joint_ids": (0, 1),
                    "command_type": "qpos",
                },
            ),
            (
                "second",
                {
                    "action_dim": 1,
                    "low": -1.0,
                    "high": 1.0,
                    "joint_ids": (1,),
                    "command_type": "qpos",
                },
            ),
            num_envs=1,
        )


def test_manager_rejects_joint_descriptor_ownership_mismatch() -> None:
    """A public term cannot hide owned joint IDs by shortening its descriptor."""
    with pytest.raises(ValueError, match="controlled joint IDs.*descriptor"):
        make_manager(
            (
                "arm",
                {
                    "action_dim": 2,
                    "low": -1.0,
                    "high": 1.0,
                    "joint_ids": (0, 1),
                    "joint_names": ("joint_0",),
                    "command_type": "qpos",
                },
            ),
            num_envs=1,
        )


def test_manager_allows_different_command_types_on_one_joint() -> None:
    """Position and velocity terms retain the existing joint-command contract."""
    manager = make_manager(
        (
            "position",
            {
                "action_dim": 1,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0,),
                "command_type": "qpos",
            },
        ),
        (
            "velocity",
            {
                "action_dim": 1,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0,),
                "command_type": "qvel",
            },
        ),
        num_envs=1,
    )

    assert manager.total_action_dim == 2


def test_manager_reset_preserves_unselected_rows() -> None:
    """Manager and term histories reset only requested environments."""
    manager = make_manager(
        (
            "arm",
            {
                "action_dim": 2,
                "low": -1.0,
                "high": 1.0,
                "joint_ids": (0, 1),
            },
        ),
        num_envs=2,
    )
    action = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    manager.process_action(action)

    manager.reset(torch.tensor([0]))

    torch.testing.assert_close(manager.action[0], torch.zeros(2))
    torch.testing.assert_close(manager.action[1], action[1])
    torch.testing.assert_close(manager.get_term("arm").raw_actions[0], torch.zeros(2))
    torch.testing.assert_close(manager.get_term("arm").raw_actions[1], action[1])
