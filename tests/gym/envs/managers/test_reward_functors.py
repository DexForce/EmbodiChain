# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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

"""Tests for reward functors."""

from __future__ import annotations

import pytest
import torch

from unittest.mock import MagicMock, Mock


class MockRobot:
    """Mock robot for reward functor tests."""

    def __init__(self, num_envs: int = 4, num_joints: int = 6):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = torch.device("cpu")

        # Mock body_data
        self.body_data = Mock()
        self.body_data.qpos = torch.zeros(num_envs, num_joints)
        self.body_data.qvel = torch.zeros(num_envs, num_joints)
        self.body_data.qpos_limits = torch.zeros(1, num_joints, 2)
        self.body_data.qpos_limits[..., 0] = -3.14
        self.body_data.qpos_limits[..., 1] = 3.14
        self.body_data.qvel_limits = torch.zeros(1, num_joints, 2)
        self.body_data.qvel_limits[..., 0] = -10.0
        self.body_data.qvel_limits[..., 1] = 10.0

    def get_qpos(self, *args, **kwargs):
        return self.body_data.qpos

    def get_qvel(self, *args, **kwargs):
        return self.body_data.qvel

    def get_qpos_limits(self, *args, **kwargs):
        return self.body_data.qpos_limits

    def get_joint_ids(self, part_name=None):
        return list(range(self.num_joints))

    def compute_fk(self, qpos=None, name=None, to_matrix=True):
        pose = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1)
        pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.5])
        return pose


class MockRigidObject:
    """Mock rigid object for reward functor tests."""

    def __init__(self, uid: str = "test_object", num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        # Default pose at origin
        self._pose = torch.eye(4).unsqueeze(0).repeat(num_envs, 1, 1)

    def get_local_pose(self, to_matrix=True):
        return self._pose

    @property
    def body_data(self):
        return self


class MockSim:
    """Mock simulation for reward functor tests."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._rigid_objects = {}
        self._robots = {}

    def get_rigid_object(self, uid: str):
        return self._rigid_objects.get(uid)

    def get_rigid_object_uid_list(self):
        return list(self._rigid_objects.keys())

    def get_robot(self, uid: str = None):
        if uid is None:
            return list(self._robots.values())[0] if self._robots else None
        return self._robots.get(uid)

    def add_rigid_object(self, obj):
        self._rigid_objects[obj.uid] = obj


class MockEnv:
    """Mock environment for reward functor tests."""

    def __init__(self, num_envs: int = 4, num_joints: int = 6):
        self.num_envs = num_envs
        self.device = torch.device("cpu")

        self.sim = MockSim(num_envs)
        self.robot = MockRobot(num_envs, num_joints)
        self.sim._robots["robot"] = self.robot

        # Add test rigid objects
        self.test_object = MockRigidObject("cube", num_envs)
        self.sim.add_rigid_object(self.test_object)

        self.target_object = MockRigidObject("target", num_envs)
        self.target_object._pose[:, :3, 3] = torch.tensor([0.5, 0.0, 0.0])
        self.sim.add_rigid_object(self.target_object)

        # Rollout state for action_smoothness_penalty
        # rollout_buffer is 2D: (num_envs, rollout_steps)
        self.current_rollout_step = 0
        self.rollout_buffer = {
            "action": torch.zeros(
                num_envs, 100, 6
            ),  # (num_envs, rollout_steps, action_dim)
            "done": torch.zeros(
                num_envs, 100, dtype=torch.bool
            ),  # (num_envs, rollout_steps)
        }


# Import functors to test
from embodichain.lab.gym.envs.managers.rewards import (
    distance_between_objects,
    joint_velocity_penalty,
    action_smoothness_penalty,
    joint_limit_penalty,
    orientation_alignment,
    success_reward,
    distance_to_target,
    incremental_distance_to_target,
)


class TestDistanceBetweenObjects:
    """Tests for distance_between_objects reward functor."""

    def test_negative_distance_reward(self):
        """Test linear negative distance reward."""
        env = MockEnv(num_envs=4)
        obs = {}
        action = {}
        info = {}

        result = distance_between_objects(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_entity_cfg=MagicMock(uid="target"),
            exponential=False,
        )

        assert result.shape == (4,)
        # Distance from origin to (0.5, 0, 0) is 0.5
        assert result[0] == pytest.approx(-0.5, abs=0.01)

    def test_exponential_reward(self):
        """Test exponential Gaussian-shaped reward."""
        env = MockEnv(num_envs=4)
        obs = {}
        action = {}
        info = {}

        result = distance_between_objects(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_entity_cfg=MagicMock(uid="target"),
            exponential=True,
            sigma=0.2,
        )

        assert result.shape == (4,)
        # exp(-0.5^2 / (2 * 0.2^2)) = exp(-0.25 / 0.08) = exp(-3.125) ≈ 0.044
        assert result[0] == pytest.approx(0.044, abs=0.01)


class TestJointVelocityPenalty:
    """Tests for joint_velocity_penalty reward functor."""

    def test_returns_negative_penalty(self):
        """Test that joint velocity penalty is negative."""
        env = MockEnv(num_envs=4, num_joints=6)
        # Set some velocities
        env.robot.body_data.qvel = torch.ones(4, 6) * 0.5
        obs = {}
        action = {}
        info = {}

        result = joint_velocity_penalty(env, obs, action, info, robot_uid="robot")

        assert result.shape == (4,)
        # L2 norm of ones * 6 joints = sqrt(6) ≈ 2.45
        assert result[0] < 0

    def test_with_part_name(self):
        """Test joint velocity penalty with part_name."""
        env = MockEnv(num_envs=4, num_joints=6)
        env.robot.body_data.qvel = torch.ones(4, 6) * 0.5
        obs = {}
        action = {}
        info = {}

        result = joint_velocity_penalty(
            env, obs, action, info, robot_uid="robot", part_name="arm"
        )

        assert result.shape == (4,)

    def test_with_joint_ids(self):
        """Test joint velocity penalty with specific joint_ids."""
        env = MockEnv(num_envs=4, num_joints=6)
        env.robot.body_data.qvel = torch.ones(4, 6) * 0.5
        obs = {}
        action = {}
        info = {}

        result = joint_velocity_penalty(
            env, obs, action, info, robot_uid="robot", joint_ids=[0, 1, 2]
        )

        assert result.shape == (4,)


class TestActionSmoothnessPenalty:
    """Tests for action_smoothness_penalty reward functor."""

    def test_zero_on_first_step(self):
        """Test that penalty is zero on first step (no previous action)."""
        env = MockEnv(num_envs=4)
        env.current_rollout_step = 0
        obs = {}
        action = torch.ones(4, 6)
        info = {}

        result = action_smoothness_penalty(env, obs, action, info)

        assert result.shape == (4,)
        assert torch.all(result == 0)

    def test_penalty_on_subsequent_steps(self):
        """Test that penalty is negative when there was a previous action."""
        env = MockEnv(num_envs=4)
        # Set rollout state to step 1 with previous action of zeros
        env.current_rollout_step = 1
        env.rollout_buffer["action"][:4, 0, :] = torch.zeros(4, 6)
        env.rollout_buffer["done"][:4, 0] = False
        obs = {}
        action = torch.ones(4, 6) * 2.0  # large action change
        info = {}

        result = action_smoothness_penalty(env, obs, action, info)

        assert result.shape == (4,)
        # All have negative penalty from action difference of 2.0
        # Norm of 2.0 across 6 dims = sqrt(6 * 4) = sqrt(24) ≈ 4.9
        assert torch.all(result < 0)
        assert result[0] == pytest.approx(-4.9, abs=0.1)

    def test_handles_dict_action(self):
        """Test action smoothness with dict action."""
        env = MockEnv(num_envs=4)
        # Set rollout state to step 1 with previous action
        env.current_rollout_step = 1
        env.rollout_buffer["action"][:4, 0, :] = torch.zeros(4, 6)
        env.rollout_buffer["done"][:4, 0] = False
        obs = {}
        action = {"qpos": torch.ones(4, 6) * 2.0}
        info = {}

        result = action_smoothness_penalty(env, obs, action, info)

        assert result.shape == (4,)
        # All have negative penalty from action difference
        assert torch.all(result < 0)


class TestJointLimitPenalty:
    """Tests for joint_limit_penalty reward functor."""

    def test_zero_when_far_from_limits(self):
        """Test that penalty is zero when joints are far from limits."""
        env = MockEnv(num_envs=4, num_joints=6)
        # Set qpos to middle of range (0.0 is middle between -3.14 and 3.14)
        env.robot.body_data.qpos = torch.zeros(4, 6)
        obs = {}
        action = {}
        info = {}

        result = joint_limit_penalty(
            env, obs, action, info, robot_uid="robot", margin=0.1
        )

        assert result.shape == (4,)
        # Should be zero since we're far from limits
        assert torch.all(result == 0)

    def test_negative_when_near_limits(self):
        """Test that penalty is negative when joints are near limits."""
        env = MockEnv(num_envs=4, num_joints=6)
        # Set qpos very close to upper limit (3.14)
        env.robot.body_data.qpos = torch.ones(4, 6) * 3.0
        obs = {}
        action = {}
        info = {}

        result = joint_limit_penalty(
            env, obs, action, info, robot_uid="robot", margin=0.1
        )

        assert result.shape == (4,)
        # Should be negative since we're within margin
        assert torch.any(result < 0)


class TestOrientationAlignment:
    """Tests for orientation_alignment reward functor."""

    def test_perfect_alignment(self):
        """Test that perfect alignment returns 1.0."""
        env = MockEnv(num_envs=4)
        # Both at identity rotation
        obs = {}
        action = {}
        info = {}

        result = orientation_alignment(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_entity_cfg=MagicMock(uid="target"),
        )

        assert result.shape == (4,)
        assert result[0] == pytest.approx(1.0, abs=0.01)

    def test_opposite_orientation(self):
        """Test that opposite orientation returns -1.0."""
        env = MockEnv(num_envs=4)
        # Set cube to 180 degree rotation around x-axis
        env.test_object._pose = torch.eye(4).unsqueeze(0).repeat(4, 1, 1)
        env.test_object._pose[:, 1, 1] = -1
        env.test_object._pose[:, 2, 2] = -1

        obs = {}
        action = {}
        info = {}

        result = orientation_alignment(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_entity_cfg=MagicMock(uid="target"),
        )

        assert result.shape == (4,)
        assert result[0] == pytest.approx(-1.0, abs=0.01)


class TestSuccessReward:
    """Tests for success_reward functor."""

    def test_returns_zero_when_no_success_key(self):
        """Test that reward is zero when success key not in info."""
        env = MockEnv(num_envs=4)
        obs = {}
        action = {}
        info = {}

        result = success_reward(env, obs, action, info)

        assert result.shape == (4,)
        assert torch.all(result == 0)

    def test_returns_one_when_successful(self):
        """Test that reward is 1.0 when successful."""
        env = MockEnv(num_envs=4)
        obs = {}
        action = {}
        info = {"success": torch.tensor([True, True, False, False])}

        result = success_reward(env, obs, action, info)

        assert result.shape == (4,)
        torch.testing.assert_close(result, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    def test_handles_bool_success(self):
        """Test that reward handles boolean success."""
        env = MockEnv(num_envs=1)
        obs = {}
        action = {}
        info = {"success": True}

        result = success_reward(env, obs, action, info)

        assert result.shape == (1,)
        assert result[0] == 1.0


class TestDistanceToTarget:
    """Tests for distance_to_target reward functor."""

    def test_requires_target_pose_key(self):
        """Test that distance_to_target raises when target_pose_key not in env."""
        env = MockEnv(num_envs=4)
        # Don't set target_pose attribute
        obs = {}
        action = {}
        info = {}

        with pytest.raises(ValueError, match="Target pose"):
            distance_to_target(
                env,
                obs,
                action,
                info,
                source_entity_cfg=MagicMock(uid="cube"),
                target_pose_key="target_pose",
            )

    def test_returns_negative_distance(self):
        """Test that distance_to_target returns negative distance."""
        env = MockEnv(num_envs=4)
        # Set target pose
        env.target_pose = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1)
        # Set cube at origin
        env.test_object._pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.0])

        obs = {}
        action = {}
        info = {}

        result = distance_to_target(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_pose_key="target_pose",
        )

        assert result.shape == (4,)
        assert result[0] == pytest.approx(-0.5, abs=0.01)

    def test_exponential_reward(self):
        """Test exponential distance reward."""
        env = MockEnv(num_envs=4)
        env.target_pose = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1)
        env.test_object._pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.0])

        obs = {}
        action = {}
        info = {}

        result = distance_to_target(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_pose_key="target_pose",
            exponential=True,
            sigma=0.2,
        )

        assert result.shape == (4,)
        # exp(-0.5^2 / (2 * 0.2^2)) is very small
        assert result[0] < 0.1


class TestIncrementalDistanceToTarget:
    """Tests for incremental_distance_to_target reward functor."""

    def test_returns_zero_on_first_call(self):
        """Test that incremental distance returns zero on first call."""
        env = MockEnv(num_envs=4)
        env.target_pose = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1)
        env.test_object._pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.0])

        obs = {}
        action = {}
        info = {}

        # First call should return zeros (initializes state)
        result = incremental_distance_to_target(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_pose_key="target_pose",
        )

        assert result.shape == (4,)
        assert torch.all(result == 0)

    def test_positive_when_getting_closer(self):
        """Test that incremental distance is positive when getting closer."""
        env = MockEnv(num_envs=4)

        # First call - set initial distance
        env.test_object._pose[:, :3, 3] = torch.tensor([0.0, 0.0, 0.0])
        env.target_pose = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1)
        env._reward_states = {}

        obs = {}
        action = {}
        info = {}

        # First call - initializes state
        _ = incremental_distance_to_target(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_pose_key="target_pose",
        )

        # Move closer
        env.test_object._pose[:, :3, 3] = torch.tensor([0.25, 0.0, 0.0])

        # Second call - should be positive (getting closer)
        result = incremental_distance_to_target(
            env,
            obs,
            action,
            info,
            source_entity_cfg=MagicMock(uid="cube"),
            target_pose_key="target_pose",
        )

        assert result.shape == (4,)
        # Distance decreased from 0.5 to 0.25, so should be positive
        assert torch.any(result > 0)
