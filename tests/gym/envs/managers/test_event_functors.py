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

"""Tests for event functors."""

from __future__ import annotations

import pytest
import torch

from unittest.mock import MagicMock, Mock


class MockRobot:
    """Mock robot for event functor tests."""

    def __init__(self, num_envs: int = 4, num_joints: int = 6):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = torch.device("cpu")
        self.joint_names = [f"joint_{i}" for i in range(num_joints)]

    def get_qpos(self, *args, **kwargs):
        return torch.zeros(self.num_envs, self.num_joints)

    def get_joint_ids(self, part_name=None):
        return list(range(self.num_joints))


class MockRigidObject:
    """Mock rigid object for event functor tests."""

    def __init__(
        self, uid: str = "test_object", num_envs: int = 4, is_dynamic: bool = True
    ):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.is_non_dynamic = not is_dynamic

        # Mock cfg
        self.cfg = Mock()
        self.cfg.shape = Mock()
        self.cfg.shape.fpath = "test.obj"
        self.cfg.attrs = Mock()
        self.cfg.attrs.mass = 1.0

        # Default pose at origin
        self._pose = torch.eye(4).unsqueeze(0).repeat(num_envs, 1, 1)
        self._mass = torch.ones(num_envs) * 1.0
        self._com = torch.zeros(num_envs, 3)

        # Mock body_data
        self.body_data = Mock()
        self.body_data.default_com_pose = torch.zeros(num_envs, 7)
        self.body_data.default_com_pose[:, 3] = 1.0  # quaternion w

    def get_local_pose(self, to_matrix=True):
        return self._pose

    def set_local_pose(self, pose, env_ids=None, obj_ids=None):
        if env_ids is not None:
            self._pose[env_ids] = pose[:, env_ids] if pose.dim() > 3 else pose
        else:
            self._pose = pose

    def get_mass(self, env_ids=None):
        if env_ids is not None:
            return self._mass[env_ids].unsqueeze(-1)
        return self._mass.unsqueeze(-1)

    def set_mass(self, mass, env_ids=None):
        if env_ids is not None:
            self._mass[env_ids] = mass
        else:
            self._mass = mass


class MockRigidObjectGroup:
    """Mock rigid object group for event functor tests."""

    def __init__(self, uid: str = "object_group", num_objects: int = 3):
        self.uid = uid
        self.num_objects = num_objects

    def set_local_pose(self, pose, env_ids, obj_ids):
        pass


class MockArticulation:
    """Mock articulation for event functor tests."""

    def __init__(self, uid: str = "test_articulation", num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")

        # Default pose at origin (position + quaternion)
        # Format: (N, 7) - position (3) + quaternion (4)
        self._pose = torch.zeros(num_envs, 7)
        self._pose[:, 3] = 1.0  # quaternion w = 1 (identity rotation)

    def get_local_pose(self, to_matrix: bool = False):
        """Returns pose in (N, 7) format: position (3) + quaternion (4)."""
        return self._pose

    def set_local_pose(self, pose, env_ids=None):
        if env_ids is not None:
            self._pose[env_ids] = pose[env_ids] if pose.dim() > 2 else pose
        else:
            self._pose = pose

    def clear_dynamics(self, env_ids=None):
        """Clear dynamics - no-op for mock."""
        pass


class MockSim:
    """Mock simulation for event functor tests."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._rigid_objects = {}
        self._articulations = {}
        self._robots = {}
        self._rigid_object_groups = {}

    def get_rigid_object(self, uid: str):
        return self._rigid_objects.get(uid)

    def get_rigid_object_uid_list(self):
        return list(self._rigid_objects.keys())

    def get_articulation_uid_list(self):
        return list(self._articulations.keys())

    def get_articulation(self, uid: str):
        return self._articulations.get(uid)

    def add_articulation(self, articulation):
        self._articulations[articulation.uid] = articulation
        return articulation

    def get_robot(self, uid: str = None):
        if uid is None:
            return list(self._robots.values())[0] if self._robots else None
        return self._robots.get(uid)

    def get_robot_uid_list(self):
        return list(self._robots.keys())

    def get_asset(self, uid: str):
        return self._rigid_objects.get(uid)

    def add_rigid_object(self, obj):
        self._rigid_objects[obj.uid] = obj
        return obj

    def remove_asset(self, uid: str):
        if uid in self._rigid_objects:
            del self._rigid_objects[uid]

    def add_robot(self, robot):
        self._robots["robot"] = robot

    def get_rigid_object_group(self, uid: str):
        return self._rigid_object_groups.get(uid)

    def update(self, step: int = 1):
        pass


class MockEnv:
    """Mock environment for event functor tests."""

    def __init__(self, num_envs: int = 4, num_joints: int = 6):
        self.num_envs = num_envs
        self.device = torch.device("cpu")

        self.sim = MockSim(num_envs)
        self.robot = MockRobot(num_envs, num_joints)
        self.sim.add_robot(self.robot)

        # Add test rigid objects
        self.test_object = MockRigidObject("cube", num_envs)
        self.sim.add_rigid_object(self.test_object)

        self.target_object = MockRigidObject("target", num_envs)
        self.target_object._pose[:, :3, 3] = torch.tensor([0.5, 0.0, 0.0])
        self.sim.add_rigid_object(self.target_object)

        # Add test articulation
        self.test_articulation = MockArticulation("articulation", num_envs)
        self.sim.add_articulation(self.test_articulation)

        # For affordance registration
        self.affordance_datas = {}


# Import functors to test
from embodichain.lab.gym.envs.managers.events import (
    resolve_uids,
    resolve_dict,
    set_detached_uids_for_env_reset,
)
from embodichain.lab.gym.envs.managers.randomization.physics import (
    randomize_rigid_object_mass,
)
from embodychain.lab.gym.envs.managers.randomization.spatial import (
    randomize_articulation_root_pose,
)


class TestResolveUids:
    """Tests for resolve_uids function."""

    def test_resolve_all_objects(self):
        """Test resolving 'all_objects' string."""
        env = MockEnv()
        # Already has cube and target added
        result = resolve_uids(env, "all_objects")

        assert "cube" in result
        assert "target" in result

    def test_resolve_all_robots(self):
        """Test resolving 'all_robots' string."""
        env = MockEnv()

        result = resolve_uids(env, "all_robots")

        assert "robot" in result

    def test_resolve_single_string(self):
        """Test resolving a single UID string."""
        env = MockEnv()

        result = resolve_uids(env, "cube")

        assert result == ["cube"]

    def test_resolve_list(self):
        """Test resolving a list of UIDs."""
        env = MockEnv()

        result = resolve_uids(env, ["cube", "target"])

        assert result == ["cube", "target"]


class TestResolveDict:
    """Tests for resolve_dict function."""

    def test_resolve_dict_with_all_objects(self):
        """Test resolving dictionary with 'all_objects' key."""
        env = MockEnv()

        input_dict = {"all_objects": {"param": "value"}}

        result = resolve_dict(env, input_dict)

        assert "cube" in result
        assert "target" in result
        assert result["cube"]["param"] == "value"


class TestRandomizeRigidObjectMass:
    """Tests for randomize_rigid_object_mass functor."""

    def test_sets_mass_in_range(self):
        """Test that mass is randomized within the specified range."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])
        mass_range = (0.5, 2.0)

        randomize_rigid_object_mass(
            env, env_ids, entity_cfg=MagicMock(uid="cube"), mass_range=mass_range
        )

        # Check masses are in range
        masses = env.test_object.get_mass()
        assert torch.all(masses >= 0.5)
        assert torch.all(masses <= 2.0)

    def test_relative_mass_randomization(self):
        """Test relative mass randomization."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Initial mass is 1.0
        randomize_rigid_object_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="cube"),
            mass_range=(-0.5, 0.5),
            relative=True,
        )

        # Final mass should be in range [0.5, 1.5]
        masses = env.test_object.get_mass()
        assert torch.all(masses >= 0.5)
        assert torch.all(masses <= 1.5)

    def test_handles_nonexistent_object(self):
        """Test that function handles non-existent object gracefully."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Should not raise - function returns early for non-existent objects
        randomize_rigid_object_mass(
            env, env_ids, entity_cfg=MagicMock(uid="nonexistent"), mass_range=(0.5, 2.0)
        )


class TestSetDetachedUidsForEnvReset:
    """Tests for set_detached_uids_for_env_reset functor."""

    def test_adds_detached_uids(self):
        """Test that detached UIDs are added to environment."""
        env = MockEnv(num_envs=4)

        # Mock add_detached_uids_for_reset
        env.add_detached_uids_for_reset = Mock()

        set_detached_uids_for_env_reset(env, None, uids=["detached_object"])

        env.add_detached_uids_for_reset.assert_called_once_with(
            uids=["detached_object"]
        )


class TestRandomizeArticulationRootPose:
    """Tests for randomize_articulation_root_pose functor."""

    def test_randomize_position_absolute(self):
        """Test absolute position randomization."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Set initial pose
        initial_pos = torch.zeros(4, 3)
        initial_pos[:, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3])
        initial_quat = torch.zeros(4, 4)
        initial_quat[:, 3] = 1.0  # identity quaternion
        env.test_articulation._pose = torch.cat([initial_pos, initial_quat], dim=1)

        # Randomize with absolute position range
        randomize_articulation_root_pose(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            position_range=([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]),
            rotation_range=None,
            relative_position=False,
            relative_rotation=False,
        )

        # Check that position was randomized within range
        pose = env.test_articulation.get_local_pose()
        pos = pose[:, :3]

        assert torch.all(pos[:, 0] >= -0.5)
        assert torch.all(pos[:, 0] <= 0.5)
        assert torch.all(pos[:, 1] >= -0.5)
        assert torch.all(pos[:, 1] <= 0.5)

    def test_randomize_position_relative(self):
        """Test relative position randomization."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Set initial pose at origin
        initial_pos = torch.zeros(4, 3)
        initial_quat = torch.zeros(4, 4)
        initial_quat[:, 3] = 1.0  # identity quaternion
        env.test_articulation._pose = torch.cat([initial_pos, initial_quat], dim=1)

        # Get initial position
        initial_pos_before = env.test_articulation.get_local_pose()[:, :3].clone()

        # Randomize with relative position range
        randomize_articulation_root_pose(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            position_range=([-0.1, -0.1, 0.0], [0.1, 0.1, 0.0]),
            rotation_range=None,
            relative_position=True,
            relative_rotation=False,
        )

        # Check that position changed
        pose = env.test_articulation.get_local_pose()
        pos = pose[:, :3]

        # Position should be different from initial
        assert torch.any(torch.abs(pos - initial_pos_before) > 1e-6)

    def test_randomize_rotation(self):
        """Test rotation randomization."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Set initial pose
        initial_pos = torch.zeros(4, 3)
        initial_quat = torch.zeros(4, 4)
        initial_quat[:, 3] = 1.0  # identity quaternion
        env.test_articulation._pose = torch.cat([initial_pos, initial_quat], dim=1)

        # Randomize with rotation range
        randomize_articulation_root_pose(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            position_range=None,
            rotation_range=([-45, -45, -45], [45, 45, 45]),
            relative_position=False,
            relative_rotation=False,
        )

        # Check that rotation changed (quaternion should not be identity anymore)
        pose = env.test_articulation.get_local_pose()
        quat = pose[:, 3:7]

        # At least some quaternions should be different from identity
        identity_quat = torch.zeros(4)
        identity_quat[3] = 1.0
        is_identity = torch.all(torch.abs(quat - identity_quat) < 1e-6, dim=1)
        assert not torch.all(is_identity), "Rotation should have changed"

    def test_handles_nonexistent_articulation(self):
        """Test that function handles non-existent articulation gracefully."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Should not raise - function returns early for non-existent articulations
        randomize_articulation_root_pose(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="nonexistent"),
            position_range=([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]),
            rotation_range=None,
        )

    def test_physics_update_step(self):
        """Test that physics update step is called when specified."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Mock the update method
        env.sim.update = Mock()

        randomize_articulation_root_pose(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            position_range=([-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]),
            rotation_range=None,
            physics_update_step=10,
        )

        # Check that update was called
        env.sim.update.assert_called_once_with(step=10)
