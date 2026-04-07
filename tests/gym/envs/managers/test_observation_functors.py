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

"""Tests for observation functors."""

from __future__ import annotations

import pytest
import torch

from unittest.mock import MagicMock, Mock, patch


class MockRobot:
    """Mock robot for observation functor tests."""

    def __init__(self, num_envs: int = 4, num_joints: int = 6):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = torch.device("cpu")
        self.joint_names = [f"joint_{i}" for i in range(num_joints)]
        self.link_names = [
            "base",
            "left_shoulder",
            "left_elbow",
            "right_shoulder",
            "right_elbow",
        ]
        self._qpos = torch.zeros(num_envs, num_joints)
        self._qvel = torch.zeros(num_envs, num_joints)
        self.user_ids = torch.tensor([[0, 1, 2, 3, 4]] * num_envs, dtype=torch.int32)

        # Mock body_data
        self.body_data = Mock()
        self.body_data.qpos = self._qpos
        self.body_data.qvel = self._qvel
        self.body_data.qpos_limits = torch.zeros(1, num_joints, 2)
        self.body_data.qpos_limits[..., 0] = -3.14
        self.body_data.qpos_limits[..., 1] = 3.14

    def get_qpos(self, *args, **kwargs):
        return self._qpos

    def get_qvel(self, *args, **kwargs):
        return self._qvel

    def compute_fk(self, qpos=None, name=None, to_matrix=True):
        # Return identity poses
        pose = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1)
        if to_matrix:
            return pose
        return pose[:, :3, 3]

    def get_joint_ids(self, part_name=None):
        return list(range(self.num_joints))

    def get_user_ids(self):
        return torch.tensor([1], device=self.device)

    def get_joint_drive(self, joint_ids=None, env_ids=None):
        num_envs = len(env_ids) if env_ids is not None else self.num_envs
        joints = len(joint_ids) if joint_ids is not None else self.num_joints
        stiffness = torch.ones((num_envs, joints), device=self.device) * 100.0
        damping = torch.ones((num_envs, joints), device=self.device) * 10.0
        max_effort = torch.ones((num_envs, joints), device=self.device) * 50.0
        max_velocity = torch.ones((num_envs, joints), device=self.device) * 5.0
        friction = torch.ones((num_envs, joints), device=self.device) * 1.0
        return stiffness, damping, max_effort, max_velocity, friction


class MockRigidObject:
    """Mock rigid object for observation functor tests."""

    def __init__(self, uid: str = "test_object", num_envs: int = 4):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        # Default pose at origin
        self._pose = torch.eye(4).unsqueeze(0).repeat(num_envs, 1, 1)
        # Default velocity (6D)
        self._vel = torch.zeros(num_envs, 6)

        # Mock body_data with vel attribute
        self.body_data = Mock()
        self.body_data.vel = self._vel

    def get_local_pose(self, to_matrix=True):
        if to_matrix:
            return self._pose
        # Return as (position, quaternion)
        pos = self._pose[:, :3, 3]
        # Simple quaternion from identity rotation
        quat = torch.zeros(self.num_envs, 4)
        quat[:, 0] = 1.0  # w=1 (identity)
        return torch.cat([pos, quat], dim=-1)

    def get_mass(self):
        """Return mock mass for each environment."""
        return torch.ones(self.num_envs, 1)

    def get_friction(self):
        """Return mock friction for each environment."""
        return torch.tensor([[0.5]]).repeat(self.num_envs, 1)

    def get_damping(self):
        """Return mock damping for each environment."""
        return torch.tensor([[0.1, 0.1]]).repeat(self.num_envs, 1)

    def get_inertia(self):
        """Return mock inertia tensor for each environment."""
        return torch.tensor([[0.1, 0.2, 0.1]]).repeat(self.num_envs, 1)

    def get_body_scale(self):
        """Return mock body scale for each environment."""
        return torch.tensor([[1.0, 1.0, 1.0]]).repeat(self.num_envs, 1)

    def get_user_ids(self):
        """Return mock user IDs for each environment."""
        return torch.ones(self.num_envs, dtype=torch.int32)

    @property
    def body(self):
        return self


class MockSensor:
    """Mock sensor for observation functor tests."""

    def __init__(self, uid: str = "camera", num_envs: int = 1):
        self.uid = uid
        self.num_envs = num_envs
        self.cfg = Mock()
        self.cfg.height = 480
        self.cfg.width = 640
        self.cfg.enable_mask = True

    def get_left_right_arena_pose(self):
        pose = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1)
        return pose, pose

    def get_arena_pose(self, to_matrix=True):
        pose = torch.eye(4).unsqueeze(0).repeat(self.num_envs, 1, 1)
        return pose

    def get_intrinsics(self):
        # Return mock intrinsic matrix
        intrinsics = torch.zeros(self.num_envs, 3, 3)
        intrinsics[:, 0, 0] = 500.0  # fx
        intrinsics[:, 1, 1] = 500.0  # fy
        intrinsics[:, 0, 2] = 320.0  # cx
        intrinsics[:, 1, 2] = 240.0  # cy
        intrinsics[:, 2, 2] = 1.0
        return intrinsics


class MockSim:
    """Mock simulation for observation functor tests."""

    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self._rigid_objects = {}
        self._robots = {}
        self._sensors = {}
        self.asset_uids = []

    def get_rigid_object(self, uid: str):
        return self._rigid_objects.get(uid)

    def get_rigid_object_uid_list(self):
        return list(self._rigid_objects.keys())

    def get_robot(self, uid: str = None):
        if uid is None:
            return list(self._robots.values())[0] if self._robots else None
        return self._robots.get(uid)

    def get_robot_uid_list(self):
        return list(self._robots.keys())

    def get_articulation(self, uid: str):
        return self._robots.get(uid)

    def get_articulation_uid_list(self):
        return list(self._robots.keys())

    def get_sensor(self, uid: str):
        return self._sensors.get(uid)

    def get_asset(self, uid: str):
        """Get an asset by UID from rigid objects or robots."""
        if uid in self._rigid_objects:
            return self._rigid_objects.get(uid)
        elif uid in self._robots:
            return self._robots.get(uid)
        return None

    def add_rigid_object(self, obj):
        self._rigid_objects[obj.uid] = obj
        self.asset_uids.append(obj.uid)

    def add_robot(self, robot):
        self._robots["robot"] = robot


class MockEnv:
    """Mock environment for observation functor tests."""

    def __init__(self, num_envs: int = 4, num_joints: int = 6):
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.active_joint_ids = list(range(num_joints))

        self.sim = MockSim(num_envs)
        self.robot = MockRobot(num_envs, num_joints)
        self.sim.add_robot(self.robot)

        # Add test rigid objects
        self.test_object = MockRigidObject("test_cube", num_envs)
        self.sim.add_rigid_object(self.test_object)

        self.target_object = MockRigidObject("target", num_envs)
        self.target_object._pose[:, :3, 3] = torch.tensor([0.5, 0.0, 0.0])
        self.sim.add_rigid_object(self.target_object)

        # Add sensor
        self.test_camera = MockSensor("camera", num_envs)
        self.sim._sensors["camera"] = self.test_camera


# Import functors to test
from embodichain.lab.gym.envs.managers.observations import (
    get_rigid_object_pose,
    get_rigid_object_velocity,
    normalize_robot_joint_data,
    get_sensor_pose_in_robot_frame,
    get_sensor_intrinsics,
    compute_semantic_mask,
    get_robot_eef_pose,
    target_position,
    get_rigid_object_physics_attributes,
    get_articulation_joint_drive,
    get_object_uid,
)


class TestGetRigidObjectPose:
    """Tests for get_rigid_object_pose functor."""

    def test_returns_matrix_pose(self):
        """Test that get_rigid_object_pose returns 4x4 matrix by default."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_rigid_object_pose(
            env, obs, entity_cfg=MagicMock(uid="test_cube"), to_matrix=True
        )

        assert result.shape == (4, 4, 4)
        # Identity matrix for default pose
        torch.testing.assert_close(result[0], torch.eye(4))

    def test_returns_position_quaternion(self):
        """Test that get_rigid_object_pose returns position+quaternion when to_matrix=False."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_rigid_object_pose(
            env, obs, entity_cfg=MagicMock(uid="test_cube"), to_matrix=False
        )

        assert result.shape == (4, 7)

    def test_returns_zero_for_nonexistent_object(self):
        """Test that get_rigid_object_pose returns zeros for non-existent object."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_rigid_object_pose(
            env, obs, entity_cfg=MagicMock(uid="nonexistent"), to_matrix=True
        )

        assert result.shape == (4, 4, 4)
        assert torch.all(result == 0)


class TestGetRigidObjectVelocity:
    """Tests for get_rigid_object_velocity functor."""

    def test_returns_velocity_shape(self):
        """Test that get_rigid_object_velocity returns correct shape."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_rigid_object_velocity(
            env, obs, entity_cfg=MagicMock(uid="test_cube")
        )

        assert result.shape == (4, 6)  # 6D velocity

    def test_returns_zero_for_nonexistent_object(self):
        """Test that get_rigid_object_velocity returns zeros for non-existent object."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_rigid_object_velocity(
            env, obs, entity_cfg=MagicMock(uid="nonexistent")
        )

        assert result.shape == (4, 6)
        assert torch.all(result == 0)


class TestNormalizeRobotJointData:
    """Tests for normalize_robot_joint_data functor."""

    def test_normalizes_to_0_1_range(self):
        """Test that joint data is normalized to [0, 1] range."""
        env = MockEnv(num_envs=4, num_joints=6)

        # Create data at limits (-3.14 and 3.14)
        data = torch.zeros(4, 6)
        data[:, 0] = -3.14  # at lower limit
        data[:, 1] = 0.0  # at middle
        data[:, 2] = 3.14  # at upper limit
        joint_ids = [0, 1, 2]

        result = normalize_robot_joint_data(
            env, data.clone(), joint_ids, limit="qpos_limits"
        )

        # Check normalization
        assert result[0, 0] == pytest.approx(0.0, abs=0.01)
        assert result[0, 1] == pytest.approx(0.5, abs=0.01)
        assert result[0, 2] == pytest.approx(1.0, abs=0.01)


class TestGetSensorIntrinsics:
    """Tests for get_sensor_intrinsics functor."""

    def test_returns_intrinsics_matrix(self):
        """Test that get_sensor_intrinsics returns correct shape."""
        env = MockEnv(num_envs=1)
        obs = {}

        # Replace the mock sensor with a proper one that will pass isinstance check
        # by using patch to mock the Camera import
        with patch("embodichain.lab.gym.envs.managers.observations.Camera", MockSensor):
            result = get_sensor_intrinsics(env, obs, entity_cfg=MagicMock(uid="camera"))

        assert result.shape == (1, 3, 3)
        # Check fx, fy are set
        assert result[0, 0, 0] == 500.0
        assert result[0, 1, 1] == 500.0


class TestGetRobotEefPose:
    """Tests for get_robot_eef_pose functor."""

    def test_returns_matrix_pose_by_default(self):
        """Test that get_robot_eef_pose returns 4x4 matrix by default."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_robot_eef_pose(env, obs)

        assert result.shape == (4, 4, 4)

    def test_returns_position_only(self):
        """Test that get_robot_eef_pose returns only position when position_only=True."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_robot_eef_pose(env, obs, position_only=True)

        assert result.shape == (4, 3)

    def test_with_part_name(self):
        """Test that get_robot_eef_pose works with part_name."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_robot_eef_pose(env, obs, part_name="arm")

        assert result.shape == (4, 4, 4)


class TestTargetPosition:
    """Tests for target_position functor."""

    def test_returns_zeros_when_not_initialized(self):
        """Test that target_position returns zeros before initialization."""
        env = MockEnv(num_envs=4)
        obs = {}

        # Without target_pose_key attribute set
        result = target_position(env, obs, target_pose_key="goal_pose")

        assert result.shape == (4, 3)
        assert torch.all(result == 0)

    def test_returns_position_from_env_attribute(self):
        """Test that target_position reads from env attribute."""
        env = MockEnv(num_envs=4)
        obs = {}

        # Set the target pose
        env.goal_pose = torch.tensor([[0.5, 0.0, 0.0]]).repeat(4, 1)

        result = target_position(env, obs, target_pose_key="goal_pose")

        assert result.shape == (4, 3)
        torch.testing.assert_close(result[0], torch.tensor([0.5, 0.0, 0.0]))

    def test_handles_matrix_pose(self):
        """Test that target_position handles 4x4 matrix poses."""
        env = MockEnv(num_envs=4)
        obs = {}

        # Set as 4x4 matrix
        pose = torch.eye(4).unsqueeze(0).repeat(4, 1, 1)
        pose[:, :3, 3] = torch.tensor([0.5, 0.3, 0.1])
        env.goal_pose = pose

        result = target_position(env, obs, target_pose_key="goal_pose")

        assert result.shape == (4, 3)
        torch.testing.assert_close(result[0], torch.tensor([0.5, 0.3, 0.1]))


class TestGetObjectUid:
    """Tests for get_object_uid functor."""

    @patch(
        "embodichain.lab.gym.envs.managers.observations.RigidObject", MockRigidObject
    )
    def test_returns_correct_shape(self):
        """Test that get_object_uid returns correct tensor shape."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_object_uid(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        assert result.shape == (4,)
        assert result.dtype == torch.int32

    @patch(
        "embodichain.lab.gym.envs.managers.observations.RigidObject", MockRigidObject
    )
    def test_returns_correct_value(self):
        """Test that get_object_uid returns correct user ID from object."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_object_uid(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        # Check value matches mock object's user_id (which is 1)
        torch.testing.assert_close(
            result, torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        )

    def test_returns_zero_for_nonexistent_object(self):
        """Test that get_object_uid returns zeros for non-existent object."""
        env = MockEnv(num_envs=4)
        obs = {}

        result = get_object_uid(env, obs, entity_cfg=MagicMock(uid="nonexistent"))

        assert result.shape == (4,)
        assert torch.all(result == 0)

    @patch(
        "embodichain.lab.gym.envs.managers.observations.RigidObject", MockRigidObject
    )
    def test_different_num_envs(self):
        """Test that functor works with different number of environments."""
        env = MockEnv(num_envs=8)
        obs = {}

        result = get_object_uid(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        assert result.shape == (8,)


class TestGetRigidObjectPhysicsAttributes:
    """Tests for get_rigid_object_physics_attributes class functor."""

    def test_returns_correct_shapes(self):
        """Test that functor returns correct tensor shapes."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        # Check shapes
        assert result["mass"].shape == (4, 1)
        assert result["friction"].shape == (4, 1)
        assert result["damping"].shape == (4, 2)
        assert result["inertia"].shape == (4, 3)

    def test_returns_correct_values(self):
        """Test that functor returns correct physics values from object."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        # Check values match mock object
        torch.testing.assert_close(result["mass"], torch.ones(4, 1))
        torch.testing.assert_close(
            result["friction"], torch.tensor([[0.5]]).repeat(4, 1)
        )
        torch.testing.assert_close(
            result["damping"], torch.tensor([[0.1, 0.1]]).repeat(4, 1)
        )
        torch.testing.assert_close(
            result["inertia"], torch.tensor([[0.1, 0.2, 0.1]]).repeat(4, 1)
        )

    def test_returns_zeros_for_nonexistent_object(self):
        """Test that functor returns zero tensors for non-existent object."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="nonexistent"))

        # Check all attributes are zero
        assert torch.all(result["mass"] == 0)
        assert torch.all(result["friction"] == 0)
        assert torch.all(result["damping"] == 0)
        assert torch.all(result["inertia"] == 0)

    def test_caches_data_across_calls(self):
        """Test that data is cached and reused on subsequent calls."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        result1 = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))
        assert len(functor._cache) == 1

        # Call again - should use cache
        result2 = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))
        assert len(functor._cache) == 1  # Still just 1 entry

        # Values should be identical
        torch.testing.assert_close(result1["mass"], result2["mass"])
        torch.testing.assert_close(result1["friction"], result2["friction"])
        torch.testing.assert_close(result1["damping"], result2["damping"])
        torch.testing.assert_close(result1["inertia"], result2["inertia"])

    def test_reset_clears_cache(self):
        """Test that reset() clears the internal cache."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        # Populate cache
        functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))
        assert len(functor._cache) == 1

        # Reset should clear cache
        functor.reset()
        assert len(functor._cache) == 0

    def test_reset_with_env_ids_clears_cache(self):
        """Test that reset(env_ids=...) clears the internal cache."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        # Populate cache
        functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))
        assert len(functor._cache) == 1

        # Reset with env_ids should still clear cache (current implementation clears all)
        functor.reset(env_ids=[0, 1])
        assert len(functor._cache) == 0

    def test_caches_multiple_objects_separately(self):
        """Test that different objects have separate cache entries."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        result1 = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))
        result2 = functor(env, obs, entity_cfg=MagicMock(uid="target"))

        # Should have 2 separate cache entries
        assert len(functor._cache) == 2
        assert "test_cube" in functor._cache
        assert "target" in functor._cache

    def test_returns_clones_not_references(self):
        """Test that returned tensors are clones, not references to cache."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        # Modify the returned result
        result["mass"][:] = 999.0

        # Get result again - should still have original value
        result2 = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        # Cache should not be affected by modification
        assert torch.allclose(result2["mass"], torch.ones(4, 1))
        assert not torch.allclose(result["mass"], torch.ones(4, 1))

    def test_different_num_envs(self):
        """Test that functor works with different number of environments."""
        env = MockEnv(num_envs=8)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_rigid_object_physics_attributes(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="test_cube"))

        # Check shapes match num_envs
        assert result["mass"].shape == (8, 1)
        assert result["inertia"].shape == (8, 3)


class TestGetArticulationJointDrive:
    """Tests for get_articulation_joint_drive class functor."""

    def test_returns_correct_shapes(self):
        """Test that the functor returns properties with correct shapes."""
        env = MockEnv(num_envs=4, num_joints=6)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_articulation_joint_drive(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="robot"))

        assert "stiffness" in result.keys()
        assert "damping" in result.keys()
        assert "max_effort" in result.keys()
        assert "max_velocity" in result.keys()
        assert "friction" in result.keys()

        assert result["stiffness"].shape == (4, 6)
        assert result["damping"].shape == (4, 6)
        assert result["max_effort"].shape == (4, 6)
        assert result["max_velocity"].shape == (4, 6)
        assert result["friction"].shape == (4, 6)

    def test_returns_correct_values(self):
        """Test that the functor returns expected mock values."""
        env = MockEnv(num_envs=4, num_joints=6)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_articulation_joint_drive(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="robot"))

        assert torch.allclose(result["stiffness"], torch.ones(4, 6) * 100.0)
        assert torch.allclose(result["damping"], torch.ones(4, 6) * 10.0)
        assert torch.allclose(result["max_effort"], torch.ones(4, 6) * 50.0)
        assert torch.allclose(result["max_velocity"], torch.ones(4, 6) * 5.0)
        assert torch.allclose(result["friction"], torch.ones(4, 6) * 1.0)

    def test_returns_zeros_for_nonexistent_object(self):
        """Test that zeros are returned for non-existent objects."""
        env = MockEnv(num_envs=4)
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_articulation_joint_drive(cfg=FunctorCfg(), env=env)

        result = functor(env, obs, entity_cfg=MagicMock(uid="does_not_exist"))

        assert torch.allclose(result["stiffness"], torch.zeros(4, 1))
        assert torch.allclose(result["damping"], torch.zeros(4, 1))
        assert torch.allclose(result["max_effort"], torch.zeros(4, 1))
        assert torch.allclose(result["max_velocity"], torch.zeros(4, 1))
        assert torch.allclose(result["friction"], torch.zeros(4, 1))

    def test_caches_data_across_calls(self):
        """Test that fetched data is cached for subsequent calls."""
        env = MockEnv(num_envs=4)
        # Verify the robot gets called
        env.sim._robots["robot"].get_joint_drive = MagicMock(
            return_value=(
                torch.ones(4, 6),
                torch.ones(4, 6),
                torch.ones(4, 6),
                torch.ones(4, 6),
                torch.ones(4, 6),
            )
        )
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_articulation_joint_drive(cfg=FunctorCfg(), env=env)

        # First call should fetch
        functor(env, obs, entity_cfg=MagicMock(uid="robot"))
        assert env.sim._robots["robot"].get_joint_drive.call_count == 1

        # Second call should use cache
        functor(env, obs, entity_cfg=MagicMock(uid="robot"))
        assert env.sim._robots["robot"].get_joint_drive.call_count == 1

    def test_reset_clears_cache(self):
        """Test that calling reset clears the cache."""
        env = MockEnv(num_envs=4)
        env.sim._robots["robot"].get_joint_drive = MagicMock(
            return_value=(
                torch.ones(4, 6),
                torch.ones(4, 6),
                torch.ones(4, 6),
                torch.ones(4, 6),
                torch.ones(4, 6),
            )
        )
        obs = {}
        from embodichain.lab.gym.envs.managers.cfg import FunctorCfg

        functor = get_articulation_joint_drive(cfg=FunctorCfg(), env=env)

        # Populate cache
        functor(env, obs, entity_cfg=MagicMock(uid="robot"))
        assert env.sim._robots["robot"].get_joint_drive.call_count == 1

        # Reset clears cache
        functor.reset()

        # Should fetch again
        functor(env, obs, entity_cfg=MagicMock(uid="robot"))
        assert env.sim._robots["robot"].get_joint_drive.call_count == 2


class TestComputeSemanticMask:
    """Tests for compute_semantic_mask functor."""

    # Layout of the synthetic 4x4 mask used in tests:
    #   [[ 1,  2,  3,  4],
    #    [ 0,  0, 10, 10],
    #    [ 1,  0,  3, 10],
    #    [ 0,  0,  0,  0]]
    #
    # Robot user_ids = [[0, 1, 2, 3, 4]] per env
    #   left_link_indices  = [1, 2] → uids 1, 2
    #   right_link_indices = [3, 4] → uids 3, 4
    # Foreground object uid = 10

    def _make_env_and_obs(self, num_envs: int = 2):
        """Create a mock env with a synthetic 4x4 semantic mask."""
        env = MockEnv(num_envs=num_envs)

        # Create a foreground object with user_id 10
        fg_obj = MockRigidObject("fg_object", num_envs)
        fg_obj.get_user_ids = lambda: torch.full((num_envs,), 10, dtype=torch.int32)
        env.sim.add_rigid_object(fg_obj)

        single = torch.tensor(
            [[1, 2, 3, 4], [0, 0, 10, 10], [1, 0, 3, 10], [0, 0, 0, 0]],
            dtype=torch.int32,
        )
        mask = single.unsqueeze(0).repeat(num_envs, 1, 1)

        obs = {"sensor": {"camera": {"mask": mask}}}
        return env, obs

    def test_returns_correct_shape(self):
        """Test that compute_semantic_mask returns shape (B, H, W, 4)."""
        env, obs = self._make_env_and_obs(num_envs=2)

        result = compute_semantic_mask(
            env,
            obs,
            entity_cfg=MagicMock(uid="camera"),
            foreground_uids=["fg_object"],
        )

        assert result.shape == (2, 4, 4, 4)

    def test_correct_channel_assignment(self):
        """Test that each semantic channel has 1s only in expected pixels."""
        env, obs = self._make_env_and_obs(num_envs=2)

        result = compute_semantic_mask(
            env,
            obs,
            entity_cfg=MagicMock(uid="camera"),
            foreground_uids=["fg_object"],
        )

        # SemanticMask channels: BACKGROUND=0, FOREGROUND=1, ROBOT_LEFT=2, ROBOT_RIGHT=3
        bg = result[0, :, :, 0]
        fg = result[0, :, :, 1]
        left = result[0, :, :, 2]
        right = result[0, :, :, 3]

        # Left robot (uids 1, 2): pixels (0,0), (0,1), (2,0)
        assert left[0, 0] == 1
        assert left[0, 1] == 1
        assert left[2, 0] == 1
        assert left.sum().item() == 3

        # Right robot (uids 3, 4): pixels (0,2), (0,3), (2,2)
        assert right[0, 2] == 1
        assert right[0, 3] == 1
        assert right[2, 2] == 1
        assert right.sum().item() == 3

        # Foreground (uid 10): pixels (1,2), (1,3), (2,3)
        assert fg[1, 2] == 1
        assert fg[1, 3] == 1
        assert fg[2, 3] == 1
        assert fg.sum().item() == 3

        # Background: 7 pixels with uid 0
        assert bg.sum().item() == 7

    def test_background_is_negation_of_foreground_and_robot(self):
        """Test that background == ~(left | right | foreground)."""
        env, obs = self._make_env_and_obs(num_envs=2)

        result = compute_semantic_mask(
            env,
            obs,
            entity_cfg=MagicMock(uid="camera"),
            foreground_uids=["fg_object"],
        )

        bg = result[0, :, :, 0]
        fg = result[0, :, :, 1]
        left = result[0, :, :, 2]
        right = result[0, :, :, 3]

        expected_bg = ~(left.bool() | right.bool() | fg.bool())
        torch.testing.assert_close(bg.bool(), expected_bg)

    def test_with_foreground_uids_not_in_assets(self):
        """Test that foreground UIDs not in asset_uids are silently ignored."""
        env, obs = self._make_env_and_obs(num_envs=2)

        result = compute_semantic_mask(
            env,
            obs,
            entity_cfg=MagicMock(uid="camera"),
            foreground_uids=["fg_object", "nonexistent_object"],
        )

        # Foreground should still only match uid 10 (from fg_object)
        fg = result[0, :, :, 1]
        assert fg[1, 2] == 1
        assert fg[1, 3] == 1
        assert fg[2, 3] == 1
        assert fg.sum().item() == 3

    def test_different_num_envs(self):
        """Test that compute_semantic_mask works with different batch sizes."""
        num_envs = 5
        env, obs = self._make_env_and_obs(num_envs=num_envs)

        result = compute_semantic_mask(
            env,
            obs,
            entity_cfg=MagicMock(uid="camera"),
            foreground_uids=["fg_object"],
        )

        assert result.shape == (num_envs, 4, 4, 4)
        # All envs have identical mask data, so results should match
        for i in range(1, num_envs):
            assert result[i].equal(result[0])
