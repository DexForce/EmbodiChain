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


class MockArticulationEntity:
    """Mock dexsim articulation entity with per-link mass support."""

    def __init__(self, link_names: list[str], default_mass: float = 1.0):
        self._link_masses: dict[str, float] = {
            name: default_mass for name in link_names
        }

    def get_mass(self, link_name: str | None = None) -> tuple[int, dict[str, float]]:
        if link_name is not None:
            return 0, {link_name: self._link_masses.get(link_name, 0.0)}
        return 0, dict(self._link_masses)

    def set_mass(self, link_name: str, mass: float) -> int:
        self._link_masses[link_name] = mass
        return 0


class MockArticulation:
    """Mock articulation for event functor tests."""

    def __init__(
        self,
        uid: str = "test_articulation",
        num_envs: int = 4,
        link_names: list[str] | None = None,
    ):
        self.uid = uid
        self.num_envs = num_envs
        self.device = torch.device("cpu")

        # Link names for the articulation
        self.link_names: list[str] = link_names or [
            "base_link",
            "link_0",
            "link_1",
            "link_2",
        ]

        # Per-environment entities with mass tracking
        self._entities = [
            MockArticulationEntity(self.link_names) for _ in range(num_envs)
        ]

        # Default pose at origin (position + quaternion)
        # Format: (N, 7) - position (3) + quaternion (4)
        self._pose = torch.zeros(num_envs, 7)
        self._pose[:, 3] = 1.0  # quaternion w = 1 (identity rotation)

        self.default_link_masses = torch.ones(
            (self.num_envs, len(self.link_names)), device=self.device
        )

    def _matrix_to_quat_pos(self, pose_matrix):
        """Convert 4x4 matrix to (position, quaternion) format.

        Args:
            pose_matrix: (N, 4, 4) transformation matrix

        Returns:
            (N, 7) tensor with position (3) + quaternion (4)
        """
        # Extract position
        pos = pose_matrix[:, :3, 3]  # (N, 3)

        # Extract rotation matrix and convert to quaternion
        rot = pose_matrix[:, :3, :3]  # (N, 3, 3)

        # Simple quaternion from rotation matrix
        # This is a simplified conversion - not full Davenport q
        quat = torch.zeros(pose_matrix.shape[0], 4)
        quat[:, 3] = 1.0  # default to identity

        # Check if rotation is close to identity
        for i in range(pose_matrix.shape[0]):
            r = rot[i]
            # Trace of rotation matrix
            trace = r[0, 0] + r[1, 1] + r[2, 2]
            if trace > 0:
                quat[i, 3] = (trace + 1.0) ** 0.5 / 2.0
                quat[i, 0] = (r[2, 1] - r[1, 2]) / (4 * quat[i, 3])
                quat[i, 1] = (r[0, 2] - r[2, 0]) / (4 * quat[i, 3])
                quat[i, 2] = (r[1, 0] - r[0, 1]) / (4 * quat[i, 3])

        # Normalize quaternion
        quat = quat / quat.norm(dim=1, keepdim=True)

        return torch.cat([pos, quat], dim=1)

    def get_local_pose(self, to_matrix: bool = False):
        """Returns pose in (N, 7) format: position (3) + quaternion (4)."""
        return self._pose

    def set_local_pose(self, pose, env_ids=None):
        """Set pose from 4x4 matrix or (N, 7) format."""
        if pose.dim() == 3:
            # 4x4 matrix format - convert to (N, 7)
            pose = self._matrix_to_quat_pos(pose)

        if env_ids is not None:
            self._pose[env_ids] = pose[env_ids] if pose.dim() > 1 else pose
        else:
            self._pose = pose

    def clear_dynamics(self, env_ids=None):
        """Clear dynamics - no-op for mock."""
        pass

    def get_mass(self, link_names=None, env_ids=None):
        """Get mass of links, matching Articulation API."""
        local_env_ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        if link_names is None:
            link_names = self.link_names
        mass_tensor = torch.zeros(
            (len(local_env_ids), len(link_names)), dtype=torch.float32
        )
        for i, env_idx in enumerate(local_env_ids):
            for j, name in enumerate(link_names):
                mass_tensor[i, j] = self._entities[env_idx]._link_masses[name]
        return mass_tensor

    def set_mass(self, mass, link_names, env_ids=None):
        """Set mass of links, matching Articulation API."""
        local_env_ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        for i, env_idx in enumerate(local_env_ids):
            for j, name in enumerate(link_names):
                self._entities[env_idx]._link_masses[name] = mass[i, j].item()


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
    randomize_articulation_mass,
)
from embodichain.lab.gym.envs.managers.randomization.spatial import (
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


class TestRandomizeArticulationMass:
    """Tests for randomize_articulation_mass functor."""

    def test_sets_all_links_mass_in_range(self):
        """Test that mass is randomized within range for all links."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])
        mass_range = (0.5, 2.0)

        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            mass_range=mass_range,
        )

        # Check all links in all envs are in range using get_mass
        masses = env.test_articulation.get_mass(env_ids=env_ids)
        assert torch.all(masses >= 0.5)
        assert torch.all(masses <= 2.0)

    def test_sets_specific_link_with_regex(self):
        """Test that only matched links are randomized using regex."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Set base_link mass to a known value
        base_mass = torch.full((4, 1), 10.0)
        env.test_articulation.set_mass(
            base_mass, link_names=["base_link"], env_ids=env_ids
        )

        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            mass_range=(0.5, 2.0),
            link_names="link_.*",
        )

        # base_link should not be changed
        base_masses = env.test_articulation.get_mass(
            link_names=["base_link"], env_ids=env_ids
        )
        assert torch.all(base_masses == 10.0), "base_link should not be changed"

        # link_0, link_1, link_2 should be randomized
        link_masses = env.test_articulation.get_mass(
            link_names=["link_0", "link_1", "link_2"], env_ids=env_ids
        )
        assert torch.all(link_masses >= 0.5)
        assert torch.all(link_masses <= 2.0)

    def test_sets_specific_link_with_list(self):
        """Test that only matched links are randomized using a list of patterns."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Set link_2 and base_link mass to known values
        known_mass = torch.full((4, 2), 10.0)
        env.test_articulation.set_mass(
            known_mass, link_names=["link_2", "base_link"], env_ids=env_ids
        )

        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            mass_range=(0.5, 2.0),
            link_names=["link_0", "link_1"],
        )

        # link_2 and base_link should not be changed
        unchanged = env.test_articulation.get_mass(
            link_names=["link_2", "base_link"], env_ids=env_ids
        )
        assert torch.all(unchanged == 10.0)

        # link_0 and link_1 should be randomized
        randomized = env.test_articulation.get_mass(
            link_names=["link_0", "link_1"], env_ids=env_ids
        )
        assert torch.all(randomized >= 0.5)
        assert torch.all(randomized <= 2.0)

    def test_relative_mass_randomization(self):
        """Test relative mass randomization adds to current mass."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Initial mass is 1.0 for all links (default)
        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            mass_range=(-0.5, 0.5),
            link_names="base_link",
            relative=True,
        )

        # Final mass for base_link should be in [0.5, 1.5]
        masses = env.test_articulation.get_mass(
            link_names=["base_link"], env_ids=env_ids
        )
        assert torch.all(masses >= 0.5)
        assert torch.all(masses <= 1.5)

    def test_handles_nonexistent_articulation(self):
        """Test that function handles non-existent articulation gracefully."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Should not raise - function returns early for non-existent articulations
        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="nonexistent"),
            mass_range=(0.5, 2.0),
        )

    def test_dict_mass_range_per_link(self):
        """Test per-link mass ranges using dict-based mass_range."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Set link_2 and base_link mass to known values
        known_mass = torch.full((4, 2), 10.0)
        env.test_articulation.set_mass(
            known_mass, link_names=["link_2", "base_link"], env_ids=env_ids
        )

        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            mass_range={
                "link_0": (0.5, 1.0),
                "link_1": (2.0, 3.0),
            },
        )

        # link_2 and base_link should not be changed
        unchanged = env.test_articulation.get_mass(
            link_names=["link_2", "base_link"], env_ids=env_ids
        )
        assert torch.all(unchanged == 10.0)

        # link_0 should be in [0.5, 1.0]
        link_0_masses = env.test_articulation.get_mass(
            link_names=["link_0"], env_ids=env_ids
        )
        assert torch.all(link_0_masses >= 0.5)
        assert torch.all(link_0_masses <= 1.0)

        # link_1 should be in [2.0, 3.0]
        link_1_masses = env.test_articulation.get_mass(
            link_names=["link_1"], env_ids=env_ids
        )
        assert torch.all(link_1_masses >= 2.0)
        assert torch.all(link_1_masses <= 3.0)

    def test_dict_mass_range_ignores_link_names(self):
        """Test that link_names is ignored when mass_range is a dict."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Set base_link mass to a known value
        base_mass = torch.full((4, 1), 10.0)
        env.test_articulation.set_mass(
            base_mass, link_names=["base_link"], env_ids=env_ids
        )

        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            mass_range={"link_0": (0.5, 1.0)},
            link_names="base_link",  # should be ignored
        )

        # base_link should not be changed
        base_masses = env.test_articulation.get_mass(
            link_names=["base_link"], env_ids=env_ids
        )
        assert torch.all(base_masses == 10.0)

        # link_0 should be randomized
        link_0_masses = env.test_articulation.get_mass(
            link_names=["link_0"], env_ids=env_ids
        )
        assert torch.all(link_0_masses >= 0.5)
        assert torch.all(link_0_masses <= 1.0)

    def test_dict_mass_range_relative(self):
        """Test relative randomization with per-link mass ranges."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        # Initial mass is 1.0 for all links (default)
        randomize_articulation_mass(
            env,
            env_ids,
            entity_cfg=MagicMock(uid="articulation"),
            mass_range={
                "base_link": (-0.5, 0.5),
                "link_0": (-0.2, 0.2),
            },
            relative=True,
        )

        # base_link should be in [0.5, 1.5]
        base_masses = env.test_articulation.get_mass(
            link_names=["base_link"], env_ids=env_ids
        )
        assert torch.all(base_masses >= 0.5)
        assert torch.all(base_masses <= 1.5)

        # link_0 should be in [0.8, 1.2]
        link_0_masses = env.test_articulation.get_mass(
            link_names=["link_0"], env_ids=env_ids
        )
        assert torch.all(link_0_masses >= 0.8)
        assert torch.all(link_0_masses <= 1.2)

    def test_handles_nonexistent_link_pattern(self):
        """Test that function raises for non-matching link patterns."""
        env = MockEnv(num_envs=4)
        env_ids = torch.tensor([0, 1, 2, 3])

        with pytest.raises(ValueError, match="Not all regular expressions"):
            randomize_articulation_mass(
                env,
                env_ids,
                entity_cfg=MagicMock(uid="articulation"),
                mass_range=(0.5, 2.0),
                link_names="nonexistent_link",
            )
