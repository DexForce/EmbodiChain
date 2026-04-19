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

import torch
import pytest

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    GraspPose,
    InteractionPoints,
    ObjectSemantics,
    ActionCfg,
    register_action,
    unregister_action,
    get_registered_actions,
)


class TestAffordance:
    """Test affordance base class and subclasses."""

    def test_affordance_base(self):
        """Test base affordance class."""
        aff = Affordance(object_label="test_object")
        assert aff.object_label == "test_object"
        assert aff.get_batch_size() == 1

    def test_grasp_pose_default(self):
        """Test GraspPose with default values."""
        grasp = GraspPose(object_label="bottle")
        assert grasp.object_label == "bottle"
        assert grasp.poses.shape == (1, 4, 4)
        assert grasp.grasp_types == ["default"]
        assert grasp.get_batch_size() == 1

    def test_grasp_pose_multiple(self):
        """Test GraspPose with multiple poses."""
        poses = torch.stack(
            [
                torch.eye(4),
                torch.eye(4),
                torch.eye(4),
            ]
        )
        grasp = GraspPose(
            object_label="bottle",
            poses=poses,
            grasp_types=["pinch", "power", "hook"],
        )
        assert grasp.get_batch_size() == 3

        # Test get_grasp_by_type
        pinch_pose = grasp.get_grasp_by_type("pinch")
        assert pinch_pose is not None
        assert torch.allclose(pinch_pose, torch.eye(4))

        nonexistent = grasp.get_grasp_by_type("nonexistent")
        assert nonexistent is None

    def test_grasp_pose_best_grasp(self):
        """Test get_best_grasp method."""
        poses = torch.stack(
            [
                torch.eye(4),
                torch.eye(4) * 2,
            ]
        )
        confidence = torch.tensor([0.7, 0.9])
        grasp = GraspPose(
            poses=poses,
            grasp_types=["low_conf", "high_conf"],
            confidence_scores=confidence,
        )

        best = grasp.get_best_grasp()
        # Should return the second pose (higher confidence)
        assert torch.allclose(best, poses[1])

    def test_interaction_points(self):
        """Test InteractionPoints class."""
        points = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ]
        )
        normals = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        interaction = InteractionPoints(
            object_label="cube",
            points=points,
            normals=normals,
            point_types=["push", "poke", "touch"],
        )

        assert interaction.get_batch_size() == 3

        # Test get_points_by_type
        push_points = interaction.get_points_by_type("push")
        assert push_points is not None
        assert torch.allclose(push_points, points[0:1])

        nonexistent = interaction.get_points_by_type("nonexistent")
        assert nonexistent is None

        # Test get_approach_direction
        approach = interaction.get_approach_direction(0)
        assert torch.allclose(approach, torch.tensor([-1.0, 0.0, 0.0]))

    def test_interaction_points_no_normals(self):
        """Test InteractionPoints without normals."""
        points = torch.tensor([[0.1, 0.2, 0.3]])
        interaction = InteractionPoints(points=points)

        # Default approach direction should be +z
        approach = interaction.get_approach_direction(0)
        assert torch.allclose(approach, torch.tensor([0.0, 0.0, 1.0]))


class TestObjectSemantics:
    """Test ObjectSemantics dataclass."""

    def test_basic_creation(self):
        """Test basic ObjectSemantics creation."""
        affordance = GraspPose(object_label="bottle")
        semantics = ObjectSemantics(
            label="bottle",
            affordance=affordance,
            geometry={"bounding_box": [0.1, 0.2, 0.3]},
            properties={"mass": 0.5, "friction": 0.8},
            uid="bottle_001",
        )

        assert semantics.label == "bottle"
        assert semantics.uid == "bottle_001"
        assert semantics.affordance.object_label == "bottle"
        assert semantics.properties["mass"] == 0.5

    def test_no_uid(self):
        """Test ObjectSemantics without UID."""
        affordance = GraspPose()
        semantics = ObjectSemantics(
            label="apple",
            affordance=affordance,
            geometry={},
            properties={},
        )

        assert semantics.uid is None


class TestActionCfg:
    """Test ActionCfg dataclass."""

    def test_defaults(self):
        """Test ActionCfg default values."""
        cfg = ActionCfg()
        assert cfg.control_part == "left_arm"
        assert cfg.interpolation_type == "linear"
        assert cfg.velocity_limit is None
        assert cfg.acceleration_limit is None

    def test_custom_values(self):
        """Test ActionCfg with custom values."""
        cfg = ActionCfg(
            control_part="right_arm",
            interpolation_type="toppra",
            velocity_limit=0.5,
            acceleration_limit=1.0,
        )
        assert cfg.control_part == "right_arm"
        assert cfg.interpolation_type == "toppra"
        assert cfg.velocity_limit == 0.5
        assert cfg.acceleration_limit == 1.0


class TestActionRegistry:
    """Test action registry functions."""

    def test_register_and_unregister(self):
        """Test registering and unregistering actions."""
        from embodichain.lab.sim.atomic_actions import AtomicAction

        class TestAction(AtomicAction):
            def execute(self, target, **kwargs):
                return PlanResult(success=True)

            def validate(self, target, **kwargs):
                return True

        # Register
        register_action("test", TestAction)
        assert "test" in get_registered_actions()

        # Unregister
        unregister_action("test")
        assert "test" not in get_registered_actions()

    def test_get_registered_actions_copy(self):
        """Test that get_registered_actions returns a copy."""
        from embodichain.lab.sim.atomic_actions import AtomicAction

        initial = get_registered_actions()

        class DummyAction(AtomicAction):
            def execute(self, target, **kwargs):
                return PlanResult(success=True)

            def validate(self, target, **kwargs):
                return True

        register_action("dummy", DummyAction)

        # Original should not contain the new action
        assert "dummy" not in initial

        # Cleanup
        unregister_action("dummy")
