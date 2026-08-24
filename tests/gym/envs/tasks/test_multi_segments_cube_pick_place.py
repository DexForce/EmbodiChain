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

"""Tests for the lazy multi-segment cube pick-and-place task."""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.lab.sim.robots import URRobotCfg

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.multi_segments.cube_pick_place import (  # noqa: E402
    DEFAULT_GRIPPER_CLOSE_QPOS,
    MultiSegmentsCubePickPlaceEnv,
)


class TestMultiSegmentsCubePickPlaceEnv:
    """Registration, config, and lazy-planning tests."""

    def test_registered_and_exported(self) -> None:
        """The new task category exports a registered environment."""
        from embodichain_tasks.multi_segments import __all__

        assert "MultiSegmentsCubePickPlaceEnv" in __all__
        spec = REGISTERED_ENVS["MultiSegmentsCubePickPlace-v1"]
        assert spec.cls is MultiSegmentsCubePickPlaceEnv
        assert spec.max_episode_steps == 1200
        assert issubclass(MultiSegmentsCubePickPlaceEnv, EmbodiedEnv)

    def test_gym_config_targets_the_registered_task(self) -> None:
        """The runnable gym config selects the task and three cycles."""
        config_path = (
            Path(__file__).parents[4]
            / "embodichain_tasks/configs/gym/multi_segments/cube_pick_place.json"
        )
        config = json.loads(config_path.read_text())

        assert config["id"] == "MultiSegmentsCubePickPlace-v1"
        assert config["env"]["extensions"]["num_cycles"] == 3
        assert config["env"]["extensions"]["grasp_hold_steps"] == 45
        assert "grasp_samples" not in config["env"]["extensions"]
        assert "force_reannotate" not in config["env"]["extensions"]
        assert len(config["env"]["extensions"]["place_positions"]) == 2
        assert config["rigid_object"][0]["uid"] == "cube"
        assert config["robot"]["class_type"] == "URRobot"
        assert config["robot"]["robot_type"] == "ur5"
        recorder = config["env"]["dataset"]["lerobot"]
        assert recorder["func"] == "LeRobotRecorder"
        assert recorder["params"]["robot_meta"] == {
            "robot_type": "UR5",
            "control_freq": 25,
        }
        assert recorder["params"]["save_path"] == "outputs/lerobot/multi_segments"

        cfg = config_to_cfg(config)

        assert isinstance(cfg.robot, URRobotCfg)
        assert cfg.robot.robot_type == "ur5"
        assert cfg.robot.control_parts["arm"] == [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        assert cfg.robot.solver_cfg["arm"].ur_type == "ur5"
        assert cfg.robot.solver_cfg["arm"].d1 == 0.089159

    def test_segments_are_planned_lazily_from_updated_scene(self) -> None:
        """Requesting the next segment observes the post-execution cube pose."""
        env = object.__new__(MultiSegmentsCubePickPlaceEnv)
        env.num_cycles = 3
        env.place_positions = ((1.0, 0.0, 0.1), (2.0, 0.0, 0.1))
        env._completed_cycles = 0
        env._last_target_position = None
        env.sim = SimpleNamespace(device=torch.device("cpu"))
        env._scene_position_for_test = 0.0
        env._planned_positions_for_test = []

        def fake_plan(
            self: MultiSegmentsCubePickPlaceEnv, target_position: torch.Tensor
        ):
            source_pose = torch.eye(4).unsqueeze(0)
            source_pose[:, 0, 3] = self._scene_position_for_test
            self._planned_positions_for_test.append(self._scene_position_for_test)
            action = torch.tensor([[self._scene_position_for_test]])
            return torch.ones(1, dtype=torch.bool), (action,), source_pose

        env._plan_pick_place_cycle = MethodType(fake_plan, env)
        segments = iter(env.create_demo_segments())

        first = next(segments)
        assert env._planned_positions_for_test == [0.0]
        assert first.metadata["planned_source_poses"][0][0][3] == 0.0

        # In the real executor the first segment actions run while the outer
        # generator is suspended. Emulate the resulting free-fall displacement.
        list(first.actions)
        env._scene_position_for_test = 0.17
        second = next(segments)
        assert env._planned_positions_for_test == [0.0, 0.17]
        assert second.metadata["planned_source_poses"][0][0][3] == pytest.approx(0.17)

        env._scene_position_for_test = -0.04
        third = next(segments)
        assert env._planned_positions_for_test == [0.0, 0.17, -0.04]
        assert third.metadata["target_position"] == pytest.approx([1.0, 0.0, 0.1])

        list(third.actions)
        try:
            next(segments)
        except StopIteration:
            pass
        else:
            raise AssertionError("Expected exactly three demo segments.")
        assert env._completed_cycles == 3

    def test_invalid_positions_are_rejected(self) -> None:
        """Every configured placement target must be an XYZ position."""
        with pytest.raises(ValueError, match="XYZ"):
            MultiSegmentsCubePickPlaceEnv._validate_place_positions([(1.0, 2.0)])

    def test_grasp_hold_is_inserted_before_lift(self) -> None:
        """The closed grasp waypoint is held before the pickup lift starts."""
        env = object.__new__(MultiSegmentsCubePickPlaceEnv)
        env.grasp_hold_steps = 2
        trajectory = torch.arange(120, dtype=torch.float32).reshape(1, 120, 1)

        augmented, clear_step = env._insert_grasp_hold(trajectory)

        assert augmented.shape == (1, 122, 1)
        assert clear_step == 78
        assert augmented[0, 75, 0] == 75
        assert augmented[0, 76:78, 0].tolist() == [75, 75]
        assert augmented[0, 78, 0] == 76

    def test_cycle_uses_grasp_service_and_motion_generator_directly(self) -> None:
        """The non-atomic example composes the two planning services itself."""

        class StubCube:
            def __init__(self) -> None:
                self.clear_dynamics_calls = 0

            def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
                assert to_matrix is True
                return torch.eye(4, dtype=torch.float32).unsqueeze(0)

            def clear_dynamics(self) -> None:
                self.clear_dynamics_calls += 1

        class StubRobot:
            def get_qpos(self) -> torch.Tensor:
                return torch.zeros(1, 8, dtype=torch.float32)

        class StubGraspPoseGenerator:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def get_best_grasp_poses(self, **kwargs: object):
                self.calls.append(kwargs)
                object_poses = kwargs["obj_poses"]
                assert isinstance(object_poses, torch.Tensor)
                return (
                    torch.ones(1, dtype=torch.bool),
                    object_poses.clone(),
                    torch.full((1,), 0.04, dtype=torch.float32),
                )

        class StubMotionGenerator:
            def __init__(self) -> None:
                self.calls: list[tuple[list[object], object]] = []

            def generate(self, target_states: list[object], options: object):
                self.calls.append((target_states, options))
                start_qpos = options.start_qpos
                sample_count = options.sample_count
                assert isinstance(start_qpos, torch.Tensor)
                assert isinstance(sample_count, int)
                return SimpleNamespace(
                    success=torch.ones(1, dtype=torch.bool),
                    positions=start_qpos.unsqueeze(1).repeat(1, sample_count, 1),
                )

        env = object.__new__(MultiSegmentsCubePickPlaceEnv)
        env.sim = SimpleNamespace(device=torch.device("cpu"))
        env._num_envs = 1
        env.robot = StubRobot()
        env._cube = StubCube()
        env._grasp_pose_generator = StubGraspPoseGenerator()
        env._motion_generator = StubMotionGenerator()
        env._arm_joint_ids = tuple(range(6))
        env._hand_joint_ids = (6, 7)
        env._hand_open_qpos = torch.zeros(1, 2)
        env._hand_close_qpos = torch.full((1, 2), DEFAULT_GRIPPER_CLOSE_QPOS)
        env._cube_mesh_vertices = torch.zeros(3, 3)
        env._cube_mesh_triangles = torch.tensor([[0, 1, 2]], dtype=torch.long)
        env.grasp_hold_steps = 0
        env.settle_max_steps = 0

        success, actions, _ = env._plan_pick_place_cycle(
            torch.tensor([0.3, -0.2, 0.1], dtype=torch.float32)
        )
        planned_actions = list(actions)

        assert success.tolist() == [True]
        assert len(env._grasp_pose_generator.calls) == 1
        assert len(env._motion_generator.calls) == 4
        assert [len(states) for states, _ in env._motion_generator.calls] == [
            2,
            1,
            2,
            1,
        ]
        assert len(planned_actions) == (
            env.PICK_SAMPLE_INTERVAL + env.PLACE_SAMPLE_INTERVAL
        )
        assert env._cube.clear_dynamics_calls == 1
        assert not hasattr(env, "_action_engine")
