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

"""Tests for the declarative repeated pick/place reference task."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.expert_program import (
    CyclicPoseTargetCfg,
    ObjectNearTargetValidatorCfg,
    PickCfg,
    PlaceCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
)
from embodichain.lab.gym.envs.managers.randomization.spatial import (
    randomize_rigid_object_pose,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.expert_program import (  # noqa: E402
    ExpertProgramRepeatedPickPlaceEnv,
)
from embodichain_tasks.expert_program._common import (  # noqa: E402
    create_ur5_skill_profile_binding,
)

_REPOSITORY_ROOT = Path(__file__).parents[4]


class TestExpertProgramRepeatedPickPlaceEnv:
    """Registration, program structure, and integration tests."""

    def test_registered_as_a_separate_reference_environment(self) -> None:
        """Only the canonical Expert Program Gym ID is registered."""
        from embodichain_tasks.expert_program import __all__

        assert "ExpertProgramRepeatedPickPlaceEnv" in __all__
        spec = REGISTERED_ENVS["ExpertProgramRepeatedPickPlace-v1"]
        assert spec.cls is ExpertProgramRepeatedPickPlaceEnv
        assert spec.max_episode_steps == 1200
        assert issubclass(ExpertProgramRepeatedPickPlaceEnv, EmbodiedEnv)
        assert "MultiSegmentsCubePickPlace-v1" not in REGISTERED_ENVS

    def test_gym_config_loads_the_declarative_three_cycle_program(self) -> None:
        """The runnable config resolves a strict source-relative program."""
        config_path = (
            _REPOSITORY_ROOT
            / "embodichain_tasks/configs/gym/expert_program/repeated_pick_place.json"
        )
        config = json.loads(config_path.read_text())

        assert config["id"] == "ExpertProgramRepeatedPickPlace-v1"
        assert config["expert_program_path"] == (
            "../../expert_program/repeated_pick_place.yaml"
        )
        assert config["sensor"] == []
        assert config["env"]["dataset"]["lerobot"]["params"]["save_path"] == (
            "outputs/lerobot/expert_program"
        )
        assert "extensions" not in config["env"]

        cfg = config_to_cfg(config, source_path=config_path)
        program = cfg.expert_program
        randomize_cube_pose = cfg.events.randomize_cube_pose

        assert program is not None
        assert program.program_id == "repeated_cube_pick_place"
        assert type(program.program) is RepeatCfg
        assert program.program.count == 3
        assert type(program.program.body) is SegmentCfg
        assert type(program.program.body.steps) is SequenceCfg
        first, second = program.program.body.steps.items
        assert type(first.call) is PickCfg
        assert type(second.call) is PlaceCfg
        assert randomize_cube_pose.func is randomize_rigid_object_pose
        assert randomize_cube_pose.mode == "reset"
        assert randomize_cube_pose.params["entity_cfg"].uid == "cube"
        assert randomize_cube_pose.params["position_range"] == [
            [-0.02, -0.02, 0.0],
            [0.02, 0.02, 0.0],
        ]
        assert randomize_cube_pose.params["rotation_range"] == [
            [0.0, 0.0, -10.0],
            [0.0, 0.0, 10.0],
        ]
        assert randomize_cube_pose.params["relative_position"] is True
        assert randomize_cube_pose.params["relative_rotation"] is True
        assert first.call.object == second.call.object == "cube"
        assert second.call.at is not None
        assert second.call.at.target == "drop_pose"
        assert program.program.body.post[0].entity == "cube"
        validator = program.program.body.validators[0]
        assert type(validator) is ObjectNearTargetValidatorCfg
        assert validator.object == "cube"
        assert validator.target == "drop_pose"
        assert validator.position_tolerance == 0.12
        target = program.targets["drop_pose"]
        assert type(target) is CyclicPoseTargetCfg
        assert tuple(pose.position for pose in target.values) == (
            (-0.40, 0.48, 0.10),
            (-0.42, -0.08, 0.10),
        )

    def test_ur5_profile_tolerates_normal_simulated_joint_tracking_lag(self) -> None:
        """The runtime does not replan for the observed 0.05-radian lag."""
        robot = SimpleNamespace(
            device=torch.device("cpu"),
            get_qpos_limits=lambda *, name: torch.tensor(
                [[[0.0, 0.08]]], dtype=torch.float32
            ),
        )

        binding = create_ur5_skill_profile_binding(
            robot,
            profile_id="test_ur5_profile",
            sample_count=120,
            skill_ids=("pick_up", "place"),
        )

        assert binding.presets[0].recovery_policy.tracking_error_threshold == 0.1

    def test_task_has_no_task_local_constraint_observer(self) -> None:
        """The reference task does not own contact-based effect evidence."""
        assert "_observe_grasp_constraint" not in (
            ExpertProgramRepeatedPickPlaceEnv.__dict__
        )

    def test_task_does_not_reimplement_program_acceptance(self) -> None:
        """The standard bridge validator is the sole task-success boundary."""
        assert "is_task_success" not in ExpertProgramRepeatedPickPlaceEnv.__dict__
