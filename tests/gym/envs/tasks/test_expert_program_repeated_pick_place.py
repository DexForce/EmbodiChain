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
from embodichain.lab.sim.atomic_actions import TrackingPolicy
from embodichain.lab.sim.skills import (
    BinaryEffectClause,
    BinaryEffectEvidenceQuery,
    BinaryEvidenceKind,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEvidenceAddress,
    EffectEvidenceCollectionContext,
    EffectEvidenceSourceRef,
    HeldObjectRelation,
    HeldObjectStateExpectation,
)

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.expert_program import (  # noqa: E402
    ExpertProgramRepeatedPickPlaceEnv,
)
from embodichain_tasks.expert_program._common import (  # noqa: E402
    GRIPPER_FINGER_LINKS,
    create_ur5_skill_profile_binding,
)

_REPOSITORY_ROOT = Path(__file__).parents[4]
_EXPECTED_PGI_FINGER_LINKS = (
    "gripper_finger1_link_1",
    "gripper_finger2_link_1",
)


class _ContactSensorStub:
    """Return one owned contact snapshot for evidence tests."""

    def __init__(self, user_ids: torch.Tensor, is_valid: torch.Tensor) -> None:
        self._data = {"user_ids": user_ids, "is_valid": is_valid}
        self.update_count = 0

    def update(self) -> None:
        """Record that the observation refreshed the physical sensor."""
        self.update_count += 1

    def get_data(self) -> dict[str, torch.Tensor]:
        """Return the configured contact tensors."""
        return self._data


def _grasp_constraint_query() -> BinaryEffectEvidenceQuery:
    """Build the exact constraint query emitted by Pick/Place monitoring."""
    expectation = HeldObjectStateExpectation(
        "held",
        HeldObjectRelation.ATTACHED,
        "cube",
        "primary",
        "manipulator",
        "manipulator",
    )
    source = EffectEvidenceSourceRef(
        CONTROL_PART_EVIDENCE_PROVIDER_ID,
        CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
        ControlPartEvidenceAddress("hand", CONSTRAINT_EFFECT_CHANNEL),
    )
    return BinaryEffectEvidenceQuery(
        BinaryEffectClause(
            "constraint",
            "held",
            source,
            BinaryEvidenceKind.CONSTRAINT,
            True,
        ),
        expectation,
    )


class TestExpertProgramRepeatedPickPlaceEnv:
    """Registration, program structure, and physical evidence tests."""

    def test_registered_as_a_separate_reference_environment(self) -> None:
        """The Expert Program task coexists with the imperative comparison."""
        from embodichain_tasks.expert_program import __all__

        assert "ExpertProgramRepeatedPickPlaceEnv" in __all__
        spec = REGISTERED_ENVS["ExpertProgramRepeatedPickPlace-v1"]
        assert spec.cls is ExpertProgramRepeatedPickPlaceEnv
        assert spec.max_episode_steps == 1200
        assert issubclass(ExpertProgramRepeatedPickPlaceEnv, EmbodiedEnv)
        assert "MultiSegmentsCubePickPlace-v1" in REGISTERED_ENVS

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
        assert config["sensor"][0]["sensor_type"] == "ContactSensor"
        assert "extensions" not in config["env"]
        configured_links = config["sensor"][0]["articulation_cfg_list"][0][
            "link_name_list"
        ]
        assert GRIPPER_FINGER_LINKS == _EXPECTED_PGI_FINGER_LINKS
        assert configured_links == list(_EXPECTED_PGI_FINGER_LINKS)

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

        assert binding.presets[0].tracking_policy == TrackingPolicy.joint_position(
            in_flight_max_abs_error=0.1,
            terminal_max_abs_error=0.1,
        )

    def test_constraint_observer_requires_both_finger_contacts(self) -> None:
        """A logical grasp is never inferred from commands or one finger."""
        env = object.__new__(ExpertProgramRepeatedPickPlaceEnv)
        env.sim = SimpleNamespace(device=torch.device("cpu"))
        env._num_envs = 2
        env._cube_user_ids = torch.tensor([[10], [20]])
        env._finger_user_ids = (
            torch.tensor([[11], [21]]),
            torch.tensor([[12], [22]]),
        )
        sensor = _ContactSensorStub(
            user_ids=torch.tensor(
                [
                    [[10, 11], [12, 10], [0, 0]],
                    [[20, 21], [20, 99], [0, 0]],
                ]
            ),
            is_valid=torch.tensor(
                [[True, True, False], [True, True, False]],
            ),
        )
        env._contact_sensor = sensor

        observation = env._observe_grasp_constraint(
            _grasp_constraint_query(),
            EffectEvidenceCollectionContext(
                timestamp=1.0,
                observation_revision=2,
                env_ids=torch.tensor([0, 1], dtype=torch.long),
            ),
        )

        assert sensor.update_count == 1
        assert observation.values.tolist() == [True, False]
        assert observation.valid is not None
        assert observation.valid.tolist() == [True, True]

    def test_task_does_not_reimplement_program_acceptance(self) -> None:
        """The standard bridge validator is the sole task-success boundary."""
        assert "is_task_success" not in ExpertProgramRepeatedPickPlaceEnv.__dict__
