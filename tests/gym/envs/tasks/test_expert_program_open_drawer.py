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

"""Tests for the Expert Program Open Drawer reference task."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import embodichain.data as data_module
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.expert_program import (
    ArticulationJointPositionValidatorCfg,
    RegisteredSemanticCallCfg,
    SegmentCfg,
)
from embodichain.lab.gym.envs.managers.randomization.spatial import (
    randomize_articulation_root_pose,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.lab.sim.atomic_actions import (
    ObjectSemantics,
    SceneEntityPose,
    SlideAffordance,
    SlideGoal,
    SlideOptions,
)
from embodichain.lab.sim.skills import RegisteredSemanticCall

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.expert_program import ExpertProgramOpenDrawerEnv  # noqa: E402
from embodichain_tasks.expert_program.open_drawer import (  # noqa: E402
    HANDLE_ENTITY_ID,
    OPEN_DRAWER_CALL_ID,
    _OpenDrawerSlideLowerer,
    _open_drawer_call_catalog,
)

_REPOSITORY_ROOT = Path(__file__).parents[4]


def _slide_semantics() -> ObjectSemantics:
    """Create minimal task-owned drawer-handle semantics."""
    return ObjectSemantics(
        label="drawer_handle",
        entity_id=HANDLE_ENTITY_ID,
        geometry={},
        affordance=SlideAffordance(
            mesh_vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
            ),
            mesh_triangles=torch.tensor([[0, 1, 2]], dtype=torch.long),
            translation_axis=torch.tensor([0.0, 1.0, 0.0]),
            joint_name="drawer_slide",
            joint_limits=(0.0, 0.25),
        ),
    )


def _open_drawer_call(**overrides: object) -> RegisteredSemanticCall:
    """Build the task's exact safe registered-call payload."""
    arguments: dict[str, object] = {
        "handle": HANDLE_ENTITY_ID,
        "direction": "pull",
        "hand_interp_steps": 12,
        "approach_distance": 0.10,
        "translation_distance": 0.18,
    }
    arguments.update(overrides)
    return RegisteredSemanticCall(
        call_id=OPEN_DRAWER_CALL_ID,
        arguments=arguments,
        resources={"primary": "manipulator"},
    )


class TestExpertProgramOpenDrawerEnv:
    """Registration, registered lowering, and success-boundary tests."""

    def test_registered_as_a_separate_reference_environment(self) -> None:
        """The canonical Drawer environment is the only exported integration."""
        from embodichain_tasks.expert_program import __all__

        assert "ExpertProgramOpenDrawerEnv" in __all__
        spec = REGISTERED_ENVS["ExpertProgramOpenDrawer-v1"]
        assert spec.cls is ExpertProgramOpenDrawerEnv
        assert spec.max_episode_steps == 600
        assert issubclass(ExpertProgramOpenDrawerEnv, EmbodiedEnv)

    def test_gym_config_loads_the_slide_open_drawer_program(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The runnable config resolves its strict registered-call program."""
        config_path = (
            _REPOSITORY_ROOT
            / "embodichain_tasks/configs/gym/expert_program/open_drawer.json"
        )
        config = json.loads(config_path.read_text())

        assert config["id"] == "ExpertProgramOpenDrawer-v1"
        assert config["expert_program_path"] == "../../expert_program/open_drawer.yaml"
        assert config["articulation"][0]["uid"] == "drawer"
        assert config["articulation"][0]["drive_pros"]["drive_type"] == "none"
        assert set(config["env"]["extensions"]) == {"translation_axis"}

        monkeypatch.setattr(data_module, "get_data_path", lambda value: value)
        cfg = config_to_cfg(config, source_path=config_path)
        program = cfg.expert_program
        randomize_drawer_pose = cfg.events.randomize_drawer_pose

        assert program is not None
        assert program.program_id == "slide_open_drawer"
        assert type(program.program) is SegmentCfg
        assert type(program.program.steps.call) is RegisteredSemanticCallCfg
        assert program.program.steps.call.call_id == OPEN_DRAWER_CALL_ID
        assert program.program.steps.call.arguments["direction"] == "pull"
        assert program.program.steps.call.arguments["translation_distance"] == 0.18
        assert program.program.post[0].entity == "drawer"
        assert program.program.post[0].preset == "articulation"
        validator = program.program.validators[0]
        assert type(validator) is ArticulationJointPositionValidatorCfg
        assert validator.articulation == "drawer"
        assert validator.joint == "cabinet_to_drawer"
        assert validator.minimum_position == 0.10
        assert validator.maximum_position is None
        assert randomize_drawer_pose.func is randomize_articulation_root_pose
        assert randomize_drawer_pose.mode == "reset"
        assert randomize_drawer_pose.params["entity_cfg"].uid == "drawer"
        assert randomize_drawer_pose.params["position_range"] == [
            [-0.01, -0.01, 0.0],
            [0.01, 0.01, 0.0],
        ]
        assert randomize_drawer_pose.params["rotation_range"] == [
            [0.0, 0.0, -2.0],
            [0.0, 0.0, 2.0],
        ]
        assert randomize_drawer_pose.params["relative_position"] is True
        assert randomize_drawer_pose.params["relative_rotation"] is True

    def test_registered_call_lowers_to_the_atomic_slide_contract(self) -> None:
        """Only validated declarative values reach Slide goal/options types."""
        semantics = _slide_semantics()
        lowering = _OpenDrawerSlideLowerer(semantics).lower(
            _open_drawer_call(),
            context=object(),  # The task lowerer does not inspect live context.
            bound=object(),  # Compiler ownership is tested in the core suite.
        )

        assert type(lowering.goal) is SlideGoal
        assert lowering.goal.semantics is semantics
        assert type(lowering.goal.target_pose) is SceneEntityPose
        assert lowering.goal.target_pose.entity_id == HANDLE_ENTITY_ID
        assert type(lowering.skill_options) is SlideOptions
        assert lowering.skill_options.direction == "pull"
        assert lowering.skill_options.hand_interp_steps == 12
        assert lowering.skill_options.approach_distance == 0.10
        assert lowering.skill_options.translation_distance == 0.18
        descriptor = _open_drawer_call_catalog().discover(OPEN_DRAWER_CALL_ID)
        assert descriptor.skill_id == "slide"
        assert descriptor.target_descriptor is not None
        assert descriptor.target_descriptor.goal_type is SlideGoal
        assert descriptor.target_descriptor.options_type is SlideOptions

    def test_registered_call_rejects_non_canonical_payload(self) -> None:
        """The task extension cannot smuggle arbitrary arguments into Slide."""
        lowerer = _OpenDrawerSlideLowerer(_slide_semantics())

        with pytest.raises(ValueError, match="exactly"):
            lowerer.lower(
                _open_drawer_call(unexpected=True),
                context=object(),
                bound=object(),
            )
        with pytest.raises(ValueError, match="canonical ID"):
            lowerer.lower(
                _open_drawer_call(handle="native_link_name"),
                context=object(),
                bound=object(),
            )

    def test_task_does_not_reimplement_program_acceptance(self) -> None:
        """The standard bridge validator is the sole task-success boundary."""
        assert "is_task_success" not in ExpertProgramOpenDrawerEnv.__dict__
