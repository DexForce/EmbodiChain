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

"""Tests for typed Expert Program configuration values."""

from __future__ import annotations

import math

import pytest

from embodichain.lab.gym.envs.expert_program import (
    ArticulationJointPositionValidatorCfg,
    CyclicPoseTargetCfg,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    HandOverCfg,
    InvokeCfg,
    ObjectNearTargetValidatorCfg,
    PickCfg,
    PlaceCfg,
    PoseCfg,
    RegisteredSemanticCallCfg,
    RepeatCfg,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    WaitStablePostCfg,
)
from embodichain.lab.gym.envs.expert_program.cfg import MAX_REPEAT_COUNT
from embodichain.utils.configclass import is_configclass


def _integration() -> ExpertProgramIntegrationCfg:
    """Return one valid provider-free integration selection."""
    return ExpertProgramIntegrationCfg(
        robot_profile="auto",
        scene_registry="env",
        runtime_preset="safe",
    )


def _pick_invoke() -> InvokeCfg:
    """Return one minimal semantic invocation."""
    return InvokeCfg(call=PickCfg(object="cube"))


def test_every_public_schema_value_uses_configclass() -> None:
    classes = (
        ExpertProgramCfg,
        ExpertProgramIntegrationCfg,
        PoseCfg,
        TargetRefCfg,
        CyclicPoseTargetCfg,
        PickCfg,
        PlaceCfg,
        HandOverCfg,
        RegisteredSemanticCallCfg,
        WaitStablePostCfg,
        ObjectNearTargetValidatorCfg,
        ArticulationJointPositionValidatorCfg,
        InvokeCfg,
        SequenceCfg,
        RepeatCfg,
        SegmentCfg,
    )

    assert all(is_configclass(cls) for cls in classes)


def test_call_configs_own_resources_and_registered_payloads() -> None:
    resources = {"primary": "left_actor"}
    arguments = {"waypoints": [1, {"enabled": True}]}
    pick = PickCfg(object="cube", resources=resources)
    registered = RegisteredSemanticCallCfg(
        call_id="example.inspect",
        arguments=arguments,
    )

    resources["primary"] = "right_actor"
    arguments["waypoints"][1]["enabled"] = False

    assert pick.resources == {"primary": "left_actor"}
    assert registered.arguments == {
        "waypoints": (1, {"enabled": True}),
    }


@pytest.mark.parametrize("count", [False, 0, -1, MAX_REPEAT_COUNT + 1])
def test_repeat_rejects_non_positive_non_integer_or_excessive_count(
    count: object,
) -> None:
    with pytest.raises(ValueError, match="count must be an integer"):
        RepeatCfg(count=count, body=_pick_invoke())


def test_program_rejects_nested_repeat_expansion_above_static_budget() -> None:
    nested = RepeatCfg(
        count=MAX_REPEAT_COUNT,
        body=RepeatCfg(count=MAX_REPEAT_COUNT, body=_pick_invoke()),
    )

    with pytest.raises(ValueError, match="expands to more than"):
        ExpertProgramCfg(
            program_id="too_large",
            integration=_integration(),
            targets={},
            program=nested,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        (
            {
                "at": TargetRefCfg(target="drop"),
                "on": "tray",
            },
            "exactly one",
        ),
    ],
)
def test_place_requires_exactly_one_typed_destination(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PlaceCfg(object="cube", **kwargs)


def test_programmatic_config_rejects_unknown_target_reference() -> None:
    program = InvokeCfg(
        call=PlaceCfg(
            object="cube",
            at=TargetRefCfg(target="missing"),
        )
    )

    with pytest.raises(ValueError, match="Unknown target reference 'missing'"):
        ExpertProgramCfg(
            program_id="missing_target",
            integration=_integration(),
            targets={},
            program=program,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {"callback": lambda: None},
        {"eval": "1 + 1"},
        {"source": "env.robot.control_parts"},
        {"bad": math.inf},
    ],
)
def test_registered_call_rejects_executable_or_non_declarative_payload(
    arguments: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RegisteredSemanticCallCfg(
            call_id="example.inspect",
            arguments=arguments,
        )


def test_pose_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="non-zero magnitude"):
        PoseCfg(
            position=(0.0, 0.0, 0.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 0.0),
        )


def test_segment_owns_post_policy_and_validator_sequences() -> None:
    post = [WaitStablePostCfg(entity="cube")]
    validators = [ObjectNearTargetValidatorCfg(object="cube", target="drop_pose")]
    segment = SegmentCfg(
        name="move_cube",
        steps=SequenceCfg(items=(_pick_invoke(),)),
        post=post,
        validators=validators,
    )

    post.clear()
    validators.clear()

    assert segment.post == (WaitStablePostCfg(entity="cube"),)
    assert segment.validators == (
        ObjectNearTargetValidatorCfg(object="cube", target="drop_pose"),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "At least one"),
        (
            {"minimum_position": 0.2, "maximum_position": 0.1},
            "less than or equal",
        ),
    ],
)
def test_articulation_joint_validator_requires_an_ordered_bound(
    kwargs: dict[str, object],
    message: str,
) -> None:
    """A joint validator must define one non-empty inclusive interval."""
    with pytest.raises(ValueError, match=message):
        ArticulationJointPositionValidatorCfg(
            articulation="drawer",
            joint="cabinet_to_drawer",
            **kwargs,
        )
