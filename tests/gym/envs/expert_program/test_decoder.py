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

"""Tests for strict Expert Program Version 1 decoding."""

from __future__ import annotations

from copy import deepcopy

import pytest

from embodichain.lab.gym.envs.expert_program import (
    MAX_REPEAT_COUNT,
    ConfigPath,
    ExpertProgramDecodeError,
    ExpertProgramIntegrationCfg,
    ExpertProgramValidationError,
    HandOverCfg,
    PickCfg,
    PlaceCfg,
    PostPolicyCfg,
    RegisteredSemanticCallCfg,
    RepeatCfg,
    SceneReferenceRole,
    SegmentCfg,
    SequenceCfg,
    TargetRefCfg,
    ValidatorCfg,
    decode_expert_program,
    decode_semantic_call,
    encode_semantic_call,
    render_config_path,
)


def _program_data() -> dict[str, object]:
    """Return the repeated-cube Version 1 example as plain JSON values."""
    return {
        "schema_version": 1,
        "program_id": "repeated_cube_pick_place",
        "integration": {
            "robot_profile": "auto",
            "scene_registry": "env",
            "runtime_preset": "safe",
        },
        "targets": {
            "drop_pose": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": [0.45, -0.20, 0.20],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    {
                        "position": [0.45, 0.00, 0.20],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    {
                        "position": [0.45, 0.20, 0.20],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                ],
            }
        },
        "program": {
            "kind": "repeat",
            "count": 3,
            "body": {
                "kind": "segment",
                "name": "move_cube",
                "steps": {
                    "kind": "sequence",
                    "items": [
                        {
                            "kind": "invoke",
                            "call": {"kind": "pick", "object": "cube"},
                        },
                        {
                            "kind": "invoke",
                            "call": {
                                "kind": "place",
                                "object": "cube",
                                "at": {
                                    "kind": "target_ref",
                                    "target": "drop_pose",
                                },
                            },
                        },
                    ],
                },
                "post": [
                    {
                        "kind": "wait_stable",
                        "entity": "cube",
                        "preset": "rigid_object",
                    }
                ],
                "validators": [
                    {
                        "kind": "object_near_target",
                        "object": "cube",
                        "target": "drop_pose",
                        "position_tolerance": 0.03,
                    }
                ],
            },
        },
    }


def _invoke(call: dict[str, object]) -> dict[str, object]:
    """Wrap one call mapping in an invoke node."""
    return {"kind": "invoke", "call": call}


@pytest.mark.parametrize(
    "payload",
    (
        {
            "kind": "pick",
            "object": "cube",
            "grasp": "cube_grasp",
            "resources": {"primary": "left"},
        },
        {
            "kind": "place",
            "object": "cube",
            "at": {"kind": "target_ref", "target": "drop_pose"},
        },
        {
            "kind": "hand_over",
            "object": "cube",
            "receiver": "right",
            "final_target": {"kind": "target_ref", "target": "drop_pose"},
        },
        {
            "kind": "operate_articulation",
            "articulation": "drawer",
            "handle": "drawer_handle",
            "target": "open",
        },
        {
            "kind": "registered",
            "call_id": "example.inspect",
            "schema_version": 1,
            "arguments": {"labels": ["front", "back"]},
        },
    ),
)
def test_public_semantic_call_codec_round_trips_exact_schema(
    payload: dict[str, object],
) -> None:
    call = decode_semantic_call(
        payload,
        target_ids=frozenset({"drop_pose"}),
    )
    encoded = encode_semantic_call(call)
    round_trip = decode_semantic_call(
        encoded,
        target_ids=frozenset({"drop_pose"}),
    )

    assert type(round_trip) is type(call)
    assert encode_semantic_call(round_trip) == encoded


def test_public_semantic_call_decoder_owns_input_and_validates_context() -> None:
    payload = {"kind": "pick", "object": "cube", "grasp": "cube_grasp"}
    context = _StaticValidationContext()

    call = decode_semantic_call(payload, validation_context=context)
    payload["object"] = "changed"

    assert type(call) is PickCfg
    assert call.object == "cube"
    assert ("call",) in context.validated_paths
    assert ("call", "object") in context.validated_paths


def test_decoder_builds_owned_repeated_cube_ast() -> None:
    data = _program_data()

    config = decode_expert_program(data)
    data["program"]["count"] = 99
    data["targets"]["drop_pose"]["values"][0]["position"][0] = -1.0

    assert type(config.program) is RepeatCfg
    assert config.program.count == 3
    assert type(config.program.body) is SegmentCfg
    assert type(config.program.body.steps) is SequenceCfg
    place = config.program.body.steps.items[1].call
    assert type(place) is PlaceCfg
    assert place.at == TargetRefCfg(target="drop_pose")
    assert config.targets["drop_pose"].values[0].position[0] == pytest.approx(0.45)


def test_decoder_supports_every_version_one_semantic_call() -> None:
    data = _program_data()
    data["program"] = {
        "kind": "sequence",
        "items": [
            _invoke(
                {
                    "kind": "pick",
                    "object": "cube",
                    "grasp": "cube_grasp",
                    "resources": {"primary": "left_actor"},
                }
            ),
            _invoke({"kind": "place", "object": "cube", "on": "tray_top"}),
            _invoke(
                {
                    "kind": "hand_over",
                    "object": "cube",
                    "receiver": "right_actor",
                    "final_target": {
                        "kind": "target_ref",
                        "target": "drop_pose",
                    },
                }
            ),
            _invoke(
                {
                    "kind": "registered",
                    "call_id": "example.inspect",
                    "schema_version": 1,
                    "arguments": {
                        "labels": ["front", "back"],
                        "options": {"confidence": 0.9},
                    },
                }
            ),
        ],
    }

    config = decode_expert_program(data)
    assert [type(node.call) for node in config.program.items] == [
        PickCfg,
        PlaceCfg,
        HandOverCfg,
        RegisteredSemanticCallCfg,
    ]
    handover = config.program.items[2].call
    assert handover.resources == {"destination": "right_actor"}
    registered = config.program.items[3].call
    assert registered.arguments == {
        "labels": ("front", "back"),
        "options": {"confidence": 0.9},
    }


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda data: data.update({"unexpected": True}),
            "$.unexpected",
        ),
        (
            lambda data: data["program"]["body"].update({"unexpected": True}),
            "$.program.body.unexpected",
        ),
        (
            lambda data: data["program"]["body"]["steps"]["items"][0]["call"].update(
                {"unexpected": True}
            ),
            "$.program.body.steps.items[0].call.unexpected",
        ),
    ],
)
def test_decoder_rejects_unknown_fields_with_complete_path(
    mutate: object,
    expected_path: str,
) -> None:
    data = _program_data()
    mutate(data)

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "unknown_field"
    assert render_config_path(error.value.path) == expected_path


def test_decoder_reports_missing_required_field_at_exact_path() -> None:
    data = _program_data()
    del data["integration"]["runtime_preset"]

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "missing_field"
    assert render_config_path(error.value.path) == "$.integration.runtime_preset"


@pytest.mark.parametrize(
    ("value", "code"),
    [
        (None, "missing_discriminator"),
        ("parallel", "unknown_discriminator"),
    ],
)
def test_decoder_rejects_missing_or_reserved_program_discriminator(
    value: str | None,
    code: str,
) -> None:
    data = _program_data()
    if value is None:
        del data["program"]["kind"]
    else:
        data["program"]["kind"] = value

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == code
    assert error.value.path == ("program", "kind")


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda data: data["targets"]["drop_pose"].update({"kind": "pose"}),
            "$.targets.drop_pose.kind",
        ),
        (
            lambda data: data["program"]["body"]["steps"]["items"][0]["call"].update(
                {"kind": "move"}
            ),
            "$.program.body.steps.items[0].call.kind",
        ),
        (
            lambda data: data["program"]["body"]["steps"]["items"][1]["call"][
                "at"
            ].update({"kind": "env_ref"}),
            "$.program.body.steps.items[1].call.at.kind",
        ),
        (
            lambda data: data["program"]["body"]["post"][0].update({"kind": "sleep"}),
            "$.program.body.post[0].kind",
        ),
        (
            lambda data: data["program"]["body"]["validators"][0].update(
                {"kind": "python"}
            ),
            "$.program.body.validators[0].kind",
        ),
    ],
)
def test_every_union_rejects_unknown_discriminator_at_exact_path(
    mutate: object,
    expected_path: str,
) -> None:
    data = _program_data()
    mutate(data)

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "unknown_discriminator"
    assert render_config_path(error.value.path) == expected_path


@pytest.mark.parametrize("schema_version", [False, 0, 3, "1"])
def test_decoder_rejects_unsupported_top_level_schema_version(
    schema_version: object,
) -> None:
    data = _program_data()
    data["schema_version"] = schema_version

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "unsupported_schema_version"
    assert error.value.path == ("schema_version",)


def test_decoder_reports_unknown_target_at_reference_site() -> None:
    data = _program_data()
    data["program"]["body"]["steps"]["items"][1]["call"]["at"]["target"] = "missing"

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "unknown_target"
    assert render_config_path(error.value.path) == (
        "$.program.body.steps.items[1].call.at.target"
    )


@pytest.mark.parametrize("count", [False, 0, MAX_REPEAT_COUNT + 1])
def test_decoder_rejects_unbounded_or_invalid_repeat_count(count: object) -> None:
    data = _program_data()
    data["program"]["count"] = count

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "invalid_repeat_count"
    assert render_config_path(error.value.path) == "$.program.count"


def test_registered_call_schema_version_error_reports_version_field() -> None:
    data = _program_data()
    data["program"] = _invoke(
        {
            "kind": "registered",
            "call_id": "example.inspect",
            "schema_version": 2,
        }
    )

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "invalid_schema_version"
    assert render_config_path(error.value.path) == "$.program.call.schema_version"


@pytest.mark.parametrize(
    ("arguments", "code", "suffix"),
    [
        ({"eval": "1 + 1"}, "forbidden_construct", ".arguments.eval"),
        (
            {"source": "env.robot.control_parts"},
            "environment_traversal",
            ".arguments.source",
        ),
        (
            {"source": "eval(1 + 1)"},
            "executable_expression",
            ".arguments.source",
        ),
        ({"callback": lambda: None}, "non_declarative_value", ".arguments.callback"),
        ({"live": object()}, "non_declarative_value", ".arguments.live"),
    ],
)
def test_decoder_rejects_executable_traversal_or_live_registered_payload(
    arguments: dict[str, object],
    code: str,
    suffix: str,
) -> None:
    data = _program_data()
    data["program"] = _invoke(
        {
            "kind": "registered",
            "call_id": "example.inspect",
            "schema_version": 1,
            "arguments": arguments,
        }
    )

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == code
    assert render_config_path(error.value.path).endswith(suffix)


def test_decoder_rejects_cyclic_input_before_ast_recursion() -> None:
    data = _program_data()
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    data["program"] = _invoke(
        {
            "kind": "registered",
            "call_id": "example.inspect",
            "schema_version": 1,
            "arguments": cyclic,
        }
    )

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(data)

    assert error.value.code == "cyclic_input"


class _StaticValidationContext:
    """Small provider-free reference catalog used by decoder tests."""

    def __init__(
        self,
        *,
        calls: set[str] | None = None,
        scene: set[str] | None = None,
    ) -> None:
        self.calls = {"pick", "place", "hand_over"} if calls is None else calls
        self.scene = {"cube", "cube_grasp", "tray_top"} if scene is None else scene
        self.validated_paths: list[ConfigPath] = []

    def validate_integration(
        self,
        integration: ExpertProgramIntegrationCfg,
        *,
        path: ConfigPath,
    ) -> None:
        if integration.robot_profile != "auto":
            raise KeyError(integration.robot_profile)
        self.validated_paths.append(path)

    def validate_semantic_call(
        self,
        call: object,
        *,
        path: ConfigPath,
    ) -> None:
        semantic_id = (
            call.call_id if type(call) is RegisteredSemanticCallCfg else call.kind
        )
        if semantic_id not in self.calls:
            raise KeyError(semantic_id)
        self.validated_paths.append(path)

    def validate_scene_reference(
        self,
        reference: str,
        *,
        role: SceneReferenceRole,
        path: ConfigPath,
    ) -> None:
        del role
        if reference not in self.scene:
            raise KeyError(reference)
        self.validated_paths.append(path)

    def validate_post_policy(
        self,
        policy: PostPolicyCfg,
        *,
        path: ConfigPath,
    ) -> None:
        if policy.kind != "wait_stable" or policy.preset != "rigid_object":
            raise KeyError(policy.preset)
        self.validated_paths.append(path)

    def validate_validator(
        self,
        validator: ValidatorCfg,
        *,
        path: ConfigPath,
    ) -> None:
        if validator.kind != "object_near_target":
            raise KeyError(validator.kind)
        self.validated_paths.append(path)


def test_decoder_runs_explicit_provider_free_validation_context() -> None:
    data = _program_data()
    context = _StaticValidationContext()

    config = decode_expert_program(data, validation_context=context)

    assert config.program_id == "repeated_cube_pick_place"
    assert ("integration",) in context.validated_paths
    assert ("program", "body", "steps", "items", 0, "call") in (context.validated_paths)
    assert ("program", "body", "post", 0, "entity") in (context.validated_paths)


def test_validation_context_failure_is_wrapped_at_exact_reference_path() -> None:
    data = _program_data()
    context = _StaticValidationContext(scene={"cube"})
    data["program"]["body"]["steps"]["items"][0]["call"]["grasp"] = "missing_grasp"

    with pytest.raises(ExpertProgramValidationError) as error:
        decode_expert_program(data, validation_context=context)

    assert error.value.code == "reference_validation_failed"
    assert render_config_path(error.value.path) == (
        "$.program.body.steps.items[0].call.grasp"
    )


def test_decoder_does_not_mutate_caller_input_on_failure() -> None:
    data = _program_data()
    data["program"]["unexpected"] = True
    before = deepcopy(data)

    with pytest.raises(ExpertProgramDecodeError):
        decode_expert_program(data)

    assert data == before
