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

"""Tests for strict serialized Expert Program loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from embodichain.lab.gym.envs.expert_program import (
    ConfigPath,
    ExpertProgramDecodeError,
    ExpertProgramValidationError,
    InvokeCfg,
    PickCfg,
    load_expert_program,
    loads_expert_program_json,
    parse_expert_program_json,
)


def _program_data() -> dict[str, object]:
    """Return one minimal complete Expert Program JSON value."""
    program: dict[str, object] = {
        "kind": "invoke",
        "call": {"kind": "pick", "object": "cube"},
    }
    return {
        "program_id": "loader_pick",
        "integration": {
            "robot_profile": "test_robot",
            "scene_registry": "test_scene",
            "runtime_preset": "safe",
        },
        "targets": {},
        "program": program,
    }


def _program_json() -> str:
    """Serialize the minimal program using standards-compliant JSON."""
    return json.dumps(_program_data())


class _RejectingValidationContext:
    """Reject integration references after recording their exact path."""

    def __init__(self) -> None:
        self.integration_paths: list[ConfigPath] = []

    def validate_integration(
        self,
        integration: object,
        *,
        path: ConfigPath,
    ) -> None:
        del integration
        self.integration_paths.append(path)
        raise KeyError("unavailable integration")

    def validate_semantic_call(self, call: object, *, path: ConfigPath) -> None:
        del call, path

    def validate_scene_reference(
        self,
        reference: str,
        *,
        role: str,
        path: ConfigPath,
    ) -> None:
        del reference, role, path

    def validate_post_policy(self, policy: object, *, path: ConfigPath) -> None:
        del policy, path

    def validate_validator(self, validator: object, *, path: ConfigPath) -> None:
        del validator, path


def test_parse_expert_program_json_preserves_predecode_mapping() -> None:
    value = parse_expert_program_json('{"host_integration_pending": [true, null, 3.5]}')

    assert value == {"host_integration_pending": [True, None, 3.5]}


@pytest.mark.parametrize("response", ["[]", "null", '"program"'])
def test_parse_expert_program_json_requires_top_level_mapping(response: str) -> None:
    with pytest.raises(ExpertProgramDecodeError) as error:
        parse_expert_program_json(response)

    assert error.value.code == "expected_mapping"
    assert error.value.path == ()


def test_loads_expert_program_json_decodes_one_plain_document() -> None:
    config = loads_expert_program_json(f"\n{_program_json()}\t")

    assert type(config.program) is InvokeCfg
    assert type(config.program.call) is PickCfg
    assert config.program.call.object == "cube"


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_load_expert_program_forwards_validation_context_for_each_format(
    tmp_path: Path,
    suffix: str,
) -> None:
    data = _program_data()
    serialized = json.dumps(data) if suffix == ".json" else yaml.safe_dump(data)
    path = tmp_path / f"program{suffix}"
    path.write_text(serialized, encoding="utf-8")
    context = _RejectingValidationContext()

    with pytest.raises(ExpertProgramValidationError) as error:
        load_expert_program(path, validation_context=context)

    assert error.value.code == "reference_validation_failed"
    assert error.value.path == ("integration",)
    assert context.integration_paths == [("integration",)]


def test_loads_expert_program_json_rejects_nested_duplicate_keys() -> None:
    duplicate = _program_json().replace(
        '"object": "cube"',
        '"object": "cube", "object": "other"',
    )

    with pytest.raises(ExpertProgramDecodeError) as error:
        loads_expert_program_json(duplicate)

    assert error.value.code == "duplicate_json_key"


@pytest.mark.parametrize(
    "invalid_response",
    [
        "```json\n{}\n```",
        f"{_program_json()} trailing text",
        f"{_program_json()} {_program_json()}",
    ],
)
def test_loads_expert_program_json_requires_one_unfenced_document(
    invalid_response: str,
) -> None:
    with pytest.raises(ExpertProgramDecodeError) as error:
        loads_expert_program_json(invalid_response)

    assert error.value.code == "invalid_json"


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_loads_expert_program_json_rejects_non_finite_numbers(number: str) -> None:
    response = f'{{"value": {number}}}'

    with pytest.raises(ExpertProgramDecodeError) as error:
        loads_expert_program_json(response)

    assert error.value.code == "non_finite_number"


def test_loads_expert_program_json_enforces_utf8_byte_limit() -> None:
    response = _program_json()
    too_small = len(response.encode("utf-8")) - 1

    with pytest.raises(ExpertProgramDecodeError) as error:
        loads_expert_program_json(response, max_bytes=too_small)

    assert error.value.code == "input_too_large"


def test_loads_expert_program_json_normalizes_invalid_utf8_text() -> None:
    response = "\ud800"

    with pytest.raises(ExpertProgramDecodeError) as error:
        loads_expert_program_json(response)

    assert error.value.code == "invalid_utf8"


def test_loads_expert_program_json_rejects_escaped_unpaired_surrogate() -> None:
    response = _program_json().replace("loader_pick", r"\ud800")

    with pytest.raises(ExpertProgramDecodeError) as error:
        loads_expert_program_json(response)

    assert error.value.code == "invalid_utf8"


def test_loads_expert_program_json_accepts_escaped_surrogate_pair() -> None:
    response = _program_json().replace("loader_pick", r"\ud83d\ude00")

    config = loads_expert_program_json(response)

    assert config.program_id == "😀"


def test_loads_expert_program_json_normalizes_oversized_integer() -> None:
    data = _program_data()
    data["targets"] = {
        "goal": {
            "kind": "cyclic_pose",
            "values": [
                {
                    "position": [10**400, 0, 0],
                    "quaternion_xyzw": [0, 0, 0, 1],
                }
            ],
        }
    }

    with pytest.raises(ExpertProgramDecodeError) as error:
        loads_expert_program_json(json.dumps(data))

    assert error.value.code == "invalid_value"
    assert error.value.path == ("targets", "goal", "values", 0)
