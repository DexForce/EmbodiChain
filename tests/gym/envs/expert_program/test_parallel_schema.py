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

"""Tests for Expert Program schema Version 2 parallel nodes."""

from __future__ import annotations

import pytest

from embodichain.lab.gym.envs.expert_program.cfg import (
    BarrierCfg,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    InvokeCfg,
    ParallelCfg,
    PickCfg,
)
from embodichain.lab.gym.envs.expert_program.decoder import (
    ExpertProgramDecodeError,
    decode_expert_program,
)


def _payload(*, schema_version: int = 2) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "program_id": "parallel_pick",
        "integration": {
            "robot_profile": "dual_arm",
            "scene_registry": "scene",
            "runtime_preset": "safe",
        },
        "targets": {},
        "program": {
            "kind": "parallel",
            "branches": [
                {
                    "kind": "invoke",
                    "call": {"kind": "pick", "object": "left_cube"},
                },
                {
                    "kind": "invoke",
                    "call": {"kind": "pick", "object": "right_cube"},
                },
            ],
            "barrier": {
                "kind": "barrier",
                "name": "both_picked",
                "timeout_steps": 200,
                "failure_policy": "fail_fast",
            },
        },
    }


def test_decode_schema_v2_parallel_with_explicit_barrier() -> None:
    config = decode_expert_program(_payload())

    assert config.schema_version == 2
    assert type(config.program) is ParallelCfg
    assert len(config.program.branches) == 2
    assert config.program.barrier == BarrierCfg(
        name="both_picked",
        timeout_steps=200,
        failure_policy="fail_fast",
    )


def test_schema_v1_is_rejected_before_program_decoding() -> None:
    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(_payload(schema_version=1))

    assert error.value.code == "unsupported_schema_version"
    assert error.value.path == ("schema_version",)


def test_parallel_requires_two_branches_and_explicit_barrier() -> None:
    payload = _payload()
    program = payload["program"]
    assert type(program) is dict
    program["branches"] = program["branches"][:1]  # type: ignore[index]
    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(payload)
    assert error.value.code == "parallel_branch_count"

    payload = _payload()
    program = payload["program"]
    assert type(program) is dict
    del program["barrier"]
    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(payload)
    assert error.value.code == "missing_field"
    assert error.value.path == ("program", "barrier")


def test_barrier_is_not_valid_as_a_standalone_program() -> None:
    with pytest.raises(TypeError, match="exact ProgramNodeCfg"):
        ExpertProgramCfg(
            schema_version=2,
            program_id="invalid_barrier",
            integration=ExpertProgramIntegrationCfg(
                robot_profile="profile",
                scene_registry="scene",
                runtime_preset="safe",
            ),
            targets={},
            program=BarrierCfg(name="orphan"),
        )


def test_parallel_cfg_rejects_nested_parallel() -> None:
    invoke = InvokeCfg(call=PickCfg(object="cube"))
    nested = ParallelCfg(
        branches=(invoke, invoke),
        barrier=BarrierCfg(name="inner"),
    )
    with pytest.raises(ValueError, match="Nested Parallel"):
        ParallelCfg(
            branches=(nested, invoke),
            barrier=BarrierCfg(name="outer"),
        )


__all__: list[str] = []
