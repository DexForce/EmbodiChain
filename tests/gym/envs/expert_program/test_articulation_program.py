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

"""Tests for declarative articulation calls in Expert Programs."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramCfg,
    ExpertProgramCompiler,
    ExpertProgramDecodeError,
    ExpertProgramIntegrationCfg,
    InvokeCfg,
    OperateArticulationCfg,
    decode_expert_program,
)
from embodichain.lab.sim.atomic_actions import Affordance, EntityState
from embodichain.lab.sim.skills.calls import OperateArticulation
from embodichain.lab.sim.skills.scene import (
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneRegistry,
)


class _NeverObserveProvider:
    """Reject state observation during static program compilation."""

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        raise AssertionError("Compilation must not observe providers.")


def _payload(call: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "program_id": "open_drawer",
        "integration": {
            "robot_profile": "manipulator",
            "scene_registry": "scene",
            "runtime_preset": "safe",
        },
        "targets": {},
        "program": {"kind": "invoke", "call": call},
    }


def _compiler() -> ExpertProgramCompiler:
    provider = _NeverObserveProvider()
    drawer = SceneArticulationRef("drawer")
    registry = SceneRegistry(
        (
            SceneEntityRegistration(ref=drawer, state_provider=provider),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("drawer_handle"),
                parent=drawer,
                native_name="handle",
                relative_pose=torch.eye(4),
                affordance=Affordance(),
            ),
        )
    )
    return ExpertProgramCompiler.from_scene_registry(registry)


def _integration() -> ExpertProgramIntegrationCfg:
    return ExpertProgramIntegrationCfg(
        robot_profile="manipulator",
        scene_registry="scene",
        runtime_preset="safe",
    )


def test_decoder_accepts_named_and_explicit_articulation_targets() -> None:
    named = decode_expert_program(
        _payload(
            {
                "kind": "operate_articulation",
                "articulation": "drawer",
                "handle": "drawer_handle",
                "target": "open",
                "resources": {"primary": "right_arm"},
            }
        )
    )
    explicit = decode_expert_program(
        _payload(
            {
                "kind": "operate_articulation",
                "articulation": "drawer",
                "target_position": 0.42,
                "target_displacement": 0.40,
            }
        )
    )

    assert type(named.program) is InvokeCfg
    assert named.program.call == OperateArticulationCfg(
        articulation="drawer",
        handle="drawer_handle",
        target="open",
        resources={"primary": "right_arm"},
    )
    assert type(explicit.program) is InvokeCfg
    assert explicit.program.call == OperateArticulationCfg(
        articulation="drawer",
        target_position=0.42,
        target_displacement=0.40,
    )


@pytest.mark.parametrize(
    ("fields", "code"),
    (
        ({"target": "open", "target_position": 0.4}, "conflicting_articulation_target"),
        ({"target_position": 0.4}, "incomplete_articulation_target"),
        ({"target_displacement": 0.2}, "incomplete_articulation_target"),
        ({"target_position": True, "target_displacement": 0.2}, "invalid_number"),
    ),
)
def test_decoder_rejects_ambiguous_or_incomplete_articulation_targets(
    fields: dict[str, object],
    code: str,
) -> None:
    call = {
        "kind": "operate_articulation",
        "articulation": "drawer",
        **fields,
    }

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_expert_program(_payload(call))

    assert error.value.code == code


def test_compiler_preserves_typed_articulation_call_without_observation() -> None:
    config = ExpertProgramCfg(
        schema_version=1,
        program_id="open_drawer",
        integration=_integration(),
        targets={},
        program=InvokeCfg(
            call=OperateArticulationCfg(
                articulation="drawer",
                handle="drawer_handle",
                target="open",
            )
        ),
    )

    segment = tuple(_compiler().compile(config))[0]
    call = segment.calls[0].call

    assert type(call) is OperateArticulation
    assert call.articulation == SceneArticulationRef("drawer")
    assert call.handle == SceneAffordanceRef("drawer_handle")
    assert call.target == "open"
    assert call.target_position is None
    assert call.target_displacement is None


__all__: list[str] = []
