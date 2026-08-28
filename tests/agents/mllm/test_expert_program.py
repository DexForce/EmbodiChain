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

"""Tests for the strict MLLM Expert Program frontend."""

from __future__ import annotations

from collections.abc import Iterable
import json

import pytest

from embodichain.agents.mllm import (
    compile_mllm_expert_program,
    decode_mllm_expert_program,
)
from embodichain.lab.expert_program import (
    CompiledProgram,
    ExpertProgramCompileError,
    ExpertProgramDecodeError,
    ExpertProgramIntegrationCfg,
    decode_expert_program,
)
from embodichain.lab.gym.envs.expert_program import (
    EnvironmentStepClock,
    ExpertProgramEnvironmentAdapter,
    PlanningObservationPort,
)
from embodichain.lab.sim.atomic_actions import AtomicActionEngine, EntityState
from embodichain.lab.semantic_skills import (
    EffectEvidenceProvider,
    Pick,
    RobotSkillProfile,
    SceneEntityRegistration,
    SceneObjectRef,
    SceneRegistry,
)


def _integration() -> ExpertProgramIntegrationCfg:
    """Return the exact trusted integration selected by the host."""
    return ExpertProgramIntegrationCfg(
        robot_profile="test_robot",
        scene_registry="test_scene",
        runtime_preset="safe",
    )


def _invoke(call: dict[str, object]) -> dict[str, object]:
    """Wrap one semantic call in an Expert Program invoke node."""
    return {"kind": "invoke", "call": call}


def _model_data(
    call: dict[str, object] | None = None,
    *,
    program: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the integration-free JSON envelope exposed to the model."""
    if call is None:
        call = {"kind": "pick", "object": "cube"}
    return {
        "program_id": "model_program",
        "targets": {},
        "program": _invoke(call) if program is None else program,
    }


def _model_json(
    call: dict[str, object] | None = None,
    *,
    program: dict[str, object] | None = None,
) -> str:
    """Serialize one integration-free model response."""
    return json.dumps(
        _model_data(
            call,
            program=program,
        )
    )


class _UnusedStateProvider:
    """Satisfy the static scene contract without allowing live observation."""

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: object,
    ) -> EntityState:
        """Fail if provider-free compilation accidentally observes the scene."""
        del timestamp, env_ids
        raise AssertionError("Provider-free compilation must not observe the scene.")


class _CompileOnlyFactory:
    """Expose only the scene snapshot needed by adapter compilation."""

    scene_registry_id = "test_scene"
    robot_profile_id = "test_robot"

    def __init__(self) -> None:
        self.scene_registry_calls = 0

    def create_scene_registry(self) -> SceneRegistry:
        """Return one canonical object registration and count compilation."""
        self.scene_registry_calls += 1
        return SceneRegistry(
            (
                SceneEntityRegistration(
                    ref=SceneObjectRef("cube"),
                    state_provider=_UnusedStateProvider(),
                    semantic_type="cube",
                ),
            )
        )

    def create_robot_skill_profile(self) -> RobotSkillProfile:
        """Reject runtime assembly in this compile-only test factory."""
        raise AssertionError("MLLM frontend compilation must not assemble a runtime.")

    def create_atomic_action_engine(
        self,
        profile: RobotSkillProfile,
    ) -> AtomicActionEngine:
        """Reject engine creation in this compile-only test factory."""
        del profile
        raise AssertionError("MLLM frontend compilation must not create an engine.")

    def create_planning_observation_provider(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        clock: EnvironmentStepClock,
    ) -> PlanningObservationPort:
        """Reject observation-port creation during provider-free compilation."""
        del scene_registry, engine, clock
        raise AssertionError("MLLM frontend compilation must not create live ports.")

    def create_effect_evidence_providers(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: PlanningObservationPort,
    ) -> Iterable[EffectEvidenceProvider]:
        """Reject evidence-provider creation during provider-free compilation."""
        del scene_registry, engine, observation_provider
        raise AssertionError("MLLM frontend compilation must not create live ports.")


def _adapter(factory: _CompileOnlyFactory) -> ExpertProgramEnvironmentAdapter:
    """Create the existing production adapter around the compile-only factory."""
    return ExpertProgramEnvironmentAdapter(factory, step_dt=0.02)


def test_decoder_injects_exact_host_integration() -> None:
    config = decode_mllm_expert_program(
        _model_json(),
        integration=_integration(),
    )

    assert config.integration.robot_profile == "test_robot"
    assert config.integration.scene_registry == "test_scene"
    assert config.integration.runtime_preset == "safe"


def test_decoder_rejects_model_controlled_integration() -> None:
    response = _model_data()
    response["integration"] = {
        "robot_profile": "attacker_robot",
        "scene_registry": "attacker_scene",
        "runtime_preset": "unsafe",
    }

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(
            json.dumps(response),
            integration=_integration(),
        )

    assert error.value.code == "model_controlled_integration"
    assert error.value.path == ("integration",)


def test_decoder_rejects_parallel_program() -> None:
    parallel = {
        "kind": "parallel",
        "branches": [
            _invoke({"kind": "pick", "object": "cube"}),
            _invoke({"kind": "pick", "object": "cube"}),
        ],
        "barrier": {"kind": "barrier", "name": "join"},
    }

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(
            _model_json(program=parallel),
            integration=_integration(),
        )

    assert error.value.code == "mllm_program_node_not_allowed"
    assert error.value.path == ("program", "kind")


def test_decoder_rejects_removed_schema_version() -> None:
    response = _model_data()
    response["schema_version"] = 2

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(
            json.dumps(response),
            integration=_integration(),
        )

    assert error.value.code == "unknown_field"
    assert error.value.path == ("schema_version",)


def test_decoder_rejects_registered_semantic_calls() -> None:
    registered = {
        "kind": "registered",
        "call_id": "vendor.inspect",
        "arguments": {},
    }

    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(
            _model_json(registered),
            integration=_integration(),
        )

    assert error.value.code == "mllm_call_not_allowed"
    assert error.value.path == ("program", "call", "kind")


@pytest.mark.parametrize(
    "call",
    [
        {"kind": "pick", "object": "cube", "resources": {"primary": "left"}},
        {
            "kind": "place",
            "object": "cube",
            "on": "tray",
            "resources": {"primary": "left"},
        },
        {
            "kind": "hand_over",
            "object": "cube",
            "resources": {"destination": "right"},
        },
    ],
)
def test_decoder_rejects_explicit_nonempty_resource_overrides(
    call: dict[str, object],
) -> None:
    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(
            _model_json(call),
            integration=_integration(),
        )

    assert error.value.code == "mllm_resource_override_not_allowed"
    assert error.value.path == ("program", "call", "resources")


def test_decoder_allows_explicit_empty_resources() -> None:
    config = decode_mllm_expert_program(
        _model_json({"kind": "pick", "object": "cube", "resources": {}}),
        integration=_integration(),
    )

    assert config.program.call.resources == {}  # type: ignore[union-attr]


def test_decoder_rejects_removed_handover_receiver_alias() -> None:
    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(
            _model_json(
                {
                    "kind": "hand_over",
                    "object": "cube",
                    "receiver": "right",
                }
            ),
            integration=_integration(),
        )

    assert error.value.code == "unknown_field"
    assert error.value.path == ("program", "call", "receiver")


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (
            {"kind": "pick", "object": "env.robot.control_parts"},
            "environment_traversal",
        ),
        ({"kind": "pick", "object": "eval(1 + 1)"}, "executable_expression"),
    ],
)
def test_decoder_reuses_executable_free_value_validation(
    call: dict[str, object],
    code: str,
) -> None:
    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(
            _model_json(call),
            integration=_integration(),
        )

    assert error.value.code == code


@pytest.mark.parametrize(
    ("response", "code"),
    [
        ("```json\n{}\n```", "invalid_json"),
        ('{"program_id": "one", "program_id": "two"}', "duplicate_json_key"),
        ('{"value": NaN}', "non_finite_number"),
        ('{"value": 1e400}', "non_finite_number"),
    ],
)
def test_decoder_propagates_strict_json_failures(response: str, code: str) -> None:
    with pytest.raises(ExpertProgramDecodeError) as error:
        decode_mllm_expert_program(response, integration=_integration())

    assert error.value.code == code


def test_compile_frontend_reuses_existing_adapter_and_compiler() -> None:
    factory = _CompileOnlyFactory()
    adapter = _adapter(factory)
    response = _model_json()

    model_compiled = compile_mllm_expert_program(
        response,
        adapter=adapter,
        integration=_integration(),
    )
    direct_data = _model_data()
    direct_data["integration"] = {
        "robot_profile": "test_robot",
        "scene_registry": "test_scene",
        "runtime_preset": "safe",
    }
    direct_compiled = adapter.compile(decode_expert_program(direct_data))

    model_call = list(model_compiled)[0].calls[0].call
    direct_call = list(direct_compiled)[0].calls[0].call
    assert type(model_compiled) is CompiledProgram
    assert type(model_call) is Pick
    assert type(direct_call) is Pick
    assert model_call.object.entity_id == direct_call.object.entity_id == "cube"
    assert factory.scene_registry_calls == 2


def test_policy_failure_does_not_touch_adapter_or_runtime() -> None:
    factory = _CompileOnlyFactory()
    adapter = _adapter(factory)
    registered = {
        "kind": "registered",
        "call_id": "vendor.inspect",
    }

    with pytest.raises(ExpertProgramDecodeError):
        compile_mllm_expert_program(
            _model_json(registered),
            adapter=adapter,
            integration=_integration(),
        )

    assert factory.scene_registry_calls == 0


def test_compile_frontend_rejects_unknown_scene_reference() -> None:
    factory = _CompileOnlyFactory()

    with pytest.raises(ExpertProgramCompileError) as error:
        compile_mllm_expert_program(
            _model_json({"kind": "pick", "object": "missing"}),
            adapter=_adapter(factory),
            integration=_integration(),
        )

    assert error.value.code == "scene_resolution_failed"
    assert error.value.path == ("program", "call", "object")
