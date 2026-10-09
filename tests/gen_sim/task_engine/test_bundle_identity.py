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

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine._task_program import configured
from embodichain.gen_sim.task_engine._task_program.identity import (
    bundle_registration_fingerprint,
)
from embodichain.gen_sim.task_engine._task_program.services import (
    _RelativePlaceLowererFactory,
    _RelativePlaceRoute,
    _UprightPlaceLowererFactory,
)
from embodichain.gen_sim.task_engine._task_program.stack_place import _StackPlaceFactory
from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.task_program.integrations import IntegrationFingerprintMismatch
from embodichain.lab.task_program.integrations._configured_composition import (
    _resolve_task_program_components,
)
from embodichain.lab.task_program.integrations.catalog import (
    SimulationTaskProgramRegistration,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SemanticCallDescriptor,
)
from embodichain.utils.utility import load_config

# Fixed v38 declaration digests, independent of the new projection implementation.
_LEGACY_REGISTRATION = (
    "1992baee8c923e814d096d18cee0e5d80575efeac1e1d23558f655d581984c00"
)
_LEGACY_CORE = "87f6a3a11d4b3fe1f143c0263b398f654a54a06edb3ef1f1529dd36fee46b3cf"
_ROUTE_CASES = [
    (
        _RelativePlaceLowererFactory,
        "ab7a2c157a297daf48af5d90ebe8b1eac0772063c85351a97a256b5ddf519fea",
    ),
    (
        _UprightPlaceLowererFactory,
        "319e615b25c847fb0e44dddd5af3cef066ffa15b0db15bf6a919dd11729fb1a0",
    ),
    (
        _StackPlaceFactory,
        "d12d778254515da53c7165a494ee733802328766e1da47ff10a5f8f31fde1c4d",
    ),
]


def _compose(
    monkeypatch: pytest.MonkeyPatch, *, mode: str | None = None
) -> configured.TaskDeployment:
    path = (
        Path(__file__).parents[3]
        / "embodichain_tasks/configs/tasks/manipulation/hand_over"
        / "task.dual_ur5_dh_pgi_140_80.yaml"
    )
    physical = _resolve_gym_components(load_config(path), base_dir=path.parent)
    program_path, task, policy = _resolve_task_program_components(
        physical.config["task_program"], base_dir=path.parent
    )
    task = deepcopy(task)
    options = task["profile"]["action_options"]["hand_over"]
    options.pop("source_hold_mode", None)
    if mode is not None:
        options["source_hold_mode"] = mode
    monkeypatch.setattr(
        configured,
        "_resolve_task_program_components",
        lambda *_a, **_kw: (program_path, task, policy),
    )
    return configured.compose_deployment(
        task_program=physical.config["task_program"],
        skill_profile=physical.embodiment_skill_profile,
        base_dir=path.parent,
    )


@pytest.fixture
def deployment(monkeypatch: pytest.MonkeyPatch) -> configured.TaskDeployment:
    return _compose(monkeypatch)


def _with_route(
    registration: SimulationTaskProgramRegistration,
    factory_type: type[_RelativePlaceLowererFactory],
    *,
    stage_id: str = "",
    height: float = 0.1,
) -> SimulationTaskProgramRegistration:
    factory = factory_type(
        (
            _RelativePlaceRoute(
                "can", "support_surface", "on", (0.0, 0.0, height), stage_id=stage_id
            ),
        )
    )
    return replace(
        registration,
        call_catalog=registration.call_catalog.with_descriptor(
            SemanticCallDescriptor(
                call_id=factory.call_id,
                spec_type=RegisteredSemanticCall,
                target_descriptor=factory.target_descriptor,
            )
        ),
        registered_semantic_lowerer_factories=(
            *registration.registered_semantic_lowerer_factories,
            factory,
        ),
    )


@pytest.mark.parametrize("mode", [None, "grasp_command", "observed"])
def test_handover_bundle_identity_uses_v38_default_and_keeps_explicit_new_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    composed = _compose(monkeypatch, mode=mode)
    identity = bundle_registration_fingerprint(composed.integration.registration)
    if mode == "observed":
        assert identity != _LEGACY_REGISTRATION
        assert composed.integration.integration_fingerprint != _LEGACY_CORE
    else:
        assert identity == _LEGACY_REGISTRATION
        assert composed.integration.integration_fingerprint == _LEGACY_CORE
    assert composed.integration.registration.fingerprint != _LEGACY_REGISTRATION
    composed.integration.registration.assert_unchanged()


@pytest.mark.parametrize("factory_type,legacy_digest", _ROUTE_CASES)
def test_empty_stage_retains_v38_route_identity_but_real_changes_do_not(
    deployment: configured.TaskDeployment,
    factory_type: type[_RelativePlaceLowererFactory],
    legacy_digest: str,
) -> None:
    registration = _with_route(deployment.integration.registration, factory_type)
    assert bundle_registration_fingerprint(registration) == legacy_digest
    for changed in (
        _with_route(
            deployment.integration.registration, factory_type, stage_id="step_02"
        ),
        _with_route(deployment.integration.registration, factory_type, height=0.2),
    ):
        assert bundle_registration_fingerprint(changed) != legacy_digest
    registration.assert_unchanged()


def test_bundle_identity_cannot_hide_post_registration_route_drift(
    deployment: configured.TaskDeployment,
) -> None:
    registration = _with_route(
        deployment.integration.registration, _RelativePlaceLowererFactory
    )
    route = registration.registered_semantic_lowerer_factories[-1].routes[0]
    object.__setattr__(route, "stage_id", "changed")
    with pytest.raises(IntegrationFingerprintMismatch, match="changed"):
        bundle_registration_fingerprint(registration)


@pytest.mark.parametrize("kind", ["place_relative", "place_upright"])
@pytest.mark.parametrize("stage_id", [" ", "\t", None, 0, False])
def test_only_exact_empty_stage_can_use_the_old_default(
    kind: str, stage_id: object
) -> None:
    with pytest.raises(ValueError, match="stage_id"):
        configured.decode_task_lowerer(
            {
                "kind": kind,
                "routes": [
                    {
                        "object_id": "can",
                        "reference_entity_id": "support_surface",
                        "relation": "on",
                        "world_displacement": [0.0, 0.0, 0.1],
                        "stage_id": stage_id,
                    }
                ],
            },
            path="test",
        )


@pytest.mark.parametrize(
    "field,value", [("source_hold_mode", "grasp_command"), ("stage_id", "")]
)
def test_bundle_identity_preserves_same_named_raw_metadata(
    monkeypatch: pytest.MonkeyPatch,
    deployment: configured.TaskDeployment,
    field: str,
    value: str,
) -> None:
    registration = deployment.integration.registration
    payload = registration.fingerprint_payload()
    type_name = (
        "embodichain.lab.sim.atomic_actions.primitives.hand_over.HandOverOptions"
        if field == "source_hold_mode"
        else "embodichain.gen_sim.task_engine._task_program.services._RelativePlaceRoute"
    )
    payload["metadata"] = {"type": type_name, "fields": {field: value}}
    before = deepcopy(payload)
    monkeypatch.setattr(
        SimulationTaskProgramRegistration,
        "fingerprint_payload",
        lambda _self: deepcopy(payload),
    )
    first = bundle_registration_fingerprint(registration)
    payload["metadata"]["fields"].pop(field)
    assert bundle_registration_fingerprint(registration) != first
    assert before["metadata"]["fields"][field] == value
