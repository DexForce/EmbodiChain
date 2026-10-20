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
from embodichain.utils.config_paths import resolve_config_path

# Main 04020afc0 changes endpoint names, motion limits, and validator declarations.
# Keep an independent golden for that declaration epoch, not an old-scene claim.
_PRE_MAIN_REGISTRATION = (
    "1992baee8c923e814d096d18cee0e5d80575efeac1e1d23558f655d581984c00"
)
_MAIN_REGISTRATION = "ca807ab4cd63103c024283efbaa928375cda2b54f2f95a7524d9e12f72f5f11e"
_PRE_MAIN_CORE = "87f6a3a11d4b3fe1f143c0263b398f654a54a06edb3ef1f1529dd36fee46b3cf"
_MAIN_CORE = "b28c58f4d801769e1bf0f6ae43c3077b60481f2ba7dd12a9fda2f26b200ae54e"
_ROUTE_CASES = [
    (
        _RelativePlaceLowererFactory,
        "45cb1095ec0d5a6cfcdb7ba3a40f2b43df27692a13612723f1bb1c7bea3b997e",
    ),
    (
        _UprightPlaceLowererFactory,
        "36ca52e06630fa9edc49f4ae0e4f904eaeb0ffc8353e2923c33dddff6e7f5ccd",
    ),
    (
        _StackPlaceFactory,
        "59f51e9f682df752da86605f98a4d20157728dc6dc7109048bfa471b838be70e",
    ),
]


def _compose(
    monkeypatch: pytest.MonkeyPatch, *, mode: str | None = None
) -> configured.TaskDeployment:
    path = resolve_config_path(
        "extensions/embodichain_tasks/configs/tasks/manipulation/hand_over/task.dual_ur5_dh_pgi_140_80.yaml"
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
def test_handover_bundle_identity_preserves_default_and_explicit_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    composed = _compose(monkeypatch, mode=mode)
    identity = bundle_registration_fingerprint(composed.integration.registration)
    if mode == "observed":
        assert identity != _MAIN_REGISTRATION
        assert composed.integration.integration_fingerprint != _MAIN_CORE
    else:
        assert identity == _MAIN_REGISTRATION
        assert composed.integration.integration_fingerprint == _MAIN_CORE
    assert identity != _PRE_MAIN_REGISTRATION
    assert composed.integration.integration_fingerprint != _PRE_MAIN_CORE
    assert composed.integration.registration.fingerprint != _MAIN_REGISTRATION
    composed.integration.registration.assert_unchanged()


@pytest.mark.parametrize("factory_type,main_digest", _ROUTE_CASES)
def test_empty_stage_preserves_route_identity_but_real_changes_do_not(
    deployment: configured.TaskDeployment,
    factory_type: type[_RelativePlaceLowererFactory],
    main_digest: str,
) -> None:
    registration = _with_route(deployment.integration.registration, factory_type)
    assert bundle_registration_fingerprint(registration) == main_digest
    for changed in (
        _with_route(
            deployment.integration.registration, factory_type, stage_id="step_02"
        ),
        _with_route(deployment.integration.registration, factory_type, height=0.2),
    ):
        assert bundle_registration_fingerprint(changed) != main_digest
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


def test_explicit_reference_links_remain_in_bundle_identity(deployment):
    registration = _with_route(
        deployment.integration.registration, _RelativePlaceLowererFactory
    )
    factory = registration.registered_semantic_lowerer_factories[-1]
    baseline = bundle_registration_fingerprint(registration)
    identities = []
    for link in ("anchor__body", "anchor__drawer"):
        changed = replace(
            factory, routes=(replace(factory.routes[0], reference_link_id=link),)
        )
        bound = replace(
            registration,
            registered_semantic_lowerer_factories=(
                *registration.registered_semantic_lowerer_factories[:-1],
                changed,
            ),
        )
        identities.append(bundle_registration_fingerprint(bound))
    assert baseline not in identities
    assert identities[0] != identities[1]


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
