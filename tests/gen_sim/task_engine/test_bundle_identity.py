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
)
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
    default = _compose(monkeypatch, mode="grasp_command")
    composed = _compose(monkeypatch, mode=mode)
    identity = bundle_registration_fingerprint(composed.integration.registration)
    default_identity = bundle_registration_fingerprint(default.integration.registration)
    assert identity == composed.integration.registration.fingerprint
    if mode == "observed":
        assert identity != default_identity
        assert (
            composed.integration.integration_fingerprint
            != default.integration.integration_fingerprint
        )
    else:
        assert identity == default_identity
        assert (
            composed.integration.integration_fingerprint
            == default.integration.integration_fingerprint
        )
    composed.integration.registration.assert_unchanged()


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


def test_experimental_open_release_changes_native_evidence_identity(
    deployment: configured.TaskDeployment,
) -> None:
    from embodichain.gen_sim.task_engine._task_program.grip_evidence import (
        with_native_grip_evidence,
    )

    original = deployment.integration.registration
    ordinary = with_native_grip_evidence(original, experimental_open_release=False)
    experiment = with_native_grip_evidence(original, experimental_open_release=True)
    assert ordinary.control_part_evidence_factory.algorithm_revision == 1
    assert experiment.control_part_evidence_factory.algorithm_revision == 2
    assert bundle_registration_fingerprint(ordinary) != bundle_registration_fingerprint(
        experiment
    )
