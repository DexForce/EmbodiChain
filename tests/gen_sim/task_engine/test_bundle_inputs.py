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


"""Source provenance seals owners without selecting any runtime policy."""

from __future__ import annotations

from copy import deepcopy
import json
import pytest
from embodichain.gen_sim.task_engine.bundle_deployment import prepare_configured_bundle
from embodichain.gen_sim.task_engine.bundle_io import BundleSnapshot
from embodichain.gen_sim.task_engine.bundle_inputs import (
    SOURCE_INPUT_PROVENANCE_FILE,
    SOURCE_INPUT_PROVENANCE_SCHEMA,
)
from .test_bundle_deployment import _files


def declared():
    values = {
        "physical.mesh": "a" * 64,
        "hand.yaml": "b" * 64,
        "model.urdf": "c" * 64,
        "annotation.json": "d" * 64,
    }
    provenance = {
        "schema_version": SOURCE_INPUT_PROVENANCE_SCHEMA,
        "groups": {
            "physical": {
                "physical.mesh": values["physical.mesh"],
                "model.urdf": values["model.urdf"],
            },
            "hand_services": {"hand.yaml": values["hand.yaml"]},
            "additional": {"annotation.json": values["annotation.json"]},
        },
    }
    return values, provenance


def prepare(tmp_path, inputs, provenance):
    return prepare_configured_bundle(
        _files(),
        base_dir=tmp_path,
        external_input_digests=inputs,
        implementation_revisions={"test": "fixed"},
        source_input_provenance=provenance,
    )


def test_provenance_seals_complete_owner_groups_and_owns_values(tmp_path):
    inputs, provenance = declared()
    provenance["groups"]["additional"]["hand.yaml"] = inputs["hand.yaml"]
    snapshot = prepare(tmp_path, inputs, provenance)
    original = json.loads(snapshot.files[SOURCE_INPUT_PROVENANCE_FILE])
    provenance["groups"]["physical"].clear()
    assert json.loads(snapshot.files[SOURCE_INPUT_PROVENANCE_FILE]) == original
    assert snapshot.identity.to_dict()["external_inputs"] == inputs
    assert (
        snapshot.identity.to_dict()["schema_version"]
        == "semantic_integration_fingerprint/v4"
    )
    assert (
        snapshot.identity.to_dict()["adapter_contract"]
        == "gen_sim.task_program/shared/v2"
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_hand",
        "extra_hand",
        "conflicting_hash",
        "configuration_value",
        "extra_field",
    ],
)
def test_provenance_rejects_missing_conflicting_or_non_inventory_data_before_preflight(
    tmp_path, monkeypatch, damage
):
    from embodichain.gen_sim.task_engine import bundle_deployment

    inputs, provenance = declared()
    if damage == "missing_hand":
        provenance["groups"]["hand_services"].clear()
    elif damage == "extra_hand":
        provenance["groups"]["hand_services"]["other.yaml"] = "e" * 64
    elif damage == "conflicting_hash":
        provenance["groups"]["physical"]["hand.yaml"] = "e" * 64
    elif damage == "configuration_value":
        provenance["groups"]["additional"]["policy"] = {"retries": 3}
    else:
        provenance["runtime_profile"] = "fallback"
    monkeypatch.setattr(
        bundle_deployment,
        "_preflight",
        lambda *a, **k: pytest.fail("Invalid provenance reached assembly"),
    )
    with pytest.raises((ValueError, TypeError)):
        prepare(tmp_path, inputs, provenance)


def test_metadata_only_owner_change_needs_new_identity_and_does_not_change_task_semantics(
    tmp_path,
):
    inputs, provenance = declared()
    first = prepare(tmp_path, inputs, provenance)
    changed = deepcopy(provenance)
    changed["groups"]["declared_source"] = changed["groups"].pop("additional")
    second = prepare(tmp_path, inputs, changed)
    assert first.identity.digest != second.identity.digest
    for key in (
        "task_spec_digest",
        "program_fingerprint",
        "integration_fingerprint",
        "registration_fingerprint",
    ):
        assert first.identity.to_dict()[key] == second.identity.to_dict()[key]
    files = dict(first.files)
    files[SOURCE_INPUT_PROVENANCE_FILE] = second.files[SOURCE_INPUT_PROVENANCE_FILE]
    with pytest.raises(ValueError, match="file changed"):
        BundleSnapshot(first.identity, files)


def test_low_level_application_owned_inputs_receive_an_explicit_default_group(tmp_path):
    snapshot = prepare_configured_bundle(
        _files(),
        base_dir=tmp_path,
        external_input_digests={"qualified-owner": "a" * 64},
        implementation_revisions={"test": "fixed"},
    )
    assert json.loads(snapshot.files[SOURCE_INPUT_PROVENANCE_FILE]) == {
        "schema_version": SOURCE_INPUT_PROVENANCE_SCHEMA,
        "groups": {"application": {"qualified-owner": "a" * 64}},
    }
