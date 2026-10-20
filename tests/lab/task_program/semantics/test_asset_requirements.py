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

"""Task asset role contract validation and snapshots."""

from __future__ import annotations

import json
import pytest

from embodichain.data.asset_catalog import AssetQuery
from embodichain.lab.task_program.semantics.asset_requirements import (
    AssetRequirements,
    AssetRoleRequirement,
    AssetRelationRequirement,
)


def _roles():
    return {
        "source": AssetRoleRequirement("cup", AssetQuery("cup")),
        "target": AssetRoleRequirement("tray", AssetQuery("tray")),
    }


def test_roles_snapshot_and_json_round_trip():
    roles = _roles()
    requirements = AssetRequirements(
        roles,
        (AssetRelationRequirement("fits_inside", "source", "target", 0.01),),
        {"mug": "cup"},
    )
    roles.clear()
    assert len(requirements.roles) == 2
    assert (
        AssetRequirements.from_dict(json.loads(json.dumps(requirements.to_dict())))
        == requirements
    )
    with pytest.raises(TypeError):
        requirements.roles["extra"] = requirements.roles["source"]


@pytest.mark.parametrize("aliases", [{"a": "a"}, {"a": "b", "b": "a"}])
def test_category_alias_cycles_rejected(aliases):
    with pytest.raises(ValueError, match="cycle"):
        AssetRequirements(_roles(), category_aliases=aliases)


def test_unknown_roles_duplicate_entities_and_fields_rejected():
    with pytest.raises(ValueError, match="unknown role"):
        AssetRequirements(
            _roles(), (AssetRelationRequirement("fits_inside", "source", "missing"),)
        )
    with pytest.raises(ValueError, match="distinct entity_id"):
        AssetRequirements({"a": _roles()["source"], "b": _roles()["source"]})
    with pytest.raises(ValueError, match="Unknown"):
        AssetRequirements.from_dict({"roles": {}, "typo": True})
    with pytest.raises(ValueError, match="Unknown"):
        AssetRoleRequirement.from_dict({"entity_id": "x", "query": {"typo": 1}})


@pytest.mark.parametrize("extent", [0, -1, float("nan"), True])
def test_invalid_max_extent_rejected(extent):
    with pytest.raises(ValueError):
        AssetRoleRequirement("cup", AssetQuery("cup"), extent)


@pytest.mark.parametrize("clearance", [-1, float("nan"), True])
def test_invalid_relation_clearance_rejected(clearance):
    with pytest.raises(ValueError):
        AssetRelationRequirement("fits_inside", "source", "target", clearance)
