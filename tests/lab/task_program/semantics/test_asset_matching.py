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

"""Tests for task-semantic matching and bounded combination selection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from embodichain.data.asset_catalog import (
    AssetDescriptor,
    AssetQuery,
    Bounds3D,
    PreparedAsset,
)
from embodichain.lab.task_program.semantics.asset_requirements import (
    AssetRequirements,
    AssetRoleRequirement,
    AssetRelationRequirement,
)
from embodichain.lab.task_program.semantics.asset_matching import (
    AssetResolutionError,
    resolve_asset_set,
)


def _asset(
    name="cup",
    *,
    category="cup",
    size=0.10,
    attributes=None,
    capabilities=(),
    container=None,
    support=None,
):
    descriptor = AssetDescriptor(
        name,
        category=category,
        attributes=attributes or {},
        capabilities=capabilities,
        bounds_m=Bounds3D((0, 0, 0), (size, size, size)),
        normalized=True,
        container_region=container,
        support_region=support,
    )
    return PreparedAsset(
        descriptor,
        f"/assets/{name}/visual.obj",
        f"/assets/{name}/collision.obj",
        hashlib.sha256(name.encode()).hexdigest(),
    )


def _requirements(
    *, category="cup", attributes=None, capabilities=(), max_extent=None, aliases=None
):
    return AssetRequirements(
        {
            "source": AssetRoleRequirement(
                "source_object",
                AssetQuery(category, attributes or {}, capabilities),
                max_extent,
            )
        },
        category_aliases=aliases or {},
    )


def _pair_requirements(kind="fits_inside", clearance=0.0):
    return AssetRequirements(
        {
            "source": AssetRoleRequirement("source_object", AssetQuery("cup")),
            "target": AssetRoleRequirement("target_object", AssetQuery("tray")),
        },
        (AssetRelationRequirement(kind, "source", "target", clearance),),
    )


def test_matches_all_semantics_and_freezes_selection():
    requirements = _requirements(
        attributes={"color": "red"}, capabilities=("antipodal_grasp",), max_extent=0.15
    )
    selected = _asset(attributes={"color": "red"}, capabilities=("antipodal_grasp",))
    candidates = {"source": [selected]}
    result = resolve_asset_set(requirements, candidates)
    candidates["source"].clear()
    assert result.assets["source"] == selected
    with pytest.raises(TypeError):
        result.assets["source"] = selected
    with pytest.raises(FrozenInstanceError):
        result.seed = 1
    manifest = result.to_dict()
    manifest["assets"]["source"]["descriptor"]["attributes"]["color"] = "blue"
    assert result.assets["source"].descriptor.attributes["color"] == "red"


@pytest.mark.parametrize(
    ("asset", "requirements", "code"),
    [
        (_asset(category="bowl"), _requirements(), "category_mismatch"),
        (_asset(category=None), _requirements(), "category_mismatch"),
        (_asset(), _requirements(attributes={"color": "red"}), "attribute_mismatch"),
        (
            _asset(attributes={"color": "blue"}),
            _requirements(attributes={"color": "red"}),
            "attribute_mismatch",
        ),
        (
            _asset(attributes={"enabled": 1}),
            _requirements(attributes={"enabled": True}),
            "attribute_mismatch",
        ),
        (
            _asset(),
            _requirements(capabilities=("antipodal_grasp",)),
            "missing_capability",
        ),
        (_asset(size=0.20), _requirements(max_extent=0.15), "extent_exceeded"),
    ],
)
def test_rejects_unknown_or_incompatible_role_metadata(asset, requirements, code):
    with pytest.raises(AssetResolutionError) as captured:
        resolve_asset_set(requirements, {"source": [asset]})
    assert code in {item.code for item in captured.value.rejections}
    assert not captured.value.budget_exhausted


def test_category_aliases_are_explicit_and_transitive():
    mug = _asset(category="mug")
    requirements = _requirements(aliases={"mug": "drinking_cup", "drinking_cup": "cup"})
    assert resolve_asset_set(requirements, {"source": [mug]}).assets["source"] == mug
    with pytest.raises(AssetResolutionError):
        resolve_asset_set(_requirements(), {"source": [mug]})


def test_combination_rejects_individually_eligible_assets_that_do_not_fit():
    source = _asset(size=0.10)
    target = _asset(
        "tray",
        category="tray",
        size=0.30,
        container=Bounds3D((0, 0, 0), (0.09, 0.20, 0.20)),
    )
    with pytest.raises(AssetResolutionError) as captured:
        resolve_asset_set(
            _pair_requirements(), {"source": [source], "target": [target]}
        )
    rejection = next(
        item
        for item in captured.value.rejections
        if item.code == "relation_does_not_fit"
    )
    assert rejection.target_role == "target"
    assert "tray" in rejection.detail


def test_search_backtracks_to_valid_joint_combination():
    small = _asset("small", size=0.08)
    large = _asset("large", size=0.20)
    tray = _asset(
        "tray",
        category="tray",
        size=0.30,
        container=Bounds3D((0, 0, 0), (0.12, 0.12, 0.12)),
    )
    result = resolve_asset_set(
        _pair_requirements(clearance=0.01),
        {"source": [large, small], "target": [tray]},
        seed=27,
    )
    assert result.assets["source"] == small


def test_container_outer_bounds_do_not_prove_inner_space():
    with pytest.raises(AssetResolutionError) as captured:
        resolve_asset_set(
            _pair_requirements(),
            {
                "source": [_asset()],
                "target": [_asset("tray", category="tray", size=0.5)],
            },
        )
    assert "missing_relation_geometry" in {
        item.code for item in captured.value.rejections
    }


def test_support_checks_horizontal_footprint_not_object_height():
    support = _asset(
        "tray",
        category="tray",
        size=0.30,
        support=Bounds3D((0, 0, 0.30), (0.30, 0.30, 0.30)),
    )
    source = _asset()
    source = replace(
        source,
        descriptor=replace(
            source.descriptor, bounds_m=Bounds3D((0, 0, 0), (0.10, 0.10, 0.80))
        ),
    )
    result = resolve_asset_set(
        _pair_requirements("supported_by", 0.01),
        {"source": [source], "target": [support]},
    )
    assert result.assets["source"] == source
    with pytest.raises(AssetResolutionError):
        resolve_asset_set(
            _pair_requirements("supported_by", 0.11),
            {"source": [source], "target": [support]},
        )


def test_seeded_selection_independent_of_provider_and_role_order():
    sources = [_asset(f"cup_{index}") for index in range(8)]
    targets = [
        _asset(
            f"tray_{index}",
            category="tray",
            size=0.30,
            container=Bounds3D((0, 0, 0), (0.20, 0.20, 0.20)),
        )
        for index in range(3)
    ]
    first = resolve_asset_set(
        _pair_requirements(), {"source": sources, "target": targets}, seed=19
    )
    second = resolve_asset_set(
        _pair_requirements(),
        {"target": targets[::-1], "source": sources[::-1]},
        seed=19,
    )
    assert first.to_dict() == second.to_dict()


def test_identity_tracks_metadata_requirements_and_content_not_cache_path():
    requirements = _requirements()
    asset = _asset()
    original = resolve_asset_set(requirements, {"source": [asset]})
    moved = replace(
        asset, visual_path="/other/visual.obj", collision_path="/other/collision.obj"
    )
    assert (
        resolve_asset_set(requirements, {"source": [moved]}).set_id == original.set_id
    )
    changed = replace(asset, content_sha256="1" * 64)
    assert (
        resolve_asset_set(requirements, {"source": [changed]}).set_id != original.set_id
    )
    changed = replace(
        asset, descriptor=replace(asset.descriptor, attributes={"color": "red"})
    )
    assert (
        resolve_asset_set(requirements, {"source": [changed]}).set_id != original.set_id
    )
    assert (
        resolve_asset_set(
            _requirements(max_extent=1), {"source": [asset]}
        ).requirements_sha256
        != original.requirements_sha256
    )


def test_candidate_budget_reports_truncation():
    candidates = [_asset(f"bad_{index}", category="bowl") for index in range(5)]
    with pytest.raises(AssetResolutionError) as captured:
        resolve_asset_set(
            _requirements(), {"source": candidates}, max_candidates_per_role=2
        )
    assert captured.value.budget_exhausted
    assert (
        sum(item.code == "category_mismatch" for item in captured.value.rejections) == 2
    )
    assert "candidate_budget" in {item.code for item in captured.value.rejections}


def test_search_budget_stops_before_complete_assignment():
    target = _asset(
        "tray",
        category="tray",
        size=0.30,
        container=Bounds3D((0, 0, 0), (0.20, 0.20, 0.20)),
    )
    with pytest.raises(AssetResolutionError) as captured:
        resolve_asset_set(
            _pair_requirements(),
            {"source": [_asset()], "target": [target]},
            max_search_nodes=1,
        )
    assert captured.value.search_nodes == 1
    assert captured.value.budget_exhausted


def test_empty_role_pool_has_actionable_diagnostic():
    with pytest.raises(AssetResolutionError) as captured:
        resolve_asset_set(_requirements(), {"source": []})
    assert captured.value.rejections[0].role == "source"
    assert captured.value.rejections[0].code == "no_candidates"


@pytest.mark.parametrize(
    "kwargs", [{"max_search_nodes": 0}, {"max_candidates_per_role": -1}, {"seed": True}]
)
def test_invalid_budget_inputs_fail_early(kwargs):
    with pytest.raises(ValueError):
        resolve_asset_set(_requirements(), {"source": [_asset()]}, **kwargs)


def test_unexpected_role_is_not_silently_ignored():
    with pytest.raises(ValueError, match="exactly match"):
        resolve_asset_set(_requirements(), {"source": [_asset()], "typo": []})


def test_duplicate_content_with_different_preparation_provenance_is_stable():
    asset = _asset()
    other_source = replace(asset, provenance={"provider": "second"})
    first = resolve_asset_set(_requirements(), {"source": [asset, other_source]})
    second = resolve_asset_set(_requirements(), {"source": [other_source, asset]})
    assert first.to_dict() == second.to_dict()
