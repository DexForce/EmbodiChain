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

"""Behavioral tests for rigid asset catalog snapshots."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from embodichain.data.asset_catalog import (
    AssetCandidate,
    AssetDescriptor,
    AssetQuery,
    Bounds3D,
    PreparedAsset,
    PreparationContext,
)


def _descriptor(**kwargs):
    values = dict(
        asset_id="cup-1",
        category="cup",
        bounds_m=Bounds3D((0, 0, 0), (0.1, 0.1, 0.2)),
        normalized=True,
    )
    values.update(kwargs)
    return AssetDescriptor(**values)


def test_descriptor_snapshots_nested_evidence_and_round_trips():
    evidence = {"category": {"kind": "declared", "sources": ["metadata.json"]}}
    descriptor = _descriptor(provenance=evidence, attributes={"color": "red"})
    evidence["category"]["sources"].append("later.json")
    assert descriptor.provenance["category"]["sources"] == ("metadata.json",)
    with pytest.raises(TypeError):
        descriptor.provenance["category"]["kind"] = "verified"
    snapshot = json.loads(json.dumps(descriptor.to_dict()))
    assert AssetDescriptor.from_dict(snapshot) == descriptor
    snapshot["attributes"]["color"] = "blue"
    assert descriptor.attributes["color"] == "red"


def test_candidate_and_prepared_round_trip_without_io():
    descriptor = _descriptor()
    candidate = AssetCandidate("candidate", "provider://cup", descriptor)
    assert AssetCandidate.from_dict(candidate.to_dict()) == candidate
    prepared = PreparedAsset(
        descriptor, "/cache/mesh.glb", "/cache/collision.obj", "a" * 64
    )
    assert PreparedAsset.from_dict(prepared.to_dict()) == prepared
    with pytest.raises(FrozenInstanceError):
        prepared.visual_path = "/other.glb"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_geometry_rejected(value):
    with pytest.raises(ValueError):
        Bounds3D((0, 0, 0), (1, 1, value))


@pytest.mark.parametrize(
    "field,value",
    [
        ("bounds_m", Bounds3D((0, 0, 0), (1, 1, 0))),
        ("container_region", Bounds3D((0, 0, 0), (1, 1, 0))),
        ("support_region", Bounds3D((0, 0, 0), (1, 1, 1))),
        ("attributes", {"invalid": float("nan")}),
        ("attributes", {"invalid": ["nested"]}),
        ("capabilities", ["container", "container"]),
    ],
)
def test_invalid_descriptor_fields_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        _descriptor(**{field: value})


def test_unknown_fields_and_unnormalized_preparation_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        AssetDescriptor.from_dict({"asset_id": "a", "unrecognized": 2})
    with pytest.raises(ValueError, match="normalized"):
        PreparedAsset(AssetDescriptor("a"), "/model.glb", "/collision.obj", "a" * 64)
    with pytest.raises(ValueError, match="absolute"):
        PreparedAsset(_descriptor(), "relative.glb", "/collision.obj", "a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        PreparedAsset(_descriptor(), "/model.glb", "/collision.obj", "short")


def test_query_and_context_are_strict_serializable_records():
    query = AssetQuery("cup", {"color": "red"}, ["antipodal_grasp"])
    assert query.capabilities == ("antipodal_grasp",)
    assert AssetQuery.from_dict(query.to_dict()) == query
    context = PreparationContext("/cache/assets")
    assert PreparationContext.from_dict(context.to_dict()) == context
    with pytest.raises(ValueError, match="absolute"):
        PreparationContext("relative")


def test_geometry_rejects_overflowing_extents():
    with pytest.raises(ValueError, match="extents must be finite"):
        Bounds3D((-1e308, 0, 0), (1e308, 1, 1))
