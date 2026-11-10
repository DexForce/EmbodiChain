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

"""Semantic gates shared by online and offline numeric data sources."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.data_pipeline.datasets import (
    LeRobotDataSource,
    OnlineDataSource,
    TensorDataSource,
    data_schema_fingerprint,
    validate_data_schema,
)

_SCHEMA = {
    "observation": {
        "names": ["position", "speed"],
        "dtype": "float32",
        "timing": "pre_action",
        "version": "features@1",
        "frame": "arena",
    },
    "action": {"names": ["request"], "semantics": "policy_request", "step_dt": 1 / 30},
}


def _data():
    return TensorDict(
        {
            "obs": torch.zeros(1, 2, 2),
            "actions": torch.zeros(1, 2, 1),
            "valid": torch.ones(1, 2, dtype=torch.bool),
        },
        batch_size=(1, 2),
    )


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("action", "semantics", None, "Explicit action semantics"),
        ("action", "semantics", " ", "Explicit action semantics"),
        ("action", "step_dt", 0, "positive action step_dt"),
        ("action", "step_dt", float("nan"), "positive action step_dt"),
        ("action", "step_dt", True, "positive action step_dt"),
        ("observation", "timing", "post_action", "pre_action"),
        ("observation", "names", ["position", "position"], "unique names"),
        ("observation", "dtype", "torch.float32", "dtype name"),
        ("observation", "version", False, "Schema version"),
    ],
)
def test_all_sources_reject_invalid_semantics_before_loading(
    tmp_path, monkeypatch, section, key, value, message
):
    schema = copy.deepcopy(_SCHEMA)
    schema[section][key] = value

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Invalid schema must be rejected before SDK loading")

    monkeypatch.setattr(LeRobotDataSource, "_load", unexpected_load)
    constructors = [
        lambda: OnlineDataSource(SimpleNamespace(), schema=schema),
        lambda: TensorDataSource(_data(), schema=schema, source_id="test"),
        lambda: LeRobotDataSource(
            tmp_path, schema=schema, source_id="test", memory_limit_bytes=1024
        ),
    ]
    for construct in constructors:
        with pytest.raises(ValueError, match=message):
            construct()


def test_schema_fingerprint_preserves_versions_and_feature_order():
    before = copy.deepcopy(_SCHEMA)
    fingerprint = data_schema_fingerprint(_SCHEMA)
    assert _SCHEMA == before
    reordered = dict(reversed(list(_SCHEMA.items())))
    assert data_schema_fingerprint(reordered) == fingerprint
    for field, value in (
        ("names", ["speed", "position"]),
        ("version", "features@2"),
        ("frame", "world"),
    ):
        changed = copy.deepcopy(_SCHEMA)
        changed["observation"][field] = value
        assert data_schema_fingerprint(changed) != fingerprint


@pytest.mark.parametrize("value", [float("inf"), object()])
def test_schema_fingerprint_rejects_unpersistable_metadata(value):
    schema = copy.deepcopy(_SCHEMA)
    schema["metadata"] = value
    with pytest.raises(ValueError, match="finite JSON"):
        data_schema_fingerprint(schema)


def test_explicit_representation_without_generic_semantics_is_supported():
    schema = copy.deepcopy(_SCHEMA)
    del schema["action"]["semantics"]
    schema["action"]["representation"] = "joint_position"
    validate_data_schema(schema)
    TensorDataSource(_data(), schema=schema, source_id="test")


def test_nested_feature_dtype_is_checked_without_flat_name_inference():
    data = _data()
    data["obs"] = TensorDict(
        {"position": data["obs"].double()}, batch_size=data.batch_size
    )
    schema = copy.deepcopy(_SCHEMA)
    del schema["observation"]["names"]
    with pytest.raises(ValueError, match="dtype mismatch"):
        TensorDataSource(data, schema=schema, source_id="test")
