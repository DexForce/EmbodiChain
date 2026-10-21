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

"""Tests for the configclass decorator."""

from __future__ import annotations

from dataclasses import MISSING, fields
from typing import ClassVar

import pytest
import torch

from embodichain.utils import configclass
from embodichain.utils.configclass import class_to_dict


@configclass
class _DeferredClassVarCfg:
    values: list[int] = []
    label: ClassVar[str] = "shared"


@configclass
class _PropertyCfg:
    values: list[int] = []

    @property
    def derived_values(self) -> list[int]:
        raise AssertionError("Configuration copying must not evaluate properties.")


@configclass
class _InheritedPropertyCfg(_PropertyCfg):
    label: str = "inherited"


class _PropertyOverriddenByFieldCfg(_PropertyCfg):
    derived_values: list[int] = [1]


@configclass
class _PostInitCfg:
    required: str = MISSING
    values: list[int] = []
    shared_values: ClassVar[list[int]] = [0]

    def __post_init__(self) -> None:
        self.values.append(1)


@pytest.mark.parametrize("cfg_type", [_PropertyCfg, _InheritedPropertyCfg])
def test_config_copying_does_not_evaluate_properties(cfg_type: type) -> None:
    supplied = [1]

    first = cfg_type(values=supplied)
    second = first.copy()
    first.values.append(2)

    assert supplied == [1]
    assert first.values == [1, 2]
    assert second.values == [1]


def test_field_overriding_an_inherited_property_is_copied() -> None:
    cfg = _PropertyOverriddenByFieldCfg()
    cfg.derived_values.append(2)

    assert _PropertyOverriddenByFieldCfg.derived_values == [1]
    assert cfg.derived_values == [1, 2]


def test_copying_preserves_post_init_missing_and_classvar_contracts() -> None:
    first = _PostInitCfg()
    second = _PostInitCfg()
    first.values.append(2)

    assert first.values == [1, 2]
    assert second.values == [1]
    assert first.required is second.required is MISSING
    assert first.shared_values is second.shared_values is _PostInitCfg.shared_values
    with pytest.raises(TypeError, match="required"):
        first.validate()


def test_deferred_classvar_is_not_converted_to_a_dataclass_field() -> None:
    first = _DeferredClassVarCfg()
    second = _DeferredClassVarCfg()
    first.values.append(1)

    assert [item.name for item in fields(_DeferredClassVarCfg)] == ["values"]
    assert first.to_dict() == {"values": [1]}
    assert second.values == []
    assert _DeferredClassVarCfg.label == "shared"


def test_legacy_serializer_preserves_nested_tensor_and_callable_contract() -> None:
    from types import SimpleNamespace

    from embodichain.utils.utility import class_to_dict as legacy_class_to_dict

    tensor = torch.tensor([1.0, 2.0])
    source = SimpleNamespace(
        nested={"items": (SimpleNamespace(tensor=tensor),)},
        callback=class_to_dict,
    )

    assert legacy_class_to_dict is class_to_dict
    result = legacy_class_to_dict(source)
    assert isinstance(result["nested"]["items"], tuple)
    assert result["nested"]["items"][0]["tensor"] is tensor
    assert result["callback"] == "embodichain.utils.configclass:class_to_dict"
