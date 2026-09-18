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

from dataclasses import fields
from typing import ClassVar

from embodichain.utils import configclass


@configclass
class _DeferredClassVarCfg:
    values: list[int] = []
    label: ClassVar[str] = "shared"


def test_deferred_classvar_is_not_converted_to_a_dataclass_field() -> None:
    first = _DeferredClassVarCfg()
    second = _DeferredClassVarCfg()
    first.values.append(1)

    assert [item.name for item in fields(_DeferredClassVarCfg)] == ["values"]
    assert first.to_dict() == {"values": [1]}
    assert second.values == []
    assert _DeferredClassVarCfg.label == "shared"
