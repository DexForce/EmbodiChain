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

"""Closed decoding for source-qualified prismatic semantic factories."""

from __future__ import annotations

from collections.abc import Mapping
from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding
from .simulation.prismatic_lowering import (
    PrismaticControlBinding,
    ArticulationSlideFactory,
    ArticulationWithdrawFactory,
    ArticulationParkFactory,
)

__all__: list[str] = []


def _decode_prismatic_lowerer(value: Mapping[str, object], *, path: str):
    from .configured import _mapping, _sequence

    kind = value["kind"]
    if kind == "articulation_park":
        _mapping(value, path=path, required=frozenset({"kind"}))
        return ArticulationParkFactory()
    config = _mapping(
        value, path=path, required=frozenset({"kind", "bindings", "table_uid"})
    )
    bindings = []
    for index, item in enumerate(
        _sequence(config["bindings"], path=f"{path}.bindings")
    ):
        raw = _mapping(
            item,
            path=f"{path}.bindings[{index}]",
            required=frozenset({"entity_id", "source"}),
        )
        bindings.append(
            PrismaticControlBinding(
                raw["entity_id"], PrismaticBinding.decode(raw["source"])
            )
        )
    cls = (
        ArticulationSlideFactory
        if kind == "prismatic_slide"
        else ArticulationWithdrawFactory
    )
    return cls(tuple(bindings), config["table_uid"])
