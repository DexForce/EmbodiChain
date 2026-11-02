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
"""Bind control occurrences to the same static alias contract as GenSim Pick."""

from __future__ import annotations

from dataclasses import make_dataclass
import hashlib
from typing import Any

__all__: list[str] = []


def make_control_factory(factory: type, route: Any) -> Any:
    """Create only trusted provider types; data supplies an occurrence identity."""
    if not route.operation_id:
        return factory(route)
    call_id = route.call_id(factory.call_id)
    suffix = hashlib.sha256(call_id.encode("utf-8")).hexdigest()[:16]
    lowerer = factory.lowerer_type
    if factory.call_id == "gen_sim.press" and route.variant == "approach_only":
        from .press_runtime import _ApproachLowerer

        lowerer = _ApproachLowerer
    lowerer_type = type(
        f"_ControlLowerer_{suffix}",
        (lowerer,),
        {"call_id": call_id, "__module__": __name__, "__slots__": ()},
    )
    factory_type = make_dataclass(
        f"_ControlFactory_{suffix}",
        [],
        bases=(factory,),
        namespace={
            "call_id": call_id,
            "lowerer_type": lowerer_type,
            "__module__": __name__,
        },
        frozen=True,
        slots=True,
    )
    return factory_type(route)
