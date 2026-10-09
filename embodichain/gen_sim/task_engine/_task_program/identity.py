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

"""GenSim bundle identity at the v38 declaration boundary."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, cast

from ..contracts import canonical_hash

if TYPE_CHECKING:
    from embodichain.lab.task_program.integrations.catalog import (
        SimulationTaskProgramRegistration,
    )

__all__: list[str] = []

_PRESET_TYPE = "embodichain.lab.task_program.semantics.profiles.SkillPolicyPreset"
_HANDOVER_TYPE = (
    "embodichain.lab.sim.atomic_actions.primitives.hand_over.HandOverOptions"
)
_PLACE_ROUTE_TYPE = (
    "embodichain.gen_sim.task_engine._task_program.services._RelativePlaceRoute"
)
_PLACE_FACTORY_TYPES = frozenset(
    {
        "embodichain.gen_sim.task_engine._task_program.services._RelativePlaceLowererFactory",
        "embodichain.gen_sim.task_engine._task_program.services._UprightPlaceLowererFactory",
        "embodichain.gen_sim.task_engine._task_program.stack_place._StackPlaceFactory",
    }
)


def bundle_registration_fingerprint(
    registration: SimulationTaskProgramRegistration,
) -> str:
    """Project only v38-equivalent defaults at their validated declaration slots.

    Full runtime identity is validated before projection. Other fields, explicit
    new behaviors, factory revisions and arbitrary metadata remain hashed.
    """
    payload = cast(dict[str, Any], registration.fingerprint_payload())
    # Never recursively strip type-shaped metadata: only real option slots qualify.
    for profile_key in ("robot_profile", "robot_profile_binding"):
        presets = payload[profile_key]["fields"]["presets"]
        for preset in (presets.values() if isinstance(presets, dict) else presets):
            if preset["type"] != _PRESET_TYPE:
                continue
            for option in preset["fields"]["_action_option_templates"].values():
                if option["type"] == _HANDOVER_TYPE:
                    fields = option["fields"]
                    if fields.get("source_hold_mode") == "grasp_command":
                        del fields["source_hold_mode"]
    for item in payload["registered_semantic_lowerer_factories"]:
        provider = item["provider"]
        if provider.get("type") not in _PLACE_FACTORY_TYPES:
            continue
        for route in provider["fields"]["routes"]:
            if route["type"] == _PLACE_ROUTE_TYPE:
                fields = route["fields"]
                if fields.get("stage_id") == "":
                    del fields["stage_id"]
    return canonical_hash(payload)
