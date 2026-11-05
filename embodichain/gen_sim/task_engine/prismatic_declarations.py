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

"""Source-qualified prismatic operation declarations for semantic bundle generation."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from embodichain.lab.sim.scene.articulation_inputs import (
    inspect_prismatic,
    inspect_prismatic_part,
)
from embodichain.lab.sim.scene.prismatic_control import validate_prismatic_placement
from embodichain.lab.task_program.integrations.simulation.prismatic_lowering import (
    PrismaticControlBinding,
)
from embodichain.lab.task_program.integrations.simulation.stability import (
    SimulationJointStabilityConditionCfg,
)
from .program_declarations import ProgramSegmentBinding
from .semantic_graph import SemanticTaskGraph

__all__ = ["PrismaticProgramDeclarations", "resolve_prismatic_program_declarations"]


@dataclass(frozen=True, slots=True)
class PrismaticProgramDeclarations:
    """Resolved source/link bindings, named conditions and complete group checks."""

    bindings: tuple[PrismaticControlBinding, ...]
    conditions: Mapping[str, SimulationJointStabilityConditionCfg]
    operation_groups: Mapping[str, tuple[ProgramSegmentBinding, ...]]

    def link_declarations(self) -> list[dict[str, str]]:
        """Return the qualified semantic/native link binding declarations."""
        return [
            dict(
                entity_id=item.entity_id,
                articulation_id=item.source.object_id,
                native_link_name=item.source.link,
            )
            for item in self.bindings
        ]

    def lowerer_declarations(self, *, table_uid: str) -> list[dict[str, Any]]:
        """Return the ordinary public source-control factories and explicit table.

        Args:
            table_uid: Exact native rigid support UID, never inferred at runtime.

        Returns:
            Closed public prismatic Slide/withdraw/Park factory declarations.
        """
        if not self.bindings:
            return []
        return [
            dict(
                kind=kind,
                bindings=[
                    dict(entity_id=item.entity_id, source=item.source.payload())
                    for item in self.bindings
                ],
                table_uid=table_uid,
            )
            for kind in ("prismatic_slide", "prismatic_withdraw")
        ] + [{"kind": "articulation_park"}]


def resolve_prismatic_program_declarations(
    graph: SemanticTaskGraph, scene: Any
) -> PrismaticProgramDeclarations:
    """Resolve complete serial E6 calls through shared authored source inspectors.

    Args:
        graph: Semantic graph with explicit group ownership/dependencies.
        scene: Frozen source scene, articulation configurations and measured table.

    Returns:
        Source-qualified control bindings and standard policy/check declarations.
        The original three-call recipe, explicit roles, target tolerance and final
        joint validator meaning are retained; no native environment is constructed.
    """
    nodes = {node["id"]: node for node in graph["nodes"]}
    configs = {value["uid"]: value for value in scene.articulations}
    bindings, conditions, groups = {}, {}, {}
    for group in graph["task_groups"]:
        if group["task_type"] != "E6":
            continue
        sequence = [nodes[key] for key in group["node_ids"]]
        if len(sequence) != 3:
            raise ValueError("E6 requires the complete Slide/withdraw/Park recipe.")
        first = sequence[0]["call"]
        args = first.get("arguments", {})
        if set(args) not in ({"object", "state"}, {"object", "state", "part"}):
            raise ValueError("E6 requires object/state and an optional part.")
        resource = first.get("resources", {}).get("primary")
        if (
            args["state"] not in {"open", "closed"}
            or type(resource) is not str
            or not resource
        ):
            raise ValueError("E6 requires an open/closed target and explicit resource.")
        expected = [
            dict(
                kind="registered",
                call_id=identity,
                arguments=dict(args),
                resources={"primary": resource},
            )
            for identity in (
                "gen_sim.articulation_slide",
                "gen_sim.articulation_withdraw",
            )
        ] + [
            dict(
                kind="registered",
                call_id="gen_sim.articulation_park",
                arguments={},
                resources={"primary": resource},
            )
        ]
        if [node["call"] for node in sequence] != expected or [
            node["role"] for node in sequence
        ] != ["primary", "cleanup", "cleanup"]:
            raise ValueError(
                "E6 requires complete explicit interaction and cleanup calls."
            )
        if any(
            current["depends_on"] != [previous["id"]]
            for previous, current in zip(sequence, sequence[1:])
        ):
            raise ValueError(
                "E6 withdrawal and Park must follow their preceding action."
            )
        uid = args["object"]
        part = args.get("part")
        if uid not in configs:
            raise ValueError(
                f"E6 articulation {uid!r} is absent from the prepared scene."
            )
        key = uid if part is None else f"{uid}::{part}"
        if key not in bindings:
            source = (
                inspect_prismatic(configs[uid])
                if part is None
                else inspect_prismatic_part(configs[uid], part)
            )
            validate_prismatic_placement(source, configs[uid], scene.table_top_z)
            bindings[key] = PrismaticControlBinding(
                f"{source.object_id}__{source.link}", source
            )
        binding = bindings[key].source
        if args["state"] == "closed" and binding.closed_position is None:
            raise ValueError(
                "Closed targets require an explicitly authored closed endpoint."
            )
        target = binding.target(args["state"])
        # This is the source recipe's declared acceptance margin, not a geometry
        # property or a native joint-limit correction.
        tolerance = min(0.03, (binding.limits[1] - binding.limits[0]) * 0.49)
        suffix = f".{binding.part_id}" if binding.part_id else ""
        preset = f"gen_sim.articulation.{binding.object_id}{suffix}.{args['state']}"
        condition = SimulationJointStabilityConditionCfg(
            entity=binding.object_id,
            joint=binding.joint,
            target=target,
            tolerance=tolerance,
        ).snapshot()
        previous = conditions.get(preset)
        if previous is not None and previous.to_dict() != condition.to_dict():
            raise ValueError(
                "One prismatic preset identity cannot own different conditions."
            )
        conditions[preset] = condition
        post = dict(kind="wait_stable", entity=binding.object_id, preset=preset)
        validator = dict(
            kind="articulation_joint_position",
            articulation=binding.object_id,
            joint=binding.joint,
            minimum_position=target - tolerance,
            maximum_position=target + tolerance,
        )
        groups[group["id"]] = tuple(
            ProgramSegmentBinding(node["id"], post=(post,), validators=(validator,))
            for node in sequence
        )
    return PrismaticProgramDeclarations(tuple(bindings.values()), conditions, groups)
