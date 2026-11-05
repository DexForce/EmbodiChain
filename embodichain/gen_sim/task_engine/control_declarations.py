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

"""Bind semantic Press/Twist recipes to shared source, policy and evidence owners."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
import math
from embodichain.lab.sim.scene.press_inputs import discover_press_source
from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
    TwistPlanningRecovery,
    required_twist_occurrences,
)
from embodichain.lab.task_program.integrations.simulation.control_assembly import (
    ControlOperationDeclaration,
)
from embodichain.lab.task_program.integrations.simulation.control_lowering import (
    PressCallDeclaration,
    TwistCallDeclaration,
)
from embodichain.lab.task_program.integrations.simulation.control_policy import (
    SimulationControlConditionCfg,
)
from embodichain.lab.task_program.integrations.hand_calibration import (
    TaskHandCalibration,
)
from .program_declarations import ProgramSegmentBinding
from .semantic_graph import SemanticTaskGraph

__all__ = ["ControlProgramDeclarations", "resolve_control_program_declarations"]


@dataclass(frozen=True, slots=True)
class ControlProgramDeclarations:
    """Pure owner declarations for source-qualified control operation groups."""

    operations: tuple[ControlOperationDeclaration, ...]
    lowerers: tuple[Mapping[str, Any], ...]
    conditions: Mapping[str, SimulationControlConditionCfg]
    operation_groups: Mapping[str, tuple[ProgramSegmentBinding, ...]]
    action_options: Mapping[str, Mapping[str, Any]]
    sensors: tuple[Mapping[str, Any], ...]
    post_policy_presets: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def link_declarations(self) -> list[dict[str, str]]:
        """Return qualified parent/native child links in declared operation order."""
        selected = {value.call.entity_id: value for value in self.operations}
        return [
            dict(
                entity_id=identity,
                articulation_id=value.binding.object_id,
                native_link_name=value.binding.link,
            )
            for identity, value in selected.items()
        ]

    def runtime_declarations(self) -> list[dict[str, Any]]:
        """Serialize through the existing closed control-operation schema."""
        result = []
        for operation in self.operations:
            press = type(operation.call) is PressCallDeclaration
            binding = asdict(operation.binding)
            binding.pop("source_path", None)
            value = dict(
                kind="press" if press else "twist",
                **asdict(operation.call),
                binding=binding,
                sensor_uid=operation.sensor_uid,
                grasp_target_id=operation.grasp_target_id,
                finger_link_names=list(operation.finger_link_names),
                arm_label=operation.arm_label,
            )
            if press:
                value["moving_mass"] = operation.moving_mass
            else:
                value.update(
                    table_uid=operation.table_uid,
                    chunk_count=operation.chunk_count,
                    success_criterion=operation.success_criterion,
                )
            result.append(value)
        return result


def resolve_control_program_declarations(
    graph: SemanticTaskGraph,
    scene: Any,
    *,
    robot_uid: str,
    hands: Sequence[TaskHandCalibration],
    table_uid: str,
    press_extra_distance: float | None = None,
    twist_chunk_angles: Mapping[str, float] | None = None,
) -> ControlProgramDeclarations:
    """Resolve explicit source controls into canonical shared runtime declarations.

    Args:
        graph: Source graph with complete operation groups and dependencies.
        scene: Frozen authored articulation configuration owner.
        robot_uid: Exact physical robot identity for the full source contact query.
        hands: Explicit model/embodiment-owned geometry and finger roles.
        table_uid: Exact native rigid support selected for Twist geometry.
        press_extra_distance: Request-owned overtravel, or source default 0.010 m.
        twist_chunk_angles: Explicit operation-angle overrides for source coarse
            requests. Their occurrence count still uses the shared fixed rule.

    Returns:
        Existing public control declarations, standard sensor mappings, typed
        phase policies and original program bindings. Child-link identities and
        separate terminal criteria stay in integration-owned post-policy preset
        selections; program root entities and reused preset strings remain unchanged.
    """
    if press_extra_distance is not None and (
        type(press_extra_distance) not in (int, float)
        or not math.isfinite(press_extra_distance)
        or not 0 <= press_extra_distance <= 0.010
    ):
        raise ValueError("press_extra_distance must be within [0,.010] metres.")
    angles = {} if twist_chunk_angles is None else dict(twist_chunk_angles)
    nodes = {node["id"]: node for node in graph["nodes"]}
    configs = {config["uid"]: config for config in scene.articulations}
    by_resource = {hand.resource_id: hand for hand in hands}
    if len(by_resource) != len(hands):
        raise ValueError("Control hand resources must be unique.")
    post_policy_presets = {}
    operations = []
    lowerers = []
    conditions = {}
    groups = {}
    options = {}
    sensors = []
    used_angles = set()
    for group in graph["task_groups"]:
        if group["task_type"] not in {"E8", "E9"}:
            continue
        press = group["task_type"] == "E9"
        family = "press" if press else "twist"
        sequence = [nodes[key] for key in group["node_ids"]]
        first = sequence[0]["call"]
        args = first.get("arguments", {})
        if set(args) != ({"object"} if press else {"object", "setting"}):
            raise ValueError(
                "Control recipe requires its exact object/setting arguments."
            )
        uid = args["object"]
        resource = first.get("resources", {}).get("primary")
        if uid not in configs or resource not in by_resource:
            raise ValueError(
                "Control recipe requires declared articulation and hand resources."
            )
        hand = by_resource[resource]
        prepare_base = f"gen_sim.{family}_prepare"
        action_base = f"gen_sim.{family}"
        suffix = "" if first.get("call_id") == prepare_base else group["id"]
        operation_id = group["id"]
        prepare_id = prepare_base + ("." + suffix if suffix else "")
        action_id = action_base + ("." + suffix if suffix else "")
        if press:
            binding, mass = discover_press_source(configs[uid])
            count = 1
            entity = f"{uid}__{binding.link}"
            call = PressCallDeclaration(
                operation_id,
                entity,
                extra_press_distance=(
                    0.010 if press_extra_distance is None else press_extra_distance
                ),
            )
            operation = ControlOperationDeclaration(
                call,
                binding,
                f"gen_sim_press_evidence" + ("." + suffix if suffix else ""),
                hand.grasp_target_id,
                hand.finger_link_names,
                hand.side,
                moving_mass=mass,
            )
            options[action_id] = dict(
                kind="press",
                approach_distance=0.08,
                press_distance=binding.stroke + call.extra_press_distance,
                hand_interp_steps=12,
            )
        else:
            binding = inspect_revolute_control(configs[uid], args["setting"])
            count = required_twist_occurrences(
                binding.target_qpos, TwistPlanningRecovery()
            )
            entity = f"{uid}::{binding.link}"
            angle = angles.get(operation_id, math.radians(5.0))
            if operation_id in angles:
                used_angles.add(operation_id)
            call = TwistCallDeclaration(operation_id, entity, chunk_angle=angle)
            operation = ControlOperationDeclaration(
                call,
                binding,
                f"gen_sim_twist_evidence" + ("." + suffix if suffix else ""),
                hand.grasp_target_id,
                hand.finger_link_names,
                hand.side,
                table_uid=table_uid,
                chunk_count=count,
            )
            options[action_id] = dict(
                kind="twist",
                hand_interp_steps=12,
                pre_grasp_distance=0.10,
                twist_angle=binding.target_qpos / binding.axis_sign,
                max_twist_angle=call.chunk_angle,
                contact_bite_depth=min(call.grasp_depth, binding.grip_depth / 2),
                contact_settle_steps=call.settle_steps,
                contact_roll=call.fixed_roll,
            )
        expected = [
            dict(
                kind="registered",
                call_id=identity,
                arguments=dict(args),
                resources={"primary": resource},
            )
            for identity in [prepare_id, *([action_id] * count)]
        ] + [
            dict(
                kind="registered",
                call_id="gen_sim.articulation_park",
                arguments={},
                resources={"primary": resource},
            )
        ]
        if (
            [node["call"] for node in sequence] != expected
            or [node["role"] for node in sequence]
            != ["primary"] * (count + 1) + ["cleanup"]
            or any(
                current["depends_on"] != [before["id"]]
                for before, current in zip(sequence, sequence[1:])
            )
        ):
            raise ValueError(
                "Control recipe requires complete prepare/actions/Park order and roles."
            )
        operations.append(operation)
        lowerers.extend(
            dict(kind=kind, operation_id=operation_id, operation_suffix=suffix)
            for kind in (family + "_prepare", family)
        )
        options[prepare_id] = {"kind": "move_joints"}
        group_bindings = []
        for index, node in enumerate(sequence):
            terminal = index == len(sequence) - 1
            phase = (
                "ready"
                if index == 0
                else "released" if press or index >= len(sequence) - 2 else "chunk"
            )
            phase_name = (
                "ready"
                if index == 0
                else (
                    "terminal"
                    if terminal
                    else (
                        "pressed"
                        if press
                        else "turned" if index == len(sequence) - 2 else "chunk"
                    )
                )
            )
            preset = f"gen_sim.{family}.{suffix or uid}.{phase_name}"
            conditions[preset] = SimulationControlConditionCfg(
                entity=entity,
                operation_id=operation_id,
                family=family,
                phase=phase,
                terminal=terminal,
            ).snapshot()
            legacy_phase = (
                "ready"
                if index == 0
                else (
                    "pressed"
                    if press
                    else "turned" if index >= len(sequence) - 2 else "chunk"
                )
            )
            legacy_preset = f"gen_sim.{family}.{suffix or uid}.{legacy_phase}"
            group_bindings.append(
                ProgramSegmentBinding(
                    node["id"],
                    post=(dict(kind="wait_stable", entity=uid, preset=legacy_preset),),
                )
            )
            post_policy_presets[node["id"]] = (preset,)
        groups[group["id"]] = tuple(group_bindings)
        sensors.append(
            dict(
                uid=operation.sensor_uid,
                sensor_type="ContactSensor",
                max_contacts_per_env=512,
                filter_need_both_actor=press,
                rigid_uid_list=[],
                articulation_cfg_list=[
                    dict(articulation_uid=robot_uid, link_name_list=[]),
                    dict(articulation_uid=uid, link_name_list=[]),
                ],
            )
        )
    if set(angles) != used_angles:
        raise ValueError("Twist angle declarations contain unknown operations.")
    return ControlProgramDeclarations(
        tuple(operations),
        tuple(lowerers),
        conditions,
        groups,
        options,
        tuple(sensors),
        post_policy_presets,
    )
