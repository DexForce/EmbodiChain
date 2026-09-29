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

"""Bind generated motion budgets to immutable requests, not program defaults."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from embodichain.lab.sim.atomic_actions import AtomicActionEngine
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionInvocation,
    ResolvedActionRequest,
)

__all__: list[str] = []

INVOCATION_POLICY_REVISION = 3


def motion_sample_declarations(
    graph: dict[str, Any], *, drawer_groups: frozenset[str] = frozenset()
) -> dict[str, int]:
    """Select budgets from each recipe's own declared intent, including cleanup."""
    staged = set(drawer_groups)
    for node in graph["nodes"]:
        call = node["call"]
        if (
            node["task_type"] in {"E2", "E4", "E6"}
            or call["kind"] == "hand_over"
            or (
                call.get("call_id") == "simulation.coordinated_transport"
                and call.get("arguments", {}).get("relation") == "on"
            )
        ):
            staged.add(node.get("task_instance_id", node["id"]))
    return {
        node["id"]: 260 if node.get("task_instance_id", node["id"]) in staged else 140
        for node in graph["nodes"]
    }


def cartesian_call_declarations(
    graph: dict[str, Any], *, drawer_groups: frozenset[str] = frozenset()
) -> list[str]:
    """Declare Cartesian sampling only for recipes whose own motion requires it."""
    selected = set(drawer_groups)
    for node in graph["nodes"]:
        call = node["call"]
        if call["kind"] == "hand_over" or call.get("call_id") in {
            "gen_sim.clear_released",
            "gen_sim.articulation_withdraw",
        }:
            selected.add(node.get("task_instance_id", node["id"]))
    return [
        node["id"]
        for node in graph["nodes"]
        if node.get("task_instance_id", node["id"]) in selected
    ]


def _invocation_ids(program: Any) -> dict[str, str]:
    result = {}
    for segment in program.iter_segments():
        if len(segment.calls) != 1 or segment.name in result:
            raise ValueError(
                "Generated invocation policies require one call per uniquely named segment."
            )
        call = segment.calls[0]
        result[segment.name] = (
            f"{program.program_id}/{segment.segment_id}:{call.segment_call_index}"
        )
    return result


def bind_cartesian_calls(declarations: object, program: Any) -> tuple[str, ...]:
    """Resolve selected segment names without inspecting other calls at runtime."""
    if type(declarations) is not list or any(
        type(value) is not str for value in declarations
    ):
        raise ValueError("GenSim cartesian_calls must be a list of segment names.")
    if len(set(declarations)) != len(declarations):
        raise ValueError("GenSim cartesian_calls must not contain duplicate names.")
    ids = _invocation_ids(program)
    if set(declarations) - ids.keys():
        raise ValueError("Cartesian policies reference missing generated calls.")
    return tuple(ids[name] for name in declarations)


def pour_receiver_declarations(graph: dict[str, Any]) -> dict[str, str]:
    """Keep a recipe's explicit staging reference local to its Pour occurrence."""
    staged = {}
    result = {}
    for node in graph["nodes"]:
        call = node["call"]
        arguments = call.get("arguments", {})
        key = (node.get("task_instance_id", node["id"]), arguments.get("object"))
        if (
            call.get("call_id") == "simulation.move_held_object"
            and "reference" in arguments
        ):
            staged[key] = arguments["reference"]
        elif call.get("call_id") == "simulation.pour":
            if key not in staged:
                raise ValueError(
                    "Every generated Pour needs its own preceding staging reference."
                )
            result[node["id"]] = staged[key]
    return result


def bind_pour_receivers(
    declarations: object, program: Any
) -> tuple[tuple[str, str], ...]:
    """Bind receivers to Pour invocation IDs, never to the held object's UID."""
    if type(declarations) is not dict or any(
        type(name) is not str
        or type(value) is not str
        or not value
        or value != value.strip()
        for name, value in declarations.items()
    ):
        raise ValueError(
            "GenSim pour_receivers must map Pour segment names to receiver IDs."
        )
    ids = _invocation_ids(program)
    pours = {
        segment.name
        for segment in program.iter_segments()
        if getattr(segment.calls[0].call, "call_id", None) == "simulation.pour"
    }
    if pours != declarations.keys():
        raise ValueError("Every generated Pour needs an exact receiver binding.")
    return tuple((ids[name], receiver) for name, receiver in declarations.items())


def bind_motion_samples(
    declarations: object, program: Any
) -> tuple[tuple[str, int], ...]:
    """Resolve exact generated segments to canonical runtime invocation IDs."""
    if type(declarations) is not dict or any(
        type(name) is not str or type(value) is not int or value < 2
        for name, value in declarations.items()
    ):
        raise ValueError(
            "GenSim motion_samples must map segment names to positive motion budgets."
        )
    ids = _invocation_ids(program)
    if ids.keys() != declarations.keys():
        raise ValueError(
            "Every generated call needs an exact motion budget; regenerate the bundle."
        )
    return tuple(
        (invocation_id, declarations[name]) for name, invocation_id in ids.items()
    )


class GenSimActionEngine(AtomicActionEngine):
    """Select call-local policies while retaining the shared planner and executor."""

    def __init__(
        self,
        *args: Any,
        motion_samples: tuple[tuple[str, int], ...] = (),
        cartesian_calls: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        self._motion_samples = dict(motion_samples)
        if any(type(value) is not str or not value for value in cartesian_calls):
            raise ValueError("Cartesian planning scopes require invocation IDs.")
        self._cartesian_calls = frozenset(cartesian_calls)
        if len(self._motion_samples) != len(motion_samples) or any(
            type(key) is not str or not key or type(value) is not int or value < 2
            for key, value in motion_samples
        ):
            raise ValueError(
                "Invocation motion budgets require unique IDs and integer sample counts."
            )
        super().__init__(*args, **kwargs)

    def _resolve(self, invocation: ActionInvocation) -> ResolvedActionRequest:
        if self._motion_samples:
            try:
                minimum = self._motion_samples[invocation.invocation_id]
            except KeyError as exc:
                raise ValueError(
                    "No motion policy was bound to this GenSim invocation."
                ) from exc
            invocation = replace(
                invocation,
                motion_policy=replace(
                    invocation.motion_policy,
                    sample_count=max(minimum, invocation.motion_policy.sample_count),
                ),
            )
        return super()._resolve(invocation)

    def _plan_request(self, request: ResolvedActionRequest, context: Any = None) -> Any:
        from .motion import cartesian_approach_scope
        from .planning_probe import initial_plan_capture

        capture = initial_plan_capture(request.invocation_id)
        enabled = request.invocation_id in self._cartesian_calls
        try:
            with cartesian_approach_scope(enabled):
                plan = super()._plan_request(request, context)
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_invocation_policy": {
                            "cartesian_approach": enabled,
                            "sample_count": request.motion_policy.sample_count,
                        },
                    },
                ),
            )
            return (
                plan if capture is None else capture.accept(request, plan, self.robot)
            )
        except Exception as exc:
            if capture is not None:
                capture.failed(exc)
            raise
