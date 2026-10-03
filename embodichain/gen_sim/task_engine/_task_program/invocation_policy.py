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

import torch

from embodichain.lab.sim.atomic_actions import AtomicActionEngine, PlanTransform
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionInvocation,
    ResolvedActionRequest,
)
from embodichain.utils import logger

__all__: list[str] = []

INVOCATION_POLICY_REVISION = 5


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


def bind_articulation_calls(program: Any) -> tuple[tuple[str, str], ...]:
    """Grant recovery only to the exact registered E6 call occurrences."""
    ids = _invocation_ids(program)
    kinds = {
        "gen_sim.articulation_slide": "slide",
        "gen_sim.articulation_withdraw": "withdraw",
    }
    return tuple(
        (ids[segment.name], kinds[segment.calls[0].call.call_id])
        for segment in program.iter_segments()
        if getattr(segment.calls[0].call, "call_id", None) in kinds
    )


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
        articulation_calls: tuple[tuple[str, str], ...] = (),
        **kwargs: Any,
    ) -> None:
        self._motion_samples = dict(motion_samples)
        if any(type(value) is not str or not value for value in cartesian_calls):
            raise ValueError("Cartesian planning scopes require invocation IDs.")
        self._cartesian_calls = frozenset(cartesian_calls)
        self._articulation_calls = dict(articulation_calls)
        self._clearance_requests: dict[str, Any] = {}
        self._clearance_parks: frozenset[str] = frozenset()
        self._clearance_simulation: Any = None
        if len(self._articulation_calls) != len(articulation_calls) or any(
            not isinstance(key, str) or not key or value not in {"slide", "withdraw"}
            for key, value in articulation_calls
        ):
            raise ValueError(
                "E6 recovery requires unique invocation IDs and known call kinds."
            )
        if len(self._motion_samples) != len(motion_samples) or any(
            type(key) is not str or not key or type(value) is not int or value < 2
            for key, value in motion_samples
        ):
            raise ValueError(
                "Invocation motion budgets require unique IDs and integer sample counts."
            )
        super().__init__(*args, **kwargs)

    def bind_release_clearance(
        self, park_requests: dict[str, tuple[Any, ...]], *, simulation: Any
    ) -> None:
        """Install exact generated cleanup/park scopes after compiler analysis."""
        if self._clearance_requests:
            raise ValueError("Released-hand planning scopes are already bound.")
        if any(
            type(chain) is not tuple or not 1 <= len(chain) <= 2
            for chain in park_requests.values()
        ):
            raise ValueError("Released-hand recovery requires bounded park tuples.")
        parks = [request for chain in park_requests.values() for request in chain]
        park_ids = [request.invocation_id for request in parks]
        if (
            len(set(park_ids)) != len(park_ids)
            or any(type(key) is not str or not key for key in park_requests)
            or any(type(key) is not str or not key for key in park_ids)
            or any(request.skill_id != "move_joints" for request in parks)
        ):
            raise ValueError(
                "Released-hand recovery requires unique configured park requests."
            )
        self._clearance_requests = dict(park_requests)
        self._clearance_parks = frozenset(park_ids)
        self._clearance_simulation = simulation

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

    def _plan_request(
        self,
        request: ResolvedActionRequest,
        context: Any = None,
        *,
        plan_transform: PlanTransform | None = None,
    ) -> Any:
        from .motion import cartesian_approach_scope
        from .motion import _velocity_retry_samples
        from .planning_probe import initial_plan_capture
        from .planning_probe import _validate_final_plan_velocity

        capture = initial_plan_capture(request.invocation_id)
        enabled = request.invocation_id in self._cartesian_calls
        alignment = (
            request.skill_id == "move_held_object"
            and getattr(request.goal, "world_yaw_free", False) is True
            and not getattr(self.motion_generator, "direct_transport_planning", False)
        )
        if context is None and (
            alignment
            or request.invocation_id in getattr(self, "_clearance_requests", {})
            or request.invocation_id in getattr(self, "_clearance_parks", ())
        ):
            context = self.initial_context()
        selected_request = request
        retime_attempts: list[int] = []
        try:
            with cartesian_approach_scope(enabled):
                kind = self._articulation_calls.get(request.invocation_id)
                if (
                    kind is not None
                    and enabled
                    and context is not None
                    and context.batch_size == 1
                    and request.motion_policy.strategy == "ik_interp"
                    and request.motion_policy.dynamic_collision_mode == "off"
                ):
                    from .articulation_recovery import planning_scope, recover

                    def build(selected: Any) -> Any:
                        return super(GenSimActionEngine, self)._plan_request(
                            selected, context, plan_transform=plan_transform
                        )

                    with planning_scope(self.motion_generator, kind) as scope:
                        plan = build(request)
                        plan = recover(scope, request, context, plan, build)
                else:
                    plan = super()._plan_request(
                        request, context, plan_transform=plan_transform
                    )
                # Some GenSim primitives (notably coordinated E5) assemble
                # their complete trajectory outside the Cartesian planner's
                # adaptive retry path.  Retry only a velocity-invalid plan
                # with a denser time grid; never relax the robot's limits.
                trajectory = getattr(plan, "joint_trajectory", None)
                if (
                    request.invocation_id in getattr(self, "_motion_samples", {})
                    and trajectory is not None
                    and request.motion_policy.sample_count is not None
                    and getattr(plan, "plan_success", None) is not None
                    and plan.plan_success.any()
                ):
                    velocity = _validate_final_plan_velocity(plan, self.robot)
                    invalid = plan.plan_success & ~velocity["valid_mask"]
                    if invalid.any():
                        # A failed fallback must not perturb the next call's
                        # candidate stream.  The first plan remains the source
                        # of truth if every denser grid is rejected.
                        devices = []
                        device = trajectory.positions.device
                        if device.type == "cuda" and device.index is not None:
                            devices = [device.index]
                        with torch.random.fork_rng(devices=devices):
                            for sample_count in _velocity_retry_samples(
                                request.motion_policy.sample_count
                            ):
                                retime_attempts.append(sample_count)
                                retry_request = replace(
                                    request,
                                    motion_policy=replace(
                                        request.motion_policy,
                                        sample_count=sample_count,
                                    ),
                                )
                                retry_plan = super()._plan_request(
                                    retry_request,
                                    context,
                                    plan_transform=plan_transform,
                                )
                                retry_trajectory = getattr(
                                    retry_plan, "joint_trajectory", None
                                )
                                if retry_trajectory is None:
                                    continue
                                retry_velocity = _validate_final_plan_velocity(
                                    retry_plan, self.robot
                                )
                                retry_success = retry_plan.plan_success
                                retry_invalid = (
                                    retry_success & ~retry_velocity["valid_mask"]
                                )
                                retained_success = retry_success | ~plan.plan_success
                                if (
                                    retry_success.any()
                                    and not retry_invalid.any()
                                    and retained_success.all()
                                ):
                                    selected_request = retry_request
                                    plan = retry_plan
                                    logger.log_info(
                                        "GenSim adaptive velocity retime accepted "
                                        f"sample_count={sample_count} for "
                                        f"{request.invocation_id}."
                                    )
                                    break
                if alignment:
                    from .held_alignment_recovery import qualify_alignment_plan

                    plan = qualify_alignment_plan(request, context, plan, self.robot)
                park_request = getattr(self, "_clearance_requests", {}).get(
                    request.invocation_id
                )
                if park_request is not None:
                    from .clearance_recovery import recover

                    def build_clearance(selected: Any, current: Any) -> Any:
                        return super(GenSimActionEngine, self)._plan_request(
                            selected,
                            current,
                            plan_transform=(
                                plan_transform
                                if selected.invocation_id == request.invocation_id
                                else None
                            ),
                        )

                    plan = recover(
                        request,
                        context,
                        plan,
                        build=build_clearance,
                        park_request=park_request,
                        robot=self.robot,
                        simulation=self._clearance_simulation,
                    )
                elif request.invocation_id in getattr(self, "_clearance_parks", ()):
                    from .clearance_recovery import validate_motion
                    from embodichain.lab.sim.atomic_actions import JointPositionTarget
                    from embodichain.lab.sim.atomic_actions.plans import PlanningFailure

                    endpoint = request.binding.endpoint("primary", "motion")
                    motion = endpoint.require_target(JointPositionTarget)
                    held = context.task.get_held_object(endpoint.task_state_key)
                    acceptance = (
                        {"valid": False, "reason": "park_requires_released_hand"}
                        if held is not None and held.active_mask.any()
                        else validate_motion(
                            plan,
                            context,
                            self.robot,
                            self._clearance_simulation,
                            motion.control_part,
                        )
                    )
                    plan = replace(
                        plan,
                        plan_success=(
                            plan.plan_success
                            if acceptance["valid"]
                            else torch.zeros_like(plan.plan_success)
                        ),
                        diagnostics=replace(
                            plan.diagnostics,
                            metadata={
                                **plan.diagnostics.metadata,
                                "gen_sim_clearance_park": acceptance,
                            },
                            **(
                                {
                                    "failure": PlanningFailure(
                                        "cleanup_park_failed", retryable=False
                                    )
                                }
                                if not acceptance["valid"]
                                else {}
                            ),
                        ),
                    )
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_invocation_policy": {
                            "cartesian_approach": enabled,
                            "sample_count": selected_request.motion_policy.sample_count,
                            "velocity_retime_attempts": retime_attempts,
                        },
                    },
                ),
            )
            return (
                plan
                if capture is None
                else capture.accept(selected_request, plan, self.robot)
            )
        except Exception as exc:
            if capture is not None:
                capture.failed(exc)
            raise
