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

from copy import deepcopy
from dataclasses import replace
from collections.abc import Iterable
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import AtomicActionEngine, PlanTransform
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionInvocation,
    ResolvedActionRequest,
)
from embodichain.utils import logger

__all__: list[str] = []

INVOCATION_POLICY_REVISION = 6
POUR_RETURN_POLICY_REVISION = 1


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


def pour_return_call_declarations(graph: dict[str, Any]) -> list[str]:
    """Select the first absolute Place after this recipe's own object's Pour."""
    pending: set[tuple[str, str]] = set()
    result = []
    for node in graph["nodes"]:
        if node["task_type"] != "E3":
            continue
        call = node["call"]
        if call.get("call_id") == "simulation.pour":
            pending.add((node["task_instance_id"], call["arguments"]["object"]))
        elif call["kind"] == "place" and call.get("at") is not None:
            key = (node["task_instance_id"], call["object"])
            if key in pending:
                result.append(node["id"])
                pending.remove(key)
    return result


def bind_pour_return_calls(declarations: object, program: Any) -> tuple[str, ...]:
    """Bind the declared returns to exact built-in absolute Place invocations."""
    from embodichain.lab.task_program.semantics.calls import Place

    if (
        type(declarations) is not list
        or any(type(name) is not str for name in declarations)
        or len(set(declarations)) != len(declarations)
    ):
        raise ValueError("GenSim pour_return_calls must be unique Place segment names.")
    ids = _invocation_ids(program)
    calls = {segment.name: segment.calls[0].call for segment in program.iter_segments()}
    if any(
        name not in calls or type(calls[name]) is not Place or calls[name].at is None
        for name in declarations
    ):
        raise ValueError(
            "Pour return policies require existing built-in absolute Place calls."
        )
    return tuple(ids[name] for name in declarations)


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
        pour_return_calls: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> None:
        self._motion_samples = dict(motion_samples)
        self._twist_session_controller = None
        self._twist_session_controllers: dict[str, Any] = {}
        if any(type(value) is not str or not value for value in cartesian_calls):
            raise ValueError("Cartesian planning scopes require invocation IDs.")
        self._cartesian_calls = frozenset(cartesian_calls)
        self._articulation_calls = dict(articulation_calls)
        if (
            type(pour_return_calls) is not tuple
            or any(
                type(identity) is not str or not identity
                for identity in pour_return_calls
            )
            or len(set(pour_return_calls)) != len(pour_return_calls)
        ):
            raise ValueError("Pour return scopes require unique invocation IDs.")
        self._pour_return_calls = frozenset(pour_return_calls)
        self._clearance_requests: dict[str, Any] = {}
        self._clearance_parks: frozenset[str] = frozenset()
        self._park_recovery_previews: dict[str, Any] = {}
        self._clearance_simulation: Any = None
        self._alignment_continuations: dict[str, Any] = {}
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
        from .clearance_recovery import _ParkPreview

        self._park_recovery_previews = {
            preview.invocation_id: preview
            for preview in parks
            if isinstance(preview, _ParkPreview)
            and preview.compiler.integration.engine is self
        }

    def bind_alignment_continuation(self, previews: dict[str, Any]) -> None:
        """Install only the factory's canonical, exact adjacent Place previews."""
        if self._alignment_continuations or any(
            identity != preview.route.alignment_id
            or preview.compiler.integration.engine is not self
            for identity, preview in previews.items()
        ):
            raise ValueError(
                "Alignment continuation previews are already bound or foreign."
            )
        self._alignment_continuations = dict(previews)

    def configure_twist_feedback(self, controller: Any) -> None:
        """Bind one E8 policy to this engine's public-session extension."""
        from .twist_feedback import TwistFeedbackState
        from .twist_runtime import GenSimTwist

        operation = getattr(getattr(controller, "route", None), "operation_id", "")
        action = self.actions.get("twist")
        if (
            not isinstance(controller, TwistFeedbackState)
            or not controller.enabled
            or controller.robot is not self.robot
            or not isinstance(action, GenSimTwist)
            or (
                getattr(action, "_operation_feedback", {}).get(operation)
                is not controller
                and action._feedback_state is not controller
            )
            or operation in self._twist_session_controllers
        ):
            raise ValueError("E8 session feedback must own the exact action and robot.")
        self._twist_session_controllers[operation] = controller
        if not operation:
            self._twist_session_controller = controller

    def start(
        self,
        invocations: Iterable[ActionInvocation],
        context: Any = None,
        *,
        eligible_mask: Any = None,
        plan_transform: PlanTransform | None = None,
    ) -> Any:
        """Retain the public runner while extending effectless E8 sessions only."""
        selected = tuple(invocations)
        controller = self._twist_session_controller
        if (
            self._twist_session_controllers
            and len(selected) == 1
            and selected[0].skill_id == "twist"
        ):
            operation = getattr(
                getattr(selected[0].goal, "semantics", None), "geometry", {}
            ).get("gen_sim_twist_operation_id", "")
            controller = self._twist_session_controllers.get(operation, controller)
        if (
            controller is None
            or not any(item.skill_id == "twist" for item in selected)
            or (eligible_mask is not None and not bool(eligible_mask.any()))
        ):
            return super().start(
                selected,
                context,
                eligible_mask=eligible_mask,
                plan_transform=plan_transform,
            )
        from .twist_session import TwistFeedbackSession

        initial = self.initial_context() if context is None else context
        return TwistFeedbackSession(
            self,
            selected,
            initial,
            revision_controller=controller,
            eligible_mask=eligible_mask,
            plan_transform=plan_transform,
        )

    def _resolve(self, invocation: ActionInvocation) -> ResolvedActionRequest:
        if invocation.invocation_id in self._pour_return_calls:
            from embodichain.lab.sim.atomic_actions.primitives.place import PlaceOptions

            if (
                invocation.skill_id != "place"
                or type(invocation.skill_options) is not PlaceOptions
            ):
                raise ValueError(
                    "Pour return requires the exact Place options contract."
                )
            invocation = replace(
                invocation,
                skill_options=replace(invocation.skill_options, lift_height=0.12),
            )
        if getattr(
            invocation, "skill_id", None
        ) == "twist" and "gen_sim_twist_operation_id" in getattr(
            getattr(invocation.goal, "semantics", None), "geometry", {}
        ):
            invocation = replace(
                invocation,
                motion_policy=replace(
                    invocation.motion_policy,
                    strategy="ik_interp",
                    sample_count=max(320, invocation.motion_policy.sample_count or 0),
                ),
                recovery_policy=replace(
                    invocation.recovery_policy, max_replans=0, max_action_retries=0
                ),
            )
        if getattr(
            invocation, "skill_id", None
        ) == "press" and "gen_sim_press_operation_id" in getattr(
            getattr(invocation.goal, "semantics", None), "geometry", {}
        ):
            invocation = replace(
                invocation,
                motion_policy=replace(
                    invocation.motion_policy,
                    strategy="ik_interp",
                    sample_count=max(260, invocation.motion_policy.sample_count or 0),
                ),
                recovery_policy=replace(
                    invocation.recovery_policy, max_replans=0, max_action_retries=0
                ),
            )
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
        from ._search_cost import measure_search, search_cost_scope

        with search_cost_scope() as cost:
            with measure_search("complete_plan_request"):
                plan = self._plan_request_impl(
                    request, context, plan_transform=plan_transform
                )
            return replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_search_cost": cost.snapshot(),
                    },
                ),
            )

    def _plan_request_impl(
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
        from .alignment_continuation import _PREVIEW

        if _PREVIEW.get():
            capture = None
        enabled = request.invocation_id in self._cartesian_calls
        alignment = (
            request.skill_id == "move_held_object"
            and getattr(request.goal, "world_yaw_free", False) is True
            and not getattr(self.motion_generator, "direct_transport_planning", False)
        )
        pick_transform = False
        closing_transform = False
        held_transform = False
        if plan_transform is not None:
            from .actions import GenSimMoveHeldObject, GenSimPickUp
            from .pick_safety import _owns_constrained_pick

            transformed_action = getattr(self, "_actions", {}).get(request.skill_id)
            closing_transform = (
                isinstance(transformed_action, GenSimPickUp)
                and transformed_action.closing_geometry_factory is not None
            )
            pick_transform = isinstance(
                transformed_action, GenSimPickUp
            ) and _owns_constrained_pick(request, context, self.motion_generator)
            held_transform = (
                isinstance(transformed_action, GenSimMoveHeldObject)
                and transformed_action.held_path_scene_vertices is not None
                and not getattr(
                    self.motion_generator, "direct_transport_planning", False
                )
            )
        if context is None and (
            alignment
            or pick_transform
            or closing_transform
            or held_transform
            or request.invocation_id in getattr(self, "_clearance_requests", {})
            or request.invocation_id in getattr(self, "_clearance_parks", ())
        ):
            context = self.initial_context()
        staging_transform_owned = False
        if alignment and plan_transform is not None:
            from .staging_recovery import owns_staging

            action = getattr(self, "_actions", {}).get(request.skill_id)
            staging_contract = getattr(action, "staging_contract", None)
            staging_transform_owned = staging_contract is not None and (
                owns_staging(
                    request,
                    context,
                    staging_contract.rigid_entity_ids,
                    staging_contract.invocation_ids,
                    self.motion_generator,
                )
            )
            if staging_transform_owned:
                transform = plan_transform

                def preserve_staging_scope(selected, current, original):
                    report = deepcopy(
                        dict(
                            original.diagnostics.metadata.get(
                                "gen_sim_staging_recovery", {}
                            )
                        )
                    )
                    transformed = transform(selected, current, original)
                    if (
                        report.get("accepted") is True
                        and getattr(transformed, "diagnostics", None) is not None
                    ):
                        transformed = replace(
                            transformed,
                            diagnostics=replace(
                                transformed.diagnostics,
                                metadata={
                                    **transformed.diagnostics.metadata,
                                    "gen_sim_staging_recovery": dict(report),
                                },
                            ),
                        )
                    return transformed

                plan_transform = preserve_staging_scope
        continuation = getattr(self, "_alignment_continuations", {}).get(
            request.invocation_id
        )
        continuation_owned, before_continuation, continuation_request = (
            False,
            None,
            None,
        )
        if continuation is not None:
            from .alignment_continuation import eligible
            from .held_alignment_recovery import _RandomState, _cuda_devices

            continuation_owned = eligible(self, request, context, continuation)
            if continuation_owned:
                continuation_request = request.snapshot()
                before_continuation = _RandomState.capture(
                    _cuda_devices(self.robot.device, context.robot.qpos.device)
                )
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
                staging_report = plan.diagnostics.metadata.get(
                    "gen_sim_staging_recovery", {}
                )
                if (
                    staging_transform_owned
                    and staging_report.get("accepted") is True
                    and plan.plan_success.any()
                ):
                    from .staging_recovery import inspect_staging_plan
                    from embodichain.lab.sim.atomic_actions.plans import PlanningFailure

                    certificate = inspect_staging_plan(
                        self.robot,
                        selected_request,
                        context,
                        plan,
                        retention_source=action.staging_contract.retention_source(
                            selected_request.invocation_id
                        ),
                        _require_retained_command=staging_report.get(
                            "retained_hand_boundary", {}
                        ).get("accepted")
                        is True,
                    )
                    plan = replace(
                        plan,
                        plan_success=(
                            plan.plan_success
                            if certificate["valid"]
                            else torch.zeros_like(plan.plan_success)
                        ),
                        diagnostics=replace(
                            plan.diagnostics,
                            metadata={
                                **plan.diagnostics.metadata,
                                "gen_sim_staging_recovery": {
                                    **staging_report,
                                    "post_transform_check": certificate,
                                },
                            },
                            **(
                                {
                                    "failure": PlanningFailure(
                                        "held_staging_safety_failed", retryable=False
                                    )
                                }
                                if not certificate["valid"]
                                else {}
                            ),
                        ),
                    )
                park_request = getattr(self, "_clearance_requests", {}).get(
                    request.invocation_id
                )
                if park_request is not None:
                    from .clearance_recovery import recover

                    def build_clearance(selected: Any, current: Any) -> Any:
                        if selected.invocation_id in getattr(
                            self, "_park_recovery_previews", {}
                        ):
                            return self._plan_request(selected, current)
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
                    from .park_recovery import recover

                    original_park = plan
                    plan = recover(
                        self,
                        selected_request,
                        context,
                        plan,
                        acceptance=acceptance,
                        build=lambda selected, current: self._plan_request(
                            selected, current, plan_transform=plan_transform
                        ),
                        simulation=self._clearance_simulation,
                    )
                    if plan is not original_park and plan.plan_success.all():
                        acceptance = plan.diagnostics.metadata["gen_sim_clearance_park"]
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
                                    "failure": (
                                        plan.diagnostics.failure
                                        if "gen_sim_park_recovery"
                                        in plan.diagnostics.metadata
                                        else PlanningFailure(
                                            "cleanup_park_failed", retryable=False
                                        )
                                    )
                                }
                                if not acceptance["valid"]
                                else {}
                            ),
                        ),
                    )
                if (
                    pick_transform
                    and _owns_constrained_pick(
                        selected_request, context, self.motion_generator
                    )
                    and plan.plan_success.any()
                ):
                    from .articulation_collision import robot_collision_scope
                    from .pick_safety import _inspect_pick_plan
                    from embodichain.lab.sim.atomic_actions.plans import PlanningFailure

                    with robot_collision_scope():
                        acceptance = _inspect_pick_plan(
                            self.robot, selected_request, context, plan
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
                                "gen_sim_pick_final_command_check": acceptance,
                            },
                            failure=(
                                plan.diagnostics.failure
                                if acceptance["valid"]
                                else PlanningFailure(
                                    "pick_robot_safety_failed", retryable=False
                                )
                            ),
                        ),
                    )
                if closing_transform and plan.plan_success.any():
                    from embodichain.lab.sim.atomic_actions.plans import PlanningFailure
                    from .articulation_collision import robot_collision_scope
                    from .pick_safety import _inspect_pick_plan

                    try:
                        with robot_collision_scope():
                            commands = _inspect_pick_plan(
                                self.robot, selected_request, context, plan
                            )
                        acceptance = (
                            transformed_action.inspect_closing_plan(
                                selected_request, context, plan
                            )
                            if commands["valid"]
                            else {"valid": False, "command_validation": commands}
                        )
                    except (
                        ValueError,
                        RuntimeError,
                        AttributeError,
                        KeyError,
                        TypeError,
                    ) as error:
                        acceptance = {"valid": False, "unavailable": str(error)}
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
                                "gen_sim_pick_final_contact_check": acceptance,
                            },
                            failure=(
                                plan.diagnostics.failure
                                if acceptance["valid"]
                                else PlanningFailure(
                                    "pick_contact_geometry_failed", retryable=False
                                )
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
            if (
                continuation_owned
                and selected_request.motion_policy.sample_count == 260
            ):
                from .alignment_continuation import recover

                plan = recover(
                    self,
                    continuation_request,
                    context,
                    plan,
                    preview=continuation,
                    build=lambda selected: self._plan_request(
                        selected, context, plan_transform=plan_transform
                    ),
                    before_original=before_continuation,
                    planned_request=selected_request,
                )
            if held_transform and plan.plan_success.any():
                from .held_path import inspect, phase_scope
                from embodichain.lab.sim.atomic_actions.plans import PlanningFailure

                # Transform consumers must qualify the actual emitted samples,
                # not reuse the pre-transform carried-mesh certificate.
                scope = phase_scope(transformed_action, selected_request, context)
                acceptance = (
                    inspect(transformed_action, selected_request, context, plan)
                    if scope["owned"]
                    else None
                )
                rejected = acceptance is not None and not acceptance["valid"]
                plan = replace(
                    plan,
                    plan_success=(
                        torch.zeros_like(plan.plan_success)
                        if rejected
                        else plan.plan_success
                    ),
                    diagnostics=replace(
                        plan.diagnostics,
                        metadata={
                            **plan.diagnostics.metadata,
                            "gen_sim_held_path": {
                                **plan.diagnostics.metadata.get(
                                    "gen_sim_held_path", {}
                                ),
                                "phase": scope,
                                "post_transform_check": acceptance,
                            },
                        },
                        failure=(
                            PlanningFailure(
                                "held_transport_path_unqualified", retryable=False
                            )
                            if rejected
                            else plan.diagnostics.failure
                        ),
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
