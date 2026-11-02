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
"""Explicit empty-arm clearance goals and exact invocation-owned qualification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    JointPositionGoal,
    JointPositionTarget,
    MoveJoints,
    MoveJointsOptions,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations.configured import (
    _mapping,
    _sequence,
    _identifier,
    _real,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall

from .articulation_collision import check_free_motion
from .motion import _joint_velocity_limits, _velocity_validity

__all__: list[str] = []

CALL_ID = "gen_sim.clear_idle_arm"


@dataclass(frozen=True, slots=True)
class IdleRoute:
    route_id: str
    clear_segment: str
    restore_segment: str
    resource: str
    joint_name: str
    offset_rad: float
    search: bool = False
    protected_control_part: str | None = None


def decode_routes(value: object, *, path: str) -> tuple[IdleRoute, ...]:
    routes = []
    for index, raw in enumerate(_sequence(value, path=path)):
        where = f"{path}[{index}]"
        required = frozenset(
            {
                "route_id",
                "clear_segment",
                "restore_segment",
                "resource",
                "joint_name",
                "offset_rad",
            }
        )
        item = _mapping(
            raw,
            path=where,
            required=required,
            optional=frozenset({"search", "protected_control_part"}),
        )
        fields = {
            name: _identifier(item[name], path=f"{where}.{name}")
            for name in required - {"offset_rad"}
        }
        offset = _real(item["offset_rad"], path=f"{where}.offset_rad")
        if offset == 0:
            raise ValueError("Idle clearance requires a nonzero declared offset.")
        search = item.get("search", False)
        protected = item.get("protected_control_part")
        if type(search) is not bool or (
            search and (type(protected) is not str or not protected)
        ):
            raise ValueError("Idle search requires a declared protected motion part.")
        routes.append(
            IdleRoute(
                **fields,
                offset_rad=offset,
                search=search,
                protected_control_part=protected,
            )
        )
    names = [
        name
        for route in routes
        for name in (route.clear_segment, route.restore_segment)
    ]
    if (
        not routes
        or len({r.route_id for r in routes}) != len(routes)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Idle clearance requires nonempty unique routes and segments.")
    return tuple(routes)


def require_empty(context: Any, resource: str) -> None:
    if context.batch_size != 1 or context.env_ids.tolist() != [0]:
        raise ValueError("Idle clearance requires B1/env0.")
    held = context.task.get_held_object(resource)
    if held is not None and held.active_mask.any():
        raise ValueError("Idle clearance cannot move a held object.")
    if any(
        resource in pair and state.active_mask.any()
        for pair, state in context.task.coordinated_held_objects.items()
    ):
        raise ValueError("Idle clearance cannot move a coordinated-held resource.")


class _IdleLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = CALL_ID
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(
        self, routes: tuple[IdleRoute, ...], robot: Any, simulation: Any = None
    ) -> None:
        self._routes, self._robot, self._simulation = (
            {r.route_id: r for r in routes},
            robot,
            simulation,
        )

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        args = dict(call.arguments)
        if set(args) != {"route"} or args["route"] not in self._routes:
            raise ValueError("Idle clearance has no exact declared route.")
        route = self._routes[args["route"]]
        if (
            dict(call.resources) != {"primary": route.resource}
            or type(option_template) is not MoveJointsOptions
        ):
            raise ValueError(
                "Idle clearance requires its declared MoveJoints resource."
            )
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        require_empty(context, endpoint.task_state_key)
        motion = endpoint.runtime_target
        names = [self._robot.joint_names[i] for i in motion.joint_ids]
        if route.joint_name not in names:
            raise ValueError("Idle joint is outside the bound motion endpoint.")
        qpos = context.robot.qpos[:, list(motion.joint_ids)].clone()
        offset = route.offset_rad
        if route.search:
            offset = select_offset(
                self._robot, self._simulation, context, motion, route
            )
        qpos[:, names.index(route.joint_name)] += offset
        return SemanticLowering(goal=JointPositionGoal(qpos))


@dataclass(frozen=True, slots=True)
class IdleFactory:
    call_id: ClassVar[str] = CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor = MoveJoints.descriptor()
    routes: tuple[IdleRoute, ...]

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> _IdleLowerer:
        if engine.robot is not robot:
            raise ValueError("Idle clearance must use the engine-owned robot.")
        return _IdleLowerer(self.routes, robot, simulation)


def select_offset(
    robot: Any, simulation: Any, context: Any, motion: Any, route: IdleRoute
) -> float:
    """Rank bounded joint proposals by FK separation, then screen their swept geometry."""
    import yourdfpy

    model = yourdfpy.URDF.load(robot.cfg.fpath, load_meshes=False)
    start = context.robot.qpos[0].detach().cpu()
    joint = robot.joint_names.index(route.joint_name)
    active = robot.get_solver(motion.control_part).end_link_name
    protected = robot.get_solver(route.protected_control_part).end_link_name
    limit = model.joint_map[route.joint_name].limit
    proposals = []
    for factor in (-1.0, -2 / 3, -1 / 3, 1 / 3, 2 / 3, 1.0):
        offset = abs(route.offset_rad) * factor
        goal = start.clone()
        goal[joint] += offset
        if not limit.lower <= float(goal[joint]) <= limit.upper:
            continue
        model.update_cfg(
            {
                name: float(value)
                for name, value in zip(robot.joint_names, goal)
                if name in model.joint_map
            }
        )
        a, b = model.get_transform(active)[:3, 3], model.get_transform(protected)[:3, 3]
        separation = float(((a - b) ** 2).sum())
        proposals.append((separation, offset, goal))
    for _, offset, goal in sorted(proposals, key=lambda item: item[0], reverse=True):
        sweep = start[None] + torch.linspace(0, 1, 101)[:, None] * (goal - start)[None]
        if check_free_motion(robot, simulation, sweep, motion.control_part)["valid"]:
            return offset
    raise ValueError("No bounded empty-arm clearance proposal passed swept geometry.")


def bind_invocations(
    program: Any, factories: tuple[IdleFactory, ...]
) -> tuple[tuple[str, str], ...]:
    if not factories:
        return ()
    if len(factories) != 1:
        raise ValueError("Idle clearance requires one factory.")
    segments = tuple(program.iter_segments())
    by_name = {s.name: (i, s) for i, s in enumerate(segments)}
    result, used = [], set()
    for route in factories[0].routes:
        if route.clear_segment not in by_name or route.restore_segment not in by_name:
            raise ValueError("Idle clearance references missing program segments.")
        start, clear = by_name[route.clear_segment]
        end, restore = by_name[route.restore_segment]
        if start >= end or any(
            len(s.calls) != 1 or s.parallel_block is not None for s in (clear, restore)
        ):
            raise ValueError("Idle clearance requires an ordered sequential pair.")
        a, b = clear.calls[0].call, restore.calls[0].call
        if (
            type(a) is not RegisteredSemanticCall
            or a.call_id != CALL_ID
            or dict(a.arguments) != {"route": route.route_id}
            or type(b) is not RegisteredSemanticCall
            or b.call_id != "simulation.park"
            or dict(b.arguments)
            or any(dict(c.resources) != {"primary": route.resource} for c in (a, b))
        ):
            raise ValueError(
                "Idle clearance/restore must match their declared resource and calls."
            )
        for s in (clear, restore):
            identity = (
                f"{program.program_id}/{s.segment_id}:{s.calls[0].segment_call_index}"
            )
            if identity in used:
                raise ValueError("Idle clearance invocation is duplicated.")
            used.add(identity)
            result.append((identity, route.resource))
    actual = {
        f"{program.program_id}/{s.segment_id}:{s.calls[0].segment_call_index}"
        for s in segments
        if len(s.calls) == 1 and getattr(s.calls[0].call, "call_id", None) == CALL_ID
    }
    if not actual <= used:
        raise ValueError("An idle clearance call lacks its exact declaration.")
    return tuple(result)


def qualify_motion(
    robot: Any, simulation: Any, request: Any, context: Any, plan: Any
) -> dict[str, Any]:
    endpoint = request.binding.endpoint("primary", "motion")
    require_empty(context, endpoint.task_state_key)
    arm = endpoint.require_target(JointPositionTarget)
    positions, dt = plan.joint_trajectory.positions, plan.joint_trajectory.dt
    if (
        positions.ndim != 3
        or positions.shape[0] != 1
        or not positions.shape[1]
        or not torch.equal(positions[:, 0], context.robot.qpos)
    ):
        raise ValueError("Idle motion must retain its exact full-robot start.")
    passive = [i for i in range(positions.shape[-1]) if i not in arm.joint_ids]
    if not torch.equal(
        positions[:, :, passive],
        context.robot.qpos[:, None, passive].expand(-1, positions.shape[1], -1),
    ):
        raise ValueError("Idle clearance changed an uncommanded joint.")
    bounds = robot.get_qpos_limits(name=arm.control_part).to(positions)
    active = positions[:, :, list(arm.joint_ids)]
    if not (
        (active >= bounds[:, None, :, 0]) & (active <= bounds[:, None, :, 1])
    ).all():
        raise ValueError("Idle clearance violates original joint limits.")
    if not _velocity_validity(
        positions, dt, _joint_velocity_limits(robot, None).to(positions)
    ).all():
        raise ValueError("Idle clearance violates original velocity limits.")
    return check_free_motion(robot, simulation, positions[0], arm.control_part)


class IdleGuardMoveJoints(MoveJoints):
    skill_id = MoveJoints.skill_id
    GoalType = MoveJoints.GoalType
    OptionsType = MoveJoints.OptionsType
    binding_contract = MoveJoints.binding_contract
    idle_invocations: Mapping[str, str] = {}
    idle_simulation: Any = None

    def inspect_final_plan(self, request: Any, context: Any, plan: Any) -> Any:
        """Recheck the actual emitted trajectory after an engine plan transform."""
        if (
            request.invocation_id not in self.idle_invocations
            or not plan.plan_success.all()
        ):
            return plan
        try:
            checked = qualify_motion(
                self.robot, self.idle_simulation, request, context, plan
            )
            if not checked["valid"]:
                raise ValueError(f"Transformed idle motion rejected: {checked}")
            return plan
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            return self.failed_plan(
                request,
                context,
                message=str(error),
                failure_code="gen_sim_idle_final_plan_unqualified",
                retryable=False,
            )

    def _plan(self, request: Any, context: Any) -> Any:
        resource = self.idle_invocations.get(request.invocation_id)
        if resource is None:
            return super()._plan(request, context)
        try:
            if request.binding.endpoint("primary", "motion").task_state_key != resource:
                raise ValueError("Idle invocation resource identity differs.")
            require_empty(context, resource)
            plan = super()._plan(request, context)
            if not plan.plan_success.all():
                return plan
            checked = qualify_motion(
                self.robot, self.idle_simulation, request, context, plan
            )
            if not checked["valid"]:
                raise ValueError(f"Idle robot/scene motion rejected: {checked}")
            return replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_idle_clearance": {
                            "resource": resource,
                            "qualification": checked,
                        },
                    },
                ),
            )
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            return self.failed_plan(
                request,
                context,
                message=str(error),
                failure_code="gen_sim_idle_clearance_unqualified",
                retryable=False,
            )
