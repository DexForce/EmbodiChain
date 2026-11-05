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

"""Explicit idle-arm clear/restore declarations lowered into ordinary MoveJoints."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, ClassVar, TYPE_CHECKING
import math
from embodichain.lab.sim.atomic_actions import (
    JointPositionGoal,
    MoveJoints,
    MoveJointsOptions,
)
from embodichain.lab.sim.atomic_actions.primitives.move_joints import JointOffsetTarget
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    SemanticIntegrationManifest,
)

if TYPE_CHECKING:
    from embodichain.lab.task_program.compiler import CompiledTaskProgram
__all__ = ["IdleClearanceRoute", "IdleClearanceFactory"]
CALL_ID = "gen_sim.clear_idle_arm"


@dataclass(frozen=True, slots=True)
class IdleClearanceRoute:
    """A paired clear/restore declaration with an explicit bounded offset."""

    route_id: str
    clear_segment: str
    restore_segment: str
    resource: str
    joint_name: str
    offset_rad: float
    search: bool = False
    protected_control_part: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "route_id",
            "clear_segment",
            "restore_segment",
            "resource",
            "joint_name",
            "protected_control_part",
        ):
            value = getattr(self, name)
            if value is None and name == "protected_control_part":
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact identity.")
        if (
            type(self.offset_rad) not in (int, float)
            or not math.isfinite(self.offset_rad)
            or self.offset_rad == 0
        ):
            raise ValueError(
                "Idle clearance requires a finite nonzero declared offset."
            )
        if (
            type(self.search) is not bool
            or self.search
            and self.protected_control_part is None
        ):
            raise ValueError("Idle search requires an explicit protected control part.")


class _IdleLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = CALL_ID
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, routes: tuple[IdleClearanceRoute, ...], robot: Any) -> None:
        self._routes = {route.route_id: route for route in routes}
        self._robot = robot

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: Any,
        bound: Any,
        option_template: Any,
    ) -> SemanticLowering:
        """Bind declared joint intent; the Atomic action owns empty-hand checking."""
        args = dict(call.arguments)
        if set(args) != {"route"} or args["route"] not in self._routes:
            raise ValueError("Idle clearance has no exact declared route.")
        route = self._routes[args["route"]]
        if (
            dict(call.resources) != {"primary": route.resource}
            or type(option_template) is not MoveJointsOptions
            or not option_template.require_free_motion
        ):
            raise ValueError(
                "Idle clearance requires its declared qualified MoveJoints resource."
            )
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        target = endpoint.runtime_target
        names = [self._robot.joint_names[index] for index in target.joint_ids]
        if route.joint_name not in names:
            raise ValueError("Idle joint is outside the bound motion endpoint.")
        protected = None
        if route.search:
            solver = self._robot.get_solver(route.protected_control_part)
            if solver is None:
                raise ValueError("Idle search protected solver is unavailable.")
            protected = solver.end_link_name
        return SemanticLowering(
            goal=JointPositionGoal(
                JointOffsetTarget(route.joint_name, route.offset_rad, protected)
            )
        )


@dataclass(frozen=True, slots=True)
class IdleClearanceFactory:
    """Build explicit idle clear goals; Atomic/Motion own qualification/search."""

    routes: tuple[IdleClearanceRoute, ...]
    call_id: ClassVar[str] = CALL_ID
    revision: ClassVar[str] = "2"
    target_descriptor = MoveJoints.descriptor()

    def __post_init__(self) -> None:
        if (
            type(self.routes) is not tuple
            or not self.routes
            or any(type(route) is not IdleClearanceRoute for route in self.routes)
        ):
            raise TypeError("Idle routes require a nonempty immutable typed tuple.")
        names = [
            name
            for route in self.routes
            for name in (route.clear_segment, route.restore_segment)
        ]
        if len(set(names)) != len(names) or len(
            {r.route_id for r in self.routes}
        ) != len(self.routes):
            raise ValueError("Idle clearance requires unique routes and segments.")

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> RegisteredSemanticLowerer:
        """Create a fresh lowerer without performing geometry or motion search.

        Args:
            simulation: Borrowed runtime; no state is read.
            robot: Exact engine robot owning static joint/solver declarations.
            scene_registry: Existing scene, unused by joint-goal binding.
            engine: Ordinary Atomic engine with free-motion services.

        Returns:
            Effectless idle-joint goal lowerer; no private skill is registered.
        """
        if engine.robot is not robot:
            raise ValueError("Idle clearance must use the engine-owned robot.")
        return _IdleLowerer(self.routes, robot)


def _bind_invocations(
    program: CompiledTaskProgram, factories: tuple[IdleClearanceFactory, ...]
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


def _validate_idle_program(program, manifest, policies, factories):
    ids = _bind_invocations(program, factories)
    if not ids:
        return
    selected = {item.invocation_id: item.preset_id for item in policies}
    expected = dict(ids)
    for segment in program.iter_segments():
        for call in segment.calls:
            identity = (
                f"{program.program_id}/{segment.segment_id}:{call.segment_call_index}"
            )
            if identity not in expected:
                continue
            linked = manifest.link_call(
                call.call, path=call.source_path, preset_id=selected.get(identity)
            )
            preset = manifest.robot_profile.presets[linked.preset_id]
            options = preset.action_option_templates.get(call.call.semantic_id)
            if (
                type(options) is not MoveJointsOptions
                or not options.require_free_motion
            ):
                raise ValueError(
                    "Idle clear/restore calls require explicit free-motion qualification."
                )
