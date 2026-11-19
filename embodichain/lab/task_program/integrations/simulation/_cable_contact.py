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

"""Cable geometry adapter for generic observed PickUp/Place contact paths."""

from __future__ import annotations

from copy import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, TYPE_CHECKING
from uuid import uuid4
import torch

from embodichain.lab.sim.atomic_skills.primitives.contact import (
    ContactPhase,
    ContactPostcondition,
    ContactPickUpGoal,
    ContactPlaceGoal,
    ResolvedContactPhase,
    ContactContinuation,
)
from embodichain.lab.sim.atomic_skills.primitives.pick_up import PickUp
from embodichain.lab.sim.atomic_skills.primitives.place import Place
from embodichain.lab.sim.atomic_skills import (
    PlanningContext,
    ResolvedSkillRequest,
    SkillDescriptor,
    SkillOptions,
)
from embodichain.lab.task_program.compiler.lowering import SemanticLowering
from embodichain.lab.task_program.language.semantic_contracts import (
    SemanticCallContract,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    BoundSemanticCall,
)
from .cable import (
    _CableLowerer,
    _CableFactory,
    _CableRuntime,
    _CableTarget,
    _ground_cable_move,
    _ground_cable_joint,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.task_program.runtime import SemanticCallExecutor


class _ObservedRobot:
    def __init__(
        self, robot: Robot, context: PlanningContext, target: torch.Tensor
    ) -> None:
        self._robot = robot
        self._target = target
        self._observed = torch.zeros_like(target)
        self._observed.index_copy_(0, context.env_ids, context.robot.qpos)

    def get_qpos(self, *, target: bool = False) -> torch.Tensor:
        return self._target if target else self._observed

    def __getattr__(self, name: str) -> object:
        return getattr(self._robot, name)


@dataclass(frozen=True, slots=True)
class CableContactOperation:
    """Declared operation whose numeric plans remain owned by PickUp or Place."""

    name: str
    kind: str
    phases: tuple[ContactPhase, ...]


def decode_operations(
    value: object, targets: Mapping[str, _CableTarget]
) -> tuple[CableContactOperation, ...]:
    """Validate contact operation paths without acquiring simulation resources."""
    from ..configured import _mapping, _sequence, _identifier

    result = []
    if not isinstance(value, Mapping):
        raise TypeError("Contact operations must be a mapping.")
    for name, raw in _mapping(
        value,
        path="cable_material.operations",
        required=frozenset(),
        optional=frozenset(value),
    ).items():
        c = _mapping(
            raw,
            path=f"cable_material.operations.{name}",
            required=frozenset({"kind", "phases"}),
        )
        if c["kind"] not in {"pick_up", "place"}:
            raise ValueError("Contact operation kind must be pick_up or place.")
        phases = []
        for raw_phase in _sequence(c["phases"], path="contact operation phases"):
            p = _mapping(
                raw_phase,
                path="contact phase",
                required=frozenset({"kind", "target"}),
                optional=frozenset({"after"}),
            )
            key = _identifier(p["target"], path="contact phase target")
            kind = p["kind"]
            if key not in targets:
                raise ValueError(f"Unknown contact target {key!r}.")
            t = targets[key]
            if (
                kind not in {"pose", "grasp", "joint"}
                or (kind == "grasp" and t.kind != "material")
                or (kind == "joint" and t.kind not in {"open", "grip_adjust"})
                or (
                    kind == "pose"
                    and t.kind in {"check", "open", "hold", "grip_adjust"}
                )
            ):
                raise ValueError(
                    "Contact phase kind does not match its declared target."
                )
            tail = []
            for raw_post in _sequence(p.get("after", ()), path="contact phase after"):
                v = _mapping(
                    raw_post,
                    path="contact postcondition",
                    required=frozenset({"kind", "target"}),
                )
                ref = _identifier(v["target"], path="contact postcondition target")
                if (
                    ref not in targets
                    or targets[ref].kind != v["kind"]
                    or v["kind"] not in {"hold", "check"}
                ):
                    raise ValueError(
                        "Contact postcondition must name a compatible hold/check target."
                    )
                tail.append(ContactPostcondition(v["kind"], ref))
            phases.append(ContactPhase(kind, key, tuple(tail)))
        if c["kind"] == "place" and not any(
            p.kind == "joint" and targets[p.target].kind == "open" for p in phases
        ):
            raise ValueError("Contact placement must declare an opening phase.")
        goal_type = ContactPickUpGoal if c["kind"] == "pick_up" else ContactPlaceGoal
        goal_type("static-preflight", tuple(phases))
        result.append(
            CableContactOperation(
                _identifier(name, path="contact operation name"),
                c["kind"],
                tuple(phases),
            )
        )
    return tuple(result)


class CableContactSource:
    """Resolve geometry from one bridge, without owning the operation sequence."""

    def __init__(self, runtime: _CableRuntime) -> None:
        from weakref import proxy

        self.runtime = proxy(runtime)
        self.source_id = uuid4().hex
        self.current: ContactContinuation | None = None
        self.observed_checks: set[tuple[int, str]] = set()
        self.rejected: torch.Tensor | None = None
        self.reports: list[dict[str, object]] = []
        self.owner: str | None = None

    def _view(
        self, context: PlanningContext, active: torch.Tensor
    ) -> tuple[_CableRuntime, torch.Tensor, torch.Tensor]:
        runtime = self.runtime
        rows = active.nonzero(as_tuple=True)[0]
        if not torch.equal(context.env_ids, runtime.env_ids):
            raise ValueError("Contact context row identity changed.")
        target = runtime.robot.get_qpos(target=True).clone()
        view = copy(runtime)
        view.env_ids = runtime.env_ids[rows]
        view.rows = runtime.rows[rows]
        view.origins = runtime.origins[rows]
        view.max_slip = runtime.max_slip.index_select(0, rows)
        view.robot = _ObservedRobot(runtime.robot, context, target)
        for name in ["home", "grasp_slip", "last_targets", "port_yaw", "grasp_active"]:
            setattr(
                view,
                name,
                {
                    k: v.index_select(0, rows)
                    for k, v in getattr(runtime, name, {}).items()
                },
            )
        view.grasps = {
            k: (d, b.index_select(0, rows)) for k, (d, b) in runtime.grasps.items()
        }
        view.pick_initial = {
            k: (d, b.index_select(0, rows))
            for k, (d, b) in runtime.pick_initial.items()
        }
        view._metric_cache_time = None
        view._metric_cache = {}
        return view, rows, target.index_select(0, context.env_ids)

    def resolve(
        self,
        phase: ContactPhase,
        request: ResolvedSkillRequest,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> ResolvedContactPhase:
        """Resolve only selected rows, before any batch-wide timing reduction."""
        runtime = self.runtime
        runtime._scope_guard()
        B = context.batch_size
        motion = request.binding.endpoint("primary", "motion").target
        hand = request.binding.endpoint("primary", "grasp").target
        valid = active_mask.clone()
        if (
            isinstance(request.goal, ContactPlaceGoal)
            and phase is request.goal.phases[0]
        ):
            grasped = runtime.grasp_active.get(motion.control_part)
            valid &= torch.zeros_like(valid) if grasped is None else grasped

        held = (
            runtime.robot.get_qpos(target=True).index_select(0, context.env_ids).clone()
        )
        if not bool(valid.any()):
            shape = (
                (B, 2, 4, 4)
                if phase.kind == "pose"
                else (B, 4, 4) if phase.kind == "grasp" else (B, 2, len(hand.joint_ids))
            )
            return ResolvedContactPhase(
                torch.zeros(shape, device=held.device), held, valid
            )
        view, rows, held = self._view(context, valid)
        data = {
            "rows": rows,
            "part": motion.control_part,
            "operation_id": request.invocation_id or repr(request.goal),
        }
        target = runtime.cfg.target(phase.target)
        if phase.kind == "pose":
            result = _ground_cable_move(
                view, target, motion.control_part, context.require_control_dt()
            )
            local = result.goal.xpos
            values = (
                torch.eye(4, device=held.device).expand(B, local.shape[1], 4, 4).clone()
            )
            values[rows] = local
            for physical in result.failed_preconditions:
                valid[torch.nonzero(context.env_ids == physical).flatten()] = False
            data["last_target"] = view.last_targets[motion.control_part]
        elif phase.kind == "grasp":
            # Resolve only target geometry; reference state is installed when execution accepts the plan.
            local = view.pose(target, motion.control_part)
            values = torch.eye(4, device=held.device).expand(B, 4, 4).clone()
            values[rows] = local
            close = request.binding.endpoint("primary", "grasp").joint_positions(
                "grasp", num_envs=B, device=held.device, dtype=held.dtype
            )
            if target.firm:
                close = torch.zeros_like(close)
            data["closed_qpos"] = close
            data["pick_z"] = view.point(target.material_distance)[0][:, 2]
            data["hand"] = hand.control_part
            if motion.control_part in view.port_yaw:
                data["port_yaw"] = view.port_yaw[motion.control_part]
        else:
            result = _ground_cable_joint(view, target, hand.control_part, view.env_ids)
            local = result.goal.target
            values = (
                held[:, None, list(hand.joint_ids)]
                .expand(-1, local.shape[1], -1)
                .clone()
            )
            values[rows] = local
            if target.kind == "open":
                data["release_part"] = target.release_part
        return ResolvedContactPhase(
            values,
            held,
            valid,
            "grasp" if phase.kind == "joint" else "motion",
            tuple(data.items()),
        )

    def hold_steps(self, target: str) -> int:
        """Read explicit settling duration from its configured target."""
        t = self.runtime.cfg.target(target)
        if t.kind != "hold":
            raise ValueError("Expected declared hold target.")
        return t.samples

    def begin(
        self,
        cursor: ContactContinuation,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> None:
        """Install accepted references without capturing pre-close grasp evidence."""
        runtime = self.runtime
        runtime._scope_guard()
        if not bool(active_mask.any()):
            return
        data = dict(cursor.resolved.data)
        owner = data["operation_id"]
        if self.owner is not None and self.owner != owner:
            raise ValueError("One contact source cannot execute concurrent operations.")
        self.current = cursor
        self.observed_checks = set()
        self.rejected = torch.zeros_like(active_mask)
        self.owner = owner
        rows = active_mask.nonzero(as_tuple=True)[0]
        original_rows = data["rows"]
        select = torch.isin(original_rows, rows)
        dest = original_rows[select]
        part = data["part"]
        target = runtime.cfg.target(cursor.phase.target)
        if "last_target" in data:
            value = runtime.last_targets.get(part, runtime.tool(part)).clone()
            value[dest] = data["last_target"][select]
            runtime.last_targets[part] = value
        if cursor.phase.kind == "grasp":
            runtime.pending_pick = (part, target)
            runtime.pending_pick_rows = active_mask.clone()
            runtime.hand_to_motion[data["hand"]] = part
            previous = runtime.pick_initial.get(
                part, (target.material_distance, torch.zeros_like(runtime.max_slip))
            )[1].clone()
            previous[dest] = data["pick_z"][select]
            runtime.pick_initial[part] = (target.material_distance, previous)
            if "port_yaw" in data:
                yaw = runtime.port_yaw.get(
                    part, torch.zeros_like(runtime.max_slip)
                ).clone()
                yaw[dest] = data["port_yaw"][select]
                runtime.port_yaw[part] = yaw
        if "release_part" in data:
            key = data["release_part"]
            if key in runtime.grasp_active:
                runtime.grasp_active[key][active_mask] = False
            if key in runtime.grasps and not bool(
                runtime.grasp_active.get(key, torch.zeros_like(active_mask)).any()
            ):
                runtime.grasps.pop(key, None)
                runtime.grasp_slip.pop(key, None)
            runtime.pending_pick = None
        self.reports.append(
            {
                "event": "phase_started",
                "phase": cursor.phase_name,
                "time": context.robot.timestamp,
                "active": active_mask.cpu().tolist(),
            }
        )

    def observe(self, action: object, runtime: SemanticCallExecutor) -> None:
        """Evaluate checkpoints only after their physical commands are acknowledged."""
        cursor = self.current
        if cursor is None:
            return
        reference = runtime.last_sent_plan_reference
        if reference is None:
            return
        index = reference[2]
        for after, target in cursor.checkpoints:
            checkpoint = (after, target)
            if index != after or checkpoint in self.observed_checks:
                continue
            self.observed_checks.add(checkpoint)
            part = dict(cursor.resolved.data)["part"]
            passed, details = self.runtime.condition(
                self.runtime.cfg.target(target), part
            )
            self.rejected |= ~passed
            self.reports.append(
                {
                    "event": "condition",
                    "phase": cursor.phase_name,
                    "condition": target,
                    "after_index": after,
                    "passed": passed.cpu().tolist(),
                    "details": details,
                }
            )

    def finish(
        self,
        cursor: ContactContinuation,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Accept rows with all required physical evidence for the completed phase."""
        if self.current is not cursor and (
            self.current is None or self.current.position != cursor.position
        ):
            raise RuntimeError("Contact phase observation is stale.")
        expected = set(cursor.checkpoints)
        if expected != self.observed_checks:
            raise RuntimeError(
                "Contact path commands were not acknowledged by their observation host."
            )
        accepted = active_mask & ~self.rejected
        metrics = self.runtime.metrics()
        self.reports.append(
            {
                "event": "phase_completed",
                "phase": cursor.phase_name,
                "time": context.robot.timestamp,
                "accepted": accepted.cpu().tolist(),
                "endpoint": metrics["end_position"].cpu().tolist(),
                "max_slip": metrics["max_grasp_slip"].cpu().tolist(),
            }
        )
        messages = tuple(
            f"env {int(context.env_ids[i])}: contact condition failed in {cursor.phase_name}"
            for i in (active_mask & ~accepted).nonzero().flatten().tolist()
        )
        if cursor.position == cursor.limit:
            self.owner = None
        return accepted, messages


class _CableCompactPickLowerer(_CableLowerer):
    call_id: ClassVar[str] = "simulation.cable_pickup"
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        return _lower_compact(self, call, "pick_up")


class _CableCompactPlaceLowerer(_CableLowerer):
    call_id: ClassVar[str] = "simulation.cable_place"
    target_descriptor: ClassVar[SkillDescriptor] = Place.descriptor()

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        return _lower_compact(self, call, "place")


def _lower_compact(
    lowerer: _CableLowerer, call: RegisteredSemanticCall, kind: str
) -> SemanticLowering:
    if set(call.arguments) != {"target"}:
        raise ValueError("Contact operations require one declared target.")
    operation = next(
        (
            v
            for v in lowerer.cfg.operations
            if v.name == call.arguments["target"] and v.kind == kind
        ),
        None,
    )
    if operation is None:
        raise ValueError("Unknown contact operation.")
    if lowerer.runtime is None or lowerer.runtime._contact_source is None:
        raise RuntimeError("Contact operation was not bound to its observation source.")
    cls = ContactPickUpGoal if kind == "pick_up" else ContactPlaceGoal
    return SemanticLowering(
        goal=cls(lowerer.runtime._contact_source.source_id, operation.phases)
    )


@dataclass(frozen=True, slots=True)
class CableCompactPickFactory(_CableFactory):
    """Bind a contact pickup declaration to the ordinary PickUp primitive."""

    call_id: ClassVar[str] = _CableCompactPickLowerer.call_id
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()
    lowerer_type: ClassVar[type[_CableLowerer]] = _CableCompactPickLowerer

    @property
    def argument_contract(self) -> SemanticCallContract:
        return SemanticCallContract(
            required={
                "target": tuple(
                    o.name for o in self.cfg.operations if o.kind == "pick_up"
                )
            },
            optional={},
        )


@dataclass(frozen=True, slots=True)
class CableCompactPlaceFactory(_CableFactory):
    """Bind a contact placement declaration to the ordinary Place primitive."""

    call_id: ClassVar[str] = _CableCompactPlaceLowerer.call_id
    target_descriptor: ClassVar[SkillDescriptor] = Place.descriptor()
    lowerer_type: ClassVar[type[_CableLowerer]] = _CableCompactPlaceLowerer

    @property
    def argument_contract(self) -> SemanticCallContract:
        return SemanticCallContract(
            required={
                "target": tuple(
                    o.name for o in self.cfg.operations if o.kind == "place"
                )
            },
            optional={},
        )


__all__: list[str] = []
