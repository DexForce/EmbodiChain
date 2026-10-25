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

"""Goal-conditioned Place recovery retaining exact release and contact targets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import math
from typing import Any, TYPE_CHECKING

import torch

from embodichain.compute.trajectory import (
    interpolate_with_distance,
    interpolate_with_nums,
)
from embodichain.lab.sim.atomic_actions import JointPositionTarget
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose, resolve_pose_goal
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, TimedTrajectory
from embodichain.lab.sim.atomic_actions.primitives.place import PlaceGoal
from embodichain.lab.task_program.semantics import (
    Pick,
    HandOver,
    Place,
    RegisteredSemanticCall,
    SceneAffordanceRef,
    SceneObjectRef,
)
from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

from .adaptive_grasp import ADAPTIVE_GRASP
from .held_alignment_recovery import _RandomState, _cuda_devices
from .motion import _CARTESIAN_APPROACH, _joint_velocity_limits
from .pick_path import _pose_path_valid
from .pick_safety import _PICK_GRASP_RULE
from . import place_recovery
from .solver_tolerances import _pose_validation_parameters

__all__: list[str] = []
PRESERVED_LATERAL_RECOVERY_REVISION = 2
PRESERVED_ON_RECOVERY_REVISION = 2

if TYPE_CHECKING:
    from .place_transport import _ContactPhases


@dataclass(frozen=True)
class _HandoverPlaceRoute:
    place_id: str
    handover_id: str
    subject: str
    source: str
    receiver: str
    reference: str


@dataclass
class _Scope:
    action: Any
    request: Any
    context: Any
    velocity_phase_allocation: bool = False
    preserved_transport: bool = False
    release_branch_index: int = 0
    release_branches: list[torch.Tensor] | None = None
    groups: list[torch.Tensor] = field(default_factory=list)
    qposes: torch.Tensor | None = None
    solves: int = 0
    receipt: dict[str, Any] = field(default_factory=dict)
    contact: _ContactPhases | None = None


_SCOPE: ContextVar[_Scope | None] = ContextVar("gen_sim_place_branch", default=None)


def handover_place_declarations(value: object) -> tuple[str, ...]:
    """Validate unique explicit invocation permissions without changing defaults."""
    if (
        type(value) not in (list, tuple)
        or any(
            type(name) is not str or not name or name != name.strip() for name in value
        )
        or len(set(value)) != len(value)
    ):
        raise ValueError("Post-handover Place requires unique exact segment names.")
    return tuple(value)


def bind_handover_places(value: object, program: Any, scene_binding: Any) -> tuple:
    """Bind only an adjacent exact receiver's rigid on-release after HandOver."""
    remaining = set(handover_place_declarations(value))
    if not remaining:
        return ()
    rigid = {item.entity_id for item in scene_binding.rigid_objects}
    segments, result = tuple(program.iter_segments()), []
    for index, segment in enumerate(segments):
        if segment.name not in remaining:
            continue
        if index == 0:
            raise ValueError("Post-handover Place has no preceding HandOver.")
        previous = segments[index - 1]
        if any(
            len(s.calls) != 1 or s.parallel_block is not None
            for s in (previous, segment)
        ):
            raise ValueError("Post-handover Place requires adjacent sequential calls.")
        transfer, place = previous.calls[0].call, segment.calls[0].call
        if (
            type(transfer) is not HandOver
            or transfer.final_target is not None
            or set(transfer.resources) != {"source", "destination"}
            or transfer.resources["source"] == transfer.resources["destination"]
            or type(place) is not RegisteredSemanticCall
            or place.call_id != "simulation.place_relative"
            or set(place.arguments) != {"object", "reference", "relation"}
            or place.arguments["relation"] != "on"
            or place.arguments["object"] != transfer.object.entity_id
            or dict(place.resources) != {"primary": transfer.resources["destination"]}
            or transfer.object.entity_id not in rigid
            or place.arguments["reference"] not in rigid
            or place.arguments["reference"] == transfer.object.entity_id
        ):
            raise ValueError(
                "Post-handover Place must preserve object, receiver and rigid support."
            )
        identity = f"{program.program_id}/{segment.segment_id}:{segment.calls[0].segment_call_index}"
        result.append(
            _HandoverPlaceRoute(
                identity,
                f"{program.program_id}/{previous.segment_id}:{previous.calls[0].segment_call_index}",
                transfer.object.entity_id,
                transfer.resources["source"],
                transfer.resources["destination"],
                place.arguments["reference"],
            )
        )
        remaining.remove(segment.name)
    if remaining:
        raise ValueError("Post-handover Place references a missing segment.")
    return tuple(result)


def owns_post_handover(action: Any, request: Any, context: Any) -> bool:
    """Retain the receiver's actual end grip and observed orientation at release."""
    route = (getattr(action, "post_handover_places", None) or {}).get(
        getattr(request, "invocation_id", None)
    )
    if route is None or not action.is_bound:
        return False
    goal, options = request.goal, request.skill_options
    if (
        _SCOPE.get() is not None
        or context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or _CARTESIAN_APPROACH.get()
        or not isinstance(goal, PlaceGoal)
        or not isinstance(goal.xpos, SceneEntityPose)
        or goal.tcp_symmetry != "none"
        or not options.preserve_current_object_orientation
        or options.max_approach_retract_z is not None
        or request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.dynamic_collision_mode != "off"
        or getattr(action.motion_generator, "direct_place_planning", False)
    ):
        return False
    subject, resource, reference = route
    endpoint = request.binding.endpoint("primary", "motion")
    if endpoint.task_state_key != resource or goal.xpos.entity_id != reference:
        return False
    held = context.get_held_object(resource)
    affordance = getattr(getattr(held, "semantics", None), "affordance", None)
    return bool(
        affordance is not None
        and held.semantics.entity_id == subject
        and affordance.get_custom_config(ADAPTIVE_GRASP) == "ends"
        and affordance.get_custom_config(_PICK_GRASP_RULE) is None
    )


def bind_on_invocations(
    routes: Iterable[Any],
    purposes: Mapping[str, str] | Iterable[tuple[str, str]],
    scene_binding: Any,
) -> frozenset[str]:
    """Bind only typed same-object/resource ordinary releases onto rigid supports."""
    routes = tuple(routes)
    if not routes:
        return frozenset()
    purposes = dict(purposes)
    rigid = {b.entity_id for b in scene_binding.rigid_objects}
    result = set()
    for route in routes:
        pick, place = route.pick_call, route.place_call
        if (
            type(pick) is not Pick
            or type(place) not in (Place, RegisteredSemanticCall)
            or purposes.get(route.pick_id) != "ordinary"
        ):
            continue
        subject = pick.object.entity_id
        if subject not in rigid or set(pick.resources) != {"primary"}:
            continue
        if dict(place.resources) != dict(pick.resources):
            continue
        if type(place) is Place and place.object.entity_id == subject:
            target = place.on
            if type(target) is SceneAffordanceRef:
                matches = [
                    b
                    for b in scene_binding.support_surfaces
                    if b.entity_id == target.entity_id
                ]
                if len(matches) != 1:
                    continue
                parent = matches[0].parent_id
                if target.entity_id != f"on__{parent}__{subject}":
                    continue
            elif type(target) is SceneObjectRef:
                parent = target.entity_id
            else:
                continue
        elif type(place) is RegisteredSemanticCall:
            args = place.arguments
            if (
                place.call_id != "simulation.place_relative"
                or args.get("object") != subject
                or args.get("relation") != "on"
                or args.get("reference_part") is not None
                or set(args)
                - {"object", "reference", "relation", "stage_id", "reference_part"}
            ):
                continue
            parent = args.get("reference")
        else:
            continue
        if parent in rigid and parent != subject:
            result.add(route.place_id)
    return frozenset(result)


def bind_lateral_invocations(
    routes: Iterable[Any],
    purposes: Mapping[str, str] | Iterable[tuple[str, str]],
    scene_binding: Any,
) -> frozenset[str]:
    """Bind ordinary same-resource lateral releases, not authored pose goals."""
    routes = tuple(routes)
    if not routes:
        return frozenset()
    rigid = {item.entity_id for item in scene_binding.rigid_objects}
    purposes = dict(purposes)
    result = set()
    for route in routes:
        pick, place = route.pick_call, route.place_call
        if (
            type(pick) is not Pick
            or pick.grasp is not None
            or type(place) is not RegisteredSemanticCall
            or place.call_id != "simulation.place_relative"
            or purposes.get(route.pick_id) != "ordinary"
            or dict(pick.resources) != dict(place.resources)
            or set(place.resources) != {"primary"}
        ):
            continue
        args = place.arguments
        if (
            set(args) - {"object", "reference", "relation", "stage_id"}
            or args.get("relation") not in {"left_of", "right_of", "front_of", "behind"}
            or args.get("object") != pick.object.entity_id
            or args.get("object") not in rigid
            or args.get("reference") not in rigid
            or args.get("reference") == args.get("object")
        ):
            continue
        result.add(route.place_id)
    return frozenset(result)


def owns(action: Any, request: Any, context: Any) -> bool:
    """Own only an unconstrained live ordinary release with a new target R."""
    return _owns_ordinary(action, request, context, preserve=False)


def _owns_ordinary(
    action: Any, request: Any, context: Any, *, preserve: bool, on_support: bool = False
) -> bool:
    if not action.is_bound or getattr(request, "invocation_id", None) not in getattr(
        action,
        (
            "ordinary_lateral_invocations"
            if preserve and not on_support
            else "ordinary_on_invocations"
        ),
        (),
    ):
        return False
    goal = getattr(request, "goal", None)
    if not isinstance(goal, PlaceGoal):
        return False
    options = request.skill_options
    if (
        _SCOPE.get() is not None
        or context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or _CARTESIAN_APPROACH.get()
        or not isinstance(goal.xpos, SceneEntityPose)
        or (not preserve and goal.xpos.world_orientation is not None)
        or goal.tcp_symmetry != "none"
        or options.preserve_current_object_orientation != preserve
        or (preserve and options.cartesian_waypoint_count != 2)
        or options.max_approach_retract_z is not None
        or request.motion_policy.strategy != "ik_interp"
        or request.motion_policy.dynamic_collision_mode != "off"
        or getattr(action.motion_generator, "direct_place_planning", False)
    ):
        return False
    key = request.binding.endpoint("primary", "motion").task_state_key
    held = context.get_held_object(key)
    affordance = getattr(getattr(held, "semantics", None), "affordance", None)
    return bool(
        affordance is not None
        and affordance.get_custom_config(ADAPTIVE_GRASP) in {None, "ordinary"}
        and affordance.get_custom_config(_PICK_GRASP_RULE) is None
    )


def recover_preserved_lateral(
    action: Any,
    request: Any,
    context: Any,
    rejected: ActionPlan,
    build: Callable[[Any, Any], ActionPlan],
) -> ActionPlan:
    """Replace only a self-colliding waypoint transport, never the release goal."""
    return _recover_preserved(
        action, request, context, rejected, build, on_support=False
    )


def recover_preserved_on(
    action: Any,
    request: Any,
    context: Any,
    rejected: ActionPlan,
    build: Callable[[Any, Any], ActionPlan],
) -> ActionPlan:
    """Recover a support-colliding free prefix without changing the on-release."""
    return _recover_preserved(
        action, request, context, rejected, build, on_support=True
    )


def _recover_preserved(
    action: Any,
    request: Any,
    context: Any,
    rejected: ActionPlan,
    build: Callable[[Any, Any], ActionPlan],
    *,
    on_support: bool,
) -> ActionPlan:
    probe = rejected.diagnostics.metadata.get("gen_sim_place_waypoint_probe", {})
    acceptance = probe.get("acceptance", {})
    if (
        rejected.plan_success.any()
        or not _owns_ordinary(
            action, request, context, preserve=True, on_support=on_support
        )
        or not (
            acceptance.get("reason") == "held_object_scene_envelope_rejected"
            and acceptance.get("entity")
            == (getattr(action, "support_surface_owners", None) or {}).get(
                request.goal.xpos.entity_id, request.goal.xpos.entity_id
            )
            if on_support
            else acceptance.get("reason") == "robot_proxy_rejected"
            and acceptance.get("collision", {}).get("kind") == "self"
        )
    ):
        return rejected
    candidate = plan(
        action,
        request,
        context,
        build,
        **(
            {"preserved_on_transport": True}
            if on_support
            else {"preserved_transport": True}
        ),
    )
    receipt = candidate.diagnostics.metadata["gen_sim_place_branch"]
    chosen = candidate if candidate.plan_success.all() else rejected
    return replace(
        chosen,
        diagnostics=replace(
            chosen.diagnostics,
            metadata={
                **chosen.diagnostics.metadata,
                "gen_sim_place_branch": receipt,
                **(
                    {"gen_sim_place_waypoint_recovery": receipt["inspection"]}
                    if candidate.plan_success.all()
                    else {}
                ),
            },
        ),
    )


def capture_keyframes(action: Any, poses: torch.Tensor) -> torch.Tensor:
    """Retain the shared Place builder's down/back poses without changing them."""
    scope = _SCOPE.get()
    if scope is not None and scope.action is action:
        if poses.ndim != 4 or poses.shape[0] != 1 or poses.shape[2:] != (4, 4):
            raise ValueError("Place keyframes require one batched SE(3) sequence.")
        if not poses.shape[1] or not torch.isfinite(poses).all():
            raise ValueError("Place keyframes must be finite and nonempty.")
        if len(scope.groups) >= 2:
            raise ValueError("Place must emit exactly one down and one back group.")
        scope.groups.append(poses.detach().clone())
    return poses


def _tolerances(robot: Any, part: str) -> tuple[float, float]:
    params = _pose_validation_parameters(robot.get_solver(part))
    result = params["pos_eps"], params["rot_eps"]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("Place requires the original finite solver tolerances.")
    return result


def _phases(plan: ActionPlan) -> tuple[Any, Any, Any]:
    approach, release, retract = plan.segments
    if (
        (approach.name, release.name, retract.name)
        != ("approach", "release", "retract")
        or approach.start != 0
        or approach.stop != release.start
        or release.stop != retract.start
        or retract.stop != plan.joint_trajectory.waypoint_count
    ):
        raise ValueError(
            "Place must retain contiguous approach/release/retract phases."
        )
    return approach, release, retract


def inspect_emitted(
    action: Any, request: Any, context: Any, plan: ActionPlan
) -> dict[str, Any]:
    """Qualify final commands and the exact release target, never physical release."""
    try:
        if not plan.plan_success.all() or plan.joint_trajectory is None:
            return {"valid": False, "reason": "plan_failed"}
        _, release, _ = _phases(plan)
        endpoint = request.binding.endpoint("primary", "motion")
        effects = plan.expected_effects
        if (
            dict(effects.held_object_updates) != {endpoint.task_state_key: None}
            or effects.coordinated_held_object_updates
            or effects.articulation_joint_updates
        ):
            return {"valid": False, "reason": "release_effect_resource_mismatch"}
        checked = place_recovery.inspect(action, request, context, plan)
        if not checked["valid"]:
            return checked
        arm = endpoint.require_target(JointPositionTarget)
        desired = resolve_pose_goal(
            request.goal.xpos, context, name="place_release_target"
        )
        if desired.ndim == 4:
            desired = desired[:, -1]
        elif desired.ndim == 2:
            desired = desired[None]
        actual = action.robot.compute_fk(
            qpos=plan.joint_trajectory.positions[:, release.start, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        valid = _pose_path_valid(
            actual, desired, *_tolerances(action.robot, arm.control_part)
        )
        return {
            **checked,
            "valid": valid,
            "release_pose_valid": valid,
            **({} if valid else {"reason": "release_pose_fk_rejected"}),
            "physical_evidence": False,
        }
    except (ValueError, RuntimeError) as error:
        return {
            "valid": False,
            "reason": "emitted_evidence_unavailable",
            "unavailable": str(error),
        }


def _qualification(
    robot: Any,
    part: str,
    q: torch.Tensor,
    desired: torch.Tensor,
    ok: torch.Tensor,
    bounds: torch.Tensor,
) -> dict[str, Any]:
    if q.shape != (*desired.shape[:-2], bounds.shape[1]):
        raise ValueError("Place IK qpos shape differs from required waypoints.")
    finite = bool(torch.isfinite(q).all())
    selected = bounds if q.ndim == 2 else bounds[:, None]
    bounded = finite and bool(((q >= selected[..., 0]) & (q <= selected[..., 1])).all())
    record = {
        "ik_ok": bool(
            torch.as_tensor(ok).numel() == math.prod(desired.shape[:-2])
            and torch.as_tensor(ok).all()
        ),
        "finite": finite,
        "in_limits": bounded,
        "fk_valid": False,
        "desired_eef": desired.detach().cpu().tolist(),
    }
    if finite:
        fk = (
            robot.compute_fk(qpos=q, name=part, to_matrix=True)
            if q.ndim == 2
            else robot.compute_batch_fk(qpos=q, name=part, to_matrix=True)
        )
        record["fk_valid"] = _pose_path_valid(fk, desired, *_tolerances(robot, part))
        if torch.isfinite(fk).all():
            record["position_error_m"] = (
                torch.linalg.vector_norm(fk[..., :3, 3] - desired[..., :3, 3], dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
            record["rotation_error_rad"] = (
                quat_error_magnitude(
                    quat_from_matrix(fk[..., :3, :3]),
                    quat_from_matrix(desired[..., :3, :3]),
                )
                .detach()
                .cpu()
                .tolist()
            )
    record["valid"] = all(
        record[k] for k in ("ik_ok", "finite", "in_limits", "fk_valid")
    )
    return record


def solve_waypoints(
    generator: Any, poses: torch.Tensor, start_qpos: torch.Tensor, control_part: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Solve release once, then every required pose once with that common seed."""
    scope = _SCOPE.get()
    if scope is None:
        return None
    scope.solves += 1
    if scope.solves != 1 or generator is not scope.action.motion_generator:
        raise ValueError("Ordinary Place permits one owned waypoint solve, no retry.")
    if len(scope.groups) != 2 or not torch.equal(poses, torch.cat(scope.groups, 1)):
        raise ValueError("Place solver targets differ from captured phase keyframes.")
    arm = scope.request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    expected = scope.context.robot.qpos[:, list(arm.joint_ids)].to(start_qpos)
    if control_part != arm.control_part or not torch.equal(start_qpos, expected):
        raise ValueError("Place solver must start at the actual owning arm state.")
    bounds = generator.robot.get_qpos_limits(name=control_part).to(start_qpos)
    if (
        bounds.shape != (*start_qpos.shape, 2)
        or not torch.isfinite(bounds).all()
        or (bounds[..., 0] > bounds[..., 1]).any()
    ):
        raise ValueError("Place requires the original finite joint bounds.")
    release = scope.groups[0][:, -1].to(start_qpos)
    if scope.preserved_transport and scope.release_branches is None:
        scope.release_branches = _release_branches(
            generator.robot,
            control_part,
            release,
            start_qpos,
            scope.context.require_control_dt(),
        )
    if scope.release_branches is not None:
        if scope.release_branch_index >= len(scope.release_branches):
            raise ValueError("No qualified Place release branch remains.")
        qrelease = scope.release_branches[scope.release_branch_index]
        ok = torch.ones(qrelease.shape[:-1], dtype=torch.bool, device=qrelease.device)
        scope.receipt["release_branch_index"] = scope.release_branch_index
        scope.receipt["release_branch_count"] = len(scope.release_branches)
    else:
        ok, qrelease = generator.robot.compute_ik(
            pose=release, name=control_part, joint_seed=start_qpos
        )
    qrelease = torch.as_tensor(qrelease).to(start_qpos)
    measured = _qualification(
        generator.robot, control_part, qrelease, release, ok, bounds
    )
    scope.receipt["release_seed"] = measured
    if measured["valid"]:
        ok, qposes = generator.robot.compute_batch_ik(
            pose=poses,
            name=control_part,
            joint_seed=qrelease[:, None].expand(-1, poses.shape[1], -1).clone(),
        )
        qposes = torch.as_tensor(qposes).to(start_qpos)
        measured = _qualification(
            generator.robot, control_part, qposes, poses, ok, bounds
        )
        scope.receipt["waypoints"] = measured
        if measured["valid"]:
            if scope.velocity_phase_allocation:
                from .place_transport import prepare_contact

                qposes, scope.contact = prepare_contact(scope, poses, qposes, arm)
            scope.qposes = qposes.detach().clone()
            return torch.ones(1, dtype=torch.bool, device=start_qpos.device), qposes
    scope.receipt["reason"] = "required_waypoint_ik_fk_limits_rejected"
    return (
        torch.zeros(1, dtype=torch.bool, device=start_qpos.device),
        start_qpos[:, None].expand(-1, poses.shape[1], -1).clone(),
    )


def _release_branches(
    robot: Any, part: str, pose: torch.Tensor, start: torch.Tensor, dt: float
) -> list[torch.Tensor] | None:
    """Reuse the existing four-branch CPU IK search without changing other solvers."""
    from embodichain.lab.sim.motion.solvers import PytorchSolver
    from .articulation_recovery import _Search

    solver = robot.get_solver(part)
    if (
        not isinstance(solver, PytorchSolver)
        or solver.device.type != "cpu"
        or solver._seed_sampler is not None
    ):
        return None
    search = _Search(robot, part, dt)
    return [q.to(start) for _, q in search.endpoints(search.to_root(pose), start)]


def _velocity_phase(
    keys: torch.Tensor, samples: int, limits: torch.Tensor, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate a fixed phase grid by each edge's slowest controlled joint."""
    if (
        keys.ndim != 3
        or keys.shape[0] != 1
        or keys.shape[1] < 2
        or limits.shape != (1, keys.shape[-1])
        or not torch.isfinite(keys).all()
        or not torch.isfinite(limits).all()
        or (limits <= 0).any()
        or not math.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Place phase requires finite single-row keys, limits and dt.")
    intervals = (
        torch.ceil(
            ((keys[:, 1:] - keys[:, :-1]).abs() / (limits[:, None] * dt)).amax(-1)[0]
        )
        .to(torch.int64)
        .clamp_min(1)
    )
    spare = samples - 1 - int(intervals.sum())
    if spare < 0:
        raise ValueError("Place phase cannot fit the original duration at its limits.")
    intervals += spare // intervals.numel()
    intervals[: spare % intervals.numel()] += 1
    values = interpolate_with_nums(keys, intervals, device=keys.device)
    values[:, intervals.cumsum(0)] = keys[:, 1:]
    return values, intervals


def _rebuild(scope: _Scope, original: ActionPlan) -> ActionPlan:
    if scope.solves != 1 or scope.qposes is None or not original.plan_success.all():
        raise ValueError("Place shared builder did not qualify every required pose.")
    action, request, context = scope.action, scope.request, scope.context
    arm = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    ids = list(arm.joint_ids)
    approach, release, retract = _phases(original)
    n_down = scope.groups[0].shape[1]
    down, back = scope.qposes[:, :n_down], scope.qposes[:, n_down:]
    anchor = down[:, -1:]
    full = original.joint_trajectory.positions.clone()
    lengths = {s.name: s.stop - s.start for s in original.segments}
    times = original.joint_trajectory.dt
    if scope.contact is not None:
        from .place_transport import compose

        full, lengths, times = compose(scope, original, arm)
        scope.groups = list(scope.contact.desired)
        scope.qposes = scope.contact.qposes
        n_down = scope.groups[0].shape[1]
        down, back = scope.qposes[:, :n_down], scope.qposes[:, n_down:]
    elif scope.velocity_phase_allocation:
        steps = original.joint_trajectory.dt[:, 1:]
        if (
            not steps.numel()
            or not (steps > 0).all()
            or not (steps == steps[0, 0]).all()
        ):
            raise ValueError("Post-handover Place requires its uniform control grid.")
        limits = _joint_velocity_limits(action.robot, None).to(full)[:, ids]
        step = float(steps[0, 0])
        scope.receipt["velocity_phase_intervals"] = {}
    for segment, keys in (
        ()
        if scope.contact is not None
        else (
            (approach, torch.cat((context.robot.qpos[:, None, ids], down), 1)),
            (retract, torch.cat((anchor, back), 1)),
        )
    ):
        samples = segment.stop - segment.start
        if scope.velocity_phase_allocation:
            values, intervals = _velocity_phase(keys, samples, limits, step)
            scope.receipt["velocity_phase_intervals"][segment.name] = intervals.tolist()
        else:
            values = interpolate_with_distance(keys, samples, device=full.device)
        full[:, segment.start : segment.stop, ids] = values
    if scope.contact is None:
        full[:, release.start : release.stop, ids] = anchor
    candidate = action.build_plan(
        request,
        context,
        success=original.plan_success,
        trajectory=TimedTrajectory.from_positions(
            full,
            env_ids=original.joint_trajectory.env_ids,
            dt=times,
        ),
        expected_effects=original.expected_effects,
        effect_candidates=original.effect_candidates,
        effect_verification=original.effect_verification,
        replannable=original.replannable,
        diagnostics=original.diagnostics,
        segment_lengths=lengths,
        scene_dependency_monitor_until=(
            scope.contact.monitor_until
            if scope.contact is not None
            else original.scene_dependency_monitor_until
        ),
        scene_dependency_end_segment=original.scene_dependency_end_segment,
    )
    candidate = replace(candidate, scene_dependencies=original.scene_dependencies)
    approach, release, retract = _phases(candidate)
    # Validate mandatory samples after phase composition, not the discarded split.
    scope.receipt["mandatory_indices"] = {}
    for segment, qkeys, desired in (
        (approach, down, scope.groups[0]),
        (retract, back, scope.groups[1]),
    ):
        values = candidate.joint_trajectory.positions[
            :, segment.start : segment.stop, ids
        ]
        indices, cursor = [], 1
        for key in qkeys[0]:
            matches = (values[0, cursor:] == key).all(-1).nonzero().flatten()
            if not matches.numel():
                raise ValueError("Place emitted phase dropped a mandatory waypoint.")
            index = cursor + int(matches[0])
            indices.append(index)
            cursor = index + 1
        scope.receipt["mandatory_indices"][segment.name] = [
            segment.start + index for index in indices
        ]
        fk = action.robot.compute_batch_fk(
            qpos=values[:, indices], name=arm.control_part, to_matrix=True
        )
        if not _pose_path_valid(
            fk, desired, *_tolerances(action.robot, arm.control_part)
        ):
            raise ValueError("Place emitted mandatory waypoint failed FK.")
    scope.receipt["emitted_mandatory_fk_valid"] = True
    scope.receipt["inspection"] = inspect_emitted(action, request, context, candidate)
    if not scope.receipt["inspection"]["valid"]:
        raise ValueError("Place emitted trajectory failed its original safety gates.")
    return candidate


def plan(
    action: Any,
    request: Any,
    context: Any,
    build: Callable[[Any, Any], ActionPlan],
    *,
    preserved_transport: bool = False,
    preserved_on_transport: bool = False,
) -> ActionPlan:
    """Build one shared Place plan and qualify it without fallback or dispatch."""
    post_handover = owns_post_handover(action, request, context)
    if preserved_transport and preserved_on_transport:
        raise ValueError("Place cannot select two preserved transport routes.")
    preserving = preserved_transport or preserved_on_transport
    owned = (
        _owns_ordinary(
            action, request, context, preserve=True, on_support=preserved_on_transport
        )
        if preserving
        else owns(action, request, context) or post_handover
    )
    if not owned:
        raise ValueError(
            "Place branch requires a bound ordinary on or post-handover invocation."
        )
    state = _RandomState.capture(
        _cuda_devices(action.robot.device, context.robot.qpos.device)
    )
    chosen = None
    branches = None
    attempts = []
    try:
        for index in range(4 if preserving else 1):
            state.restore()
            scope = _Scope(
                action,
                request,
                context,
                velocity_phase_allocation=post_handover or preserving,
                preserved_transport=preserving,
                release_branch_index=index,
                release_branches=branches,
            )
            token = _SCOPE.set(scope)
            try:
                chosen = _rebuild(scope, build(request, context))
            except (ValueError, RuntimeError) as error:
                scope.receipt["unavailable"] = str(error)
            finally:
                _SCOPE.reset(token)
            branches = scope.release_branches
            attempts.append(dict(scope.receipt))
            if chosen is not None or branches is None or index + 1 >= len(branches):
                break
    finally:
        state.restore()
    accepted = chosen is not None
    if not accepted:
        chosen = action.failed_plan(
            request,
            context,
            message="Ordinary Place waypoint branch was not qualified.",
            failure_code="gen_sim_place_branch_unqualified",
            retryable=False,
        )
    return replace(
        chosen,
        diagnostics=replace(
            chosen.diagnostics,
            metadata={
                **chosen.diagnostics.metadata,
                "gen_sim_place_branch": {
                    **scope.receipt,
                    "accepted": accepted,
                    "physical_evidence": False,
                    "scope": "goal_conditioned_waypoint_ik_and_emitted_phase_checks",
                    "mandatory_counts": [p.shape[1] for p in scope.groups],
                    "solve_requests": scope.solves,
                    "retries": len(attempts) - 1,
                    **({"release_branch_attempts": attempts} if preserving else {}),
                    "timing_unchanged": scope.receipt.get("timing_unchanged", True),
                },
            },
        ),
    )
