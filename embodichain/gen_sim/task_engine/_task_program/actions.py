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

"""GenSim-only policies on shared Atomic Action planning and execution."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.affordance import (
    AntipodalAffordance,
    AxisAlignAffordance,
)
from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
    AffordanceSample,
)
from embodichain.lab.sim.atomic_actions.core import ObjectSemantics
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
)
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
    MoveHeldObject,
)
from embodichain.lab.sim.atomic_actions.primitives.pour import Pour
from embodichain.lab.sim.atomic_actions.primitives.place import (
    Place,
    PlaceGoal,
    PlaceOptions,
)
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp, PickUpOptions
from embodichain.lab.sim.atomic_actions.primitives.hand_over import HandOver
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    PlanningFailure,
)
from embodichain.lab.sim.atomic_actions.state import TaskState
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    to_full_robot_trajectory,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

from .adaptive_grasp import (
    ADAPTIVE_GRASP,
    grasp_strategies,
    observed_axis,
    opposite_grasp_end,
    select_grasp,
)
from .articulation_collision import robot_collision_scope
from .held_alignment_recovery import _RandomState, _cuda_devices
from .pick_safety import (
    _PICK_GRASP_RULE,
    _inspect_pick_motion,
    _inspect_pick_plan,
    _owns_constrained_pick,
)
from .state_handoff import (
    GenSimStateHandoff,
    StagingContract,
    require_release_effect,
)

__all__ = [
    "GenSimPickUp",
    "GenSimHandOver",
    "GenSimMoveHeldObject",
    "GenSimPlace",
    "GenSimPour",
]


@dataclass
class _PickSelection:
    direction: torch.Tensor | None = None
    request: object = None
    trajectory: tuple | None = None
    constrained: bool = False
    closing_geometry: Any = None
    object_pose: torch.Tensor | None = None
    candidates: AffordancePoseCandidates | None = None
    candidate_routes: list[tuple[int, AffordancePoseCandidates, torch.Tensor]] = field(
        default_factory=list
    )
    raw_candidate_routes: list[tuple[int, AffordancePoseCandidates, torch.Tensor]] = (
        field(default_factory=list)
    )
    pour_preview: Any = None
    ik_audit: dict[str, Any] = field(default_factory=dict)
    active_ik_audit: dict[str, Any] | None = None
    depth_recovery_attempts: int = 0
    depth_trace: list[dict[str, Any]] = field(default_factory=list)
    closing_scene_checks: list[dict[str, Any]] = field(default_factory=list)
    native_candidate_expanders: dict[int, Callable[[], AffordancePoseCandidates]] = (
        field(default_factory=dict)
    )


def _pick_ik_tensor(value: Any) -> dict[str, Any]:
    tensor = torch.as_tensor(value)
    if not hasattr(value, "dtype") and tensor.is_floating_point():
        tensor = torch.as_tensor(value, dtype=torch.float64)
    tensor = tensor.detach().cpu()
    record = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    if tensor.is_complex() or not bool(torch.isfinite(tensor).all()):
        return {**record, "available": False, "reason": "nonfinite_or_complex"}
    return {**record, "available": True, "values": tensor.tolist()}


@contextmanager
def _pick_depth_random_scope(checkpoint: _RandomState | None):
    if checkpoint is None:
        yield
        return
    saved = _RandomState.capture(set(checkpoint.cuda))
    checkpoint.restore()
    try:
        yield
    finally:
        saved.restore()


_PICK_SELECTION: ContextVar[_PickSelection | None] = ContextVar(
    "gen_sim_pick_selection", default=None
)


@dataclass
class _ReceiveSelection:
    resources: object
    held: object
    alternate: torch.Tensor
    request: object
    screen: Any
    sampled: bool = False
    exchange_pose: torch.Tensor | None = None
    root_poses: tuple[torch.Tensor, torch.Tensor] | None = None


_RECEIVE_SELECTION: ContextVar[_ReceiveSelection | None] = ContextVar(
    "gen_sim_receive_selection", default=None
)


class GenSimPickUp(PickUp):
    """Apply explicit call-local grasp rules without changing shared sampling."""

    skill_id = PickUp.skill_id
    GoalType = PickUp.GoalType
    OptionsType = PickUp.OptionsType
    binding_contract = PickUp.binding_contract
    adaptive_unconstrained = False
    pick_purposes: dict[str, str] | None = None
    continuation_previews: dict | None = None
    path_simulation: object | None = None
    path_excluded_invocations: frozenset[str] = frozenset()
    closing_geometry_factory: Any = None
    pour_continuations: dict | None = None
    pour_scene_vertices: dict | None = None
    pour_region_geometries: dict | None = None
    pour_gravity_world: tuple[float, float, float] | None = None

    @staticmethod
    def _native_handover_source(request: Any, screen: Any) -> bool:
        """Opt in the lowerer's end-grasp contract, never an ordinary/Pour alias."""
        if screen is None or screen.revision != 8:
            return False
        semantics = getattr(getattr(request, "goal", None), "semantics", None)
        affordance = getattr(semantics, "affordance", None)
        return (
            affordance is not None
            and affordance.get_custom_config(ADAPTIVE_GRASP) == "ends"
        )

    def _approach_alignment_mask(self, grasp_poses, options, approach_direction):
        aligned = super()._approach_alignment_mask(
            grasp_poses, options, approach_direction
        )
        selection = _PICK_SELECTION.get()
        if selection is not None and selection.closing_geometry is not None:
            if selection.object_pose is None:
                raise ValueError("Pick closing screen has no exact object reference.")
            aligned = aligned & selection.closing_geometry.mask(
                grasp_poses, selection.object_pose
            )
        if selection is not None and selection.pour_preview is not None:
            if selection.object_pose is None:
                raise ValueError("Pour look-ahead requires the exact Pick object pose.")
            transforms = pose_inv(selection.object_pose)[:, None, None] @ grasp_poses
            aligned &= selection.pour_preview.valid_mask(transforms)
        if (
            selection is not None
            and selection.active_ik_audit is not None
            and not selection.active_ik_audit.get("omitted")
        ):
            selection.active_ik_audit["alignment_mask"] = _pick_ik_tensor(aligned)
        return aligned

    def _iter_downstream_eef_targets(self, object_to_eef_variants, options):
        selection = _PICK_SELECTION.get()
        if selection is None or selection.pour_preview is None:
            yield from super()._iter_downstream_eef_targets(
                object_to_eef_variants, options
            )
            return
        preview, group = selection.pour_preview, selection.active_ik_audit
        targets, valid = preview.targets(object_to_eef_variants)
        if group is None:
            yield from targets
            return
        if not group.get("omitted"):
            group["object_to_eef_variants"] = _pick_ik_tensor(object_to_eef_variants)
            group["geometry_valid_mask"] = _pick_ik_tensor(valid)
            if "operands" not in selection.ik_audit:
                selection.ik_audit["operands"] = {
                    name: _pick_ik_tensor(getattr(preview, name))
                    for name in (
                        "source_vertices",
                        "receiver_vertices",
                        "staging",
                        "receiver_pose",
                        "pad_line_tcp",
                        "gravity_world",
                    )
                }
                selection.ik_audit["recipe"] = {
                    name: (
                        value
                        if value is None or isinstance(value, str)
                        else _pick_ik_tensor(value)
                    )
                    for name, value in asdict(preview.route).items()
                }
        for role, target in zip(
            ("staging", "peak", "restore", "return_transport", "return_place"),
            targets,
            strict=True,
        ):
            if group is not None:
                group["next_role"] = role
            yield target

    def inspect_closing_plan(
        self, request: Any, context: Any, plan: Any, *, screen: Any = None
    ) -> dict[str, Any]:
        """Screen actual close TCP samples, including a transformed plan's poses."""
        from embodichain.lab.sim.atomic_actions.goals import (
            _resolve_object_pose,
            resolve_pose_goal,
        )
        from embodichain.lab.sim.atomic_actions.trajectory_ops import (
            resolve_pose_target,
        )

        if screen is None:
            screen = self.closing_geometry_factory(request, context)
        if screen is None:
            return {"valid": True, "scope": "not_a_bound_rigid_pick"}
        close = next(
            (segment for segment in plan.segments if segment.name == "close"), None
        )
        if close is None or close.stop <= close.start:
            raise ValueError("Pick closing requires its actual emitted close phase.")
        from embodichain.lab.sim.atomic_actions.trajectory_ops import (
            interpolate_hand_qpos,
        )

        hand_endpoint = request.binding.endpoint("primary", "grasp")
        hand = hand_endpoint.require_target(JointPositionTarget)
        hand_commands = plan.joint_trajectory.positions[
            :, close.start : close.stop, list(hand.joint_ids)
        ]
        opened, closed = (
            hand_endpoint.joint_positions(
                command,
                num_envs=context.batch_size,
                device=hand_commands.device,
                dtype=hand_commands.dtype,
            )
            for command in ("open", "grasp")
        )
        options = request.skill_options
        expected = torch.cat(
            (
                interpolate_hand_qpos(
                    opened, closed, n_waypoints=options.hand_interp_steps
                ),
                closed[:, None].expand(-1, options.grasp_settle_steps, -1),
            ),
            dim=1,
        )
        if not torch.equal(hand_commands, expected):
            return {"valid": False, "reason": "emitted_hand_closing_commands_changed"}
        endpoint = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        positions = plan.joint_trajectory.positions[
            :, close.start : close.stop, list(endpoint.joint_ids)
        ]
        poses = self.robot.compute_batch_fk(
            qpos=positions, name=endpoint.control_part, to_matrix=True
        )
        goal = request.goal
        object_pose = (
            _resolve_object_pose(goal.semantics, context, name="pickup_object_pose")
            if goal.object_pose is None
            else resolve_pose_target(
                resolve_pose_goal(goal.object_pose, context, name="object_pose"),
                num_envs=context.batch_size,
                device=poses.device,
            )
        )
        valid = bool(screen.mask(poses, object_pose).all())
        acceptance = {
            "valid": valid,
            "emitted_close_samples": poses.shape[1],
            **screen.report(),
        }
        if (
            valid
            and screen.revision == 8
            and (
                (self.pick_purposes or {}).get(request.invocation_id) == "ordinary"
                or self._native_handover_source(request, screen)
            )
        ):
            scene = self._inspect_closing_scene(
                request,
                context,
                plan.joint_trajectory.positions[0, close.start : close.stop],
                screen,
            )
            acceptance["closing_scene"] = scene
            acceptance["valid"] &= bool(scene["valid"])
        return acceptance

    def _qualify_closing_plan(self, request: Any, context: Any, plan: Any) -> Any:
        selection = _PICK_SELECTION.get()
        screen = None if selection is None else selection.closing_geometry
        if screen is None:
            return plan
        acceptance = (
            self.inspect_closing_plan(request, context, plan, screen=screen)
            if plan.plan_success.any()
            else {"valid": False, **screen.report()}
        )
        if "closing_scene" in acceptance:
            selection.closing_scene_checks.append(
                {"stage": "emitted", **acceptance["closing_scene"]}
            )
        if (
            acceptance["valid"]
            and screen.revision in (7, 8)
            and (
                (self.pick_purposes or {}).get(request.invocation_id) == "ordinary"
                or self._native_handover_source(request, screen)
            )
            and "closing_scene" not in acceptance
        ):
            close = next(s for s in plan.segments if s.name == "close")
            scene = self._inspect_closing_scene(
                request,
                context,
                plan.joint_trajectory.positions[0, close.start : close.stop],
                screen,
            )
            acceptance["closing_scene"] = scene
            acceptance["valid"] &= bool(scene["valid"])
            selection.closing_scene_checks.append({"stage": "emitted", **scene})
        metadata = {
            **plan.diagnostics.metadata,
            "gen_sim_pick_contact_geometry": acceptance,
        }
        return replace(
            plan,
            plan_success=(
                plan.plan_success
                if acceptance["valid"]
                else torch.zeros_like(plan.plan_success)
            ),
            diagnostics=replace(
                plan.diagnostics,
                metadata=metadata,
                failure=(
                    plan.diagnostics.failure
                    if acceptance["valid"] or not plan.plan_success.any()
                    else PlanningFailure(
                        "gen_sim_pick_contact_unqualified", retryable=False
                    )
                ),
            ),
        )

    def _compute_batch_candidate_ik(
        self,
        poses: torch.Tensor,
        joint_seed: torch.Tensor,
        manipulator: JointPositionTarget,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the episode's parallel budget only for large CPU IK batches."""
        from .motion import _CPU_BATCH_THREADS
        from embodichain.lab.sim.motion.solvers.pytorch_solver import PytorchSolver

        parent = super()._compute_batch_candidate_ik

        def solve():
            selection = _PICK_SELECTION.get()
            group = None if selection is None else selection.active_ik_audit
            record = None
            if (
                group is not None
                and not group.get("omitted")
                and len(group["calls"]) < 8
            ):
                seed = (
                    joint_seed[:, None, None, :].expand(*poses.shape[:3], -1)
                    if joint_seed.ndim == 2
                    else joint_seed
                )
                record = {
                    "role": (
                        group["next_role"]
                        if "next_role" in group
                        else ("pre_grasp", "grasp", "lift")[len(group["calls"])]
                    ),
                    "candidate_shape": list(poses.shape[:3]),
                    "control_part": manipulator.control_part,
                    "joint_ids": list(manipulator.joint_ids),
                    "targets": _pick_ik_tensor(poses.reshape(poses.shape[0], -1, 4, 4)),
                    "joint_seed": _pick_ik_tensor(
                        seed.reshape(poses.shape[0], -1, len(manipulator.joint_ids))
                    ),
                    "torch_cpu_rng_state": torch.get_rng_state().tolist(),
                    "torch_cuda_rng_states": {
                        str(i): torch.cuda.get_rng_state(i).tolist()
                        for i in _cuda_devices(poses.device)
                    },
                    "torch_intraop_threads": torch.get_num_threads(),
                }
                group["calls"].append(record)
            elif group is not None:
                group["dropped_calls"] = group.get("dropped_calls", 0) + 1
                group["trace_complete"] = False
                selection.ik_audit.update(trace_complete=False, truncated=True)
                selection.ik_audit["omitted_calls"] += 1
            result = parent(poses, joint_seed, manipulator)
            if record is not None:
                record.update(
                    raw_success=_pick_ik_tensor(result[0]),
                    output_qpos=_pick_ik_tensor(result[1]),
                )
            return result

        threads = _CPU_BATCH_THREADS.get()
        if (
            threads is None
            or poses.device.type != "cpu"
            or poses.shape[0] != 1
            or poses.shape[1] * poses.shape[2] < 100
            or not isinstance(
                self.robot.get_solver(manipulator.control_part), PytorchSolver
            )
        ):
            return solve()
        previous = torch.get_num_threads()
        if previous >= threads:
            return solve()
        try:
            torch.set_num_threads(threads)
            return solve()
        finally:
            torch.set_num_threads(previous)

    def _shared_grasp_variants(self, *args):
        from .pour_continuation import CandidatePreview, PourContinuation

        selection = _PICK_SELECTION.get()
        preview = None if selection is None else selection.pour_preview
        if (
            type(preview) is not CandidatePreview
            or type(preview.route) is not PourContinuation
        ):
            return super()._select_feasible_grasp_variants(*args)
        if not selection.ik_audit:
            selection.ik_audit = {
                "scope": "planning_inputs_not_physical",
                "group_limit": 8,
                "trace_complete": True,
                "truncated": False,
                "omitted_groups": 0,
                "omitted_calls": 0,
                "groups": [],
            }
        groups = selection.ik_audit["groups"]
        previous, group = selection.active_ik_audit, None
        if len(groups) < 8:
            group = {
                "calls": [],
                "trace_complete": True,
                "object_pose": _pick_ik_tensor(args[2]),
            }
            groups.append(group)
        else:
            group = {"calls": [], "omitted": True}
            selection.ik_audit["omitted_groups"] += 1
            selection.ik_audit.update(trace_complete=False, truncated=True)
        selection.active_ik_audit = group
        try:
            return super()._select_feasible_grasp_variants(*args)
        except Exception:
            group["trace_complete"] = False
            selection.ik_audit["trace_complete"] = False
            raise
        finally:
            selection.active_ik_audit = previous

    def _select_feasible_grasp_variants(
        self,
        grasp_xpos: torch.Tensor,
        start_qpos: torch.Tensor,
        object_poses: torch.Tensor,
        manipulator: JointPositionTarget,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
        *,
        candidate_costs: torch.Tensor | None = None,
        candidate_limit: int | None = None,
        first_feasible: bool = False,
        candidate_route: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefilter grasp columns, leaving IK and roll selection to shared PickUp."""
        selection = _PICK_SELECTION.get()
        checkpoint = (
            _RandomState.capture(_cuda_devices(grasp_xpos.device))
            if first_feasible and candidate_route is not None
            else None
        )
        if selection is not None:
            selection.object_pose = object_poses
        if candidate_limit is None and (
            options.rotate_upright is not None
            or options.approach_alignment_max_angle is None
        ):
            return self._shared_grasp_variants(
                grasp_xpos,
                start_qpos,
                object_poses,
                manipulator,
                options,
                approach_direction,
            )
        mirrored = grasp_xpos.clone()
        mirrored[..., :3, :2] *= -1
        variants = torch.stack((grasp_xpos, mirrored), dim=2)
        eef_variants = variants @ options.grasp_frame_to_eef.to(
            device=self.device, dtype=grasp_xpos.dtype
        )
        # Legacy callers retain their IK population and random draws. Native
        # proposals qualify both final TCP rolls before spending candidate slots.
        aligned = super()._approach_alignment_mask(
            eef_variants, options, approach_direction
        )
        # Keep the full environment axis and both roll variants of a retained
        # candidate. Only columns rejected by every row can be removed safely.
        keep = aligned.any(dim=(0, 2))
        indices = torch.nonzero(keep, as_tuple=False).flatten()
        selected = eef_variants[:, :, 0].clone()
        success = torch.zeros(
            grasp_xpos.shape[:2], dtype=torch.bool, device=self.device
        )
        if candidate_limit is not None:
            if (
                grasp_xpos.shape[0] != 1
                or candidate_costs is None
                or candidate_costs.shape != grasp_xpos.shape[:2]
            ):
                raise ValueError(
                    "Native candidate ranking requires B1 and aligned costs."
                )
            eligible = indices[torch.isfinite(candidate_costs[0, indices])]
            order = torch.argsort(candidate_costs[0, eligible], stable=True)
            ranked = eligible[order]
            retained, count, checked = [], 0, 0
            while checked < len(ranked) and count < candidate_limit:
                width = (
                    min(50, candidate_limit - count)
                    if first_feasible
                    else candidate_limit - count
                )
                stop = min(len(ranked), checked + width)
                chunk = ranked[checked:stop]
                qualified = self._approach_alignment_mask(
                    eef_variants[:, chunk], options, approach_direction
                ).any(dim=(0, 2))
                accepted = chunk[qualified]
                retained.append(accepted)
                count += len(accepted)
                checked = stop
                if first_feasible and len(accepted):
                    poses, valid = self._shared_grasp_variants(
                        grasp_xpos[:, accepted],
                        start_qpos,
                        object_poses,
                        manipulator,
                        options,
                        approach_direction,
                    )
                    selected[:, accepted] = poses
                    success[:, accepted] = valid
                    if valid.any():
                        break
            indices = torch.cat(retained) if retained else ranked[:0]
            if selection is not None:
                selection.closing_scene_checks.append(
                    {
                        "stage": "native_candidate_limit",
                        "raw_candidates": grasp_xpos.shape[1],
                        "qualified_before_limit": len(indices),
                        "qualification_exhaustive": checked == len(ranked),
                        "native_candidates_queried": checked,
                        "direction_eligible_candidates": len(ranked),
                        "retained": len(indices),
                        "limit": candidate_limit,
                        "physical_evidence": False,
                        "first_feasible_batch": first_feasible,
                    }
                )
            if first_feasible:
                if (
                    selection is not None
                    and candidate_route is not None
                    and success.any()
                    and checked < len(ranked)
                    and count < candidate_limit
                ):

                    def expand():
                        with _pick_depth_random_scope(checkpoint):
                            poses, valid = self._select_feasible_grasp_variants(
                                grasp_xpos,
                                start_qpos,
                                object_poses,
                                manipulator,
                                options,
                                approach_direction,
                                candidate_costs=candidate_costs,
                                candidate_limit=candidate_limit,
                            )
                        return AffordancePoseCandidates(poses, candidate_costs, valid)

                    selection.native_candidate_expanders[candidate_route] = expand
                return selected, success
        if len(indices):
            poses, valid = self._shared_grasp_variants(
                grasp_xpos[:, indices],
                start_qpos,
                object_poses,
                manipulator,
                options,
                approach_direction,
            )
            selected[:, indices] = poses
            success[:, indices] = valid
        return selected, success

    def _expand_native_pick_candidates(self, selection: _PickSelection) -> None:
        """Restore the original filtered recovery population only after path failure."""
        if selection.depth_recovery_attempts >= 4:
            return
        for index, (strategy, pool, direction) in enumerate(selection.candidate_routes):
            expand = selection.native_candidate_expanders.pop(strategy, None)
            if expand is not None:
                selection.candidate_routes[index] = (strategy, expand(), direction)

    def _plan(self, request, context):
        if self.adaptive_unconstrained:
            goal, options = request.goal, request.skill_options
            affordance = goal.semantics.affordance
            if (
                goal.grasp_xpos is None
                and options.fixed_object_to_eef is None
                and isinstance(affordance, AntipodalAffordance)
                and affordance.get_custom_config(ADAPTIVE_GRASP) is None
                and affordance.get_custom_config(_PICK_GRASP_RULE) is None
                and options.pick_object_part == "center"
                and options.rotate_upright is None
                and options.obj_upright_direction is None
                and torch.equal(
                    options.approach_direction,
                    options.approach_direction.new_tensor([0.0, 0.0, -1.0]),
                )
            ):
                semantics = deepcopy(goal.semantics)
                purpose = (
                    "ordinary"
                    if self.pick_purposes is None
                    else self.pick_purposes.get(request.invocation_id)
                )
                if purpose not in {"ordinary", "pour"}:
                    raise ValueError(
                        "Unbound GenSim Pick purpose; regenerate the bundle."
                    )
                semantics.affordance.set_custom_config(ADAPTIVE_GRASP, purpose)
                request = replace(request, goal=replace(goal, semantics=semantics))
        selection = _PickSelection(request=request)
        pour_route = (self.pour_continuations or {}).get(request.invocation_id)
        if pour_route is not None:
            from .pour_continuation import prepare_preview

            selection.pour_preview = prepare_preview(self, request, context, pour_route)
        if self.closing_geometry_factory is not None:
            selection.closing_geometry = self.closing_geometry_factory(request, context)

        def finish(plan):
            plan = self._qualify_closing_plan(request, context, plan)
            if selection.closing_scene_checks:
                plan = replace(
                    plan,
                    diagnostics=replace(
                        plan.diagnostics,
                        metadata={
                            **plan.diagnostics.metadata,
                            "gen_sim_pick_closing_scene_search": deepcopy(
                                selection.closing_scene_checks
                            ),
                        },
                    ),
                )
            if selection.depth_trace:
                plan = replace(
                    plan,
                    diagnostics=replace(
                        plan.diagnostics,
                        metadata={
                            **plan.diagnostics.metadata,
                            "gen_sim_pick_native_depth_search": {
                                "attempts": deepcopy(selection.depth_trace),
                                "extra_full_paths": selection.depth_recovery_attempts,
                                "shared_recovery_budget": 4,
                                "physical_evidence": False,
                            },
                        },
                    ),
                )
            if selection.ik_audit and not plan.plan_success.any():
                plan = replace(
                    plan,
                    diagnostics=replace(
                        plan.diagnostics,
                        metadata={
                            **plan.diagnostics.metadata,
                            "gen_sim_pour_pick_ik_audit": deepcopy(selection.ik_audit),
                        },
                    ),
                )
            return plan

        token = _PICK_SELECTION.set(selection)
        try:
            with robot_collision_scope():
                from .pick_candidate_path import owns_fixed

                if owns_fixed(self, request, context):
                    return self._plan_fixed_pick_candidate(request, context, selection)
                first = finish(super()._plan(request, context))
                preview = (self.continuation_previews or {}).get(
                    getattr(request, "invocation_id", None)
                )
                if preview is not None:
                    from .pick_continuation import eligible, recover

                    if eligible(self, request, context, selection, preview):
                        first = finish(
                            recover(self, request, context, first, selection, preview)
                        )
                from .pick_path import eligible as owns_path, rebuild

                if owns_path(self, request, context, selection):
                    qualified = finish(
                        rebuild(self, request, context, first, selection)
                    )
                    if qualified.plan_success.all():
                        return finish(qualified)
                    self._expand_native_pick_candidates(selection)
                    first = self._retry_ordinary_pick_candidates(
                        request, context, qualified, selection
                    )
                if selection.constrained:
                    first = self._recover_constrained_pick_roll(request, context, first)
                if selection.raw_candidate_routes and not first.plan_success.any():
                    from .pick_candidate_path import eligible as owns_candidates

                    if owns_candidates(self, request, context, selection):
                        first = self._recover_raw_pick_candidates(
                            request, context, first, selection
                        )
                        if first.plan_success.all():
                            return finish(first)
                return finish(first)
        finally:
            _PICK_SELECTION.reset(token)

    def _qualify_pick_candidate(self, request, context, selection, pose, direction):
        """Use the shared builder only after qualifying this exact Pick branch."""
        from .articulation_collision import check_free_motion
        from .pick_candidate_path import trajectory
        from .pick_continuation import preview_place

        exact, receipt = trajectory(self, request, context, pose, direction)
        record = {"trajectory": receipt}
        if exact is None:
            return None, record
        prior = selection.request, selection.trajectory, selection.direction
        selection.request, selection.trajectory, selection.direction = (
            request,
            exact,
            direction,
        )
        try:
            # Retain PickUp's effects, phase guards and original command timing.
            candidate = self._qualify_closing_plan(
                request, context, super()._plan(request, context)
            )
            record["emitted_plan"] = _inspect_pick_plan(
                self.robot, request, context, candidate
            )
            if not candidate.plan_success.all() or not record["emitted_plan"]["valid"]:
                return None, record
            arm = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            stop = next(
                segment.stop
                for segment in candidate.segments
                if segment.name == "approach"
            )
            record["approach_scene"] = check_free_motion(
                self.robot,
                self.path_simulation,
                candidate.joint_trajectory.positions[0, :stop],
                arm.control_part,
            )
            if not record["approach_scene"]["valid"]:
                return None, record
            preview = (self.continuation_previews or {}).get(request.invocation_id)
            if preview is not None:
                record["place_preview"] = preview_place(
                    self, request, context, candidate, preview, safety=True
                )
                if not record["place_preview"]["valid"]:
                    return None, record
            return candidate, record
        finally:
            selection.request, selection.trajectory, selection.direction = prior

    def _plan_fixed_pick_candidate(self, request, context, selection):
        """Qualify one declared grasp as the main route, without resampling."""
        from embodichain.lab.sim.atomic_actions.goals import _resolve_object_pose

        state = _RandomState.capture(_cuda_devices(self.device, self.robot.device))
        record = {}
        candidate = None
        try:
            if self.path_simulation is None or selection.closing_geometry is None:
                raise ValueError(
                    "Fixed Pick requires scene and native closing screens."
                )
            object_pose = _resolve_object_pose(
                request.goal.semantics, context, name="pickup_object_pose"
            ).to(device=self.device, dtype=context.robot.qpos.dtype)
            pose = object_pose @ request.skill_options.fixed_object_to_eef.to(
                object_pose
            )
            direction = request.skill_options.approach_direction.to(pose)
            direction = (direction / torch.linalg.vector_norm(direction))[None]
            selection.object_pose = object_pose
            record["selected_pose"] = pose.detach().cpu().tolist()
            record["approach_direction"] = direction.detach().cpu().tolist()
            if self._approach_alignment_mask(
                pose[:, None, None], request.skill_options, direction[:, None, None]
            ).all():
                candidate, qualification = self._qualify_pick_candidate(
                    request, context, selection, pose, direction
                )
                record.update(qualification)
            else:
                record["reason"] = "approach_alignment_rejected"
        except (
            ImportError,
            ValueError,
            RuntimeError,
            AttributeError,
            KeyError,
            TypeError,
        ) as error:
            record["unavailable"] = str(error)
        finally:
            state.restore()
        accepted = candidate is not None
        if not accepted:
            candidate = self.failed_plan(
                request,
                context,
                message="The declared fixed Pick was not qualified.",
                failure_code="gen_sim_fixed_pick_unqualified",
                retryable=False,
            )
        return replace(
            candidate,
            diagnostics=replace(
                candidate.diagnostics,
                metadata={
                    **candidate.diagnostics.metadata,
                    "gen_sim_fixed_pick": {
                        "accepted": accepted,
                        "fixed_object_to_eef": request.skill_options.fixed_object_to_eef.detach()
                        .cpu()
                        .tolist(),
                        **record,
                        "resampled": False,
                        "physical_evidence": False,
                    },
                },
            ),
        )

    def _recover_raw_pick_candidates(self, request, context, first, selection):
        """Qualify legal geometry discarded by the coarse endpoint-IK screen."""
        from .pick_candidate_path import geometric_variants
        from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

        previous = first.diagnostics.metadata.get("gen_sim_pick_candidate_retry", {})
        budget = max(
            0, 4 - len(previous.get("attempts", ())) - selection.depth_recovery_attempts
        )
        if first.plan_success.any() or budget == 0:
            return first
        state = _RandomState.capture(_cuda_devices(self.device, self.robot.device))
        prior = selection.request, selection.trajectory, selection.direction
        attempts = []
        chosen = first
        seen = [
            torch.as_tensor(item["selected_pose"], device=self.device)
            for item in previous.get("attempts", ())
            if item.get("selected_pose") is not None
        ]
        try:
            routes = []
            arm = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            start = self.robot.compute_fk(
                qpos=context.robot.qpos[:, list(arm.joint_ids)],
                name=arm.control_part,
                to_matrix=True,
            )
            for strategy, raw, direction in selection.raw_candidate_routes:
                pool = geometric_variants(
                    self, raw, request, context, direction, qualify=False
                )
                rotation = quat_from_matrix(pool.poses[0, :, :3, :3])
                rotation_error = quat_error_magnitude(
                    rotation, quat_from_matrix(start[0, :3, :3]).expand_as(rotation)
                )
                # Preserve shared Pick's nearest-roll preference within each
                # geometric cost, without discarding the other legal branch.
                preferred = torch.argsort(rotation_error, stable=True)
                order = torch.argsort(
                    pool.costs[0, preferred].masked_fill(
                        ~pool.valid[0, preferred], torch.inf
                    ),
                    stable=True,
                )
                order = preferred[order]
                routes.append(
                    (
                        strategy,
                        pool,
                        direction,
                        iter(order[pool.valid[0, order]].tolist()),
                    )
                )
            # Share the same bounded budget across declared approach strategies.
            while routes and len(attempts) < budget:
                remaining = []
                for strategy, pool, direction, indices in routes:
                    if len(attempts) >= budget:
                        break
                    index = next(indices, None)
                    while index is not None:
                        pose = pool.poses[:, index]
                        duplicate = any(torch.equal(pose, value) for value in seen)
                        if not duplicate and bool(
                            self._approach_alignment_mask(
                                pose[:, None, None],
                                request.skill_options,
                                direction[:, None, None],
                            ).all()
                        ):
                            break
                        index = next(indices, None)
                    if index is None:
                        continue
                    remaining.append((strategy, pool, direction, indices))
                    if len(attempts) >= budget:
                        break
                    pose = pool.poses[:, index].clone()
                    seen.append(pose)
                    state.restore()
                    selected = replace(
                        request,
                        goal=replace(request.goal, grasp_xpos=pose),
                        skill_options=replace(
                            request.skill_options,
                            approach_direction=direction[0],
                            rotate_upright=None,
                            obj_upright_direction=None,
                        ),
                    )
                    candidate, qualification = self._qualify_pick_candidate(
                        selected, context, selection, pose, direction
                    )
                    record = {
                        "strategy": strategy,
                        "candidate_id": index // 2,
                        "roll_variant": index % 2,
                        "selected_pose": pose.detach().cpu().tolist(),
                        **qualification,
                    }
                    attempts.append(record)
                    if candidate is None:
                        continue
                    chosen = replace(
                        candidate,
                        diagnostics=replace(
                            candidate.diagnostics,
                            metadata={
                                **candidate.diagnostics.metadata,
                                "affordance_sample": {
                                    "grasp": {
                                        "candidate_ids": [index // 2],
                                        "selected_poses": pose.detach().cpu().tolist(),
                                        "adaptive_grasp": {
                                            "strategy_indices": [strategy],
                                            "roll_variants": [index % 2],
                                            "approach_directions": direction.tolist(),
                                        },
                                    }
                                },
                            },
                        ),
                    )
                    break
                if chosen.plan_success.all():
                    break
                routes = remaining
        except (
            ImportError,
            ValueError,
            RuntimeError,
            AttributeError,
            KeyError,
            TypeError,
        ) as error:
            attempts.append({"unavailable": str(error)})
        finally:
            selection.request, selection.trajectory, selection.direction = prior
            state.restore()
        return replace(
            chosen,
            diagnostics=replace(
                chosen.diagnostics,
                metadata={
                    **chosen.diagnostics.metadata,
                    "gen_sim_pick_raw_candidate_retry": {
                        "budget": budget,
                        "attempts": attempts,
                        "accepted": bool(chosen.plan_success.all()),
                        "source": "geometry_before_endpoint_ik",
                        "qualification_scope": "continuous_pick_full_robot_limits_and_approach_scene_discrete_meshes",
                    },
                },
            ),
        )

    def _retry_ordinary_pick_candidates(
        self, request: Any, context: Any, first: Any, selection: _PickSelection
    ) -> Any:
        """Replan at most four already-filtered grasps with their own approach."""
        budget = max(0, 4 - selection.depth_recovery_attempts)
        if first.plan_success.any() or not selection.candidate_routes or not budget:
            return first
        from .pick_path import rebuild
        from .pick_continuation import preview_place

        key = request.binding.endpoint("primary", "motion").task_state_key
        original = first.expected_effects.held_object_updates.get(key)
        seen = [] if original is None else [original.grasp_xpos]
        sampled = first.diagnostics.metadata.get("affordance_sample", {}).get(
            "grasp", {}
        )
        if sampled.get("selected_poses") is not None:
            seen.append(torch.as_tensor(sampled["selected_poses"], device=self.device))
        preview = (self.continuation_previews or {}).get(request.invocation_id)
        state = _RandomState.capture(_cuda_devices(self.device, self.robot.device))
        prior = (selection.request, selection.trajectory, selection.direction)
        attempts: list[dict[str, Any]] = []
        chosen = first

        def candidates():
            deferred = []
            for strategy, pool, direction in selection.candidate_routes:
                order = torch.argsort(
                    pool.costs[0].masked_fill(~pool.valid[0], torch.inf), stable=True
                )
                for index in order[pool.valid[0, order]].tolist():
                    pose = pool.poses[:, index]
                    if any(torch.equal(pose, value) for value in seen):
                        continue
                    # Spend the bounded retry budget on different wrist branches
                    # before millimetre perturbations of the same grasp.
                    similar = any(
                        bool(
                            torch.linalg.vector_norm(
                                pose[:, :3, 3] - value[:, :3, 3], dim=-1
                            ).amax()
                            < 0.01
                        )
                        and bool(
                            (
                                (pose[:, :3, :3].transpose(-1, -2) @ value[:, :3, :3])
                                .diagonal(dim1=-2, dim2=-1)
                                .sum(-1)
                                .amin()
                                - 1
                            )
                            / 2
                            > 0.9396926208
                        )
                        for value in seen
                    )
                    if similar:
                        deferred.append(
                            (strategy, index, pose.clone(), direction.clone())
                        )
                        continue
                    seen.append(pose)
                    yield strategy, index, pose.clone(), direction.clone()
            yield from deferred

        try:
            for strategy, index, pose, direction in candidates():
                if len(attempts) >= budget:
                    break
                state.restore()
                selected = replace(
                    request,
                    goal=replace(request.goal, grasp_xpos=pose),
                    skill_options=replace(
                        request.skill_options, approach_direction=direction[0]
                    ),
                )
                selection.request, selection.trajectory, selection.direction = (
                    selected,
                    None,
                    direction,
                )
                candidate = self._qualify_closing_plan(
                    selected, context, super()._plan(selected, context)
                )
                candidate = rebuild(self, selected, context, candidate, selection)
                candidate = self._qualify_closing_plan(selected, context, candidate)
                record = {
                    "strategy": strategy,
                    "candidate_id": index,
                    "selected_pose": pose.detach().cpu().tolist(),
                    "approach_direction": direction.detach().cpu().tolist(),
                    "path_success": candidate.plan_success.tolist(),
                    "failure_code": (
                        None
                        if candidate.diagnostics.failure is None
                        else candidate.diagnostics.failure.code
                    ),
                }
                attempts.append(record)
                if not candidate.plan_success.all():
                    continue
                if preview is not None:
                    record["place_preview"] = preview_place(
                        self, selected, context, candidate, preview, safety=True
                    )
                    if not record["place_preview"]["valid"]:
                        continue
                samples = dict(
                    candidate.diagnostics.metadata.get("affordance_sample", {})
                )
                samples["grasp"] = {
                    "candidate_ids": [index],
                    "selected_poses": pose.detach().cpu().tolist(),
                    "adaptive_grasp": {
                        "strategy_indices": [strategy],
                        "approach_directions": direction.tolist(),
                    },
                }
                chosen = replace(
                    candidate,
                    diagnostics=replace(
                        candidate.diagnostics,
                        metadata={
                            **candidate.diagnostics.metadata,
                            "affordance_sample": samples,
                        },
                    ),
                )
                break
        finally:
            selection.request, selection.trajectory, selection.direction = prior
            state.restore()
        return replace(
            chosen,
            diagnostics=replace(
                chosen.diagnostics,
                metadata={
                    **chosen.diagnostics.metadata,
                    "gen_sim_pick_candidate_retry": {
                        "budget": 4,
                        "attempts": attempts,
                        "accepted": bool(chosen.plan_success.all()),
                    },
                },
            ),
        )

    def _recover_constrained_pick_roll(self, request, context, first):
        """Keep failure-only roll recovery; reject robot-unsafe successful Pick paths."""
        goal, options = request.goal, request.skill_options
        affordance = goal.semantics.affordance
        if not _owns_constrained_pick(request, context, self.motion_generator):
            return first
        endpoint = request.binding.endpoint("primary", "motion")
        held = first.expected_effects.held_object_updates.get(endpoint.task_state_key)
        if held is None or held.semantics.entity_id != goal.semantics.entity_id:
            return (
                replace(
                    first,
                    plan_success=torch.zeros_like(first.plan_success),
                    diagnostics=replace(
                        first.diagnostics,
                        failure=PlanningFailure(
                            "pick_attachment_unavailable", retryable=False
                        ),
                    ),
                )
                if first.plan_success.any()
                else first
            )
        from .grasp_filter import TaskGraspPoseGenerator

        calibration = options.grasp_frame_to_eef.to(held.grasp_xpos)
        original_grasp = held.grasp_xpos @ pose_inv(calibration)
        grasp_pose = original_grasp.clone()
        grasp_pose[..., :3, :2] *= -1
        choices = [(grasp_pose, None, "grasp_z_roll_180")]
        object_pose = held.grasp_xpos @ pose_inv(held.object_to_eef)
        hand = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        original_success = bool(first.plan_success.all())
        attempts = []

        def finish(
            plan, reason, alternate=None, candidate_id=None, variant="grasp_z_roll_180"
        ):
            accepted = reason == "accepted"
            metadata = dict(plan.diagnostics.metadata)
            if accepted and alternate is not None:
                samples = dict(first.diagnostics.metadata.get("affordance_sample", {}))
                samples["grasp"] = {
                    **samples.get("grasp", {}),
                    "selected_poses": alternate.detach().cpu().tolist(),
                }
                if candidate_id is not None:
                    samples["grasp"]["candidate_ids"] = [candidate_id]
                metadata["affordance_sample"] = samples
            details = {
                "accepted": accepted,
                "reason": reason,
                "variant": variant,
                "original_plan_success": first.plan_success.tolist(),
            }
            metadata[
                (
                    "gen_sim_pick_safety"
                    if original_success
                    else "gen_sim_pick_roll_recovery"
                )
            ] = {**details, "attempts": attempts}
            return replace(
                plan,
                plan_success=(
                    plan.plan_success
                    if accepted
                    else torch.zeros_like(plan.plan_success)
                ),
                diagnostics=replace(
                    plan.diagnostics,
                    metadata=metadata,
                    **(
                        {
                            "failure": PlanningFailure(
                                "pick_robot_safety_failed", retryable=False
                            )
                        }
                        if original_success and not accepted
                        else {}
                    ),
                ),
            )

        arm = endpoint.require_target(JointPositionTarget)
        start = _inspect_pick_motion(self.robot, context.robot.qpos, arm.control_part)
        if not start["valid"]:
            attempts.append({"variant": "observed_start", **start})
            return finish(first, "start_collision_or_unavailable")
        if original_success:
            certificate = _inspect_pick_plan(self.robot, request, context, first)
            if certificate["valid"]:
                return first
            attempts.append({"variant": "original", **certificate})
            if "unavailable" in certificate:
                return finish(first, "certificate_unavailable")
            if certificate.get("reason") == "initial_velocity_rejected":
                from .pick_safety import _prepend_pick_opening

                opened, opening_report = _prepend_pick_opening(
                    self, request, context, first
                )
                attempts.append({"variant": "observed_hand_opening", **opening_report})
                if opened is not None:
                    return finish(opened, "accepted", variant="observed_hand_opening")
            pool = _PICK_SELECTION.get().candidates
            if pool is not None:
                indices = torch.argsort(
                    pool.costs[0].masked_fill(~pool.valid[0], torch.inf), stable=True
                )
                for index in indices[pool.valid[0, indices]][:4].tolist():
                    pose = pool.poses[:, index] @ pose_inv(calibration)
                    mirror = pose.clone()
                    mirror[..., :3, :2] *= -1
                    choices.extend(
                        (candidate, index, f"candidate_{index}_{roll}")
                        for candidate, roll in (
                            (pose, "original"),
                            (mirror, "z_roll_180"),
                        )
                        if not any(
                            torch.equal(candidate, previous)
                            for previous in [
                                original_grasp,
                                *(choice[0] for choice in choices),
                            ]
                        )
                    )
        # Every alternate is a complete shared PickUp plan; keep its proposed
        # attachment and emitted commands together and restore all extra draws.
        state = _RandomState.capture(_cuda_devices(self.device, self.robot.device))
        try:
            generator = self.planning_services.grasp_pose_generator(hand.target_id)
            if not isinstance(generator, TaskGraspPoseGenerator):
                raise ValueError(
                    "Constrained Pick roll recovery requires its grasp filter."
                )
            generator = generator.for_pick(
                *affordance.get_custom_config(_PICK_GRASP_RULE),
                affordance.mesh_vertices,
                affordance.mesh_triangles,
            )
            for grasp_pose, candidate_id, variant in choices:
                alternate = grasp_pose @ calibration
                reason = "grasp_filter_rejected"
                if generator.accepts_grasp_poses(
                    grasp_pose,
                    object_pose,
                    affordance.mesh_vertices,
                    affordance.mesh_triangles,
                ).all():
                    direction = options.approach_direction.to(alternate)
                    direction = direction / torch.linalg.vector_norm(direction)
                    reason = "approach_alignment_rejected"
                    if self._approach_alignment_mask(
                        alternate[:, None, None], options, direction
                    ).all():
                        selected = replace(
                            request, goal=replace(goal, grasp_xpos=alternate)
                        )
                        second = self._qualify_closing_plan(
                            selected, context, super()._plan(selected, context)
                        )
                        certificate = (
                            _inspect_pick_plan(self.robot, request, context, second)
                            if second.plan_success.all()
                            else {"valid": False, "reason": "planning_failed"}
                        )
                        attempts.append({"variant": variant, **certificate})
                        if certificate["valid"]:
                            return finish(
                                second, "accepted", alternate, candidate_id, variant
                            )
                        reason = certificate.get("reason", "robot_collision_rejected")
                        if "unavailable" in certificate:
                            return finish(first, "certificate_unavailable")
                if not original_success:
                    return finish(first, reason)
            return finish(first, "safe_candidates_exhausted")
        finally:
            state.restore()

    def _get_full_pickup_trajectory(
        self,
        grasp_xpos,
        start_arm_qpos,
        last_qpos,
        motion_policy,
        options,
        approach_direction,
        manipulator,
        end_effector,
        hand_open_qpos,
        hand_grasp_qpos,
        interpolation_dt,
    ):
        selection = _PICK_SELECTION.get()
        if selection is not None and selection.trajectory is not None:
            return selection.trajectory
        if selection is not None and selection.direction is not None:
            approach_direction = selection.direction
        return super()._get_full_pickup_trajectory(
            grasp_xpos,
            start_arm_qpos,
            last_qpos,
            motion_policy,
            options,
            approach_direction,
            manipulator,
            end_effector,
            hand_open_qpos,
            hand_grasp_qpos,
            interpolation_dt,
        )

    def _resolve_grasp_pose(
        self,
        semantics,
        object_pose,
        start_qpos,
        manipulator,
        grasp_target_id,
        options,
        approach_direction,
        context,
        *,
        sample_key,
    ):
        affordance = semantics.affordance
        mode = affordance.get_custom_config(ADAPTIVE_GRASP)
        rule_selector = affordance.get_custom_config(_PICK_GRASP_RULE)
        selection = _PICK_SELECTION.get()
        if (
            mode in {"ordinary", "pour", "ends"}
            and selection is not None
            and rule_selector is None
            and options.pick_object_part == "center"
            and options.rotate_upright is None
            and options.obj_upright_direction is None
        ):
            axis = observed_axis(affordance, object_pose)
            tcp = self.robot.compute_fk(
                qpos=start_qpos, name=manipulator.control_part, to_matrix=True
            )
            generator = self.planning_services.grasp_pose_generator(grasp_target_id)
            if (
                selection.closing_geometry is not None
                and selection.closing_geometry.revision in (5, 6, 7, 8)
            ):
                from .grasp_filter import TaskGraspPoseGenerator

                if isinstance(generator, TaskGraspPoseGenerator):
                    generator = generator.for_native_closing(selection.closing_geometry)
            choices = grasp_strategies(
                axis,
                object_pose,
                tcp,
                ends=mode == "ends",
                purpose="handover_source" if mode == "ends" else mode,
            )
            return self._select_adaptive_pick(
                affordance,
                generator,
                object_pose,
                start_qpos,
                manipulator,
                options,
                context,
                sample_key,
                choices,
                selection,
                **({"axis_region_policy": "supported_end"} if mode == "ends" else {}),
            )
        if rule_selector is None:
            return super()._resolve_grasp_pose(
                semantics,
                object_pose,
                start_qpos,
                manipulator,
                grasp_target_id,
                options,
                approach_direction,
                context,
                sample_key=sample_key,
            )
        if not isinstance(affordance, AntipodalAffordance):
            raise ValueError("GenSim Pick requires AntipodalAffordance.")
        generator = self.planning_services.grasp_pose_generator(grasp_target_id)
        if rule_selector is not None:
            from .grasp_filter import TaskGraspPoseGenerator

            if (
                not isinstance(generator, TaskGraspPoseGenerator)
                or options.pick_object_part != "center"
                or len(rule_selector) != 2
                or rule_selector[0] != semantics.entity_id
            ):
                raise ValueError(
                    "Constrained Pick requires its call-local grasp filter."
                )
            generator = generator.for_pick(
                *rule_selector, affordance.mesh_vertices, affordance.mesh_triangles
            )
            if selection is not None:
                selection.constrained = True
        candidates = affordance.get_grasp_candidates(
            generator,
            object_pose,
            approach_direction,
            obj_longest_axis=None,
            is_positive_part=True,
        )
        if selection is not None:
            selection.raw_candidate_routes.append(
                (0, candidates, approach_direction.reshape(1, 3).clone())
            )
        poses, ik_success = self._select_feasible_grasp_variants(
            candidates.poses,
            start_qpos,
            object_pose,
            manipulator,
            options,
            approach_direction,
        )
        feasible = AffordancePoseCandidates(
            poses=poses, costs=candidates.costs, valid=candidates.valid & ik_success
        )
        if selection is not None:
            selection.candidates = feasible
        return affordance.sample_candidates(
            feasible,
            sampling=context.affordance_sampling,
            env_ids=context.env_ids,
            key=sample_key,
            reference_poses=object_pose,
        )

    def _native_pick_endpoint_clear(
        self, request: Any, context: Any, path: torch.Tensor, lengths: dict[str, int]
    ) -> bool:
        """Reject an impossible open-hand endpoint before planning more prefixes."""
        from .articulation_collision import check_free_motion

        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        stop = lengths["approach"]
        opening = check_free_motion(
            self.robot, self.path_simulation, path[0, stop - 1 : stop], arm.control_part
        )
        opened = opening["valid"]
        selection = _PICK_SELECTION.get()
        screen = None if selection is None else selection.closing_geometry
        if screen is not None and screen.revision in (7, 8):
            selection.closing_scene_checks.append({"stage": "open_endpoint", **opening})
        if not opened or screen is None or screen.revision not in (7, 8):
            return bool(opened)
        scene = self._inspect_closing_scene(
            request, context, path[0, stop : stop + lengths["close"]], screen
        )
        selection.closing_scene_checks.append({"stage": "candidate", **scene})
        return bool(scene["valid"])

    def _inspect_closing_scene(
        self, request: Any, context: Any, positions: torch.Tensor, screen: Any
    ) -> dict[str, Any]:
        from .pick_closing_scene import check_closing_scene
        from embodichain.lab.sim.atomic_actions.goals import (
            _resolve_object_pose,
            resolve_pose_goal,
        )
        from embodichain.lab.sim.atomic_actions.trajectory_ops import (
            resolve_pose_target,
        )

        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Pick closing scene revision 7 requires native B1/env0.")
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        tcp = self.robot.compute_batch_fk(
            qpos=positions[None, :1, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )[0, 0]
        goal = request.goal
        object_pose = (
            _resolve_object_pose(goal.semantics, context, name="pickup_object_pose")
            if goal.object_pose is None
            else resolve_pose_target(
                resolve_pose_goal(goal.object_pose, context, name="object_pose"),
                num_envs=context.batch_size,
                device=positions.device,
            )
        )
        contact = screen.inspect(
            (pose_inv(object_pose)[0] @ tcp).detach().cpu().double().numpy()
        )
        if not contact["valid"]:
            return {"valid": False, "source_closing": contact}
        result = check_closing_scene(
            self.robot,
            self.path_simulation,
            positions,
            arm.control_part,
            arm.joint_ids,
            screen.identity["simulation_uid"],
            contact_fraction=contact["nominal_closing_fraction"],
            grid_samples=screen.steps,
        )
        if result["valid"] and screen.revision == 8:
            from .pick_native_scene import build_native_rigid_scene

            if screen.rigid_scene_screen is None:
                screen.rigid_scene_screen = build_native_rigid_scene(
                    self.path_simulation, screen
                )
            certificate = screen.rigid_scene_screen.inspect(
                tcp.detach().cpu().double().numpy(), contact["checked_samples"]
            )
            result = {
                **result,
                "valid": bool(certificate["valid"]),
                "native_rigid_pair_rest": certificate,
            }
        return {
            **result,
            "nominal_closing_fraction": contact["nominal_closing_fraction"],
            "nominal_grid_samples": screen.steps,
            "post_contact_commands_screened": False,
        }

    def _select_adaptive_pick(
        self,
        affordance,
        generator,
        object_pose,
        start_qpos,
        manipulator,
        options,
        context,
        sample_key,
        choices,
        selection,
        *,
        axis_region_policy: Literal["mesh_extent", "supported_end"] = "mesh_extent",
    ):
        request = selection.request
        grasp = request.binding.endpoint("primary", "grasp")
        hand = grasp.require_target(JointPositionTarget)
        commands = [
            grasp.joint_positions(
                name,
                num_envs=context.batch_size,
                device=self.device,
                dtype=context.robot.qpos.dtype,
            )
            for name in ("open", "grasp")
        ]
        success = torch.zeros(context.batch_size, dtype=torch.bool, device=self.device)
        poses = object_pose.clone()
        directions = torch.zeros_like(object_pose[:, :3, 3])
        strategies = torch.full_like(success, -1, dtype=torch.long)
        candidate_ids = torch.full_like(strategies, -1)
        full, lengths = None, None
        attempts = []
        native_source = self._native_handover_source(
            request, selection.closing_geometry
        )
        depth_enabled = (
            selection.closing_geometry is not None
            and selection.closing_geometry.revision in (6, 7, 8)
            and context.batch_size == 1
            and not selection.constrained
            and (axis_region_policy == "mesh_extent" or native_source)
            and (
                request.invocation_id not in self.path_excluded_invocations
                or native_source
            )
        )
        depths = selection.closing_geometry.depth_offsets() if depth_enabled else (0.0,)
        alternatives = ((choice, depth) for depth in depths for choice in choices)
        original_pools: dict[int, AffordancePoseCandidates] = {}
        original_rng: dict[int, _RandomState] = {}
        selected_depth = object_pose.new_zeros(context.batch_size)
        for index, (choice, depth) in enumerate(alternatives):
            choice_index = index % len(choices)
            if depth and selection.depth_recovery_attempts >= 4:
                break
            effective = options
            canonical_shift = None
            pool_costs = None
            if depth:
                transform = options.grasp_frame_to_eef.clone()
                transform[:3, 3] -= depth * transform[:3, 2]
                effective = replace(options, grasp_frame_to_eef=transform)
                canonical_shift = transform @ pose_inv(options.grasp_frame_to_eef)

            def observe_pool(pool: AffordancePoseCandidates) -> None:
                selection.candidates = pool
                if (self.pick_purposes or {}).get(request.invocation_id) == "ordinary":
                    selection.candidate_routes.append(
                        (index, pool, choice.direction.clone())
                    )

            def observe_raw_pool(pool: AffordancePoseCandidates) -> None:
                nonlocal pool_costs
                pool_costs = pool.costs
                if depth_enabled and not depth:
                    original_pools[choice_index] = pool
                    original_rng[choice_index] = _RandomState.capture(
                        _cuda_devices(object_pose.device)
                    )
                if canonical_shift is not None:
                    pool = replace(
                        pool, poses=pool.poses @ canonical_shift.to(pool.poses)
                    )
                selection.raw_candidate_routes.append(
                    (index, pool, choice.direction.clone())
                )

            def feasible(values, approach):
                limit = (
                    getattr(generator, "native_proposal_limit", None)
                    if choice.axis is None
                    else None
                )
                return self._select_feasible_grasp_variants(
                    values,
                    start_qpos,
                    object_pose,
                    manipulator,
                    effective,
                    approach[:, None, None, :],
                    **(
                        {
                            "candidate_costs": pool_costs,
                            "candidate_limit": limit,
                            "candidate_route": index,
                            "first_feasible": (
                                (self.pick_purposes or {}).get(request.invocation_id)
                                == "ordinary"
                                and not selection.constrained
                                and request.invocation_id
                                not in (self.continuation_previews or {})
                                and not getattr(
                                    getattr(context, "affordance_sampling", None),
                                    "enabled",
                                    False,
                                )
                            ),
                        }
                        if limit is not None
                        else {}
                    ),
                )

            with _pick_depth_random_scope(
                original_rng[choice_index] if depth else None
            ):
                sample, direction = select_grasp(
                    affordance,
                    generator,
                    object_pose,
                    (choice,),
                    context,
                    f"{sample_key}:strategy_{choice_index}",
                    feasible=feasible,
                    candidate_observer=observe_pool,
                    raw_candidate_observer=observe_raw_pool,
                    **(
                        {"candidate_pool": original_pools[choice_index]}
                        if depth
                        else {}
                    ),
                    **(
                        {"axis_region_policy": axis_region_policy}
                        if axis_region_policy != "mesh_extent"
                        else {}
                    ),
                )
                path_rng = (
                    _RandomState.capture(_cuda_devices(object_pose.device))
                    if depth
                    else None
                )
            viable = sample.success & ~success
            attempt = {"strategy": index, "grasp_feasible": sample.success.tolist()}
            if depth_enabled:
                attempt["retreat_m"] = depth
                selection.depth_trace.append(attempt)
            attempts.append(attempt)
            if not viable.any():
                continue
            if depth:
                selection.depth_recovery_attempts += 1
            # Screen the exact trajectory once, then return that same path to
            # PickUp's effect builder rather than planning it again from new seeds.
            with _pick_depth_random_scope(path_rng):
                (
                    path_ok,
                    candidate_path,
                    candidate_lengths,
                ) = super()._get_full_pickup_trajectory(
                    sample.poses,
                    start_qpos,
                    context.last_qpos,
                    request.motion_policy,
                    effective,
                    direction,
                    manipulator,
                    hand,
                    *commands,
                    context.require_control_dt(),
                )
            accepted = viable & path_ok
            attempt["path_feasible"] = path_ok.tolist()
            if (
                depth_enabled
                and bool(accepted.any())
                and (
                    (self.pick_purposes or {}).get(request.invocation_id) == "ordinary"
                    or native_source
                )
            ):
                clear = self._native_pick_endpoint_clear(
                    request, context, candidate_path, candidate_lengths
                )
                key = (
                    "open_and_precontact_scene_valid"
                    if selection.closing_geometry.revision in (7, 8)
                    else "open_endpoint_scene_valid"
                )
                attempt[key] = bool(clear)
                accepted &= bool(clear)
            if not accepted.any():
                self._expand_native_pick_candidates(selection)
            if full is None:
                full = context.last_qpos[:, None].expand_as(candidate_path).clone()
                lengths = candidate_lengths
            if candidate_path.shape != full.shape or candidate_lengths != lengths:
                raise ValueError("Adaptive Pick strategies must preserve phase timing.")
            full = torch.where(accepted[:, None, None], candidate_path, full)
            poses = torch.where(accepted[:, None, None], sample.poses, poses)
            directions = torch.where(accepted[:, None], direction, directions)
            strategies = torch.where(accepted, index, strategies)
            candidate_ids = torch.where(
                accepted,
                candidate_ids.new_tensor(sample.metadata["candidate_ids"]),
                candidate_ids,
            )
            success |= accepted
            selected_depth = torch.where(
                accepted,
                selected_depth.new_full(selected_depth.shape, depth),
                selected_depth,
            )
            if success.all():
                break
        if full is not None:
            selection.trajectory = (success, full, lengths)
        selection.direction = directions
        return AffordanceSample(
            success,
            poses,
            {
                "key": sample_key,
                "candidate_ids": candidate_ids.tolist(),
                "selected_poses": poses.detach().cpu().tolist(),
                "adaptive_grasp": {
                    **(
                        {"selected_retreat_m": selected_depth.tolist()}
                        if depth_enabled
                        else {}
                    ),
                    **(
                        {"axis_region_policy": axis_region_policy}
                        if axis_region_policy != "mesh_extent"
                        else {}
                    ),
                    "strategy_indices": strategies.tolist(),
                    "approach_directions": directions.tolist(),
                    "attempts": attempts,
                },
            },
        )


class GenSimHandOver(HandOver):
    """Choose receiving geometry locally; retain shared transfer and effects."""

    skill_id = HandOver.skill_id
    GoalType = HandOver.GoalType
    OptionsType = HandOver.OptionsType
    binding_contract = HandOver.binding_contract
    environment_screen_factory: Any = None
    post_place_exits: dict[str, tuple[str, str, str]] | None = None

    def _owns_post_place_exit(self) -> bool:
        selection = _RECEIVE_SELECTION.get()
        if (
            not self.post_place_exits
            or selection is None
            or selection.alternate.numel() != 1
        ):
            return False
        held, resources = selection.held, selection.resources
        route = (self.post_place_exits or {}).get(selection.request.invocation_id)
        if route is None:
            return False
        return bool(
            route
            == (
                held.semantics.entity_id,
                resources.first.task_state_key,
                resources.second.task_state_key,
            )
            and held.semantics.affordance.get_custom_config(ADAPTIVE_GRASP) == "ends"
        )

    def _source_retreat_waypoints(self, *args, **kwargs):
        result = super()._source_retreat_waypoints(*args, **kwargs)
        if not self._owns_post_place_exit():
            return result
        selection = _RECEIVE_SELECTION.get()
        if selection.root_poses is None or result.shape != (1, 5, 4, 4):
            raise ValueError("Post-handover withdrawal requires exact bound arm roots.")
        source, receiver = selection.root_poses
        direction = (source[:, :3, 3] - receiver[:, :3, 3]).clone()
        direction[:, 2] = 0
        length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        if not torch.isfinite(direction).all() or (length <= 1e-6).any():
            raise ValueError(
                "Post-handover withdrawal has no horizontal arm separation."
            )
        result[:, -2:, :3, 3] += (
            direction[:, None]
            / length[:, None]
            * result.new_tensor([0.5, 1.0])[None, :, None]
            * kwargs["retreat_distance"]
        )
        return result

    def _resolve_existing_hold_receive(
        self,
        request,
        context,
        resources,
        exchange_pose,
        approach_direction,
        *,
        receive_axis,
        receive_positive,
        receive_center_axis,
        source_root,
        destination_root,
    ):
        selection = _RECEIVE_SELECTION.get()
        if selection is not None:
            selection.exchange_pose = exchange_pose
            selection.root_poses = (source_root, destination_root)
        original, sample = super()._resolve_existing_hold_receive(
            request,
            context,
            resources,
            exchange_pose,
            approach_direction,
            receive_axis=receive_axis,
            receive_positive=receive_positive,
            receive_center_axis=receive_center_axis,
            source_root=source_root,
            destination_root=destination_root,
        )
        return (
            selection.exchange_pose if selection is not None else original,
            sample,
        )

    def _plan_existing_hold(self, request, context, resources, held):
        mode = held.semantics.affordance.get_custom_config(ADAPTIVE_GRASP)
        if mode not in {"ordinary", "pour", "ends"}:
            return super()._plan_existing_hold(request, context, resources, held)
        screen = None
        if self.environment_screen_factory is not None:
            try:
                screen = self.environment_screen_factory(request, context, resources)
            except (
                AttributeError,
                KeyError,
                TypeError,
                ValueError,
                RuntimeError,
            ) as error:
                return self.failed_plan(
                    request,
                    context,
                    message=f"HandOver receiver environment geometry unavailable: {error}",
                )
        selection = _ReceiveSelection(
            resources,
            held,
            torch.zeros(context.batch_size, dtype=torch.bool, device=self.device),
            request,
            screen,
        )
        token = _RECEIVE_SELECTION.set(selection)
        try:
            first = super()._plan_existing_hold(request, context, resources, held)
            first = self._qualify_receiver_environment(first, screen)
            if first.plan_success.all() or not selection.sampled:
                return first
            selection.alternate = ~first.plan_success
            # Retry planning only. Never dispatch commands, alter the source
            # attachment, or consume randomness belonging to later calls.
            devices = (
                [context.robot.qpos.device.index] if context.robot.qpos.is_cuda else []
            )
            with torch.random.fork_rng(devices=devices):
                second = super()._plan_existing_hold(request, context, resources, held)
                second = self._qualify_receiver_environment(second, screen)
            metadata = {
                "receive_direction_attempts": [
                    {
                        "plan_success": plan.plan_success.tolist(),
                        "metadata": dict(plan.diagnostics.metadata),
                    }
                    for plan in (first, second)
                ],
                "selected_receive_attempt": torch.where(
                    first.plan_success, 0, torch.where(second.plan_success, 1, -1)
                ).tolist(),
            }
            if not (second.plan_success & ~first.plan_success).any():
                return replace(
                    first,
                    diagnostics=replace(
                        first.diagnostics,
                        metadata={**first.diagnostics.metadata, **metadata},
                    ),
                )
            merged = self._merge_receive_plans(
                request, context, first, second, metadata
            )
            return self._qualify_receiver_environment(merged, screen)
        finally:
            _RECEIVE_SELECTION.reset(token)

    def _qualify_receiver_environment(
        self, plan: ActionPlan, screen: Any
    ) -> ActionPlan:
        exit_check = self._owns_post_place_exit()
        if not plan.plan_success.any() or (screen is None and not exit_check):
            return plan
        try:
            evidence = screen.inspect(plan) if screen is not None else {"valid": True}
            if exit_check:
                selection = _RECEIVE_SELECTION.get()
                exit_evidence = _inspect_pick_motion(
                    self.robot,
                    plan.joint_trajectory.positions[0, -1:],
                    selection.resources.second.arm.control_part,
                )
                evidence = {
                    **evidence,
                    "post_handover_exit": exit_evidence,
                    "valid": evidence.get("valid") is True
                    and exit_evidence.get("valid") is True,
                }
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
            evidence = {"valid": False, "unavailable": str(error)}
        valid = evidence.get("valid") is True
        return replace(
            plan,
            plan_success=(
                plan.plan_success if valid else torch.zeros_like(plan.plan_success)
            ),
            diagnostics=replace(
                plan.diagnostics,
                metadata={
                    **plan.diagnostics.metadata,
                    "gen_sim_handover_environment": evidence,
                },
                failure=(
                    plan.diagnostics.failure
                    if valid
                    else PlanningFailure(
                        "handover_receiver_environment_unqualified", retryable=False
                    )
                ),
            ),
        )

    def _merge_receive_plans(self, request, context, first, second, metadata):
        """Retain successful rows and shared HandOver's planned effect contracts."""
        keep = first.plan_success
        if first.segments != second.segments:
            raise ValueError("HandOver direction retries must preserve phase timing.")
        trajectory = replace(
            second.joint_trajectory,
            positions=torch.where(
                keep[:, None, None],
                first.joint_trajectory.positions,
                second.joint_trajectory.positions,
            ),
            dt=torch.where(
                keep[:, None], first.joint_trajectory.dt, second.joint_trajectory.dt
            ),
        )

        def effects(name):
            previous, retry = getattr(first, name), getattr(second, name)
            hypothetical = TaskState(
                batch_size=context.batch_size, device=context.robot.qpos.device
            )
            hypothetical = previous.apply(hypothetical, keep)
            hypothetical = retry.apply(hypothetical, ~keep & second.plan_success)
            return StateDelta(
                held_object_updates={
                    key: hypothetical.get_held_object(key)
                    for key in previous.held_object_updates.keys()
                    | retry.held_object_updates.keys()
                }
            )

        return self.build_plan(
            request,
            context,
            success=keep | second.plan_success,
            trajectory=trajectory,
            expected_effects=effects("expected_effects"),
            effect_candidates=effects("effect_candidates"),
            diagnostics=PlannerDiagnostics(
                backend=first.diagnostics.backend,
                metadata=metadata,
            ),
            segment_lengths={
                segment.name: segment.stop - segment.start for segment in first.segments
            },
            scene_dependency_monitor_until=first.scene_dependency_monitor_until,
        )

    def _resolve_grasp(
        self,
        affordance,
        object_pose,
        approach_direction,
        grasp_target_id,
        *,
        obj_longest_axis,
        is_positive_part,
        center_axis=None,
        context,
        sample_key,
    ):
        selection = _RECEIVE_SELECTION.get()
        if selection is None:
            return super()._resolve_grasp(
                affordance,
                object_pose,
                approach_direction,
                grasp_target_id,
                obj_longest_axis=obj_longest_axis,
                is_positive_part=is_positive_part,
                center_axis=center_axis,
                context=context,
                sample_key=sample_key,
            )
        resources, held = selection.resources, selection.held
        axis = observed_axis(affordance, object_pose)
        source = object_pose @ held.object_to_eef.to(object_pose)
        destination = self.robot.compute_fk(
            qpos=context.robot.qpos[:, list(resources.second.arm.joint_ids)],
            name=resources.second.arm.control_part,
            to_matrix=True,
        )
        positive = opposite_grasp_end(axis, object_pose, source, destination)
        choices = grasp_strategies(
            axis,
            object_pose,
            destination,
            ends=True,
            positive=positive,
            purpose="handover_receive",
        )
        choice = replace(
            choices[0],
            direction=torch.where(
                selection.alternate[:, None], choices[1].direction, choices[0].direction
            ),
        )
        candidate_report = None
        raw_pool = None

        def observe_pool(pool):
            nonlocal raw_pool
            raw_pool = pool

        def feasible(poses, _direction):
            nonlocal candidate_report
            if raw_pool is None or raw_pool.poses is not poses:
                raise ValueError(
                    "HandOver candidate screen requires its original pool."
                )
            eligible = (
                raw_pool.valid
                & torch.isfinite(raw_pool.costs)
                & torch.isfinite(poses).all(dim=(-1, -2))
            )
            lengths = self._compute_existing_hold_segment_lengths(
                selection.request.motion_policy.sample_count,
                selection.request.skill_options,
            )
            valid, evidence = selection.screen.candidate_mask(
                poses,
                eligible,
                resources=resources,
                approach_samples=lengths["approach"],
                pre_grasp_distance=selection.request.skill_options.pre_grasp_distance,
            )
            selected_poses = poses
            if (
                not valid.any()
                and eligible.any()
                and selection.exchange_pose is not None
                and selection.root_poses is not None
            ):
                from .handover_exchange import select_exchange

                station, selected_poses, valid, station_evidence = select_exchange(
                    selection.screen,
                    self.robot,
                    selection.request,
                    context,
                    resources,
                    held,
                    selection.exchange_pose,
                    *selection.root_poses,
                    raw_pool,
                    valid,
                    lengths["approach"],
                    selection.request.skill_options.pre_grasp_distance,
                )
                selection.exchange_pose = station
                evidence = {**evidence, "exchange_selection": station_evidence}
            candidate_report = {
                **evidence,
                "raw_valid_count": raw_pool.valid.sum(-1).tolist(),
                "sampler_eligible_count": eligible.sum(-1).tolist(),
                "original_candidates": [
                    {
                        "env_row": row,
                        "pool_index": column,
                        "pose": poses[row, column].detach().cpu().tolist(),
                        "cost": float(raw_pool.costs[row, column]),
                    }
                    for row, column in eligible.nonzero(as_tuple=False).tolist()
                ],
            }
            return selected_poses, valid

        screen_candidates = (
            selection.screen is not None and context.affordance_sampling is None
        )
        sample, _ = select_grasp(
            affordance,
            self.planning_services.grasp_pose_generator(grasp_target_id),
            object_pose,
            (choice,),
            context,
            sample_key,
            feasible=feasible if screen_candidates else None,
            raw_candidate_observer=observe_pool if screen_candidates else None,
            **(
                {"axis_region_policy": "supported_end"}
                if selection.request.skill_options.receive_pick_object_part == "center"
                and affordance.get_custom_config(_PICK_GRASP_RULE) is None
                else {}
            ),
        )
        selection.sampled = True
        reference = (
            {"reference_poses": selection.exchange_pose.detach().cpu().tolist()}
            if selection.exchange_pose is not None
            else {}
        )
        return replace(
            sample,
            metadata={
                **sample.metadata,
                **reference,
                **(
                    {"gen_sim_handover_candidate_environment": candidate_report}
                    if candidate_report is not None
                    else {}
                ),
                "adaptive_grasp": {
                    **sample.metadata["adaptive_grasp"],
                    "strategy_indices": torch.where(
                        sample.success, selection.alternate.long(), -1
                    ).tolist(),
                },
            },
        )


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(1e-6)


def jaw_tilt_axis(
    jaw_line: torch.Tensor, upright: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the projected jaw axis, visible tilt direction, and validity."""
    up = _unit(upright)
    raw = jaw_line - (jaw_line * up).sum(-1, keepdim=True) * up
    valid = torch.isfinite(raw).all(-1) & (torch.linalg.vector_norm(raw, dim=-1) > 1e-5)
    axis = _unit(raw)
    direction = _unit(torch.linalg.cross(axis, up))
    return axis, direction, valid


class GenSimMoveHeldObject(MoveHeldObject):
    """GenSim transport with task-owned staging and retention evidence."""

    skill_id = MoveHeldObject.skill_id
    GoalType = MoveHeldObject.GoalType
    OptionsType = MoveHeldObject.OptionsType
    binding_contract = MoveHeldObject.binding_contract

    def __init__(
        self,
        *,
        staging_contract: StagingContract | None = None,
    ) -> None:
        super().__init__()
        self.staging_contract = staging_contract or StagingContract.empty()

    def _plan(self, request, context):
        if getattr(self.motion_generator, "direct_transport_planning", False) is True:
            return super()._plan(request, context)
        goal = request.goal
        target = goal.object_target_pose
        if isinstance(target, torch.Tensor) and target.ndim == 3:
            motion = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            handoff = GenSimStateHandoff.from_request(
                request, context, participant="GenSim transport"
            )
            task_key = handoff.resource
            held = handoff.held
            if held is not None:
                arm_qpos = arm_qpos_from_state(context, list(motion.joint_ids))
                start = self.robot.compute_fk(
                    qpos=arm_qpos, name=motion.control_part, to_matrix=True
                )
                target_eef = target @ held.object_to_eef
                lifted = start.clone()
                lifted[..., 2, 3] = (
                    torch.maximum(start[..., 2, 3], target_eef[..., 2, 3]) + 0.06
                )
                above_eef = target_eef.clone()
                above_eef[..., 2, 3] = lifted[..., 2, 3]
                above = above_eef @ pose_inv(held.object_to_eef)
                goal = replace(
                    goal,
                    object_target_pose=torch.stack(
                        (lifted @ pose_inv(held.object_to_eef), above, target), dim=1
                    ),
                )
                request = replace(request, goal=goal)
        from .held_alignment_recovery import (
            AlignmentCollisionUnavailable,
            recover,
            world_yaw_free_collision_scope,
        )
        from .staging_recovery import owns_staging, recover_staging
        from embodichain.lab.sim.atomic_actions.plans import PlanningFailure

        before_original = (
            _RandomState.capture(
                _cuda_devices(self.device, self.robot.device, context.robot.qpos.device)
            )
            if owns_staging(
                request,
                context,
                self.staging_contract.rigid_entity_ids,
                self.staging_contract.invocation_ids,
                self.motion_generator,
            )
            else None
        )

        def build(selected):
            with world_yaw_free_collision_scope(
                self.robot, selected, context
            ) as selected_scope:
                candidate = super(GenSimMoveHeldObject, self)._plan(selected, context)
                candidate = recover(
                    selected_scope,
                    selected,
                    context,
                    candidate,
                    lambda exact: super(GenSimMoveHeldObject, self)._plan(
                        exact, context
                    ),
                )
            return candidate, selected_scope

        try:
            plan, scope = build(request)
        except AlignmentCollisionUnavailable as error:
            plan = self.failed_plan(request, context, message=str(error))
            return replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    failure=PlanningFailure(
                        "held_alignment_collision_unavailable", retryable=False
                    ),
                ),
            )
        plan = recover_staging(
            plan,
            request,
            context,
            self.robot,
            scope,
            before_original,
            lambda selected: build(selected)[0],
            retention_source=self.staging_contract.retention_source(
                request.invocation_id
            ),
        )
        handoff = GenSimStateHandoff.from_request(
            request, context, participant="GenSim transport"
        )
        held = handoff.held
        if held is not None and plan.plan_success.any():
            plan = handoff.retain(plan, held)
        return plan


class GenSimPlace(Place):
    """Scope Place recovery and the approach-only scene guard to GenSim."""

    skill_id = Place.skill_id
    GoalType = Place.GoalType
    OptionsType = Place.OptionsType
    binding_contract = Place.binding_contract
    recovery_scene_vertices: dict | None = None
    ordinary_on_invocations: frozenset[str] = frozenset()
    world_yaw_free_routes: dict | None = None
    world_yaw_simulation: Any = None
    post_handover_places: dict[str, tuple[str, str, str]] | None = None

    def _translation_keyframes(
        self,
        start_xpos: torch.Tensor,
        target_xpos: torch.Tensor,
        options: PlaceOptions,
    ) -> torch.Tensor:
        from .place_branch import capture_keyframes

        return capture_keyframes(
            self, super()._translation_keyframes(start_xpos, target_xpos, options)
        )

    @staticmethod
    def _with_contact_monitor_boundary(plan: ActionPlan) -> ActionPlan:
        approach = next((s for s in plan.segments if s.name == "approach"), None)
        if approach is None or not plan.scene_dependencies:
            return plan
        # Preserve v38: expected contact motion must not replan release/retract.
        monitor_until = {
            entity_id: min(
                approach.stop,
                plan.scene_dependency_monitor_until.get(entity_id, approach.stop),
            )
            for entity_id in plan.scene_dependencies
        }
        return replace(plan, scene_dependency_monitor_until=monitor_until)

    def _plan(self, request, context):
        first = self._plan_without_yaw(request, context)
        from . import place_branch

        if (
            place_branch.owns_post_handover(self, request, context)
            and not first.plan_success.any()
        ):
            candidate = place_branch.plan(
                self,
                request,
                context,
                lambda selected, current: super(GenSimPlace, self)._plan(
                    selected, current
                ),
            )
            handoff = GenSimStateHandoff.from_request(
                request, context, participant="GenSim Place"
            )
            return require_release_effect(
                self._with_contact_monitor_boundary(candidate), handoff.resource
            )
        if not self.world_yaw_free_routes:
            return first
        from .place_yaw import recover

        return recover(self, request, context, first, self._plan_without_yaw)

    def _plan_without_yaw(self, request, context):
        handoff = GenSimStateHandoff.from_request(
            request, context, participant="GenSim Place"
        )
        if getattr(self.motion_generator, "direct_place_planning", False) is True:
            return require_release_effect(
                self._with_contact_monitor_boundary(super()._plan(request, context)),
                handoff.resource,
            )
        from .motion import (
            _joint_velocity_limits,
            _velocity_validity,
            place_ik_recovery,
        )
        from .place_recovery import eligible, recover
        from . import place_branch

        if place_branch.owns(self, request, context):
            build = lambda selected, current: super(GenSimPlace, self)._plan(
                selected, current
            )
            with place_ik_recovery(defer_retries=True):
                original = build(request, context)
            accepted = place_branch.inspect_emitted(self, request, context, original)
            plan = (
                original
                if accepted["valid"]
                else place_branch.plan(self, request, context, build)
            )
            return require_release_effect(
                self._with_contact_monitor_boundary(plan), handoff.resource
            )

        from .place_recovery import preflight

        keyframes = (
            preflight(self, request, context) if self.recovery_scene_vertices else None
        )
        if keyframes is not None and not keyframes["valid"]:
            failed = self.failed_plan(
                request,
                context,
                message="Required ordinary Place waypoints were not IK-qualified.",
                failure_code="gen_sim_place_keyframes_unqualified",
                retryable=False,
            )
            return replace(
                failed,
                diagnostics=replace(
                    failed.diagnostics,
                    metadata={
                        **failed.diagnostics.metadata,
                        "gen_sim_place_keyframe_preflight": keyframes,
                    },
                ),
            )
        early = bool(self.recovery_scene_vertices) and eligible(self, request, context)
        before = (
            _RandomState.capture(
                _cuda_devices(self.robot.device, context.robot.qpos.device)
            )
            if early
            else None
        )
        build = lambda selected, current: super(GenSimPlace, self)._plan(
            selected, current
        )
        with place_ik_recovery(defer_retries=early) as recovery:
            plan = super()._plan(request, context)
        early_fallback = None
        if early and not plan.plan_success.any():
            candidate = recover(self, request, context, plan, build)
            if candidate.plan_success.all():
                early_fallback = candidate
            acceptance = candidate.diagnostics.metadata.get(
                "gen_sim_place_waypoint_recovery", {}
            )
            # Large turns retain the complete Cartesian search before fallback.
            if (
                candidate.plan_success.all()
                and acceptance["rotation_max_rad"] <= torch.pi / 4
            ):
                plan = candidate
            else:
                before.restore()
                with place_ik_recovery() as recovery:
                    plan = super()._plan(request, context)
        if not plan.plan_success.any() and self.recovery_scene_vertices:
            plan = recover(self, request, context, plan, build)
        if recovery.used and plan.plan_success.any():
            trajectory = plan.joint_trajectory
            limits = _joint_velocity_limits(self.robot, None).to(trajectory.positions)
            if not _velocity_validity(
                trajectory.positions, trajectory.dt, limits
            ).all():
                if early_fallback is not None:
                    plan = early_fallback
                else:
                    return self.failed_plan(
                        request,
                        context,
                        message="Recovered Place exceeds final resampled joint velocity limits.",
                    )
        if not plan.plan_success.any():
            plan = self._retry_place_routes(request, context, plan)
        if early_fallback is not None:
            measured = plan.diagnostics.metadata.get("gen_sim_place_waypoint_recovery")
            earlier = early_fallback.diagnostics.metadata[
                "gen_sim_place_waypoint_recovery"
            ]
            keys = ("rotation_max_rad", "rotation_path_rad")
            if not plan.plan_success.any() or (
                measured is not None
                and tuple(measured[k] for k in keys) > tuple(earlier[k] for k in keys)
            ):
                plan = early_fallback
        plan = require_release_effect(
            self._with_contact_monitor_boundary(plan), handoff.resource
        )
        if keyframes is not None:
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_place_keyframe_preflight": keyframes,
                    },
                ),
            )
        return plan

    def _retry_place_routes(
        self, request: Any, context: Any, first: ActionPlan
    ) -> ActionPlan:
        """Try two approach vias without changing required release waypoints."""
        target = request.goal
        options = request.skill_options
        from .motion import _CARTESIAN_APPROACH, place_ik_recovery
        from .place_recovery import inspect
        from .pick_path import _pose_path_valid

        if (
            not isinstance(target, PlaceGoal)
            or first.plan_success.any()
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or not self.recovery_scene_vertices
            or request.motion_policy.strategy != "ik_interp"
            or request.motion_policy.dynamic_collision_mode != "off"
            or options.max_approach_retract_z is not None
            or target.tcp_symmetry != "none"
            or _CARTESIAN_APPROACH.get()
        ):
            return first
        endpoint = request.binding.endpoint("primary", "motion")
        held = context.get_held_object(endpoint.task_state_key)
        if held is None or held.semantics.affordance.get_custom_config(
            ADAPTIVE_GRASP
        ) in {"pour", "ends"}:
            return first
        from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
        from embodichain.lab.sim.atomic_actions.trajectory_ops import (
            resolve_pose_target,
        )

        xpos = resolve_pose_target(
            resolve_pose_goal(target.xpos, context, name="place_candidate_xpos"),
            num_envs=context.batch_size,
            device=self.device,
        )
        if xpos.ndim == 3:
            xpos = xpos.unsqueeze(1)
        arm = endpoint.require_target(JointPositionTarget)
        start = self.robot.compute_fk(
            qpos=context.robot.qpos[:, list(arm.joint_ids)],
            name=arm.control_part,
            to_matrix=True,
        )
        if not _inspect_pick_motion(self.robot, context.robot.qpos, arm.control_part)[
            "valid"
        ]:
            return first
        parameters = self.robot.get_solver(arm.control_part).get_iteration_params()
        state = _RandomState.capture(
            _cuda_devices(self.robot.device, context.robot.qpos.device)
        )
        attempts = []
        chosen = first
        try:
            for clearance in (0.05, 0.10):
                state.restore()
                via = start.clone()
                via[:, :2, 3] = (start[:, :2, 3] + xpos[:, 0, :2, 3]) / 2
                via[:, 2, 3] = (
                    torch.maximum(start[:, 2, 3], xpos[:, :, 2, 3].amax(1)) + clearance
                )
                selected = replace(
                    request,
                    goal=replace(target, xpos=torch.cat((via[:, None], xpos), dim=1)),
                )
                with place_ik_recovery():
                    candidate = super()._plan(selected, context)
                acceptance = inspect(self, selected, context, candidate)
                record = {"clearance_m": clearance, "acceptance": acceptance}
                attempts.append(record)
                if not acceptance["valid"]:
                    continue
                release = next(s for s in candidate.segments if s.name == "release")
                actual = self.robot.compute_fk(
                    qpos=candidate.joint_trajectory.positions[
                        :, release.start, list(arm.joint_ids)
                    ],
                    name=arm.control_part,
                    to_matrix=True,
                )
                record["release_pose_valid"] = _pose_path_valid(
                    actual, xpos[:, -1], parameters["pos_eps"], parameters["rot_eps"]
                )
                if record["release_pose_valid"]:
                    chosen = replace(
                        candidate,
                        scene_dependencies=first.scene_dependencies,
                        scene_dependency_monitor_until=first.scene_dependency_monitor_until,
                        scene_dependency_end_segment=first.scene_dependency_end_segment,
                    )
                    break
        finally:
            state.restore()
        return replace(
            chosen,
            diagnostics=replace(
                chosen.diagnostics,
                metadata={
                    **chosen.diagnostics.metadata,
                    "gen_sim_place_candidate_retry": {
                        "budget": 2,
                        "attempts": attempts,
                        "accepted": bool(chosen.plan_success.all()),
                        "release_pose_unchanged": True,
                    },
                },
            ),
        )


class GenSimPour(Pour):
    """GenSim Pour using measured pad-line geometry, not TCP-column assumptions."""

    skill_id = Pour.skill_id
    GoalType = Pour.GoalType
    OptionsType = Pour.OptionsType
    binding_contract = Pour.binding_contract
    continuations: dict | None = None
    scene_vertices: dict | None = None
    gravity_world: tuple[float, float, float] | None = None

    def __init__(self, pour_receivers: dict[str, str] | None = None) -> None:
        super().__init__()
        self._pour_receivers = dict(pour_receivers or {})

    def _receiver_id(self, invocation_id: str | None) -> str | None:
        if not self._pour_receivers:
            return None
        try:
            return self._pour_receivers[invocation_id]
        except KeyError as exc:
            raise ValueError(
                "Unbound GenSim Pour receiver; regenerate the bundle."
            ) from exc

    def _plan(self, request, context):
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        motion_target = motion.require_target(JointPositionTarget)
        handoff = GenSimStateHandoff.from_request(
            request, context, participant="GenSim Pour"
        )
        held = handoff.require_held()
        arm_qpos = arm_qpos_from_state(context, list(motion_target.joint_ids))
        eef = self.robot.compute_fk(
            qpos=arm_qpos, name=motion_target.control_part, to_matrix=True
        )
        object_to_eef = held.object_to_eef.to(self.device, torch.float32)
        object_pose = eef @ pose_inv(object_to_eef)
        continuation = (self.continuations or {}).get(request.invocation_id)
        if continuation is not None:
            if handoff.resource != continuation.resource:
                raise ValueError(
                    "Canonical Pour resource differs from its bound recipe."
                )
            object_poses, metadata = self._canonical_arc(
                request, context, continuation, held, object_pose, motion_target
            )
            return self._build_tilt_plan(
                request,
                context,
                handoff,
                held,
                arm_qpos,
                motion_target,
                grasp,
                object_to_eef,
                object_poses,
                metadata,
                retain=True,
            )
        prefix = motion_target.control_part.split("_arm")[0]
        names = set(self.robot.link_names)
        pad_names = (
            ["left_inner_finger_pad", "left_right_inner_finger_pad"]
            if prefix == "left"
            else ["right_left_inner_finger_pad", "right_inner_finger_pad"]
        )
        if not all(name in names for name in pad_names):
            raise ValueError(
                f"GenSim Pour cannot resolve both pad links for {prefix!r}."
            )
        pad_poses = [
            self.robot.get_link_pose(name, to_matrix=True) for name in pad_names
        ]
        jaw_world = pad_poses[1][:, :3, 3] - pad_poses[0][:, :3, 3]
        jaw_local = torch.bmm(
            object_pose[:, :3, :3].transpose(1, 2), jaw_world[..., None]
        ).squeeze(-1)
        affordance = deepcopy(held.semantics.affordance)
        if not isinstance(affordance, AxisAlignAffordance):
            raise ValueError("GenSim Pour requires an axis-aware grasp affordance.")
        upright = _unit(affordance.internal_axis.to(self.device, torch.float32))
        axis, direction, valid = jaw_tilt_axis(jaw_local, upright.expand_as(jaw_local))
        if not valid.all():
            raise ValueError(
                "GenSim Pour found a jaw line parallel to the vessel upright axis."
            )
        if not torch.allclose(axis, axis[0:1].expand_as(axis), atol=2e-2, rtol=0):
            raise ValueError(
                "GenSim Pour requires one stable jaw-line axis across environments."
            )
        receiver_id = self._receiver_id(request.invocation_id)
        if receiver_id is not None:
            receiver = context.scene.entities.get(receiver_id)
            if receiver is None or receiver.pose is None:
                raise ValueError(
                    f"GenSim Pour receiver {receiver_id!r} is not observed."
                )
            receiver_pose = receiver.pose.to(
                device=self.device, dtype=object_pose.dtype
            )
            if receiver_pose.ndim == 2:
                receiver_pose = receiver_pose.unsqueeze(0).expand(
                    context.batch_size, -1, -1
                )
            receiver_direction = receiver_pose[:, :3, 3] - object_pose[:, :3, 3]
            upright_world = object_pose[:, :3, :3] @ upright.expand_as(axis)[..., None]
            axis_world = object_pose[:, :3, :3] @ axis[..., None]
            receiver_direction = receiver_direction - (
                receiver_direction * upright_world.squeeze(-1)
            ).sum(-1, keepdim=True) * upright_world.squeeze(-1)
            receiver_direction = receiver_direction - (
                receiver_direction * axis_world.squeeze(-1)
            ).sum(-1, keepdim=True) * axis_world.squeeze(-1)
            visible_direction = torch.linalg.cross(
                axis_world.squeeze(-1), upright_world.squeeze(-1)
            )
            signed_visible = visible_direction * float(
                request.skill_options.rotate_angle
            )
            if torch.linalg.vector_norm(receiver_direction, dim=-1).min() <= 1e-5:
                raise ValueError("GenSim Pour receiver direction is degenerate.")
            flip = (signed_visible * receiver_direction).sum(-1, keepdim=True) < 0
            axis = torch.where(flip, -axis, axis)
        if not torch.allclose(axis, axis[0:1].expand_as(axis), atol=2e-2, rtol=0):
            raise ValueError(
                "GenSim Pour receiver direction selects inconsistent tilt signs across environments."
            )
        affordance.internal_axis = axis[0].detach().clone()
        semantics = ObjectSemantics(
            affordance=affordance,
            geometry=deepcopy(held.semantics.geometry),
            entity_id=held.semantics.entity_id,
            properties=deepcopy(held.semantics.properties),
            label=held.semantics.label,
        )
        count = request.motion_policy.sample_count
        if count < 3:
            raise ValueError("GenSim Pour needs at least three trajectory samples.")
        first = count // 2
        angles = torch.cat(
            (
                torch.linspace(0.0, 1.0, first + 1, device=self.device)[1:],
                torch.linspace(1.0, 0.0, count - first + 1, device=self.device)[1:],
            )
        ) * float(request.skill_options.rotate_angle)
        try:
            rotations = axis_angle_to_rotation_matrix(
                axis[:, None] * angles[None, :, None]
            )
            object_poses = object_pose[:, None].expand(-1, count, -1, -1).clone()
            pivot = torch.zeros(3, device=self.device, dtype=object_pose.dtype)
            vertices = getattr(affordance, "mesh_vertices", None)
            if isinstance(vertices, torch.Tensor) and vertices.numel():
                vertices = vertices.to(device=self.device, dtype=object_pose.dtype)
                center = (vertices.amin(0) + vertices.amax(0)) * 0.5
                up0 = upright.reshape(-1, 3)[0]
                top = (vertices @ up0).amax()
                pivot = center + up0 * (top - (center * up0).sum())
            anchor = (
                object_pose[:, None, :3, 3]
                + torch.einsum("bij,j->bi", object_pose[:, :3, :3], pivot)[:, None]
            )
            object_poses[:, :, :3, :3] = object_pose[:, None, :3, :3] @ rotations
            object_poses[:, :, :3, 3] = anchor - torch.einsum(
                "bnij,j->bni", object_poses[:, :, :3, :3], pivot
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "GenSim Pour geometry shape failure: "
                f"object_pose={tuple(object_pose.shape)}, axis={tuple(axis.shape)}, "
                f"angles={tuple(angles.shape)}, rotations={tuple(rotations.shape) if 'rotations' in locals() else None}, "
                f"pivot={tuple(pivot.shape) if 'pivot' in locals() else None}, "
                f"vertices={tuple(vertices.shape) if 'vertices' in locals() and isinstance(vertices, torch.Tensor) else None}"
            ) from exc
        metadata = {
            "pad_line_local": jaw_local.detach().cpu().tolist(),
            "receiver_entity_id": receiver_id,
            "tilt_axis_local": axis.detach().cpu().tolist(),
            "tilt_direction_local": direction.detach().cpu().tolist(),
        }
        return self._build_tilt_plan(
            request,
            context,
            handoff,
            held,
            arm_qpos,
            motion_target,
            grasp,
            object_to_eef,
            object_poses,
            metadata,
        )

    def _canonical_arc(self, request, context, route, held, object_pose, motion_target):
        from .pour_continuation import observed_pose, pad_line_world
        from .pour_geometry import pour_frames

        if (
            context.batch_size != 1
            or held.semantics.entity_id != route.object_id
            or self._receiver_id(request.invocation_id) != route.receiver_id
            or request.motion_policy.sample_count != route.sample_count
            or float(request.skill_options.rotate_angle) != route.angle_rad
        ):
            raise ValueError("Canonical Pour differs from its bound recipe.")
        source = self.scene_vertices[route.object_id][0].detach().cpu().numpy()
        receiver = self.scene_vertices[route.receiver_id][0].detach().cpu().numpy()
        receiver_pose = observed_pose(context, route.receiver_id)
        jaw_world = pad_line_world(self.robot, motion_target)
        jaw_local = (object_pose[:, :3, :3].transpose(1, 2) @ jaw_world[..., None])[
            0, :, 0
        ]
        frames = pour_frames(
            source,
            receiver,
            object_pose[0].detach().cpu().numpy(),
            receiver_pose[0].detach().cpu().numpy(),
            jaw_local.detach().cpu().numpy(),
            route.angle_rad,
            route.return_pose.detach().cpu().numpy(),
            sample_count=route.sample_count,
            gravity_world=self.gravity_world,
        )
        poses = torch.as_tensor(
            frames["object_arc"], device=self.device, dtype=object_pose.dtype
        )[None]
        metadata = {
            "pad_line_local": jaw_local[None].detach().cpu().tolist(),
            "receiver_entity_id": route.receiver_id,
            "tilt_axis_local": [frames["tilt_axis_object"].tolist()],
            "gen_sim_pour_geometry": {
                "revision": 2 if route.proxy_position_tolerance_m is not None else 1,
                **(
                    {"proxy_position_tolerance_m": route.proxy_position_tolerance_m}
                    if route.proxy_position_tolerance_m is not None
                    else {}
                ),
                "source_entity_id": route.object_id,
                "receiver_entity_id": route.receiver_id,
                "physical_evidence": False,
                "fluid_transfer": "unmodeled",
                "planning_pose_source": "held_fk_and_existing_object_to_eef",
                "observed_source_entry_pose": observed_pose(context, route.object_id)[0]
                .detach()
                .cpu()
                .tolist(),
                "entry_pose": frames["object_arc"][0].tolist(),
                "peak_pose": frames["object_arc"][frames["peak_index"]].tolist(),
                "restore_pose": frames["object_arc"][-1].tolist(),
                "pivot_object": frames["pivot_object"].tolist(),
                "pivot_world": frames["pivot_world"].tolist(),
                "receiver_pose": receiver_pose[0].detach().cpu().tolist(),
                "gravity_world": list(self.gravity_world),
                "target_roles": list(frames["target_roles"]),
                "receiver_bearing_m": frames["receiver_bearing_m"],
                "receiver_bearing_tie": frames["receiver_bearing_tie"],
                "tilt_direction_policy": frames["tilt_direction_policy"],
                "effective_angle_rad": frames["effective_angle_rad"],
                "peak_index": frames["peak_index"],
                "sample_count": route.sample_count,
            },
        }
        return poses, metadata

    def _build_tilt_plan(
        self,
        request,
        context,
        handoff,
        held,
        arm_qpos,
        motion_target,
        grasp,
        object_to_eef,
        object_poses,
        metadata,
        *,
        retain=False,
    ):
        # ik_interp prepends the observed start qpos; pass the remaining
        # count-1 Cartesian keyframes so every output sample is grounded.
        eef_targets = (object_poses @ object_to_eef[:, None])[:, 1:]
        options = request.motion_policy.to_motion_gen_options(
            start_qpos=arm_qpos,
            control_part=motion_target.control_part,
            interpolation_dt=context.require_control_dt(),
        )
        if options.strategy == "ik_interp":
            options.preserve_cartesian_samples = True
        before_ik_rng = torch.get_rng_state().tolist() if retain else None
        targets = build_pose_plan_states(eef_targets)
        result = self.motion_generator.generate(targets, options=options)
        recovered = False
        if (
            retain
            and options.strategy == "ik_interp"
            and getattr(context, "batch_size", None) == 1
            and result.positions is not None
            and result.dt is not None
            and not result.success.all()
        ):
            from .motion import (
                _joint_velocity_limits,
                _velocity_validity,
                resolve_continuous_ik_path,
            )

            limits = _joint_velocity_limits(self.robot, motion_target.control_part).to(
                result.positions
            )
            if (
                torch.isfinite(result.positions).all()
                and not _velocity_validity(result.positions, result.dt, limits).all()
            ):
                retry = resolve_continuous_ik_path(
                    self.motion_generator, targets, options, operation="retained Pour"
                )
                if retry is not None:
                    expected = (1, object_poses.shape[1], arm_qpos.shape[1])
                    if (
                        retry.positions.shape != expected
                        or not torch.equal(retry.positions[:, 0], arm_qpos)
                        or not torch.equal(retry.dt, result.dt)
                        or not retry.success.all()
                    ):
                        raise ValueError(
                            "Continuous Pour recovery changed its start or time grid."
                        )
                    result, recovered = retry, True
        if result.positions is None or result.dt is None or not result.success.all():
            failed = self.failed_plan(
                request,
                context,
                message="GenSim Pour tilt arc has no feasible IK path.",
            )
            if not retain:
                return failed
            return replace(
                failed,
                diagnostics=replace(
                    failed.diagnostics,
                    metadata={
                        **failed.diagnostics.metadata,
                        **metadata,
                        "gen_sim_pour_failed_kinematics": {
                            "scope": "planning_inputs_not_execution_or_ik_certificate",
                            "entry_arm_qpos": arm_qpos.detach().cpu().tolist(),
                            "torch_cpu_rng_state_before_generate": before_ik_rng,
                            "torch_intraop_threads": torch.get_num_threads(),
                            "object_to_eef": object_to_eef.detach().cpu().tolist(),
                            "required_eef_targets": eef_targets.detach().cpu().tolist(),
                            "per_waypoint_status": "not_exposed_by_shared_result",
                        },
                    },
                ),
            )
        grasp_target = grasp.require_target(JointPositionTarget)
        hand_ids = list(grasp_target.joint_ids)
        hand_qpos = grasp.joint_positions(
            "grasp",
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        base = context.last_qpos.clone()
        base[:, hand_ids] = hand_qpos
        _, timed = to_full_robot_trajectory(
            result,
            base_qpos=base,
            joint_ids=list(motion_target.joint_ids),
            env_ids=context.env_ids,
            control_dt=context.require_control_dt(),
        )
        if recovered:
            certificate = _inspect_pick_motion(
                self.robot, timed.positions[0], motion_target.control_part
            )
            metadata = {
                **metadata,
                "gen_sim_pour_continuous_ik": {
                    "unchanged_targets_and_timing": True,
                    "robot_proxy": certificate,
                },
            }
            if not certificate["valid"]:
                return self.failed_plan(
                    request,
                    context,
                    message="Continuous Pour recovery failed final robot-proxy qualification.",
                    retryable=False,
                )
        plan = self.build_plan(
            request,
            context,
            success=result.success
            & context.task.exclusive_held_object_mask(handoff.resource),
            trajectory=timed,
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name, metadata=metadata
            ),
            # Reuse the seed command as the hold for native pre-pour verification.
            segment_lengths=(
                {"entry_hold": 1, "pour": timed.waypoint_count - 1}
                if retain
                else {"pour": timed.waypoint_count}
            ),
        )
        return handoff.retain(plan, held) if retain else plan
