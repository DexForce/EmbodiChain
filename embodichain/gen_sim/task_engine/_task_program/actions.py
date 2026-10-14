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

"""GenSim-only action wrappers; the shared Atomic Action implementations stay unchanged."""

from __future__ import annotations

from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

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
from embodichain.lab.sim.atomic_actions.primitives.place import Place, PlaceGoal
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
    candidates: AffordancePoseCandidates | None = None
    candidate_routes: list[tuple[int, AffordancePoseCandidates, torch.Tensor]] = field(
        default_factory=list
    )
    raw_candidate_routes: list[tuple[int, AffordancePoseCandidates, torch.Tensor]] = (
        field(default_factory=list)
    )


_PICK_SELECTION: ContextVar[_PickSelection | None] = ContextVar(
    "gen_sim_pick_selection", default=None
)


@dataclass
class _ReceiveSelection:
    resources: object
    held: object
    alternate: torch.Tensor
    sampled: bool = False


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

    def _compute_batch_candidate_ik(
        self,
        poses: torch.Tensor,
        joint_seed: torch.Tensor,
        manipulator: JointPositionTarget,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the episode's parallel budget only for large CPU IK batches."""
        from .motion import _CPU_BATCH_THREADS
        from embodichain.lab.sim.motion.solvers.pytorch_solver import PytorchSolver

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
            return super()._compute_batch_candidate_ik(poses, joint_seed, manipulator)
        previous = torch.get_num_threads()
        if previous >= threads:
            return super()._compute_batch_candidate_ik(poses, joint_seed, manipulator)
        try:
            torch.set_num_threads(threads)
            return super()._compute_batch_candidate_ik(poses, joint_seed, manipulator)
        finally:
            torch.set_num_threads(previous)

    def _select_feasible_grasp_variants(
        self,
        grasp_xpos: torch.Tensor,
        start_qpos: torch.Tensor,
        object_poses: torch.Tensor,
        manipulator: JointPositionTarget,
        options: PickUpOptions,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefilter grasp columns, leaving IK and roll selection to shared PickUp."""
        if (
            options.rotate_upright is not None
            or options.approach_alignment_max_angle is None
        ):
            return super()._select_feasible_grasp_variants(
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
        aligned = self._approach_alignment_mask(
            eef_variants, options, approach_direction
        )
        # Keep the full environment axis and both roll variants of a retained
        # candidate. Only columns rejected by every row can be removed safely.
        keep = aligned.any(dim=(0, 2))
        selected = eef_variants[:, :, 0].clone()
        success = torch.zeros(
            grasp_xpos.shape[:2], dtype=torch.bool, device=self.device
        )
        if keep.any():
            poses, valid = super()._select_feasible_grasp_variants(
                grasp_xpos[:, keep],
                start_qpos,
                object_poses,
                manipulator,
                options,
                approach_direction,
            )
            selected[:, keep] = poses
            success[:, keep] = valid
        return selected, success

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
        token = _PICK_SELECTION.set(selection)
        try:
            with robot_collision_scope():
                first = super()._plan(request, context)
                preview = (self.continuation_previews or {}).get(
                    getattr(request, "invocation_id", None)
                )
                if preview is not None:
                    from .pick_continuation import eligible, recover

                    if eligible(self, request, context, selection, preview):
                        first = recover(
                            self, request, context, first, selection, preview
                        )
                from .pick_path import eligible as owns_path, rebuild

                if owns_path(self, request, context, selection):
                    qualified = rebuild(self, request, context, first, selection)
                    if qualified.plan_success.all():
                        return qualified
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
                            return first
                return first
        finally:
            _PICK_SELECTION.reset(token)

    def _recover_raw_pick_candidates(self, request, context, first, selection):
        """Qualify legal geometry discarded by the coarse endpoint-IK screen."""
        from .articulation_collision import check_free_motion
        from .pick_candidate_path import geometric_variants, trajectory
        from .pick_continuation import preview_place
        from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

        previous = first.diagnostics.metadata.get("gen_sim_pick_candidate_retry", {})
        budget = max(0, 4 - len(previous.get("attempts", ())))
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
                pool = geometric_variants(self, raw, request, context, direction)
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
                    index = next(indices, None)
                    while index is not None and any(
                        torch.equal(pool.poses[:, index], value) for value in seen
                    ):
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
                    exact, receipt = trajectory(
                        self, selected, context, pose, direction
                    )
                    record = {
                        "strategy": strategy,
                        "candidate_id": index // 2,
                        "roll_variant": index % 2,
                        "selected_pose": pose.detach().cpu().tolist(),
                        "trajectory": receipt,
                    }
                    attempts.append(record)
                    if exact is None:
                        continue
                    selection.request, selection.trajectory, selection.direction = (
                        selected,
                        exact,
                        direction,
                    )
                    # Shared PickUp builds commands/effects from the screened
                    # trajectory; do not lose its accepted IK branch by replanning.
                    candidate = super()._plan(selected, context)
                    record["emitted_plan"] = _inspect_pick_plan(
                        self.robot, selected, context, candidate
                    )
                    if (
                        not candidate.plan_success.all()
                        or not record["emitted_plan"]["valid"]
                    ):
                        continue
                    arm = selected.binding.endpoint("primary", "motion").require_target(
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
                        continue
                    preview = (self.continuation_previews or {}).get(
                        request.invocation_id
                    )
                    if preview is not None:
                        record["place_preview"] = preview_place(
                            self, selected, context, candidate, preview, safety=True
                        )
                        if not record["place_preview"]["valid"]:
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
        if first.plan_success.any() or not selection.candidate_routes:
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
                if len(attempts) >= 4:
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
                candidate = super()._plan(selected, context)
                candidate = rebuild(self, selected, context, candidate, selection)
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
                        second = super()._plan(
                            replace(request, goal=replace(goal, grasp_xpos=alternate)),
                            context,
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
        for index, choice in enumerate(choices):

            def observe_pool(pool: AffordancePoseCandidates) -> None:
                selection.candidates = pool
                if (self.pick_purposes or {}).get(request.invocation_id) == "ordinary":
                    selection.candidate_routes.append(
                        (index, pool, choice.direction.clone())
                    )

            def observe_raw_pool(pool: AffordancePoseCandidates) -> None:
                selection.raw_candidate_routes.append(
                    (index, pool, choice.direction.clone())
                )

            sample, direction = select_grasp(
                affordance,
                generator,
                object_pose,
                (choice,),
                context,
                f"{sample_key}:strategy_{index}",
                feasible=lambda values, approach: self._select_feasible_grasp_variants(
                    values,
                    start_qpos,
                    object_pose,
                    manipulator,
                    options,
                    approach[:, None, None, :],
                ),
                candidate_observer=observe_pool,
                raw_candidate_observer=observe_raw_pool,
            )
            viable = sample.success & ~success
            attempt = {"strategy": index, "grasp_feasible": sample.success.tolist()}
            attempts.append(attempt)
            if not viable.any():
                continue
            # Screen the exact trajectory once, then return that same path to
            # PickUp's effect builder rather than planning it again from new seeds.
            (
                path_ok,
                candidate_path,
                candidate_lengths,
            ) = super()._get_full_pickup_trajectory(
                sample.poses,
                start_qpos,
                context.last_qpos,
                request.motion_policy,
                options,
                direction,
                manipulator,
                hand,
                *commands,
                context.require_control_dt(),
            )
            accepted = viable & path_ok
            attempt["path_feasible"] = path_ok.tolist()
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

    def _plan_existing_hold(self, request, context, resources, held):
        mode = held.semantics.affordance.get_custom_config(ADAPTIVE_GRASP)
        if mode not in {"ordinary", "pour", "ends"}:
            return super()._plan_existing_hold(request, context, resources, held)
        selection = _ReceiveSelection(
            resources,
            held,
            torch.zeros(context.batch_size, dtype=torch.bool, device=self.device),
        )
        token = _RECEIVE_SELECTION.set(selection)
        try:
            first = super()._plan_existing_hold(request, context, resources, held)
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
            return self._merge_receive_plans(request, context, first, second, metadata)
        finally:
            _RECEIVE_SELECTION.reset(token)

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
        sample, _ = select_grasp(
            affordance,
            self.planning_services.grasp_pose_generator(grasp_target_id),
            object_pose,
            (choice,),
            context,
            sample_key,
        )
        selection.sampled = True
        return replace(
            sample,
            metadata={
                **sample.metadata,
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
    held_path_scene_vertices: dict[str, torch.Tensor] | None = None

    def __init__(self, *, staging_contract: StagingContract | None = None) -> None:
        super().__init__()
        self.staging_contract = staging_contract or StagingContract.empty()

    def _plan(self, request, context):
        if getattr(self.motion_generator, "direct_transport_planning", False) is True:
            return super()._plan(request, context)
        source_request = request
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
        if self.held_path_scene_vertices is not None:
            from .held_path import qualify_or_recover

            plan = qualify_or_recover(self, source_request, context, plan)
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
        return require_release_effect(
            self._with_contact_monitor_boundary(plan), handoff.resource
        )

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
        task_key = handoff.resource
        held = handoff.require_held()
        arm_qpos = arm_qpos_from_state(context, list(motion_target.joint_ids))
        eef = self.robot.compute_fk(
            qpos=arm_qpos, name=motion_target.control_part, to_matrix=True
        )
        object_to_eef = held.object_to_eef.to(self.device, torch.float32)
        object_pose = eef @ pose_inv(object_to_eef)
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
        result = self.motion_generator.generate(
            build_pose_plan_states(eef_targets), options=options
        )
        if result.positions is None or result.dt is None or not result.success.all():
            return self.failed_plan(
                request,
                context,
                message="GenSim Pour tilt arc has no feasible IK path.",
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
        previous = None
        metadata = {
            "pad_line_local": jaw_local.detach().cpu().tolist(),
            "receiver_entity_id": receiver_id,
            "tilt_axis_local": axis.detach().cpu().tolist(),
            "tilt_direction_local": direction.detach().cpu().tolist(),
        }
        return self.build_plan(
            request,
            context,
            success=result.success & context.task.exclusive_held_object_mask(task_key),
            trajectory=timed,
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name, metadata=metadata
            ),
            segment_lengths={"pour": timed.waypoint_count},
        )
