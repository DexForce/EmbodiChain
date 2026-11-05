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

"""Scoped native closing qualification, separate from observed grasp evidence."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from copy import deepcopy
from collections.abc import Iterator
from typing import Any
import torch

from embodichain.lab.sim.atomic_actions.source_native_contact import (
    NativeClosingQuery,
    NativeRigidRestScene,
)
from embodichain.lab.sim.motion.planners.utils import RejectedMotionCandidate
from ..bindings import JointPositionTarget
from ..planning_probes import PickRigidSceneProvider
from ..goals import _resolve_object_pose, resolve_pose_goal
from embodichain.utils.math import pose_inv
from embodichain.lab.sim.motion.mesh_collision import (
    PrecontactMotionQuery,
    MotionCollisionReport,
)
from ..trajectory_ops import resolve_pose_target, interpolate_hand_qpos
from ..plans import PlanQualification, PlanningFailure
from ._pick_adaptive import current_pick_state

__all__: list[str] = []


def native_handover_source(request: Any) -> bool:
    """Select only the source revision-eight end-grasp contract."""
    return (
        request.skill_options.native_closing_revision == 8
        and request.skill_options.adaptive_grasp == "handover_source"
    )


def native_pour_source(request: Any) -> bool:
    """Select the declared native Pour-purpose candidate contract."""
    return (
        request.skill_options.native_closing_revision is not None
        and request.skill_options.adaptive_grasp == "pour"
    )


def requires_native_scene(request: Any) -> bool:
    return (
        request.skill_options.qualify_fixed_grasp
        or request.skill_options.adaptive_grasp == "ordinary"
        or native_handover_source(request)
    )


def validate_closing_request(action: Any, request: Any, context: Any) -> None:
    if request.skill_options.native_closing_revision is None:
        return
    if request.skill_options.native_closing_revision in (6, 7, 8, 9) and (
        (
            request.recovery_policy.planning_recovery is None
            and not request.skill_options.qualify_fixed_grasp
        )
        or (
            not (
                native_handover_source(request)
                or native_pour_source(request)
                or request.skill_options.constrained_grasp
                or request.skill_options.qualify_fixed_grasp
            )
            and (
                request.skill_options.adaptive_grasp != "ordinary"
                or not request.skill_options.require_qualified_approach
            )
        )
        or (
            request.skill_options.full_opening_clearance
            and not request.skill_options.constrained_grasp
        )
    ):
        raise ValueError(
            "Native Pick revisions 6-9 require bounded ordinary/Pour or revision-eight end-grasp recovery without the global opening envelope."
        )
    if (
        requires_native_scene(request)
        and request.skill_options.native_closing_revision >= 7
        and not isinstance(
            action.planning_services.free_motion_query, PrecontactMotionQuery
        )
    ):
        raise ValueError(
            "Native Pick revision 7 requires the explicit precontact scene query capability."
        )
    if request.skill_options.native_closing_revision >= 8:
        if requires_native_scene(request) and not isinstance(
            action.planning_services.pick_closing_query_provider, PickRigidSceneProvider
        ):
            raise ValueError(
                "Native Pick revisions 8-9 require a native rigid rest-scene provider."
            )
    if (
        request.skill_options.native_closing_revision >= 8
        and not request.skill_options.constrained_grasp
        and not request.skill_options.qualify_fixed_grasp
    ):
        from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator

        target = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        generator = action.planning_services.grasp_pose_generator(target.target_id)
        if (
            not isinstance(generator, AntipodalGraspPoseGenerator)
            or generator.collision_cfg.filter_ground_collision
        ):
            raise ValueError(
                "Deferred native center proposals require Antipodal geometry without a ground-filter override."
            )
    if context.batch_size != 1 or context.env_ids.tolist() != [0]:
        raise ValueError("Native Pick closing requires B1/env0.")
    if action.planning_services.pick_closing_query_provider is None:
        raise ValueError("Native Pick closing requires a query provider.")
    motion = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    grasp = request.binding.endpoint("primary", "grasp").require_target(
        JointPositionTarget
    )
    cfg = action.planning_services.hand_geometry_cfg(grasp.target_id)
    if cfg.sample_count != 65:
        raise ValueError("Source native closing requires exactly 65 declared samples.")
    if cfg.motion_part != motion.control_part or cfg.hand_part != grasp.control_part:
        raise ValueError(
            "Native closing calibration does not match the resolved endpoints."
        )


def _object_pose(request: Any, context: Any, device: torch.device) -> torch.Tensor:
    goal = request.goal
    return (
        _resolve_object_pose(goal.semantics, context, name="pickup_object_pose")
        if goal.object_pose is None
        else resolve_pose_target(
            resolve_pose_goal(goal.object_pose, context, name="object_pose"),
            num_envs=context.batch_size,
            device=device,
        )
    )


@contextmanager
def closing_query_scope(
    action: Any, request: Any, context: Any
) -> Iterator[NativeClosingQuery | None]:
    if request.skill_options.native_closing_revision is None:
        yield None
        return
    validate_closing_request(action, request, context)
    grasp = request.binding.endpoint("primary", "grasp").require_target(
        JointPositionTarget
    )
    cfg = action.planning_services.hand_geometry_cfg(grasp.target_id)
    query = action.planning_services.pick_closing_query_provider.prepare(
        request, context, hand_geometry_cfg=cfg
    )
    if not isinstance(query, NativeClosingQuery):
        raise TypeError("Native Pick provider must return a NativeClosingQuery.")
    try:
        if (
            query.closed
            or query.revision != request.skill_options.native_closing_revision
            or query.steps != 65
        ):
            raise ValueError(
                "Native Pick query is closed or uses a different closing law/grid."
            )
        if (
            query.scope.observation_timestamp != context.robot.timestamp
            or query.scope.scene_version != context.scene.version
            or query.scope.env_id != 0
        ):
            raise ValueError(
                "Native Pick query does not belong to this observation scope."
            )
        if query.revision >= 7 and (
            type(query.identity.get("simulation_uid")) is not str
            or not query.identity["simulation_uid"]
        ):
            raise ValueError(
                "Native precontact query requires the exact captured source UID."
            )
        captured = query.object_pose_at_capture
        expected = _object_pose(request, context, action.device)
        if not isinstance(captured, torch.Tensor) or not torch.equal(
            captured.to(expected), expected
        ):
            raise ValueError("Native Pick query has a different captured object pose.")
        yield query
    finally:
        query.close()


def inspect_precontact_scene(
    action: Any,
    request: Any,
    context: Any,
    positions: torch.Tensor,
    query: NativeClosingQuery,
) -> dict[str, Any]:
    arm = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    tcp = action.robot.compute_batch_fk(
        qpos=positions[None, :1, list(arm.joint_ids)],
        name=arm.control_part,
        to_matrix=True,
    )[0, 0]
    pose = _object_pose(request, context, positions.device)
    contact = query.inspect((pose_inv(pose)[0] @ tcp).detach().cpu().double().numpy())
    if not contact["valid"]:
        return {"valid": False, "source_closing": contact}
    source_uid = query.identity.get("simulation_uid")
    if type(source_uid) is not str or not source_uid:
        raise ValueError(
            "Native precontact query requires the exact captured source UID."
        )
    result = action.planning_services.free_motion_query.inspect_precontact_closure(
        positions,
        control_part=arm.control_part,
        arm_joint_ids=arm.joint_ids,
        source_uid=source_uid,
        contact_fraction=contact["nominal_closing_fraction"],
        grid_samples=query.steps,
    )
    if (
        not isinstance(result, MotionCollisionReport)
        or result.scope != "discrete_precontact_meshes"
        or result.excluded_rigid_uid != source_uid
    ):
        raise ValueError(
            "Native Pick requires the exact source-excluding precontact query scope."
        )
    receipt = result.metadata()
    if result.valid and query.revision in (8, 9):
        scene = query.rigid_scene_screen
        if scene is None:
            scene = action.planning_services.pick_closing_query_provider.prepare_rigid_scene(
                query
            )
        if (
            not isinstance(scene, NativeRigidRestScene)
            or scene is not query.rigid_scene_screen
            or scene.closed
            or scene.scope != query.scope
        ):
            if isinstance(scene, NativeRigidRestScene):
                scene.close()
            raise ValueError(
                "Native rigid scene must attach to the exact closing-query scope."
            )
        certificate = query.rigid_scene_screen.inspect(
            tcp.detach().cpu().double().numpy(), contact["checked_samples"]
        )
        receipt = {
            **receipt,
            "valid": bool(certificate["valid"]),
            "native_rigid_pair_rest": certificate,
        }
    return {
        **receipt,
        "nominal_closing_fraction": contact["nominal_closing_fraction"],
        "nominal_grid_samples": query.steps,
        "post_contact_commands_screened": False,
    }


def inspect_closing_plan(
    action: Any, request: Any, context: Any, plan: Any, query: NativeClosingQuery
) -> PlanQualification:
    """Qualify exact emitted close commands/FK samples; never replan a trajectory."""
    valid = False
    if plan.plan_success.any():
        close = next((s for s in plan.segments if s.name == "close"), None)
        if close is None or close.stop <= close.start:
            raise ValueError(
                "Native Pick closing requires the actual emitted close phase."
            )
        hand_endpoint = request.binding.endpoint("primary", "grasp")
        hand = hand_endpoint.require_target(JointPositionTarget)
        commands = plan.joint_trajectory.positions[
            :, close.start : close.stop, list(hand.joint_ids)
        ]
        opened, closed = (
            hand_endpoint.joint_positions(
                command,
                num_envs=context.batch_size,
                device=commands.device,
                dtype=commands.dtype,
            )
            for command in ("open", "grasp")
        )
        expected = torch.cat(
            (
                interpolate_hand_qpos(
                    opened, closed, n_waypoints=request.skill_options.hand_interp_steps
                ),
                closed[:, None].expand(
                    -1, request.skill_options.grasp_settle_steps, -1
                ),
            ),
            1,
        )
        if not torch.equal(commands, expected):
            receipt = {
                "valid": False,
                "reason": "emitted_hand_closing_commands_changed",
            }
        else:
            motion = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            poses = action.robot.compute_batch_fk(
                qpos=plan.joint_trajectory.positions[
                    :, close.start : close.stop, list(motion.joint_ids)
                ],
                name=motion.control_part,
                to_matrix=True,
            )
            valid = bool(
                query.mask(poses, _object_pose(request, context, poses.device)).all()
            )
            receipt = {
                "valid": valid,
                "emitted_close_samples": poses.shape[1],
                **query.report(),
            }
            if valid and query.revision >= 7 and requires_native_scene(request):
                scene = inspect_precontact_scene(
                    action,
                    request,
                    context,
                    plan.joint_trajectory.positions[0, close.start : close.stop],
                    query,
                )
                receipt["closing_scene"] = scene
                valid &= bool(scene["valid"])
                receipt["valid"] = valid

    else:
        receipt = {"valid": False, **query.report()}
    return PlanQualification(
        plan.plan_success & valid,
        replace(
            plan.diagnostics,
            metadata={**plan.diagnostics.metadata, "pick_contact_geometry": receipt},
            failure=(
                plan.diagnostics.failure
                if valid or not plan.plan_success.any()
                else PlanningFailure("pick_contact_unqualified", retryable=False)
            ),
        ),
    )


def finish_closing_plan(action: Any, request: Any, context: Any, plan: Any) -> Any:
    state = current_pick_state()
    if state is None or state.closing_query is None:
        return plan
    certificate = inspect_closing_plan(
        action, request, context, plan, state.closing_query
    )
    contact = certificate.diagnostics.metadata.get("pick_contact_geometry", {})
    if "closing_scene" in contact:
        state.closing_scene_checks.append(
            {"stage": "emitted", **contact["closing_scene"]}
        )
    if state.closing_scene_checks:
        certificate = replace(
            certificate,
            diagnostics=replace(
                certificate.diagnostics,
                metadata={
                    **certificate.diagnostics.metadata,
                    "pick_closing_scene_search": deepcopy(state.closing_scene_checks),
                },
            ),
        )
    if state.depth_trace:
        certificate = replace(
            certificate,
            diagnostics=replace(
                certificate.diagnostics,
                metadata={
                    **certificate.diagnostics.metadata,
                    "pick_native_depth_search": {
                        "attempts": deepcopy(state.depth_trace),
                        "extra_full_paths": state.depth_recovery_attempts,
                        "shared_recovery_budget": request.recovery_policy.planning_recovery.max_candidate_paths,
                        "physical_evidence": False,
                    },
                },
            ),
        )
    rejected = plan.plan_success & ~certificate.valid_mask
    if not rejected.any():
        return replace(plan, diagnostics=certificate.diagnostics)
    # B1 means a currently successful path cannot contain an earlier rejected
    # row. Keep this candidate's original samples before executable rows hold.
    candidate = RejectedMotionCandidate(
        plan.joint_trajectory.positions, plan.joint_trajectory.dt, rejected
    )
    return replace(
        plan,
        plan_success=certificate.valid_mask,
        diagnostics=certificate.diagnostics,
        commands=action._authorize_command_targets(
            request, plan.commands, active_mask=certificate.valid_mask
        ),
        joint_trajectory=plan.joint_trajectory.hold_rows(
            certificate.valid_mask, context.robot.qpos
        ),
        rejected_candidate=candidate,
    )
