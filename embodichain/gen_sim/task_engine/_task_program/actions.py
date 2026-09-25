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
from dataclasses import dataclass, replace

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
    require_shared_task_state_key,
)
from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
    HeldObjectPoseGoal,
    MoveHeldObject,
)
from embodichain.lab.sim.atomic_actions.primitives.pour import Pour
from embodichain.lab.sim.atomic_actions.primitives.place import Place
from embodichain.lab.sim.atomic_actions.primitives.pick_up import PickUp
from embodichain.lab.sim.atomic_actions.primitives.hand_over import HandOver
from embodichain.lab.sim.atomic_actions.plans import PlannerDiagnostics
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

__all__ = [
    "GenSimPickUp",
    "GenSimHandOver",
    "GenSimMoveHeldObject",
    "GenSimPlace",
    "GenSimPour",
]

_PICK_GRASP_RULE = "gen_sim.pick_grasp_rule"


@dataclass
class _PickSelection:
    direction: torch.Tensor | None = None
    request: object = None
    trajectory: tuple | None = None


_PICK_SELECTION: ContextVar[_PickSelection | None] = ContextVar(
    "gen_sim_pick_selection", default=None
)
_RECEIVE_SELECTION: ContextVar[tuple | None] = ContextVar(
    "gen_sim_receive_selection", default=None
)


class GenSimPickUp(PickUp):
    """Apply explicit call-local grasp rules without changing shared sampling."""

    skill_id = PickUp.skill_id
    GoalType = PickUp.GoalType
    OptionsType = PickUp.OptionsType
    binding_contract = PickUp.binding_contract
    adaptive_unconstrained = False

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
                semantics.affordance.set_custom_config(ADAPTIVE_GRASP, "free")
                request = replace(request, goal=replace(goal, semantics=semantics))
        token = _PICK_SELECTION.set(_PickSelection(request=request))
        try:
            return super()._plan(request, context)
        finally:
            _PICK_SELECTION.reset(token)

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
            mode in {"free", "ends"}
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
            choices = grasp_strategies(axis, object_pose, tcp, ends=mode == "ends")
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
        candidates = affordance.get_grasp_candidates(
            generator,
            object_pose,
            approach_direction,
            obj_longest_axis=None,
            is_positive_part=True,
        )
        poses, ik_success = self._select_feasible_grasp_variants(
            candidates.poses,
            start_qpos,
            object_pose,
            manipulator,
            options,
            approach_direction,
        )
        return affordance.sample_candidates(
            AffordancePoseCandidates(
                poses=poses,
                costs=candidates.costs,
                valid=candidates.valid & ik_success,
            ),
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
        token = _RECEIVE_SELECTION.set(
            (resources, held) if mode in {"free", "ends"} else None
        )
        try:
            return super()._plan_existing_hold(request, context, resources, held)
        finally:
            _RECEIVE_SELECTION.reset(token)

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
        resources, held = selection
        axis = observed_axis(affordance, object_pose)
        source = object_pose @ held.object_to_eef.to(object_pose)
        destination = self.robot.compute_fk(
            qpos=context.robot.qpos[:, list(resources.second.arm.joint_ids)],
            name=resources.second.arm.control_part,
            to_matrix=True,
        )
        positive = opposite_grasp_end(axis, object_pose, source, destination)
        sample, _ = select_grasp(
            affordance,
            self.planning_services.grasp_pose_generator(grasp_target_id),
            object_pose,
            grasp_strategies(
                axis, object_pose, destination, ends=True, positive=positive
            ),
            context,
            sample_key,
        )
        return sample


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

    def _plan(self, request, context):
        goal = request.goal
        target = goal.object_target_pose
        if isinstance(target, torch.Tensor) and target.ndim == 3:
            motion = request.binding.endpoint("primary", "motion").require_target(
                JointPositionTarget
            )
            task_key = require_shared_task_state_key(
                request.binding.endpoint("primary", "motion"),
                request.binding.endpoint("primary", "grasp"),
                participant="GenSim transport",
            )
            held = context.get_held_object(task_key)
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
        plan = super()._plan(request, context)
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        task_key = require_shared_task_state_key(
            motion, grasp, participant="GenSim transport"
        )
        held = context.get_held_object(task_key)
        if held is not None and plan.plan_success.any():
            plan = replace(
                plan,
                expected_effects=StateDelta(held_object_updates={task_key: held}),
            )
        return plan


class GenSimPlace(Place):
    """Keep ordinary Place unchanged unless its Cartesian IK plan fails."""

    skill_id = Place.skill_id
    GoalType = Place.GoalType
    OptionsType = Place.OptionsType
    binding_contract = Place.binding_contract

    def _plan(self, request, context):
        from .motion import (
            _joint_velocity_limits,
            _velocity_validity,
            place_ik_recovery,
        )

        with place_ik_recovery() as recovery:
            plan = super()._plan(request, context)
        if recovery.used and plan.plan_success.any():
            trajectory = plan.joint_trajectory
            limits = _joint_velocity_limits(self.robot, None).to(trajectory.positions)
            if not _velocity_validity(
                trajectory.positions, trajectory.dt, limits
            ).all():
                return self.failed_plan(
                    request,
                    context,
                    message="Recovered Place exceeds final resampled joint velocity limits.",
                )
        return plan


class GenSimPour(Pour):
    """GenSim Pour using measured pad-line geometry, not TCP-column assumptions."""

    skill_id = Pour.skill_id
    GoalType = Pour.GoalType
    OptionsType = Pour.OptionsType
    binding_contract = Pour.binding_contract

    def __init__(self, pour_receivers: dict[str, str] | None = None) -> None:
        super().__init__()
        self._pour_receivers = dict(pour_receivers or {})

    def _plan(self, request, context):
        motion = request.binding.endpoint("primary", "motion")
        grasp = request.binding.endpoint("primary", "grasp")
        motion_target = motion.require_target(JointPositionTarget)
        task_key = require_shared_task_state_key(
            motion, grasp, participant="GenSim Pour"
        )
        held = context.get_held_object(task_key)
        if held is None:
            raise ValueError("GenSim Pour requires an exclusively held object.")
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
        receiver_id = self._pour_receivers.get(held.semantics.entity_id)
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
        ) * abs(float(request.skill_options.rotate_angle))
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
