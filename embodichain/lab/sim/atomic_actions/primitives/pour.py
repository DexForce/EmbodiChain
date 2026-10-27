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

"""Pour atomic action implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils.math import axis_angle_to_rotation_matrix, pose_inv

from embodichain.lab.sim.atomic_actions.affordance import AxisAlignAffordance
from embodichain.lab.sim.atomic_actions.goals import (
    PoseGoalValue,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
    split_joint_trajectory_at_pose,
    plan_named_arm_trajectory,
    pad_trajectory_to_length,
    resample_planned_trajectory,
)
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    resolve_pose_target,
)


def _validate_frame(pose: torch.Tensor, name: str) -> None:
    """Require finite proper transforms before geometric path construction."""
    if not pose.is_floating_point() or not torch.isfinite(pose).all():
        raise ValueError(f"{name} must contain finite floating rigid transforms.")
    if not torch.allclose(
        pose[..., 3, :],
        pose.new_tensor([0, 0, 0, 1]).expand_as(pose[..., 3, :]),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(f"{name} must have a homogeneous bottom row.")
    rotation = pose[..., :3, :3]
    if not torch.allclose(
        rotation.transpose(-1, -2) @ rotation,
        torch.eye(3).to(pose).expand_as(rotation),
        atol=1e-6,
        rtol=0,
    ) or not torch.allclose(
        torch.linalg.det(rotation),
        pose.new_ones(rotation.shape[:-2]),
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(f"{name} must have a proper rotation.")


@dataclass(frozen=True, slots=True, eq=False)
class PourTarget:
    """Geometry of a receiving opening and the held object's mouth.

    Args:
        receiver_pose: World opening pose, optionally a live scene reference.
            Its +Z is the outward opening normal.
        source_mouth_pose: Mouth frame relative to the held object. Its +Z
            points out of the mouth when the source is upright.
        receiver_radius: Usable circular opening radius, in metres.
        mouth_radius: Conservative source mouth radius, in metres.
    """

    receiver_pose: PoseGoalValue
    source_mouth_pose: torch.Tensor
    receiver_radius: float
    mouth_radius: float

    def __post_init__(self) -> None:
        validate_pose_goal(self.receiver_pose, "receiver_pose", allow_waypoints=False)
        if isinstance(self.receiver_pose, torch.Tensor):
            _validate_frame(self.receiver_pose, "receiver_pose")
        elif self.receiver_pose.relative_pose is not None:
            _validate_frame(
                self.receiver_pose.relative_pose, "receiver_pose.relative_pose"
            )
        pose = self.source_mouth_pose
        if not isinstance(pose, torch.Tensor) or pose.shape != (4, 4):
            raise ValueError("source_mouth_pose must be one (4, 4) transform.")
        _validate_frame(pose, "source_mouth_pose")
        if (
            any(
                isinstance(v, bool) or not math.isfinite(v) or v <= 0
                for v in (self.receiver_radius, self.mouth_radius)
            )
            or self.mouth_radius >= self.receiver_radius
        ):
            raise ValueError("Positive mouth_radius must be less than receiver_radius.")
        object.__setattr__(self, "source_mouth_pose", pose.clone())
        if isinstance(self.receiver_pose, torch.Tensor):
            object.__setattr__(self, "receiver_pose", self.receiver_pose.clone())
        else:
            object.__setattr__(self, "receiver_pose", self.receiver_pose.snapshot())


@dataclass(frozen=True, slots=True, eq=False)
class PourGoal:
    """Pour toward an opening, or rotate in place when no target is supplied.

    Args:
        target: Receiving and source-mouth geometry. Omitting it retains the
            original signed rotation about the source's internal axis.
    """

    target: PourTarget | None = None

    def __post_init__(self) -> None:
        if self.target is not None and type(self.target) is not PourTarget:
            raise TypeError("target must be a PourTarget or None.")


@dataclass(frozen=True, slots=True, eq=False)
class PourOptions(ActionOptions):
    """Per-invocation pouring behavior."""

    rotate_angle: float = math.pi / 4.0
    """Signed rotation about the held object's local internal axis, in radians."""

    dwell_duration: float = 0.5
    """Minimum time to remain at the poured orientation, in seconds."""

    mouth_height: float = 0.08
    """Preferred mouth height above the receiver opening plane, in metres."""

    max_mouth_height: float = 0.25
    """Maximum mouth height considered for a targeted pour, in metres."""

    clearance: float = 0.025
    """Source bounding-box clearance above the receiver plane, in metres."""

    approach_distance: float = 0.04
    """Extra lateral clearance outside the receiving opening before tilt."""

    max_rotation_step: float = math.pi / 36
    """Maximum angular spacing of mouth-compensated Cartesian keyframes."""

    def __post_init__(self) -> None:
        if not math.isfinite(self.rotate_angle):
            raise ValueError("rotate_angle must be finite.")
        if not math.isfinite(self.dwell_duration) or self.dwell_duration < 0.0:
            raise ValueError("dwell_duration must be finite and non-negative.")
        for name in (
            "mouth_height",
            "max_mouth_height",
            "clearance",
            "approach_distance",
            "max_rotation_step",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.max_mouth_height < self.mouth_height:
            raise ValueError("max_mouth_height must be at least mouth_height.")
        if self.max_rotation_step > math.pi / 12:
            raise ValueError("max_rotation_step must not exceed fifteen degrees.")


class Pour(AtomicAction[PourGoal, PourOptions]):
    """Rotate and return an exclusively held object about its internal axis."""

    skill_id: ClassVar[str] = "pour"
    GoalType: ClassVar[type] = PourGoal
    OptionsType: ClassVar[type] = PourOptions
    open_loop: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset(
                    {
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
                ),
                grasp_commands={GRASP_COMMAND: JointPositionCommand},
            ),
        ),
    )

    def _plan(
        self,
        request: ResolvedActionRequest[PourGoal, PourOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a held-object rotation followed by the inverse rotation."""
        goal = self.require_goal(request)
        options = request.skill_options
        motion_endpoint = request.binding.endpoint("primary", "motion")
        grasp_endpoint = request.binding.endpoint("primary", "grasp")
        manipulator = motion_endpoint.require_target(JointPositionTarget)
        end_effector = grasp_endpoint.require_target(JointPositionTarget)
        control_part = manipulator.control_part
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        task_state_key = require_shared_task_state_key(
            motion_endpoint,
            grasp_endpoint,
            participant="Pour primary participant",
        )

        held_object = context.get_held_object(task_state_key)
        if held_object is None:
            raise ValueError(
                "Pour requires an object held by task-state resource "
                f"{task_state_key!r} - run PickUp first."
            )
        affordance = held_object.semantics.affordance
        if goal.target is None and not isinstance(affordance, AxisAlignAffordance):
            raise ValueError(
                "Pour requires the held object to use an AxisAlignAffordance."
            )
        eligible = context.task.exclusive_held_object_mask(task_state_key)
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message="Held object is not exclusive to the task-state resource.",
            )

        start_arm_qpos = arm_qpos_from_state(context, arm_joint_ids)
        current_eef_pose = self.robot.compute_fk(
            qpos=start_arm_qpos,
            name=control_part,
            to_matrix=True,
        )
        object_to_eef = held_object.object_to_eef.to(
            device=self.device,
            dtype=torch.float32,
        )
        current_object_pose = torch.bmm(current_eef_pose, pose_inv(object_to_eef))

        if goal.target is not None:
            if getattr(affordance, "mesh_scope", "object") != "object":
                raise ValueError("Targeted Pour requires whole-object source geometry.")
            return self._plan_targeted(
                request,
                context,
                goal.target,
                current_object_pose,
                object_to_eef,
                start_arm_qpos,
                eligible,
                arm_joint_ids,
                hand_joint_ids,
                getattr(affordance, "mesh_vertices", None),
            )

        internal_axis = affordance.internal_axis.to(
            device=self.device,
            dtype=torch.float32,
        )
        internal_axis = internal_axis / torch.linalg.vector_norm(internal_axis)
        world_axis = torch.matmul(current_object_pose[:, :3, :3], internal_axis)
        rotation_delta = axis_angle_to_rotation_matrix(
            world_axis * options.rotate_angle
        )
        target_object_pose = current_object_pose.clone()
        target_object_pose[:, :3, :3] = torch.bmm(
            rotation_delta,
            current_object_pose[:, :3, :3],
        )
        poured_eef_pose = torch.bmm(target_object_pose, object_to_eef)

        # The object remains rigidly attached to the EEF throughout Pour.  The
        # first target performs the requested rotation about the object's
        # internal axis; the second target returns to the EEF pose observed by
        # FK at start_qpos, which is exactly the inverse rotation by the same
        # signed angle.  Supplying both targets in one call lets planners such
        # as cuRobo chain the two collision-aware legs without another public
        # MotionGenerator.generate invocation.
        eef_targets = torch.stack([poured_eef_pose, current_eef_pose], dim=1)

        result = self.motion_generator.generate(
            build_pose_plan_states(eef_targets),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_arm_qpos,
                control_part=control_part,
                interpolation_dt=context.require_control_dt(),
            ),
        )
        assert isinstance(result.success, torch.Tensor)
        assert result.positions is not None
        assert result.dt is not None
        success = result.success & eligible

        control_dt = context.require_control_dt()
        minimum_pour = max(2, request.motion_policy.sample_count // 2)
        minimum_return = max(2, request.motion_policy.sample_count - minimum_pour)
        pour_arm, return_arm = split_joint_trajectory_at_pose(
            result.positions,
            poured_eef_pose,
            robot=self.robot,
            control_part=control_part,
            first_sample_count=minimum_pour,
            second_sample_count=minimum_return,
            dt=result.dt,
            control_dt=control_dt,
        )
        dwell_steps = math.ceil(options.dwell_duration / control_dt)
        dwell_arm = pour_arm[:, -1:].expand(-1, dwell_steps, -1)
        return_arm = return_arm[:, 1:]
        arm_trajectory = torch.cat((pour_arm, dwell_arm, return_arm), dim=1)
        full = (
            context.last_qpos.unsqueeze(1)
            .expand(-1, arm_trajectory.shape[1], -1)
            .clone()
        )
        full[:, :, arm_joint_ids] = arm_trajectory
        # A contact-blocked finger measurement is not a holding command.
        # Retain closing effort while rotating, dwelling and returning.
        hold_hand_qpos = grasp_endpoint.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        full[:, :, hand_joint_ids] = hold_hand_qpos.unsqueeze(1)
        timed = TimedTrajectory.from_uniform_step(
            full,
            env_ids=context.env_ids,
            step_dt=control_dt,
        )

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=timed,
            segment_lengths={
                "pour": pour_arm.shape[1],
                "dwell": dwell_steps,
                "return": return_arm.shape[1],
            },
        )

    def _target_paths(
        self,
        target: PourTarget,
        context: PlanningContext,
        current: torch.Tensor,
        vertices: torch.Tensor,
        options: PourOptions,
        height_offset: float,
        fractions: torch.Tensor | None = None,
        report_geometry: bool = True,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, object]]:
        """Build side approach, mouth-controlled tilt, return and withdrawal."""
        receiver = resolve_pose_target(
            resolve_pose_goal(target.receiver_pose, context, name="receiver_pose"),
            num_envs=context.batch_size,
            device=self.device,
        )
        _validate_frame(receiver, "receiver_pose")
        mouth = target.source_mouth_pose.to(current)
        normal = receiver[:, :3, 2]
        source_up = current[:, :3, :3] @ mouth[:3, 2]
        cross = torch.linalg.cross(source_up, normal)
        sin = torch.linalg.vector_norm(cross, dim=-1)
        cos = (source_up * normal).sum(-1).clamp(-1, 1)
        align = axis_angle_to_rotation_matrix(
            cross / sin[:, None].clamp_min(1e-8) * torch.atan2(sin, cos)[:, None]
        )
        upright = align @ current[:, :3, :3]
        current_mouth = (current @ mouth)[:, :3, 3]
        direction = receiver[:, :3, 3] - current_mouth
        direction -= (direction * normal).sum(-1, keepdim=True) * normal
        norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        direction = torch.where(
            norm > 1e-6, direction / norm.clamp_min(1e-6), receiver[:, :3, 0]
        )
        axis = torch.linalg.cross(normal, direction)
        if fractions is None:
            count = max(2, math.ceil(options.rotate_angle / options.max_rotation_step))
            fractions = torch.linspace(0, 1, count + 1, device=self.device)
        else:
            count = fractions.numel() - 1
        rotations = (
            axis_angle_to_rotation_matrix(
                axis[:, None] * (fractions * options.rotate_angle)[None, :, None]
            )
            @ upright[:, None]
        )
        if vertices.ndim == 2:
            vertices = vertices[None].expand(context.batch_size, -1, -1)
        if vertices.ndim != 3 or vertices.shape[0] not in (1, context.batch_size):
            raise ValueError("Pour source mesh must have shape (V,3) or (B,V,3).")
        # Eight local AABB corners bound the source body throughout the tilt.
        bounds = torch.stack((vertices.amin(-2), vertices.amax(-2)), dim=1).to(current)
        bits = torch.tensor(
            [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
            device=self.device,
            dtype=torch.long,
        )
        corners = bounds[:, bits, torch.arange(3, device=self.device)]
        relative = corners - mouth[:3, 3]
        upright_relative = (upright[:, None] @ relative[..., None]).squeeze(-1)
        cosine = (upright_relative * normal[:, None]).sum(-1)
        sine = (
            torch.linalg.cross(axis[:, None], upright_relative) * normal[:, None]
        ).sum(-1)
        # A corner's normal displacement is a*cos(theta)+b*sin(theta).
        # Its minimum lies at an endpoint or atan2(-b, -a). Taking the full
        # angular envelope also covers extrema between the sampled poses.
        critical = torch.atan2(-sine, -cosine).clamp(0.0, options.rotate_angle)
        lowest = torch.stack(
            (
                cosine,
                cosine * math.cos(options.rotate_angle)
                + sine * math.sin(options.rotate_angle),
                cosine * critical.cos() + sine * critical.sin(),
            ),
            dim=-1,
        ).amin(dim=(-2, -1))
        height = (options.clearance - lowest).clamp_min(options.mouth_height)
        height += height_offset
        heights = height[:, None].expand(-1, count + 1)
        valid = (cos > 0.5) & (height <= options.max_mouth_height)
        planar = (
            relative - (relative * mouth[:3, 2]).sum(-1, keepdim=True) * mouth[:3, 2]
        )
        radius = torch.linalg.vector_norm(planar, dim=-1).amax(-1)
        stand_off = target.receiver_radius + radius + options.approach_distance
        progress = ((fractions - 0.2) / 0.6).clamp(0, 1)
        progress = progress.square() * (3 - 2 * progress)
        mouth_positions = (
            receiver[:, None, :3, 3]
            - direction[:, None]
            * stand_off[:, None, None]
            * (1 - progress)[None, :, None]
            + normal[:, None] * heights[:, :, None]
        )
        poses = current[:, None].expand(-1, count + 1, -1, -1).clone()
        poses[:, :, :3, :3] = rotations
        poses[:, :, :3, 3] = mouth_positions - (rotations @ mouth[:3, 3, None]).squeeze(
            -1
        )
        paths = (poses[:, :1], poses[:, 1:], poses[:, :-1].flip(1), current[:, None])
        return (
            valid,
            paths,
            (
                {
                    "direction_world": direction.tolist(),
                    "mouth_height_range_m": torch.stack(
                        (heights.amin(-1), heights.amax(-1)), -1
                    ).tolist(),
                }
                if report_geometry
                else {}
            ),
        )

    def _plan_targeted(
        self,
        request: ResolvedActionRequest[PourGoal, PourOptions],
        context: PlanningContext,
        target: PourTarget,
        current: torch.Tensor,
        object_to_eef: torch.Tensor,
        start: torch.Tensor,
        eligible: torch.Tensor,
        arm_ids: list[int],
        hand_ids: list[int],
        vertices: torch.Tensor | None,
    ) -> ActionPlan:
        """Try bounded heights while keeping accepted rows independently."""
        options = request.skill_options
        if not 0 < options.rotate_angle <= math.pi / 2:
            raise ValueError("Targeted Pour requires a positive tilt of at most pi/2.")
        if (
            vertices is None
            or not vertices.numel()
            or not torch.isfinite(vertices).all()
        ):
            raise ValueError("Targeted Pour requires a finite non-empty source mesh.")
        motion = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        success = torch.zeros_like(eligible)
        selected: list[torch.Tensor] | None = None
        offsets = current.new_full((context.batch_size,), -1.0)
        selected_heights = current.new_zeros((context.batch_size, 2))
        selected_directions = current.new_zeros((context.batch_size, 3))
        geometry: dict[str, object] = {}
        for offset in (0.0, 0.025, 0.05):
            valid, paths, geometry = self._target_paths(
                target, context, current, vertices, options, offset
            )
            valid &= eligible
            if not (valid & ~success).any():
                continue
            seed = start
            parts = []
            for phase_index, path in enumerate(paths):
                eef = path @ object_to_eef[:, None]
                if (
                    phase_index in (1, 2)
                    and request.motion_policy.strategy == "ik_interp"
                ):
                    # Retiming covers the complete tilt with one progress law;
                    # every emitted Cartesian sample compensates the mouth.
                    def sample(
                        progress: torch.Tensor, *, reverse: bool = phase_index == 2
                    ) -> torch.Tensor:
                        _, sampled, _ = self._target_paths(
                            target,
                            context,
                            current,
                            vertices,
                            options,
                            offset,
                            fractions=1 - progress if reverse else progress,
                            report_geometry=False,
                        )
                        poses = torch.cat((sampled[0], sampled[1]), dim=1)
                        return poses @ object_to_eef[:, None]

                    motion_options = request.motion_policy.to_motion_gen_options(
                        start_qpos=seed,
                        control_part=motion.control_part,
                        sample_count=path.shape[1] + 1,
                        interpolation_dt=context.require_control_dt(),
                    )
                    motion_options.preserve_cartesian_samples = True
                    motion_options.cartesian_path_sampler = sample
                    result = self.motion_generator.generate(
                        build_pose_plan_states(eef),
                        options=motion_options,
                    )
                    planned = result.success
                    arm = resample_planned_trajectory(
                        result.positions,
                        path.shape[1] + 1,
                        dt=result.dt,
                        control_dt=context.require_control_dt(),
                    )
                else:
                    planned, arm = plan_named_arm_trajectory(
                        self.motion_generator,
                        motion.control_part,
                        seed,
                        eef,
                        max(
                            2,
                            request.motion_policy.sample_count // 4,
                            path.shape[1] + 1,
                        ),
                        request.motion_policy,
                        context.require_control_dt(),
                    )
                valid &= planned
                parts.append(arm)
                seed = arm[:, -1]
            take = valid & ~success
            if selected is None:
                selected = parts
            else:
                for i, part in enumerate(parts):
                    size = max(part.shape[1], selected[i].shape[1])
                    selected[i] = torch.where(
                        take[:, None, None],
                        pad_trajectory_to_length(part, size),
                        pad_trajectory_to_length(selected[i], size),
                    )
            offsets = torch.where(take, offsets.new_tensor(offset), offsets)
            selected_heights = torch.where(
                take[:, None],
                current.new_tensor(geometry["mouth_height_range_m"]),
                selected_heights,
            )
            selected_directions = torch.where(
                take[:, None],
                current.new_tensor(geometry["direction_world"]),
                selected_directions,
            )
            success |= take
            if (success | ~eligible).all():
                break
        if selected is None or not success.any():
            return self.failed_plan(
                request, context, message="No feasible mouth-controlled pouring path."
            )
        prepare, pour, back, retreat = selected
        dwell_steps = math.ceil(options.dwell_duration / context.require_control_dt())
        dwell = pour[:, -1:].expand(-1, dwell_steps, -1)
        parts = [prepare, pour[:, 1:], dwell, back[:, 1:], retreat[:, 1:]]
        arm_path = torch.cat(parts, dim=1)
        full = context.last_qpos[:, None].expand(-1, arm_path.shape[1], -1).clone()
        full[:, :, arm_ids] = arm_path
        hand = request.binding.endpoint("primary", "grasp").joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        full[:, :, hand_ids] = hand[:, None]
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                full, env_ids=context.env_ids, step_dt=context.require_control_dt()
            ),
            segment_lengths=dict(
                zip(
                    ("prepare", "pour", "dwell", "return", "retreat"),
                    [part.shape[1] for part in parts],
                )
            ),
            diagnostics=PlannerDiagnostics(
                backend=self.planning_services.planner_name,
                metadata={
                    "pour_target": {
                        "direction_world": selected_directions.tolist(),
                        "mouth_height_range_m": selected_heights.tolist(),
                    },
                    "height_offset_m": offsets.tolist(),
                },
            ),
        )


__all__ = ["Pour", "PourGoal", "PourOptions", "PourTarget"]
