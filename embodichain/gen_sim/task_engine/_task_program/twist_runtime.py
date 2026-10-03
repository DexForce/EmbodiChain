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
"""E8 lowering and measured acceptance around the unchanged public Twist."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Any, ClassVar
from contextvars import ContextVar

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionControlOverrides,
    Twist,
    TwistGoal,
    TwistOptions,
    TwistAffordance,
    ObjectSemantics,
    SceneEntityPose,
    MoveJoints,
    MoveJointsOptions,
    JointPositionGoal,
    JointPositionCommand,
    JointPositionTarget,
    RecoveryPolicy,
    GRASP_COMMAND,
    OPEN_COMMAND,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.trajectory_ops import resolve_pose_target
from embodichain.lab.sim.sensors import ContactSensorCfg, ArticulationContactFilterCfg
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations.extensions import (
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.task_program.semantics import (
    SceneLinkRef,
    SceneArticulationRef,
    SkillPolicyPreset,
)
from embodichain.utils import logger

from .press_runtime import (
    PressContactSensor,
    _closed_finger_tip_offset,
    _park_clearance,
)
from .twist_binding import (
    TwistRoute,
    PREPARE_CALL,
    TWIST_CALL,
    TWIST_REVISION,
    discover_twist,
)
from .articulation_binding import PARK_CALL

__all__: list[str] = []
SENSOR_UID = "gen_sim_twist_evidence"
ANGLE_TOLERANCE = math.radians(5.0)
INITIAL_TOLERANCE = math.radians(2.0)
INITIAL_VELOCITY_TOLERANCE = math.radians(2.0)
STARTUP_MAX_DURATION = 32.0
STARTUP_MAX_SAMPLES = 8192
PARENT_CONTACT_FRACTION_LIMIT = 0.25
PARENT_CONTACT_SCORE_LIMIT = 0.05
COARSE_TURN_THRESHOLD = math.radians(15.0)
QPOS_PATH_DEADBAND = math.radians(0.5)
_ROLL_CANDIDATES = (
    0.0,
    math.pi / 2,
    -math.pi / 2,
    math.pi,
    3 * math.pi / 4,
    -3 * math.pi / 4,
    math.pi / 4,
    -math.pi / 4,
)
_AXIAL_CANDIDATES = (-0.25, 0.0, 0.25, 0.5)
TWIST_CANDIDATE_FALLBACKS = 3


@dataclass
class _TwistCandidateState:
    """Run-local E8 candidate cursor; never shared with public Twist."""

    forced_index: int | None = None
    order: tuple[int, ...] = ()
    selected_index: int | None = None
    fallback_count: int = 0
    locked: bool = False
    contact_path: float = 0.0
    last_contact_qpos: float | None = None
    goal_complete: bool = False


_CANDIDATE_STATE: ContextVar[_TwistCandidateState | None] = ContextVar(
    "gen_sim_twist_candidate_state", default=None
)


def _candidate_state() -> _TwistCandidateState:
    state = _CANDIDATE_STATE.get()
    if state is None:
        state = _TwistCandidateState()
        _CANDIDATE_STATE.set(state)
    return state


def _candidate_table(grip_depth: float = 0.01) -> tuple[tuple[int, float, float], ...]:
    if not math.isfinite(grip_depth) or grip_depth <= 0.0:
        raise ValueError("E8 axial candidates require a positive grip depth.")
    values = [
        (roll, fraction * grip_depth)
        for roll in _ROLL_CANDIDATES[:4]
        for fraction in _AXIAL_CANDIDATES
    ]
    return tuple((index, roll, shift) for index, (roll, shift) in enumerate(values))


def _candidate_contact_decision(
    state: _TwistCandidateState,
    *,
    target_contact: bool,
    already_reached: bool,
) -> tuple[bool, bool]:
    """Return ``(post_policy_success, fallback_used)`` for one E8 chunk."""
    if target_contact or already_reached:
        state.locked = True
        state.forced_index = state.selected_index
        return True, False
    if (
        not state.locked
        and state.fallback_count < TWIST_CANDIDATE_FALLBACKS
        and len(state.order) > state.fallback_count + 1
    ):
        state.fallback_count += 1
        state.forced_index = state.order[state.fallback_count]
        return True, True
    return False, False


class GenSimTwist(Twist):
    """Use public Twist trajectories with a measured remaining joint angle."""

    skill_id = Twist.skill_id
    GoalType = Twist.GoalType
    OptionsType = Twist.OptionsType
    binding_contract = Twist.binding_contract

    @staticmethod
    def _current_qpos(request: Any, context: Any) -> float | None:
        geometry = request.goal.semantics.geometry
        measured = geometry.get("gen_sim_twist_measured_qpos")
        if measured is not None:
            if type(measured) is not float or not math.isfinite(measured):
                raise ValueError("E8 requires a finite freshly measured joint angle.")
            return measured
        articulation_id = geometry.get("gen_sim_twist_articulation_id")
        joint = geometry.get("gen_sim_twist_joint")
        if not isinstance(articulation_id, str) or not isinstance(joint, str):
            return None
        state = context.get_articulation_joint_state(articulation_id, joint)
        if state is None:
            return None
        values = state.position.reshape(-1)
        if values.numel() != context.batch_size:
            raise ValueError("E8 articulation qpos observation has the wrong batch.")
        value = float(values[0])
        return value if math.isfinite(value) else None

    @staticmethod
    def _candidate_request(request: Any, roll: float, shift: float) -> Any:
        affordance = request.goal.semantics.affordance
        if not isinstance(affordance, TwistAffordance):
            raise TypeError("E8 requires a TwistAffordance.")
        geometry = request.goal.semantics.geometry
        point = tuple(
            p + shift * axis
            for p, axis in zip(
                geometry.get(
                    "gen_sim_twist_grasp_point",
                    geometry["gen_sim_twist_outer_point"],
                ),
                geometry["gen_sim_twist_axis"],
            )
        )
        goal = replace(
            request.goal,
            semantics=replace(
                request.goal.semantics,
                affordance=replace(
                    affordance,
                    grasp_position=point,
                    grasp_roll=roll,
                    grasp_roll_range=None,
                ),
            ),
        )
        return replace(request, goal=goal)

    def _candidate_geometry_score(
        self, plan: Any, request: Any, context: Any
    ) -> tuple[float, float, float, float]:
        """Rank successful IK by parent clearance, pad contact, then motion."""
        geometry = request.goal.semantics.geometry
        target_points = geometry.get("gen_sim_twist_target_points")
        parent_points = geometry.get("gen_sim_twist_parent_points")
        target_to_parent = geometry.get("gen_sim_twist_target_to_parent")
        if not target_points or not parent_points or target_to_parent is None:
            return (1.0, 1.0, 1.0, float("inf"))
        close = next((s for s in plan.segments if s.name == "close"), None)
        if close is None or plan.joint_trajectory is None:
            return (1.0, 1.0, 1.0, float("inf"))
        qpos = plan.joint_trajectory.positions[:, close.stop - 1]
        target_pose = resolve_pose_target(
            resolve_pose_goal(request.goal.target_pose, context, name="target_pose"),
            num_envs=1,
            device=qpos.device,
        )[0]
        target = qpos.new_tensor(target_points)
        parent = qpos.new_tensor(parent_points)
        target_world = target @ target_pose[:3, :3].T + target_pose[:3, 3]
        parent_pose = target_pose @ qpos.new_tensor(target_to_parent)
        parent_world = parent @ parent_pose[:3, :3].T + parent_pose[:3, 3]
        names = [
            name
            for name in self.robot.link_names
            if name.startswith((geometry.get("gen_sim_twist_arm", "right") + "_"))
            if "finger_pad" in name
        ]
        if not names:
            return (1.0, 1.0, 1.0, float("inf"))
        pads = self.robot.compute_fk(
            qpos=qpos,
            link_names=names,
            qpos_joint_names=self.robot.joint_names,
        )
        # Full-articulation FK is root-local; the target snapshot is arena-local.
        pads = self.robot.get_local_pose(to_matrix=True).to(pads)[:, None] @ pads
        distances = []
        parent_close = []
        for index, name in enumerate(names):
            vertices, _ = self.robot.get_link_vert_face(name)
            pose = pads[0, index]
            world = vertices.to(pose) @ pose[:3, :3].T + pose[:3, 3]
            distances.append(torch.cdist(world, target_world).amin())
            parent_close.append(torch.cdist(world, parent_world).amin(-1))
        dual_pad_distance = float(torch.stack(distances).amax())
        parent_fraction = float(torch.cat(parent_close).le(0.003).float().mean())
        motion = float(torch.linalg.vector_norm(qpos - context.robot.qpos).item())
        parent_violation = float(parent_fraction > PARENT_CONTACT_SCORE_LIMIT)
        return parent_violation, parent_fraction, dual_pad_distance, motion

    @staticmethod
    def _annotate_plan(plan: Any, *, roll: float, candidate_index: int) -> Any:
        diagnostics = plan.diagnostics
        if diagnostics is None:
            return plan
        return replace(
            plan,
            diagnostics=replace(
                diagnostics,
                metadata={
                    **diagnostics.metadata,
                    "gen_sim_twist_roll": roll,
                    "gen_sim_twist_roll_candidate": candidate_index,
                },
            ),
        )

    def _plan(self, request: Any, context: Any) -> Any:
        from .motion import _joint_velocity_limits, _velocity_validity

        geometry = request.goal.semantics.geometry
        angle = geometry["gen_sim_twist_angle"]
        current_qpos = self._current_qpos(request, context)
        if current_qpos is not None:
            target_qpos = geometry["gen_sim_twist_target_qpos"]
            axis_sign = geometry["gen_sim_twist_axis_sign"]
            angle = (target_qpos - current_qpos) / axis_sign
        chunk_angle = geometry.get("gen_sim_twist_chunk_angle", math.pi)
        if abs(angle) > chunk_angle:
            angle = math.copysign(chunk_angle, angle)
        if type(angle) is not float or not math.isfinite(angle) or abs(angle) > math.pi:
            raise ValueError(
                "E8 requires a finite calibrated turn of at most pi radians."
            )
        base_request = replace(
            request, skill_options=replace(request.skill_options, twist_angle=angle)
        )
        state = _candidate_state()
        if state.goal_complete:
            if (
                current_qpos is not None
                and abs(current_qpos - geometry["gen_sim_twist_target_qpos"])
                <= ANGLE_TOLERANCE
            ):
                return self.build_plan(
                    request,
                    context,
                    success=torch.ones(
                        context.batch_size, dtype=torch.bool, device=self.device
                    ),
                    trajectory=TimedTrajectory.from_uniform_step(
                        context.robot.qpos.unsqueeze(1).clone(),
                        env_ids=context.env_ids,
                        step_dt=context.require_control_dt(),
                    ),
                    expected_effects=StateDelta(),
                    diagnostics=None,
                    segment_lengths={"approach": 1},
                )
            state.goal_complete = False
        table = _candidate_table(geometry["gen_sim_twist_grip_depth"])
        candidates = (
            [item for item in table if item[0] == state.forced_index]
            if state.forced_index is not None
            else list(table)
        )
        ranked: list[tuple[tuple[float, float, float, float], int, Any, Any]] = []
        for candidate_index, roll, shift in candidates:
            candidate_request = self._candidate_request(base_request, roll, shift)
            candidate_plan = super()._plan(candidate_request, context)
            candidate_plan = self._annotate_plan(
                candidate_plan, roll=roll, candidate_index=candidate_index
            )
            if candidate_plan.plan_success.any():
                ranked.append(
                    (
                        self._candidate_geometry_score(
                            candidate_plan, candidate_request, context
                        ),
                        candidate_index,
                        candidate_plan,
                        candidate_request,
                    )
                )
        plan = None
        planned_request = base_request
        if ranked:
            ranked.sort(key=lambda item: (item[0], item[1]))
            selected_score, selected, plan, planned_request = ranked[0]
            state.selected_index = selected
            if state.forced_index is None:
                state.order = tuple(item[1] for item in ranked)
            plan = replace(
                plan,
                diagnostics=replace(
                    plan.diagnostics,
                    metadata={
                        **plan.diagnostics.metadata,
                        "gen_sim_twist_candidate_order": list(state.order),
                        "gen_sim_twist_candidate_score": list(selected_score),
                        "gen_sim_twist_candidate_fallback_limit": TWIST_CANDIDATE_FALLBACKS,
                        "gen_sim_twist_parent_contact_score_limit": PARENT_CONTACT_SCORE_LIMIT,
                    },
                ),
            )
        else:
            if candidates:
                selected, _, _ = candidates[0]
                state.selected_index = selected
            plan = super()._plan(base_request, context)
        request = planned_request
        if not plan.plan_success.any():
            logger.log_warning(f"E8 public Twist planning rejected: {plan.diagnostics}")
        if plan.plan_success.any():
            settle = request.goal.semantics.geometry["gen_sim_twist_settle_steps"]
            if settle:
                trajectory = plan.joint_trajectory
                pieces, lengths = [], {}
                for segment in plan.segments:
                    piece = trajectory.positions[:, segment.start : segment.stop]
                    if segment.name in {"close", "twist"}:
                        piece = torch.cat(
                            (piece, piece[:, -1:].expand(-1, settle, -1)), dim=1
                        )
                    pieces.append(piece)
                    lengths[segment.name] = piece.shape[1]
                plan = self.build_plan(
                    request,
                    context,
                    success=plan.plan_success,
                    trajectory=TimedTrajectory.from_uniform_step(
                        torch.cat(pieces, dim=1),
                        env_ids=trajectory.env_ids,
                        step_dt=(
                            float(trajectory.dt[0, 1].item())
                            if trajectory.dt.shape[1] > 1
                            else context.require_control_dt()
                        ),
                    ),
                    expected_effects=plan.expected_effects,
                    diagnostics=plan.diagnostics,
                    segment_lengths=lengths,
                )
            trajectory = plan.joint_trajectory
            limits = _joint_velocity_limits(self.robot, None).to(trajectory.positions)
            if not _velocity_validity(
                trajectory.positions, trajectory.dt, limits
            ).all():
                logger.log_warning(
                    "E8 public Twist trajectory exceeds joint velocity limits."
                )
                return self.failed_plan(
                    request,
                    context,
                    message="E8 Twist exceeds resampled joint velocity limits.",
                )
            # Keep free-space invalidation, then permit the intended contact
            # rotation. The next chunk lowers from a fresh physical observation.
            if plan.scene_dependencies:
                plan = replace(plan, scene_dependency_end_segment="reach")
        return plan


def _grasp_tip_offset(
    robot: Any, context: Any, bound: Any, width: float
) -> tuple[float, JointPositionCommand]:
    """Match the measured aperture and return its tip offset and grasp command."""
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("E8 knob width must be finite and positive.")
    motion = bound.binding.action_binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    grasp = bound.binding.action_binding.endpoint("primary", "grasp")
    hand = grasp.require_target(JointPositionTarget)
    prefix = motion.control_part.removesuffix("_arm") + "_"
    names = [
        n
        for n in robot.link_names
        if n.startswith(prefix) and "finger" in n and "pad" in n
    ]
    if len(names) != 2:
        names = [
            n
            for n in robot.link_names
            if n.startswith(prefix)
            and "finger" in n
            and "knuckle" not in n
            and not n.endswith("_joint")
        ]
    if len(names) != 2:
        raise ValueError("E8 aperture calibration requires two explicit finger pads.")
    solver = robot.cfg.solver_cfg[motion.control_part]
    qpos = context.robot.qpos.clone()
    commands = [
        grasp.joint_positions(
            c, num_envs=context.batch_size, device=qpos.device, dtype=qpos.dtype
        )
        for c in (OPEN_COMMAND, GRASP_COMMAND)
    ]
    vertices = [robot.get_link_vert_face(n)[0] for n in names]

    def measure(fraction: float) -> tuple[float, float]:
        qpos[:, list(hand.joint_ids)] = commands[0] + fraction * (
            commands[1] - commands[0]
        )
        poses = robot.compute_fk(
            qpos=qpos,
            link_names=[solver.end_link_name, *names],
            qpos_joint_names=robot.joint_names,
        )
        inverse = torch.linalg.inv(poses[0, 0] @ poses.new_tensor(solver.tcp))
        clouds = []
        for i, v in enumerate(vertices, start=1):
            t = inverse @ poses[0, i]
            clouds.append(v.to(t) @ t[:3, :3].T + t[:3, 3])
        clouds.sort(key=lambda p: float(p[:, 0].mean()))
        return float(clouds[1][:, 0].min() - clouds[0][:, 0].max()), max(
            float(p[:, 2].max()) for p in clouds
        )

    open_gap, closed_gap = measure(0.0)[0], measure(1.0)[0]
    if math.isclose(width, open_gap):
        width = open_gap
    elif math.isclose(width, closed_gap):
        width = closed_gap
    if (
        not math.isfinite(open_gap)
        or not math.isfinite(closed_gap)
        or open_gap <= closed_gap
        or not closed_gap <= width <= open_gap
    ):
        raise ValueError("E8 knob width is outside the measured gripper opening.")
    low, high = 0.0, 1.0
    for _ in range(12):
        mid = (low + high) / 2
        if measure(mid)[0] > width:
            low = mid
        else:
            high = mid
    # The closed-side bound adds only the bisection's numerical contact margin.
    offset = measure(high)[1]
    command = JointPositionCommand(commands[0] + high * (commands[1] - commands[0]))
    return offset, command


def with_twist_options(profile: Any, routes: tuple[TwistRoute, ...]) -> Any:
    if not routes:
        return profile
    (route,) = routes
    return replace(
        profile,
        presets=tuple(
            SkillPolicyPreset(
                p.preset_id,
                required_planner=p.required_planner,
                action_option_templates={
                    **p.action_option_templates,
                    PREPARE_CALL: MoveJointsOptions(),
                    TWIST_CALL: TwistOptions(
                        hand_interp_steps=12,
                        pre_grasp_distance=0.10,
                        twist_angle=route.binding.target_qpos / route.binding.axis_sign,
                    ),
                },
                motion_policy=replace(
                    p.motion_policy,
                    strategy="ik_interp",
                    sample_count=max(320, p.motion_policy.sample_count),
                ),
                recovery_policy=RecoveryPolicy(max_replans=0, max_action_retries=0),
                workflow_recovery_policy=replace(
                    p.workflow_recovery_policy, max_recovery_attempts=0
                ),
                effect_assurance=p.effect_assurance,
                effect_monitors=p.effect_monitors,
                tracking_policy=p.tracking_policy,
                runner_cfg=p.runner_cfg,
            )
            for p in profile.presets
        ),
    )


def _bounded_twist_qpos(
    observed: torch.Tensor, limits: tuple[float, float]
) -> float | None:
    """Accept native-dtype boundary roundoff without accepting physical overtravel."""
    value = float(observed)
    epsilon = torch.finfo(observed.dtype).eps * max(1.0, *(abs(v) for v in limits))
    if (
        not math.isfinite(value)
        or not limits[0] - epsilon <= value <= limits[1] + epsilon
    ):
        return None
    return min(limits[1], max(limits[0], value))


class _TwistLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = TWIST_CALL
    target_descriptor = Twist.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(
        self,
        route: TwistRoute,
        simulation: Any,
        robot: Any,
        registry: Any,
    ) -> None:
        self.route, self.robot = route, robot
        b = route.binding
        entry = registry.lookup(route.link_id, expected_type=SceneLinkRef)
        if (
            entry.parent != SceneArticulationRef(b.object_id)
            or entry.native_name != b.link
        ):
            raise ValueError("E8 joint/link ownership changed.")
        self.art = simulation.get_articulation(b.object_id)
        actual = discover_twist(
            {
                "uid": self.art.uid,
                "fpath": self.art.cfg.fpath,
                "body_scale": self.art.cfg.body_scale,
                "fix_base": self.art.cfg.root_props.fixed_base,
            },
            b.target_setting,
        )
        if actual.binding != b or simulation.num_envs != 1:
            raise ValueError(
                "E8 requires one environment and unchanged source-qualified geometry."
            )
        self.joint_index = self.art.joint_names.index(b.joint)

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        b = self.route.binding
        if dict(call.arguments) != {"object": b.object_id, "setting": b.target_setting}:
            raise ValueError("E8 must retain its exact target and setting.")
        endpoint = bound.binding.action_binding.endpoint("primary", "motion")
        if (
            context.task.get_held_object(endpoint.task_state_key) is not None
            or context.task.coordinated_held_objects
        ):
            raise ValueError("E8 requires an empty hand.")
        target = endpoint.require_target(JointPositionTarget)
        if self.call_id == PREPARE_CALL:
            return SemanticLowering(
                goal=JointPositionGoal(
                    context.robot.qpos[:, list(target.joint_ids)].clone()
                )
            )
        if type(option_template) is not TwistOptions:
            raise TypeError("E8 requires public TwistOptions.")
        observed = self.art.get_qpos()[0, self.joint_index]
        qpos = _bounded_twist_qpos(observed, b.limits)
        if qpos is None:
            raise ValueError("E8 calibrated joint qpos is not finite or within limits.")
        rest = float(self.art.get_link_physical_attr(b.link)[0].rest_offset)
        pad_names = [
            n
            for n in self.robot.link_names
            if n.startswith(self.route.arm + "_") and "finger_pad" in n
        ]
        pad_rest = (
            max(
                float(v.rest_offset)
                for v in self.robot.get_link_physical_attr(pad_names)
            )
            if pad_names
            else 0.0
        )
        offset, grasp_command = _grasp_tip_offset(
            self.robot, context, bound, b.grip_width + 2 * (rest + pad_rest)
        )
        if self.route.variant == "closed_tip":
            offset = _closed_finger_tip_offset(self.robot, context, bound)
        bite = min(self.route.grasp_depth, b.grip_depth / 2)
        shift = b.grip_depth / 2 if self.route.variant == "centroid" else bite - offset
        point = tuple(p + shift * a for p, a in zip(b.outer_point, b.axis))
        angle = (
            0.0
            if self.route.variant == "no_twist"
            else (b.target_qpos - qpos) / b.axis_sign
        )
        if abs(angle) > self.route.chunk_angle:
            angle = math.copysign(self.route.chunk_angle, angle)
        logger.log_info(
            f"E8 target={b.target_qpos:.6f} rad, remaining={angle:.6f}, finger_offset={offset:.6f}, bite={bite:.6f}"
        )
        target_pose = self.art.get_link_pose(b.link, to_matrix=True)[0]
        parent_pose = self.art.get_link_pose(b.parent, to_matrix=True)[0]
        target_vertices, _ = self.art.get_link_vert_face(b.link)
        parent_vertices, _ = self.art.get_link_vert_face(b.parent)

        def sample(vertices: torch.Tensor, limit: int = 128) -> list[list[float]]:
            if vertices.shape[0] <= limit:
                return vertices.detach().cpu().tolist()
            indices = (
                torch.linspace(0, vertices.shape[0] - 1, limit, device=vertices.device)
                .round()
                .long()
            )
            return vertices[indices].detach().cpu().tolist()

        target_to_parent = torch.linalg.inv(target_pose) @ parent_pose
        return SemanticLowering(
            goal=TwistGoal(
                ObjectSemantics(
                    entity_id=self.route.link_id,
                    geometry={
                        "gen_sim_twist_angle": angle,
                        "gen_sim_twist_measured_qpos": qpos,
                        "gen_sim_twist_target_qpos": b.target_qpos,
                        "gen_sim_twist_axis_sign": b.axis_sign,
                        "gen_sim_twist_articulation_id": b.object_id,
                        "gen_sim_twist_joint": b.joint,
                        "gen_sim_twist_chunk_angle": self.route.chunk_angle,
                        "gen_sim_twist_settle_steps": self.route.settle_steps,
                        "gen_sim_twist_outer_point": list(b.outer_point),
                        "gen_sim_twist_grip_depth": b.grip_depth,
                        "gen_sim_twist_grasp_point": list(point),
                        "gen_sim_twist_axis": list(b.axis),
                        "gen_sim_twist_arm": self.route.arm,
                        "gen_sim_twist_target_points": sample(target_vertices),
                        "gen_sim_twist_parent_points": sample(parent_vertices),
                        "gen_sim_twist_target_to_parent": target_to_parent.detach()
                        .cpu()
                        .tolist(),
                    },
                    affordance=TwistAffordance(
                        grasp_position=point,
                        axis_origin=b.origin,
                        twist_axis=torch.tensor(b.axis),
                        joint_name=b.joint,
                        joint_limits=b.limits,
                    ),
                ),
                SceneEntityPose(self.route.link_id),
            ),
            control_overrides=ActionControlOverrides(
                endpoints={"primary": {"grasp": {GRASP_COMMAND: grasp_command}}}
            ),
        )


class _TwistPrepareLowerer(_TwistLowerer):
    call_id: ClassVar[str] = PREPARE_CALL
    target_descriptor = MoveJoints.descriptor()


@dataclass(frozen=True, slots=True)
class TwistPrepareFactory(RegisteredSemanticLowererFactory):
    route: TwistRoute
    call_id: ClassVar[str] = PREPARE_CALL
    revision: ClassVar[str] = TWIST_REVISION
    target_descriptor = MoveJoints.descriptor()

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> Any:
        if engine.robot is not robot:
            raise ValueError("E8 must use the integration-owned robot.")
        lowerer = (
            _TwistPrepareLowerer if self.call_id == PREPARE_CALL else _TwistLowerer
        )
        return lowerer(self.route, simulation, robot, scene_registry)


@dataclass(frozen=True, slots=True)
class TwistFactory(TwistPrepareFactory):
    call_id: ClassVar[str] = TWIST_CALL
    target_descriptor = Twist.descriptor()


class TwistContactSensor(PressContactSensor):
    """Reuse the read-only substep collector; qpos values here are radians."""

    def configure(self, route: TwistRoute, robot: Any) -> None:
        super().configure(route, robot)
        self.table_actor = int(self.get_actor_ids("table")[0, 0])
        self.robot_actors = set(
            self.get_actor_ids(robot.uid, robot.link_names)[0].tolist()
        )
        dt = float(self._sim.sim_config.physics_dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("E8 startup observation requires a positive physics dt.")
        self._startup_sample_limit = min(
            STARTUP_MAX_SAMPLES, math.ceil(STARTUP_MAX_DURATION / dt) + 1
        )
        self._startup_epoch = -1
        self._finalized_startup: dict[str, Any] | None = None
        self.update()
        self._begin_startup()

    def _begin_startup(self) -> None:
        self.clock = 0.0
        self._startup_epoch += 1
        self._startup_recording = True
        self._startup_overflow = False
        self.startup_trace: list[dict[str, Any]] = []
        self._capture_startup()

    def _capture_startup(self) -> None:
        if (
            len(self.startup_trace) >= self._startup_sample_limit
            or self.clock > STARTUP_MAX_DURATION + 1e-8
        ):
            self._startup_overflow = True
            return
        data = self.get_data()
        mask = data["is_valid"][0]
        pairs = data["user_ids"][0, mask].tolist()
        impulses = data["impulse"][0, mask].tolist()
        qpos = float(self.art.get_qpos()[0, self.joint_index])
        qvel = float(self.art.get_qvel()[0, self.joint_index])
        valid = (
            self.dropped_contacts == 0
            and all(math.isfinite(value) for value in (self.clock, qpos, qvel))
            and all(
                bool(torch.isfinite(data[key][0, mask]).all())
                for key in ("impulse", "position", "normal", "distance")
            )
        )
        self.startup_trace.append(
            {
                "timestamp": self.clock,
                "qpos": qpos,
                "qvel": qvel,
                "contact_pairs": pairs,
                "impulses": impulses,
                "table_target_contact": any(
                    self.target_actor in pair
                    and self.table_actor in pair
                    and impulse > 0
                    for pair, impulse in zip(pairs, impulses)
                ),
                "robot_target_contact": any(
                    self.target_actor in pair
                    and any(actor in self.robot_actors for actor in pair)
                    and impulse > 0
                    for pair, impulse in zip(pairs, impulses)
                ),
                "valid": valid,
                "dropped_contacts": self.dropped_contacts,
            }
        )

    def update_physics_step(self, dt: float) -> None:
        super().update_physics_step(dt)
        if self._startup_recording and not self._startup_overflow:
            self._capture_startup()

    def _startup_record(self, *, complete: bool) -> dict[str, Any]:
        record = {
            "scope": "reset_to_ready_physics_substeps",
            "epoch": self._startup_epoch,
            "complete": complete,
            "overflow": self._startup_overflow,
            "sample_limit": self._startup_sample_limit,
            "duration_limit": STARTUP_MAX_DURATION,
            "table_actor_id": self.table_actor,
            "robot_actor_ids": sorted(self.robot_actors),
            "trace": deepcopy(self.startup_trace),
        }
        record["summary"] = _evaluate_twist_startup(record)
        return record

    def startup_evidence(self) -> dict[str, Any]:
        """Retain the completed attempt even if error cleanup resets the scene."""
        if self._finalized_startup is not None:
            return deepcopy(self._finalized_startup)
        return self._startup_record(complete=False)

    def startup_check(self) -> dict[str, Any]:
        return self.startup_evidence()["summary"]

    def arm(self) -> None:
        self._startup_recording = False
        self._finalized_startup = self._startup_record(complete=True)
        super().arm()
        self.phase = "twist"

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        if hasattr(self, "route"):
            if self._startup_recording and len(self.startup_trace) > 1:
                self._finalized_startup = self._startup_record(complete=False)
            self._begin_startup()

    def capture(self, phase: str) -> Any:
        sample = super().capture(phase)
        if len(self.trace) % 20 == 0:
            part = self.route.arm + "_eef"
            self.trace[-1]["hand_qpos"] = self.robot.get_qpos(part).tolist()
            self.trace[-1]["hand_target"] = self.robot.get_qpos(
                part, target=True
            ).tolist()
        return sample


def _evaluate_twist_startup(evidence: dict[str, Any]) -> dict[str, Any]:
    trace = evidence["trace"]
    valid = bool(
        evidence["complete"]
        and not evidence["overflow"]
        and len(trace) >= 2
        and all(frame["valid"] and frame["dropped_contacts"] == 0 for frame in trace)
        and all(
            math.isfinite(frame[key])
            for frame in trace
            for key in ("timestamp", "qpos", "qvel")
        )
        and all(
            current["timestamp"] > previous["timestamp"]
            for previous, current in zip(trace, trace[1:])
        )
    )
    table_contacts = sum(frame["table_target_contact"] for frame in trace)
    robot_contacts = sum(frame["robot_target_contact"] for frame in trace)
    motion_stable = _initial_twist_state_stable(
        [frame["qpos"] for frame in trace], [frame["qvel"] for frame in trace]
    )
    reason = (
        "invalid_startup_evidence"
        if not valid
        else (
            "startup_table_contact"
            if table_contacts
            else (
                "startup_robot_contact"
                if robot_contacts
                else ("startup_stable" if motion_stable else "startup_motion")
            )
        )
    )
    return {
        "accepted": valid
        and not table_contacts
        and not robot_contacts
        and motion_stable,
        "reason": reason,
        "sample_count": len(trace),
        "table_contact_samples": table_contacts,
        "robot_contact_samples": robot_contacts,
        "max_abs_qpos": max((abs(frame["qpos"]) for frame in trace), default=None),
        "max_abs_qvel": max((abs(frame["qvel"]) for frame in trace), default=None),
        "initial_angle_tolerance": INITIAL_TOLERANCE,
        "initial_velocity_tolerance": INITIAL_VELOCITY_TOLERANCE,
    }


def ensure_sensor(simulation: Any, robot: Any, route: TwistRoute) -> TwistContactSensor:
    sensor = simulation.get_sensor(SENSOR_UID)
    if sensor is not None:
        if not isinstance(sensor, TwistContactSensor) or sensor.route != route:
            raise ValueError("E8 contact sensor identity conflicts.")
        return sensor
    factories = simulation.SUPPORTED_SENSOR_TYPES
    simulation.SUPPORTED_SENSOR_TYPES = {
        **factories,
        "GenSimTwistContact": TwistContactSensor,
    }
    try:
        sensor = simulation.add_sensor(
            ContactSensorCfg(
                uid=SENSOR_UID,
                sensor_type="GenSimTwistContact",
                max_contacts_per_env=512,
                filter_need_both_actor=False,
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(articulation_uid=robot.uid),
                    ArticulationContactFilterCfg(
                        articulation_uid=route.binding.object_id
                    ),
                ],
            )
        )
    finally:
        simulation.SUPPORTED_SENSOR_TYPES = factories
    sensor.configure(route, robot)
    return sensor


def evaluate_twist(route: TwistRoute, samples: list, *, final_qpos: float) -> dict:
    b = route.binding
    result = {
        "accepted": False,
        "criterion": "qpos_converged_with_contact",
        "target_qpos": b.target_qpos,
        "observed_qpos": final_qpos,
        "angle_tolerance": ANGLE_TOLERANCE,
        "coarse_turn_threshold": COARSE_TURN_THRESHOLD,
        "coarse_turn_reached": False,
        "target_contact_seen": False,
        "reason": "invalid_observation",
    }
    if (
        not samples
        or any(
            not s.valid
            or not math.isfinite(s.qpos)
            or not math.isfinite(s.timestamp)
            or s.timestamp < 0
            or type(s.target_contact) is not bool
            for s in samples
        )
        or any(b.timestamp <= a.timestamp for a, b in zip(samples, samples[1:]))
    ):
        return result
    if abs(samples[0].qpos) > INITIAL_TOLERANCE or samples[0].target_contact:
        return {**result, "reason": "unstable_initial_state"}
    if any(not b.limits[0] - 0.02 <= s.qpos <= b.limits[1] + 0.02 for s in samples):
        return {**result, "reason": "joint_out_of_bounds"}
    sign = math.copysign(1.0, b.target_qpos - samples[0].qpos)
    signed_contact_travel, uncontacted_reverse_travel, contact_path = 0.0, 0.0, 0.0
    for previous, current in zip(samples, samples[1:]):
        delta = sign * (current.qpos - previous.qpos)
        if previous.target_contact and current.target_contact:
            signed_contact_travel += delta
            contact_path += abs(delta)
        elif delta < 0.0:
            # Release/regrasp reversals must not let repeated excursions earn credit.
            uncontacted_reverse_travel -= delta
    directional_travel = max(0.0, signed_contact_travel - uncontacted_reverse_travel)
    error = abs(final_qpos - b.target_qpos)
    target_contact_seen = any(s.target_contact for s in samples)
    required_directional_travel = min(
        COARSE_TURN_THRESHOLD, abs(b.target_qpos - samples[0].qpos)
    )
    accepted = (
        math.isfinite(final_qpos)
        and target_contact_seen
        and directional_travel >= required_directional_travel
        and error <= ANGLE_TOLERANCE
    )
    return {
        **result,
        "accepted": accepted,
        "contact_travel": directional_travel,
        "signed_contact_travel": signed_contact_travel,
        "uncontacted_reverse_travel": uncontacted_reverse_travel,
        "contact_path": contact_path,
        "target_contact_seen": target_contact_seen,
        "directional_travel_required": required_directional_travel,
        "target_error": error,
        "coarse_turn_reached": contact_path >= COARSE_TURN_THRESHOLD,
        "reason": (
            "qpos_converged"
            if accepted
            else (
                "target_contact_missing"
                if not target_contact_seen
                else (
                    "target_qpos_not_reached"
                    if error > ANGLE_TOLERANCE
                    else "direction_not_confirmed"
                )
            )
        ),
    }


def _initial_twist_state_stable(
    positions: list[float], velocities: list[float]
) -> bool:
    """Reject a stationary-looking joint with unresolved angular velocity."""
    return bool(
        positions
        and len(positions) == len(velocities)
        and all(math.isfinite(q) and abs(q) <= INITIAL_TOLERANCE for q in positions)
        and all(
            math.isfinite(v) and abs(v) <= INITIAL_VELOCITY_TOLERANCE
            for v in velocities
        )
    )


class TwistAcceptancePort:
    """Observe initial stability, contact rotation, released setting and Park."""

    def __init__(
        self,
        delegate: Any,
        simulation: Any,
        robot: Any,
        route: TwistRoute,
        step_dt: float,
    ) -> None:
        self.delegate, self.robot, self.route, self.dt = delegate, robot, route, step_dt
        self.sensor = ensure_sensor(simulation, robot, route)
        self.results, self.metadata = {}, {}
        self._chunk_cursor = 0
        self._candidate_state = _TwistCandidateState()
        _CANDIDATE_STATE.set(self._candidate_state)

    def validate_policy(self, policy: Any, *, segment: Any) -> None:
        if policy.cfg.preset not in {
            self.route.preset("ready"),
            self.route.preset("chunk"),
            self.route.preset("turned"),
        }:
            return self.delegate.validate_policy(policy, segment=segment)
        if (
            policy.cfg.kind != "wait_stable"
            or policy.entity.entity_id != self.route.binding.object_id
        ):
            raise ValueError("E8 physical policy identity differs from its knob.")

    def _accumulate_contact_path(self, samples: list) -> None:
        """Accumulate debounced target-contact qpos travel across chunks."""
        state = self._candidate_state
        for sample in samples:
            if sample.target_contact is not True:
                state.last_contact_qpos = None
                continue
            if state.last_contact_qpos is not None:
                delta = abs(sample.qpos - state.last_contact_qpos)
                if delta >= QPOS_PATH_DEADBAND:
                    state.contact_path += delta
                    state.last_contact_qpos = sample.qpos
            else:
                state.last_contact_qpos = sample.qpos

    def _clearance(self) -> float:
        bounds = []
        for name in self.sensor.finger_names:
            pose = self.robot.get_link_pose(name, to_matrix=True)[0]
            vertices, _ = self.robot.get_link_vert_face(name)
            world = vertices.to(pose) @ pose[:3, :3].T + pose[:3, 3]
            bounds.append(torch.stack((world.amin(0), world.amax(0))))
        return _park_clearance(self.sensor.art, bounds)

    def actions(self, policy: Any, *, segment: Any, active_mask: torch.Tensor) -> Any:
        self.validate_policy(policy, segment=segment)
        ready = policy.cfg.preset == self.route.preset("ready")
        chunk = policy.cfg.preset == self.route.preset("chunk")
        if not ready and not chunk and policy.cfg.preset != self.route.preset("turned"):
            yield from self.delegate.actions(
                policy, segment=segment, active_mask=active_mask
            )
            return
        s, b = self.sensor, self.route.binding
        hold = self.robot.get_qpos().clone()
        values, velocities = [], []
        for _ in range(math.ceil(0.5 / self.dt)):
            yield hold.clone()
            values.append(float(s.art.get_qpos()[0, s.joint_index]))
            velocities.append(float(s.art.get_qvel()[0, s.joint_index]))
        stable = bool(
            max(values) - min(values) <= math.radians(1.0)
            and all(
                math.isfinite(v) and abs(v) <= INITIAL_VELOCITY_TOLERANCE
                for v in velocities
            )
        )
        recent = s.samples[-max(1, math.ceil(0.15 / s._sim.sim_config.physics_dt)) :]
        released = bool(recent) and all(
            sample.valid and sample.target_contact is False for sample in recent
        )
        if ready:
            self._candidate_state.forced_index = None
            self._candidate_state.order = ()
            self._candidate_state.selected_index = None
            self._candidate_state.fallback_count = 0
            self._candidate_state.locked = False
            self._candidate_state.contact_path = 0.0
            self._candidate_state.last_contact_qpos = None
            self._candidate_state.goal_complete = False
            s.update()
            s.arm()
            s.other_qpos = s.art.get_qpos().clone()
            self._chunk_cursor = len(s.samples)
            startup = s.startup_check()
            good = (
                startup["accepted"]
                and _initial_twist_state_stable(values, velocities)
                and s.samples[0].valid
                and s.samples[0].target_contact is False
            )
            result = {
                "accepted": good,
                "phase": "initial_stable",
                "reason": (
                    startup["reason"]
                    if not startup["accepted"]
                    else ("initial_stable" if good else "unstable_initial_state")
                ),
                "joint_positions": values,
                "joint_velocities": velocities,
                "initial_velocity_tolerance": INITIAL_VELOCITY_TOLERANCE,
                "startup_check": startup,
            }
        elif chunk:
            start = self._chunk_cursor
            samples = s.samples[start:]
            trace = s.trace[start:]
            self._accumulate_contact_path(samples)
            qpos = float(s.art.get_qpos()[0, s.joint_index])
            target_contacts = [sample.target_contact for sample in samples]
            parent_contacts = [bool(item.get("parent_contact")) for item in trace]
            parent_fraction = (
                sum(parent_contacts) / len(parent_contacts) if parent_contacts else 1.0
            )
            target_error = abs(qpos - b.target_qpos)
            already_reached = target_error <= ANGLE_TOLERANCE
            coarse_reached = self._candidate_state.contact_path >= COARSE_TURN_THRESHOLD
            valid_chunk = bool(
                samples
                and all(sample.valid for sample in samples)
                and all(sample.target_contact is not None for sample in samples)
                and math.isfinite(qpos)
                and b.limits[0] - 0.02 <= qpos <= b.limits[1] + 0.02
            )
            good, fallback = False, False
            if valid_chunk:
                good, fallback = _candidate_contact_decision(
                    self._candidate_state,
                    target_contact=any(target_contacts),
                    already_reached=already_reached,
                )
            goal_check = evaluate_twist(self.route, s.samples, final_qpos=qpos)
            self._candidate_state.goal_complete = bool(
                goal_check["accepted"]
                and stable
                and released
                and self._clearance() >= 0.04
            )
            result = {
                "accepted": good,
                "phase": "bounded_chunk",
                "sample_count": len(samples),
                "target_contact": any(target_contacts),
                "parent_contact_fraction": parent_fraction,
                "observed_qpos": qpos,
                "target_error": target_error,
                "already_reached": already_reached,
                "parent_contact_fraction_limit": PARENT_CONTACT_FRACTION_LIMIT,
                "parent_contact_warning": parent_fraction
                > PARENT_CONTACT_FRACTION_LIMIT,
                "candidate_selected": self._candidate_state.selected_index,
                "candidate_fallback": fallback,
                "candidate_fallback_count": self._candidate_state.fallback_count,
                "candidate_fallback_limit": TWIST_CANDIDATE_FALLBACKS,
                "coarse_turn_threshold": COARSE_TURN_THRESHOLD,
                "cumulative_contact_path": self._candidate_state.contact_path,
                "coarse_turn_reached": coarse_reached,
                "early_stop": False,
                "goal_complete": self._candidate_state.goal_complete,
            }
            self._chunk_cursor = len(s.samples)
        else:
            result = evaluate_twist(self.route, s.samples, final_qpos=values[-1])
            clearance = self._clearance()
            qpos = s.art.get_qpos().clone()
            qpos[:, s.joint_index] = s.other_qpos[:, s.joint_index]
            others_unchanged = bool(
                torch.isfinite(qpos).all()
                and torch.allclose(qpos, s.other_qpos, atol=INITIAL_TOLERANCE, rtol=0)
            )
            good = bool(
                result["accepted"]
                and stable
                and released
                and clearance >= 0.04
                and others_unchanged
            )
            parent_contacts = [bool(item.get("parent_contact")) for item in s.trace]
            parent_fraction = (
                sum(parent_contacts) / len(parent_contacts) if parent_contacts else 1.0
            )
            result.update(
                accepted=good,
                contact_released=released,
                stable=stable,
                joint_velocities=velocities,
                clearance=clearance,
                others_unchanged=others_unchanged,
                parent_contact_fraction=parent_fraction,
                parent_contact_fraction_limit=PARENT_CONTACT_FRACTION_LIMIT,
                parent_contact_warning=parent_fraction > PARENT_CONTACT_FRACTION_LIMIT,
                terminal=any(c.call.semantic_id == PARK_CALL for c in segment.calls),
                coarse_turn_threshold=COARSE_TURN_THRESHOLD,
                cumulative_contact_path=self._candidate_state.contact_path,
                coarse_turn_reached=(
                    self._candidate_state.contact_path >= COARSE_TURN_THRESHOLD
                ),
                early_stop=False,
            )
            s.phase = "cleanup"
        s.acceptance = deepcopy(result)
        logger.log_info(f"E8 acceptance: {result}")
        s.armed = good and not result.get("terminal", False)
        self.results[id(policy)] = torch.full_like(active_mask, good) & active_mask
        self.metadata[id(policy)] = result

    def post_policy_result(self, policy: Any, *, segment: Any) -> Any:
        return (
            self.results[id(policy)].clone()
            if id(policy) in self.results
            else self.delegate.post_policy_result(policy, segment=segment)
        )

    def post_policy_metadata(self, policy: Any, *, segment: Any) -> Any:
        return (
            deepcopy(self.metadata[id(policy)])
            if id(policy) in self.metadata
            else self.delegate.post_policy_metadata(policy, segment=segment)
        )
