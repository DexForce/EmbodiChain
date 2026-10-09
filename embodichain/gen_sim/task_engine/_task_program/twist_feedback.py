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
"""Explicitly opted-in, bounded E8 native position feedback state."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
import time
from typing import Any, Callable

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    JointPositionTarget,
)
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    JointPositionPayload,
    RuntimeCommandFrame,
)

from .twist_geometry_guard import _matrix
from .twist_session import DueObservationRequest, RevisionControllerStop

__all__: list[str] = []

MAX_STEP_M = 0.008
MAX_BIAS_M = 0.020
MAX_PATH_M = 0.030
MAX_CHUNK_CORRECTIONS = 3
MAX_EPISODE_CORRECTIONS = 12
MAX_WALL_SECONDS = 180.0
MAX_SEED_DEVIATION = 0.05
POSITION_TOL_M = 0.002
ROTATION_TOL_RAD = math.radians(3.0)
MAX_WINDOW_SPREAD_M = 0.0005
MIN_PARENT_GAP_M = 0.003


def _vector(value: Any) -> np.ndarray:
    value = np.array(value, dtype=float, copy=True)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Feedback requires finite position vectors.")
    return value


def _bounded_update(
    bias: Any, error: Any, count: int, path: float
) -> tuple[np.ndarray, np.ndarray, float]:
    bias, increment = _vector(bias), _vector(error)
    if type(count) is not int or not 0 <= count < MAX_CHUNK_CORRECTIONS:
        raise ValueError("E8 chunk correction budget exhausted.")
    if not math.isfinite(path) or not 0 <= path <= MAX_PATH_M:
        raise ValueError("E8 correction path budget is invalid.")
    magnitude = math.hypot(*bias)
    if magnitude > MAX_BIAS_M:
        raise ValueError("Existing E8 bias exceeds its original bound.")
    step = math.hypot(*increment)
    if step > MAX_STEP_M:
        increment *= MAX_STEP_M / step
    proposed = bias + increment
    if (
        math.hypot(*proposed) > MAX_BIAS_M
        or float(np.linalg.norm(proposed)) > MAX_BIAS_M
    ):
        distance = math.hypot(*increment)
        direction = increment / distance
        projection = float(bias @ direction)
        available = (MAX_BIAS_M - magnitude) * (MAX_BIAS_M + magnitude)
        root = math.hypot(projection, math.sqrt(max(0.0, available)))
        remaining = (
            available / (root + projection) if projection > 0 else root - projection
        )
        high = min(1.0, remaining / distance)
        low = 0.0
        for _ in range(64):
            middle = (low + high) / 2
            candidate = bias + increment * middle
            if (
                math.hypot(*candidate) <= MAX_BIAS_M
                and float(np.linalg.norm(candidate)) <= MAX_BIAS_M
            ):
                low = middle
            else:
                high = middle
        increment *= low
        proposed = bias + increment
    if not np.any(proposed != bias):
        raise ValueError("E8 cumulative position budget has no remaining step.")
    new_path = path + math.hypot(*increment)
    if new_path > MAX_PATH_M:
        raise ValueError("E8 cumulative correction path exhausted.")
    return proposed, increment, new_path


def _points(vertices: Any, pose: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Feedback collision inputs are incomplete/nonfinite.")
    return vertices @ pose[:3, :3].T + pose[:3, 3]


def _gap(first: np.ndarray, second: np.ndarray) -> float:
    distance = np.maximum(
        np.maximum(first.min(0) - second.max(0), second.min(0) - first.max(0)), 0
    )
    return float(np.linalg.norm(distance))


def _rotation_error(actual: np.ndarray, nominal: np.ndarray) -> float:
    return min(
        math.acos(float(np.clip((np.trace(actual.T @ candidate) - 1) / 2, -1, 1)))
        for candidate in (nominal, nominal @ np.diag([-1.0, -1.0, 1.0]))
    )


@dataclass
class _Chunk:
    request: Any
    plan: Any
    desired: np.ndarray | None
    desired_standoff: np.ndarray | None
    closed_qpos: torch.Tensor | None
    hand_ids: tuple[int, ...]
    selected_metadata: dict[str, Any]
    command_baseline: torch.Tensor
    first_wall: float
    original_planned_at: float | None = None
    original_timeout: float | None = None
    active_revision: int = 0
    active_generation: int = -1
    corrections: int = 0
    path_length: float = 0.0
    inspected: set[tuple[int, str]] = field(default_factory=set)
    last_contact_time: float = -1.0
    repair_required: bool = False
    resume_phase: str = "close"


@dataclass
class _Reservation:
    revision: int
    previous_generation: int
    phase: str
    bias: np.ndarray
    increment: np.ndarray
    path_length: float
    command_seed: torch.Tensor
    planned_request: Any = None
    planned_plan: Any = None


class TwistFeedbackState:
    """Own one route's native samples and correction ledger, not an executor.

    Successful planning only reserves a replacement. The next due observation
    confirms its installed revision/generation before committing position bias.
    """

    controller_id = "gen_sim.twist.precontact"
    revision = "1"
    supported_skill_ids = ("twist",)

    def __init__(
        self,
        route: Any,
        simulation: Any,
        robot: Any,
        sensor: Any,
        geometry_guard: Any,
        *,
        wall_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.route, self.simulation, self.robot, self.sensor, self.geometry_guard = (
            route,
            simulation,
            robot,
            sensor,
            geometry_guard,
        )
        self.enabled = route.feedback_enabled
        self.grip_extra_m = route.grip_extra_m
        self.wall_clock = wall_clock
        if type(self.enabled) is not bool or self.grip_extra_m not in {0.0, 0.002}:
            raise ValueError(
                "E8 feedback requires an explicit route opt-in and paired grip extra."
            )
        if (
            sensor.route != route
            or sensor.robot is not robot
            or sensor.art is not simulation.get_articulation(route.binding.object_id)
            or geometry_guard.route != route
            or geometry_guard.robot is not robot
            or geometry_guard.simulation is not simulation
            or geometry_guard.sensor is not sensor
        ):
            raise ValueError("E8 feedback must borrow its exact route/robot/sensor.")
        self.art = sensor.art
        self.bias = np.zeros(3)
        self.episode_corrections = 0
        self.samples: deque[dict[str, Any]] = deque(maxlen=12)
        self.chunks: dict[str, _Chunk] = {}
        self.pending: dict[str, _Reservation] = {}
        self.audit: deque[dict[str, Any]] = deque(maxlen=256)
        self.sample_error: str | None = None
        self.last_parent_contact_at = -1.0

    def reset(self) -> None:
        """Clear episode-owned state without stepping or changing native targets."""
        self.bias[:] = 0
        self.episode_corrections = 0
        self.samples.clear()
        self.chunks.clear()
        self.pending.clear()
        self.audit.clear()
        self.sample_error = None
        self.last_parent_contact_at = -1.0

    def capture_native(self) -> None:
        """Read after the existing sensor capture; never fetch/update the sensor."""
        if not self.enabled:
            return
        try:
            row = self.sensor.trace[-1]
            timestamp = float(row["timestamp"])
            if (
                not math.isfinite(timestamp)
                or type(row["parent_contact"]) is not bool
                or type(row["target_contact"]) is not bool
                or type(row["dropped_contacts"]) is not int
                or row["dropped_contacts"] < 0
            ):
                raise ValueError("Native feedback contact fields are invalid.")
            if self.samples and timestamp < self.samples[-1]["timestamp"]:
                self.reset()
            if row["parent_contact"]:
                self.last_parent_contact_at = timestamp
            solver = self.robot.cfg.solver_cfg[self.route.arm + "_arm"]
            pose = self.robot.get_link_pose(solver.end_link_name, to_matrix=True)[0]
            pose = pose @ torch.as_tensor(
                solver.tcp, dtype=pose.dtype, device=pose.device
            )
            tcp = _matrix(pose).copy()
            self.samples.append(
                {
                    "timestamp": timestamp,
                    "tcp": tcp,
                    "valid": row["valid"] is True,
                    "dropped": row["dropped_contacts"],
                    "parent_contact": row["parent_contact"] is True,
                    "target_contact": row["target_contact"] is True,
                }
            )
        except (
            ValueError,
            KeyError,
            AttributeError,
            IndexError,
            RuntimeError,
            TypeError,
        ) as error:
            self.sample_error = str(error)

    def _tcp(self, qpos: torch.Tensor, control_part: str) -> np.ndarray:
        solver = self.robot.cfg.solver_cfg[control_part]
        fk = self.robot.compute_fk(
            qpos=qpos,
            link_names=[solver.end_link_name],
            qpos_joint_names=self.robot.joint_names,
        )
        root = self.robot.get_local_pose(to_matrix=True).to(fk)
        value = (root[:, None] @ fk)[0, 0] @ torch.as_tensor(
            solver.tcp, dtype=fk.dtype, device=fk.device
        )
        return _matrix(value).copy()

    def record_plan(self, action: Any, request: Any, context: Any, plan: Any) -> None:
        """Record only a finalized selected plan; never count a proposal as installed."""
        if not self.enabled or not bool(plan.plan_success.all()):
            return
        if (
            action.robot is not self.robot
            or context.batch_size != 1
            or not request.invocation_id
        ):
            raise ValueError("Feedback plan belongs to another runtime/call.")
        key = request.invocation_id
        request = request.snapshot()
        plan = plan.snapshot()
        if request.revision > 0:
            reservation = self.pending.get(key)
            if (
                reservation is None
                or request.revision != reservation.revision
                or not request.goal.semantics.geometry.get("gen_sim_twist_continuation")
            ):
                raise ValueError("Feedback replacement has no matching reservation.")
            chunk = self.chunks[key]
            elapsed = float(self.wall_clock()) - chunk.first_wall
            if not math.isfinite(elapsed) or not 0 <= elapsed <= MAX_WALL_SECONDS:
                raise ValueError("Original E8 wall deadline exhausted during planning.")
            reservation.planned_request, reservation.planned_plan = request, plan
            return
        if key in self.chunks:
            return
        motion = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        hand = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        reach = next((s for s in plan.segments if s.name == "reach"), None)
        close = next((s for s in plan.segments if s.name == "close"), None)
        nominal = (
            None
            if reach is None
            else self._tcp(
                plan.joint_trajectory.positions[:, reach.stop - 1], motion.control_part
            )
        )
        if nominal is not None:
            nominal[:3, 3] -= self.bias
        standoff = None if nominal is None else nominal.copy()
        if standoff is not None:
            standoff[:3, 3] -= nominal[:3, 2] * request.skill_options.pre_grasp_distance
        baseline = self.robot.get_qpos(target=True).clone()
        contact_epoch = float(context.robot.timestamp)
        if baseline.shape != context.robot.qpos.shape or not bool(
            torch.isfinite(baseline).all()
        ):
            raise ValueError("Initial owned whole-robot command target is unavailable.")
        if not math.isfinite(contact_epoch):
            raise ValueError("Initial feedback native timestamp is invalid.")
        self.chunks[key] = _Chunk(
            request,
            plan,
            nominal,
            standoff,
            (
                None
                if close is None
                else plan.joint_trajectory.positions[:, close.stop - 1].clone()
            ),
            tuple(hand.joint_ids),
            deepcopy(
                dict(plan.diagnostics.metadata) if plan.diagnostics is not None else {}
            ),
            baseline,
            float(self.wall_clock()),
            last_contact_time=contact_epoch,
        )

    def apply_bias(self, request: Any, context: Any) -> Any:
        """Bias the private command goal, never the native planning context."""
        if not self.enabled:
            return request
        pending = self.pending.get(request.invocation_id)
        bias = (
            pending.bias
            if pending is not None and request.revision == pending.revision
            else self.bias
        )
        pose = _matrix(
            self.art.get_link_pose(self.route.binding.link, to_matrix=True)[0]
        )
        affordance = request.goal.semantics.affordance
        local_bias = pose[:3, :3].T @ bias
        point = np.asarray(affordance.grasp_position) + local_bias
        geometry = request.goal.semantics.geometry
        candidate_point = geometry.get(
            "gen_sim_twist_grasp_point", geometry.get("gen_sim_twist_outer_point")
        )
        if candidate_point is not None:
            geometry = {
                **geometry,
                "gen_sim_twist_grasp_point": tuple(
                    np.asarray(candidate_point) + local_bias
                ),
            }
        return replace(
            request,
            goal=replace(
                request.goal,
                semantics=replace(
                    request.goal.semantics,
                    geometry=geometry,
                    affordance=replace(affordance, grasp_position=tuple(point)),
                ),
            ),
        )

    def plan_continuation(self, action: Any, request: Any, context: Any) -> Any:
        """Plan from the separate reserved command seed; the root owns final guards."""
        from .twist_continuation import plan_continuation

        reservation = self.pending.get(request.invocation_id)
        chunk = self.chunks.get(request.invocation_id)
        if (
            not self.enabled
            or reservation is None
            or chunk is None
            or reservation.revision != request.revision
        ):
            raise ValueError("E8 continuation has no owned staged correction.")
        continuation = replace(chunk, resume_phase=reservation.phase)
        return plan_continuation(
            action,
            request,
            context,
            continuation,
            command_seed=reservation.command_seed,
            bias=reservation.bias,
        )

    def _seed(
        self, chunk: _Chunk, context: Any, frame: RuntimeCommandFrame | None
    ) -> torch.Tensor:
        if (
            frame is None
            or not torch.equal(frame.env_ids, context.env_ids)
            or frame.active_mask.tolist() != [True]
        ):
            raise ValueError("Correction requires a matching accepted SEND frame.")
        expected = {
            chunk.request.binding.endpoint("primary", role)
            .require_target(JointPositionTarget)
            .address_fingerprint
            for role in ("motion", "grasp")
        }
        if {
            command.target.address_fingerprint for command in frame.commands
        } != expected:
            raise ValueError(
                "Accepted command addresses differ from the original call."
            )
        seed = chunk.command_baseline.clone().to(context.robot.qpos)
        readback = self.robot.get_qpos(target=True).clone().to(seed)
        controlled = set()
        for command in frame.commands:
            if not isinstance(command.target, JointPositionTarget) or not isinstance(
                command.payload, JointPositionPayload
            ):
                raise ValueError("Feedback requires typed joint-position commands.")
            ids = list(command.target.joint_ids)
            values = command.payload.positions.to(seed)
            if not torch.equal(readback[:, ids], values):
                raise ValueError(
                    "Accepted SEND does not match current command-target readback."
                )
            seed[:, ids] = values
            controlled.update(ids)
        untouched = [index for index in range(seed.shape[1]) if index not in controlled]
        if untouched and not torch.equal(readback[:, untouched], seed[:, untouched]):
            raise ValueError(
                "Unaddressed command targets changed outside the owned call."
            )
        if (
            seed.shape != context.robot.qpos.shape
            or not bool(torch.isfinite(seed).all())
            or not bool(torch.isfinite(context.robot.qpos).all())
            or float((seed - context.robot.qpos).abs().max()) > MAX_SEED_DEVIATION
        ):
            raise ValueError(
                "Separate command seed exceeds the native observation bound."
            )
        return seed

    def _gaps(self, chunk: _Chunk, phase: str) -> tuple[list[float], list[float]]:
        names = tuple(name for name in self.sensor.finger_names if "pad" in name)
        if len(names) != 2 or chunk.closed_qpos is None:
            raise ValueError("Feedback requires two pad colliders and a closed target.")
        parent = _matrix(
            self.art.get_link_pose(self.route.binding.parent, to_matrix=True)[0]
        )
        parents = [
            _points(mesh.vertices, parent)
            for mesh in self.sensor.geometry.parent_collisions
        ]
        if not parents:
            raise ValueError("Feedback parent collision inputs are missing.")
        qpos = self.robot.get_qpos().clone()
        qpos[:, list(chunk.hand_ids)] = chunk.closed_qpos[:, list(chunk.hand_ids)].to(
            qpos
        )
        if phase == "reach":
            qpos = chunk.closed_qpos.clone()
        fk = self.robot.compute_fk(
            qpos=qpos, link_names=list(names), qpos_joint_names=self.robot.joint_names
        )
        predicted = self.robot.get_local_pose(to_matrix=True).to(fk)[:, None] @ fk
        native_gaps, predicted_gaps = [], []
        for index, name in enumerate(names):
            local = self.sensor.finger_collision_points[name]
            native = _points(
                local, _matrix(self.robot.get_link_pose(name, to_matrix=True)[0])
            )
            endpoint = _points(local, _matrix(predicted[0, index]))
            if phase == "reach":
                endpoint += self.samples[-1]["tcp"][:3, 3] - (
                    chunk.desired_standoff[:3, 3] + self.bias
                )
            native_gaps.append(min(_gap(native, points) for points in parents))
            predicted_gaps.append(min(_gap(endpoint, points) for points in parents))
        return native_gaps, predicted_gaps

    def _installed(self, key: str, chunk: _Chunk, due: DueObservationRequest) -> None:
        if (
            type(due.attempt_generation) is not int
            or due.attempt_generation < 0
            or due.attempt_generation < chunk.active_generation
        ):
            raise ValueError("Feedback plan generation is invalid or moved backward.")
        reservation = self.pending.get(key)
        if reservation is None:
            if due.invocation_revision != chunk.active_revision:
                raise ValueError("A foreign revision replaced the feedback-owned plan.")
        elif due.invocation_revision == reservation.revision:
            if (
                due.attempt_generation <= reservation.previous_generation
                or reservation.planned_request is None
                or reservation.planned_plan is None
            ):
                raise ValueError(
                    "Feedback revision was not installed from its reserved plan."
                )
            self.bias = reservation.bias.copy()
            chunk.corrections += 1
            chunk.path_length = reservation.path_length
            self.episode_corrections += 1
            chunk.request, chunk.plan = (
                reservation.planned_request,
                reservation.planned_plan,
            )
            close = chunk.plan.segment("close")
            chunk.closed_qpos = chunk.plan.joint_trajectory.positions[
                :, close.stop - 1
            ].clone()
            chunk.resume_phase = reservation.phase
            chunk.active_revision = reservation.revision
            self.pending.pop(key)
            self.audit.append(
                {
                    "decision": "installed",
                    "invocation_id": key,
                    "revision": due.invocation_revision,
                    "generation": due.attempt_generation,
                    "corrections": chunk.corrections,
                    "episode_corrections": self.episode_corrections,
                    "bias_m": self.bias.tolist(),
                    "path_m": chunk.path_length,
                }
            )
        elif due.invocation_revision != chunk.active_revision:
            raise ValueError("Feedback pending revision lost its scheduling identity.")
        chunk.active_generation = due.attempt_generation

    def propose(
        self,
        context: Any,
        due: DueObservationRequest,
        *,
        previous_command: RuntimeCommandFrame | None,
    ) -> ActionInvocation | RevisionControllerStop | None:
        if not self.enabled:
            return None
        stop_diagnostics: dict[str, Any] = {}
        try:
            if (
                due.skill_id != "twist"
                or context.batch_size != 1
                or due.env_mask.tolist() != [True]
                or not due.invocation_id
            ):
                raise ValueError(
                    "Feedback due-cycle scope differs from its one-env Twist."
                )
            chunk = self.chunks.get(due.invocation_id)
            if chunk is None:
                raise ValueError("No finalized selected E8 plan was recorded.")
            self._installed(due.invocation_id, chunk, due)
            if chunk.desired is None or chunk.closed_qpos is None:
                return None
            if chunk.original_planned_at is None:
                chunk.original_planned_at, chunk.original_timeout = (
                    due.original_planned_at,
                    due.original_action_timeout,
                )
            elif (chunk.original_planned_at, chunk.original_timeout) != (
                due.original_planned_at,
                due.original_action_timeout,
            ):
                raise ValueError("Feedback logical deadline identity changed.")
            now = float(context.robot.timestamp)
            wall_elapsed = float(self.wall_clock()) - chunk.first_wall
            if (
                not np.isfinite(
                    [
                        now,
                        chunk.original_planned_at,
                        chunk.original_timeout,
                        wall_elapsed,
                    ]
                ).all()
                or chunk.original_timeout <= 0
                or now < chunk.original_planned_at
                or wall_elapsed < 0
            ):
                raise ValueError("Original E8 logical/wall clocks are invalid.")
            if (
                now - chunk.original_planned_at > chunk.original_timeout
                or wall_elapsed > MAX_WALL_SECONDS
            ):
                raise ValueError("Original E8 logical/wall deadline exhausted.")
            if self.sample_error is not None:
                raise ValueError(self.sample_error)
            phase = due.segment_name
            fresh = [
                sample
                for sample in self.samples
                if chunk.last_contact_time < sample["timestamp"] <= now
            ]
            parent_since_due = (
                chunk.last_contact_time < self.last_parent_contact_at <= now
            ) or any(sample["parent_contact"] for sample in fresh)
            chunk.last_contact_time = now
            if phase in {"approach", "reach", "close"} and parent_since_due:
                raise ValueError("Pre-contact E8 motion touched the parent.")
            inspected = (due.invocation_revision, phase)
            if (
                phase not in {"reach", "close"}
                or inspected in chunk.inspected
                or (
                    due.invocation_revision > 0
                    and phase == "reach"
                    and chunk.resume_phase == "close"
                )
            ):
                return None
            chunk.inspected.add(inspected)
            window = list(self.samples)[-3:]
            if len(window) != 3 or any(
                not sample["valid"]
                or sample["dropped"]
                or not np.isfinite(sample["tcp"]).all()
                for sample in window
            ):
                raise ValueError("Native E8 feedback window is invalid/incomplete.")
            stamps = np.asarray([sample["timestamp"] for sample in window])
            if (
                not np.isfinite(stamps).all()
                or not np.all(np.diff(stamps) > 0)
                or stamps[-1] > now + 1e-9
                or now - stamps[-1] > 1.1 * float(self.simulation.sim_config.physics_dt)
            ):
                raise ValueError("Native E8 feedback window is stale.")
            desired = chunk.desired_standoff if phase == "reach" else chunk.desired
            desired = _matrix(desired)
            positions = np.asarray([sample["tcp"][:3, 3] for sample in window])
            errors = desired[:3, 3] - positions
            error = errors.mean(0)
            angle = max(
                _rotation_error(sample["tcp"][:3, :3], desired[:3, :3])
                for sample in window
            )
            spread = float(np.linalg.norm(positions - positions.mean(0), axis=1).max())
            native_gaps, predicted = self._gaps(chunk, phase)
            if (
                not np.isfinite(native_gaps + predicted).all()
                or any(sample["parent_contact"] for sample in window)
                or min(native_gaps) < MIN_PARENT_GAP_M
            ):
                raise ValueError("Native E8 parent clearance is unsafe.")
            alignment_diagnostics = {
                "angle_deg": math.degrees(angle),
                "spread_mm": spread * 1000,
                "position_error_mm": float(np.linalg.norm(errors, axis=1).max()) * 1000,
                "window_timestamps": stamps.tolist(),
                "invocation_id": due.invocation_id,
                "attempt_generation": due.attempt_generation,
                "due_phase": phase,
                "due_revision": due.invocation_revision,
                "due_waypoint": due.next_waypoint_index,
                "angle_limit_deg": math.degrees(ROTATION_TOL_RAD),
                "spread_limit_mm": MAX_WINDOW_SPREAD_M * 1000,
            }
            if angle > ROTATION_TOL_RAD or spread > MAX_WINDOW_SPREAD_M:
                stop_diagnostics = {
                    "code": "native_tracking_window_not_stable_aligned",
                    **alignment_diagnostics,
                    "alignment_accepted": False,
                }
                details = "; ".join(
                    f"{name}={value}"
                    for name, value in alignment_diagnostics.items()
                    if name not in {"invocation_id", "attempt_generation"}
                )
                raise ValueError(
                    f"Native E8 tracking window is not stable/aligned. {details}"
                )
            self.audit.append(
                {
                    "decision": "tracking_window",
                    **alignment_diagnostics,
                    "alignment_accepted": True,
                }
            )
            if not chunk.repair_required:
                axial = abs(float(error @ desired[:3, 2]))
                lateral = float(
                    np.linalg.norm(error - (error @ desired[:3, 2]) * desired[:3, 2])
                )
                depth, radius = (
                    float(self.route.binding.grip_depth),
                    float(self.route.binding.grip_width) / 2,
                )
                if not np.isfinite([depth, radius]).all() or min(depth, radius) <= 0:
                    raise ValueError("E8 grip dimensions are invalid.")
                chunk.repair_required = axial > math.nextafter(
                    depth, math.inf
                ) or lateral > math.nextafter(radius, math.inf)
                if not chunk.repair_required:
                    return None
            if phase == "reach" and (
                chunk.corrections > 0 or min(predicted) >= MIN_PARENT_GAP_M
            ):
                return None
            if chunk.corrections == 0 and min(predicted) >= MIN_PARENT_GAP_M:
                return None
            if float(np.linalg.norm(errors, axis=1).max()) <= POSITION_TOL_M:
                if phase == "close" and min(predicted) < MIN_PARENT_GAP_M:
                    raise ValueError("Predicted closed E8 parent clearance is unsafe.")
                return None
            if any(sample["target_contact"] for sample in window):
                raise ValueError(
                    "E8 correction cannot move an established target contact."
                )
            if (
                self.episode_corrections >= MAX_EPISODE_CORRECTIONS
                or due.invocation_id in self.pending
            ):
                raise ValueError("E8 correction episode/pending budget exhausted.")
            bias, increment, path = _bounded_update(
                self.bias, error, chunk.corrections, chunk.path_length
            )
            seed = self._seed(chunk, context, previous_command)
            original = chunk.request
            geometry = dict(original.goal.semantics.geometry)
            geometry.update(
                gen_sim_twist_continuation=True, gen_sim_twist_resume_phase=phase
            )
            live = self.art.get_qpos()
            geometry["gen_sim_twist_measured_qpos"] = float(
                live[0, self.art.joint_names.index(self.route.binding.joint)]
            )
            goal = replace(
                original.goal,
                semantics=replace(original.goal.semantics, geometry=geometry),
            )
            invocation = ActionInvocation(
                skill_id=original.skill_id,
                goal=goal,
                binding=original.binding,
                motion_policy=original.motion_policy,
                tracking_policy=original.tracking_policy,
                recovery_policy=original.recovery_policy,
                phase_effect_gates=original.phase_effect_gates,
                skill_options=original.skill_options,
                invocation_id=original.invocation_id,
                revision=due.invocation_revision + 1,
            )
            self.pending[due.invocation_id] = _Reservation(
                invocation.revision,
                due.attempt_generation,
                phase,
                bias,
                increment,
                path,
                seed,
            )
            self.audit.append(
                {
                    "decision": "reserved",
                    "invocation_id": due.invocation_id,
                    "revision": invocation.revision,
                    "increment_m": increment.tolist(),
                    "elapsed_s": now - chunk.original_planned_at,
                }
            )
            return invocation
        except (
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            RuntimeError,
        ) as error:
            self.audit.append(
                {"decision": "stopped", "reason": str(error), **stop_diagnostics}
            )
            return RevisionControllerStop(str(error))
