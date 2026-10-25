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

"""Physical replay and observation collection for Atomic Skill benchmarks."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ..contracts import (
    ExecutionObservation,
    ReplayObservationTrace,
    ReplayObservationWindow,
)
from .atomic_evaluation import signed_rotation_about_axis

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_skills.plans import CompiledTrajectory
    from embodichain.lab.sim.objects import Robot

    from ..atomic_skill import AtomicSkillCaseProvider
    from ..models import BenchmarkCase
    from .atomic_objects import AtomicArticulationHandle, AtomicObjectHandle

__all__ = ["AtomicReplayRuntime"]


@dataclass(frozen=True)
class _ReplayMetrics:
    """Tracking and articulation-effect accumulators from one replay."""

    squared_tracking_error: torch.Tensor
    tracking_value_count: int
    physics_steps: int | None = None
    articulation_joint_initial: torch.Tensor | None = None
    articulation_joint_final: torch.Tensor | None = None
    articulation_joint_peak_signed_delta: torch.Tensor | None = None
    articulation_joint_effect_end: torch.Tensor | None = None
    object_rotation_peak_signed_rad: torch.Tensor | None = None
    replay_traces: Mapping[str, ReplayObservationTrace] = field(default_factory=dict)


@dataclass
class _ReplayTraceAccumulator:
    """Mutable samples for one validated replay observation window."""

    window: ReplayObservationWindow
    tcp_pose: list[torch.Tensor] = field(default_factory=list)
    object_pose: list[torch.Tensor] = field(default_factory=list)
    articulation_joint: list[torch.Tensor] = field(default_factory=list)
    gripper_qpos: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    scene_object_poses: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def finish(self) -> ReplayObservationTrace:
        """Stack synchronized samples into the public immutable trace."""
        return ReplayObservationTrace(
            tcp_pose=torch.stack(self.tcp_pose) if self.tcp_pose else None,
            object_pose=torch.stack(self.object_pose) if self.object_pose else None,
            articulation_joint=(
                torch.stack(self.articulation_joint)
                if self.articulation_joint
                else None
            ),
            gripper_qpos={
                part: torch.stack(values) for part, values in self.gripper_qpos.items()
            },
            scene_object_poses={
                object_id: torch.stack(values)
                for object_id, values in self.scene_object_poses.items()
            },
        )


@dataclass(frozen=True)
class _PhysicsReplaySettings:
    """Shared physics-step counts for timed evaluation and video replay."""

    steps_per_waypoint: int
    hold_steps: int
    hold_sim_steps: int
    joint_tracking_tolerance_rad: float
    clock: str = "fixed_samples"


class AtomicReplayRuntime:
    """Replay trajectories against explicitly bound simulator state.

    Create one runtime at each evaluation or video boundary so its object and
    gripper bindings belong to the currently configured scenario. This runtime
    owns neither scene reset nor simulator teardown.

    Args:
        simulation: Simulator receiving the replay's physics steps.
        robot: Robot receiving full-joint position commands.
        control_part: Arm used for end-effector and final-state observations.
        track_config: Track-owned physics replay settings.
        objects: Configured movable object handles indexed by object ID.
        articulations: Configured articulation handles indexed by object ID.
        gripper_states: Open and grasp commands for every configured gripper.
    """

    def __init__(
        self,
        *,
        simulation: SimulationManager,
        robot: Robot,
        control_part: str,
        track_config: Mapping[str, object],
        objects: Mapping[str, AtomicObjectHandle],
        articulations: Mapping[str, AtomicArticulationHandle],
        gripper_states: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.simulation = simulation
        self.robot = robot
        self.control_part = control_part
        self._track_config = track_config
        self._objects = objects
        self._articulations = articulations
        self._gripper_states = gripper_states

    def execute(
        self,
        compiled: CompiledTrajectory,
        case: BenchmarkCase,
        provider: AtomicSkillCaseProvider,
    ) -> ExecutionObservation | None:
        """Replay a finite full-robot trajectory and collect physical evidence.

        Args:
            compiled: Full-robot trajectory with one planning result per row.
            case: Frozen benchmark inputs and expected initial robot state.
            provider: Skill-specific observation and effect-window declarations.

        Returns:
            Execution observations, or ``None`` when no successful finite row
            can be replayed. Failed rows remain at their frozen start state.
        """
        if self.simulation is None or self.robot is None or self._track_config is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        if not self.is_replayable(compiled):
            return None
        object_handle = self._objects.get(case.object_id or "")
        initial_object_position = (
            None
            if object_handle is None
            else object_handle.entity.get_local_pose(to_matrix=True)[:, :3, 3].clone()
        )
        settings = self._physics_settings()
        self._synchronize()
        started = time.perf_counter()
        replay_metrics = self.replay_physics(
            compiled, case, provider, collect_metrics=True
        )
        self._synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if replay_metrics is None:
            raise RuntimeError("Timed Atomic Skill replay did not return metrics.")
        observed = self.robot.get_qpos()
        tracking = torch.sqrt(
            replay_metrics.squared_tracking_error
            / max(replay_metrics.tracking_value_count, 1)
        )
        trajectory = compiled.trajectory.positions
        physics_steps = replay_metrics.physics_steps
        if physics_steps is None:
            physics_steps = (
                trajectory.shape[1] * settings.steps_per_waypoint
                + settings.hold_steps * settings.hold_sim_steps
            )
        simulated_execution_time = physics_steps * float(
            self.simulation.sim_config.physics_dt
        )
        final_tcp = self.robot.compute_fk(
            self.robot.get_qpos(name=self.control_part),
            name=self.control_part,
            to_matrix=True,
        )
        object_lift = None
        final_object_pose = None
        if object_handle is not None and initial_object_position is not None:
            final_object_pose = object_handle.entity.get_local_pose(
                to_matrix=True
            ).clone()
            final_object_position = final_object_pose[:, :3, 3]
            object_lift = final_object_position[:, 2] - initial_object_position[:, 2]
        return ExecutionObservation(
            # A compiled trajectory carries one planning result per batch row.
            # Physical replay is allowed when at least one row succeeded, but
            # failed rows must never be promoted to execution success merely
            # because their held command happened to track the controller.
            execution_success=(
                compiled.plan_success.to(device=observed.device)
                & torch.isfinite(observed).all(dim=1)
                & (tracking <= settings.joint_tracking_tolerance_rad)
            ),
            final_tcp_pose=final_tcp,
            joint_tracking_rmse_rad=tracking,
            execution_time_ms=elapsed_ms,
            task_completion_time_s=simulated_execution_time,
            object_lift_delta_m=object_lift,
            final_arm_qpos=self.robot.get_qpos(name=self.control_part).clone(),
            final_object_pose=final_object_pose,
            articulation_joint_initial=replay_metrics.articulation_joint_initial,
            articulation_joint_final=replay_metrics.articulation_joint_final,
            articulation_joint_peak_signed_delta=(
                replay_metrics.articulation_joint_peak_signed_delta
            ),
            articulation_joint_effect_end=replay_metrics.articulation_joint_effect_end,
            object_rotation_peak_signed_rad=(
                replay_metrics.object_rotation_peak_signed_rad
            ),
            scene_object_final_poses={
                str(object_id): self._object_handle(str(object_id))
                .entity.get_local_pose(to_matrix=True)
                .clone()
                for object_id in dict(
                    case.case_parameters.get("scene_object_poses", {})
                )
            },
            final_gripper_qpos={
                part: self.robot.get_qpos(name=part).clone()
                for part in self._gripper_states
            },
            gripper_open_qpos={
                part: states[0].clone() for part, states in self._gripper_states.items()
            },
            gripper_grasp_qpos={
                part: states[1].clone() for part, states in self._gripper_states.items()
            },
            replay_traces=replay_metrics.replay_traces,
        )

    def _physics_settings(self) -> _PhysicsReplaySettings:
        """Resolve and validate the common physics-replay step counts."""
        if self._track_config is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        physics = dict(self._track_config.get("physics", {}))
        settings = _PhysicsReplaySettings(
            steps_per_waypoint=int(physics.get("steps_per_waypoint", 4)),
            hold_steps=int(physics.get("hold_steps", 80)),
            hold_sim_steps=int(physics.get("hold_sim_steps", 2)),
            joint_tracking_tolerance_rad=float(
                physics.get("joint_tracking_tolerance_rad", 0.05)
            ),
            clock=str(physics.get("clock", "fixed_samples")),
        )
        if settings.clock not in {"fixed_samples", "compiled"}:
            raise ValueError("physics.clock must be fixed_samples or compiled.")
        if (
            settings.steps_per_waypoint < 1
            or settings.hold_steps < 0
            or settings.hold_sim_steps < 1
        ):
            raise ValueError("Atomic Skill physics step counts are invalid.")
        return settings

    @staticmethod
    def is_replayable(compiled: CompiledTrajectory) -> bool:
        """Return whether successful rows contain a finite nonempty trajectory.

        Args:
            compiled: Full-robot trajectory and per-row planning results.

        Returns:
            Whether at least one row succeeded and all successful rows are finite.
        """
        trajectory = compiled.trajectory.positions
        plan_success = compiled.plan_success.to(
            device=trajectory.device, dtype=torch.bool
        )
        if trajectory.shape[1] == 0 or not bool(plan_success.any().item()):
            return False
        return bool(torch.isfinite(trajectory[plan_success]).all().item())

    @staticmethod
    def is_recordable(compiled: CompiledTrajectory) -> bool:
        """Check whether a rollout can produce a diagnostic replay video.

        Args:
            compiled: Full-robot trajectory, including failed planning rows.

        Returns:
            Whether every row is finite and at least one waypoint is present.
        """
        trajectory = compiled.trajectory.positions
        return trajectory.shape[1] > 0 and bool(torch.isfinite(trajectory).all().item())

    def _capture_replay_samples(
        self,
        accumulators: Sequence[_ReplayTraceAccumulator],
        *,
        object_handle: AtomicObjectHandle | None,
        scene_object_handles: Mapping[str, AtomicObjectHandle],
        articulation_handle: AtomicArticulationHandle | None,
        articulation_joint_name: str | None,
    ) -> None:
        """Append one synchronized simulator sample to active windows."""
        if not accumulators:
            return
        if self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        channels = frozenset(
            channel
            for accumulator in accumulators
            for channel in accumulator.window.channels
        )
        tcp_pose = None
        object_pose = None
        articulation_joint = None
        grippers: dict[str, torch.Tensor] = {}
        scene_object_poses: dict[str, torch.Tensor] = {}
        if "tcp_pose" in channels:
            tcp_pose = self.robot.compute_fk(
                self.robot.get_qpos(name=self.control_part),
                name=self.control_part,
                to_matrix=True,
            ).clone()
        if "object_pose" in channels and object_handle is not None:
            object_pose = object_handle.entity.get_local_pose(to_matrix=True).clone()
        if (
            "articulation_joint" in channels
            and articulation_handle is not None
            and articulation_joint_name is not None
        ):
            articulation_joint = articulation_handle.joint_qpos(articulation_joint_name)
        if "gripper_qpos" in channels:
            grippers = {
                part: self.robot.get_qpos(name=part).clone()
                for part in self._gripper_states
            }
        if "scene_object_pose" in channels:
            requested_ids = {
                object_id
                for accumulator in accumulators
                for object_id in accumulator.window.scene_object_ids
            }
            scene_object_poses = {
                object_id: scene_object_handles[object_id]
                .entity.get_local_pose(to_matrix=True)
                .clone()
                for object_id in requested_ids
            }
        for accumulator in accumulators:
            requested = accumulator.window.channels
            if "tcp_pose" in requested and tcp_pose is not None:
                accumulator.tcp_pose.append(tcp_pose)
            if "object_pose" in requested and object_pose is not None:
                accumulator.object_pose.append(object_pose)
            if "articulation_joint" in requested and articulation_joint is not None:
                accumulator.articulation_joint.append(articulation_joint)
            if "gripper_qpos" in requested:
                for part, qpos in grippers.items():
                    accumulator.gripper_qpos.setdefault(part, []).append(qpos)
            if "scene_object_pose" in requested:
                for object_id in accumulator.window.scene_object_ids:
                    accumulator.scene_object_poses.setdefault(object_id, []).append(
                        scene_object_poses[object_id]
                    )

    def replay_physics(
        self,
        compiled: CompiledTrajectory,
        case: BenchmarkCase,
        provider: AtomicSkillCaseProvider | None,
        *,
        collect_metrics: bool,
    ) -> _ReplayMetrics | None:
        """Replay one compiled trajectory with the common physics contract.

        Args:
            compiled: Full-robot trajectory and per-row planning results.
            case: Frozen benchmark inputs, including failed-row hold positions.
            provider: Optional skill-specific observation and effect declarations.
            collect_metrics: Accumulate tracking and physical observations when
                true; video replay uses the same stepping without accumulation.

        Returns:
            Collected replay metrics when requested, otherwise ``None``.
        """
        if self.simulation is None or self.robot is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        trajectory = compiled.trajectory.positions
        plan_success = compiled.plan_success.to(
            device=trajectory.device, dtype=torch.bool
        )
        if plan_success.shape != (trajectory.shape[0],):
            raise ValueError(
                "Compiled plan_success must contain one value per trajectory row."
            )
        if bool((~plan_success).any().item()):
            if case.full_start_qpos is None:
                raise ValueError(
                    "Atomic replay requires full_start_qpos to hold failed rows."
                )
            hold_qpos = case.full_start_qpos.to(
                device=trajectory.device, dtype=trajectory.dtype
            )
            expected_hold_shape = (trajectory.shape[0], trajectory.shape[2])
            if hold_qpos.shape != expected_hold_shape:
                raise ValueError(
                    "Atomic full_start_qpos must match trajectory batch and joint "
                    "dimensions."
                )
            trajectory = torch.where(
                plan_success[:, None, None],
                trajectory,
                hold_qpos[:, None, :],
            )
        object_handle = self._objects.get(case.object_id or "")
        articulation_handle = self._articulations.get(case.object_id or "")
        settings = self._physics_settings()
        arrival_steps = self._arrival_steps(compiled, settings)
        lift_start = None if provider is None else provider.lift_segment_start(compiled)
        dynamics_cleared = False
        squared_tracking_error: torch.Tensor | None = None
        tracking_value_count = 0
        articulation_initial: torch.Tensor | None = None
        articulation_final: torch.Tensor | None = None
        articulation_peak: torch.Tensor | None = None
        articulation_joint_name: str | None = None
        effect_start = 0
        effect_stop = 0
        effect_sign = 1.0
        articulation_effect_end: torch.Tensor | None = None
        effect_end_index = -1
        rotation_window = (
            None
            if provider is None or not collect_metrics or object_handle is None
            else provider.object_rotation_segment(compiled)
        )
        rotation_axis_local: torch.Tensor | None = None
        rotation_sign = 1.0
        rotation_reference: torch.Tensor | None = None
        rotation_axis_world: torch.Tensor | None = None
        rotation_peak: torch.Tensor | None = None
        if rotation_window is not None:
            rotation_axis_local = torch.tensor(
                case.case_parameters["object_rotation_axis"],
                dtype=torch.float32,
                device=trajectory.device,
            )
            rotation_axis_local = rotation_axis_local / torch.linalg.vector_norm(
                rotation_axis_local
            )
            rotation_sign = float(case.case_parameters.get("object_rotation_sign", 1.0))
            if rotation_sign not in (-1.0, 1.0):
                raise ValueError("object_rotation_sign must be 1 or -1.")
        if collect_metrics:
            squared_tracking_error = torch.zeros(
                case.batch_size, dtype=trajectory.dtype, device=trajectory.device
            )
            if articulation_handle is not None:
                joint_value = case.case_parameters.get("target_joint")
                if not isinstance(joint_value, str) or not joint_value:
                    raise ValueError(
                        "Articulation cases must freeze a non-empty target_joint."
                    )
                articulation_joint_name = joint_value
                effect_sign = float(case.case_parameters.get("effect_joint_sign", 1.0))
                if not math.isfinite(effect_sign) or abs(effect_sign) <= 1.0e-8:
                    raise ValueError("effect_joint_sign must be finite and non-zero.")
                articulation_initial = articulation_handle.joint_qpos(joint_value)
                articulation_peak = torch.zeros_like(articulation_initial)
                segment_name = (
                    None
                    if provider is None
                    else provider.articulation_effect_segment(case)
                )
                if segment_name is None:
                    raise RuntimeError(
                        f"Atomic skill {case.skill_id!r} has an articulation target "
                        "but declares no effect segment."
                    )
                segment = compiled.segment(0, segment_name)
                # A passive articulation can respond a few physics steps after
                # the command segment ends.  Start at the manipulation segment,
                # then observe through release, retract, and the final hold.
                effect_start, effect_stop = segment.start, trajectory.shape[1]
                effect_end_index = segment.stop - 1
        observation_windows = (
            ()
            if provider is None or not collect_metrics
            else provider.replay_observation_windows(compiled, case)
        )
        if len({window.observation_id for window in observation_windows}) != len(
            observation_windows
        ):
            raise ValueError("Replay observation IDs must be unique within one case.")
        waypoint_count = trajectory.shape[1]
        for window in observation_windows:
            if window.stop > waypoint_count:
                raise ValueError(
                    f"Replay observation {window.observation_id!r} exceeds the "
                    "compiled trajectory."
                )
            if window.include_settle and window.stop != waypoint_count:
                raise ValueError(
                    "Only a window ending at the trajectory boundary may include "
                    "terminal settle."
                )
        trace_accumulators = {
            window.observation_id: _ReplayTraceAccumulator(window)
            for window in observation_windows
        }
        requested_scene_object_ids = {
            object_id
            for window in observation_windows
            for object_id in window.scene_object_ids
        }
        missing_scene_object_ids = requested_scene_object_ids - set(self._objects)
        if missing_scene_object_ids:
            raise ValueError(
                "Replay observations reference unknown scene objects: "
                f"{sorted(missing_scene_object_ids)}."
            )
        scene_object_handles = {
            object_id: self._objects[object_id]
            for object_id in requested_scene_object_ids
        }
        for waypoint_index in range(trajectory.shape[1]):
            self._capture_replay_samples(
                [
                    trace_accumulators[window.observation_id]
                    for window in observation_windows
                    if window.start == waypoint_index
                ],
                object_handle=object_handle,
                scene_object_handles=scene_object_handles,
                articulation_handle=articulation_handle,
                articulation_joint_name=articulation_joint_name,
            )
            if rotation_window is not None and waypoint_index == rotation_window[0]:
                assert object_handle is not None and rotation_axis_local is not None
                rotation_reference = (
                    object_handle.entity.get_local_pose(to_matrix=True)[:, :3, :3]
                    .to(device=trajectory.device, dtype=torch.float32)
                    .clone()
                )
                rotation_axis_world = rotation_reference @ rotation_axis_local
                rotation_peak = torch.zeros(
                    case.batch_size, dtype=torch.float32, device=trajectory.device
                )
            positions = trajectory[:, waypoint_index]
            self.robot.set_qpos(positions, target=True)
            for _ in range(arrival_steps[waypoint_index]):
                self.simulation.update(step=1)
                if (
                    rotation_window is not None
                    and rotation_reference is not None
                    and rotation_axis_world is not None
                    and rotation_peak is not None
                    and rotation_window[0] <= waypoint_index < rotation_window[1]
                ):
                    assert object_handle is not None
                    current_rotation = object_handle.entity.get_local_pose(
                        to_matrix=True
                    )[:, :3, :3].to(device=trajectory.device, dtype=torch.float32)
                    rotation_peak = torch.maximum(
                        rotation_peak,
                        signed_rotation_about_axis(
                            rotation_reference,
                            current_rotation,
                            rotation_axis_world,
                        )
                        * rotation_sign,
                    )
                if (
                    articulation_handle is not None
                    and articulation_joint_name is not None
                    and articulation_initial is not None
                    and articulation_peak is not None
                    and effect_start <= waypoint_index < effect_stop
                ):
                    current_joint = articulation_handle.joint_qpos(
                        articulation_joint_name
                    )
                    articulation_peak = torch.maximum(
                        articulation_peak,
                        (current_joint - articulation_initial) * effect_sign,
                    )
                self._capture_replay_samples(
                    [
                        trace_accumulators[window.observation_id]
                        for window in observation_windows
                        if window.start <= waypoint_index < window.stop
                    ],
                    object_handle=object_handle,
                    scene_object_handles=scene_object_handles,
                    articulation_handle=articulation_handle,
                    articulation_joint_name=articulation_joint_name,
                )
            if (
                articulation_handle is not None
                and articulation_joint_name is not None
                and waypoint_index == effect_end_index
            ):
                articulation_effect_end = articulation_handle.joint_qpos(
                    articulation_joint_name
                )
            if squared_tracking_error is not None:
                observed = self.robot.get_qpos()
                squared_tracking_error += ((observed - positions) ** 2).sum(dim=1)
                tracking_value_count += observed.shape[1]
            if (
                object_handle is not None
                and lift_start is not None
                and not dynamics_cleared
                and waypoint_index + 1 >= lift_start
            ):
                object_handle.entity.clear_dynamics()
                dynamics_cleared = True
        final_command = trajectory[:, -1]
        settling = [
            trace_accumulators[window.observation_id]
            for window in observation_windows
            if window.include_settle
        ]
        self._capture_replay_samples(
            settling,
            object_handle=object_handle,
            scene_object_handles=scene_object_handles,
            articulation_handle=articulation_handle,
            articulation_joint_name=articulation_joint_name,
        )
        for _ in range(settings.hold_steps):
            self.robot.set_qpos(final_command, target=True)
            self.simulation.update(step=settings.hold_sim_steps)
            if (
                articulation_handle is not None
                and articulation_joint_name is not None
                and articulation_initial is not None
                and articulation_peak is not None
            ):
                current_joint = articulation_handle.joint_qpos(articulation_joint_name)
                articulation_peak = torch.maximum(
                    articulation_peak,
                    (current_joint - articulation_initial) * effect_sign,
                )
            if squared_tracking_error is not None:
                observed = self.robot.get_qpos()
                squared_tracking_error += ((observed - final_command) ** 2).sum(dim=1)
                tracking_value_count += observed.shape[1]
            self._capture_replay_samples(
                settling,
                object_handle=object_handle,
                scene_object_handles=scene_object_handles,
                articulation_handle=articulation_handle,
                articulation_joint_name=articulation_joint_name,
            )
        if collect_metrics:
            if squared_tracking_error is None:
                raise RuntimeError("Timed replay lost its tracking accumulator.")
            if articulation_handle is not None and articulation_joint_name is not None:
                articulation_final = articulation_handle.joint_qpos(
                    articulation_joint_name
                )
            return _ReplayMetrics(
                squared_tracking_error=squared_tracking_error,
                tracking_value_count=tracking_value_count,
                physics_steps=sum(arrival_steps)
                + settings.hold_steps * settings.hold_sim_steps,
                articulation_joint_initial=articulation_initial,
                articulation_joint_final=articulation_final,
                articulation_joint_peak_signed_delta=articulation_peak,
                articulation_joint_effect_end=articulation_effect_end,
                object_rotation_peak_signed_rad=(
                    None if rotation_peak is None else rotation_peak * rotation_sign
                ),
                replay_traces={
                    observation_id: accumulator.finish()
                    for observation_id, accumulator in trace_accumulators.items()
                    if (
                        accumulator.tcp_pose
                        or accumulator.object_pose
                        or accumulator.articulation_joint
                        or accumulator.gripper_qpos
                        or accumulator.scene_object_poses
                    )
                },
            )
        return None

    def _arrival_steps(
        self, compiled: CompiledTrajectory, settings: _PhysicsReplaySettings
    ) -> list[int]:
        """Quantize the shared compiled clock once, preserving segment indices."""
        count = compiled.trajectory.positions.shape[1]
        if settings.clock == "fixed_samples":
            return [settings.steps_per_waypoint] * count
        dt = compiled.trajectory.dt
        if dt is None or dt.shape != compiled.trajectory.positions.shape[:2]:
            raise ValueError(
                "Compiled-clock replay requires aligned arrival intervals."
            )
        if (
            not bool(torch.isfinite(dt).all())
            or bool((dt < 0).any())
            or bool((dt[:, 0] != 0).any())
        ):
            raise ValueError(
                "Compiled-clock replay requires finite nonnegative intervals and a zero first arrival."
            )
        # Rows padded with zero-time terminal holds wait on the common clock.
        # Cumulative quantization bounds rounding error to one physics tick.
        physics_dt = float(self.simulation.sim_config.physics_dt)
        if not math.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError("physics_dt must be finite and positive.")
        arrivals = dt.to(torch.float64).amax(dim=0).cumsum(dim=0).cpu().tolist()
        result = []
        previous = 0
        for arrival in arrivals:
            ratio = arrival / physics_dt
            nearest = round(ratio)
            tick = (
                nearest
                if math.isclose(
                    ratio, nearest, abs_tol=max(1.0e-7, abs(ratio) * 1.0e-7)
                )
                else math.ceil(ratio)
            )
            result.append(tick - previous)
            previous = tick
        return result

    def hold_static(self) -> None:
        """Hold the current scene so a failed-case debug video has frames."""
        if self.simulation is None:
            raise RuntimeError("Atomic Skill runtime is not configured.")
        settings = self._physics_settings()
        for _ in range(max(settings.hold_steps, 1)):
            self.simulation.update(step=settings.hold_sim_steps)

    @staticmethod
    def _synchronize() -> None:
        """Synchronize CUDA around physical wall-time measurement."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _object_handle(self, object_id: str) -> AtomicObjectHandle:
        """Resolve a scene observation's object with an actionable error."""
        try:
            return self._objects[object_id]
        except KeyError as exc:
            raise ValueError(
                f"Unknown atomic object {object_id!r}; configured objects: "
                f"{sorted(self._objects)}."
            ) from exc
