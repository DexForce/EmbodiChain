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

"""Simulation-backed Task Program post-policies and validators.

The port in this module deliberately consumes the same explicit
:class:`SimulationSceneBinding` used to construct the semantic scene registry.
It never scans a simulation or guesses a native entity from a canonical name.
Post-policy actions remain inside the normal Gym ``env.step()`` path owned by
:class:`TaskProgramDemoBridge`. Rows eligible for the policy reuse full drive targets
so physical contact does not erase position-control preload; rows already
inactive use fresh measured-position holds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, MutableMapping
from copy import copy, deepcopy
from dataclasses import dataclass, fields, replace
import math
from types import MappingProxyType
from typing import Any, Literal, TYPE_CHECKING

import torch

from embodichain.compute.geometry.rigid import rigid_pose_displacement
from embodichain.compute.trajectory.stability import PositionRangeMonitor
from ._joint_window import JointWindowRecord
from embodichain.utils import configclass

from embodichain.lab.evaluation import (
    CompiledEvaluation,
    EvaluationStatus,
    RuleSession,
)
from embodichain.lab.evaluation.observations import (
    _advance_observation,
    _combine_selections,
    _rule_report,
)
from embodichain.lab.evaluation._recording import _is_segment_start_field
from embodichain.lab.evaluation._predicates import (
    _joint_range,
    _lift,
    _relative_position,
    _upright,
)
from embodichain.lab.gym.envs.objectives.conditions import evaluate_near_target
from embodichain.lab.gym.envs.settling import (
    DynamicSettleMonitor,
    DynamicSettleMonitorCfg,
    DynamicSettleSample,
    DynamicSettleState,
)

from embodichain.lab.task_program.compiler import (
    CompiledArticulationJointPositionValidator,
    CompiledConditionValidator,
    CompiledObjectNearRelativeTargetValidator,
    CompiledObjectNearTargetValidator,
    CompiledObjectStateValidator,
    CompiledPostPolicy,
    CompiledHoldPostPolicy,
    CompiledTaskProgramSegment,
    CompiledTaskProgramValidator,
)
from .bindings import SimulationSceneBinding
from .stability import (
    SimulationStabilityConditionCfg,
    SimulationJointStabilityConditionCfg,
)
from embodichain.lab.task_program.semantics.scene import SceneLinkRef

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers.observation_types import ObservationFrame
    from embodichain.lab.sim.atomic_skills.state import PlanningContext
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.sensors.base_sensor import BaseSensor
    from embodichain.lab.sim.sensors.support_observation import (
        SupportControlClock,
        NativeSupportResult,
        SupportObservationRegistry,
    )


@dataclass(frozen=True, slots=True)
class _SimulationSettleTarget:
    """One canonical entity resolved through an explicit native binding."""

    canonical_id: str
    kind: str
    native_entity: Any


@dataclass
class _PolicyLifetime:
    closed: bool = False
    observation_clock: object | None = None
    observation_generation: int = 0


@configclass
class SimulationSettleMonitorCfg(DynamicSettleMonitorCfg):
    """Explicit simulation measurement selection for an existing settle monitor.

    Plain :class:`DynamicSettleMonitorCfg` values retain reported velocities.
    Pose deltas are opt-in for rigid objects, require a control timestep, and
    leave the first observation unresolved. Preset names never select behavior.
    """

    measurement: Literal["reported_velocity", "pose_delta"] = "reported_velocity"
    """Read backend velocities or derive speeds from consecutive observed poses."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.measurement not in ("reported_velocity", "pose_delta"):
            raise ValueError(
                "Unsupported settle measurement; use reported_velocity or pose_delta."
            )

    def snapshot(self) -> SimulationSettleMonitorCfg:
        """Own and revalidate thresholds, cadence and explicit measurement mode."""
        return SimulationSettleMonitorCfg(
            **{field.name: getattr(self, field.name) for field in fields(self)}
        )


def default_simulation_settle_presets() -> Mapping[str, DynamicSettleMonitorCfg]:
    """Return independently owned built-in post-policy presets."""
    return MappingProxyType(
        {
            "rigid_object": DynamicSettleMonitorCfg(
                linear_velocity_threshold=0.03,
                angular_velocity_threshold=0.20,
                min_steps=10,
                max_steps=240,
                check_interval_steps=2,
                required_stable_checks=3,
            ),
            "articulation": DynamicSettleMonitorCfg(
                linear_velocity_threshold=0.02,
                angular_velocity_threshold=0.10,
                min_steps=10,
                max_steps=120,
                check_interval_steps=2,
                required_stable_checks=3,
            ),
        }
    )


def _json_speed_values(value: torch.Tensor) -> list[float | None]:
    """Convert speed evidence to finite JSON numbers or explicit unknowns."""
    return [
        float(item) if math.isfinite(float(item)) else None
        for item in value.detach().cpu().tolist()
    ]


class SimulationSegmentPolicyPort:
    """Execute built-in segment policies against explicitly bound simulation data.

    Args:
        simulation: Live simulation used only for UIDs declared in
            ``scene_binding``.
        robot: Live robot used to produce full target-qpos holds while the
            post-policy observes settling.
        scene_binding: Exact canonical-to-native scene declaration.
        settle_presets: Named settling policies. ``None`` installs the shared
            ``rigid_object`` and ``articulation`` presets.
        step_dt: Control interval in seconds; required only by explicit
            pose-delta measurement declarations.
        env_ids: Optional stable logical row IDs. They describe correlation,
            not simulator row indices; simulator rows remain ordered exactly as
            returned by the robot and bound entities.
        support_registry: Explicit runtime-owned native support observers for
            declared support conditions.
        sensor_suite: Host-owned sensor mapping driving those same instances.

    The same instance implements both ``SegmentPostPolicyPort`` and
    ``SegmentValidatorPort``. Unknown policy types, presets, canonical IDs, or
    native entities fail before an action is emitted.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        scene_binding: SimulationSceneBinding,
        *,
        settle_presets: (
            Mapping[
                str,
                DynamicSettleMonitorCfg
                | SimulationStabilityConditionCfg
                | SimulationJointStabilityConditionCfg,
            ]
            | None
        ) = None,
        env_ids: torch.Tensor | None = None,
        select_rows: bool = False,
        step_dt: float | None = None,
        support_registry: SupportObservationRegistry | None = None,
        sensor_suite: MutableMapping[str, BaseSensor] | None = None,
    ) -> None:
        if type(scene_binding) is not SimulationSceneBinding:
            raise TypeError("scene_binding must be exactly SimulationSceneBinding.")
        qpos = self._read_robot_qpos(robot, target=False)
        self._robot_qpos_shape = qpos.shape
        self._robot_qpos_device = qpos.device
        if env_ids is None:
            env_ids = torch.arange(
                qpos.shape[0],
                dtype=torch.long,
                device=qpos.device,
            )
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor or None.")
        if type(select_rows) is not bool:
            raise TypeError("select_rows must be bool")
        if env_ids.dtype != torch.long or env_ids.ndim != 1 or not env_ids.numel():
            raise ValueError("env_ids must be a nonempty int64 vector")
        if select_rows:
            if bool((env_ids < 0).any()) or bool((env_ids >= qpos.shape[0]).any()):
                raise ValueError("selected env_ids must index physical rows")
        elif env_ids.shape != (qpos.shape[0],):
            raise ValueError("env_ids must have one identifier per simulator row")
        if env_ids.device != qpos.device:
            raise ValueError("env_ids and robot qpos must share a device.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must contain unique values.")

        selected_presets = (
            default_simulation_settle_presets()
            if settle_presets is None
            else settle_presets
        )
        if not isinstance(selected_presets, Mapping) or not selected_presets:
            raise ValueError("settle_presets must be a non-empty mapping.")
        normalized_presets: dict[
            str,
            DynamicSettleMonitorCfg
            | SimulationStabilityConditionCfg
            | SimulationJointStabilityConditionCfg,
        ] = {}
        for preset_id, cfg in selected_presets.items():
            if (
                type(preset_id) is not str
                or not preset_id
                or preset_id != preset_id.strip()
            ):
                raise ValueError(
                    "Settle preset IDs must be non-empty strings without outer "
                    "whitespace."
                )
            if not isinstance(cfg, DynamicSettleMonitorCfg) and type(cfg) not in (
                SimulationStabilityConditionCfg,
                SimulationJointStabilityConditionCfg,
            ):
                raise TypeError(
                    "settle_presets require dynamic monitors or typed stability conditions."
                )
            normalized_presets[preset_id] = cfg.snapshot()

        if step_dt is not None and (
            isinstance(step_dt, bool)
            or not isinstance(step_dt, (int, float))
            or not math.isfinite(step_dt)
            or step_dt <= 0
        ):
            raise ValueError("step_dt must be finite and positive when supplied.")
        if step_dt is None and any(
            type(cfg)
            in (SimulationStabilityConditionCfg, SimulationJointStabilityConditionCfg)
            or (
                isinstance(cfg, SimulationSettleMonitorCfg)
                and cfg.measurement == "pose_delta"
            )
            for cfg in normalized_presets.values()
        ):
            raise ValueError(
                "Pose-delta or condition settling requires an explicit control step_dt."
            )

        self._simulation = simulation
        self._robot = robot
        self._scene_binding = scene_binding
        self._step_dt = step_dt
        self._env_ids = env_ids.clone()
        self._row_indices = (
            env_ids.clone()
            if select_rows
            else torch.arange(qpos.shape[0], device=qpos.device)
        )
        self._settle_presets = MappingProxyType(normalized_presets)
        (
            self._settle_targets,
            self._objects,
            self._articulations,
        ) = self._resolve_native_entities()
        self._joint_conditions = {}
        for name, cfg in normalized_presets.items():
            if type(cfg) is SimulationJointStabilityConditionCfg:
                art = self._articulations.get(cfg.entity)
                if art is None or cfg.joint not in art.joint_names:
                    raise ValueError(
                        "Joint stability requires its exact bound articulation and joint."
                    )
                self._joint_conditions[name] = (art, art.joint_names.index(cfg.joint))
        self._reference_links = {
            binding.entity_id: (
                self._articulations[binding.articulation_id],
                binding.native_link_name,
            )
            for binding in scene_binding.links
        }
        self._post_policy_results: dict[int, dict[str, object]] = {}
        self._post_policy_success: dict[int, torch.Tensor] = {}
        self._validator_results: dict[int, dict[str, object]] = {}
        self._validator_active_masks: dict[int, torch.Tensor] = {}
        self._lift_baselines: dict[int, torch.Tensor] = {}
        self._evaluation_specs: dict[str, CompiledEvaluation] = {}
        self._evaluation_groups: tuple[
            tuple[tuple[str, ...], CompiledEvaluation], ...
        ] = ()
        self._evaluation_fields: tuple[str, ...] = ()
        self._evaluation_initial = {}
        self._evaluation_sessions: dict[str, RuleSession] = {}
        self._evaluation_start: int | None = None
        self._evaluation_active: torch.Tensor | None = None
        self._evaluation_rows: tuple[int, ...] = ()
        self._lifetime = _PolicyLifetime()
        self._observation_clock: SupportControlClock | None = None
        self._observation_generation = 0
        self._hold_observer = None
        self._target_observer = None
        self._condition_runtime = None
        conditions = {
            name: cfg
            for name, cfg in normalized_presets.items()
            if type(cfg) is SimulationStabilityConditionCfg
        }
        if conditions:
            from ._stability_runtime import _ConditionStabilityRuntime

            self._condition_runtime = _ConditionStabilityRuntime(
                simulation,
                robot,
                scene_binding,
                conditions,
                step_dt=step_dt,
                registry=support_registry,
                sensor_suite=sensor_suite,
            )

    def bind_clock(self, clock: SupportControlClock) -> None:
        """Bind the existing environment clock for explicitly declared conditions.

        Args:
            clock: Canonical environment execution clock, never advanced here.
        """
        self._require_open()
        if self._condition_runtime is not None:
            self._condition_runtime.bind_clock(clock)
        if self._lifetime.observation_clock is not clock:
            self._lifetime.observation_clock = clock
            self._lifetime.observation_generation += 1
        self._observation_clock = clock
        self._observation_generation = self._lifetime.observation_generation

    @property
    def observation_clock(self) -> SupportControlClock | None:
        """Return this session's current observation clock, or None before binding.

        A later session can supersede this owner's observation binding. Reading
        a superseded clock raises instead of assigning new facts to an old
        execution. This guard does not manage active actions or cancellation.
        """
        self._require_open()
        if self._observation_clock is None:
            return None
        self.validate_observation_scope(self._observation_clock)
        return self._observation_clock

    def validate_observation_scope(self, clock: SupportControlClock) -> None:
        """Require the actual bridge's current observation ownership.

        Args:
            clock: Exact canonical clock supplied by the requesting bridge.

        Raises:
            ValueError: The session is unbound or superseded by another clock.
            RuntimeError: Runtime resources are closed.
        """
        self._require_open()
        if (
            clock is None
            or self._observation_clock is not clock
            or self._observation_generation != self._lifetime.observation_generation
        ):
            raise ValueError(
                "Observation capture belongs to another runtime/session clock."
            )

    def capture_support_observation(
        self, preset_id: str, *, clock: SupportControlClock
    ) -> NativeSupportResult:
        """Capture current support from this exact declared runtime session.

        Args:
            preset_id: Existing named condition in the sealed settle table.
            clock: The requesting bridge's canonical clock object.

        Returns:
            Owned native support with explicit validity/error masks, using the
            native rest-envelope COM tolerance. A policy-specific tolerance is
            confined to that policy's historical measurement. No task predicate
            or historical support is reconstructed here.

        Raises:
            RuntimeError: Runtime resources are closed.
            ValueError: The preset has no support declaration or this session
                no longer owns the observer's runtime clock binding.
            KeyError: The preset is not registered.
        """
        self._require_open()
        if self._condition_runtime is None:
            raise ValueError("No native support condition is registered.")
        return self._condition_runtime.capture_current_support(preset_id, clock=clock)

    def close_runtime_resources(self) -> None:
        """Mark condition sessions closed without destroying borrowed resources."""
        self._lifetime.closed = True
        if self._condition_runtime is not None:
            self._condition_runtime.close()

    def _require_open(self) -> None:
        if self._lifetime.closed:
            raise RuntimeError("Simulation policy runtime resources are closed.")

    @property
    def settle_preset_ids(self) -> tuple[str, ...]:
        """Return installed post-policy preset IDs in declaration order."""
        return tuple(self._settle_presets)

    def new_task_program_session(self) -> SimulationSegmentPolicyPort:
        """Create fresh episode state while retaining the validated scene bindings.

        Returns:
            A port sharing the same live scene and robot, with independent
            validator baselines, results and post-policy state.
        """
        self._require_open()
        session = copy(self)
        session._hold_observer = None
        session._target_observer = None
        session._post_policy_results = {}
        session._post_policy_success = {}
        session._validator_results = {}
        session._validator_active_masks = {}
        session._lift_baselines = {}
        session._evaluation_specs = {}
        session._evaluation_groups = ()
        session._evaluation_fields = ()
        session._evaluation_initial = {}
        session._evaluation_sessions = {}
        session._evaluation_start = None
        session._evaluation_active = None
        session._evaluation_rows = ()
        if self._condition_runtime is not None:
            session._condition_runtime = self._condition_runtime.new_session()
        return session

    def validate_policy(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> None:
        """Validate one post-policy against static bindings without observation.

        This method reads only the compiled declaration, installed preset
        table, and entities resolved when the port was constructed. It never
        samples velocity or qpos and never emits a controller action.
        """
        if type(policy) not in (CompiledPostPolicy, CompiledHoldPostPolicy):
            raise TypeError("policy must be exactly CompiledPostPolicy.")
        self._validate_segment_membership(segment, policy, kind="post policy")
        if type(policy) is CompiledHoldPostPolicy:
            if self._step_dt is None or policy.entity.entity_id not in self._objects:
                raise ValueError("Hold requires a rigid object and control step_dt.")
            return
        if policy.cfg.kind != "wait_stable":
            raise ValueError(
                f"Unsupported compiled post-policy kind {policy.cfg.kind!r}."
            )
        if policy.cfg.preset not in self._settle_presets:
            raise KeyError(
                f"Unknown settle preset {policy.cfg.preset!r}; available presets "
                f"are {sorted(self._settle_presets)}."
            )
        if (
            type(self._settle_presets[policy.cfg.preset])
            is SimulationJointStabilityConditionCfg
        ):
            if (
                policy.entity.entity_id
                != self._settle_presets[policy.cfg.preset].entity
            ):
                raise ValueError(
                    "Joint stability differs from the declared articulation."
                )
            return
        if (
            type(self._settle_presets[policy.cfg.preset])
            is SimulationStabilityConditionCfg
        ):
            self._condition_runtime.validate_policy(policy, segment=segment)
            return
        entity_id = policy.entity.entity_id
        target = self._settle_targets.get(entity_id)
        if target is None:
            raise KeyError(
                f"Canonical settle entity {entity_id!r} has no explicit native "
                "dynamic binding."
            )
        preset = self._settle_presets[policy.cfg.preset]
        if (
            isinstance(preset, SimulationSettleMonitorCfg)
            and preset.measurement == "pose_delta"
            and target.kind != "rigid_object"
        ):
            raise ValueError("Pose-delta settling requires a rigid-object target.")
        if target.kind == "rigid_object" and bool(
            getattr(target.native_entity, "is_non_dynamic", False)
        ):
            raise ValueError(
                f"Canonical settle entity {entity_id!r} is static or kinematic."
            )

    def actions(
        self,
        policy: Any,
        *,
        segment: Any,
        active_mask: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        """Yield full target-qpos hold actions until rows settle or time out.

        Args:
            policy: Exact compiled ``wait_stable`` policy.
            segment: Exact segment that owns ``policy``.
            active_mask: Rows that remain eligible after runtime execution and
                preceding post-policies. Inactive rows are held safely but do
                not participate in settling, timeout, or success results.

        Yields:
            Fresh full target-qpos hold commands consumed by ordinary
            ``env.step()``. Reading drive targets instead of measured joint
            positions preserves contact preload in position-controlled tools.
            Rows inactive when the policy starts use fresh measured qpos holds.

        Timeout is a normal row-local result boundary. Timed-out rows are
        exposed through :meth:`post_policy_result` and
        :meth:`post_policy_metadata`; no batch-level exception is raised.
        """
        self._require_open()
        self.validate_policy(policy, segment=segment)
        active_mask = self._validate_active_mask(active_mask)
        if type(policy) is CompiledHoldPostPolicy:
            yield from self._hold_actions(
                policy, segment=segment, active_mask=active_mask
            )
            return
        preset = self._settle_presets[policy.cfg.preset]
        if type(preset) is SimulationJointStabilityConditionCfg:
            yield from self._joint_condition_actions(
                policy, segment=segment, active_mask=active_mask
            )
            return
        if type(preset) is SimulationStabilityConditionCfg:
            yield from self._condition_runtime.actions(
                policy, segment=segment, active_mask=active_mask
            )
            return
        entity_id = policy.entity.entity_id
        target = self._settle_targets[entity_id]
        explicit_measurement = isinstance(preset, SimulationSettleMonitorCfg)
        use_pose_delta = explicit_measurement and preset.measurement == "pose_delta"
        measurement_metadata = (
            {"measurement_source": preset.measurement} if explicit_measurement else {}
        )

        result_key = id(policy)
        self._post_policy_results.pop(result_key, None)
        self._post_policy_success.pop(result_key, None)
        if not bool(active_mask.any().item()):
            self._post_policy_success[result_key] = active_mask.clone()
            self._post_policy_results[result_key] = {
                "kind": policy.cfg.kind,
                "entity_id": entity_id,
                "preset": policy.cfg.preset,
                **measurement_metadata,
                "source_path": list(policy.source_path),
                "status": "skipped",
                "active_mask": active_mask.detach().cpu().tolist(),
                "thresholds": self._settle_threshold_metadata(preset),
                "state": self._empty_settle_state_metadata(active_mask),
            }
            return

        active_rows = self._row_indices[active_mask]
        monitor = DynamicSettleMonitor(preset, self._env_ids[active_mask])
        previous_pose: torch.Tensor | None = None
        elapsed_steps = 0
        while True:
            self._require_open()

            if use_pose_delta:
                sample, previous_pose = self._measure_pose_delta_settle_target(
                    target, row_indices=active_rows, previous_pose=previous_pose
                )
            else:
                sample = self._measure_settle_target(target, row_indices=active_rows)
            state = monitor.observe(
                (sample,),
                elapsed_steps=elapsed_steps,
            )
            settled_mask = torch.zeros_like(active_mask)
            settled_mask[active_mask] = state.settled_mask
            self._post_policy_results[result_key] = {
                "kind": policy.cfg.kind,
                "entity_id": entity_id,
                "preset": policy.cfg.preset,
                **measurement_metadata,
                "source_path": list(policy.source_path),
                "active_mask": active_mask.detach().cpu().tolist(),
                "status": (
                    "settled"
                    if bool(state.settled_mask.all().item())
                    else (
                        "timed_out"
                        if bool(state.timeout_mask.any().item())
                        else "running"
                    )
                ),
                "thresholds": self._settle_threshold_metadata(preset),
                "state": self._expand_settle_state_metadata(state, active_mask),
            }
            self._post_policy_success[result_key] = settled_mask
            if bool(state.settled_mask.all().item()):
                return
            if bool(state.timeout_mask.any().item()):
                return
            yield self._hold_robot_qpos(active_mask)
            elapsed_steps += 1

    def bind_target_observer(
        self, observer: Callable[[str, object], torch.Tensor | None]
    ) -> None:
        """Bind a bridge-local reader of targets frozen during call grounding."""
        if not callable(observer):
            raise TypeError("Target observer must be callable.")
        self._target_observer = observer

    def bind_hold_observer(
        self,
        observer: Callable[
            [str, torch.Tensor],
            tuple[PlanningContext, torch.Tensor, torch.Tensor | None],
        ],
    ) -> None:
        """Bind the bridge's measured runtime observer to this policy session."""
        if not callable(observer):
            raise TypeError("Hold observer must be callable.")
        self._hold_observer = observer

    def _hold_actions(self, policy, *, segment, active_mask):
        from embodichain.lab.task_program.semantics.calls import (
            SemanticRelativePose,
        )
        from embodichain.lab.sim.atomic_skills.goals import resolve_pose_goal

        cfg = policy.cfg
        clock = self.observation_clock
        if clock is None or not callable(self._hold_observer):
            raise ValueError(
                "Hold requires a bound execution clock and measured observer."
            )
        start = clock.step_index
        start_time = clock.now()
        required = math.ceil(cfg.duration_s / self._step_dt)
        accepted = active_mask.clone()
        key = id(policy)
        target = None
        elapsed = 0
        status = "cancelled"
        reasons = [None] * active_mask.numel()
        observed_samples = torch.zeros_like(active_mask, dtype=torch.long)
        valid_pose_samples = torch.zeros_like(observed_samples)
        attachment_samples = torch.zeros_like(observed_samples)
        maximum_position_error = torch.zeros_like(active_mask, dtype=torch.float32)
        maximum_orientation_error = torch.zeros_like(maximum_position_error)
        try:
            for tick in range(required + 1):
                self._require_open()
                self.validate_observation_scope(clock)
                if clock.step_index != start + tick or not math.isclose(
                    clock.now(),
                    start_time + tick * self._step_dt,
                    rel_tol=0,
                    abs_tol=1e-8,
                ):
                    raise ValueError(
                        "Hold requires acknowledged consecutive environment steps."
                    )
                if not accepted.any():
                    status = "failed" if active_mask.any() else "skipped"
                    break
                context, attached, previous_target = self._hold_observer(
                    cfg.object, accepted
                )
                self.validate_observation_scope(clock)
                if clock.step_index != start + tick:
                    raise ValueError("Hold observation changed its execution clock.")
                if target is None:
                    preceding = segment.calls[-1]
                    reuse = getattr(
                        getattr(preceding.call, "object", None), "entity_id", None
                    ) == cfg.object and any(
                        item == policy.target_selection
                        for item in preceding.target_selections
                    )
                    declaration = policy.target_pose
                    target = (
                        previous_target.clone()
                        if reuse and previous_target is not None
                        else resolve_pose_goal(
                            (
                                declaration.to_goal()
                                if type(declaration) is SemanticRelativePose
                                else declaration.to_matrix()
                            ),
                            context,
                            name="hold.at",
                        )
                    )
                pose = context.scene.entities[cfg.object].pose.to(target)
                if pose.ndim == 2:
                    pose = pose.unsqueeze(0).expand(active_mask.numel(), -1, -1)
                if target.ndim == 2:
                    target = target.unsqueeze(0).expand_as(pose).clone()
                position_error, rotation_error = rigid_pose_displacement(target, pose)
                finite = torch.isfinite(position_error) & torch.isfinite(rotation_error)
                within = (
                    finite
                    & (position_error <= cfg.position_tolerance)
                    & (rotation_error <= cfg.orientation_tolerance)
                )
                current = attached & within
                observed_samples += accepted.to(torch.long)
                valid_pose = accepted & finite
                valid_pose_samples += valid_pose.to(torch.long)
                attachment_samples += (accepted & attached).to(torch.long)
                maximum_position_error = torch.maximum(
                    maximum_position_error, torch.where(valid_pose, position_error, 0.0)
                )
                maximum_orientation_error = torch.maximum(
                    maximum_orientation_error,
                    torch.where(valid_pose, rotation_error, 0.0),
                )
                diagnostics = (
                    torch.stack((accepted & ~current, attached, finite), dim=-1)
                    .detach()
                    .cpu()
                    .tolist()
                )
                for row, (failed, row_attached, row_finite) in enumerate(diagnostics):
                    if failed:
                        reasons[row] = (
                            "attachment_unavailable_or_lost"
                            if not row_attached
                            else (
                                "invalid_observation"
                                if not row_finite
                                else "pose_out_of_tolerance"
                            )
                        )
                accepted &= current
                elapsed = tick
                if not accepted.any():
                    status = "failed"
                    break
                if tick == required:
                    status = (
                        "satisfied"
                        if bool((accepted | ~active_mask).all())
                        else "failed"
                    )
                    break
                if accepted.any():
                    yield self._hold_robot_qpos(accepted)
        except GeneratorExit:
            raise
        except BaseException:
            status = "interrupted"
            raise
        finally:
            if status not in ("satisfied", "failed", "skipped"):
                accepted = torch.zeros_like(active_mask)
            self._post_policy_success[key] = accepted.clone()
            self._post_policy_results[key] = dict(
                kind="hold",
                status=status,
                object_id=cfg.object,
                target_id=policy.target_selection.target_id,
                duration_s=cfg.duration_s,
                elapsed_s=elapsed * self._step_dt,
                required_steps=required,
                elapsed_steps=elapsed,
                position_tolerance=cfg.position_tolerance,
                orientation_tolerance=cfg.orientation_tolerance,
                env_ids=self._env_ids.cpu().tolist(),
                active_mask=active_mask.cpu().tolist(),
                accepted_mask=accepted.cpu().tolist(),
                failure_reasons=reasons,
                observation_count=observed_samples.cpu().tolist(),
                attachment_verified_observations=attachment_samples.cpu().tolist(),
                max_position_error=_json_speed_values(
                    torch.where(
                        valid_pose_samples > 0, maximum_position_error, float("nan")
                    )
                ),
                max_orientation_error=_json_speed_values(
                    torch.where(
                        valid_pose_samples > 0, maximum_orientation_error, float("nan")
                    )
                ),
                target_pose=None if target is None else target.detach().cpu().tolist(),
            )

    def _joint_condition_actions(
        self,
        policy: CompiledPostPolicy,
        *,
        segment: CompiledTaskProgramSegment,
        active_mask: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        """Orchestrate sampling and holds around the pure coordinate monitor."""
        cfg = self._settle_presets[policy.cfg.preset]
        assert type(cfg) is SimulationJointStabilityConditionCfg
        art, index = self._joint_conditions[policy.cfg.preset]
        key = id(policy)
        self._post_policy_results.pop(key, None)
        self._post_policy_success.pop(key, None)
        clock = self.observation_clock
        hold = self._read_robot_qpos(self._robot, target=False).clone()
        anchor = art.get_qpos()[:, index].clone()
        if active_mask.shape != anchor.shape or active_mask.device != anchor.device:
            raise ValueError("Joint stability rows must match the observed batch.")
        monitor = PositionRangeMonitor(
            anchor,
            required_steps=math.ceil(cfg.duration / self._step_dt),
            position_drift=cfg.position_drift,
        )
        timeout = math.ceil(cfg.timeout / self._step_dt)
        if clock is not None and not math.isclose(
            clock.step_dt, self._step_dt, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(
                "Joint observation clock differs from the control cadence."
            )
        start = None if clock is None else (clock.step_index, clock.now())
        history = JointWindowRecord(
            min(timeout + 1, cfg.max_observation_samples),
            initial=anchor,
            condition=cfg.to_condition_dict(),
            env_ids=self._env_ids,
            native_uid=next(
                value.simulation_uid
                for value in self._scene_binding.articulations
                if value.entity_id == cfg.entity
            ),
            segment_id=segment.segment_id,
            policy_source_path=list(policy.source_path),
            observation_generation=self._observation_generation,
            clock_start=start,
            step_dt=self._step_dt,
        )
        timestamp = None
        step = None
        qpos = anchor
        finite = torch.isfinite(qpos)
        unavailable = torch.zeros_like(active_mask)
        accepted = torch.zeros_like(active_mask)
        completed = False
        termination = "cancelled"
        interrupted_reason = None
        try:
            for tick in range(timeout + 1):
                self._require_open()
                if clock is not None:
                    self.validate_observation_scope(clock)
                    if clock.step_index != start[0] + tick or not math.isclose(
                        clock.now(),
                        start[1] + tick * self._step_dt,
                        rel_tol=0,
                        abs_tol=1e-8,
                    ):
                        raise ValueError(
                            "Joint stability requires acknowledged environment steps."
                        )
                observed = art.get_qpos()[:, index].clone()
                if (
                    observed.shape != anchor.shape
                    or observed.device != anchor.device
                    or observed.dtype != anchor.dtype
                ):
                    raise ValueError(
                        "Joint observations changed batch shape, dtype or device."
                    )
                if clock is not None:
                    acquired_at, acquired_step = clock.now(), clock.step_index
                    self.validate_observation_scope(clock)
                    if acquired_step != start[0] + tick or not math.isclose(
                        acquired_at,
                        start[1] + tick * self._step_dt,
                        rel_tol=0,
                        abs_tol=1e-8,
                    ):
                        raise ValueError(
                            "The execution clock changed during joint acquisition."
                        )
                    timestamp, step = acquired_at, acquired_step
                qpos = observed
                history.append(qpos, step, timestamp)
                finite = torch.isfinite(qpos)
                unavailable |= active_mask & ~finite
                within = (qpos - cfg.target).abs() <= cfg.tolerance
                accepted = (
                    monitor.observe(qpos, within, elapsed_steps=tick) & active_mask
                )
                if cfg.invalid_observation_policy == "isolate_rows":
                    accepted &= ~unavailable
                    done = (accepted | unavailable | ~active_mask).all()
                else:
                    done = (accepted | ~active_mask).all()
                if (
                    done
                    or (
                        cfg.invalid_observation_policy == "abort_batch"
                        and not finite[active_mask].all()
                    )
                    or tick == timeout
                ):
                    termination = (
                        "satisfied"
                        if (accepted | ~active_mask).all()
                        else (
                            "invalid_observation"
                            if (
                                cfg.invalid_observation_policy == "abort_batch"
                                and not finite[active_mask].all()
                            )
                            or (unavailable | accepted | ~active_mask).all()
                            else "timed_out"
                        )
                    )
                    completed = True
                    break
                yield hold.clone()
        except GeneratorExit:
            raise
        except BaseException as error:
            termination = "interrupted"
            interrupted_reason = str(error) or type(error).__name__
            raise
        finally:
            # Never retain a success when an exact policy execution was cancelled.
            result = accepted if completed else torch.zeros_like(accepted)
            if completed:
                self._post_policy_success[key] = result.clone()
            self._post_policy_results[key] = history.finish(
                qpos,
                active_mask=active_mask,
                unavailable=unavailable,
                accepted=result,
                completed=completed,
                termination=termination,
                invalid_observation_policy=cfg.invalid_observation_policy,
                interruption_reason=interrupted_reason,
            )

    def post_policy_result(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> torch.Tensor:
        """Return the latest independently owned per-row settling result."""
        if type(policy) not in (CompiledPostPolicy, CompiledHoldPostPolicy):
            raise TypeError("policy must be exactly CompiledPostPolicy.")
        self._validate_segment_membership(segment, policy, kind="post policy")
        if (
            type(self._settle_presets.get(getattr(policy.cfg, "preset", None)))
            is SimulationStabilityConditionCfg
        ):
            return self._condition_runtime.post_policy_result(policy, segment=segment)
        result = self._post_policy_success.get(id(policy))
        if result is None:
            raise RuntimeError("Post-policy result is unavailable before execution.")
        return result.clone()

    def post_policy_metadata(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> Mapping[str, object]:
        """Return the latest JSON-safe settling trace for one policy.

        Dynamic-settle traces retain their latest running or terminal status.
        Joint-window records become available on completion, cancellation or
        interruption, with execution termination separate from row conditions.
        """
        if type(policy) not in (CompiledPostPolicy, CompiledHoldPostPolicy):
            raise TypeError("policy must be exactly CompiledPostPolicy.")
        self._validate_segment_membership(segment, policy, kind="post policy")
        if (
            type(self._settle_presets.get(getattr(policy.cfg, "preset", None)))
            is SimulationStabilityConditionCfg
        ):
            return self._condition_runtime.post_policy_metadata(policy, segment=segment)
        metadata = self._post_policy_results.get(id(policy))
        if metadata is None:
            raise RuntimeError("Post-policy metadata is unavailable before execution.")
        return deepcopy(metadata)

    def begin_segment(self, *, segment: Any, active_mask: torch.Tensor | None) -> None:
        """Capture lift baselines just before the segment plans and executes.

        This method only reads observed poses. Ineligible rows remain rejected,
        and non-finite observations reject their own rows during validation.
        """
        self._require_open()
        if type(segment) is not CompiledTaskProgramSegment:
            raise TypeError("segment must be exactly CompiledTaskProgramSegment.")
        active = (
            torch.ones_like(self._env_ids, dtype=torch.bool)
            if active_mask is None
            else self._validate_active_mask(active_mask)
        )
        from .bindings import _bind_evaluation_joint_names

        self._evaluation_specs = _bind_evaluation_joint_names(
            {
                validator.cfg.ref: validator.spec
                for validator in segment.validators
                if type(validator) is CompiledConditionValidator
            },
            self._robot,
        )
        self._evaluation_groups = _combine_selections(self._evaluation_specs)
        required = tuple(
            dict.fromkeys(
                field
                for spec in self._evaluation_specs.values()
                for field in spec.required_fields
            )
        )
        self._evaluation_fields = tuple(
            field for field in required if not _is_segment_start_field(field)
        )
        self._evaluation_initial = {}
        self._evaluation_sessions.clear()
        self._evaluation_start = None
        self._evaluation_active = None
        self._evaluation_rows = ()
        if self._evaluation_specs:
            clock = self.observation_clock
            if clock is None:
                raise ValueError("Condition validators require a bound control clock.")
            self._evaluation_start = clock.step_index + 1
            self._evaluation_active = active.clone()
            self._evaluation_rows = tuple(
                self._row_indices[active].detach().cpu().tolist()
            )
            for field in required:
                if _is_segment_start_field(field):
                    uid = field.removeprefix("rigid_objects/").rsplit("/", 1)[0]
                    self._evaluation_initial[field] = self._segment_start_pose(uid)
        for validator in segment.validators:
            self.validate_validator(validator, segment=segment)
            if type(validator) is CompiledObjectStateValidator:
                if id(validator) in self._validator_active_masks:
                    raise RuntimeError(
                        "Segment validator baseline was already captured."
                    )
                self._validator_active_masks[id(validator)] = active.clone()
                if validator.cfg.kind == "object_lifted":
                    self._lift_baselines[id(validator)] = self._read_pose(
                        self._objects[validator.object.entity_id],
                        entity_id=validator.object.entity_id,
                        require_finite=False,
                    )

    def _segment_start_pose(self, uid):
        from embodichain.lab.gym.envs.managers.observation_types import (
            ObservationSample,
        )
        from embodichain.lab.gym.envs.managers.observations import _state_descriptor

        pose = (
            self._simulation.get_rigid_object(uid)
            .get_local_pose(to_matrix=False)
            .detach()
            .clone()
        )
        valid = torch.isfinite(pose).all(-1) & (pose[:, 3:] != 0).any(-1)
        descriptor = _state_descriptor(uid, "segment_start_pose", (7,), pose.dtype, ())
        return ObservationSample(descriptor, pose, valid, (None,) * len(pose))

    def _observe_segment_frame(
        self, frame: ObservationFrame, *, step: int, episode_owner: EpisodeContext
    ) -> None:
        start = self._evaluation_start
        if start is None or not self._evaluation_rows:
            return
        self._require_open()
        if self._evaluation_initial:
            frame = replace(frame, fields={**frame.fields, **self._evaluation_initial})
        measurements = {}
        for refs, spec in self._evaluation_groups:
            session = self._evaluation_sessions.get(refs[0])
            if session is None:
                rows = self._evaluation_rows
                session = RuleSession(
                    spec,
                    env_ids=rows,
                    episode_ids=tuple(episode_owner.episode_ids[row] for row in rows),
                    start_step=start,
                    device=self._env_ids.device,
                    scope=f"segment@{start}",
                )
                self._evaluation_sessions.update(dict.fromkeys(refs, session))
            _advance_observation(
                session,
                frame,
                step=step,
                episode_owner=episode_owner,
                measurements=measurements,
            )

    def _validate_condition(
        self, validator: CompiledConditionValidator
    ) -> torch.Tensor:
        active = self._evaluation_active
        if active is None or validator.cfg.ref not in self._evaluation_specs:
            raise RuntimeError(
                "Condition validation requires the current segment to start."
            )
        session = self._evaluation_sessions.get(validator.cfg.ref)
        status = torch.full_like(self._env_ids, EvaluationStatus.NOT_RUN)
        status[active] = EvaluationStatus.UNAVAILABLE
        witness_steps = torch.full(
            (active.numel(), 2), -1, dtype=torch.long, device=active.device
        )
        witness_revisions = torch.full_like(witness_steps, -1)
        graph_trace = {}
        if session is not None:
            snapshot = session.snapshot(closed=True)
            root = snapshot.rules[validator.cfg.ref]
            status[active] = root.status
            if root.witness_steps is not None:
                witness_steps[active] = root.witness_steps
                witness_revisions[active] = root.witness_revisions
            graph_trace = {
                "evaluation_identity": self._evaluation_specs[
                    validator.cfg.ref
                ].identity,
                "rules": {
                    name: _rule_report(snapshot.rules[name])
                    for name, _ in validator.spec._nodes
                },
            }
        accepted = status == EvaluationStatus.SATISFIED
        self._validator_results[id(validator)] = {
            "kind": validator.cfg.kind,
            "ref": validator.cfg.ref,
            "env_ids": self._env_ids.cpu().tolist(),
            "status": [
                EvaluationStatus(value).name.lower() for value in status.cpu().tolist()
            ],
            **graph_trace,
            "witness_steps": witness_steps.cpu().tolist(),
            "witness_revisions": witness_revisions.cpu().tolist(),
            "accepted_mask": accepted.cpu().tolist(),
        }
        return accepted

    def validate_validator(
        self,
        validator: Any,
        *,
        segment: Any,
    ) -> None:
        """Validate one validator against static bindings without observation."""
        if type(validator) not in (
            CompiledConditionValidator,
            CompiledObjectNearTargetValidator,
            CompiledObjectNearRelativeTargetValidator,
            CompiledArticulationJointPositionValidator,
            CompiledObjectStateValidator,
        ):
            raise TypeError("validator must be an exact compiled validator.")
        self._validate_segment_membership(segment, validator, kind="validator")
        if type(validator) is CompiledConditionValidator:
            return
        if type(validator) is CompiledObjectStateValidator:
            for ref in (validator.object, validator.reference):
                bindings = (
                    self._reference_links
                    if type(ref) is SceneLinkRef
                    else self._objects
                )
                if ref is not None and ref.entity_id not in bindings:
                    raise KeyError(
                        f"Canonical validator object {ref.entity_id!r} has no explicit object binding."
                    )
            return
        if type(validator) is CompiledObjectNearTargetValidator:
            if validator.cfg.kind != "object_near_target":
                raise ValueError(
                    f"Unsupported compiled validator kind {validator.cfg.kind!r}."
                )
            entity_id = validator.object.entity_id
            if entity_id not in self._objects:
                raise KeyError(
                    f"Canonical validator object {entity_id!r} has no explicit "
                    "object binding."
                )
            return

        if type(validator) is CompiledObjectNearRelativeTargetValidator:
            if validator.cfg.kind != "object_near_relative_target":
                raise ValueError(
                    f"Unsupported compiled validator kind {validator.cfg.kind!r}."
                )
            for role, entity_id in (
                ("object", validator.object.entity_id),
                ("reference", validator.reference.entity_id),
            ):
                if entity_id not in self._objects:
                    raise KeyError(
                        f"Canonical validator {role} {entity_id!r} has no explicit "
                        "object binding."
                    )
            return

        if validator.cfg.kind != "articulation_joint_position":
            raise ValueError(
                f"Unsupported compiled validator kind {validator.cfg.kind!r}."
            )
        entity_id = validator.articulation.entity_id
        articulation = self._articulations.get(entity_id)
        if articulation is None:
            raise KeyError(
                f"Canonical validator articulation {entity_id!r} has no explicit "
                "articulation binding."
            )
        self._articulation_joint_index(
            articulation,
            entity_id=entity_id,
            joint_name=validator.cfg.joint,
        )

    def validate(self, validator: Any, *, segment: Any) -> torch.Tensor:
        """Observe one explicitly bound entity against its validator contract.

        Args:
            validator: Exact compiled segment validator.
            segment: Exact segment that owns ``validator``.

        Returns:
            Boolean tensor with one result per simulation row.
        """
        self._require_open()
        self.validate_validator(validator, segment=segment)
        if type(validator) is CompiledConditionValidator:
            return self._validate_condition(validator)
        if type(validator) is CompiledObjectStateValidator:
            return self._validate_object_state(validator)
        if type(validator) is CompiledArticulationJointPositionValidator:
            return self._validate_articulation_joint_position(validator)
        if type(validator) is CompiledObjectNearRelativeTargetValidator:
            return self._validate_object_near_relative_target(validator)
        if type(validator) is not CompiledObjectNearTargetValidator:
            raise TypeError("validator must be an exact compiled validator.")
        entity_id = validator.object.entity_id
        entity = self._objects[entity_id]
        pose = self._read_pose(entity, entity_id=entity_id)
        current_position = pose[:, :3, 3]
        from embodichain.lab.task_program.semantics.calls import (
            SemanticRelativePose,
        )

        if type(validator.target_pose) is SemanticRelativePose:
            if not callable(self._target_observer):
                raise ValueError(
                    "Scene-relative acceptance requires a grounded target observer."
                )
            resolved = self._target_observer(entity_id, validator.target_pose)
            if resolved is None:
                raise ValueError(
                    "Scene-relative acceptance has no matching grounded call target."
                )
            target_position = resolved.to(current_position)[..., :3, 3]
        else:
            target_position = validator.target_pose.position.to(
                device=current_position.device,
                dtype=current_position.dtype,
            )
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0).expand_as(current_position)
        elif target_position.shape != current_position.shape:
            raise ValueError(
                "Validator target batch must be unbatched or match simulator rows."
            )
        measured = evaluate_near_target(
            current_position,
            target_position,
            float(validator.cfg.position_tolerance),
            inclusive=True,
        )
        error = measured.metrics["distance"]
        accepted = measured.satisfied
        self._validator_results[id(validator)] = {
            "kind": validator.cfg.kind,
            "object_id": entity_id,
            "target_id": validator.target_selection.target_id,
            "target_value_index": validator.target_selection.value_index,
            "source_path": list(validator.source_path),
            "position_tolerance": float(validator.cfg.position_tolerance),
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "object_position": current_position.detach().cpu().tolist(),
            "target_position": target_position.detach().cpu().tolist(),
            "position_error": error.detach().cpu().tolist(),
            "accepted_mask": accepted.detach().cpu().tolist(),
        }
        return accepted

    def validator_metadata(
        self,
        validator: Any,
        *,
        segment: Any,
    ) -> Mapping[str, object]:
        """Return an owned JSON-safe trace for one completed validator."""
        if type(validator) not in (
            CompiledConditionValidator,
            CompiledObjectNearTargetValidator,
            CompiledObjectNearRelativeTargetValidator,
            CompiledArticulationJointPositionValidator,
            CompiledObjectStateValidator,
        ):
            raise TypeError("validator must be an exact compiled validator.")
        self._validate_segment_membership(segment, validator, kind="validator")
        metadata = self._validator_results.get(id(validator))
        if metadata is None:
            raise RuntimeError("Validator metadata is unavailable before validation.")
        return deepcopy(metadata)

    def _validate_object_near_relative_target(
        self,
        validator: CompiledObjectNearRelativeTargetValidator,
    ) -> torch.Tensor:
        """Compare one object with a fresh reference pose plus displacement."""
        object_id = validator.object.entity_id
        reference_id = validator.reference.entity_id
        object_pose = self._read_pose(
            self._objects[object_id],
            entity_id=object_id,
        )
        reference_pose = self._read_pose(
            self._objects[reference_id],
            entity_id=reference_id,
        )
        object_position = object_pose[:, :3, 3]
        reference_position = reference_pose[:, :3, 3]
        displacement = object_position.new_tensor(validator.cfg.displacement)
        target_position = reference_position + displacement
        dtype = torch.promote_types(object_position.dtype, target_position.dtype)
        measured = evaluate_near_target(
            object_position.to(dtype),
            target_position.to(dtype),
            float(validator.cfg.position_tolerance),
        )
        error = measured.metrics["distance"]
        accepted = measured.satisfied & torch.isfinite(error)
        self._validator_results[id(validator)] = {
            "kind": validator.cfg.kind,
            "object_id": object_id,
            "reference_id": reference_id,
            "source_path": list(validator.source_path),
            "displacement": list(validator.cfg.displacement),
            "position_tolerance": float(validator.cfg.position_tolerance),
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "object_position": object_position.detach().cpu().tolist(),
            "reference_position": reference_position.detach().cpu().tolist(),
            "target_position": target_position.detach().cpu().tolist(),
            "position_error": error.detach().cpu().tolist(),
            "accepted_mask": accepted.detach().cpu().tolist(),
        }
        return accepted

    def _validate_object_state(
        self, validator: CompiledObjectStateValidator
    ) -> torch.Tensor:
        """Measure a declared object contract without changing physics state."""
        entity_id = validator.object.entity_id
        pose = self._read_pose(
            self._objects[entity_id], entity_id=entity_id, require_finite=False
        )
        accepted = torch.isfinite(pose).all(dim=(1, 2))
        active = self._validator_active_masks.get(id(validator))
        if active is not None:
            accepted &= active.to(device=accepted.device)
        evidence: dict[str, object] = {}
        if validator.cfg.kind == "object_lifted":
            baseline = self._lift_baselines.get(id(validator))
            if baseline is None:
                raise RuntimeError(
                    "object_lifted requires begin_segment before execution."
                )
            baseline = baseline.to(device=pose.device, dtype=pose.dtype)
            measured = _lift(pose, baseline, float(validator.cfg.minimum_lift))
            accepted &= measured.valid & measured.satisfied
            evidence = {
                "minimum_lift": validator.cfg.minimum_lift,
                "initial_height": _json_speed_values(baseline[:, 2, 3]),
                "measured_lift": _json_speed_values(measured.metrics["lift"]),
            }
        elif validator.cfg.kind == "object_relative_position":
            assert validator.reference is not None
            if type(validator.reference) is SceneLinkRef:
                articulation, native_name = self._reference_links[
                    validator.reference.entity_id
                ]
                reference = articulation.get_link_pose(
                    native_name,
                    env_ids=self._row_indices.detach().cpu().tolist(),
                    to_matrix=True,
                )
                if (
                    not isinstance(reference, torch.Tensor)
                    or not reference.is_floating_point()
                    or reference.shape != pose.shape
                ):
                    raise ValueError("Reference link pose must have shape (B, 4, 4).")
            else:
                reference = self._read_pose(
                    self._objects[validator.reference.entity_id],
                    entity_id=validator.reference.entity_id,
                    require_finite=False,
                )
            measured = _relative_position(
                pose,
                reference.to(device=pose.device, dtype=pose.dtype),
                pose.new_tensor(validator.cfg.position_offset),
                validator.cfg.position_tolerance,
                parent_axes=validator.cfg.reference_frame == "parent",
            )
            accepted &= measured.satisfied
            error = measured.metrics["position_error"]
            evidence = {
                "reference_id": validator.reference.entity_id,
                "reference_frame": validator.cfg.reference_frame,
                "position_offset": list(validator.cfg.position_offset),
                "position_tolerance": list(validator.cfg.position_tolerance),
                "position_error": [_json_speed_values(row) for row in error],
            }
        elif validator.cfg.kind == "object_upright":
            measured = _upright(pose, float(validator.cfg.maximum_tilt_degrees))
            accepted &= measured.satisfied
            evidence = {
                "maximum_tilt_degrees": validator.cfg.maximum_tilt_degrees,
                "tilt_degrees": _json_speed_values(measured.metrics["tilt_degrees"]),
            }
        else:
            raise ValueError(
                f"Unsupported object-state validator {validator.cfg.kind!r}."
            )
        self._validator_results[id(validator)] = {
            "kind": validator.cfg.kind,
            "object_id": entity_id,
            "source_path": list(validator.source_path),
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "accepted_mask": accepted.detach().cpu().tolist(),
            **evidence,
        }
        return accepted.to(device=self._env_ids.device)

    def _validate_articulation_joint_position(
        self,
        validator: CompiledArticulationJointPositionValidator,
    ) -> torch.Tensor:
        """Evaluate one measured articulation joint against inclusive bounds."""
        entity_id = validator.articulation.entity_id
        articulation = self._articulations[entity_id]
        joint_index = self._articulation_joint_index(
            articulation,
            entity_id=entity_id,
            joint_name=validator.cfg.joint,
        )
        qpos = self._read_articulation_qpos(articulation, entity_id=entity_id)
        if joint_index >= qpos.shape[1]:
            raise ValueError(
                f"Joint {validator.cfg.joint!r} resolves to index {joint_index}, "
                f"outside articulation {entity_id!r} qpos width {qpos.shape[1]}."
            )
        position = qpos[:, joint_index]
        accepted = _joint_range(
            {"position": position},
            {
                "lower": validator.cfg.minimum_position,
                "upper": validator.cfg.maximum_position,
            },
        ).satisfied
        self._validator_results[id(validator)] = {
            "kind": validator.cfg.kind,
            "articulation_id": entity_id,
            "joint": validator.cfg.joint,
            "source_path": list(validator.source_path),
            "minimum_position": validator.cfg.minimum_position,
            "maximum_position": validator.cfg.maximum_position,
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "joint_position": [
                float(value) if math.isfinite(float(value)) else None
                for value in position.detach().cpu().tolist()
            ],
            "accepted_mask": accepted.detach().cpu().tolist(),
        }
        return accepted

    @staticmethod
    def _read_robot_qpos(robot: Robot, *, target: bool) -> torch.Tensor:
        """Capture one finite current- or target-qpos full-robot batch."""
        if type(target) is not bool:
            raise TypeError("target must be a bool.")
        mode = "target" if target else "current"
        call = f"robot.get_qpos(target={target})"
        get_qpos = getattr(robot, "get_qpos", None)
        if not callable(get_qpos):
            raise TypeError(f"robot must provide {call}.")
        qpos = get_qpos(target=target)
        if (
            not isinstance(qpos, torch.Tensor)
            or not qpos.is_floating_point()
            or qpos.dim() != 2
            or qpos.shape[0] == 0
            or qpos.shape[1] == 0
        ):
            raise ValueError(
                f"{call} must return {mode} floating full-qpos shape (B, J)."
            )
        if not bool(torch.isfinite(qpos).all().item()):
            raise ValueError(f"{call} must return finite {mode} qpos values.")
        return qpos.clone()

    def _hold_robot_qpos(self, active_mask: torch.Tensor) -> torch.Tensor:
        """Keep initial active rows on targets and inactive rows on current qpos."""
        target_qpos = self._read_robot_qpos(self._robot, target=True)
        if (
            target_qpos.shape != self._robot_qpos_shape
            or target_qpos.device != self._robot_qpos_device
        ):
            raise ValueError(
                "robot.get_qpos(target=True) target full qpos must match the "
                "construction-time current full qpos shape and device."
            )
        if bool(active_mask.all().item()):
            return target_qpos.index_select(0, self._row_indices)

        current_qpos = self._read_robot_qpos(self._robot, target=False)
        if (
            current_qpos.shape != self._robot_qpos_shape
            or current_qpos.device != self._robot_qpos_device
        ):
            raise ValueError(
                "robot.get_qpos(target=False) current full qpos must match the "
                "construction-time current full qpos shape and device."
            )
        hold_qpos = current_qpos.index_select(0, self._row_indices)
        hold_qpos[active_mask] = target_qpos.index_select(0, self._row_indices)[
            active_mask
        ]
        return hold_qpos

    def _validate_active_mask(self, active_mask: torch.Tensor) -> torch.Tensor:
        """Return one owned row mask aligned with the simulator batch."""
        if not isinstance(active_mask, torch.Tensor):
            raise TypeError("active_mask must be a torch.Tensor.")
        if active_mask.dtype != torch.bool or active_mask.shape != self._env_ids.shape:
            raise ValueError(
                "active_mask must be bool with one value per simulator row."
            )
        if active_mask.device != self._env_ids.device:
            raise ValueError("active_mask and env_ids must share a device.")
        return active_mask.clone()

    @staticmethod
    def _settle_threshold_metadata(
        preset: DynamicSettleMonitorCfg,
    ) -> dict[str, float | int]:
        """Serialize one settling preset without exposing mutable state."""
        return {
            "linear_velocity": float(preset.linear_velocity_threshold),
            "angular_velocity": float(preset.angular_velocity_threshold),
            "min_steps": preset.min_steps,
            "max_steps": preset.max_steps,
            "check_interval_steps": preset.check_interval_steps,
            "required_stable_checks": preset.required_stable_checks,
        }

    def _empty_settle_state_metadata(
        self,
        active_mask: torch.Tensor,
    ) -> dict[str, object]:
        """Return a full-batch trace for a policy with no eligible rows."""
        batch_size = self._env_ids.numel()
        return {
            "elapsed_steps": 0,
            "observation_count": 0,
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "active_mask": active_mask.detach().cpu().tolist(),
            "stable_counts": [0] * batch_size,
            "settled_mask": [False] * batch_size,
            "timeout_mask": [False] * batch_size,
            "checked": False,
            "max_linear_speed": [None] * batch_size,
            "max_angular_speed": [None] * batch_size,
        }

    def _expand_settle_state_metadata(
        self,
        state: DynamicSettleState,
        active_mask: torch.Tensor,
    ) -> dict[str, object]:
        """Expand active-row monitor state to the stable full-batch ordering."""
        stable_counts = torch.zeros_like(self._env_ids)
        settled_mask = torch.zeros_like(active_mask)
        timeout_mask = torch.zeros_like(active_mask)
        max_linear_speed = torch.full(
            active_mask.shape,
            float("inf"),
            dtype=state.max_linear_speed.dtype,
            device=active_mask.device,
        )
        max_angular_speed = torch.full_like(max_linear_speed, float("inf"))
        stable_counts[active_mask] = state.stable_counts
        settled_mask[active_mask] = state.settled_mask
        timeout_mask[active_mask] = state.timeout_mask
        max_linear_speed[active_mask] = state.max_linear_speed
        max_angular_speed[active_mask] = state.max_angular_speed
        return {
            "elapsed_steps": state.elapsed_steps,
            "observation_count": state.observation_count,
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "active_mask": active_mask.detach().cpu().tolist(),
            "stable_counts": stable_counts.detach().cpu().tolist(),
            "settled_mask": settled_mask.detach().cpu().tolist(),
            "timeout_mask": timeout_mask.detach().cpu().tolist(),
            "checked": state.checked,
            "max_linear_speed": _json_speed_values(max_linear_speed),
            "max_angular_speed": _json_speed_values(max_angular_speed),
        }

    @staticmethod
    def _validate_segment_membership(
        segment: Any,
        member: CompiledPostPolicy | CompiledTaskProgramValidator,
        *,
        kind: str,
    ) -> None:
        """Require the supplied compiled value to belong to the exact segment."""
        if type(segment) is not CompiledTaskProgramSegment:
            raise TypeError("segment must be exactly CompiledTaskProgramSegment.")
        values = (
            segment.post_policies
            if type(member) in (CompiledPostPolicy, CompiledHoldPostPolicy)
            else segment.validators
        )
        if not any(value is member for value in values):
            raise ValueError(
                f"Compiled {kind} does not belong to the supplied segment."
            )

    def _resolve_native_entities(
        self,
    ) -> tuple[
        Mapping[str, _SimulationSettleTarget],
        Mapping[str, Any],
        Mapping[str, Any],
    ]:
        """Resolve only explicitly declared canonical/native pairs."""
        settle_targets: dict[str, _SimulationSettleTarget] = {}
        objects: dict[str, Any] = {}
        articulation_targets: dict[str, _SimulationSettleTarget] = {}

        for binding in self._scene_binding.rigid_objects:
            entity = self._require_native(
                "get_rigid_object",
                canonical_id=binding.entity_id,
                simulation_uid=binding.simulation_uid,
            )
            target = _SimulationSettleTarget(
                binding.entity_id,
                "rigid_object",
                entity,
            )
            settle_targets[binding.entity_id] = target
            objects[binding.entity_id] = entity

        for binding in self._scene_binding.rigidized_articulations:
            entity = self._require_native(
                "get_articulation",
                canonical_id=binding.entity_id,
                simulation_uid=binding.simulation_uid,
            )
            settle_targets[binding.entity_id] = _SimulationSettleTarget(
                binding.entity_id,
                "articulation",
                entity,
            )
            objects[binding.entity_id] = entity

        for binding in self._scene_binding.articulations:
            entity = self._require_native(
                "get_articulation",
                canonical_id=binding.entity_id,
                simulation_uid=binding.simulation_uid,
            )
            target = _SimulationSettleTarget(
                binding.entity_id,
                "articulation",
                entity,
            )
            settle_targets[binding.entity_id] = target
            articulation_targets[binding.entity_id] = target

        for binding in self._scene_binding.links:
            settle_targets[binding.entity_id] = self._require_parent_target(
                articulation_targets,
                child_id=binding.entity_id,
                parent_id=binding.articulation_id,
            )
        for binding in self._scene_binding.antipodal_grasps:
            parent = settle_targets.get(binding.object_id)
            if parent is None or parent.kind != "rigid_object":
                raise KeyError(
                    f"Affordance {binding.entity_id!r} references unavailable rigid "
                    f"object {binding.object_id!r}."
                )
            settle_targets[binding.entity_id] = parent
        for binding in self._scene_binding.rigidized_articulation_grasps:
            parent = settle_targets.get(binding.object_id)
            if parent is None or parent.kind != "articulation":
                raise KeyError(
                    f"Affordance {binding.entity_id!r} references unavailable "
                    f"rigidized articulation {binding.object_id!r}."
                )
            settle_targets[binding.entity_id] = parent
        articulations = {
            entity_id: target.native_entity
            for entity_id, target in articulation_targets.items()
        }
        return (
            MappingProxyType(settle_targets),
            MappingProxyType(objects),
            MappingProxyType(articulations),
        )

    @staticmethod
    def _require_parent_target(
        targets: Mapping[str, _SimulationSettleTarget],
        *,
        child_id: str,
        parent_id: str,
    ) -> _SimulationSettleTarget:
        """Resolve a child to one explicitly declared articulation root."""
        target = targets.get(parent_id)
        if target is None:
            raise KeyError(
                f"Canonical entity {child_id!r} references unavailable parent "
                f"{parent_id!r}."
            )
        return target

    def _require_native(
        self,
        getter_name: str,
        *,
        canonical_id: str,
        simulation_uid: str,
    ) -> Any:
        """Resolve one explicitly selected native simulation entity."""
        getter = getattr(self._simulation, getter_name, None)
        if not callable(getter):
            raise TypeError(f"simulation must provide {getter_name}().")
        entity = getter(simulation_uid)
        if entity is None:
            raise KeyError(
                f"Native entity {simulation_uid!r} selected for canonical entity "
                f"{canonical_id!r} was not found."
            )
        return entity

    def _measure_settle_target(
        self,
        target: _SimulationSettleTarget,
        *,
        row_indices: torch.Tensor,
    ) -> DynamicSettleSample:
        """Measure physical bodies for explicitly selected simulator rows."""
        if target.kind == "articulation":
            body_data = getattr(target.native_entity, "body_data", None)
            velocity = getattr(body_data, "body_link_vel", None)
            if not isinstance(velocity, torch.Tensor):
                raise RuntimeError(
                    f"Articulation settle target {target.canonical_id!r} has no "
                    "body_link_vel tensor."
                )
            selected = velocity.index_select(0, row_indices.to(velocity.device))
            if selected.dim() != 3 or selected.shape[-1] != 6:
                raise ValueError(
                    "Articulation body_link_vel must have shape (B, N, 6)."
                )
            linear_velocity = selected[..., :3]
            angular_velocity = selected[..., 3:]
        else:
            body_data = getattr(target.native_entity, "body_data", None)
            linear_velocity = getattr(body_data, "lin_vel", None)
            angular_velocity = getattr(body_data, "ang_vel", None)
            if not isinstance(linear_velocity, torch.Tensor) or not isinstance(
                angular_velocity,
                torch.Tensor,
            ):
                raise RuntimeError(
                    f"Rigid settle target {target.canonical_id!r} has no linear/"
                    "angular velocity tensors."
                )
            rows = row_indices.to(linear_velocity.device)
            linear_velocity = linear_velocity.index_select(0, rows)
            angular_velocity = angular_velocity.index_select(
                0,
                row_indices.to(angular_velocity.device),
            )
            if (
                linear_velocity.shape != angular_velocity.shape
                or linear_velocity.dim() < 2
                or linear_velocity.shape[-1] != 3
            ):
                raise ValueError(
                    "Rigid body velocities must have equal shape (B, ..., 3)."
                )

        linear_speed = torch.linalg.vector_norm(linear_velocity, dim=-1).reshape(
            row_indices.numel(),
            -1,
        )
        angular_speed = torch.linalg.vector_norm(angular_velocity, dim=-1).reshape(
            row_indices.numel(),
            -1,
        )
        device = self._env_ids.device
        return DynamicSettleSample(
            entity_id=target.canonical_id,
            linear_speed=linear_speed.to(device=device),
            angular_speed=angular_speed.to(device=device),
        )

    def _measure_pose_delta_settle_target(
        self,
        target: _SimulationSettleTarget,
        *,
        row_indices: torch.Tensor,
        previous_pose: torch.Tensor | None,
    ) -> tuple[DynamicSettleSample, torch.Tensor]:
        """Derive rigid-object speed from consecutive observed poses.

        The first observation is unresolved because no temporal evidence exists.
        Selection comes from the explicit preset declaration, never its name.
        """
        if target.kind != "rigid_object":
            raise ValueError("Pose-delta settling requires a rigid-object target.")
        pose = self._read_pose(
            target.native_entity,
            entity_id=target.canonical_id,
        )
        pose = pose.index_select(0, row_indices.to(pose.device))
        row_count = row_indices.numel()
        if previous_pose is None:
            speed = torch.full(
                (row_count, 1),
                float("inf"),
                dtype=pose.dtype,
                device=pose.device,
            )
            return (
                DynamicSettleSample(
                    entity_id=target.canonical_id,
                    linear_speed=speed.to(device=self._env_ids.device),
                    angular_speed=speed.to(device=self._env_ids.device),
                ),
                pose,
            )
        if previous_pose.shape != pose.shape or previous_pose.device != pose.device:
            raise ValueError(
                "Consecutive rigid-object poses must have equal shape and device."
            )

        assert self._step_dt is not None
        displacement, rotation = rigid_pose_displacement(pose, previous_pose)
        linear_speed = displacement / self._step_dt
        angular_speed = rotation / self._step_dt
        device = self._env_ids.device
        return (
            DynamicSettleSample(
                entity_id=target.canonical_id,
                linear_speed=linear_speed.unsqueeze(-1).to(device=device),
                angular_speed=angular_speed.unsqueeze(-1).to(device=device),
            ),
            pose,
        )

    @staticmethod
    def _articulation_joint_index(
        articulation: Any,
        *,
        entity_id: str,
        joint_name: str,
    ) -> int:
        """Resolve one explicitly named native articulation joint."""
        names = getattr(articulation, "joint_names", None)
        if names is None or isinstance(names, (str, bytes)):
            raise TypeError(
                f"Articulation {entity_id!r} must expose an iterable joint_names."
            )
        try:
            normalized = tuple(names)
        except TypeError as exc:
            raise TypeError(
                f"Articulation {entity_id!r} must expose an iterable joint_names."
            ) from exc
        if not all(type(name) is str and name for name in normalized):
            raise ValueError(
                f"Articulation {entity_id!r} joint_names must be non-empty strings."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Articulation {entity_id!r} joint_names must be unique.")
        try:
            return normalized.index(joint_name)
        except ValueError as exc:
            raise KeyError(
                f"Joint {joint_name!r} was not found on articulation "
                f"{entity_id!r}; available joints are {sorted(normalized)}."
            ) from exc

    def _read_articulation_qpos(
        self,
        articulation: Any,
        *,
        entity_id: str,
    ) -> torch.Tensor:
        """Read one articulation position batch in simulator row order."""
        getter = getattr(articulation, "get_qpos", None)
        if not callable(getter):
            raise TypeError(
                f"Native articulation for {entity_id!r} must provide get_qpos()."
            )
        qpos = getter()
        if not isinstance(qpos, torch.Tensor) or not qpos.is_floating_point():
            raise TypeError("Articulation get_qpos() must return a floating tensor.")
        batch_size = self._robot_qpos_shape[0]
        if qpos.dim() != 2 or qpos.shape[0] != batch_size or qpos.shape[1] == 0:
            raise ValueError(
                f"Articulation {entity_id!r} qpos must have shape "
                f"({batch_size}, J) with J > 0."
            )
        return qpos.index_select(0, self._row_indices)

    def _read_pose(
        self, entity: Any, *, entity_id: str, require_finite: bool = True
    ) -> torch.Tensor:
        """Read one rigid-object pose batch in simulator row order."""
        getter = getattr(entity, "get_local_pose", None)
        if not callable(getter):
            raise TypeError(
                f"Native rigid object for {entity_id!r} must provide "
                "get_local_pose()."
            )
        pose = getter(to_matrix=True)
        if not isinstance(pose, torch.Tensor) or not pose.is_floating_point():
            raise TypeError("get_local_pose(to_matrix=True) must return a tensor.")
        batch_size = int(self._env_ids.numel())
        if pose.shape == (4, 4):
            pose = pose.unsqueeze(0).expand(batch_size, -1, -1)
        elif pose.shape == (self._robot_qpos_shape[0], 4, 4):
            pose = pose.index_select(0, self._row_indices)
        else:
            raise ValueError(
                f"Rigid object {entity_id!r} pose must have shape "
                f"({batch_size}, 4, 4)."
            )
        if require_finite and not bool(torch.isfinite(pose).all().item()):
            raise ValueError(f"Rigid object {entity_id!r} pose must be finite.")
        return pose.clone()

    def measure_support_geometry(
        self,
        preset_id: str,
        *,
        native_support: Mapping[str, Any],
        object_pose: Any,
        reference_pose: Any,
    ) -> Mapping[str, Any]:
        """Measure a captured support snapshot using the existing region geometry.

        Args:
            preset_id: Registered full-footprint support condition.
            native_support: Owned current or recorded native support measurements.
            object_pose: Matching observed payload poses, including exact historical slots.
            reference_pose: Corresponding observed support poses.

        Returns:
            Owned footprint and contact-height measurements. This method samples
            no contacts/live poses and computes no task acceptance decision.
        """
        self._require_open()
        if self._condition_runtime is None:
            raise ValueError("No support region condition is registered.")
        return self._condition_runtime.measure_support_geometry(
            preset_id,
            native_support=native_support,
            object_pose=object_pose,
            reference_pose=reference_pose,
        )


__all__ = ["SimulationSettleMonitorCfg"]
