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

"""Declared control post-policy assembly using shared physical validators."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING, asdict, replace
import hashlib
import json
import math
from typing import Any, Literal, TYPE_CHECKING

import torch

from embodichain.utils import configclass
from embodichain.lab.sim.atomic_actions.articulation_evidence import (
    PressEventTarget,
    TwistEventTarget,
)
from embodichain.lab.sim.atomic_actions.control_validation import (
    TwistStartupSample,
    evaluate_twist_startup,
    validate_press_ready,
    validate_press_release,
    validate_twist_ready,
    validate_twist_terminal,
)
from embodichain.lab.sim.scene.control_mass_observation import (
    verify_scaled_control_mass,
)
from embodichain.lab.sim.scene.press_inputs import PressBinding
from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding
from embodichain.lab.sim.sensors.contact.sampling import (
    ContactEvidenceSampler,
    CONTACT_ACTOR_IDS,
    CONTACT_IMPULSES,
    CONTACT_VALID_MASK,
)
from embodichain.lab.sim.sensors.contact.observation import JointContactObserver
from .control_observation import (
    ControlContactActors,
    ControlObservationWindow,
    ControlObservationSnapshot,
    ControlPolicyObservation,
)
from .twist_session import TwistOperationSession

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot, Articulation
    from embodichain.lab.sim.scene.control_measurement import NativeControlGeometry
    from embodichain.lab.sim.sensors.support_observation import SupportControlClock

__all__ = ["SimulationControlConditionCfg", "ControlOperationRuntime"]


@configclass
class SimulationControlConditionCfg:
    """Explicit named control-policy selection; physical rules stay in Atomic."""

    entity: str = MISSING
    operation_id: str = MISSING
    family: Literal["press", "twist"] = MISSING
    phase: Literal["ready", "chunk", "released"] = MISSING
    terminal: bool = False

    def snapshot(self) -> SimulationControlConditionCfg:
        """Own and validate the complete declaration without native access."""
        self.validate()
        value = deepcopy(self)
        for name in ("entity", "operation_id"):
            item = getattr(value, name)
            if type(item) is not str or not item or item != item.strip():
                raise ValueError("Control policy identities must be canonical.")
        if value.family not in ("press", "twist") or value.phase not in (
            "ready",
            "chunk",
            "released",
        ):
            raise ValueError("Unknown control policy family or phase.")
        if (
            type(value.terminal) is not bool
            or value.terminal
            and value.phase != "released"
        ):
            raise ValueError("Only released control policies may be terminal.")
        if value.family == "press" and value.phase == "chunk":
            raise ValueError("Press does not expose Twist chunk policies.")
        return value


class ControlOperationRuntime:
    """Own one declared operation's sensor windows and hold orchestration.

    Robot/geometry/sensor inputs are borrowed. The host registers the sensor in
    its ordinary substep suite, supplies a fresh operation per bridge/episode,
    and binds the canonical execution clock. No simulator is stepped here.
    Actor IDs, model link rosters and route provenance are explicit declarations.
    """

    def __init__(
        self,
        robot: Robot,
        articulation: Articulation,
        binding: PressBinding | RevoluteControlBinding,
        geometry: NativeControlGeometry,
        sampler: ContactEvidenceSampler,
        actors: ControlContactActors,
        *,
        entity_id: str,
        operation_id: str,
        program_fingerprint: str,
        episode_id: str,
        observer_id: str,
        route_payload: Mapping[str, Any],
        control_dt: float,
        physics_dt: float,
        simulation_time: Callable[[], float],
        validate_scope: Callable[[], None],
        twist_session: TwistOperationSession | None = None,
        table_actor: int | None = None,
        robot_actors: Sequence[int] = (),
        twist_criterion: Literal[
            "contact_backed_coarse_turn", "calibrated_setting"
        ] = "contact_backed_coarse_turn",
    ) -> None:
        if (
            robot.num_instances != 1
            or articulation.num_instances != 1
            or sampler.num_envs != 1
        ):
            raise ValueError("Control post-policies support native B=1 only.")
        self.family = (
            "press"
            if type(binding) is PressBinding
            else "twist" if type(binding) is RevoluteControlBinding else None
        )
        if self.family is None or (self.family == "twist") != (
            twist_session is not None
        ):
            raise ValueError("Control bindings and operation session kinds differ.")
        if getattr(articulation, "uid", None) != binding.object_id:
            raise ValueError(
                "Control source binding differs from the native articulation UID."
            )
        if geometry.robot is not robot or geometry.articulation is not articulation:
            raise ValueError("Control geometry belongs to another physical owner.")
        for value in (
            entity_id,
            operation_id,
            program_fingerprint,
            episode_id,
            observer_id,
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Control operation identities must be exact.")
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
            for value in (control_dt, physics_dt)
        ):
            raise ValueError("Control and physics cadences must be positive.")
        payload = json.loads(json.dumps(route_payload, allow_nan=False))
        physical = asdict(binding)
        physical.pop("source_path", None)
        if payload.get("operation_id") != operation_id or payload.get(
            "binding"
        ) != json.loads(json.dumps(physical)):
            raise ValueError(
                "Control provenance differs from the exact source binding."
            )
        self._criterion = (
            twist_criterion if self.family == "twist" else "contact_backed_press"
        )
        if self.family == "twist" and twist_criterion not in (
            "contact_backed_coarse_turn",
            "calibrated_setting",
        ):
            raise ValueError("Unknown explicit Twist event criterion.")
        if payload.get("success_criterion", self._criterion) != self._criterion:
            raise ValueError(
                "Legacy route projection differs from the declared event criterion."
            )
        if self.family == "twist" and (
            payload.get("feedback_enabled", False)
            or payload.get("grip_extra_m", 0) != 0
        ):
            raise ValueError("Twist feedback correction remains unsupported.")
        self._payload = payload
        self._digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        self.robot, self.articulation, self.binding, self.geometry = (
            robot,
            articulation,
            binding,
            geometry,
        )
        self.entity_id, self.operation_id = entity_id, operation_id
        self._program, self._episode = program_fingerprint, episode_id
        self._sampler, self._actors, self._observer_id = sampler, actors, observer_id
        self._sensor_uid = getattr(sampler.sensor, "uid", None)
        self._topology_revision = sampler.source_stamp().topology_revision
        self._dt, self._physics_dt = float(control_dt), float(physics_dt)
        self._time, self._scope_guard = simulation_time, validate_scope
        self._joint = articulation.joint_names.index(binding.joint)
        self._twist = twist_session
        if twist_session is not None and (
            twist_session.scope.program_fingerprint != program_fingerprint
            or twist_session.scope.operation_id != operation_id
            or twist_session.scope.episode_id != episode_id
        ):
            raise ValueError(
                "Control operation differs from its declared Twist cursor scope."
            )
        self._table, self._robot_actors = table_actor, tuple(robot_actors)
        if self.family == "twist" and (
            type(table_actor) is not int
            or table_actor < 0
            or not self._robot_actors
            or any(type(value) is not int or value < 0 for value in self._robot_actors)
        ):
            raise ValueError(
                "Twist startup requires exact table and complete robot actor IDs."
            )
        self.mass_evidence = (
            verify_scaled_control_mass(articulation, binding)
            if self.family == "twist"
            else {"policy": "native", "verified": False, "scope": "not_requested"}
        )
        self._clock = None
        self._closed = False
        self._finished = False
        self._executing = False
        self._origin = float(simulation_time())
        self._observer = JointContactObserver(
            sampler,
            lambda: articulation.get_qpos()[:, self._joint],
            simulation_time,
            retain_source_stamp=True,
            joint_velocity=(
                (lambda: articulation.get_qvel()[:, self._joint])
                if self.family == "twist"
                else None
            ),
        )
        self._window = None
        self._startup_window = None
        self._startup_snapshot = None
        self._final_snapshot = None
        self._other_qpos = None
        self._reference_qpos_observation = None
        self._policy_observation = None
        self._fact_revision = 0
        self._hold_frames = []
        self._joint_limits_observation = None
        self._current_qpos_observation = None
        self._clearance_observation = None
        self._retreat_verified = False
        self._receipt: dict[str, Any] | None = None
        sampler.sensor.add_sampling_observer(observer_id, self._observer)
        try:
            if self.family == "twist":
                self.start_startup(refresh=True)
        except Exception:
            sampler.sensor.remove_sampling_observer(
                observer_id, expected=self._observer
            )
            self._observer.close()
            raise

    def _open(self) -> None:
        if self._closed:
            raise RuntimeError("Control operation is closed.")
        self._scope_guard()

    def validate_scope(self) -> None:
        """Reject closed or replaced bridge/episode ownership without native reads."""
        self._open()

    @property
    def observation_scope(self) -> dict[str, object]:
        """Return owned declared provenance without consulting live native state.

        Returns:
            dict[str, builtins.object]: JSON metadata for the declared program, episode, operation,
            clock, native actors and observer identity. These declarations do
            not certify that a physical observation is current or available.
        """
        return {
            "program_fingerprint": self._program,
            "episode_id": self._episode,
            "operation_id": self.operation_id,
            "entity_id": self.entity_id,
            "env_ids": [0],
            "clock": "simulation_time",
            "legacy_time_origin": self._origin,
            "physics_dt": self._physics_dt,
            "control_dt": self._dt,
            "joint_index": self._joint,
            "family": self.family,
            "sensor_uid": self._sensor_uid,
            "observer_id": self._observer_id,
            "topology_revision": self._topology_revision,
            "contact_actors": {
                "target": self._actors.target,
                "parent": self._actors.parent,
                "hand": list(self._actors.hand),
            },
            "startup_actors": {"table": self._table, "robot": list(self._robot_actors)},
        }

    @property
    def route_payload(self) -> dict[str, Any]:
        """Return the owned plain declaration used by the original route digest."""
        return deepcopy(self._payload)

    def validate_clock(self, clock: SupportControlClock) -> None:
        """Reject shared episode ownership before mutating any clock binding."""
        self._open()
        if self._clock is not None and self._clock is not clock:
            raise ValueError(
                "Control operation requires a fresh service for each bridge clock."
            )
        if not math.isclose(clock.step_dt, self._dt, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Control service and bridge cadences differ.")

    def bind_clock(self, clock: SupportControlClock) -> None:
        """Bind once to one bridge; sharing state across a new episode is rejected."""
        self.validate_clock(clock)
        self._clock = clock

    def _new_window(self, generation: int) -> ControlObservationWindow:
        return ControlObservationWindow(
            self._observer,
            generation,
            self._actors,
            legacy_time_origin=self._origin,
            physics_dt=self._physics_dt,
            current_time=self._time,
            validate_scope=self._open,
        )

    def start_startup(self, *, refresh: bool = False) -> None:
        """Start the declared reset-to-ready window at the prepare-call boundary."""
        self._open()
        if self.family != "twist":
            raise ValueError("Startup qualification belongs to Twist.")
        generation = self._observer.begin_window(
            "startup",
            max_samples=min(8192, math.ceil(32.0 / self._physics_dt) + 1),
            max_duration=32.0,
        )
        stamp = None if refresh else self._observer.latest_source_stamp()
        self._observer.capture(refresh=refresh, source_stamp=stamp)
        self._startup_window = self._new_window(generation)

    def _arm(self) -> None:
        generation = self._observer.begin_window("prepare")
        self._observer.capture(refresh=True)
        self._window = self._new_window(generation)
        self._observer.set_phase(self.family)
        self._retreat_verified = False

    def _startup(self) -> dict[str, object]:
        if self._startup_window is None:
            raise ValueError("Twist startup was not explicitly prepared.")
        snapshot = self._startup_window.snapshot()
        self._startup_snapshot = snapshot
        frames = []
        for record, sample in zip(snapshot.records, snapshot.samples):
            data = record.frame.batches
            velocity = data.get("articulation.joint_velocity")
            qvel = float("nan") if velocity is None else float(velocity.values[0])
            valid = (
                sample.valid
                and velocity is not None
                and bool(velocity.valid[0])
                and math.isfinite(qvel)
            )
            mask = data[CONTACT_VALID_MASK].values[0]
            pairs = data[CONTACT_ACTOR_IDS].values[0, mask].tolist()
            impulses = data[CONTACT_IMPULSES].values[0, mask].tolist()
            table = any(
                self._actors.target in pair and self._table in pair and impulse > 0
                for pair, impulse in zip(pairs, impulses)
            )
            robot = any(
                self._actors.target in pair
                and any(actor in self._robot_actors for actor in pair)
                and impulse > 0
                for pair, impulse in zip(pairs, impulses)
            )
            frames.append(
                TwistStartupSample(
                    sample.timestamp,
                    sample.qpos,
                    qvel,
                    table,
                    robot,
                    bool(valid),
                    record.dropped_contacts,
                )
            )
        result = evaluate_twist_startup(
            frames, complete=True, overflow=self._observer.overflow
        )
        if not snapshot.available:
            result.update(accepted=False, reason="invalid_startup_evidence")
        return result

    def _fact_frame(self, values: Mapping[str, torch.Tensor]) -> Any:
        """Own values already read by this policy; never query a native owner here."""
        from embodichain.lab.sim.evidence import (
            PhysicalEvidenceBatch,
            PhysicalEvidenceFrame,
        )

        self._fact_revision += 1
        timestamp = float(self._time())
        ids = torch.tensor([0], dtype=torch.long)
        batches = {}
        for name, value in values.items():
            owned = value.detach().cpu()
            valid = torch.isfinite(owned).reshape(1, -1).all(dim=1)
            batches[name] = PhysicalEvidenceBatch(
                name,
                owned,
                valid,
                (None if bool(valid[0]) else "Nonfinite control policy measurement.",),
                timestamp,
                ids,
                self._fact_revision,
            )
        return PhysicalEvidenceFrame(timestamp, self._fact_revision, ids, batches)

    def _holds(
        self, duration: float, *, target: bool, measure: bool
    ) -> Iterator[torch.Tensor]:
        if self._clock is None:
            raise ValueError(
                "Control operation requires an acknowledged execution clock."
            )
        hold = self.robot.get_qpos(target=target).clone()
        self._positions, self._velocities = [], []
        self._hold_frames = []
        start, time = self._clock.step_index, self._clock.now()
        self._hold_start_step = start
        self._hold_end_step = start
        for tick in range(math.ceil(duration / self._dt)):
            self._open()
            if self._clock.step_index != start + tick or not math.isclose(
                self._clock.now(), time + tick * self._dt, rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError(
                    "Control hold requires acknowledged environment steps."
                )
            yield hold.clone()
            self._open()
            if self._clock.step_index != start + tick + 1 or not math.isclose(
                self._clock.now(), time + (tick + 1) * self._dt, rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError(
                    "Control hold completion requires an acknowledged environment step."
                )
            self._hold_end_step = self._clock.step_index
            if measure:
                position = self.articulation.get_qpos()[0, self._joint]
                self._positions.append(float(position))
                values = {"articulation.joint_position": position.reshape(1)}
                if self.family == "twist":
                    velocity = self.articulation.get_qvel()[0, self._joint]
                    self._velocities.append(float(velocity))
                    values["articulation.joint_velocity"] = velocity.reshape(1)
                self._hold_frames.append(self._fact_frame(values))

    def validate(self, cfg: SimulationControlConditionCfg) -> None:
        """Check pure declaration ownership before any post-policy read or action."""
        if (
            cfg.entity != self.entity_id
            or cfg.operation_id != self.operation_id
            or cfg.family != self.family
        ):
            raise ValueError("Control post-policy differs from its declared operation.")

    def actions(
        self, cfg: SimulationControlConditionCfg, *, preset: str
    ) -> Iterator[torch.Tensor]:
        """Execute one owned policy iterator; overlapping operations are rejected."""
        self._open()
        if type(cfg) is not SimulationControlConditionCfg:
            raise TypeError("Control policies require exact typed declarations.")
        cfg = cfg.snapshot()
        self.validate(cfg)
        if self._finished:
            raise RuntimeError(
                "Control operation already reached its terminal boundary."
            )
        if self._executing:
            raise ValueError("Control operation already has an executing post-policy.")
        self._executing = True
        complete = False
        try:
            yield from self._actions_scoped(cfg, preset=preset)
            complete = True
        finally:
            self._executing = False
            if not complete:
                self.abort()

    def _actions_scoped(
        self, cfg: SimulationControlConditionCfg, *, preset: str
    ) -> Iterator[torch.Tensor]:
        """Yield source hold grids and delegate physical decisions to Atomic."""
        self._open()
        self.validate(cfg)
        if self._finished:
            raise RuntimeError(
                "Control operation already reached its terminal boundary."
            )
        self._receipt = None
        self._policy_observation = None
        self._joint_limits_observation = None
        self._current_qpos_observation = None
        self._clearance_observation = None
        ready = cfg.phase == "ready"
        target_hold = (
            (not ready)
            if self.family == "press"
            else bool(not ready and self._twist.snapshot().goal_complete)
        )
        yield from self._holds(
            0.2 if self.family == "press" and not ready else 0.5,
            target=target_hold,
            measure=self.family == "twist" or ready,
        )
        if ready:
            startup = self._startup() if self.family == "twist" else None
            if self._twist is not None:
                self._twist.reset_ready()
            self._sampler.sensor.update()
            self._arm()
            projection = self._window.projection()
            initial = projection.samples[0]
            if self.family == "press":
                target = PressEventTarget(
                    self.binding.released_position, self.binding.pressed_position
                )

                def read_limits() -> torch.Tensor:
                    limits = self.articulation.get_qpos_limits()[0, self._joint]
                    self._joint_limits_observation = self._fact_frame(
                        {"articulation.joint_limits": limits.reshape(1, 2)}
                    )
                    return limits

                result = validate_press_ready(
                    target,
                    self._positions,
                    initial,
                    read_limits=read_limits,
                    expected_limits=self.binding.limits,
                    window_available=projection.available,
                )
            else:
                self._other_qpos = self.articulation.get_qpos().clone()
                self._reference_qpos_observation = self._fact_frame(
                    {"articulation.joint_positions": self._other_qpos}
                )
                result = validate_twist_ready(
                    startup,
                    self._positions,
                    self._velocities,
                    initial,
                    window_available=projection.available,
                )
                if result["accepted"]:
                    self._twist.bind_control_observation(
                        self._window, self._event_target()
                    )
        else:
            if self._window is None:
                raise ValueError("Control operation has no measured ready window.")
            projection = self._window.projection()
            if self.family == "press":
                geometry = self.geometry.press_clearances(terminal=cfg.terminal)
                self._clearance_observation = self._fact_frame(
                    {
                        "control." + key: torch.tensor([value], dtype=torch.float64)
                        for key, value in geometry.items()
                        if value is not None
                    }
                )
                observed = self.articulation.get_qpos()[0, self._joint]
                self._current_qpos_observation = self._fact_frame(
                    {"articulation.joint_position": observed.reshape(1)}
                )
                result = validate_press_release(
                    PressEventTarget(
                        self.binding.released_position, self.binding.pressed_position
                    ),
                    projection.samples,
                    parent_contacts=projection.parent_contacts,
                    physics_dt=self._physics_dt,
                    terminal=cfg.terminal,
                    retreat_verified=self._retreat_verified,
                    observed_qpos=float(observed),
                    window_available=projection.available,
                    **geometry,
                )
                self._retreat_verified = result["retreat_verified"]
            elif cfg.phase == "chunk":
                observed = self.articulation.get_qpos()[0, self._joint]
                self._current_qpos_observation = self._fact_frame(
                    {"articulation.joint_position": observed.reshape(1)}
                )
                result = self._twist.observe_chunk(
                    observed_qpos=float(observed),
                    stable_positions=self._positions,
                    stable_velocities=self._velocities,
                    feedback_timestamp=self._clock.now(),
                )
            else:
                clearance = self.geometry.park_clearance()
                self._clearance_observation = self._fact_frame(
                    {
                        "control.park_clearance": torch.tensor(
                            [clearance], dtype=torch.float64
                        )
                    }
                )
                current = self.articulation.get_qpos().clone()
                self._current_qpos_observation = self._fact_frame(
                    {"articulation.joint_positions": current}
                )
                current[:, self._joint] = self._other_qpos[:, self._joint]
                unchanged = bool(
                    torch.isfinite(current).all()
                    and torch.allclose(
                        current, self._other_qpos, atol=math.radians(2.0), rtol=0
                    )
                )
                result = validate_twist_terminal(
                    self._event_target(),
                    projection.samples,
                    parent_contacts=projection.parent_contacts,
                    positions=self._positions,
                    velocities=self._velocities,
                    physics_dt=self._physics_dt,
                    clearance=clearance,
                    others_unchanged=unchanged,
                    terminal=cfg.terminal,
                    early_stop=self._twist.snapshot().goal_complete,
                    window_available=projection.available,
                )
            if cfg.phase == "released":
                self._observer.set_phase("cleanup")
        projection = self._window.projection()
        available = projection.available and result.get("observation_available", True)
        if (
            ready
            and self.family == "twist"
            and startup["reason"] == "invalid_startup_evidence"
        ):
            available = False
        result.update(
            observation_available=bool(available),
            observation_errors=list(
                dict.fromkeys(
                    (*result.get("observation_errors", ()), *projection.errors)
                )
            ),
        )
        self._receipt = self._stamp(result, preset, projection.samples)
        self._policy_observation = ControlPolicyObservation(
            phase=cfg.phase,
            terminal=cfg.terminal,
            control_dt=self._dt,
            joint_index=self._joint,
            generation=projection.generation,
            legacy_time_origin=projection.legacy_time_origin,
            physics_dt=projection.physics_dt,
            window=(
                (projection.samples[0].timestamp, projection.samples[-1].timestamp)
                if projection.samples
                else None
            ),
            sample_count=len(projection.samples),
            hold_start_step=self._hold_start_step,
            hold_end_step=self._hold_end_step,
            hold_frames=tuple(self._hold_frames),
            joint_limits=self._joint_limits_observation,
            reference_qpos=self._reference_qpos_observation,
            current_qpos=self._current_qpos_observation,
            clearance=self._clearance_observation,
        )
        if cfg.terminal or not result["accepted"]:
            self._final_snapshot = self._window.snapshot()
            self._observer.pause_window()
            self._finished = True
            if self._twist is not None:
                self._twist.close()

    def _event_target(self) -> TwistEventTarget:
        return TwistEventTarget(
            self.binding.target_qpos,
            self.binding.limits,
            success_criterion=self._criterion,
        )

    def _stamp(
        self, result: Mapping[str, Any], preset: str, samples: Sequence[Any]
    ) -> dict[str, Any]:
        return {
            **deepcopy(dict(result)),
            "kind": "control_operation",
            "preset": preset,
            "control": {
                "revision": 1,
                "operation_id": self.operation_id,
                "object_id": self.binding.object_id,
                "joint": self.binding.joint,
                "link": self.binding.link,
                "source_sha256": self.binding.source_sha256,
                "route_digest": self._digest,
                "criterion": self._criterion,
                "sample_count": len(samples),
                "window": (
                    [samples[0].timestamp, samples[-1].timestamp] if samples else None
                ),
            },
        }

    def receipt(self) -> dict[str, Any]:
        """Return the last completed owned receipt; never a partial or stale success."""
        if self._receipt is None:
            raise RuntimeError(
                "Control receipt is unavailable before policy completion."
            )
        return deepcopy(self._receipt)

    def recorded_measurements(self) -> ControlPolicyObservation:
        """Copy physical reads retained from the last actual completed policy."""
        if self._policy_observation is None:
            raise ValueError("No completed control policy measurements were retained.")
        return self._policy_observation.snapshot()

    def recorded_window(self, *, startup: bool = False) -> ControlObservationSnapshot:
        """Copy retained historical raw evidence without consulting current native state."""
        value = self._startup_snapshot if startup else self._final_snapshot
        if value is None:
            raise ValueError("No completed control evidence window was retained.")
        return replace(
            value,
            records=tuple(record.snapshot() for record in value.records),
            trace=deepcopy(value.trace),
        )

    def abort(self) -> None:
        """Invalidate an incomplete policy without native reads or guessed steps."""
        self._receipt = None
        self._finished = True
        self._observer.pause_window()
        if self._twist is not None:
            self._twist.close()

    def close(self) -> None:
        """Detach only the owned observer; never destroy borrowed native resources."""
        if not self._closed:
            self._sampler.sensor.remove_sampling_observer(
                self._observer_id, expected=self._observer
            )
            self._observer.close()
            if self._twist is not None:
                self._twist.close()
            self._closed = True
