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

"""Gym scene preparation and owned physical initial-state restoration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import inspect
from numbers import Real
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.sensors import ContactSensor

from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene.expansion import SceneVariant
from embodichain.lab.task_program.integrations.scene_expansion import (
    ScenePreparationResult,
)

from embodichain.lab.sim.scene.query import SceneView
from embodichain.lab.sim.scene.state import (
    EntityState as _EntityInitialState,
    SceneState,
    capture_state,
    restore_state,
    _preflight_restore,
)
from embodichain.lab.sim.scene.operations import (
    SceneChangeSet,
    apply_changes,
    _preflight_changes,
)
from embodichain.lab.sim.scene.validation import (
    SceneCheckReport,
    SettlingCfg,
    compare_states,
    settle_scene,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = ["SimulationSceneExpansionHost", "SimulationSceneInitialState"]


def _validation_result(report: SceneCheckReport) -> ValidationResult:
    return ValidationResult(
        tuple(
            ValidationCheck(
                check.name,
                "unavailable" if check.status == "unverified" else check.status,
                check.message,
                check.metrics,
            )
            for check in report.checks
        )
    )


@dataclass(frozen=True)
class SimulationSceneInitialState:
    """Owned B=1 physical state for a fixed-topology scene host.

    Args:
        parent_scene_id: Asset/configuration identity supplied by the candidate.
        physics_backend: Backend used for capture and required for restoration.
        step_dt: Authoritative Gym control period.

    Snapshots include every supported rigid body, robot and articulation, with
    velocities, full joint state and controller targets. Storage is immutable
    numeric tuples, independent of simulator read buffers. Restoration clears
    solver history and external forces; this is an episode initial-state
    snapshot, not a bitwise mid-trajectory checkpoint. Physical parameters and
    asset contents are fixed for the lifetime of the owning host.
    """

    parent_scene_id: str
    physics_backend: str
    step_dt: float
    _states: tuple[_EntityInitialState, ...] = field(repr=False)
    _owner: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("parent_scene_id", "physics_backend"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"Snapshot {name} must be a nonempty stripped string")
        if (
            isinstance(self.step_dt, bool)
            or not isinstance(self.step_dt, Real)
            or not math.isfinite(self.step_dt)
            or self.step_dt <= 0
        ):
            raise ValueError("Snapshot step_dt must be finite and positive")
        states = tuple(self._states)
        if any(not isinstance(state, _EntityInitialState) for state in states) or len(
            {state.uid for state in states}
        ) != len(states):
            raise ValueError("Snapshot requires unique fully validated entity states")
        object.__setattr__(
            self,
            "_states",
            tuple(sorted(states, key=lambda state: (state.kind, state.uid))),
        )

    @property
    def initial_state_id(self) -> str:
        """Return a content identity independent of physical slot assignments.

        Returns:
            SHA-256 identity of scene provenance and complete captured state.
        """
        payload = self.to_metadata()
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return "scene-initial:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_metadata(self) -> dict[str, object]:
        """Export the actual initial state for the existing episode metadata.

        Returns:
            JSON-compatible poses, velocities, joint/controller state and scene
            provenance. Native handles and temporary environment IDs are absent.
        """
        return {
            "schema": "simulation-scene-initial/v1",
            "parent_scene_id": self.parent_scene_id,
            "physics_backend": self.physics_backend,
            "step_dt": self.step_dt,
            "entities": [asdict(state) for state in self._states],
        }


class SimulationSceneExpansionHost:
    """Prepare existing-rigid-object variants through the Gym reset lifecycle.

    Args:
        environment: Initialized B=1 Gym environment, optionally wrapped.
        initial_validator: Required checks of the actual scene after settling.
            The callback may include task semantics, support and workspace IK.
        settle_steps: Nonnegative physics substeps before initial-state checks.
        settling: Optional measured convergence policy; overrides settle_steps.
            Failed or unverified convergence rejects the candidate.

    Pass this host as ``prepare_scene`` to ``execute_scene_variant``. The reset
    hook applies the candidate after ordinary reset events and before physical
    objectives, final observations and recorder seeding. Each subsequent Task
    Program bridge builds fresh providers and a fresh planner collision world.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        *,
        initial_validator: Callable[[], ValidationResult],
        settle_steps: int = 0,
        settling: SettlingCfg | None = None,
    ) -> None:
        self._env = getattr(environment, "unwrapped", environment)
        if self._env.num_envs != 1:
            raise ValueError("scene expansion currently requires B=1")
        if not callable(initial_validator):
            raise TypeError("initial_validator must be callable")
        try:
            inspect.signature(initial_validator).bind()
        except (TypeError, ValueError) as error:
            raise TypeError("initial_validator must accept zero arguments") from error
        if type(settle_steps) is not int or settle_steps < 0:
            raise ValueError("settle_steps must be a nonnegative integer")
        if not math.isfinite(self._env.step_dt) or self._env.step_dt <= 0:
            raise ValueError("step_dt must be finite and positive")
        if settling is not None and (
            not isinstance(settling, SettlingCfg) or settling.mode != "convergence"
        ):
            raise ValueError("settling must explicitly select convergence mode")
        if settling is not None:
            settling.__post_init__()
        self._settling = settling
        self._settling_report: SceneCheckReport | None = None
        self._view = SceneView(self._env.sim)
        self._validator = initial_validator
        self._settle_steps = settle_steps
        self._owner = object()
        self._initial_state: SimulationSceneInitialState | None = None
        self._topology = self._describe_entities()
        self._physical_properties = self._measure_physical_properties()

    @property
    def initial_state(self) -> SimulationSceneInitialState | None:
        """Return the last actual initial state, absent before preparation.

        Returns:
            Captured immutable state, including states rejected by the validator.
        """
        return self._initial_state

    def _entities(self) -> tuple[tuple[str, str, object], ...]:
        return self._view.entities()

    def _describe_entities(self) -> tuple[tuple[object, ...], ...]:
        return self._view.describe_entities()

    def _check_topology(self) -> None:
        self._view.check_topology()

    def _measure_physical_properties(self) -> dict[str, torch.Tensor]:
        """Own the live physical parameters that must stay fixed across resets."""
        properties = {}
        for kind, uid, entity in self._entities():
            # Static bodies deliberately have no RigidBodyData. Their material
            # parameters remain measurable, but dynamic mass properties do not.
            values = (
                {}
                if kind == "rigid_object" and entity.body_type == "static"
                else {
                    name: getattr(entity.body_data, name)
                    for name in ("mass", "inertia", "com_pose")
                }
            )
            if kind == "rigid_object":
                values.update(
                    friction=entity.get_friction(env_ids=[0]),
                    damping=entity.get_damping(env_ids=[0]),
                )
            else:
                values.update(
                    {
                        name: getattr(entity.body_data, name)
                        for name in (
                            "joint_stiffness",
                            "joint_damping",
                            "joint_friction",
                            "joint_armature",
                            "qpos_limits",
                            "qvel_limits",
                        )
                    }
                )
            for name, value in values.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or not value.is_floating_point()
                    or value.ndim == 0
                    or value.shape[0] != 1
                    or bool(torch.isnan(value).any())
                    or (
                        not name.endswith("limits")
                        and not bool(torch.isfinite(value).all())
                    )
                ):
                    raise ValueError(f"Cannot measure physical parameter {uid}.{name}")
                properties[f"{uid}.{name}"] = value.detach().clone()
        return properties

    def _check_physical_properties(self) -> None:
        actual = self._measure_physical_properties()
        if any(
            actual[name].shape != expected.shape
            or not torch.allclose(actual[name], expected, atol=1e-6, rtol=1e-6)
            for name, expected in self._physical_properties.items()
        ):
            raise ValueError(
                "Scene physical parameters changed; reconstruct the scene host"
            )

    def _physical_state(self, snapshot: SimulationSceneInitialState) -> SceneState:
        return SceneState(
            snapshot.physics_backend,
            (0,),
            (snapshot._states,),
            self._env.sim,
            self._topology,
        )

    def capture_initial_state(
        self, *, parent_scene_id: str
    ) -> SimulationSceneInitialState:
        """Capture every supported physical entity without advancing physics.

        Args:
            parent_scene_id: Stable source scene/asset/configuration identity.

        Returns:
            Owned full initial state suitable for restoration on this host.

        Raises:
            ValueError: If topology changed or any state cannot be captured fully.
        """
        if (
            type(parent_scene_id) is not str
            or not parent_scene_id
            or parent_scene_id != parent_scene_id.strip()
        ):
            raise ValueError("parent_scene_id must be a nonempty stripped string")
        self._check_topology()
        self._check_physical_properties()
        physical = capture_state(self._view)
        snapshot = SimulationSceneInitialState(
            parent_scene_id,
            self._env.sim.physics_backend,
            float(self._env.step_dt),
            physical.batches[0],
            self._owner,
        )
        # Validate all numeric payloads before a snapshot can reach restoration.
        snapshot.initial_state_id
        return snapshot

    def _after_state_write(self) -> None:
        # Settling is preparation, not a recorded policy/expert control action.
        for sensor in self._env.sensors.values():
            sensor.reset(env_ids=[0])
            if isinstance(sensor, ContactSensor):
                # Refresh current evidence without advancing physics or history.
                sensor.update()
        observation_manager = getattr(self._env, "observation_manager", None)
        if observation_manager is not None:
            observation_manager.reset(env_ids=[0])

    def _run_reset(
        self, callback: Callable[[], ScenePreparationResult]
    ) -> ScenePreparationResult:
        results = []

        def prepare() -> None:
            if results:
                raise RuntimeError("Scene preparation reset hook must run exactly once")
            results.append(callback())

        self._env.reset(
            options={"save_data": False, "scene_expansion_prepare": prepare}
        )
        if len(results) != 1:
            raise RuntimeError(
                "Environment reset did not invoke scene_expansion_prepare"
            )
        return results[0]

    def _result(
        self,
        snapshot: SimulationSceneInitialState,
        *,
        restored_from: SimulationSceneInitialState | None = None,
    ) -> ScenePreparationResult:
        validation = self._validator()
        if not isinstance(validation, ValidationResult):
            raise TypeError("initial_validator must return a ValidationResult")
        if not validation.checks:
            validation = ValidationResult(
                (
                    ValidationCheck(
                        "scene.initial_checks",
                        "unavailable",
                        "No initial checks were provided",
                    ),
                )
            )
        metadata = {
            "initial_state": snapshot.to_metadata(),
            "settle_steps": self._settle_steps if restored_from is None else 0,
        }
        if restored_from is not None:
            restore_checks = _validation_result(
                compare_states(
                    self._physical_state(restored_from), self._physical_state(snapshot)
                )
            )
            validation = ValidationResult((*restore_checks.checks, *validation.checks))
            metadata["restored_from_initial_state_id"] = restored_from.initial_state_id
        if restored_from is None and self._settling_report is not None:
            validation = ValidationResult(
                (*_validation_result(self._settling_report).checks, *validation.checks)
            )
            metadata["settle_steps"] = self._settling_report.checks[0].metrics["steps"]
        self._initial_state = snapshot
        return ScenePreparationResult(
            snapshot.initial_state_id,
            validation,
            metadata,
        )

    def __call__(self, variant: SceneVariant) -> ScenePreparationResult:
        """Apply, settle and check one candidate before recording starts.

        Args:
            variant: Existing-rigid-object pose changes in the local arena frame.

        Returns:
            Actual state identity, required checks and the captured state metadata.

        Raises:
            ValueError: For unsupported variants/topology before any reset writes.
        """
        if not isinstance(variant, SceneVariant):
            raise TypeError("variant must be a SceneVariant")
        self._check_topology()
        if self._settling is not None:
            if self._settling.mode != "convergence":
                raise ValueError("settling must explicitly select convergence mode")
            self._settling.__post_init__()
        # Preflight the entire state, not only the moved object, before resetting.
        self.capture_initial_state(parent_scene_id=variant.parent_scene_id)
        changes = SceneChangeSet(
            {change.entity_uid: change.arena_pose for change in variant.pose_changes},
            tuple(change.entity_uid for change in variant.pose_changes),
        )
        _preflight_changes(self._view, changes)

        def prepare() -> ScenePreparationResult:
            self._check_topology()
            self._check_physical_properties()
            apply_changes(self._view, changes)
            self._settling_report = None
            if self._settling is not None:
                self._settling_report = settle_scene(
                    self._view,
                    lambda count: self._env.sim.update(step=count),
                    self._settling,
                )
            elif self._settle_steps:
                self._env.sim.update(step=self._settle_steps)
            self._after_state_write()
            return self._result(
                self.capture_initial_state(parent_scene_id=variant.parent_scene_id)
            )

        return self._run_reset(prepare)

    def restore_initial_state(
        self, snapshot: SimulationSceneInitialState
    ) -> ScenePreparationResult:
        """Restore a captured state after normal reset events and recheck it.

        Args:
            snapshot: Immutable state captured by this host with identical assets,
                topology, physics backend and control period.

        Returns:
            Fresh checks and the actual restored state identity, linked to the
            source snapshot. Restoration does not settle again. A mandatory
            state check requires measured joint positions within absolute
            tolerance ``1e-4`` to account for native joint-limit clipping. Every
            other field, including controller targets, uses ``1e-5``. Quaternion
            signs are equivalent. Backend round trips may produce a different
            content hash even when this passes.

        Raises:
            ValueError: For foreign snapshots or changed topology before writes.
        """
        if not isinstance(snapshot, SimulationSceneInitialState):
            raise TypeError("snapshot must be a SimulationSceneInitialState")
        self._check_topology()
        if snapshot._owner is not self._owner:
            raise ValueError("Snapshot belongs to another scene host")
        if (
            snapshot.physics_backend != self._env.sim.physics_backend
            or snapshot.step_dt != self._env.step_dt
        ):
            raise ValueError(
                "Snapshot backend/control period differs from the environment"
            )
        physical = self._physical_state(snapshot)
        self._check_physical_properties()
        _preflight_restore(self._view, physical, qpos_tolerance=1e-4)

        def prepare() -> ScenePreparationResult:
            self._check_physical_properties()
            restore_state(self._view, physical, qpos_tolerance=1e-4)
            self._after_state_write()
            actual = self.capture_initial_state(
                parent_scene_id=snapshot.parent_scene_id
            )
            return self._result(actual, restored_from=snapshot)

        return self._run_reset(prepare)
