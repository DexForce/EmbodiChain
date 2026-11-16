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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import inspect
from numbers import Real
from types import MappingProxyType
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
    apply_changes_batch,
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
    """Owned physical state for one row of a fixed-topology scene host.

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
    _env_id: int = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self._env_id) is not int or self._env_id < 0:
            raise ValueError("Snapshot environment ID must be nonnegative")
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
        environment: Initialized Gym environment, optionally wrapped.
        env_ids: Ordered rows to prepare; defaults to the complete World.
        initial_validator: Required checks of the actual scene after settling.
            Called with each physical environment ID. It may include task
            semantics, support and workspace IK.
        settle_steps: Nonnegative physics substeps before initial-state checks.
        settling: Optional measured convergence policy; overrides settle_steps.
            Failed or unverified convergence rejects the candidate.

    Pass this host as ``prepare_scene`` to ``execute_scene_variants``. The reset
    hook applies the candidate after ordinary reset events and before physical
    objectives, final observations and recorder seeding. Each subsequent Task
    Program bridge builds fresh providers and a fresh planner collision world.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        *,
        initial_validator: Callable[[int], ValidationResult],
        env_ids: Sequence[int] | None = None,
        settle_steps: int = 0,
        settling: SettlingCfg | None = None,
    ) -> None:
        self._env = getattr(environment, "unwrapped", environment)
        self._view = SceneView(
            self._env.sim,
            env_ids=tuple(range(self._env.num_envs)) if env_ids is None else env_ids,
        )
        if not callable(initial_validator):
            raise TypeError("initial_validator must be callable")
        try:
            inspect.signature(initial_validator).bind(self._view.env_ids[0])
        except (TypeError, ValueError) as error:
            raise TypeError(
                "initial_validator must accept an environment ID"
            ) from error
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
        if (settle_steps or settling is not None) and set(self._view.env_ids) != set(
            range(self._env.num_envs)
        ):
            raise ValueError("Settling requires all environment rows in the World")
        self._settling = settling
        self._settling_report: SceneCheckReport | None = None
        self._validator = initial_validator
        self._settle_steps = settle_steps
        self._owner = object()
        self._initial_states: Mapping[int, SimulationSceneInitialState] | None = None
        self._topology = self._describe_entities()
        self._physical_properties = self._measure_physical_properties()

    @property
    def initial_states(self) -> Mapping[int, SimulationSceneInitialState] | None:
        """Return the last actual initial states, absent before preparation.

        Returns:
            Immutable row snapshots, including rows rejected by their validator.
        """
        return self._initial_states

    def _entities(self) -> tuple[tuple[str, str, object], ...]:
        return self._view.entities()

    def _describe_entities(self) -> tuple[tuple[object, ...], ...]:
        return self._view.describe_entities()

    def _check_topology(self) -> None:
        self._view.check_topology()

    def _measure_physical_properties(self) -> dict[str, torch.Tensor]:
        """Own the live physical parameter baseline for this scene selection."""
        from embodichain.lab.sim.scene.physics import capture_physics_properties

        return capture_physics_properties(self._view)

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

    def _physical_state(
        self, snapshots: Mapping[int, SimulationSceneInitialState]
    ) -> SceneState:
        return SceneState(
            self._env.sim.physics_backend,
            self._view.env_ids,
            tuple(snapshots[row]._states for row in self._view.env_ids),
            self._env.sim,
            self._topology,
        )

    def capture_initial_states(
        self, *, parent_scene_ids: Mapping[int, str]
    ) -> Mapping[int, SimulationSceneInitialState]:
        """Capture every supported physical entity without advancing physics.

        Args:
            parent_scene_ids: Stable source identities for exactly the selected rows.

        Returns:
            Owned row snapshots suitable for restoration on this host.

        Raises:
            ValueError: If topology changed or any state cannot be captured fully.
        """
        self._check_rows(parent_scene_ids)
        if any(
            type(value) is not str or not value or value != value.strip()
            for value in parent_scene_ids.values()
        ):
            raise ValueError("parent_scene_ids must contain nonempty stripped strings")
        self._check_topology()
        self._check_physical_properties()
        physical = capture_state(self._view)
        snapshots = {}
        for row, states in zip(physical.env_ids, physical.batches, strict=True):
            snapshot = SimulationSceneInitialState(
                parent_scene_ids[row],
                self._env.sim.physics_backend,
                float(self._env.step_dt),
                states,
                self._owner,
                row,
            )
            snapshot.initial_state_id
            snapshots[row] = snapshot
        return MappingProxyType(snapshots)

    def _check_rows(self, values: Mapping[int, object]) -> None:
        if (
            not isinstance(values, Mapping)
            or any(type(row) is not int for row in values)
            or set(values) != set(self._view.env_ids)
        ):
            raise ValueError("Inputs must exactly match the selected environment rows")

    def _after_state_write(self) -> None:
        # Settling is preparation, not a recorded policy/expert control action.
        for sensor in self._env.sensors.values():
            sensor.reset(env_ids=list(self._view.env_ids))
            if isinstance(sensor, ContactSensor):
                # Refresh current evidence without advancing physics or history.
                sensor.update()
        observation_manager = getattr(self._env, "observation_manager", None)
        if observation_manager is not None:
            observation_manager.reset(env_ids=list(self._view.env_ids))

    def _run_reset(
        self, callback: Callable[[], Mapping[int, ScenePreparationResult]]
    ) -> Mapping[int, ScenePreparationResult]:
        results = []

        def prepare() -> None:
            if results:
                raise RuntimeError("Scene preparation reset hook must run exactly once")
            results.append(callback())

        self._env.reset(
            options={
                "reset_ids": list(self._view.env_ids),
                "save_data": False,
                "episode_prepare": prepare,
            }
        )
        if len(results) != 1:
            raise RuntimeError("Environment reset did not invoke episode_prepare")
        return results[0]

    def _result(
        self,
        snapshots: Mapping[int, SimulationSceneInitialState],
        *,
        restored_from: Mapping[int, SimulationSceneInitialState] | None = None,
    ) -> Mapping[int, ScenePreparationResult]:
        results = {}
        for row in self._view.env_ids:
            snapshot = snapshots[row]
            validation = self._validator(row)
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
                expected = SceneState(
                    snapshot.physics_backend,
                    (row,),
                    (restored_from[row]._states,),
                    self._env.sim,
                    self._topology,
                )
                actual = SceneState(
                    snapshot.physics_backend,
                    (row,),
                    (snapshot._states,),
                    self._env.sim,
                    self._topology,
                )
                checks = _validation_result(compare_states(expected, actual))
                validation = ValidationResult((*checks.checks, *validation.checks))
                metadata["restored_from_initial_state_id"] = restored_from[
                    row
                ].initial_state_id
            elif self._settling_report is not None:
                checks = _validation_result(self._settling_report)
                validation = ValidationResult((*checks.checks, *validation.checks))
                metadata["settle_steps"] = self._settling_report.checks[0].metrics[
                    "steps"
                ]
            results[row] = ScenePreparationResult(
                snapshot.initial_state_id, validation, metadata
            )
        self._initial_states = MappingProxyType(dict(snapshots))
        return MappingProxyType(results)

    def __call__(
        self, variants: Mapping[int, SceneVariant]
    ) -> Mapping[int, ScenePreparationResult]:
        """Prepare selected candidates together, then check each actual row.

        Args:
            variants: Exactly one existing-entity proposal per selected row.

        Returns:
            Actual initial-state identities and independent validation results.

        Raises:
            ValueError: For invalid selections or topology before reset writes.
        """
        self._check_rows(variants)
        variants = MappingProxyType(dict(variants))
        if any(not isinstance(variant, SceneVariant) for variant in variants.values()):
            raise TypeError("variants must contain SceneVariant objects")
        self._check_topology()
        if self._settling is not None:
            if self._settling.mode != "convergence":
                raise ValueError("settling must explicitly select convergence mode")
            self._settling.__post_init__()
        parent_ids = {row: variant.parent_scene_id for row, variant in variants.items()}
        self.capture_initial_states(parent_scene_ids=parent_ids)
        changes = {
            row: SceneChangeSet(
                {
                    change.entity_uid: change.arena_pose
                    for change in variant.pose_changes
                },
                tuple(change.entity_uid for change in variant.pose_changes),
            )
            for row, variant in variants.items()
        }
        for row, change in changes.items():
            _preflight_changes(SceneView(self._env.sim, env_ids=(row,)), change)

        def prepare() -> Mapping[int, ScenePreparationResult]:
            self._check_topology()
            self._check_physical_properties()
            apply_changes_batch(self._view, changes)
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
                self.capture_initial_states(parent_scene_ids=parent_ids)
            )

        return self._run_reset(prepare)

    def restore_initial_states(
        self, snapshots: Mapping[int, SimulationSceneInitialState]
    ) -> Mapping[int, ScenePreparationResult]:
        """Restore all selected rows after reset events and validate each row.

        Args:
            snapshots: Owned row snapshots with unchanged assets, topology,
                physical backend and control period. Rows cannot be reassigned.

        Returns:
            Independent initial checks and measured round-trip checks. Measured
            joint positions allow absolute tolerance ``1e-4``; other fields use
            ``1e-5``, with quaternion-sign equivalence. No settling is repeated.

        Raises:
            ValueError: For foreign snapshots or changed topology before writes.
        """
        self._check_rows(snapshots)
        snapshots = MappingProxyType(dict(snapshots))
        self._check_topology()
        for row, snapshot in snapshots.items():
            if not isinstance(snapshot, SimulationSceneInitialState):
                raise TypeError(
                    "snapshots must contain SimulationSceneInitialState objects"
                )
            if snapshot._owner is not self._owner or snapshot._env_id != row:
                raise ValueError(
                    "Snapshot belongs to another scene host or environment row"
                )
            if (
                snapshot.physics_backend != self._env.sim.physics_backend
                or snapshot.step_dt != self._env.step_dt
            ):
                raise ValueError(
                    "Snapshot backend/control period differs from the environment"
                )
        physical = self._physical_state(snapshots)
        self._check_physical_properties()
        _preflight_restore(self._view, physical, qpos_tolerance=1e-4)

        def prepare() -> Mapping[int, ScenePreparationResult]:
            self._check_physical_properties()
            restore_state(self._view, physical, qpos_tolerance=1e-4)
            self._after_state_write()
            actual = self.capture_initial_states(
                parent_scene_ids={
                    row: snapshot.parent_scene_id for row, snapshot in snapshots.items()
                }
            )
            return self._result(actual, restored_from=snapshots)

        return self._run_reset(prepare)
