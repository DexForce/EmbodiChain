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
from numbers import Real
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from embodichain.lab.sim.scene_expansion import SceneVariant
from embodichain.lab.task_program.integrations.scene_expansion import (
    ScenePreparationResult,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = ["SimulationSceneExpansionHost", "SimulationSceneInitialState"]

_RESTORE_ABSOLUTE_TOLERANCE = 1e-5
# Native physics may put measured joints a few microradians beyond their
# limits; set_state clips positions while restoring them. Targets and all
# other physical fields retain the stricter tolerance above.
_RESTORE_QPOS_ABSOLUTE_TOLERANCE = 1e-4


def _row(value: torch.Tensor, *, width: int, name: str) -> tuple[float, ...]:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.shape != (1, width)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must have finite floating shape (1,{width})")
    return tuple(value.detach().cpu()[0].tolist())


@dataclass(frozen=True)
class _EntityInitialState:
    kind: str
    uid: str
    body_type: str
    joint_names: tuple[str, ...]
    pose: tuple[float, ...]
    velocity: tuple[float, ...]
    qpos: tuple[float, ...] = ()
    qvel: tuple[float, ...] = ()
    target_qpos: tuple[float, ...] = ()
    target_qvel: tuple[float, ...] = ()
    qf: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("rigid_object", "robot", "articulation"):
            raise ValueError("Unknown snapshot physical entity kind")
        if type(self.uid) is not str or not self.uid or self.uid != self.uid.strip():
            raise ValueError("Snapshot entity UID must be a nonempty stripped string")
        names = tuple(self.joint_names)
        if any(
            type(name) is not str or not name or name != name.strip() for name in names
        ) or len(set(names)) != len(names):
            raise ValueError("Snapshot joint_names must contain unique nonempty names")
        if self.kind == "rigid_object":
            if names or self.body_type not in ("static", "dynamic", "kinematic"):
                raise ValueError("Rigid snapshot body type/joint names are invalid")
        elif self.body_type:
            raise ValueError("Articulation snapshots cannot declare rigid body_type")
        object.__setattr__(self, "joint_names", names)
        for name, width in (
            ("pose", 7),
            ("velocity", 6),
            *(
                (name, len(names))
                for name in ("qpos", "qvel", "target_qpos", "target_qvel", "qf")
            ),
        ):
            values = tuple(getattr(self, name))
            if len(values) != width or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                for value in values
            ):
                raise ValueError(
                    f"Snapshot {name} must contain {width} finite real values"
                )
            object.__setattr__(self, name, tuple(map(float, values)))
        if not math.isclose(
            sum(value * value for value in self.pose[3:]), 1.0, abs_tol=1e-5
        ):
            raise ValueError("Snapshot pose quaternion must be normalized")
        if (
            self.kind == "rigid_object"
            and self.body_type != "dynamic"
            and any(self.velocity)
        ):
            raise ValueError("Moving non-dynamic rigid bodies cannot be restored")


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
    ) -> None:
        self._env = getattr(environment, "unwrapped", environment)
        if self._env.num_envs != 1:
            raise ValueError("scene expansion currently requires B=1")
        if not callable(initial_validator):
            raise TypeError("initial_validator must be callable")
        if type(settle_steps) is not int or settle_steps < 0:
            raise ValueError("settle_steps must be a nonnegative integer")
        if not math.isfinite(self._env.step_dt) or self._env.step_dt <= 0:
            raise ValueError("step_dt must be finite and positive")
        self._validator = initial_validator
        self._settle_steps = settle_steps
        self._owner = object()
        self._initial_state: SimulationSceneInitialState | None = None
        self._topology = self._describe_entities()

    @property
    def initial_state(self) -> SimulationSceneInitialState | None:
        """Return the last actual initial state, absent before preparation.

        Returns:
            Captured immutable state, including states rejected by the validator.
        """
        return self._initial_state

    def _entities(self) -> tuple[tuple[str, str, object], ...]:
        sim = self._env.sim
        for method in (
            "get_rigid_object_group_uid_list",
            "get_deformable_object_uid_list",
            "get_rigid_constraint_uid_list",
        ):
            getter = getattr(sim, method, None)
            if callable(getter) and getter():
                raise ValueError(
                    f"scene snapshots do not support entities from {method}"
                )
        entities = []
        for kind in ("rigid_object", "robot", "articulation"):
            for uid in getattr(sim, f"get_{kind}_uid_list")():
                entity = getattr(sim, f"get_{kind}")(uid)
                if entity is None:
                    raise ValueError(f"Scene entity {uid!r} disappeared")
                entities.append((kind, uid, entity))
        if len({uid for _, uid, _ in entities}) != len(entities):
            raise ValueError("Physical entity UIDs must be unique across scene kinds")
        return tuple(sorted(entities, key=lambda value: (value[0], value[1])))

    def _describe_entities(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                kind,
                uid,
                getattr(entity, "body_type", ""),
                () if kind == "rigid_object" else tuple(entity.joint_names),
                id(entity),
            )
            for kind, uid, entity in self._entities()
        )

    def _check_topology(self) -> None:
        if self._describe_entities() != self._topology:
            raise ValueError(
                "Scene entity set or bindings changed; reconstruct the scene host"
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
        states = []
        for kind, uid, entity in self._entities():
            if kind == "rigid_object":
                full = _row(entity.body_state, width=13, name=f"{uid}.body_state")
                if entity.body_type != "dynamic" and any(full[7:]):
                    raise ValueError(
                        "Moving non-dynamic rigid bodies cannot be restored"
                    )
                states.append(
                    _EntityInitialState(
                        kind, uid, entity.body_type, (), full[:7], full[7:]
                    )
                )
                continue
            joint_names = tuple(entity.joint_names)
            dof = len(joint_names)
            # Clone in the reader's call: body_data reuses its state buffers.
            live = {
                key: value.clone()
                for key, value in entity.body_data.fetch_state().items()
            }
            states.append(
                _EntityInitialState(
                    kind,
                    uid,
                    "",
                    joint_names,
                    _row(live["root_pose"], width=7, name=f"{uid}.root_pose"),
                    _row(
                        torch.cat((live["root_lin_vel"], live["root_ang_vel"]), dim=-1),
                        width=6,
                        name=f"{uid}.root_velocity",
                    ),
                    qpos=_row(live["qpos"], width=dof, name=f"{uid}.qpos"),
                    qvel=_row(live["qvel"], width=dof, name=f"{uid}.qvel"),
                    target_qpos=_row(
                        entity.get_qpos(target=True),
                        width=dof,
                        name=f"{uid}.target_qpos",
                    ),
                    target_qvel=_row(
                        entity.get_qvel(target=True),
                        width=dof,
                        name=f"{uid}.target_qvel",
                    ),
                    qf=_row(entity.get_qf(), width=dof, name=f"{uid}.qf"),
                )
            )
        snapshot = SimulationSceneInitialState(
            parent_scene_id,
            self._env.sim.physics_backend,
            float(self._env.step_dt),
            tuple(states),
            self._owner,
        )
        # Validate all numeric payloads before a snapshot can reach restoration.
        snapshot.initial_state_id
        return snapshot

    def _after_state_write(self) -> None:
        # Settling is preparation, not a recorded policy/expert control action.
        for sensor in self._env.sensors.values():
            sensor.reset(env_ids=[0])
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
        metadata = {
            "initial_state": snapshot.to_metadata(),
            "settle_steps": self._settle_steps if restored_from is None else 0,
        }
        if restored_from is not None:
            max_error = 0.0
            max_qpos_error = 0.0
            max_other_error = 0.0
            failed = []
            for expected, actual in zip(
                restored_from._states, snapshot._states, strict=True
            ):
                for name in (
                    "pose",
                    "velocity",
                    "qpos",
                    "qvel",
                    "target_qpos",
                    "target_qvel",
                    "qf",
                ):
                    before = getattr(expected, name)
                    after = getattr(actual, name)
                    # q and -q represent the same physical root orientation.
                    if (
                        name == "pose"
                        and sum(a * b for a, b in zip(before[3:], after[3:])) < 0
                    ):
                        after = (*after[:3], *(-value for value in after[3:]))
                    error = max(
                        (abs(a - b) for a, b in zip(before, after, strict=True)),
                        default=0.0,
                    )
                    max_error = max(max_error, error)
                    if name == "qpos":
                        max_qpos_error = max(max_qpos_error, error)
                        tolerance = _RESTORE_QPOS_ABSOLUTE_TOLERANCE
                    else:
                        max_other_error = max(max_other_error, error)
                        tolerance = _RESTORE_ABSOLUTE_TOLERANCE
                    if error > tolerance:
                        failed.append(f"{expected.uid}.{name}")
            check = ValidationCheck(
                "scene.restore_state",
                "failed" if failed else "passed",
                (
                    "State differs after restoration: " + ", ".join(failed)
                    if failed
                    else "Measured qpos restored within 1e-4; all other saved physical fields within 1e-5; quaternion sign equivalence allowed"
                ),
                {
                    "max_abs_state_error": max_error,
                    "absolute_tolerance": _RESTORE_ABSOLUTE_TOLERANCE,
                    "qpos_absolute_tolerance": _RESTORE_QPOS_ABSOLUTE_TOLERANCE,
                    "max_abs_qpos_error": max_qpos_error,
                    "max_abs_other_state_error": max_other_error,
                },
            )
            validation = ValidationResult((check, *validation.checks))
            metadata["restored_from_initial_state_id"] = restored_from.initial_state_id
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
        # Preflight the entire state, not only the moved object, before resetting.
        self.capture_initial_state(parent_scene_id=variant.parent_scene_id)
        entities = {uid: (kind, entity) for kind, uid, entity in self._entities()}
        for change in variant.pose_changes:
            selected = entities.get(change.entity_uid)
            if selected is None or selected[0] != "rigid_object":
                raise ValueError(
                    f"Pose changes require an existing rigid object: {change.entity_uid!r}"
                )

        def prepare() -> ScenePreparationResult:
            self._check_topology()
            for change in variant.pose_changes:
                entity = entities[change.entity_uid][1]
                entity.clear_dynamics(env_ids=[0])
                pose = torch.tensor(
                    change.arena_pose, dtype=torch.float32, device=self._env.device
                ).unsqueeze(0)
                entity.set_local_pose(pose, env_ids=[0])
            if self._settle_steps:
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
        expected = tuple(description[:4] for description in self._topology)
        recorded = tuple(
            (state.kind, state.uid, state.body_type, state.joint_names)
            for state in snapshot._states
        )
        if recorded != expected:
            raise ValueError(
                "Snapshot entity set, body types or joint names differ from the scene"
            )
        entities = {uid: entity for _, uid, entity in self._entities()}

        def tensor(values: tuple[float, ...]) -> torch.Tensor:
            return torch.tensor(
                values, dtype=torch.float32, device=self._env.device
            ).unsqueeze(0)

        def prepare() -> ScenePreparationResult:
            self._check_topology()
            for state in snapshot._states:
                entity = entities[state.uid]
                if state.kind == "rigid_object":
                    entity.clear_dynamics(env_ids=[0])
                    entity.set_local_pose(tensor(state.pose), env_ids=[0])
                    if state.body_type == "dynamic":
                        entity.set_velocity(
                            lin_vel=tensor(state.velocity[:3]),
                            ang_vel=tensor(state.velocity[3:]),
                            env_ids=[0],
                        )
                else:
                    entity.set_state(
                        env_ids=[0],
                        root_pose=tensor(state.pose),
                        root_velocity=tensor(state.velocity),
                        qpos=tensor(state.qpos),
                        qvel=tensor(state.qvel),
                        target_qpos=tensor(state.target_qpos),
                        target_qvel=tensor(state.target_qvel),
                        qf=tensor(state.qf),
                        clear_dynamics=True,
                    )
            self._after_state_write()
            actual = self.capture_initial_state(
                parent_scene_id=snapshot.parent_scene_id
            )
            return self._result(actual, restored_from=snapshot)

        return self._run_reset(prepare)
