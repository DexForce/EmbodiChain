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

"""Production simulation assembly for Gym-backed Task Programs.

This module owns the reusable live wiring between declarative simulation
bindings and :class:`TaskProgramEnvironmentAdapter`.  A task supplies one
immutable registration; this factory constructs the motion generator,
atomic-action engine, planning observation port, effect-evidence providers,
and segment-policy port without task-local motion code.

The resulting runtime is intentionally Gym-only.  Its buffered command sink
must remain attached to :class:`TaskProgramDemoBridge`, which advances the shared
clock only after an ordinary ``env.step()`` consumes a yielded command.  It is
therefore an internal Task Program executor, not a third user-facing action
entry point.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
import math
from uuid import uuid4
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from weakref import ReferenceType, ref
from types import MappingProxyType
from typing import Literal, Protocol, TYPE_CHECKING

import torch

from embodichain.lab.sim.evidence import PhysicalEvidenceRequest, PhysicalEvidenceBatch
from embodichain.lab.sim.atomic_actions.grip_evidence import GripEvidenceObservationPort

from embodichain.lab.gym.envs.expert_trajectory import JointCommandMode
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EntityState,
    ObservedArticulationJointState,
    PlanningContext,
    RobotObservation,
    SceneProvider,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.motion.planners import (
    BasePlannerCfg,
    CuroboAutoGenCfg,
    CuroboPlannerCfg,
    CuroboWorldCfg,
    NeuralPlannerCfg,
    ToppraPlannerCfg,
    TrapezoidalPlannerCfg,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
)
from embodichain.lab.task_program.semantics.evidence import (
    ControlPartRobotEvidenceSource,
    ControlPartSimulationEvidenceProvider,
    EffectEvidenceProvider,
    EffectEvidenceCollectionContext,
    EffectEvidenceQueryValue,
    EffectEvidenceBatch,
    SceneArticulationEvidenceProvider,
)
from embodichain.lab.task_program.runtime.parallel_executor import (
    ParallelCommandSafetyValidator,
)
from embodichain.lab.task_program.semantics.profiles import RobotSkillProfile
from embodichain.lab.task_program.semantics.scene import (
    SceneCollisionRole,
    SceneCollisionWorldMode,
    RegistrySceneProvider,
    SceneRegistry,
)

from embodichain.lab.gym.envs.task_program.bridge import (
    EnvironmentStepClock,
    TaskProgramDemoBridge,
    GymPlanningObservationProvider,
)
from ..catalog import SimulationTaskProgramRegistration
from ..environment import (
    TaskProgramEnvironmentAdapter,
    TaskProgramEnvironmentFactory,
    PlanningObservationPort,
)
from .pour_projection import ResolvedPourProjectionRoute
from embodichain.lab.task_program.compiler.lowering import SemanticCallCompiler
from .policies import SimulationSegmentPolicyPort
from .stability import SimulationStabilityConditionCfg
from ..runtime_services import TaskProgramRuntimeServiceContext
from ..invocation_policy import task_program_fingerprint, BoundPostPolicyPresets
from .control_assembly import ControlOperationAssembly, assemble_control_operation
from .twist_session import TwistOperationRegistry
from .articulation_session import (
    ArticulationContinuationRegistry,
    bind_articulation_invocations,
)
from embodichain.lab.task_program.runtime.lifecycle import (
    TaskProgramPlanLifecycleFactory,
    CompositeTaskProgramPlanLifecycleFactory,
)

from embodichain.lab.task_program.compiler.program import CompiledTaskProgram

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from .observation_services import SimulationObservationServices
    from ..invocation_policy import InvocationPolicyBinding
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
    from embodichain.toolkits.graspkit import GraspPoseGenerator


MotionGeneratorFactory = Callable[[], MotionGenerator]
"""Zero-argument factory that must return one fresh motion generator."""


@dataclass
class _RuntimeOwner:
    closed: bool = False

    def require_open(self) -> None:
        if self.closed:
            raise RuntimeError("Simulation Task Program runtime resources are closed.")


class _OwnedEffectEvidenceProvider(EffectEvidenceProvider):
    """Keep factory-provided live evidence inside the factory lifetime."""

    def __init__(self, provider: EffectEvidenceProvider, owner: _RuntimeOwner) -> None:
        self._provider = provider
        self._owner = owner
        self.provider_id = provider.provider_id
        self.revision = provider.revision

    def collect(
        self,
        queries: tuple[EffectEvidenceQueryValue, ...],
        context: EffectEvidenceCollectionContext,
    ) -> Mapping[str, EffectEvidenceBatch]:
        self._owner.require_open()
        return self._provider.collect(queries, context)


@dataclass(frozen=True, slots=True)
class _ControlBridgeServices:
    runtime_context: TaskProgramRuntimeServiceContext
    policy_port: SimulationSegmentPolicyPort
    twist_registry: TwistOperationRegistry | None
    operations: tuple[ControlOperationAssembly, ...]
    owner: _RuntimeOwner
    scope_guard: Callable[[], None]
    articulation_registry: ArticulationContinuationRegistry | None = None
    operation_lifetime: _RuntimeOwner = field(default_factory=_RuntimeOwner)

    @property
    def plan_lifecycle_factory(self) -> TaskProgramPlanLifecycleFactory | None:
        factories = tuple(
            value
            for value in (self.twist_registry, self.articulation_registry)
            if value is not None
        )
        return (
            None
            if not factories
            else (
                factories[0]
                if len(factories) == 1
                else CompositeTaskProgramPlanLifecycleFactory(factories)
            )
        )

    def validate_scope(self) -> None:
        self.scope_guard()
        self.runtime_context.validate_scope()

    def close(self) -> None:
        if self.operation_lifetime.closed:
            return
        self.operation_lifetime.closed = True
        errors = []
        resources = [operation.close for operation in self.operations]
        if self.twist_registry is not None:
            resources.append(self.twist_registry.close)
        if self.articulation_registry is not None:
            resources.append(self.articulation_registry.close)
        for close in resources:
            try:
                close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def close_host_resources(self) -> None:
        self.owner.closed = True
        try:
            self.close()
        finally:
            self.policy_port.close_runtime_resources()


class _OwnedGripEvidenceProvider(_OwnedEffectEvidenceProvider):
    """Forward only an explicitly declared physical observation capability."""

    def __init__(
        self,
        provider: EffectEvidenceProvider,
        owner: _RuntimeOwner,
        scope_guard: Callable[[], None],
    ) -> None:
        if not isinstance(provider, GripEvidenceObservationPort):
            raise TypeError("Grip observation forwarding requires its explicit port.")
        super().__init__(provider, owner)
        self._grip_port = provider
        self._scope_guard = scope_guard

    def capture_grip_observation(
        self, request: PhysicalEvidenceRequest, *, object_id: str, control_part: str
    ) -> tuple[PhysicalEvidenceBatch, dict[str, object]]:
        self._owner.require_open()
        self._scope_guard()
        try:
            return self._grip_port.capture_grip_observation(
                request, object_id=object_id, control_part=control_part
            )
        finally:
            self._owner.require_open()
            self._scope_guard()


_PLANNER_CFG_TYPES: Mapping[str, type[BasePlannerCfg]] = {
    "curobo": CuroboPlannerCfg,
    "neural": NeuralPlannerCfg,
    "toppra": ToppraPlannerCfg,
    "trapezoidal": TrapezoidalPlannerCfg,
}


def _planner_cfg_from_config(
    value: Mapping[str, object] | None,
    *,
    robot_uid: str,
) -> BasePlannerCfg | None:
    """Build one typed planner config from an execution-policy declaration.

    The robot UID and simulation-manager instance are runtime-owned values and
    are deliberately excluded from the serialized policy.  cuRobo's live
    collision objects are likewise attached by the simulation factory after
    the selected environment has been initialized.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("planner_config must be a mapping or None.")
    unknown = sorted(set(value).difference({"type", "config"}))
    if unknown:
        raise ValueError(f"planner_config contains unsupported fields: {unknown}.")
    planner_type = value.get("type")
    if type(planner_type) is not str or not planner_type:
        raise ValueError("planner_config.type must be a non-empty string.")
    cfg_type = _PLANNER_CFG_TYPES.get(planner_type)
    if cfg_type is None:
        raise ValueError(
            f"planner_config.type {planner_type!r} is unsupported; "
            f"supported types are {sorted(_PLANNER_CFG_TYPES)}."
        )
    options = value.get("config", {})
    if not isinstance(options, Mapping):
        raise TypeError("planner_config.config must be a mapping.")
    options = deepcopy(dict(options))
    reserved = sorted(
        {"planner_type", "robot_uid", "sim_instance_id"}.intersection(options)
    )
    if reserved:
        raise ValueError(
            "planner_config.config cannot override runtime-owned fields: "
            f"{reserved}."
        )

    if planner_type == "curobo":
        if "world" in options:
            world = options["world"]
            if not isinstance(world, Mapping):
                raise TypeError("planner_config.config.world must be a mapping.")
            world = deepcopy(dict(world))
            if "rigid_objects" in world:
                raise ValueError(
                    "planner_config.config.world.rigid_objects is runtime-owned; "
                    "declare scene collision roles instead."
                )
            if "dynamic_obstacle_names" in world:
                raise ValueError(
                    "planner_config.config.world.dynamic_obstacle_names is "
                    "runtime-owned; declare dynamic scene collision roles instead."
                )
            options["world"] = CuroboWorldCfg(**world)
        if "auto_gen" in options:
            auto_gen = options["auto_gen"]
            if not isinstance(auto_gen, Mapping):
                raise TypeError("planner_config.config.auto_gen must be a mapping.")
            options["auto_gen"] = CuroboAutoGenCfg(**deepcopy(dict(auto_gen)))

    try:
        planner_cfg = cfg_type(robot_uid=robot_uid, **options)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid planner_config for planner type {planner_type!r}: {exc}"
        ) from exc
    planner_cfg.validate()
    return planner_cfg


class SimulationTaskProgramEnvironment(Protocol):
    """Minimal Gym environment surface used by the simulation factory."""

    sim: SimulationManager
    robot: Robot

    @property
    def step_dt(self) -> float:
        """Return the authoritative Gym control cadence in seconds."""


def _positive_finite(value: float, *, field_name: str) -> float:
    """Validate one positive finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return normalized


def _non_negative_finite(value: float, *, field_name: str) -> float:
    """Validate one non-negative finite real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return normalized


def _robot_uid(robot: Robot) -> str:
    """Return one strict live robot UID."""
    uid = getattr(robot, "uid", None)
    if type(uid) is not str or not uid or uid != uid.strip():
        raise ValueError(
            "robot.uid must be a non-empty string without outer whitespace."
        )
    return uid


def _full_robot_tensor(
    robot: Robot,
    getter_name: str,
    *,
    required: bool,
    reference: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Read and validate one full-robot floating state tensor."""
    getter = getattr(robot, getter_name, None)
    if not callable(getter):
        if required:
            raise TypeError(f"robot must provide {getter_name}().")
        return None
    value = getter()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"robot.{getter_name}() must return a torch.Tensor.")
    if not value.is_floating_point() or value.dim() != 2:
        raise ValueError(
            f"robot.{getter_name}() must return floating shape (B, robot_dof)."
        )
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"robot.{getter_name}() dimensions must be non-zero.")
    if reference is not None and (
        value.shape != reference.shape or value.device != reference.device
    ):
        raise ValueError(
            f"robot.{getter_name}() must match robot.get_qpos() shape and device."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"robot.{getter_name}() must contain only finite values.")
    return value.clone()


class SharedTickSceneProvider(SceneProvider):
    """Share one immutable scene snapshot across consumers in the same tick.

    ``RegistrySceneProvider`` is stateful: every call observes native entities
    and updates material-change baselines.  Planning observations and multiple
    evidence providers can legitimately request the same timestamp.  This
    wrapper always delegates one full-batch request per tick, then returns the
    exact snapshot or an owned ordered-row projection to later consumers.

    Args:
        delegate: Authoritative native scene observer.
        full_env_ids: Complete ordered environment identity selection.
        require_open: Optional host-lifetime check, called before cached or live reads.
        runtime_scope_guard: Optional host epoch validator, checked around cached
            or live reads without acquiring additional physical data.
    """

    def __init__(
        self,
        delegate: RegistrySceneProvider,
        full_env_ids: torch.Tensor,
        *,
        require_open: Callable[[], None] | None = None,
        runtime_scope_guard: Callable[[], None] | None = None,
    ) -> None:
        if type(delegate) is not RegistrySceneProvider:
            raise TypeError("delegate must be exactly RegistrySceneProvider.")
        if (
            not isinstance(full_env_ids, torch.Tensor)
            or full_env_ids.dtype != torch.long
            or full_env_ids.dim() != 1
            or full_env_ids.numel() == 0
        ):
            raise ValueError("full_env_ids must be a non-empty 1D int64 tensor.")
        if torch.unique(full_env_ids).numel() != full_env_ids.numel():
            raise ValueError("full_env_ids must be unique.")
        if runtime_scope_guard is not None and not callable(runtime_scope_guard):
            raise TypeError("runtime_scope_guard must be callable or None.")
        self._require_open = require_open
        self._runtime_scope_guard = runtime_scope_guard
        self._delegate = delegate
        self._full_env_ids = full_env_ids.clone()
        self._row_by_env_id = {
            int(env_id): row
            for row, env_id in enumerate(full_env_ids.detach().cpu().tolist())
        }
        self._timestamp: float | None = None
        self._snapshot: SceneSnapshot | None = None

    @property
    def collision_entity_ids(self) -> tuple[str, ...]:
        """Return canonical dynamic collision IDs from the delegate."""
        return self._delegate.collision_entity_ids

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> SceneSnapshot:
        """Return the single shared snapshot for ``timestamp`` and ``env_ids``."""
        if self._require_open is not None:
            self._require_open()
        if self._runtime_scope_guard is not None:
            self._runtime_scope_guard()
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise TypeError("timestamp must be a real number.")
        normalized_timestamp = float(timestamp)
        if not math.isfinite(normalized_timestamp) or normalized_timestamp < 0.0:
            raise ValueError("timestamp must be finite and non-negative.")
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.dim() != 1
            or env_ids.numel() == 0
        ):
            raise ValueError("env_ids must be a non-empty 1D int64 tensor.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")
        if env_ids.device != self._full_env_ids.device:
            raise ValueError("env_ids must share the full simulation batch device.")
        try:
            rows = tuple(
                self._row_by_env_id[int(env_id)]
                for env_id in env_ids.detach().cpu().tolist()
            )
        except KeyError as exc:
            raise ValueError(
                f"Environment ID {int(exc.args[0])} is absent from full_env_ids."
            ) from exc

        if self._timestamp is not None:
            if normalized_timestamp < self._timestamp:
                raise ValueError("Shared scene snapshot timestamps must be monotonic.")
            if normalized_timestamp == self._timestamp:
                assert self._snapshot is not None
                result = self._select_rows(self._snapshot, rows)
                if self._runtime_scope_guard is not None:
                    if self._require_open is not None:
                        self._require_open()
                    self._runtime_scope_guard()
                return result

        snapshot = self._delegate.snapshot(
            timestamp=normalized_timestamp,
            env_ids=self._full_env_ids.clone(),
        )
        if not isinstance(snapshot, SceneSnapshot):
            raise TypeError(
                "RegistrySceneProvider.snapshot() must return SceneSnapshot."
            )
        if snapshot.timestamp != normalized_timestamp:
            raise ValueError("Scene snapshot timestamp must match the requested tick.")
        if self._runtime_scope_guard is not None:
            result = self._select_rows(snapshot, rows)
            if self._require_open is not None:
                self._require_open()
            self._runtime_scope_guard()
            self._timestamp = normalized_timestamp
            self._snapshot = snapshot
            return result
        self._timestamp = normalized_timestamp
        self._snapshot = snapshot
        return self._select_rows(snapshot, rows)

    def _select_rows(
        self,
        snapshot: SceneSnapshot,
        rows: tuple[int, ...],
    ) -> SceneSnapshot:
        """Project one cached full-batch snapshot to an ordered row subset."""
        full_size = int(self._full_env_ids.numel())
        if rows == tuple(range(full_size)):
            return snapshot
        entities: dict[str, EntityState] = {}
        for entity_id, state in snapshot.entities.items():
            pose = state.pose
            if pose.dim() == 3:
                if pose.shape[0] != full_size:
                    raise ValueError(
                        f"Scene entity {entity_id!r} batch does not match "
                        "full_env_ids."
                    )
                index = torch.tensor(rows, dtype=torch.long, device=pose.device)
                pose = pose.index_select(0, index)
            entities[entity_id] = EntityState(pose, confidence=state.confidence)

        articulation_joints: dict[tuple[str, str], ObservedArticulationJointState] = {}
        for address, state in snapshot.articulation_joints.items():
            position = state.position
            valid = state.valid_mask
            if position.dim() == 2:
                if position.shape[0] != full_size:
                    raise ValueError(
                        f"Scene articulation joint {address!r} batch does not "
                        "match full_env_ids."
                    )
                index = torch.tensor(rows, dtype=torch.long, device=position.device)
                position = position.index_select(0, index)
                if valid is not None:
                    valid = valid.index_select(0, index.to(valid.device))
            articulation_joints[address] = ObservedArticulationJointState(
                position,
                valid,
            )

        revisions = snapshot.collision_world_revisions(full_size)
        return SceneSnapshot(
            timestamp=snapshot.timestamp,
            version=snapshot.version,
            entities=entities,
            collision_world_revision=tuple(revisions[row] for row in rows),
            collision_entity_ids=snapshot.collision_entity_ids,
            articulation_joints=articulation_joints,
        )


class SimulationPlanningObservationProvider(GymPlanningObservationProvider):
    """Gym planning observations backed by live robot and shared scene state.

    Args:
        robot: Live robot supplying the synchronized observation.
        scene_provider: Provider sharing scene observations within a host tick.
        clock: Environment control clock.
        env_ids: Stable IDs in native robot row order.
        owner_token: Factory identity used to reject mixed runtime components.
        runtime_scope_guard: Optional host epoch/lifetime validator. It must read
            no robot/sensor state and must not advance simulation.
        capture_root_pose: Capture the robot's arena-local root transform in
            each observation. The default leaves ``RobotObservation.root_pose``
            unset and performs no root-pose query.
        hold_qpos_source: Use the latest captured observation by default, or
            explicitly read live controller position targets for normal holds.
        world_stamp_reader: Optional factual native source reader. Capture rejects
            a changed source between the robot and scene observations.
    """

    def __init__(
        self,
        robot: Robot,
        scene_provider: SharedTickSceneProvider,
        clock: EnvironmentStepClock,
        env_ids: torch.Tensor,
        *,
        owner_token: object,
        runtime_scope_guard: Callable[[], None] | None = None,
        capture_root_pose: bool = False,
        hold_qpos_source: Literal["observed", "drive_target"] = "observed",
        world_stamp_reader: Callable[[], PhysicalObservationStamp] | None = None,
    ) -> None:
        if type(scene_provider) is not SharedTickSceneProvider:
            raise TypeError("scene_provider must be exactly SharedTickSceneProvider.")
        if type(clock) is not EnvironmentStepClock:
            raise TypeError("clock must be exactly EnvironmentStepClock.")
        if type(capture_root_pose) is not bool:
            raise TypeError("capture_root_pose must be a bool.")
        if hold_qpos_source not in ("observed", "drive_target"):
            raise ValueError("hold_qpos_source must be observed or drive_target.")
        if runtime_scope_guard is not None:
            if not callable(runtime_scope_guard):
                raise TypeError("runtime_scope_guard must be callable or None.")
            runtime_scope_guard()
        qpos = _full_robot_tensor(robot, "get_qpos", required=True)
        assert qpos is not None
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.shape != (qpos.shape[0],)
        ):
            raise ValueError("env_ids must be int64 with one ID per robot row.")
        if env_ids.device != qpos.device:
            raise ValueError("env_ids and robot qpos must share a device.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")
        self._robot = robot
        self._scene_provider = scene_provider
        self._clock = clock
        self._env_ids = env_ids.clone()
        self._owner_token = owner_token
        if world_stamp_reader is not None and not callable(world_stamp_reader):
            raise TypeError("world_stamp_reader must be callable or None.")
        self._world_stamp_reader = world_stamp_reader
        self._runtime_scope_guard = runtime_scope_guard
        self._capture_root_pose = capture_root_pose
        self._hold_qpos_source = hold_qpos_source
        super().__init__(self._capture)

    @property
    def scene_provider(self) -> SharedTickSceneProvider:
        """Return the snapshot-sharing scene provider used by evidence ports."""
        return self._scene_provider

    @property
    def env_ids(self) -> torch.Tensor:
        """Return stable ordered simulation row IDs."""
        return self._env_ids.clone()

    def is_owned_by(self, owner_token: object) -> bool:
        """Return whether this provider belongs to one factory instance."""
        return self._owner_token is owner_token

    def validate_runtime_scope(self) -> None:
        """Validate factory lifetime and the explicit host scope without data reads."""
        if isinstance(self._owner_token, _RuntimeOwner):
            self._owner_token.require_open()
        if self._runtime_scope_guard is not None:
            self._runtime_scope_guard()

    def current_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return owned cached joints only while their runtime scope is current.

        Args:
            env_ids: Ordered subset of this provider's stable environment IDs.

        Returns:
            Owned cached positions without querying the robot.
        """
        self.validate_runtime_scope()
        result = super().current_qpos(env_ids)
        if self._runtime_scope_guard is not None:
            self.validate_runtime_scope()
        return result

    def hold_qpos(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the configured source for unaddressed robot joint holds.

        The default delegates to the latest captured observation without a new
        robot read. ``drive_target`` reads controller targets explicitly, keeping
        tracking error from changing an idle manipulator's commanded position.

        Args:
            env_ids: Ordered subset of this provider's stable environment IDs.

        Returns:
            Owned full hold positions in the requested order.
        """
        self.validate_runtime_scope()
        if self._hold_qpos_source == "observed":
            return self.current_qpos(env_ids)
        if (
            not isinstance(env_ids, torch.Tensor)
            or env_ids.dtype != torch.long
            or env_ids.dim() != 1
            or env_ids.numel() == 0
        ):
            raise ValueError("env_ids must be a non-empty one-dimensional long tensor.")
        if env_ids.device != self._env_ids.device:
            raise ValueError("env_ids must share the simulation environment device.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must be unique.")
        row_by_id = {
            int(env_id): row
            for row, env_id in enumerate(self._env_ids.detach().cpu().tolist())
        }
        try:
            rows = [
                row_by_id[int(env_id)] for env_id in env_ids.detach().cpu().tolist()
            ]
        except KeyError as exc:
            raise ValueError(
                f"Environment ID {int(exc.args[0])} is absent from the simulation."
            ) from exc
        target_qpos = self._robot.get_qpos(target=True)
        if (
            not isinstance(target_qpos, torch.Tensor)
            or not target_qpos.is_floating_point()
            or target_qpos.shape
            != (self._env_ids.numel(), int(getattr(self._robot, "dof", 0)))
            or target_qpos.device != self._env_ids.device
            or not bool(torch.isfinite(target_qpos).all().item())
        ):
            raise ValueError(
                "robot.get_qpos(target=True) must return finite floating full-qpos "
                "shape (num_envs, robot_dof) on the simulation device."
            )
        index = torch.tensor(rows, dtype=torch.long, device=target_qpos.device)
        result = target_qpos.index_select(0, index).clone()
        if self._runtime_scope_guard is not None:
            self.validate_runtime_scope()
        return result

    def _capture(self, task_state: TaskState) -> PlanningContext:
        """Capture one synchronized robot and scene observation."""
        self.validate_runtime_scope()
        world_stamp = (
            None if self._world_stamp_reader is None else self._world_stamp_reader()
        )
        qpos = _full_robot_tensor(self._robot, "get_qpos", required=True)
        assert qpos is not None
        if (
            qpos.shape[0] != self._env_ids.numel()
            or qpos.device != self._env_ids.device
        ):
            raise ValueError("Robot batch shape or device changed after assembly.")
        qvel = _full_robot_tensor(
            self._robot,
            "get_qvel",
            required=False,
            reference=qpos,
        )
        if qvel is None:
            qvel = torch.zeros_like(qpos)
        qeffort = _full_robot_tensor(
            self._robot,
            "get_qf",
            required=False,
            reference=qpos,
        )
        timestamp = self._clock.now()
        scene = self._scene_provider.snapshot(
            timestamp=timestamp,
            env_ids=self._env_ids.clone(),
        )
        root_pose = None
        if self._capture_root_pose:
            root_pose = self._robot.get_local_pose(to_matrix=True)
            if not isinstance(root_pose, torch.Tensor):
                raise TypeError("Robot root observation must be a tensor.")
            root_pose = root_pose.to(qpos)
        if (
            self._world_stamp_reader is not None
            and self._world_stamp_reader() != world_stamp
        ):
            raise RuntimeError(
                "Native world source changed during planning observation."
            )
        context = PlanningContext(
            world_source_stamp=world_stamp,
            robot=RobotObservation(
                timestamp=timestamp,
                qpos=qpos,
                qvel=qvel,
                qeffort=qeffort,
                root_pose=root_pose,
            ),
            task=task_state,
            scene=scene,
            env_ids=self._env_ids,
            control_dt=self._clock.step_dt,
        )
        if self._runtime_scope_guard is not None:
            self.validate_runtime_scope()
        return context


class SimulationTaskProgramFactory(TaskProgramEnvironmentFactory):
    """Build every live Task Program component from one task registration.

    Args:
        simulation: Exact live simulation that owns ``robot`` and scene UIDs.
        robot: Exact robot selected for planning and evidence acquisition.
        registration: Frozen scene, profile, and extension declarations.
        step_dt: Authoritative Gym control cadence.
        joint_command_mode: Environment-owned expert joint command mode.
        planner_cfg: Explicit planner configuration.  ``None`` selects TOPPRA
            for ``robot.uid``.
        planner_config: Optional executable-free planner declaration from an
            execution policy. It is converted to ``planner_cfg`` after the
            selected runtime robot is known.
        motion_generator_factory: Optional fresh-generator factory.  It is
            mutually exclusive with ``planner_cfg`` and intended for custom
            planners and isolated tests.
        grasp_pose_generators: Standalone grasp-pose services keyed by grasp
            endpoint target ID, normally a robot control-part name.
        hand_geometry_cfgs: Explicit hand calibration settings keyed by grasp target ID.
        translation_threshold: Material scene translation threshold.
        rotation_threshold: Material scene rotation threshold.
        sensor_suite: Environment-owned mapping used to drive declared support
            sensors through the normal physics-substep lifecycle.

        capture_root_pose: Capture an owned arena-local robot root transform in
            each planning observation. Disabled by default.
        hold_qpos_source: Source for normal controller holds. ``observed`` keeps
            the captured-qpos behavior; ``drive_target`` explicitly reads targets.

    The Gym cadence is attached to each live :class:`PlanningContext`.  Motion
    policy presets describe behavior; they do not own the environment's
    command period.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        registration: SimulationTaskProgramRegistration,
        *,
        step_dt: float,
        joint_command_mode: JointCommandMode = "position",
        planner_cfg: BasePlannerCfg | None = None,
        motion_generator_factory: MotionGeneratorFactory | None = None,
        grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
        hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg] | None = None,
        translation_threshold: float = 1.0e-4,
        rotation_threshold: float = 1.0e-3,
        sensor_suite: MutableMapping[str, object] | None = None,
        episode_identity: Callable[[], str] | None = None,
        capture_root_pose: bool | None = None,
        hold_qpos_source: Literal["observed", "drive_target"] | None = None,
    ) -> None:
        if type(registration) is not SimulationTaskProgramRegistration:
            raise TypeError(
                "registration must be exactly SimulationTaskProgramRegistration."
            )
        registration.assert_unchanged()
        if registration.recorded_contacts_cfg is not None:
            registration.recorded_contacts_cfg.validate_num_envs(simulation.num_envs)
        declared_services = registration.planning_services
        if declared_services is not None:
            if capture_root_pose is not None or hold_qpos_source is not None:
                raise ValueError(
                    "Registered planning services own observation settings; remove constructor overrides."
                )
            declared_services = declared_services.snapshot()
            capture_root_pose = declared_services.capture_root_pose
            hold_qpos_source = declared_services.hold_qpos_source
            registration.validate_num_envs(simulation.num_envs)
            registration.validate_physics(simulation.physics_backend)
        else:
            capture_root_pose = (
                False if capture_root_pose is None else capture_root_pose
            )
            hold_qpos_source = (
                "observed" if hold_qpos_source is None else hold_qpos_source
            )
        if type(capture_root_pose) is not bool:
            raise TypeError("capture_root_pose must be a bool.")
        if hold_qpos_source not in ("observed", "drive_target"):
            raise ValueError("hold_qpos_source must be observed or drive_target.")
        registration.assert_unchanged()
        selected_scene_binding = registration.scene_binding
        selected_profile_binding = registration.robot_profile_binding
        selected_endpoint_adapters = dict(registration.endpoint_adapter_map)
        selected_settle_presets = registration.settle_presets
        if planner_cfg is not None and motion_generator_factory is not None:
            raise ValueError(
                "planner_cfg and motion_generator_factory are mutually exclusive."
            )
        if planner_cfg is not None and not isinstance(planner_cfg, BasePlannerCfg):
            raise TypeError("planner_cfg must be a BasePlannerCfg or None.")
        if motion_generator_factory is not None and not callable(
            motion_generator_factory
        ):
            raise TypeError("motion_generator_factory must be callable or None.")
        if grasp_pose_generators is not None and not isinstance(
            grasp_pose_generators, Mapping
        ):
            raise TypeError("grasp_pose_generators must be a mapping or None.")
        if hand_geometry_cfgs is not None and not isinstance(
            hand_geometry_cfgs, Mapping
        ):
            raise TypeError("hand_geometry_cfgs must be a mapping or None.")
        if joint_command_mode not in ("position", "position_velocity"):
            raise ValueError(
                "joint_command_mode must be 'position' or 'position_velocity'."
            )

        robot_uid = _robot_uid(robot)
        get_robot = getattr(simulation, "get_robot", None)
        if not callable(get_robot):
            raise TypeError("simulation must provide get_robot().")
        if get_robot(robot_uid) is not robot:
            raise ValueError(
                f"simulation.get_robot({robot_uid!r}) must return the exact "
                "selected robot."
            )
        selected_planner_cfg = deepcopy(planner_cfg)
        if (
            selected_planner_cfg is not None
            and selected_planner_cfg.robot_uid != robot_uid
        ):
            raise ValueError(
                f"planner_cfg.robot_uid must equal selected robot UID {robot_uid!r}."
            )

        self._simulation = simulation
        self._robot = robot
        self._registration = registration
        self._scene_binding = selected_scene_binding
        self._robot_profile_binding = selected_profile_binding
        self._step_dt = _positive_finite(step_dt, field_name="step_dt")
        self._joint_command_mode = joint_command_mode
        self._planner_cfg = selected_planner_cfg
        self._motion_generator_factory = motion_generator_factory
        self._grasp_pose_generators = (
            {} if grasp_pose_generators is None else dict(grasp_pose_generators)
        )
        self._hand_geometry_cfgs = (
            {} if hand_geometry_cfgs is None else deepcopy(dict(hand_geometry_cfgs))
        )
        self._endpoint_adapters = selected_endpoint_adapters
        self._translation_threshold = _non_negative_finite(
            translation_threshold,
            field_name="translation_threshold",
        )
        self._rotation_threshold = _non_negative_finite(
            rotation_threshold,
            field_name="rotation_threshold",
        )
        self._owner_token = _RuntimeOwner()
        self._coordinated_grasp_services = None
        if declared_services is not None and declared_services.coordinated_grasps:
            from .coordinated_grasp import ConfiguredCoordinatedGraspServices

            self._coordinated_grasp_services = ConfiguredCoordinatedGraspServices(
                robot,
                generators=self._grasp_pose_generators,
                geometry_cfgs=self._hand_geometry_cfgs,
                declarations=declared_services.coordinated_grasps,
                validate_owner=self._owner_token.require_open,
            )
            self._grasp_pose_generators = dict(
                self._coordinated_grasp_services.generators
            )
        self._capture_root_pose = capture_root_pose
        self._hold_qpos_source = hold_qpos_source
        self._capture_native_revision = (
            declared_services is not None and declared_services.capture_native_revision
        )
        self._observation_source_id = (
            uuid4().hex if self._capture_native_revision else None
        )

        self._sensor_suite = sensor_suite
        self._episode_identity = episode_identity
        self._control_bridge_services: list[_ControlBridgeServices] = []
        self._observation_services: list[SimulationObservationServices] = []
        if registration.requires_scoped_control_services and not callable(
            episode_identity
        ):
            raise ValueError(
                "Configured controls require an explicit current episode identity reader."
            )

        qpos = _full_robot_tensor(robot, "get_qpos", required=True)
        assert qpos is not None
        self._env_ids = torch.arange(
            qpos.shape[0],
            dtype=torch.long,
            device=qpos.device,
        )
        self._support_registry = None
        self._runtime_bridge_ref: ReferenceType[TaskProgramDemoBridge] | None = None
        self._support_clock: EnvironmentStepClock | None = None
        conditions = [
            cfg
            for cfg in selected_settle_presets.values()
            if type(cfg) is SimulationStabilityConditionCfg
        ]
        if any(cfg.support_sensor is not None for cfg in conditions):
            if not isinstance(sensor_suite, MutableMapping):
                raise ValueError(
                    "Supported stability requires the environment sensor suite."
                )
            from embodichain.lab.sim.sensors.support_observation import (
                SupportObservationRegistry,
            )

            self._support_registry = SupportObservationRegistry(
                simulation, acquisition_clock=lambda: simulation.simulation_time
            )
        try:
            self._segment_policy_port = SimulationSegmentPolicyPort(
                simulation,
                robot,
                selected_scene_binding,
                settle_presets=selected_settle_presets,
                env_ids=self._env_ids,
                step_dt=self._step_dt,
                support_registry=self._support_registry,
                sensor_suite=sensor_suite,
                control_declarations={
                    value.call.operation_id: value
                    for value in registration.control_operations
                },
            )
        except Exception:
            if self._support_registry is not None:
                self._support_registry.close()
            raise

    def validate_runtime_bridge_creation(self) -> None:
        """Reject an overlapping support bridge before replacing its clock.

        A completed or safely closed segment iterator releases its clock owner.
        Unacknowledged actions retain ownership until the normal abort handshake.
        Non-support factories retain their existing independent assembly behavior.
        """
        self._owner_token.require_open()
        if self._support_registry is None or self._runtime_bridge_ref is None:
            return
        previous = self._runtime_bridge_ref()
        if previous is not None and not previous.execution_finished:
            raise RuntimeError(
                "An active support bridge still owns the execution clock; "
                "finish or safely close its segment iterator before creating another bridge."
            )

    def bind_runtime_bridge(self, bridge: TaskProgramDemoBridge) -> None:
        """Retain the successful bridge's support-clock ownership weakly.

        Args:
            bridge: Actual bridge returned by this factory's environment adapter.
        """
        self.validate_runtime_bridge_creation()
        if (
            type(bridge) is not TaskProgramDemoBridge
            or bridge.clock is not self._support_clock
        ):
            raise ValueError(
                "The runtime bridge must use this factory's assembled clock."
            )
        if self._support_registry is not None:
            self._runtime_bridge_ref = ref(bridge)

    def close_runtime_resources(self) -> None:
        """Close owned observation hooks without stepping or destroying resources."""
        if self._owner_token.closed:
            return
        self._owner_token.closed = True
        if self._coordinated_grasp_services is not None:
            self._coordinated_grasp_services.close()

        for services in tuple(self._observation_services):
            services.close()
        self._observation_services.clear()
        for services in self._control_bridge_services:
            services.close_host_resources()
        self._control_bridge_services.clear()
        self._segment_policy_port.close_runtime_resources()
        if self._support_registry is not None:
            self._support_registry.close()

    @classmethod
    def from_environment(
        cls,
        environment: SimulationTaskProgramEnvironment,
        *,
        registration: SimulationTaskProgramRegistration,
        planner_cfg: BasePlannerCfg | None = None,
        planner_config: Mapping[str, object] | None = None,
        motion_generator_factory: MotionGeneratorFactory | None = None,
        grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
        hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg] | None = None,
        translation_threshold: float = 1.0e-4,
        rotation_threshold: float = 1.0e-3,
        capture_root_pose: bool | None = None,
        hold_qpos_source: Literal["observed", "drive_target"] | None = None,
    ) -> SimulationTaskProgramFactory:
        """Create a registered factory from the standard Gym environment surface."""
        simulation = getattr(environment, "sim", None)
        robot = getattr(environment, "robot", None)
        try:
            step_dt = environment.step_dt
        except AttributeError as exc:
            raise TypeError("environment must expose step_dt.") from exc
        if simulation is None or robot is None:
            raise TypeError("environment must expose non-None sim and robot values.")
        if planner_cfg is not None and planner_config is not None:
            raise ValueError("planner_cfg and planner_config are mutually exclusive.")
        if planner_config is not None:
            planner_cfg = _planner_cfg_from_config(
                planner_config,
                robot_uid=_robot_uid(robot),
            )
        expert_trajectory_cfg = getattr(
            getattr(environment, "cfg", None),
            "expert_trajectory",
            None,
        )
        joint_command_mode = getattr(
            expert_trajectory_cfg,
            "joint_command_mode",
            "position",
        )
        episode_identity = None
        if registration.requires_scoped_control_services or (
            registration.planning_services is not None
            and registration.planning_services.released_hand_continuations
        ):
            context = getattr(environment, "episode_context", None)
            if context is None or not callable(getattr(context, "snapshot", None)):
                raise ValueError(
                    "Control services require the environment's public EpisodeContext."
                )

            def episode_identity() -> str:
                if environment.episode_context is not context:
                    raise RuntimeError(
                        "Control environment episode owner was replaced."
                    )
                rows = context.snapshot([0])
                if (
                    len(rows) != 1
                    or rows[0].get("env_id") != 0
                    or rows[0].get("initialized") is not True
                    or type(rows[0].get("episode")) is not int
                    or rows[0]["episode"] < 0
                ):
                    raise ValueError(
                        "Control services require an initialized actual episode."
                    )
                return str(rows[0]["episode"])

        return cls(
            simulation,
            robot,
            registration,
            step_dt=step_dt,
            joint_command_mode=joint_command_mode,
            planner_cfg=planner_cfg,
            motion_generator_factory=motion_generator_factory,
            grasp_pose_generators=grasp_pose_generators,
            hand_geometry_cfgs=hand_geometry_cfgs,
            translation_threshold=translation_threshold,
            rotation_threshold=rotation_threshold,
            sensor_suite=getattr(environment, "sensors", None),
            episode_identity=episode_identity,
            capture_root_pose=capture_root_pose,
            hold_qpos_source=hold_qpos_source,
        )

    @property
    def scene_registry_id(self) -> str:
        """Return the exact configured scene-registry ID."""
        return self._scene_binding.registry_id

    @property
    def robot_profile_id(self) -> str:
        """Return the exact configured robot-profile ID."""
        return self._robot_profile_binding.profile_id

    @property
    def step_dt(self) -> float:
        """Return the authoritative Gym control cadence."""
        return self._step_dt

    @property
    def segment_policy_port(self) -> SimulationSegmentPolicyPort:
        """Return the shared simulation post-policy and validator port."""
        return self._segment_policy_port

    @property
    def task_program_registration(self) -> SimulationTaskProgramRegistration:
        """Return the exact standard registration owned by this factory."""
        return self._registration

    def registration_owned_segment_policy_ports(
        self,
    ) -> tuple[SimulationSegmentPolicyPort, SimulationSegmentPolicyPort]:
        """Return registration-owned post-policy and validator ports."""
        return self._segment_policy_port, self._segment_policy_port

    def create_registered_semantic_lowerers(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        runtime_services: TaskProgramRuntimeServiceContext | None = None,
    ) -> tuple[RegisteredSemanticLowerer, ...]:
        """Create fresh registration-owned lowerers for this runtime assembly."""
        return self._registration.create_registered_semantic_lowerers(
            simulation=self._simulation,
            robot=self._robot,
            scene_registry=scene_registry,
            engine=engine,
            runtime_services=runtime_services,
        )

    def create_control_bridge_services(
        self,
        program: CompiledTaskProgram,
        clock: EnvironmentStepClock,
        *,
        invocation_policy_bindings: tuple[InvocationPolicyBinding, ...] = (),
    ) -> _ControlBridgeServices:
        """Assemble exact program/episode operations before live lowerer construction."""
        self._owner_token.require_open()
        if (
            not self._registration.requires_scoped_control_services
            or self._episode_identity is None
        ):
            raise ValueError(
                "No explicit control services and episode reader are declared."
            )
        self._registration.assert_unchanged()
        self._registration.catalog.validate_control_program(program)
        articulation_bindings = bind_articulation_invocations(
            program, self._registration, invocation_policy_bindings
        )
        episode = self._episode_identity()
        owner = _RuntimeOwner()

        def guard() -> None:
            self._owner_token.require_open()
            owner.require_open()
            if self._episode_identity() != episode:
                raise RuntimeError("Control bridge belongs to a replaced episode.")

        def observed(context: PlanningContext) -> None:
            guard()
            if (
                context.env_ids.tolist() != [0]
                or context.robot.timestamp != clock.now()
                or context.scene.timestamp != clock.now()
            ):
                raise ValueError(
                    "Control planning observation differs from its bridge clock."
                )

        fingerprint = task_program_fingerprint(program)
        selected = {
            call_id: declaration.call.operation_id
            for call_id, declaration, prepare in self._registration.catalog.control_calls
            if not prepare and declaration.chunk_count is not None
        }
        invocations = {
            value.call.operation_id: []
            for value in self._registration.control_operations
        }
        for segment in program.iter_segments():
            for call in segment.calls:
                operation_id = selected.get(getattr(call.call, "call_id", None))
                if operation_id is not None:
                    invocations[operation_id].append(
                        f"{program.program_id}/{segment.segment_id}:{call.segment_call_index}"
                    )
        built = []
        port = None
        try:
            for declaration in self._registration.control_operations:
                built.append(
                    assemble_control_operation(
                        declaration,
                        simulation=self._simulation,
                        robot=self._robot,
                        sensor_suite=self._sensor_suite,
                        hand_geometry_cfgs=self._hand_geometry_cfgs,
                        program_fingerprint=fingerprint,
                        episode_id=episode,
                        invocation_ids=invocations[declaration.call.operation_id],
                        control_dt=self._step_dt,
                        validate_scope=guard,
                        validate_observation=observed,
                    )
                )
            port = SimulationSegmentPolicyPort(
                self._simulation,
                self._robot,
                self._scene_binding,
                settle_presets=self._registration.settle_presets,
                env_ids=self._env_ids,
                step_dt=self._step_dt,
                support_registry=self._support_registry,
                sensor_suite=self._sensor_suite,
                control_operations={
                    value.operation.operation_id: value.operation for value in built
                },
                **(
                    {
                        "bound_post_policy_presets": BoundPostPolicyPresets(
                            program, self._registration.post_policy_bindings
                        )
                    }
                    if self._registration.post_policy_bindings
                    else {}
                ),
            )
            port.bind_clock(clock)
            sessions = tuple(
                value.twist_session
                for value in built
                if value.twist_session is not None
            )
            registry = TwistOperationRegistry(sessions) if sessions else None
            articulation_registry = (
                ArticulationContinuationRegistry(
                    self._robot,
                    articulation_bindings,
                    program_fingerprint=fingerprint,
                    episode_id=episode,
                    validate_scope=guard,
                    validate_observation=observed,
                )
                if articulation_bindings
                else None
            )
            services = _ControlBridgeServices(
                port.runtime_service_context(
                    program_fingerprint=fingerprint, episode_id=episode
                ),
                port,
                registry,
                tuple(built),
                owner,
                guard,
                articulation_registry=articulation_registry,
            )
            self._control_bridge_services.append(services)
            return services
        except Exception:
            owner.closed = True
            if port is not None:
                port.close_runtime_resources()
            for value in built:
                value.close()
            raise

    def create_scene_registry(self) -> SceneRegistry:
        """Build one fresh authoritative registry from the task registration."""
        self._owner_token.require_open()
        registry = self._scene_binding.build(self._simulation)
        self._registration.validate_scene_registry(registry)
        return registry

    def create_robot_skill_profile(self) -> RobotSkillProfile:
        """Build the robot profile used to lower calls to Atomic Skills."""
        self._owner_token.require_open()
        profile = self._robot_profile_binding.build(self._robot)
        self._registration.validate_robot_profile(profile)
        return profile

    def create_atomic_action_engine(
        self,
        profile: RobotSkillProfile,
        *,
        pour_projection_routes: tuple[ResolvedPourProjectionRoute, ...] = (),
        planning_program: CompiledTaskProgram | None = None,
        twist_planning_provider: TwistOperationRegistry | None = None,
        articulation_continuation_provider: (
            ArticulationContinuationRegistry | None
        ) = None,
    ) -> AtomicActionEngine:
        """Create a fresh engine around the selected planner and exact profile."""
        self._owner_token.require_open()
        if not isinstance(profile, RobotSkillProfile):
            raise TypeError("profile must be a RobotSkillProfile.")
        if profile.profile_id != self.robot_profile_id:
            raise ValueError(
                f"profile ID must be {self.robot_profile_id!r}, got "
                f"{profile.profile_id!r}."
            )
        continuation = None
        declared_services = self._registration.planning_services
        if (
            declared_services is not None
            and declared_services.place_continuation_bindings
        ):
            if planning_program is None:
                raise ValueError(
                    "Place continuations require a compiled planning_program before engine creation."
                )
            from .place_continuation import SimulationPlaceContinuationProvider

            continuation = SimulationPlaceContinuationProvider(
                self._robot,
                planning_program,
                self._registration,
                bindings=declared_services.place_continuation_bindings,
                require_open=self._owner_token.require_open,
            )
        released_provider = None
        if (
            declared_services is not None
            and declared_services.released_hand_continuations
        ):
            if not callable(self._episode_identity):
                raise ValueError(
                    "Released continuation requires the actual episode identity reader."
                )
            if planning_program is None:
                raise ValueError(
                    "Released continuations require the exact compiled program scope."
                )
            from .released_continuation import (
                SimulationReleasedHandContinuationProvider,
            )

            released_provider = SimulationReleasedHandContinuationProvider(
                self._robot,
                planning_program,
                self._registration,
                bindings=declared_services.released_hand_continuations,
                require_open=self._owner_token.require_open,
                read_source_stamp=self.world_source_stamp,
                read_episode=self._episode_identity,
            )
        motion_generator = self._create_motion_generator()
        if motion_generator.robot is not self._robot:
            raise ValueError(
                "Motion generator must own the exact robot selected by the factory."
            )
        free_motion_query = None
        services = self._registration.planning_services
        if services is not None and services.free_motion_query == "discrete_mesh":
            from embodichain.lab.sim.motion.mesh_collision import (
                SimulationFreeMotionQuery,
            )

            free_motion_query = SimulationFreeMotionQuery(self._robot, self._simulation)
        joint_prefix = None
        if services is not None and services.observed_curobo is not None:
            from embodichain.lab.sim.atomic_actions.curobo_queries import (
                CuroboJointPrefixPlanner,
            )
            from embodichain.lab.sim.scene.collision_sources import (
                SimulationCollisionShapeReader,
            )
            from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
                CuroboPlannerCfg,
                CuroboWorldCfg,
            )
            from embodichain.lab.sim.motion.planners.curobo.observed import (
                ObservedCuroboQueryFactory,
            )
            from .collision_observation import SimulationCollisionObservationProvider

            declared = services.observed_curobo.snapshot()
            roster = services.collision_roster.snapshot()
            shapes = SimulationCollisionShapeReader(
                self._simulation,
                rigid_objects=dict(roster.rigid_objects),
                articulation_links={
                    key: (uid, link) for key, uid, link in roster.articulation_links
                },
            )
            cfg = CuroboPlannerCfg(
                robot_uid=self._robot.uid,
                sim_instance_id=self._simulation.instance_id,
                world=CuroboWorldCfg(max_voxel_count=declared.max_voxel_count),
                use_cuda_graph=False,
                warmup_iterations=0,
                preserve_plan_samples=True,
                interpolation_dt=self._step_dt,
                cuda_device=declared.cuda_device,
            )
            queries = ObservedCuroboQueryFactory(self._robot, cfg, shapes)
            world = SimulationCollisionObservationProvider(
                self._simulation, roster, self.world_source_stamp
            )
            joint_prefix = CuroboJointPrefixPlanner(queries, world)
        place_scene = None
        if services is not None and services.place_scene_bindings:
            from .place_scene import SimulationPlaceSceneProvider

            roster = services.collision_roster.snapshot()
            logical_by_native = {
                item.simulation_uid: item.entity_id
                for item in self._scene_binding.articulations
            }
            articulation_uids = {
                logical_by_native.get(uid, uid): uid
                for _, uid, _ in roster.articulation_links
            }
            place_scene = SimulationPlaceSceneProvider(
                self._simulation,
                rigid_uids=dict(roster.rigid_objects),
                articulation_uids=articulation_uids,
                link_bindings=tuple(
                    (logical_by_native.get(uid, uid), link, key)
                    for key, uid, link in roster.articulation_links
                ),
                bindings=services.place_scene_bindings,
                containers=self._scene_binding.containers,
            )
        handover_scene = None
        if services is not None and services.native_handover_scene:
            from .handover_scene import SimulationHandOverSceneProvider

            roster = services.collision_roster
            required = tuple(
                dict.fromkeys(
                    [
                        *(uid for _, uid in roster.rigid_objects),
                        *(uid for _, uid, _ in roster.articulation_links),
                    ]
                )
            )
            handover_scene = SimulationHandOverSceneProvider(
                self._robot,
                self._simulation,
                scene_binding=self._scene_binding,
                required_owner_uids=required,
                read_source_stamp=self.world_source_stamp,
            )
        cargo_geometry = None
        if services is not None and services.capture_coordinated_cargo:
            from embodichain.lab.sim.scene.mesh_snapshot import capture_scene_meshes

            cargo_geometry = capture_scene_meshes(
                self._simulation,
                rigid_uids={
                    item.entity_id: item.simulation_uid
                    for item in self._scene_binding.rigid_objects
                },
            )
        held_planning = None
        if services is not None and services.held_staging_bindings:
            from .held_planning import (
                SimulationHeldPlanningProvider,
                _declared_authorities,
            )
            from embodichain.lab.sim.atomic_actions.planning_probes import (
                HeldStagingBinding,
            )

            held_planning = SimulationHeldPlanningProvider(
                self._robot,
                self._simulation,
                bindings={
                    value.invocation_id: HeldStagingBinding(
                        rigid_entity_id=value.rigid_entity_id,
                        articulation_scene=value.articulation_scene,
                    )
                    for value in services.held_staging_bindings
                },
                retained_authorities=_declared_authorities(
                    profile,
                    self._registration.invocation_policy_bindings,
                    services.held_staging_bindings,
                ),
                read_source_stamp=self.world_source_stamp,
            )
        pour_projection = None
        declared_pour = () if services is None else services.pour_projection_bindings
        if (
            type(pour_projection_routes) is not tuple
            or {value.pick_id for value in pour_projection_routes}
            != {value.pick_invocation_id for value in declared_pour}
            or len(pour_projection_routes) != len(declared_pour)
        ):
            raise ValueError("Pour projection requires exact compiled role bindings.")
        if pour_projection_routes:
            from embodichain.lab.sim.scene.pour_inputs import (
                capture_pour_vertices,
                capture_pour_faces,
                configured_pour_gravity,
            )
            from embodichain.lab.sim.scene.pour_region import PourRegionGeometry
            from .pour_projection import SimulationPourProjectionProvider

            native_ids = {
                value.entity_id: value.simulation_uid
                for value in self._scene_binding.rigid_objects
            }
            identifiers = tuple(
                dict.fromkeys(
                    identifier
                    for route in pour_projection_routes
                    for identifier in (route.object_id, route.receiver_id)
                )
            )
            entities = {
                identifier: self._simulation.get_rigid_object(native_ids[identifier])
                for identifier in identifiers
            }
            vertices = {
                identifier: capture_pour_vertices(entity)
                for identifier, entity in entities.items()
            }
            gravity = configured_pour_gravity(self._simulation)
            faces = {
                identifier: capture_pour_faces(entities[identifier])
                for identifier in dict.fromkeys(
                    route.receiver_id
                    for route in pour_projection_routes
                    if route.proxy_position_tolerance_m is not None
                )
            }
            geometries = {
                budget: PourRegionGeometry(vertices, faces, gravity, budget)
                for budget in dict.fromkeys(
                    route.proxy_position_tolerance_m
                    for route in pour_projection_routes
                    if route.proxy_position_tolerance_m is not None
                )
            }
            pour_projection = SimulationPourProjectionProvider(
                self._robot,
                routes=pour_projection_routes,
                vertices=vertices,
                gravity_world=gravity,
                hand_geometry_cfgs=self._hand_geometry_cfgs,
                region_geometries={
                    route.pick_id: geometries[route.proxy_position_tolerance_m]
                    for route in pour_projection_routes
                    if route.proxy_position_tolerance_m is not None
                },
                read_source_stamp=self.world_source_stamp,
            )
        pick_closing = None
        if services is not None and services.pick_closing_bindings:
            from .pick_closing import SimulationPickClosingProvider

            pick_closing = SimulationPickClosingProvider(
                self._robot,
                self._simulation,
                rigid_uids={
                    item.entity_id: item.simulation_uid
                    for item in self._scene_binding.rigid_objects
                },
                bindings=services.pick_closing_bindings,
                read_source_stamp=self.world_source_stamp,
            )
        slide_clearance = None
        if services is not None and services.slide_preshape_bindings:
            from .slide_preshape import SimulationSlideHandClearanceProvider

            ordered_cfgs = {
                resource.endpoints["grasp"].control_part: self._hand_geometry_cfgs[
                    resource.endpoints["grasp"].control_part
                ]
                for resource in profile.resources.values()
                if "motion" in resource.endpoints and "grasp" in resource.endpoints
            }
            slide_clearance = SimulationSlideHandClearanceProvider(
                self._robot,
                self._simulation,
                bindings=services.slide_preshape_bindings,
                hand_geometry_cfgs=ordered_cfgs,
                read_source_stamp=self.world_source_stamp,
            )
        drawer_planning = None
        if services is not None and services.drawer_bindings:
            from .drawer_planning import SimulationDrawerPlanningProvider

            drawer_planning = SimulationDrawerPlanningProvider(
                self._robot,
                self._simulation,
                bindings=services.drawer_bindings,
                read_source_stamp=self.world_source_stamp,
            )
        measured = None
        if services is not None and services.measured_rigid_bindings:
            from .measured_rigid import SimulationMeasuredRigidProvider

            measured = SimulationMeasuredRigidProvider(
                self._robot,
                self._simulation,
                bindings=services.measured_rigid_bindings,
                rigid_uids={
                    value.entity_id: value.simulation_uid
                    for value in self._scene_binding.rigid_objects
                },
                containers=self._scene_binding.containers,
                hand_geometry_cfgs=self._hand_geometry_cfgs,
                read_source_stamp=self.world_source_stamp,
            )
        engine = AtomicActionEngine(
            motion_generator,
            coordinated_cargo_geometry=cargo_geometry,
            held_staging_provider=held_planning,
            slide_hand_clearance_provider=slide_clearance,
            drawer_planning_provider=drawer_planning,
            retained_command_provider=held_planning,
            free_motion_query=free_motion_query,
            joint_prefix_planner=joint_prefix,
            place_scene_provider=place_scene,
            place_continuation_provider=continuation,
            released_hand_provider=released_provider,
            measured_rigid_provider=measured,
            handover_scene_provider=handover_scene,
            pick_closing_query_provider=pick_closing,
            pour_grasp_projection_provider=pour_projection,
            control_profiles=profile.action_control_profiles(),
            grasp_pose_generators=self._grasp_pose_generators,
            hand_geometry_cfgs=self._hand_geometry_cfgs,
            twist_planning_provider=twist_planning_provider,
            articulation_continuation_provider=articulation_continuation_provider,
        )
        self._registration.validate_engine(engine)
        return engine

    def bind_compiler_planning_services(
        self, engine: AtomicActionEngine, compiler: SemanticCallCompiler
    ) -> None:
        """Bind declared continuation services to the one assembled compiler."""
        self._owner_token.require_open()
        provider = engine.planning_services.place_continuation_provider
        if provider is not None:
            from .place_continuation import SimulationPlaceContinuationProvider

            if type(provider) is not SimulationPlaceContinuationProvider:
                raise TypeError("Standard continuation service ownership differs.")
            provider.bind_compiler(engine, compiler)

        released = engine.planning_services.released_hand_provider
        if released is not None:
            from .released_continuation import (
                SimulationReleasedHandContinuationProvider,
            )

            if type(released) is not SimulationReleasedHandContinuationProvider:
                raise TypeError("Released continuation service has another owner.")
            released.bind_compiler(engine, compiler)

    def close_compiler_planning_services(self, engine: AtomicActionEngine) -> None:
        """Close this bridge's released continuation authority, without native work."""
        provider = engine.planning_services.released_hand_provider
        if provider is not None:
            provider.close()

    def world_source_stamp(self) -> PhysicalObservationStamp:
        """Read this factory's native-world source without sampling or stepping.

        Returns:
            Actual Default contact revision and manager time for this lifetime.
        """
        self._owner_token.require_open()
        if self._observation_source_id is None:
            raise ValueError("Native observation capture was not declared.")
        from embodichain.lab.sim.sensors.contact.sampling import native_contact_revision

        return PhysicalObservationStamp(
            self._observation_source_id,
            "default.contacts",
            native_contact_revision(self._simulation),
            self._simulation.simulation_time,
        )

    def create_planning_observation_provider(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        clock: EnvironmentStepClock,
        runtime_scope_guard: Callable[[], None] | None = None,
    ) -> PlanningObservationPort:
        """Create one planning port and planner-validated shared scene provider."""
        self.validate_runtime_bridge_creation()
        if runtime_scope_guard is not None:
            if not callable(runtime_scope_guard):
                raise TypeError("runtime_scope_guard must be callable or None.")
            runtime_scope_guard()
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not self._robot:
            raise ValueError("engine must own the exact factory robot.")
        if type(clock) is not EnvironmentStepClock:
            raise TypeError("clock must be exactly EnvironmentStepClock.")
        if clock.step_dt != self._step_dt:
            raise ValueError("clock.step_dt must equal the factory Gym cadence.")
        provider = scene_registry.make_planning_scene_provider(
            engine.motion_generator,
            batch_size=int(self._env_ids.numel()),
            translation_threshold=self._translation_threshold,
            rotation_threshold=self._rotation_threshold,
        )
        shared = SharedTickSceneProvider(
            provider,
            self._env_ids,
            require_open=self._owner_token.require_open,
            runtime_scope_guard=runtime_scope_guard,
        )
        if not self._registration.requires_scoped_control_services:
            self._segment_policy_port.bind_clock(clock)
        self._support_clock = clock
        return SimulationPlanningObservationProvider(
            self._robot,
            shared,
            clock,
            self._env_ids,
            owner_token=self._owner_token,
            runtime_scope_guard=runtime_scope_guard,
            capture_root_pose=self._capture_root_pose,
            hold_qpos_source=self._hold_qpos_source,
            world_stamp_reader=(
                self.world_source_stamp if self._capture_native_revision else None
            ),
        )

    def create_effect_evidence_providers(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: PlanningObservationPort,
    ) -> Iterable[EffectEvidenceProvider]:
        """Create built-in control-part and articulation evidence providers."""
        self._owner_token.require_open()
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if not isinstance(engine, AtomicActionEngine):
            raise TypeError("engine must be an AtomicActionEngine.")
        if engine.robot is not self._robot:
            raise ValueError("engine must own the exact factory robot.")
        if type(observation_provider) is not SimulationPlanningObservationProvider:
            raise TypeError(
                "observation_provider must be exactly "
                "SimulationPlanningObservationProvider."
            )
        if not observation_provider.is_owned_by(self._owner_token):
            raise ValueError("observation_provider belongs to another factory.")
        scene_provider = observation_provider.scene_provider
        providers: list[EffectEvidenceProvider] = []
        if isinstance(self._robot, ControlPartRobotEvidenceSource):
            registered_provider = (
                self._registration.create_control_part_evidence_provider(
                    simulation=self._simulation,
                    robot=self._robot,
                    scene_registry=scene_registry,
                    engine=engine,
                    scene_provider=scene_provider,
                )
            )
            providers.append(
                registered_provider
                if registered_provider is not None
                else ControlPartSimulationEvidenceProvider(
                    self._robot,
                    scene_provider=scene_provider,
                )
            )
        providers.append(
            SceneArticulationEvidenceProvider(scene_provider=scene_provider)
        )

        def require_grip_scope() -> None:
            observation_provider.validate_runtime_scope()
            if self._simulation.get_robot(self._robot.uid) is not self._robot:
                raise ValueError("Native grip runtime robot ownership changed.")

        return tuple(
            (
                _OwnedGripEvidenceProvider(
                    provider, self._owner_token, require_grip_scope
                )
                if isinstance(provider, GripEvidenceObservationPort)
                else _OwnedEffectEvidenceProvider(provider, self._owner_token)
            )
            for provider in providers
        )

    def create_parallel_command_safety_validator(
        self,
        *,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        observation_provider: PlanningObservationPort,
    ) -> ParallelCommandSafetyValidator:
        """Create the registration-owned validator for this runtime assembly."""
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        if (
            not isinstance(engine, AtomicActionEngine)
            or engine.robot is not self._robot
        ):
            raise ValueError("engine must own the exact factory robot.")
        if type(observation_provider) is not SimulationPlanningObservationProvider:
            raise TypeError(
                "observation_provider must be exactly "
                "SimulationPlanningObservationProvider."
            )
        if not observation_provider.is_owned_by(self._owner_token):
            raise ValueError("observation_provider belongs to another factory.")
        validator = self._registration.create_parallel_safety_validator(
            simulation=self._simulation,
            robot=self._robot,
            scene_registry=scene_registry,
            engine=engine,
        )
        if validator is None:
            raise RuntimeError(
                "The task registration does not declare a parallel safety factory."
            )
        return validator

    def create_observation_services(
        self,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        *,
        observations: ObservationManager,
    ) -> SimulationObservationServices:
        """Create declared physical collectors for this exact fresh host bridge.

        Args:
            environment: Initialized host with this factory's exact robot/simulation.
            bridge: Active actual bridge before its first execution step.
            observations: Actual environment-owned raw ObservationManager.

        Returns:
            Real state, support and declared control/contact/Cargo services. Only
            selected collectors acquire data or install hooks. Call ``close``
            before replacing the episode; host teardown also releases live owners.

        Raises:
            RuntimeError: The factory has closed.
            ValueError: The environment belongs to another factory.
        """
        from .observation_services import SimulationObservationServices

        self._owner_token.require_open()
        if (
            environment.sim is not self._simulation
            or environment.robot is not self._robot
        ):
            raise ValueError(
                "Observation services belong to another environment owner."
            )
        services = SimulationObservationServices(
            environment, bridge, self._registration, observations=observations
        )
        self._observation_services.append(services)

        def release() -> None:
            if services in self._observation_services:
                self._observation_services.remove(services)

        services._owner_release = release
        return services

    def create_adapter(self) -> TaskProgramEnvironmentAdapter:
        """Create the exact Gym adapter with shared simulation policy ports."""
        self._owner_token.require_open()
        try:
            return TaskProgramEnvironmentAdapter(
                self,
                step_dt=self._step_dt,
                joint_command_mode=self._joint_command_mode,
                registration=self._registration,
            )
        except Exception:
            self.close_runtime_resources()
            raise

    def _create_motion_generator(self) -> MotionGenerator:
        """Create and validate one exact motion generator."""
        if self._motion_generator_factory is not None:
            generator = self._motion_generator_factory()
        else:
            planner_cfg = (
                ToppraPlannerCfg(robot_uid=_robot_uid(self._robot))
                if self._planner_cfg is None
                else deepcopy(self._planner_cfg)
            )
            planner_cfg.sim_instance_id = self._simulation.instance_id
            if isinstance(planner_cfg, CuroboPlannerCfg):
                self._bind_curobo_scene(planner_cfg)
            generator = MotionGenerator(MotionGenCfg(planner_cfg=planner_cfg))
        if not isinstance(generator, MotionGenerator):
            raise TypeError(
                "motion_generator_factory must return a MotionGenerator instance."
            )
        return generator

    def _bind_curobo_scene(self, planner_cfg: CuroboPlannerCfg) -> None:
        """Attach live scene collision objects to a configured cuRobo planner."""
        if planner_cfg.world.rigid_objects is not None:
            return
        collision_bindings = tuple(
            binding
            for binding in self._scene_binding.rigid_objects
            if binding.collision_role is not SceneCollisionRole.NONE
        )
        unsupported_bindings = tuple(
            binding.entity_id
            for bindings in (
                self._scene_binding.rigidized_articulations,
                self._scene_binding.articulations,
                self._scene_binding.links,
            )
            for binding in bindings
            if binding.collision_role is not SceneCollisionRole.NONE
        )
        if unsupported_bindings:
            raise ValueError(
                "Configured cuRobo collision worlds support only rigid-object "
                "scene bindings; unsupported collision entities: "
                f"{sorted(unsupported_bindings)}."
            )

        objects: dict[str, object] = {}
        get_rigid_object = getattr(self._simulation, "get_rigid_object", None)
        if collision_bindings and not callable(get_rigid_object):
            raise TypeError("simulation must provide get_rigid_object().")
        for binding in collision_bindings:
            entity = get_rigid_object(binding.simulation_uid)
            if entity is None:
                raise KeyError(
                    f"Simulation UID {binding.simulation_uid!r} selected for "
                    f"collision entity {binding.entity_id!r} was not found."
                )
            objects[binding.entity_id] = entity

        world = planner_cfg.world
        dynamic_obstacle_names = [
            binding.entity_id
            for binding in collision_bindings
            if binding.collision_role is SceneCollisionRole.DYNAMIC
        ]
        world_values = {
            field.name: getattr(world, field.name) for field in fields(CuroboWorldCfg)
        }
        world_values["rigid_objects"] = objects or None
        world_values["dynamic_obstacle_names"] = dynamic_obstacle_names
        collision_mode = self._scene_binding.collision_world_mode
        if collision_mode is not None:
            world_values["multi_env"] = (
                collision_mode is SceneCollisionWorldMode.PER_ENV
            )
        planner_cfg.world = CuroboWorldCfg(**world_values)


@dataclass(frozen=True, slots=True, init=False)
class SimulationTaskProgramAdapterFactory:
    """Create one standard simulation adapter after an environment is initialized.

    Args:
        registration: Immutable provider-free task integration.
        grasp_pose_generator_factories: Zero-argument factories keyed by runtime
            grasp endpoint target ID. Each environment receives fresh services.
        hand_geometry_cfg_factories: Zero-argument factories for explicit hand
            calibration settings. No geometry is captured by these factories.
        planner_config: Optional execution-policy planner declaration. The live
            factory supplies the selected robot and simulation instance values.
    """

    _registration: SimulationTaskProgramRegistration
    _grasp_pose_generator_factories: Mapping[
        str,
        Callable[[], GraspPoseGenerator],
    ]
    _planner_config: Mapping[str, object] | None
    _hand_geometry_cfg_factories: Mapping[str, Callable[[], RobotHandGeometryCfg]]

    def __init__(
        self,
        registration: SimulationTaskProgramRegistration,
        *,
        grasp_pose_generator_factories: (
            Mapping[str, Callable[[], GraspPoseGenerator]] | None
        ) = None,
        hand_geometry_cfg_factories: (
            Mapping[str, Callable[[], RobotHandGeometryCfg]] | None
        ) = None,
        planner_config: Mapping[str, object] | None = None,
    ) -> None:
        if type(registration) is not SimulationTaskProgramRegistration:
            raise TypeError(
                "registration must be exactly SimulationTaskProgramRegistration."
            )
        if grasp_pose_generator_factories is not None and not isinstance(
            grasp_pose_generator_factories,
            Mapping,
        ):
            raise TypeError("grasp_pose_generator_factories must be a mapping or None.")
        if planner_config is not None and not isinstance(planner_config, Mapping):
            raise TypeError("planner_config must be a mapping or None.")
        factories: dict[str, Callable[[], GraspPoseGenerator]] = {}
        for target_id, factory in (grasp_pose_generator_factories or {}).items():
            if (
                type(target_id) is not str
                or not target_id
                or target_id != target_id.strip()
            ):
                raise ValueError(
                    "grasp_pose_generator_factories keys must be non-empty strings "
                    "without outer whitespace."
                )
            if not callable(factory):
                raise TypeError(
                    "grasp_pose_generator_factories values must be callable."
                )
            factories[target_id] = factory
        if hand_geometry_cfg_factories is not None and not isinstance(
            hand_geometry_cfg_factories, Mapping
        ):
            raise TypeError("hand_geometry_cfg_factories must be a mapping or None.")
        geometry_factories = {}
        for target_id, factory in (hand_geometry_cfg_factories or {}).items():
            if (
                type(target_id) is not str
                or not target_id
                or target_id != target_id.strip()
            ):
                raise ValueError(
                    "Hand geometry factory keys must be non-empty exact identifiers."
                )
            if not callable(factory):
                raise TypeError(
                    "Hand geometry configuration factories must be callable."
                )
            geometry_factories[target_id] = factory
        object.__setattr__(self, "_registration", registration)
        object.__setattr__(
            self, "_hand_geometry_cfg_factories", MappingProxyType(geometry_factories)
        )
        object.__setattr__(
            self,
            "_grasp_pose_generator_factories",
            MappingProxyType(factories),
        )
        object.__setattr__(
            self,
            "_planner_config",
            (
                None
                if planner_config is None
                else MappingProxyType(deepcopy(dict(planner_config)))
            ),
        )

    @property
    def registration(self) -> SimulationTaskProgramRegistration:
        """Return the exact static registration owned by this factory."""
        return self._registration

    def create_adapter(
        self,
        environment: object,
    ) -> TaskProgramEnvironmentAdapter:
        """Create fresh grasp services and bind the initialized environment."""
        from embodichain.toolkits.graspkit import GraspPoseGenerator

        self._registration.assert_unchanged()
        generators: dict[str, GraspPoseGenerator] = {}
        for target_id, factory in self._grasp_pose_generator_factories.items():
            generator = factory()
            if not isinstance(generator, GraspPoseGenerator):
                raise TypeError(
                    "A grasp-pose generator factory must return a "
                    "GraspPoseGenerator instance."
                )
            generators[target_id] = generator
        kwargs: dict[str, object] = {
            "registration": self._registration,
            "grasp_pose_generators": generators,
        }
        if self._hand_geometry_cfg_factories:
            from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

            geometry_cfgs = {}
            for target_id, factory in self._hand_geometry_cfg_factories.items():
                cfg = factory()
                if not isinstance(cfg, RobotHandGeometryCfg):
                    raise TypeError(
                        "A hand geometry factory must return RobotHandGeometryCfg."
                    )
                cfg = deepcopy(cfg)
                cfg.validate_values()
                geometry_cfgs[target_id] = cfg
            kwargs["hand_geometry_cfgs"] = geometry_cfgs
        if self._planner_config is not None:
            kwargs["planner_config"] = self._planner_config
        return create_simulation_task_program_adapter(environment, **kwargs)


def create_simulation_task_program_adapter(
    environment: SimulationTaskProgramEnvironment,
    *,
    registration: SimulationTaskProgramRegistration,
    planner_cfg: BasePlannerCfg | None = None,
    planner_config: Mapping[str, object] | None = None,
    motion_generator_factory: MotionGeneratorFactory | None = None,
    grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
    hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg] | None = None,
    translation_threshold: float = 1.0e-4,
    rotation_threshold: float = 1.0e-3,
) -> TaskProgramEnvironmentAdapter:
    """Create a complete production adapter from one standard Gym environment.

    This is the intended task-side one-line integration. Standard tasks pass
    one immutable ``registration`` that owns their pre-simulation catalog and
    every runtime extension.

    Args:
        environment: Standard Gym simulation environment exposing ``sim``,
            ``robot``, and ``step_dt``.
        registration: Required immutable task registration.
        planner_cfg: Optional planner configuration owned by the factory.
        planner_config: Optional executable-free planner declaration from an
            execution policy.
        motion_generator_factory: Optional factory for one fresh motion generator.
        grasp_pose_generators: Standalone grasp-pose services keyed by grasp
            endpoint target ID.
        hand_geometry_cfgs: Explicit hand calibrations keyed by grasp endpoint target ID.
        translation_threshold: Scene translation revision threshold.
        rotation_threshold: Scene rotation revision threshold.

    Returns:
        Complete production Task Program environment adapter.
    """
    factory = SimulationTaskProgramFactory.from_environment(
        environment,
        registration=registration,
        planner_cfg=planner_cfg,
        planner_config=planner_config,
        motion_generator_factory=motion_generator_factory,
        grasp_pose_generators=grasp_pose_generators,
        hand_geometry_cfgs=hand_geometry_cfgs,
        translation_threshold=translation_threshold,
        rotation_threshold=rotation_threshold,
    )
    return factory.create_adapter()


__all__: list[str] = []
