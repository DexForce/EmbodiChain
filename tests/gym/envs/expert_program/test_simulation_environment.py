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

"""Tests for reusable simulation-backed Expert Program assembly."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import inspect
import json
import textwrap
from types import MethodType, SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
import torch

from embodichain.agents.mllm import compile_mllm_expert_program
from embodichain.lab.gym.envs.expert_program import (
    AntipodalGraspAffordanceBinding,
    CompiledProgram,
    ControlCommandStateEvidenceTracker,
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    ExpertProgramCompiler,
    ExpertProgramCfg,
    ExpertProgramIntegrationCfg,
    ExpertProgramEnvironmentAdapter,
    ExpertProgramRuntimeAssembly,
    HandOverCfg,
    InvokeCfg,
    RobotResourceBinding,
    SharedTickSceneProvider,
    SimulationExpertProgramRegistration,
    SimulationExpertProgramFactory,
    SimulationPlanningObservationProvider,
    SimulationRigidObjectBinding,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    create_simulation_expert_program_adapter,
    decode_expert_program,
)
from embodichain.lab.gym.envs.expert_program.bridge import EnvironmentStepClock
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    Affordance,
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    CARTESIAN_POSE_CAPABILITY,
    CommandAcknowledgement,
    EntityState,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    HeldObjectState,
    MotionPolicy,
    ObservedArticulationJointState,
    PlanningContext,
    StateDelta,
    TaskState,
    TrackingPolicy,
)
from embodichain.lab.sim.atomic_actions.runner import ExecutionRunnerCfg
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.bindings import RuntimeEndpointTarget
from embodichain.lab.sim.atomic_actions.control import ControlPartCommandProfile
from embodichain.lab.sim.atomic_actions.runtime_commands import (
    EndpointCommand,
    JointPositionPayload,
    RuntimeCommandFrame,
)
from embodichain.lab.sim.planners import MotionGenerator
from embodichain.lab.sim.skills import (
    AtomicSkills,
    BoundSemanticCall,
    EndpointResolution,
    HandOver,
    HandOverPoseProvider,
    HandOverPoseTargets,
    Pick,
    Place,
    RelationTargetGrounder,
    ResourceEndpoint,
    ResourceEndpointAdapter,
    SceneArticulationRef,
    SceneEntityRegistration,
    SceneRegistry,
    SemanticCallSpec,
    SemanticObjectTarget,
    SemanticPose,
    SemanticRelationTarget,
    SkillPolicyPreset,
)
from embodichain.lab.sim.skills.effects import (
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    BinaryEffectClause,
    BinaryEvidenceKind,
    ControlPartEvidenceAddress,
    EffectEvidenceSourceRef,
    HeldObjectRelation,
    HeldObjectStateExpectation,
)
from embodichain.lab.sim.skills.evidence import (
    BinaryEffectEvidenceQuery,
    BinaryEffectEvidenceBatch,
    BinaryEffectObservation,
    EffectEvidenceCollectionContext,
    PoseRelationEvidenceBatch,
)
from embodichain.lab.sim.skills.runtime import (
    SkillEffectTrace,
    SkillResult,
    SkillRuntime,
    SkillStatus,
)
from embodichain.lab.sim.skills.scene import SceneObjectRef

_BATCH_SIZE = 3
_ROBOT_DOF = 2
_STEP_DT = 0.04
_TRACKER_ENV_IDS = torch.tensor((7, 3, 11), dtype=torch.long)
_HAND_OPEN_POSITION = 0.0
_HAND_GRASP_POSITION = 0.8
_HAND_INTERMEDIATE_POSITION = 0.4
_DUAL_ROBOT_DOF = 4
_RELEASE_SEPARATION = 0.2
_DIRECT_PLACE_TARGET = SemanticPose(
    position=(0.0, 0.0, 0.0),
    quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
)
_QUICKSTART_MAX_LINES = 15


def _command_state_tracker() -> ControlCommandStateEvidenceTracker:
    """Build a three-row tracker for one semantic gripper profile."""
    profile = ControlPartCommandProfile.joint_positions(
        open=torch.tensor((_HAND_OPEN_POSITION,)),
        grasp=torch.tensor((_HAND_GRASP_POSITION,)),
    )
    return ControlCommandStateEvidenceTracker(
        {"hand": profile},
        _TRACKER_ENV_IDS,
    )


def _hand_command_frame(
    *,
    env_ids: tuple[int, ...],
    positions: tuple[float, ...],
    active: tuple[bool, ...] | None = None,
) -> RuntimeCommandFrame:
    """Build one row-addressed semantic hand command frame."""
    batch_size = len(env_ids)
    if len(positions) != batch_size:
        raise ValueError("positions must have one value per environment ID.")
    if active is None:
        active = (True,) * batch_size
    return RuntimeCommandFrame(
        commands=(
            EndpointCommand(
                target=JointPositionTarget("hand", (1,)),
                payload=JointPositionPayload(
                    torch.tensor(positions, dtype=torch.float32).unsqueeze(1)
                ),
            ),
        ),
        active_mask=torch.tensor(active, dtype=torch.bool),
        env_ids=torch.tensor(env_ids, dtype=torch.long),
        hold_duration=torch.full((batch_size,), _STEP_DT),
    )


def _hand_state_observation(
    tracker: ControlCommandStateEvidenceTracker,
    *env_ids: int,
) -> BinaryEffectObservation:
    """Observe command-state evidence in an explicit stable-ID order."""
    expectation = HeldObjectStateExpectation(
        expectation_id="held-cube",
        relation=HeldObjectRelation.ATTACHED,
        object_id="cube",
        slot_id="primary",
        resource_id="manipulator",
        task_state_key="held-cube",
    )
    query = BinaryEffectEvidenceQuery(
        BinaryEffectClause(
            clause_id="hand-constraint",
            expectation_id=expectation.expectation_id,
            source=EffectEvidenceSourceRef(
                CONTROL_PART_EVIDENCE_PROVIDER_ID,
                CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
                ControlPartEvidenceAddress("hand", CONSTRAINT_EFFECT_CHANNEL),
            ),
            evidence_kind=BinaryEvidenceKind.CONSTRAINT,
            expected=True,
        ),
        expectation,
    )
    context = EffectEvidenceCollectionContext(
        timestamp=0.0,
        observation_revision=0,
        env_ids=torch.tensor(env_ids, dtype=torch.long),
    )
    return tracker.observe(query, context)


class _CountingEntityProvider:
    """Return row-addressed poses and record every native acquisition."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, torch.Tensor]] = []

    def observe(self, *, timestamp: float, env_ids: torch.Tensor) -> EntityState:
        """Return one distinct x translation for each environment ID."""
        self.calls.append((timestamp, env_ids.clone()))
        pose = torch.eye(4).repeat(env_ids.numel(), 1, 1)
        pose[:, 0, 3] = env_ids.to(dtype=pose.dtype)
        return EntityState(pose)


class _CountingJointProvider:
    """Return row-addressed articulation state and record acquisitions."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, torch.Tensor]] = []

    def observe_joints(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> dict[str, ObservedArticulationJointState]:
        """Return one scalar joint position per environment ID."""
        self.calls.append((timestamp, env_ids.clone()))
        position = env_ids.to(dtype=torch.float32).unsqueeze(1)
        return {
            "slide": ObservedArticulationJointState(
                position,
                torch.ones(env_ids.numel(), dtype=torch.bool),
            )
        }


def _shared_scene_provider() -> tuple[
    SharedTickSceneProvider,
    _CountingEntityProvider,
    _CountingJointProvider,
]:
    """Build one full-batch registry provider with observable acquisitions."""
    entity_provider = _CountingEntityProvider()
    joint_provider = _CountingJointProvider()
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneArticulationRef("drawer"),
                state_provider=entity_provider,
                joint_state_provider=joint_provider,
            ),
        )
    )
    delegate = registry.make_scene_provider(batch_size=_BATCH_SIZE)
    full_env_ids = torch.arange(_BATCH_SIZE, dtype=torch.long)
    return (
        SharedTickSceneProvider(delegate, full_env_ids),
        entity_provider,
        joint_provider,
    )


def test_shared_tick_scene_provider_projects_partial_rows_without_resampling() -> None:
    """Planning full batch and evidence subsets share one native acquisition."""
    provider, entity_provider, joint_provider = _shared_scene_provider()
    full_env_ids = torch.arange(_BATCH_SIZE, dtype=torch.long)

    full = provider.snapshot(timestamp=0.0, env_ids=full_env_ids)
    subset = provider.snapshot(
        timestamp=0.0,
        env_ids=torch.tensor((2, 0), dtype=torch.long),
    )

    assert len(entity_provider.calls) == 1
    assert len(joint_provider.calls) == 1
    assert torch.equal(entity_provider.calls[0][1], full_env_ids)
    assert full.entities["drawer"].pose[:, 0, 3].tolist() == [0.0, 1.0, 2.0]
    assert subset.entities["drawer"].pose[:, 0, 3].tolist() == [2.0, 0.0]
    joint = subset.articulation_joints[("drawer", "slide")]
    assert joint.position[:, 0].tolist() == [2.0, 0.0]
    assert joint.valid_mask is not None and joint.valid_mask.tolist() == [True, True]
    assert subset.collision_world_revision == (0, 0)


def test_shared_tick_scene_provider_captures_full_batch_when_subset_arrives_first() -> (
    None
):
    """A partial first consumer cannot poison the delegate's stable batch."""
    provider, entity_provider, joint_provider = _shared_scene_provider()
    requested = torch.tensor((1,), dtype=torch.long)

    first = provider.snapshot(timestamp=0.0, env_ids=requested)
    second = provider.snapshot(
        timestamp=0.0,
        env_ids=torch.tensor((2, 1), dtype=torch.long),
    )

    expected_full = torch.arange(_BATCH_SIZE, dtype=torch.long)
    assert torch.equal(entity_provider.calls[0][1], expected_full)
    assert torch.equal(joint_provider.calls[0][1], expected_full)
    assert len(entity_provider.calls) == 1
    assert first.entities["drawer"].pose[:, 0, 3].tolist() == [1.0]
    assert second.entities["drawer"].pose[:, 0, 3].tolist() == [2.0, 1.0]


def test_shared_tick_scene_provider_rejects_unknown_or_regressing_rows() -> None:
    """Unknown correlations and time regressions fail before native sampling."""
    provider, entity_provider, _ = _shared_scene_provider()
    provider.snapshot(
        timestamp=0.5,
        env_ids=torch.tensor((0, 2), dtype=torch.long),
    )

    with pytest.raises(ValueError, match="absent from full_env_ids"):
        provider.snapshot(
            timestamp=0.5,
            env_ids=torch.tensor((3,), dtype=torch.long),
        )
    with pytest.raises(ValueError, match="monotonic"):
        provider.snapshot(
            timestamp=0.4,
            env_ids=torch.tensor((0,), dtype=torch.long),
        )

    assert len(entity_provider.calls) == 1


def test_command_state_tracker_correlates_open_and_grasp_across_subsets() -> None:
    """Stable IDs, not subset row positions, own accepted gripper state."""
    tracker = _command_state_tracker()

    tracker.accepted(
        _hand_command_frame(
            env_ids=(3, 7),
            positions=(_HAND_GRASP_POSITION, _HAND_OPEN_POSITION),
        )
    )
    observation = _hand_state_observation(tracker, 7, 11, 3)

    assert tracker.tracked_control_parts == ("hand",)
    assert observation.values.tolist() == [False, False, True]
    assert observation.valid is not None
    assert observation.valid.tolist() == [True, False, True]
    assert observation.acquisition_errors[0] is None
    assert observation.acquisition_errors[1] is not None
    assert observation.acquisition_errors[2] is None


def test_command_state_tracker_preserves_intermediate_and_inactive_rows() -> None:
    """Unrecognized targets and inactive rows cannot overwrite prior evidence."""
    tracker = _command_state_tracker()
    tracker.accepted(
        _hand_command_frame(
            env_ids=(7, 3),
            positions=(_HAND_OPEN_POSITION, _HAND_GRASP_POSITION),
        )
    )

    tracker.accepted(
        _hand_command_frame(
            env_ids=(3, 7),
            positions=(
                _HAND_INTERMEDIATE_POSITION,
                _HAND_INTERMEDIATE_POSITION,
            ),
        )
    )
    after_intermediate = _hand_state_observation(tracker, 3, 7)
    assert after_intermediate.values.tolist() == [True, False]
    assert after_intermediate.valid is not None
    assert after_intermediate.valid.tolist() == [True, True]

    tracker.accepted(
        _hand_command_frame(
            env_ids=(3, 7),
            positions=(_HAND_OPEN_POSITION, _HAND_GRASP_POSITION),
            active=(False, True),
        )
    )
    after_inactive_row = _hand_state_observation(tracker, 3, 7)
    assert after_inactive_row.values.tolist() == [True, True]
    assert after_inactive_row.valid is not None
    assert after_inactive_row.valid.tolist() == [True, True]


def test_command_state_tracker_cancel_invalidates_target_state() -> None:
    """Cancelling a hand destination invalidates every correlated hand row."""
    tracker = _command_state_tracker()
    frame = _hand_command_frame(
        env_ids=(7, 3),
        positions=(_HAND_OPEN_POSITION, _HAND_GRASP_POSITION),
    )
    tracker.accepted(frame)

    tracker.cancelled(frame.targets)
    observation = _hand_state_observation(tracker, 3, 7)

    assert observation.values.tolist() == [False, False]
    assert observation.valid is not None
    assert observation.valid.tolist() == [False, False]
    assert all(error is not None for error in observation.acquisition_errors)


def test_command_state_tracker_discard_invalidates_all_state() -> None:
    """A fail-closed sink discard removes every accepted row state."""
    tracker = _command_state_tracker()
    tracker.accepted(
        _hand_command_frame(
            env_ids=(11, 3),
            positions=(_HAND_GRASP_POSITION, _HAND_OPEN_POSITION),
        )
    )

    tracker.discarded()
    observation = _hand_state_observation(tracker, 11, 3)

    assert observation.values.tolist() == [False, False]
    assert observation.valid is not None
    assert observation.valid.tolist() == [False, False]


def test_command_state_tracker_rejects_unknown_environment_ids() -> None:
    """Unknown correlation IDs fail before tracker state can be mutated or read."""
    tracker = _command_state_tracker()

    with pytest.raises(ValueError, match="absent from tracker env_ids"):
        tracker.accepted(
            _hand_command_frame(
                env_ids=(99,),
                positions=(_HAND_GRASP_POSITION,),
            )
        )
    with pytest.raises(ValueError, match="absent from tracker env_ids"):
        _hand_state_observation(tracker, 99)

    observation = _hand_state_observation(tracker, 7, 3, 11)
    assert observation.valid is not None
    assert observation.valid.tolist() == [False, False, False]


class _Robot:
    """Minimal typed robot surface used by the production factory."""

    uid = "robot"
    device = torch.device("cpu")
    dof = _ROBOT_DOF
    control_parts = {"arm": ("joint_0",)}

    def __init__(self) -> None:
        self.qpos = torch.zeros(_BATCH_SIZE, _ROBOT_DOF)

    def get_qpos(
        self,
        name: str | None = None,
        target: bool = False,
    ) -> torch.Tensor:
        """Return full or control-part positions."""
        del target
        return self.qpos if name is None else self.qpos[:, :1]

    def get_qvel(
        self,
        name: str | None = None,
        target: bool = False,
    ) -> torch.Tensor:
        """Return zero measured velocities."""
        return torch.zeros_like(self.get_qpos(name=name, target=target))

    def get_qf(self, name: str | None = None) -> torch.Tensor:
        """Return zero measured effort."""
        return torch.zeros_like(self.get_qpos(name=name))

    def get_joint_ids(self, name: str) -> list[int]:
        """Resolve the only declared control part."""
        if name != "arm":
            raise KeyError(name)
        return [0]

    def get_solver(self, name: str) -> object:
        """Return a configured solver marker for Cartesian capability."""
        if name != "arm":
            raise KeyError(name)
        return object()

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str | None = None,
        env_ids: list[int] | None = None,
        to_matrix: bool = False,
    ) -> torch.Tensor:
        """Return identity endpoint poses for evidence adapter validation."""
        del name, env_ids
        if not to_matrix:
            raise ValueError("Tests require matrix FK output.")
        return torch.eye(4).repeat(qpos.shape[0], 1, 1)


class _EvidenceRobot(_Robot):
    """Joint-backed arm and hand with a mutable measured endpoint pose."""

    control_parts = {
        "arm": ("joint_0",),
        "hand": ("joint_1",),
    }

    def __init__(self) -> None:
        super().__init__()
        self.endpoint_pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)

    def get_qpos(
        self,
        name: str | None = None,
        target: bool = False,
    ) -> torch.Tensor:
        """Return the full state or the selected control-part state."""
        del target
        if name is None:
            return self.qpos
        joint_id = self.get_joint_ids(name)[0]
        return self.qpos[:, joint_id : joint_id + 1]

    def get_joint_ids(self, name: str) -> list[int]:
        """Resolve the disjoint arm and hand joints."""
        if name == "arm":
            return [0]
        if name == "hand":
            return [1]
        raise KeyError(name)

    def get_solver(self, name: str) -> object:
        """Return the configured arm solver marker."""
        if name != "arm":
            raise KeyError(name)
        return object()

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str | None = None,
        env_ids: list[int] | None = None,
        to_matrix: bool = False,
    ) -> torch.Tensor:
        """Return the live arm endpoint pose for requested simulator rows."""
        del qpos
        if name != "arm" or not to_matrix:
            raise ValueError("Evidence FK requires the arm matrix pose.")
        rows = list(range(_BATCH_SIZE)) if env_ids is None else env_ids
        return self.endpoint_pose[rows].clone()


class _DualRobot(_Robot):
    """Four-part dual-arm robot used for provider-aware helper preflight."""

    uid = "dual_robot"
    dof = _DUAL_ROBOT_DOF
    control_parts = {
        "left_arm": ("left_arm_joint",),
        "left_hand": ("left_hand_joint",),
        "right_arm": ("right_arm_joint",),
        "right_hand": ("right_hand_joint",),
    }
    _joint_ids = {
        "left_arm": (0,),
        "left_hand": (1,),
        "right_arm": (2,),
        "right_hand": (3,),
    }

    def __init__(self) -> None:
        self.qpos = torch.zeros(_BATCH_SIZE, self.dof)

    def get_qpos(
        self,
        name: str | None = None,
        target: bool = False,
    ) -> torch.Tensor:
        """Return full state or the selected one-joint control part."""
        del target
        if name is None:
            return self.qpos
        return self.qpos[:, list(self._joint_ids[name])]

    def get_joint_ids(self, name: str) -> list[int]:
        """Resolve one disjoint arm or hand joint."""
        return list(self._joint_ids[name])

    def get_solver(self, name: str) -> object:
        """Return configured solver markers for both motion endpoints."""
        if name not in {"left_arm", "right_arm"}:
            raise KeyError(name)
        return object()

    def compute_fk(
        self,
        qpos: torch.Tensor,
        name: str | None = None,
        env_ids: list[int] | None = None,
        to_matrix: bool = False,
    ) -> torch.Tensor:
        """Return identity arm endpoint poses for runtime assembly checks."""
        del env_ids
        if name not in {"left_arm", "right_arm"} or not to_matrix:
            raise ValueError("Dual-arm evidence requires an arm matrix pose.")
        return torch.eye(4).repeat(qpos.shape[0], 1, 1)


class _RigidObject:
    """Mutable batched rigid object with the mesh surface required by binding."""

    def __init__(self) -> None:
        self.pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)

    def get_local_pose(self, *, to_matrix: bool = False) -> torch.Tensor:
        """Return the current measured object pose."""
        if not to_matrix:
            raise ValueError("Tests require matrix object poses.")
        return self.pose.clone()

    def get_vertices(
        self,
        *,
        env_ids: list[int],
        scale: bool = True,
    ) -> torch.Tensor:
        """Return one minimal triangular mesh per requested row."""
        del scale
        vertices = torch.tensor(((0.0, 0.0, 0.0), (0.04, 0.0, 0.0), (0.0, 0.04, 0.0)))
        return vertices.unsqueeze(0).repeat(len(env_ids), 1, 1)

    def get_triangles(self, *, env_ids: list[int]) -> torch.Tensor:
        """Return one valid triangle per requested row."""
        return (
            torch.tensor(((0, 1, 2),), dtype=torch.long)
            .unsqueeze(0)
            .repeat(len(env_ids), 1, 1)
        )


class _ForwardedRelationGrounder(RelationTargetGrounder):
    """Sentinel relation grounder installed only to prove helper forwarding."""

    capability: ClassVar[str] = "test.place_relation"
    affordance_type: ClassVar[type[Affordance]] = Affordance
    affordance_revision: ClassVar[str] = "test-v1"

    def ground(
        self,
        relation: SemanticRelationTarget,
        *,
        affordance: Affordance,
        context: PlanningContext,
    ) -> torch.Tensor:
        """Return a direct identity target when explicitly exercised."""
        del relation, affordance, context
        return torch.eye(4)


class _ForwardedHandOverPoseProvider(HandOverPoseProvider):
    """Sentinel embodiment provider installed only through the standard helper."""

    provider_id: ClassVar[str] = "test.handover_pose"

    def resolve(
        self,
        call: HandOver,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
    ) -> HandOverPoseTargets:
        """Return owned direct targets without embedding task-side motion code."""
        del call, context, bound
        pose = SemanticPose(
            position=(0.0, 0.0, 0.5),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        return HandOverPoseTargets(
            middle=SemanticObjectTarget(pose=pose),
            final=SemanticObjectTarget(pose=pose),
        )


@dataclass(frozen=True, slots=True)
class _MobileEndpoint(ResourceEndpoint):
    """Non-joint endpoint used by the standard simulation factory test."""

    controller_id: str


@dataclass(frozen=True, slots=True)
class _MobileTarget(RuntimeEndpointTarget):
    """Runtime destination for the test mobile controller."""

    controller_id: str

    @property
    def transport_id(self) -> str:
        """Return the matching test Gym transport ID."""
        return "test.mobile_velocity"

    @property
    def target_id(self) -> str:
        """Return the selected controller ID."""
        return self.controller_id


class _MobileEndpointAdapter(ResourceEndpointAdapter):
    """Resolve a mobile endpoint without consulting robot control parts."""

    adapter_id: ClassVar[str] = "test.mobile_velocity"
    endpoint_type: ClassVar[type[ResourceEndpoint]] = _MobileEndpoint

    def resolve(
        self,
        endpoint: ResourceEndpoint,
        *,
        engine: Any,
    ) -> EndpointResolution:
        """Resolve one exclusive controller claim."""
        del engine
        if not isinstance(endpoint, _MobileEndpoint):
            raise TypeError("_MobileEndpointAdapter requires _MobileEndpoint.")
        return EndpointResolution(
            runtime_target=_MobileTarget(endpoint.controller_id),
            claim_tokens=frozenset({f"controller:{endpoint.controller_id}"}),
        )


class _MobileTransportEncoder:
    """Minimal Gym encoder registered for the custom mobile target."""

    @property
    def transport_id(self) -> str:
        """Return the custom mobile transport ID."""
        return "test.mobile_velocity"

    def encode(
        self,
        command: EndpointCommand,
        *,
        base_action: Any,
        active_mask: torch.Tensor,
    ) -> Any:
        """Preserve the base action in this assembly-only test transport."""
        del command, active_mask
        return base_action.clone()

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        base_action: Any,
        context: Any,
    ) -> Any:
        """Preserve the base action for a mobile safe hold."""
        del targets, context
        return base_action.clone()


class _MobileRobot:
    """Full-state robot fixture with no control-parts or joint-ID surface."""

    uid = "mobile_robot"
    device = torch.device("cpu")
    dof = _ROBOT_DOF

    def __init__(self) -> None:
        self.qpos = torch.zeros(_BATCH_SIZE, _ROBOT_DOF)

    def get_qpos(self) -> torch.Tensor:
        """Return the full controller hold state."""
        return self.qpos

    def get_qvel(self) -> torch.Tensor:
        """Return the full measured velocity state."""
        return torch.zeros_like(self.qpos)

    def get_qf(self) -> torch.Tensor:
        """Return the full measured effort state."""
        return torch.zeros_like(self.qpos)


class _Simulation:
    """Minimal simulation registry for one exact robot."""

    def __init__(
        self,
        robot: _Robot,
        rigid_objects: dict[str, _RigidObject] | None = None,
    ) -> None:
        self.robot = robot
        self.rigid_objects = {} if rigid_objects is None else dict(rigid_objects)

    def get_robot(self, uid: str) -> _Robot | None:
        """Resolve the selected robot UID."""
        return self.robot if uid == self.robot.uid else None

    def get_rigid_object(self, uid: str) -> _RigidObject | None:
        """Resolve one explicitly registered rigid-object UID."""
        return self.rigid_objects.get(uid)


def _profile_binding() -> SimulationRobotSkillProfileBinding:
    """Build one motion-only profile with an intentionally wrong cadence."""
    return SimulationRobotSkillProfileBinding(
        profile_id="robot_profile",
        resources=(
            ControlPartResourceBinding(
                resource_id="manipulator",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="arm",
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                    ),
                ),
            ),
        ),
        presets=(
            SkillPolicyPreset(
                "safe",
                motion_policy=MotionPolicy(control_dt=0.01),
                tracking_policy=TrackingPolicy.joint_position(
                    in_flight_max_abs_error=0.037,
                    terminal_max_abs_error=0.019,
                ),
            ),
        ),
        default_preset="safe",
    )


def _handover_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Declare two disjoint manipulators and one selected pose provider ID."""
    motion_capabilities = frozenset(
        {
            CARTESIAN_POSE_CAPABILITY,
            FORWARD_KINEMATICS_CAPABILITY,
        }
    )
    resources = tuple(
        ControlPartResourceBinding(
            resource_id=side,
            endpoints=(
                ControlPartEndpointBinding(
                    endpoint_id="motion",
                    control_part=f"{side}_arm",
                    capabilities=motion_capabilities,
                ),
                ControlPartEndpointBinding(
                    endpoint_id="grasp",
                    control_part=f"{side}_hand",
                    capabilities=frozenset({GRASP_CAPABILITY}),
                    command_preset=f"{side}_hand_commands",
                ),
            ),
        )
        for side in ("left", "right")
    )
    command_presets = tuple(
        ControlPartCommandPreset(
            preset_id=f"{side}_hand_commands",
            control_part=f"{side}_hand",
            commands={
                "open": (_HAND_OPEN_POSITION,),
                "grasp": (_HAND_GRASP_POSITION,),
            },
        )
        for side in ("left", "right")
    )
    return SimulationRobotSkillProfileBinding(
        profile_id="handover_profile",
        resources=resources,
        command_presets=command_presets,
        defaults={
            "hand_over": {"source": "left", "destination": "right"},
        },
        presets=(SkillPolicyPreset("safe"),),
        default_preset="safe",
        grounding_providers={
            "hand_over": _ForwardedHandOverPoseProvider.provider_id,
        },
    )


def _handover_helper_inputs() -> tuple[
    SimpleNamespace,
    SimulationSceneBinding,
    SimulationRobotSkillProfileBinding,
]:
    """Build standard-helper inputs for one provider-aware HandOver program."""
    robot = _DualRobot()
    cube = _RigidObject()
    simulation = _Simulation(robot, {"cube_native": cube})
    environment = SimpleNamespace(
        sim=simulation,
        robot=robot,
        step_dt=_STEP_DT,
    )
    scene_binding = SimulationSceneBinding(
        registry_id="handover_scene",
        rigid_objects=(
            SimulationRigidObjectBinding(
                entity_id="cube",
                simulation_uid="cube_native",
                default_grasp_affordance="cube_grasp",
            ),
        ),
        antipodal_grasps=(
            AntipodalGraspAffordanceBinding(
                entity_id="cube_grasp",
                object_id="cube",
                native_name="body",
                revision="1",
            ),
        ),
    )
    return environment, scene_binding, _handover_profile_binding()


def _handover_program() -> ExpertProgramCfg:
    """Build one external-held-state HandOver call for static preflight."""
    return ExpertProgramCfg(
        schema_version=1,
        program_id="handover_preflight",
        integration=ExpertProgramIntegrationCfg(
            robot_profile="handover_profile",
            scene_registry="handover_scene",
            runtime_preset="safe",
        ),
        program=InvokeCfg(call=HandOverCfg(object="cube")),
    )


def _motion_generator(robot: _Robot) -> MotionGenerator:
    """Build a type-checkable motion-generator test double."""
    generator = MagicMock(spec=MotionGenerator)
    generator.robot = robot
    generator.device = robot.device
    generator.planner = SimpleNamespace(cfg=SimpleNamespace(planner_type="test"))
    generator.dynamic_collision_entity_ids = ()
    generator.collision_world_entity_ids = ()
    generator.supports_dynamic_collision_world = False
    generator.collision_world_batch_mode = None
    return generator


def _factory() -> tuple[SimulationExpertProgramFactory, _Robot]:
    """Create one production factory around CPU-only test doubles."""
    robot = _Robot()
    simulation = _Simulation(robot)
    return (
        SimulationExpertProgramFactory(
            simulation,  # type: ignore[arg-type]
            robot,  # type: ignore[arg-type]
            SimulationExpertProgramRegistration(
                scene_binding=SimulationSceneBinding(registry_id="scene"),
                robot_profile_binding=_profile_binding(),
            ),
            step_dt=_STEP_DT,
            motion_generator_factory=lambda: _motion_generator(robot),
        ),
        robot,
    )


def _evidence_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Declare one manipulation resource with exact open/grasp semantics."""
    motion_capabilities = frozenset(
        {
            BATCH_INVERSE_KINEMATICS_CAPABILITY,
            CARTESIAN_POSE_CAPABILITY,
            FORWARD_KINEMATICS_CAPABILITY,
        }
    )
    return SimulationRobotSkillProfileBinding(
        profile_id="evidence_profile",
        resources=(
            ControlPartResourceBinding(
                resource_id="manipulator",
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part="arm",
                        capabilities=motion_capabilities,
                    ),
                    ControlPartEndpointBinding(
                        endpoint_id="grasp",
                        control_part="hand",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        command_preset="hand_commands",
                    ),
                ),
            ),
        ),
        command_presets=(
            ControlPartCommandPreset(
                preset_id="hand_commands",
                control_part="hand",
                commands={
                    "open": (_HAND_OPEN_POSITION,),
                    "grasp": (_HAND_GRASP_POSITION,),
                },
            ),
        ),
        defaults={
            "pick_up": {"primary": "manipulator"},
            "place": {"primary": "manipulator"},
        },
        presets=(SkillPolicyPreset("evidence"),),
        default_preset="evidence",
    )


def _pick_evidence_plan(action: Any, request: Any, context: Any) -> Any:
    """Build one grasp frame and an identity object-to-endpoint expectation."""
    goal = action.require_goal(request)
    trajectory = context.robot.qpos.unsqueeze(1).clone()
    trajectory[:, 0, 1] = _HAND_GRASP_POSITION
    relation = torch.eye(4).repeat(context.batch_size, 1, 1)
    held = HeldObjectState(
        semantics=goal.semantics,
        object_to_eef=relation,
        grasp_xpos=relation,
    )
    return action.build_plan(
        request,
        context,
        success=torch.ones(context.batch_size, dtype=torch.bool),
        trajectory=trajectory,
        expected_effects=StateDelta(
            held_object_updates={"manipulator": held},
        ),
        replannable=False,
        scene_dependency_monitor_until={"cube": 0},
    )


def _place_evidence_plan(action: Any, request: Any, context: Any) -> Any:
    """Build one open frame and the matching held-object removal delta."""
    trajectory = context.robot.qpos.unsqueeze(1).clone()
    trajectory[:, 0, 1] = _HAND_OPEN_POSITION
    return action.build_plan(
        request,
        context,
        success=torch.ones(context.batch_size, dtype=torch.bool),
        trajectory=trajectory,
        expected_effects=StateDelta(
            held_object_updates={"manipulator": None},
        ),
        replannable=False,
    )


def _evidence_integration() -> ExpertProgramIntegrationCfg:
    """Return the host-owned integration shared by all frontend paths."""
    return ExpertProgramIntegrationCfg(
        robot_profile="evidence_profile",
        scene_registry="evidence_scene",
        runtime_preset="evidence",
    )


def _evidence_adapter_runtime() -> tuple[
    ExpertProgramEnvironmentAdapter,
    ExpertProgramRuntimeAssembly,
    _EvidenceRobot,
    _RigidObject,
]:
    """Assemble the production adapter and Pick/Place evidence chain."""
    robot = _EvidenceRobot()
    cube = _RigidObject()
    simulation = _Simulation(robot, {"cube_native": cube})
    scene_binding = SimulationSceneBinding(
        registry_id="evidence_scene",
        rigid_objects=(
            SimulationRigidObjectBinding(
                entity_id="cube",
                simulation_uid="cube_native",
                default_grasp_affordance="cube_grasp",
            ),
        ),
        antipodal_grasps=(
            AntipodalGraspAffordanceBinding(
                entity_id="cube_grasp",
                object_id="cube",
                native_name="body",
                revision="1",
            ),
        ),
    )
    factory = SimulationExpertProgramFactory(
        simulation,  # type: ignore[arg-type]
        robot,  # type: ignore[arg-type]
        SimulationExpertProgramRegistration(
            scene_binding=scene_binding,
            robot_profile_binding=_evidence_profile_binding(),
        ),
        step_dt=_STEP_DT,
        motion_generator_factory=lambda: _motion_generator(robot),
    )
    adapter = factory.create_adapter(
        runner_cfg=ExecutionRunnerCfg(
            minimum_cycle_time=0.0,
            hold_on_completion=False,
        )
    )
    assembly = adapter.assemble_runtime(_evidence_integration())
    pick_action = assembly.engine.actions["pick_up"]
    place_action = assembly.engine.actions["place"]
    pick_action._plan = MethodType(_pick_evidence_plan, pick_action)
    place_action._plan = MethodType(_place_evidence_plan, place_action)
    return adapter, assembly, robot, cube


def _evidence_runtime() -> tuple[
    ExpertProgramRuntimeAssembly,
    _EvidenceRobot,
    _RigidObject,
]:
    """Assemble the production Pick/Place evidence chain on CPU fixtures."""
    _, assembly, robot, cube = _evidence_adapter_runtime()
    return assembly, robot, cube


def _consume_buffered_action(
    assembly: ExpertProgramRuntimeAssembly,
    robot: _EvidenceRobot,
) -> None:
    """Apply one accepted Gym action and advance the authoritative clock."""
    processed = assembly.command_sink.pop()
    if not isinstance(processed.value, torch.Tensor):
        raise TypeError("Joint-backed evidence actions must be tensors.")
    robot.qpos = processed.value.clone()
    assembly.clock.advance_after_env_step()


def _accept_hand_command(
    assembly: ExpertProgramRuntimeAssembly,
    robot: _EvidenceRobot,
    position: float,
) -> None:
    """Accept and consume one semantic hand command through the Gym sink."""
    assert assembly.command_sink.pending_count == 0
    frame = _hand_command_frame(
        env_ids=tuple(range(_BATCH_SIZE)),
        positions=(position,) * _BATCH_SIZE,
    )
    acknowledgement = assembly.command_sink.send(frame, timeout=1.0)
    assert acknowledgement.accepted
    _consume_buffered_action(assembly, robot)


def _sample_effect(
    assembly: ExpertProgramRuntimeAssembly,
    robot: _EvidenceRobot,
    *,
    expected_trace_count: int,
    advance_clock: bool = True,
) -> tuple[Any, SkillEffectTrace]:
    """Advance one fresh environment tick and return its production trace."""
    if advance_clock:
        assembly.clock.advance_after_env_step()
    result = assembly.runtime.step()
    assert len(result.effects) == expected_trace_count
    while assembly.command_sink.pending_count:
        _consume_buffered_action(assembly, robot)
    return result, result.effects[-1]


class _SynchronousEvidenceClock:
    """Advance the fixture's simulation clock during standalone facade waits."""

    def __init__(self, clock: EnvironmentStepClock) -> None:
        self._clock = clock

    def now(self) -> float:
        """Return the simulation fixture's authoritative time."""
        return self._clock.now()

    def sleep(self, duration: float) -> None:
        """Advance the exact number of fixture ticks requested by the runner."""
        steps = self._clock.steps_for_duration(duration)
        if steps:
            self._clock.advance_after_env_step(steps)


class _ImmediateEvidenceCommandSink:
    """Apply accepted endpoint frames immediately for standalone CPU execution."""

    def __init__(
        self,
        assembly: ExpertProgramRuntimeAssembly,
        robot: _EvidenceRobot,
        cube: _RigidObject,
    ) -> None:
        self._encoder = assembly.command_encoder
        self._observer = assembly.accepted_command_observer
        self._clock = assembly.clock
        self._robot = robot
        self._cube = cube

    def send(
        self,
        command: RuntimeCommandFrame,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Apply one command and publish its accepted semantic hand state."""
        assert timeout > 0.0
        action = self._encoder.encode(command)
        if not isinstance(action, torch.Tensor):
            raise TypeError("The CPU quickstart fixture requires tensor actions.")
        self._robot.qpos = action.clone()
        if self._observer is None:
            raise RuntimeError("The evidence fixture requires an accepted observer.")
        self._observer.accepted(command.snapshot())
        if torch.allclose(
            action[:, 1],
            torch.full_like(action[:, 1], _HAND_OPEN_POSITION),
        ):
            self._cube.pose[:, 0, 3] = _RELEASE_SEPARATION
        self._clock.advance_after_env_step()
        return CommandAcknowledgement.accepted_ack()

    def hold(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        context: PlanningContext,
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Apply the encoder's observed-position hold immediately."""
        assert timeout > 0.0
        action = self._encoder.encode_hold(targets, context)
        if not isinstance(action, torch.Tensor):
            raise TypeError("The CPU quickstart fixture requires tensor actions.")
        self._robot.qpos = action.clone()
        return CommandAcknowledgement.accepted_ack()

    def cancel(
        self,
        targets: tuple[RuntimeEndpointTarget, ...],
        *,
        timeout: float,
    ) -> CommandAcknowledgement:
        """Clear accepted command evidence for cancelled destinations."""
        assert timeout > 0.0
        if self._observer is not None:
            self._observer.cancelled(targets)
        return CommandAcknowledgement.accepted_ack()


class _QuickstartRuntimeProvider:
    """Explicit provider used by the public ``AtomicSkills.from_env`` path."""

    def __init__(self, runtime: SkillRuntime) -> None:
        self._runtime = runtime
        self.presets: list[str] = []

    def create_skill_runtime(self, *, preset: str) -> SkillRuntime:
        """Return the configured canonical runtime and record preset selection."""
        self.presets.append(preset)
        return self._runtime


def _quickstart_runtime_provider() -> _QuickstartRuntimeProvider:
    """Build a synchronous provider from the shared production CPU fixture."""
    assembly, robot, cube = _evidence_runtime()
    runtime = SkillRuntime.from_components(
        assembly.compiler,
        assembly.observation_provider,
        _ImmediateEvidenceCommandSink(assembly, robot, cube),
        assembly.evidence_collector,
        clock=_SynchronousEvidenceClock(assembly.clock),
        runner_cfg=ExecutionRunnerCfg(
            minimum_cycle_time=0.0,
            hold_on_completion=False,
        ),
    )
    return _QuickstartRuntimeProvider(runtime)


def _documented_pick_place_quickstart(
    runtime_provider: _QuickstartRuntimeProvider,
) -> SkillResult:
    """Run the application-facing quickstart, excluding scene construction."""
    skills = AtomicSkills.from_env(runtime_provider, preset="evidence")
    cube = skills.scene.object("cube")
    return skills.run(
        Pick(object=cube),
        Place(object=cube, at=_DIRECT_PLACE_TARGET),
    )


def _python_pick_place_calls() -> tuple[SemanticCallSpec, ...]:
    """Return the application-facing calls used by both acceptance paths."""
    cube = SceneObjectRef("cube")
    return (
        Pick(object=cube),
        Place(object=cube, at=_DIRECT_PLACE_TARGET),
    )


def _pick_place_program_data() -> dict[str, object]:
    """Return the integration-free program shared with the MLLM frontend."""
    return {
        "schema_version": 1,
        "program_id": "pick_place_equivalence",
        "targets": {
            "place_target": {
                "kind": "cyclic_pose",
                "values": [
                    {
                        "position": _DIRECT_PLACE_TARGET.position.tolist(),
                        "quaternion_wxyz": (
                            _DIRECT_PLACE_TARGET.quaternion_wxyz.tolist()
                        ),
                    }
                ],
            }
        },
        "program": {
            "kind": "sequence",
            "items": [
                {
                    "kind": "invoke",
                    "call": {"kind": "pick", "object": "cube"},
                },
                {
                    "kind": "invoke",
                    "call": {
                        "kind": "place",
                        "object": "cube",
                        "at": {
                            "kind": "target_ref",
                            "target": "place_target",
                        },
                    },
                },
            ],
        },
    }


def _compiled_program_calls(program: CompiledProgram) -> tuple[SemanticCallSpec, ...]:
    """Flatten one provider-free compiled program into semantic calls."""
    return tuple(
        compiled_call.call for segment in program for compiled_call in segment.calls
    )


def _decoded_pick_place_calls(
    adapter: ExpertProgramEnvironmentAdapter,
) -> tuple[SemanticCallSpec, ...]:
    """Decode and compile the config equivalent of the Python calls."""
    data = _pick_place_program_data()
    data["integration"] = {
        "robot_profile": "evidence_profile",
        "scene_registry": "evidence_scene",
        "runtime_preset": "evidence",
    }
    return _compiled_program_calls(adapter.compile(decode_expert_program(data)))


def _mllm_pick_place_calls(
    adapter: ExpertProgramEnvironmentAdapter,
) -> tuple[SemanticCallSpec, ...]:
    """Compile the same program through the strict MLLM frontend."""
    program = compile_mllm_expert_program(
        json.dumps(_pick_place_program_data()),
        adapter=adapter,
        integration=_evidence_integration(),
    )
    return _compiled_program_calls(program)


def _capture_grounded_invocations(
    monkeypatch: pytest.MonkeyPatch,
    assembly: ExpertProgramRuntimeAssembly,
) -> list[ActionInvocation[Any, Any]]:
    """Record the production compiler's final lowering without replacing it."""
    invocations: list[ActionInvocation[Any, Any]] = []
    ground = assembly.compiler.ground

    def recording_ground(*args: Any, **kwargs: Any) -> Any:
        grounded = ground(*args, **kwargs)
        invocations.append(grounded.invocation)
        return grounded

    monkeypatch.setattr(assembly.compiler, "ground", recording_ground)
    return invocations


def _run_evidence_pick_place(
    assembly: ExpertProgramRuntimeAssembly,
    robot: _EvidenceRobot,
    cube: _RigidObject,
    calls: tuple[SemanticCallSpec, ...],
    *,
    skills: AtomicSkills | None = None,
) -> tuple[SkillResult, HeldObjectState]:
    """Drive one happy-path workflow through accepted commands and live evidence."""
    entry = assembly.runtime if skills is None else skills
    result = entry.start(calls, workflow_id="pick_place_equivalence")
    verified_pick: HeldObjectState | None = None
    for _ in range(32):
        while assembly.command_sink.pending_count:
            _consume_buffered_action(assembly, robot)
        if result.terminal:
            break
        if result.current_call_index == 1:
            if verified_pick is None:
                verified_pick = result.task_state.get_held_object("manipulator")
            cube.pose[:, 0, 3] = _RELEASE_SEPARATION
        assembly.clock.advance_after_env_step()
        result = entry.step()

    assert result.status is SkillStatus.COMPLETED
    assert verified_pick is not None
    assert result.task_state.get_held_object("manipulator") is None
    return result, verified_pick


def _assert_typed_equivalent(
    actual: object,
    expected: object,
    *,
    path: str = "value",
) -> None:
    """Compare nested typed compiler output, including owned tensor values."""
    assert type(actual) is type(expected), path
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        torch.testing.assert_close(actual, expected)
        return
    if isinstance(actual, Mapping):
        assert isinstance(expected, Mapping)
        assert tuple(actual) == tuple(expected)
        for key in actual:
            _assert_typed_equivalent(
                actual[key],
                expected[key],
                path=f"{path}[{key!r}]",
            )
        return
    if isinstance(actual, Sequence) and not isinstance(actual, (str, bytes)):
        assert isinstance(expected, Sequence)
        assert len(actual) == len(expected)
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_typed_equivalent(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if is_dataclass(actual) and not isinstance(actual, type):
        assert is_dataclass(expected) and not isinstance(expected, type)
        for data_field in fields(actual):
            _assert_typed_equivalent(
                getattr(actual, data_field.name),
                getattr(expected, data_field.name),
                path=f"{path}.{data_field.name}",
            )
        return
    assert actual == expected, path


def _assert_invocation_equivalent(
    actual: ActionInvocation[Any, Any],
    expected: ActionInvocation[Any, Any],
) -> None:
    """Compare semantic lowering while ignoring engine-instance owner UUIDs."""
    assert actual.skill_id == expected.skill_id
    assert actual.invocation_id == expected.invocation_id
    assert actual.revision == expected.revision
    _assert_typed_equivalent(actual.goal, expected.goal, path="invocation.goal")
    _assert_typed_equivalent(
        actual.binding.endpoints,
        expected.binding.endpoints,
        path="invocation.binding.endpoints",
    )
    _assert_typed_equivalent(
        actual.motion_policy,
        expected.motion_policy,
        path="invocation.motion_policy",
    )
    _assert_typed_equivalent(
        actual.recovery_policy,
        expected.recovery_policy,
        path="invocation.recovery_policy",
    )
    _assert_typed_equivalent(
        actual.skill_options,
        expected.skill_options,
        path="invocation.skill_options",
    )
    _assert_typed_equivalent(
        actual.control_overrides,
        expected.control_overrides,
        path="invocation.control_overrides",
    )


def test_simulation_factory_aligns_every_motion_policy_to_gym_step() -> None:
    """Cadence alignment preserves the exact registered tracking contract."""
    factory, _ = _factory()

    profile = factory.create_robot_skill_profile()

    assert profile.presets["safe"].motion_policy.control_dt == pytest.approx(_STEP_DT)
    assert profile.presets["safe"].tracking_policy == TrackingPolicy.joint_position(
        in_flight_max_abs_error=0.037,
        terminal_max_abs_error=0.019,
    )


def test_mllm_config_and_atomic_skills_share_invocations_and_verified_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All public frontends reach equivalent invocations and verified state."""
    (
        _,
        python_assembly,
        python_robot,
        python_cube,
    ) = _evidence_adapter_runtime()
    (
        config_adapter,
        config_assembly,
        config_robot,
        config_cube,
    ) = _evidence_adapter_runtime()
    (
        mllm_adapter,
        mllm_assembly,
        mllm_robot,
        mllm_cube,
    ) = _evidence_adapter_runtime()
    python_invocations = _capture_grounded_invocations(monkeypatch, python_assembly)
    config_invocations = _capture_grounded_invocations(monkeypatch, config_assembly)
    mllm_invocations = _capture_grounded_invocations(monkeypatch, mllm_assembly)
    runtime_provider = _QuickstartRuntimeProvider(python_assembly.runtime)
    python_skills = AtomicSkills.from_env(runtime_provider, preset="evidence")

    python_result, python_held = _run_evidence_pick_place(
        python_assembly,
        python_robot,
        python_cube,
        _python_pick_place_calls(),
        skills=python_skills,
    )
    config_result, config_held = _run_evidence_pick_place(
        config_assembly,
        config_robot,
        config_cube,
        _decoded_pick_place_calls(config_adapter),
    )
    mllm_result, mllm_held = _run_evidence_pick_place(
        mllm_assembly,
        mllm_robot,
        mllm_cube,
        _mllm_pick_place_calls(mllm_adapter),
    )

    assert runtime_provider.presets == ["evidence"]
    assert (
        len(python_invocations) == len(config_invocations) == len(mllm_invocations) == 2
    )
    for python_invocation, config_invocation, mllm_invocation in zip(
        python_invocations,
        config_invocations,
        mllm_invocations,
        strict=True,
    ):
        _assert_invocation_equivalent(python_invocation, config_invocation)
        _assert_invocation_equivalent(python_invocation, mllm_invocation)
    _assert_typed_equivalent(python_held, config_held)
    _assert_typed_equivalent(python_held, mllm_held)
    _assert_typed_equivalent(python_result, config_result)
    _assert_typed_equivalent(python_result, mllm_result)


def test_atomic_skills_from_env_runs_documented_pick_place_quickstart() -> None:
    """The small public facade executes without exposing core motion plumbing."""
    provider = _quickstart_runtime_provider()

    result = _documented_pick_place_quickstart(provider)

    source = textwrap.dedent(inspect.getsource(_documented_pick_place_quickstart))
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    executable = function.body[1:]  # Exclude the helper's docstring.
    assert executable[-1].end_lineno is not None
    assert executable[-1].end_lineno - executable[0].lineno + 1 <= (
        _QUICKSTART_MAX_LINES
    )
    identifiers = {
        identifier
        for node in ast.walk(function)
        for identifier in (
            node.id if isinstance(node, ast.Name) else None,
            node.attr if isinstance(node, ast.Attribute) else None,
        )
        if identifier is not None
    }
    assert identifiers.isdisjoint(
        {
            "qpos",
            "matrix",
            "planner",
            "session",
            "MotionGenerator",
            "PlanningContext",
            "ExecutionSession",
        }
    )
    assert provider.presets == ["evidence"]
    assert result.status is SkillStatus.COMPLETED
    assert result.success_mask.tolist() == [True] * _BATCH_SIZE
    assert [call.semantic_id for call in result.calls] == ["pick", "place"]
    assert result.task_state.get_held_object("manipulator") is None


def test_simulation_factory_builds_shared_observation_and_evidence_ports() -> None:
    """Observation and both built-in evidence providers share one scene source."""
    factory, robot = _factory()
    registry = factory.create_scene_registry()
    profile = factory.create_robot_skill_profile()
    engine = factory.create_atomic_action_engine(profile)
    clock = EnvironmentStepClock(_STEP_DT)

    observation = factory.create_planning_observation_provider(
        scene_registry=registry,
        engine=engine,
        clock=clock,
    )
    assert type(observation) is SimulationPlanningObservationProvider
    context = observation.observe(TaskState.empty(_BATCH_SIZE, robot.device))
    providers = tuple(
        factory.create_effect_evidence_providers(
            scene_registry=registry,
            engine=engine,
            observation_provider=observation,
        )
    )
    accepted_command_observer = factory.create_accepted_runtime_command_observer(
        scene_registry=registry,
        engine=engine,
        observation_provider=observation,
    )

    assert context.robot.timestamp == pytest.approx(0.0)
    assert torch.equal(observation.current_qpos(context.env_ids), robot.qpos)
    assert accepted_command_observer is observation.command_state_tracker
    assert len(providers) == 2
    assert all(
        getattr(provider, "_scene_provider") is observation.scene_provider
        for provider in providers
    )


def test_simulation_factory_returns_exact_environment_adapter() -> None:
    """The convenience path remains compatible with the exact-type mixin check."""
    factory, _ = _factory()

    adapter = factory.create_adapter()

    assert type(adapter) is ExpertProgramEnvironmentAdapter
    assert adapter.step_dt == pytest.approx(_STEP_DT)
    assert factory.segment_policy_port is not None


def test_simulation_helper_consumes_registered_semantic_grounding_extensions() -> None:
    """Both registration-owned grounding seams reach the compiler unchanged."""
    robot = _Robot()
    environment = SimpleNamespace(
        sim=_Simulation(robot),
        robot=robot,
        step_dt=_STEP_DT,
    )
    relation_grounder = _ForwardedRelationGrounder()
    handover_provider = _ForwardedHandOverPoseProvider()
    adapter = create_simulation_expert_program_adapter(
        environment,  # type: ignore[arg-type]
        registration=SimulationExpertProgramRegistration(
            scene_binding=SimulationSceneBinding(registry_id="scene"),
            robot_profile_binding=_profile_binding(),
            relation_grounders=(relation_grounder,),
            handover_pose_providers=(handover_provider,),
        ),
        motion_generator_factory=lambda: _motion_generator(robot),
    )

    assembly = adapter.assemble_runtime(
        ExpertProgramIntegrationCfg(
            robot_profile="robot_profile",
            scene_registry="scene",
            runtime_preset="safe",
        )
    )

    assert tuple(assembly.compiler.relation_grounders.values()) == (relation_grounder,)
    assert tuple(assembly.compiler.handover_pose_providers.values()) == (
        handover_provider,
    )


def test_handover_registration_is_fail_closed_without_selected_provider() -> None:
    """A profile-selected provider must be installed before simulation startup."""
    _, scene_binding, profile_binding = _handover_helper_inputs()

    with pytest.raises(ValueError, match="selects handover pose provider"):
        SimulationExpertProgramRegistration(
            scene_binding=scene_binding,
            robot_profile_binding=profile_binding,
        )


def test_simulation_helper_uses_registered_handover_provider_for_preflight() -> None:
    """A registration-owned embodiment provider satisfies standard preflight."""
    environment, scene_binding, profile_binding = _handover_helper_inputs()
    robot = environment.robot
    provider = _ForwardedHandOverPoseProvider()
    adapter = create_simulation_expert_program_adapter(
        environment,  # type: ignore[arg-type]
        registration=SimulationExpertProgramRegistration(
            scene_binding=scene_binding,
            robot_profile_binding=profile_binding,
            handover_pose_providers=(provider,),
        ),
        motion_generator_factory=lambda: _motion_generator(robot),
    )

    bridge = adapter.create_bridge(adapter.compile(_handover_program()))

    assert bridge is not None


def test_simulation_helper_assembles_mobile_endpoint_and_transport_without_joints() -> (
    None
):
    """The one-line factory path supports a custom non-joint controller."""
    robot = _MobileRobot()
    simulation = _Simulation(robot)  # type: ignore[arg-type]
    profile_binding = SimulationRobotSkillProfileBinding(
        profile_id="mobile_profile",
        resources=(
            RobotResourceBinding(
                resource_id="mobile_base",
                endpoints={
                    "motion": _MobileEndpoint(
                        controller_id="base_velocity",
                        capabilities=frozenset({"motion.base.velocity"}),
                    )
                },
            ),
        ),
        presets=(SkillPolicyPreset("runtime"),),
        default_preset="runtime",
    )
    environment = SimpleNamespace(
        sim=simulation,
        robot=robot,
        step_dt=_STEP_DT,
    )

    adapter = create_simulation_expert_program_adapter(
        environment,  # type: ignore[arg-type]
        registration=SimulationExpertProgramRegistration(
            scene_binding=SimulationSceneBinding(registry_id="mobile_scene"),
            robot_profile_binding=profile_binding,
        ),
        motion_generator_factory=lambda: _motion_generator(robot),  # type: ignore[arg-type]
        endpoint_adapters={_MobileEndpoint: _MobileEndpointAdapter()},
        runtime_transports=(_MobileTransportEncoder(),),
    )
    assembly = adapter.assemble_runtime(
        ExpertProgramIntegrationCfg(
            robot_profile="mobile_profile",
            scene_registry="mobile_scene",
            runtime_preset="runtime",
        )
    )

    endpoint = assembly.robot_profile.resources["mobile_base"].endpoints["motion"]
    assert isinstance(endpoint, _MobileEndpoint)
    assert "test.mobile_velocity" in assembly.command_encoder.transport_ids
    assert assembly.engine.skill_profile is not None
    resolved = assembly.engine.skill_profile.resources["mobile_base"]
    assert isinstance(resolved.endpoints["motion"].runtime_target, _MobileTarget)
    assert resolved.claim.claim_tokens == frozenset({"controller:base_velocity"})


def test_pick_place_effects_require_accepted_hand_state_and_live_pose() -> None:
    """Production Pick/Place evidence stays conjunctive through runtime traces."""
    assembly, robot, cube = _evidence_runtime()
    assert type(assembly.accepted_command_observer) is (
        ControlCommandStateEvidenceTracker
    )
    cube.pose[:, 0, 3] = 0.2
    result = assembly.runtime.start(
        (
            Pick(object=SceneObjectRef("cube")),
            Place(
                object=SceneObjectRef("cube"),
                at=SemanticPose(
                    position=(0.0, 0.0, 0.0),
                    quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
                ),
            ),
        ),
        workflow_id="production_evidence_chain",
    )
    assert result.status is SkillStatus.RUNNING

    result = assembly.runtime.step()
    assert assembly.command_sink.pending_count == 1
    assert len(result.effects) == 0
    _consume_buffered_action(assembly, robot)
    result = assembly.runtime.step()
    assert len(result.effects) == 0
    while assembly.command_sink.pending_count:
        _consume_buffered_action(assembly, robot)

    result, pick_pose_missing = _sample_effect(
        assembly,
        robot,
        expected_trace_count=1,
    )
    pick_pose = pick_pose_missing.evidence["destination.pose"]
    pick_constraint = pick_pose_missing.evidence["destination.constraint"]
    assert type(pick_pose) is PoseRelationEvidenceBatch
    assert type(pick_constraint) is BinaryEffectEvidenceBatch
    assert pick_pose.object_to_endpoint[:, 0, 3].tolist() == pytest.approx(
        [-0.2] * _BATCH_SIZE
    )
    assert pick_constraint.values.tolist() == [True] * _BATCH_SIZE
    assert pick_constraint.valid.tolist() == [True] * _BATCH_SIZE
    assert not pick_pose_missing.success_mask.any()

    cube.pose = torch.eye(4).repeat(_BATCH_SIZE, 1, 1)
    assembly.command_sink.discard_pending()
    result, pick_command_missing = _sample_effect(
        assembly,
        robot,
        expected_trace_count=2,
    )
    pick_pose = pick_command_missing.evidence["destination.pose"]
    pick_constraint = pick_command_missing.evidence["destination.constraint"]
    torch.testing.assert_close(
        pick_pose.object_to_endpoint,
        torch.eye(4).repeat(_BATCH_SIZE, 1, 1),
    )
    assert pick_constraint.valid.tolist() == [False] * _BATCH_SIZE
    assert not pick_command_missing.success_mask.any()

    _accept_hand_command(assembly, robot, _HAND_GRASP_POSITION)
    result, pick_first_complete_sample = _sample_effect(
        assembly,
        robot,
        expected_trace_count=3,
        advance_clock=False,
    )
    assert not pick_first_complete_sample.success_mask.any()
    result, pick_success = _sample_effect(
        assembly,
        robot,
        expected_trace_count=4,
    )
    assert pick_success.call_index == 0
    assert pick_success.effect_spec.semantic_id == "pick"
    assert pick_success.success_mask.tolist() == [True] * _BATCH_SIZE
    assert (
        pick_success.evidence["destination.constraint"].values.tolist()
        == [True] * _BATCH_SIZE
    )
    assert result.task_state.get_held_object("manipulator") is not None
    assert result.current_call_index == 1

    result = assembly.runtime.step()
    assert assembly.command_sink.pending_count == 1
    assert len(result.effects) == 4
    _consume_buffered_action(assembly, robot)
    result = assembly.runtime.step()
    assert len(result.effects) == 4
    while assembly.command_sink.pending_count:
        _consume_buffered_action(assembly, robot)

    result, place_pose_missing = _sample_effect(
        assembly,
        robot,
        expected_trace_count=5,
    )
    place_pose = place_pose_missing.evidence["source.pose"]
    place_constraint = place_pose_missing.evidence["source.constraint"]
    assert type(place_pose) is PoseRelationEvidenceBatch
    assert type(place_constraint) is BinaryEffectEvidenceBatch
    torch.testing.assert_close(
        place_pose.object_to_endpoint,
        torch.eye(4).repeat(_BATCH_SIZE, 1, 1),
    )
    assert place_constraint.values.tolist() == [False] * _BATCH_SIZE
    assert place_constraint.valid.tolist() == [True] * _BATCH_SIZE
    assert not place_pose_missing.success_mask.any()

    cube.pose[:, 0, 3] = 0.2
    assembly.command_sink.discard_pending()
    result, place_command_missing = _sample_effect(
        assembly,
        robot,
        expected_trace_count=6,
    )
    place_pose = place_command_missing.evidence["source.pose"]
    place_constraint = place_command_missing.evidence["source.constraint"]
    assert place_pose.object_to_endpoint[:, 0, 3].tolist() == pytest.approx(
        [-0.2] * _BATCH_SIZE
    )
    assert place_constraint.valid.tolist() == [False] * _BATCH_SIZE
    assert not place_command_missing.success_mask.any()

    _accept_hand_command(assembly, robot, _HAND_OPEN_POSITION)
    result, place_first_complete_sample = _sample_effect(
        assembly,
        robot,
        expected_trace_count=7,
        advance_clock=False,
    )
    assert not place_first_complete_sample.success_mask.any()
    result, place_success = _sample_effect(
        assembly,
        robot,
        expected_trace_count=8,
    )
    assert result.status is SkillStatus.COMPLETED
    assert place_success.call_index == 1
    assert place_success.effect_spec.semantic_id == "place"
    assert place_success.success_mask.tolist() == [True] * _BATCH_SIZE
    assert (
        place_success.evidence["source.constraint"].values.tolist()
        == [False] * _BATCH_SIZE
    )
    assert result.task_state.get_held_object("manipulator") is None
    assert [len(call.effects) for call in result.calls] == [4, 4]
    assert assembly.command_sink.accepted_action_count >= 4
