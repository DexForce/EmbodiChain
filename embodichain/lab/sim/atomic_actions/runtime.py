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

"""Engine-owned planning resources shared by atomic actions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from contextlib import contextmanager
import time
from types import MappingProxyType
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import torch

from embodichain.toolkits.graspkit import GraspPoseGenerator
from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    ClosingContactPolicy,
    NativeClosingQuery,
    NativeOpeningQuery,
    NativeQueryScope,
    NativeRigidRestScene,
    _neutral_identity,
    _rigid_pose,
)

from .bindings import ActionBinding, EndpointBinding, JointPositionTarget
from .control import ActionControlOverrides, ControlPartCommandProfile
from .core import resolve_runtime_device
from .requirements import (
    DisjointResourceSlots,
    DisjointSlotEndpoints,
    SkillBindingContract,
)
from .tracking import (
    JOINT_POSITION_CHANNEL,
    EndpointTrackingChannelBinding,
    EndpointTrackingFeedbackAddress,
    TrackingFeedbackSourceRef,
    TrackingProjectorRef,
    TrackingRuntime,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg


from .grasp_geometry import GraspOpeningTarget


def _physical_identity(
    extra: Mapping[str, object] | None, physical: Mapping[str, object]
) -> dict[str, object]:
    result = _neutral_identity({} if extra is None else extra)
    if set(result) & set(physical):
        raise ValueError("Caller metadata cannot override captured native identity.")
    return {**result, **physical}


class ActionPlanningServices:
    """Engine-scoped registry of planning services used by atomic actions.

    The registry itself belongs to one engine. Grasp generators are retained by
    reference so a composition root can reuse an already prepared standalone
    service (and its geometry cache) in direct and atomic-action call paths.
    """

    def __init__(
        self,
        motion_generator: MotionGenerator,
        control_profiles: Mapping[str, ControlPartCommandProfile] | None = None,
        tracking_runtime: TrackingRuntime | None = None,
        grasp_pose_generators: Mapping[str, GraspPoseGenerator] | None = None,
        *,
        hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg] | None = None,
    ) -> None:
        """Retain shared services and snapshot caller-owned calibration data.

        Args:
            motion_generator: Motion backend and exact owning robot.
            control_profiles: Direct-core command profiles by control part.
            tracking_runtime: Optional shared endpoint tracking services.
            grasp_pose_generators: Existing grasp services by endpoint target ID.
            hand_geometry_cfgs: Explicit Lab hand settings by grasp target ID;
                copied independently without capturing geometry or reading state.
        """
        self._motion_generator = motion_generator
        self._robot: Robot = motion_generator.robot
        self._device = resolve_runtime_device(motion_generator.device)
        self._binding_owner_id = uuid4().hex
        if tracking_runtime is not None and not isinstance(
            tracking_runtime,
            TrackingRuntime,
        ):
            raise TypeError("tracking_runtime must be a TrackingRuntime or None.")
        self._tracking_runtime = tracking_runtime or TrackingRuntime.with_builtins()
        self._control_profiles = self._snapshot_control_profiles(
            {} if control_profiles is None else control_profiles
        )
        self._grasp_pose_generators = self._snapshot_grasp_pose_generators(
            {} if grasp_pose_generators is None else grasp_pose_generators
        )
        self._hand_geometry_cfgs = self._snapshot_hand_geometry_cfgs(
            {} if hand_geometry_cfgs is None else hand_geometry_cfgs
        )

    @property
    def motion_generator(self) -> MotionGenerator:
        """Return the single motion generator owned by the engine."""
        return self._motion_generator

    @property
    def robot(self) -> Robot:
        """Return the robot planned by this service set."""
        return self._robot

    @property
    def device(self) -> torch.device:
        """Return the concrete device used for planning."""
        return self._device

    @property
    def binding_owner_id(self) -> str:
        """Return the opaque identity required by this engine's bindings."""
        return self._binding_owner_id

    @property
    def tracking_runtime(self) -> TrackingRuntime:
        """Return the engine-owned typed tracking runtime."""
        return self._tracking_runtime

    @property
    def control_profiles(self) -> Mapping[str, ControlPartCommandProfile]:
        """Return owned direct-core command profiles by control-part name."""
        return MappingProxyType(
            {
                name: profile.snapshot()
                for name, profile in self._control_profiles.items()
            }
        )

    @property
    def grasp_pose_generators(self) -> Mapping[str, GraspPoseGenerator]:
        """Return grasp-pose services keyed by runtime endpoint target ID."""
        return MappingProxyType(dict(self._grasp_pose_generators))

    def grasp_pose_generator(self, target_id: str) -> GraspPoseGenerator:
        """Resolve the generator installed for one grasp endpoint target.

        Args:
            target_id: Runtime target ID, normally a robot control-part name.

        Returns:
            The installed standalone grasp-pose generator.

        Raises:
            KeyError: If no generator is installed for ``target_id``.
        """
        try:
            return self._grasp_pose_generators[target_id]
        except KeyError as exc:
            raise KeyError(
                f"No grasp-pose generator is installed for endpoint target "
                f"{target_id!r}; available targets are "
                f"{sorted(self._grasp_pose_generators)}."
            ) from exc

    @property
    def hand_geometry_cfgs(self) -> Mapping[str, RobotHandGeometryCfg]:
        """Return independently owned hand calibrations by grasp target ID."""
        return MappingProxyType(deepcopy(self._hand_geometry_cfgs))

    def hand_geometry_cfg(self, target_id: str) -> RobotHandGeometryCfg:
        """Resolve an owned hand calibration without capturing robot geometry.

        Args:
            target_id: Runtime grasp endpoint target ID.

        Returns:
            An independent configuration whose pad-link order is preserved.
            Consumers sampling an invocation's closing stroke must replace its
            open/closed commands with the current resolved endpoint commands.

        Raises:
            KeyError: No explicit calibration is installed for the target.
        """
        try:
            return deepcopy(self._hand_geometry_cfgs[target_id])
        except KeyError as exc:
            raise KeyError(
                f"No hand geometry calibration is installed for endpoint target "
                f"{target_id!r}; available targets are "
                f"{sorted(self._hand_geometry_cfgs)}."
            ) from exc

    @contextmanager
    def prepare_grasp_query(
        self,
        target_id: str,
        simulation: SimulationManager,
        *,
        source_uid: str,
        open_qpos: torch.Tensor,
        grasp_qpos: torch.Tensor,
        observation_qpos: torch.Tensor,
        object_pose: torch.Tensor,
        scope: NativeQueryScope,
        policy: ClosingContactPolicy,
        opening: GraspOpeningTarget | None = None,
        include_rigid_scene: bool = False,
        gravity_in_object: np.ndarray | None = None,
        identity: Mapping[str, object] | None = None,
        report_metadata: Mapping[str, object] | None = None,
    ) -> Iterator[NativeClosingQuery]:
        """Prepare and close a scoped exact-shape grasp query through shared services.

        Args:
            target_id: Grasp endpoint with an explicitly installed hand calibration.
            simulation: Exact B1 Default simulation owning this service's Robot.
            source_uid: Registered rigid target UID, selected by the caller.
            open_qpos: Current resolved full hand-part open command, shape (1, D).
            grasp_qpos: Current resolved close command in the same order and shape.
            observation_qpos: Same-observation full named robot state, shape (1, J).
            object_pose: Same-observation target-to-world pose, shape (1, 4, 4).
            scope: Fresh preparation identity, timestamp, scene version and env0.
            policy: Explicit geometric rules; experiment IDs are not interpreted.
            opening: Optional same-scope, already-grounded release target.
            include_rigid_scene: Attach ordered native rigid rest-clearance queries.
            gravity_in_object: Explicit gravity for horizontal pad-line qualification.
            identity: Neutral caller labels; captured physical keys are reserved.
            report_metadata: Neutral source provenance for diagnostics only.

        Yields:
            Owned GraspKit query. Exiting the context closes all query resources,
            including on failure or cancellation, without destroying native entities.

        Raises:
            KeyError: The grasp endpoint has no declared hand calibration.
            ValueError: Policy, scope, geometry, ownership or backend/batch is unsupported.
        """
        from embodichain.lab.sim.scene._collision_backends import _require_newton
        from embodichain.lab.sim.scene._newton_queries import _ConvexCollection
        from embodichain.lab.sim.scene.native_hand import (
            capture_native_hand_geometry,
            capture_native_rigid_geometry,
            capture_native_rigid_obstacles,
        )

        _require_newton()
        if not isinstance(scope, NativeQueryScope):
            raise TypeError("scope must be NativeQueryScope.")
        if not isinstance(policy, ClosingContactPolicy):
            raise TypeError("policy must be ClosingContactPolicy.")
        policy = deepcopy(policy)
        policy.__post_init__()
        if type(include_rigid_scene) is not bool:
            raise TypeError("include_rigid_scene must be boolean.")
        if opening is not None and (
            not isinstance(opening, GraspOpeningTarget)
            or opening.scope != scope
            or not policy.opening_check
        ):
            raise ValueError(
                "Opening needs an enabled policy and the same observation scope."
            )
        started = time.monotonic()
        geometry = capture_native_hand_geometry(
            self.robot,
            simulation,
            source_uid=source_uid,
            hand_geometry_cfg=self.hand_geometry_cfg(target_id),
            open_qpos=open_qpos,
            grasp_qpos=grasp_qpos,
            observation_qpos=observation_qpos,
            object_pose=object_pose,
            include_contact_offsets=policy.contact_halo != "ignore",
        )
        with NativeClosingQuery(
            geometry.parts,
            geometry.target.meshes,
            geometry.target.rest_offset,
            scope=scope,
            policy=policy,
            collision_factory=_ConvexCollection,
            collision_metadata=_ConvexCollection.metadata(),
            report_metadata=report_metadata,
            target_contact_offset=geometry.target.contact_offset,
            gravity_in_object=gravity_in_object,
        ) as query:
            query.identity = _physical_identity(identity, geometry.identity)
            query.identity.setdefault("object_id", source_uid)
            query.object_pose_at_capture = geometry.object_pose
            query.build_seconds = time.monotonic() - started
            if opening is not None:
                support = capture_native_rigid_geometry(
                    simulation,
                    geometry,
                    opening.support_uid,
                    include_contact_offsets=True,
                )
                if not support.meshes:
                    raise ValueError(
                        "Native support has no enabled physical collision shapes."
                    )
                query.opening_screen = NativeOpeningQuery(
                    query,
                    support.meshes,
                    opening.reference_pose,
                    opening.object_target_pose,
                    support.rest_offset,
                    support.contact_offset,
                    scope=opening.scope,
                    identity=_physical_identity(
                        opening.identity,
                        {
                            "reference_simulation_uid": opening.support_uid,
                            "native_reference_path": support.path,
                            "native_robot_path": geometry.identity["native_robot_path"],
                        },
                    ),
                )
            if include_rigid_scene:
                targets = capture_native_rigid_obstacles(simulation, geometry)
                query.rigid_scene_screen = NativeRigidRestScene(
                    geometry.parts,
                    targets,
                    scope=scope,
                    collision_factory=_ConvexCollection,
                )
            yield query

    @staticmethod
    def _snapshot_hand_geometry_cfgs(
        values: Mapping[str, RobotHandGeometryCfg],
    ) -> dict[str, RobotHandGeometryCfg]:
        if not isinstance(values, Mapping):
            raise TypeError("hand_geometry_cfgs must be a mapping or None.")
        if not values:
            return {}
        from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

        configs = {}
        for target_id, cfg in values.items():
            if (
                type(target_id) is not str
                or not target_id
                or target_id != target_id.strip()
            ):
                raise ValueError(
                    "hand_geometry_cfgs keys must be non-empty strings without outer whitespace."
                )
            if not isinstance(cfg, RobotHandGeometryCfg):
                raise TypeError(
                    "hand_geometry_cfgs values must be RobotHandGeometryCfg instances."
                )
            owned = deepcopy(cfg)
            owned.validate_values()
            configs[target_id] = owned
        return configs

    @staticmethod
    def _snapshot_grasp_pose_generators(
        values: Mapping[str, GraspPoseGenerator],
    ) -> dict[str, GraspPoseGenerator]:
        """Validate an endpoint-to-generator service mapping."""
        if not isinstance(values, Mapping):
            raise TypeError("grasp_pose_generators must be a mapping or None.")
        generators: dict[str, GraspPoseGenerator] = {}
        for target_id, generator in values.items():
            if (
                type(target_id) is not str
                or not target_id
                or target_id != target_id.strip()
            ):
                raise ValueError(
                    "grasp_pose_generators keys must be non-empty strings "
                    "without outer whitespace."
                )
            if not isinstance(generator, GraspPoseGenerator):
                raise TypeError(
                    "grasp_pose_generators values must be GraspPoseGenerator "
                    "instances."
                )
            generators[target_id] = generator
        return generators

    @property
    def planner_name(self) -> str:
        """Return the configured planner backend name."""
        planner_cfg = getattr(
            getattr(self._motion_generator, "planner", None), "cfg", None
        )
        planner_name = getattr(planner_cfg, "planner_type", None)
        return "unknown" if planner_name is None else str(planner_name)

    def bind_control_parts(
        self,
        contract: SkillBindingContract,
        endpoints: Mapping[str, Mapping[str, str]],
        *,
        task_state_keys: Mapping[str, str] | None = None,
    ) -> ActionBinding:
        """Build a generic binding from explicit robot control-part names.

        This is the advanced direct-core construction path. Higher-level
        binding layers may produce the same :class:`ActionBinding` through
        their own resource resolution.

        Args:
            contract: Typed endpoint contract for the bound skill.
            endpoints: Nested ``slot_id -> endpoint_id -> control_part`` mapping.
            task_state_keys: Optional stable task-state key for each resource
                slot. If omitted, a slot inherits the control part of its
                ``motion`` endpoint. A slot without ``motion`` can be inferred
                from its sole control part, or otherwise uses its stable direct
                binding resource ID.

        Returns:
            Engine-owned generic endpoint binding.
        """
        if not isinstance(contract, SkillBindingContract):
            raise TypeError("contract must be a SkillBindingContract.")
        if not isinstance(endpoints, Mapping):
            raise TypeError("endpoints must be a slot-to-endpoint mapping.")
        expected = {
            (slot.slot_id, requirement.endpoint_id): requirement
            for slot in contract.slots
            for requirement in slot.endpoints
        }
        supplied: dict[tuple[str, str], str] = {}
        for slot_id, slot_endpoints in endpoints.items():
            if not isinstance(slot_id, str) or not slot_id.strip():
                raise ValueError("Binding slot IDs must be non-empty strings.")
            if not isinstance(slot_endpoints, Mapping):
                raise TypeError(f"Binding slot {slot_id!r} must contain a mapping.")
            for endpoint_id, control_part in slot_endpoints.items():
                key = (slot_id, endpoint_id)
                if key in supplied:
                    raise ValueError(
                        f"Binding endpoint {slot_id}.{endpoint_id} repeats."
                    )
                if not isinstance(endpoint_id, str) or not endpoint_id.strip():
                    raise ValueError("Binding endpoint IDs must be non-empty strings.")
                if not isinstance(control_part, str) or not control_part.strip():
                    raise ValueError("Control-part names must be non-empty strings.")
                supplied[key] = control_part
        if set(supplied) != set(expected):
            missing = sorted(set(expected) - set(supplied))
            extra = sorted(set(supplied) - set(expected))
            raise ValueError(
                "Direct binding must cover the skill contract exactly: "
                f"missing={missing}, extra={extra}."
            )
        slot_ids = {slot.slot_id for slot in contract.slots}
        if task_state_keys is not None:
            if not isinstance(task_state_keys, Mapping):
                raise TypeError("task_state_keys must be a slot-to-key mapping.")
            for slot_id, task_state_key in task_state_keys.items():
                if (
                    not isinstance(slot_id, str)
                    or not slot_id
                    or slot_id != slot_id.strip()
                ):
                    raise ValueError(
                        "task_state_keys slot IDs must be non-empty strings "
                        "without outer whitespace."
                    )
                if not isinstance(task_state_key, str) or not task_state_key.strip():
                    raise ValueError(
                        "task_state_keys values must be non-empty strings."
                    )
                if task_state_key != task_state_key.strip():
                    raise ValueError(
                        "task_state_keys values must not contain outer whitespace."
                    )
            supplied_task_slots = set(task_state_keys)
            if supplied_task_slots != slot_ids:
                missing = sorted(slot_ids - supplied_task_slots)
                extra = sorted(supplied_task_slots - slot_ids)
                raise ValueError(
                    "task_state_keys must cover the binding slots exactly: "
                    f"missing={missing}, extra={extra}."
                )
        if not expected:
            binding = ActionBinding(owner_id=self.binding_owner_id)
            self.validate_binding(binding, contract)
            return binding

        control_parts = getattr(self.robot, "control_parts", None)
        if not isinstance(control_parts, Mapping):
            raise TypeError("Direct control-part binding requires Robot.control_parts.")
        available = sorted(str(name) for name in control_parts)
        resolved_task_state_keys: dict[str, str]
        if task_state_keys is not None:
            resolved_task_state_keys = dict(task_state_keys)
        else:
            resolved_task_state_keys = {}
            for slot in contract.slots:
                motion_key = (slot.slot_id, "motion")
                if motion_key in supplied:
                    resolved_task_state_keys[slot.slot_id] = supplied[motion_key]
                    continue
                slot_control_parts = {
                    supplied[(slot.slot_id, endpoint.endpoint_id)]
                    for endpoint in slot.endpoints
                }
                if len(slot_control_parts) != 1:
                    resolved_task_state_keys[slot.slot_id] = f"direct.{slot.slot_id}"
                    continue
                resolved_task_state_keys[slot.slot_id] = next(iter(slot_control_parts))
        resolved: list[EndpointBinding] = []
        for key, requirement in expected.items():
            slot_id, endpoint_id = key
            control_part = supplied[key]
            if control_part not in control_parts:
                raise ValueError(
                    f"Endpoint {slot_id}.{endpoint_id} references control part "
                    f"{control_part!r}, but Robot.control_parts contains {available}."
                )
            joint_ids = tuple(self.robot.get_joint_ids(name=control_part))
            if not joint_ids:
                raise ValueError(f"Control part {control_part!r} contains no joints.")
            profile = self._control_profiles.get(control_part)
            commands = {} if profile is None else profile.commands
            for name, command_type in requirement.required_commands.items():
                command = commands.get(name)
                if not isinstance(command, command_type):
                    raise ValueError(
                        f"Endpoint {slot_id}.{endpoint_id} requires command {name!r} "
                        f"of type {command_type.__name__}."
                    )
            target = JointPositionTarget(control_part, joint_ids)
            resolved.append(
                EndpointBinding(
                    slot_id=slot_id,
                    endpoint_id=endpoint_id,
                    resource_id=f"direct.{slot_id}",
                    adapter_id="control_part",
                    target=target,
                    task_state_key=resolved_task_state_keys[slot_id],
                    tracking_channels={
                        JOINT_POSITION_CHANNEL: EndpointTrackingChannelBinding(
                            channel_id=JOINT_POSITION_CHANNEL,
                            source=TrackingFeedbackSourceRef(
                                provider_id="planning_context.robot",
                                revision="1",
                                address=EndpointTrackingFeedbackAddress(
                                    target=target,
                                    channel_id=JOINT_POSITION_CHANNEL,
                                ),
                            ),
                            projector=TrackingProjectorRef(
                                projector_id="joint_position_payload",
                                revision="1",
                            ),
                        )
                    },
                    capabilities=requirement.capabilities,
                    commands=commands,
                    claim_tokens=frozenset({f"robot.control_part:{control_part}"}),
                    joint_ids=joint_ids,
                )
            )
        binding = ActionBinding(
            owner_id=self.binding_owner_id,
            endpoints=tuple(resolved),
        )
        self.validate_binding(binding, contract)
        return binding

    def validate_binding(
        self,
        binding: ActionBinding,
        contract: SkillBindingContract,
    ) -> None:
        """Validate endpoint coverage, ownership, capabilities, and claims."""
        if not isinstance(binding, ActionBinding):
            raise TypeError("binding must be an ActionBinding.")
        if binding.owner_id != self.binding_owner_id:
            raise ValueError("ActionBinding belongs to another engine instance.")
        expected = {
            (slot.slot_id, requirement.endpoint_id): requirement
            for slot in contract.slots
            for requirement in slot.endpoints
        }
        if set(binding.endpoint_keys) != set(expected):
            missing = sorted(set(expected) - set(binding.endpoint_keys))
            extra = sorted(set(binding.endpoint_keys) - set(expected))
            raise ValueError(
                "ActionBinding must cover the skill contract exactly: "
                f"missing={missing}, extra={extra}."
            )
        for key, requirement in expected.items():
            endpoint = binding.endpoint(*key)
            missing_capabilities = requirement.capabilities - endpoint.capabilities
            if missing_capabilities:
                raise ValueError(
                    f"Endpoint {key[0]}.{key[1]} is missing capabilities "
                    f"{sorted(missing_capabilities)}."
                )
            for name, command_type in requirement.required_commands.items():
                command = endpoint.commands.get(name)
                if not isinstance(command, command_type):
                    raise ValueError(
                        f"Endpoint {key[0]}.{key[1]} requires command {name!r} "
                        f"of type {command_type.__name__}."
                    )
        for slot in contract.slots:
            for constraint in slot.constraints:
                if not isinstance(constraint, DisjointSlotEndpoints):
                    continue
                selected = [
                    binding.endpoint(slot.slot_id, endpoint_id)
                    for endpoint_id in constraint.endpoint_ids
                ]
                self._validate_disjoint(selected, label=f"slot {slot.slot_id!r}")
        for constraint in contract.constraints:
            if not isinstance(constraint, DisjointResourceSlots):
                continue
            for index, left_slot in enumerate(constraint.slots):
                left = [
                    endpoint
                    for endpoint in binding.endpoints
                    if endpoint.slot_id == left_slot
                ]
                for right_slot in constraint.slots[index + 1 :]:
                    right = [
                        endpoint
                        for endpoint in binding.endpoints
                        if endpoint.slot_id == right_slot
                    ]
                    self._validate_disjoint(
                        left + right,
                        label=f"slots {left_slot!r} and {right_slot!r}",
                        only_across=len(left),
                    )

    def apply_command_overrides(
        self,
        binding: ActionBinding,
        overrides: ActionControlOverrides,
    ) -> ActionBinding:
        """Apply endpoint-scoped commands to an owned validated binding."""
        if not isinstance(overrides, ActionControlOverrides):
            raise TypeError("overrides must be an ActionControlOverrides.")
        if overrides.is_empty:
            return ActionBinding(binding.owner_id, binding.endpoints)
        return binding.with_command_overrides(overrides.as_flat_mapping())

    @staticmethod
    def _validate_disjoint(
        endpoints: list[EndpointBinding],
        *,
        label: str,
        only_across: int | None = None,
    ) -> None:
        """Reject overlapping destination, claim-token, or joint ownership."""
        pairs = (
            (
                (left, right)
                for left in endpoints[:only_across]
                for right in endpoints[only_across:]
            )
            if only_across is not None
            else (
                (left, right)
                for index, left in enumerate(endpoints)
                for right in endpoints[index + 1 :]
            )
        )
        for left, right in pairs:
            same_destination = left.destination_key == right.destination_key
            overlapping_tokens = left.claim_tokens & right.claim_tokens
            left_joints = set(left.joint_ids)
            right_joints = set(right.joint_ids)
            if same_destination or overlapping_tokens or left_joints & right_joints:
                raise ValueError(
                    f"ActionBinding violates disjoint constraint for {label}: "
                    f"{left.key} conflicts with {right.key}."
                )

    def _snapshot_control_profiles(
        self,
        profiles: Mapping[str, ControlPartCommandProfile],
    ) -> Mapping[str, ControlPartCommandProfile]:
        """Validate control-part profile ownership and freeze snapshots."""
        if not isinstance(profiles, Mapping):
            raise TypeError("control_profiles must be a mapping.")
        control_parts = getattr(self.robot, "control_parts", None)
        if not isinstance(control_parts, Mapping):
            if profiles:
                raise TypeError(
                    "Control-part command profiles require Robot.control_parts."
                )
            control_parts = {}
        snapshots: dict[str, ControlPartCommandProfile] = {}
        available = sorted(str(name) for name in control_parts)
        for name, profile in profiles.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    "control_profiles keys must be non-empty control-part names."
                )
            if name not in control_parts:
                raise ValueError(
                    f"Control profile references unknown control part {name!r}; "
                    f"Robot.control_parts contains {available}."
                )
            if not isinstance(profile, ControlPartCommandProfile):
                raise TypeError(
                    "control_profiles values must be "
                    "ControlPartCommandProfile instances."
                )
            snapshots[name] = profile.snapshot()
        return MappingProxyType(snapshots)


__all__ = ["ActionPlanningServices", "GraspOpeningTarget"]
