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

"""Exact occurrence bindings and current physical facts for measured rigid planning."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import uuid4
import torch
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose, resolve_pose_goal
from embodichain.lab.sim.atomic_actions.measured_rigid import (
    MeasuredInsidePlanningInputs,
    MeasuredRigidPlanningSession,
)
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.atomic_actions.source_native_contact import (
    NativeQueryScope,
    NativeClosingQuery,
    NativeContactPart,
    build_native_closing_query,
)
from embodichain.lab.sim.scene.native_motion_scene import (
    NativeRestSceneQuery,
    NativeRestCollisionPart,
    capture_native_rest_scene,
    capture_native_arm_parts,
)
from embodichain.lab.sim.scene.planar_observation import (
    NativePlanarPlacement,
    capture_planar_contents,
)
from .bindings import ContainerAffordanceBinding

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ResolvedActionRequest,
        PlanningContext,
        ObjectSemantics,
    )
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg

__all__ = ["MeasuredRigidInvocationBinding", "SimulationMeasuredRigidProvider"]


@dataclass(frozen=True, slots=True)
class MeasuredRigidInvocationBinding:
    """Bind one declared ordinary Pick or exact rigid-inside Place occurrence."""

    invocation_id: str
    object_id: str
    container_id: str | None = None
    collision_backend: Literal["newton", "fcl"] = field(default="newton", kw_only=True)
    """Explicit closing backend for this measured rigid planning occurrence."""

    def __post_init__(self) -> None:
        if type(self.collision_backend) is not str or self.collision_backend not in {
            "newton",
            "fcl",
        }:
            raise ValueError("collision_backend must be 'newton' or 'fcl'.")
        for name in ("invocation_id", "object_id", "container_id"):
            value = getattr(self, name)
            if value is None and name == "container_id":
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    "Measured rigid declarations require exact identities."
                )


class _MeasuredSession:
    physics_backend = "default"

    def __init__(self, provider, request, context, binding):
        self._provider, self._request, self._context = provider, request, context
        self.subject_id, self.container_id = binding.object_id, binding.container_id
        self._collision_backend = binding.collision_backend
        self.source_uid = provider._rigids[self.subject_id]
        self._stamp = context.world_source_stamp
        self._qpos = context.robot.qpos.detach().clone()
        self._root = (
            context.robot.root_pose.detach().clone()
            if context.robot.root_pose is not None
            else None
        )
        self._queries = []
        self._scene = None
        self._closed = False
        self.validate_observation()

    def validate_observation(self) -> None:
        if self._closed:
            raise RuntimeError("Measured rigid observation session is closed.")
        provider = self._provider
        robot = provider.robot
        sim = provider._simulation
        if (
            type(self._stamp) is not PhysicalObservationStamp
            or self._stamp.source_kind != "default.contacts"
            or provider._read_stamp() != self._stamp
            or sim.get_robot(robot.uid) is not robot
            or self._root is None
            or not torch.equal(robot.get_qpos().to(self._qpos), self._qpos)
            or not torch.equal(
                robot.get_local_pose(to_matrix=True).to(self._root), self._root
            )
        ):
            raise ValueError(
                "Measured rigid original native observation changed or is unavailable."
            )

    def closing_query(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        semantics: ObjectSemantics,
    ) -> NativeClosingQuery:
        self.validate_observation()
        if (
            request.invocation_id != self._request.invocation_id
            or context is not self._context
            or semantics.entity_id != self.subject_id
        ):
            raise ValueError(
                "Measured closing query changed its exact occurrence/source."
            )
        endpoint = request.binding.endpoint("primary", "grasp")
        hand = endpoint.require_target(JointPositionTarget)
        motion = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        cfg = deepcopy(self._provider._hands[hand.target_id])
        if (
            cfg.sample_count != 65
            or cfg.hand_part != hand.control_part
            or cfg.motion_part != motion.control_part
        ):
            raise ValueError(
                "Measured closing requires its source 65-sample exact hand calibration."
            )
        opened, closed = [
            endpoint.joint_positions(
                command,
                num_envs=1,
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            )
            for command in ("open", "grasp")
        ]
        pose = resolve_pose_goal(
            SceneEntityPose(self.subject_id), context, name="measured_subject"
        )
        if pose.shape == (4, 4):
            pose = pose[None]
        query = build_native_closing_query(
            self._provider.robot,
            self._provider._simulation,
            source_uid=self.source_uid,
            hand_geometry_cfg=cfg,
            open_qpos=opened,
            grasp_qpos=closed,
            observation_qpos=context.robot.qpos,
            object_pose=pose,
            scope=NativeQueryScope(
                str(uuid4()), context.robot.timestamp, context.scene.version
            ),
            revision=4,
            collision_backend=self._collision_backend,
        )
        self._queries.append(query)
        self.validate_observation()
        return query

    def scene_query(self) -> NativeRestSceneQuery:
        self.validate_observation()
        if self._scene is None:
            self._scene = capture_native_rest_scene(
                self._provider._simulation,
                self._provider.robot,
                scope=NativeQueryScope(
                    str(uuid4()),
                    self._context.robot.timestamp,
                    self._context.scene.version,
                ),
                validate_observation=self.validate_observation,
            )
        return self._scene

    def arm_parts(
        self, control_part: str, hand: Sequence[NativeContactPart]
    ) -> tuple[NativeRestCollisionPart, ...]:
        self.validate_observation()
        return capture_native_arm_parts(
            self._provider._simulation,
            self._provider.robot,
            control_part,
            hand,
            validate_observation=self.validate_observation,
        )

    def inside_inputs(self, context: PlanningContext) -> MeasuredInsidePlanningInputs:
        self.validate_observation()
        if context is not self._context or self.container_id is None:
            raise ValueError("Measured inside inputs have no exact source occurrence.")
        provider = self._provider
        container = provider._containers[self.container_id]
        geometry = NativePlanarPlacement(
            provider._simulation,
            self.source_uid,
            provider._rigids[container.parent_id],
            batch_size=1,
        )
        contents = capture_planar_contents(
            provider._simulation,
            tuple(
                (name, provider._rigids[name])
                for name in sorted(
                    set(provider._rigids) - {self.subject_id, container.parent_id}
                )
            ),
        )
        result = MeasuredInsidePlanningInputs(
            self.subject_id,
            self.container_id,
            container.parent_id,
            container.release_clearance,
            geometry,
            contents,
        )
        self.validate_observation()
        return result

    def close(self) -> None:
        if self._closed:
            return
        for query in self._queries:
            query.close()
        if self._scene is not None:
            self._scene.close()
        self._queries.clear()
        self._closed = True


class SimulationMeasuredRigidProvider:
    """Resolve static exact source bindings and borrow current physical owners."""

    def __init__(
        self,
        robot: Robot,
        simulation: SimulationManager,
        *,
        bindings: tuple[MeasuredRigidInvocationBinding, ...],
        rigid_uids: Mapping[str, str],
        containers: tuple[ContainerAffordanceBinding, ...],
        hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg],
        read_source_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        """Bind declarations without starting simulation or building any action plan.

        Args:
            robot: Exact existing Robot used by the standard engine.
            simulation: Existing Default/B1 physical owner.
            bindings: Immutable exact invocation/subject/inside selections.
            rigid_uids: Canonical logical/native rigid owners.
            containers: Existing authoritative semantic container frames.
            hand_geometry_cfgs: Existing explicit hand topology/calibration configs.
            read_source_stamp: Guarded actual native observation reader.
        """
        if (
            simulation.num_envs != 1
            or not simulation.is_default_backend
            or simulation.get_robot(robot.uid) is not robot
        ):
            raise ValueError(
                "Measured rigid services require the exact Default/B1 Robot."
            )
        if type(bindings) is not tuple or any(
            type(value) is not MeasuredRigidInvocationBinding for value in bindings
        ):
            raise TypeError("Measured bindings require immutable typed declarations.")
        self._bindings = {value.invocation_id: value for value in bindings}
        if len(self._bindings) != len(bindings):
            raise ValueError("Duplicate measured rigid occurrence.")
        self.robot, self._simulation, self._read_stamp = (
            robot,
            simulation,
            read_source_stamp,
        )
        self._rigids = dict(rigid_uids)
        self._containers = {value.entity_id: value for value in containers}
        self._hands = deepcopy(dict(hand_geometry_cfgs))
        if not callable(read_source_stamp):
            raise TypeError("Measured services require their native source reader.")
        if any(
            value.object_id not in self._rigids
            or (
                value.container_id is not None
                and (
                    value.container_id not in self._containers
                    or self._containers[value.container_id].parent_id
                    not in self._rigids
                )
            )
            for value in bindings
        ):
            raise ValueError(
                "Measured manipulation requires exact rigid scene bindings."
            )

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> MeasuredRigidPlanningSession | None:
        """Create one fresh observation lifetime only for a declared exact call."""
        binding = self._bindings.get(request.invocation_id)
        if binding is None:
            return None
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.timestamp != context.scene.timestamp
        ):
            raise ValueError("Measured services require synchronized B1/env0 facts.")
        return _MeasuredSession(self, request, context, binding)


def _validate_measured_program(compiled, manifest, policies, bindings):
    from embodichain.lab.task_program.semantics import Pick, Place, SceneAffordanceRef
    from embodichain.lab.sim.atomic_actions import PickUpOptions, PlaceOptions

    if not bindings and not any(
        (isinstance(option, PickUpOptions) and option.measured_preshape)
        or (isinstance(option, PlaceOptions) and option.measured_inside)
        for preset in manifest.robot_profile.presets.values()
        for option in preset.action_option_templates.values()
    ):
        return
    declared = {value.invocation_id: value for value in bindings}
    selected = {value.invocation_id: value.preset_id for value in policies}
    seen = set()
    for segment in compiled.iter_segments():
        for call in segment.calls:
            identity = (
                f"{compiled.program_id}/{segment.segment_id}:{call.segment_call_index}"
            )
            linked = manifest.link_call(call.call, preset_id=selected.get(identity))
            preset = manifest.robot_profile.presets[linked.preset_id]
            option = preset.action_option_template(call.call.semantic_id)
            active = (type(option) is PickUpOptions and option.measured_preshape) or (
                type(option) is PlaceOptions and option.measured_inside
            )
            binding = declared.get(identity)
            if not active:
                if binding is not None:
                    raise ValueError(
                        "Measured binding requires its explicit typed action option."
                    )
                continue
            if binding is None:
                raise ValueError(
                    "Measured action requires its exact occurrence binding."
                )
            if (
                len(segment.calls) != 1
                or segment.parallel_block is not None
                or segment.repeat_frames
            ):
                raise ValueError(
                    "Measured actions require single serial source occurrences."
                )
            if (
                preset.recovery_policy.planning_budget is None
                or preset.recovery_policy.planning_budget.wall_seconds != 90.0
            ):
                raise ValueError(
                    "Measured rigid planning requires its explicit source 90-second budget."
                )
            if (
                preset.motion_policy.strategy != "ik_interp"
                or preset.motion_policy.dynamic_collision_mode != "off"
            ):
                raise ValueError(
                    "Measured actions require original ik_interp/off planning."
                )
            if type(call.call) is Pick:
                if (
                    binding.container_id is not None
                    or binding.object_id != call.call.object.entity_id
                    or call.call.grasp is not None
                    or option.adaptive_grasp != "ordinary"
                    or option.native_closing_revision not in (None, 4)
                ):
                    raise ValueError(
                        "Measured preshape requires the original ordinary Pick and optional native-four contract."
                    )
            elif type(call.call) is Place:
                if (
                    type(call.call.inside) is not SceneAffordanceRef
                    or binding.container_id != call.call.inside.entity_id
                    or binding.object_id != call.call.object.entity_id
                ):
                    raise ValueError(
                        "Measured inside binding changed its exact subject/container."
                    )
            else:
                raise ValueError(
                    "Measured calls require canonical Pick or inside Place."
                )
            seen.add(identity)
    if set(declared) != seen:
        raise ValueError("Measured bindings include missing source occurrences.")
