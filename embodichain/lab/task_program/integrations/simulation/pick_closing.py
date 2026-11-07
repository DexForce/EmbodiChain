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

"""Explicit same-observation native Pick query assembly, without skill policy."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import uuid4
from weakref import WeakKeyDictionary
import numpy as np
import torch
from embodichain.lab.sim.atomic_actions.goals import SceneEntityPose, resolve_pose_goal
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.atomic_actions.source_native_contact import (
    NativeQueryScope,
    NativeOpeningTarget,
    NativeClosingQuery,
    NativeRigidRestScene,
    build_native_closing_query,
    build_native_rigid_rest_scene,
)
from embodichain.lab.sim.scene.physical_parameters import observed_gravity
from .targets import (
    AbsolutePoseTarget,
    SceneEntityTarget,
    ConfiguredPoseTarget,
    resolve_pose_target,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        PlanningContext,
        ResolvedActionRequest,
    )
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
__all__ = ["PickClosingBinding", "SimulationPickClosingProvider"]


@dataclass(frozen=True, slots=True)
class PickClosingBinding:
    """Bind an exact Pick occurrence and optional already-declared opening target.

    The opening pose describes the placed object's world pose, not a hand pose.
    The reference pairs with either an explicit pose or an exact future Place
    occurrence. The latter consumes this Pick request's already-grounded final
    downstream pose, preserving observed orientation without re-grounding Place.
    Absence enables no opening gate.
    """

    invocation_id: str
    object_id: str
    opening_reference_id: str | None = None
    opening_object_target: ConfiguredPoseTarget | None = None
    opening_place_invocation_id: str | None = None
    collision_backend: Literal["newton", "fcl"] = field(default="newton", kw_only=True)
    """Backend shared by this occurrence's closing, opening and rest screens."""

    def __post_init__(self) -> None:
        if type(self.collision_backend) is not str or self.collision_backend not in {
            "newton",
            "fcl",
        }:
            raise ValueError("collision_backend must be 'newton' or 'fcl'.")
        for name in (
            "invocation_id",
            "object_id",
            "opening_reference_id",
            "opening_place_invocation_id",
        ):
            value = getattr(self, name)
            if (
                name in {"opening_reference_id", "opening_place_invocation_id"}
                and value is None
            ):
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} requires an exact identity.")
        modes = int(self.opening_object_target is not None) + int(
            self.opening_place_invocation_id is not None
        )
        if (self.opening_reference_id is None and modes != 0) or (
            self.opening_reference_id is not None and modes != 1
        ):
            raise ValueError(
                "Native opening requires a reference and exactly one target mode."
            )
        if self.opening_place_invocation_id == self.invocation_id:
            raise ValueError("Native opening Place and Pick occurrences must differ.")
        if self.opening_object_target is not None and type(
            self.opening_object_target
        ) not in (AbsolutePoseTarget, SceneEntityTarget):
            raise TypeError("Opening target requires a typed declared pose.")


class SimulationPickClosingProvider:
    """Build Lab-owned native queries from exact occurrence and native mappings."""

    def __init__(
        self,
        robot: Robot,
        simulation: SimulationManager,
        *,
        rigid_uids: Mapping[str, str],
        bindings: tuple[PickClosingBinding, ...],
        read_source_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        """Own declarations; native geometry is acquired only by prepare.

        Args:
            robot: Exact borrowed simulation-owned robot.
            simulation: Existing Default/B1 native owner.
            rigid_uids: Explicit canonical entity-to-native UID mapping.
            bindings: Unique exact execution-occurrence selections.
            read_source_stamp: Complete guarded native observation source reader.
        """
        if (
            simulation.physics_backend != "default"
            or simulation.num_envs != 1
            or simulation.get_robot(robot.uid) is not robot
        ):
            raise ValueError("Native Pick requires the exact Default/B1 robot owner.")
        if not callable(read_source_stamp):
            raise TypeError("Native Pick requires an observation source reader.")
        if type(bindings) is not tuple or any(
            type(value) is not PickClosingBinding for value in bindings
        ):
            raise TypeError(
                "Native Pick bindings must be immutable typed declarations."
            )
        self._bindings = {value.invocation_id: value for value in bindings}
        if len(self._bindings) != len(bindings):
            raise ValueError("Duplicate native Pick occurrence.")
        self._rigid = dict(rigid_uids)
        if any(
            type(k) is not str or not k or type(v) is not str or not v
            for k, v in self._rigid.items()
        ):
            raise ValueError(
                "Native Pick mappings require explicit entity and UID strings."
            )
        for value in bindings:
            if value.object_id not in self._rigid or (
                value.opening_reference_id is not None
                and value.opening_reference_id not in self._rigid
            ):
                raise ValueError(
                    "Native Pick subjects and opening references require rigid mappings."
                )
        self.robot, self._simulation, self._read_stamp = (
            robot,
            simulation,
            read_source_stamp,
        )
        self._query_stamps: WeakKeyDictionary[
            NativeClosingQuery, PhysicalObservationStamp
        ] = WeakKeyDictionary()

    def _check_stamp(self, stamp: PhysicalObservationStamp) -> None:
        if (
            type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
            or self._read_stamp() != stamp
        ):
            raise ValueError(
                "Native Pick observation source changed or is unavailable."
            )

    def prepare(
        self,
        request: ResolvedActionRequest,
        context: PlanningContext,
        *,
        hand_geometry_cfg: RobotHandGeometryCfg,
    ) -> NativeClosingQuery:
        """Capture current resolved hand commands and the exact observed source.

        Args:
            request: Selected canonical Pick request with explicit native revision.
            context: B1/env0 same-timestamp robot and scene observation.
            hand_geometry_cfg: Shared hand calibration; source grid remains 65.

        Returns:
            Fresh owned closing query. Atomic closes it after use; rejected
            post-capture source checks close it immediately without a retry.
        """
        binding = self._bindings.get(request.invocation_id)
        if (
            request.skill_id != "pick_up"
            or binding is None
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
        ):
            raise ValueError(
                "Native Pick requires an explicitly bound B1/env0 occurrence."
            )
        if (
            request.goal.semantics.entity_id != binding.object_id
            or context.robot.timestamp != context.scene.timestamp
        ):
            raise ValueError("Native Pick source identity or observation time differs.")
        self._check_stamp(context.world_source_stamp)
        hand = request.binding.endpoint("primary", "grasp")
        target = hand.require_target(JointPositionTarget)
        if tuple(self.robot.get_joint_ids(target.control_part)) != tuple(
            target.joint_ids
        ):
            raise ValueError("Native Pick hand joint ownership changed.")
        pose = resolve_pose_goal(
            (
                SceneEntityPose(binding.object_id)
                if request.goal.object_pose is None
                else request.goal.object_pose
            ),
            context,
            name="native_pick_object",
        )
        if request.goal.object_pose is not None:
            from embodichain.lab.sim.atomic_actions.trajectory_ops import (
                resolve_pose_target as resolve_batched_pose_target,
            )

            pose = resolve_batched_pose_target(
                pose, num_envs=context.batch_size, device=context.robot.qpos.device
            )
        if pose.shape == (4, 4):
            pose = pose[None]
        if pose.shape != (1, 4, 4):
            raise ValueError("Native Pick object pose requires B1 rigid coordinates.")
        scope = NativeQueryScope(
            str(uuid4()), context.robot.timestamp, context.scene.version
        )
        opening = None
        if binding.opening_reference_id is not None:
            reference = resolve_pose_goal(
                SceneEntityPose(binding.opening_reference_id),
                context,
                name="native_opening_reference",
            )
            if binding.opening_place_invocation_id is not None:
                if (
                    request.skill_options.native_closing_revision not in range(3, 10)
                    or request.skill_options.adaptive_grasp != "ordinary"
                ):
                    raise ValueError(
                        "Grounded opening targets require ordinary native Pick revision 3-9."
                    )
                targets = request.skill_options.downstream_object_target_poses
                if not targets:
                    raise ValueError(
                        "Native opening lacks its grounded final Place target."
                    )
                target = targets[-1]
            else:
                target = resolve_pose_target(binding.opening_object_target)
            placed = resolve_pose_goal(target, context, name="native_opening_target")
            opening = NativeOpeningTarget(
                self._rigid[binding.opening_reference_id],
                reference.reshape(4, 4),
                placed.reshape(4, 4),
                scope,
            )
        gravity = None
        if request.skill_options.native_closing_revision == 9:
            frame = pose.detach().cpu().double().numpy()
            if (
                not np.isfinite(frame).all()
                or not np.allclose(frame[0, 3], (0, 0, 0, 1))
                or not np.allclose(
                    frame[0, :3, :3].T @ frame[0, :3, :3], np.eye(3), atol=1e-5
                )
                or not np.isclose(np.linalg.det(frame[0, :3, :3]), 1.0, atol=1e-5)
            ):
                raise ValueError(
                    "Pad-line quality requires a proper observed object pose."
                )
            gravity = frame[0, :3, :3].T @ np.asarray(
                observed_gravity(self._simulation)
            )
        opened, grasped = (
            hand.joint_positions(
                command,
                num_envs=1,
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            )
            for command in ("open", "grasp")
        )
        query = build_native_closing_query(
            self.robot,
            self._simulation,
            source_uid=self._rigid[binding.object_id],
            hand_geometry_cfg=hand_geometry_cfg,
            open_qpos=opened,
            grasp_qpos=grasped,
            observation_qpos=context.robot.qpos,
            object_pose=pose,
            scope=scope,
            revision=request.skill_options.native_closing_revision,
            collision_backend=binding.collision_backend,
            gravity_in_object=gravity,
            opening=opening,
        )
        try:
            self._check_stamp(context.world_source_stamp)
            self._query_stamps[query] = context.world_source_stamp
        except BaseException:
            query.close()
            raise
        return query

    def prepare_rigid_scene(self, query: NativeClosingQuery) -> NativeRigidRestScene:
        """Attach a rigid rest scene only under the original query source stamp.

        Args:
            query: Open query created by this exact provider.

        Returns:
            Its attached scene, owned and closed by the closing query. Closed or
            foreign queries reject; changed sources never trigger recapture.
        """
        stamp = self._query_stamps.get(query)
        if stamp is None or query.closed:
            raise ValueError("Rigid Pick scene requires this provider's open query.")
        self._check_stamp(stamp)
        try:
            scene = build_native_rigid_rest_scene(self._simulation, query)
            self._check_stamp(stamp)
            return scene
        except BaseException:
            query.close()
            raise


def _validate_pick_program(
    compiled,
    manifest,
    policies,
    bindings,
    *,
    free_motion_query=False,
    continuation_routes=(),
):
    from embodichain.lab.task_program.semantics import Pick, RegisteredSemanticCall
    from embodichain.lab.sim.atomic_actions import PickUpOptions

    selected = {value.invocation_id: value.preset_id for value in policies}
    declared = {value.invocation_id: value for value in bindings}
    encountered = set()
    for segment in compiled.iter_segments():
        for call in segment.calls:
            occurrence = (
                f"{compiled.program_id}/{segment.segment_id}:{call.segment_call_index}"
            )
            linked = manifest.link_call(
                call.call, path=call.source_path, preset_id=selected.get(occurrence)
            )
            preset = manifest.robot_profile.presets[linked.preset_id]
            option = preset.action_option_templates.get(linked.call.semantic_id)
            binding = declared.get(occurrence)
            native = (
                isinstance(option, PickUpOptions)
                and option.native_closing_revision is not None
            )
            if not native:
                if binding is not None:
                    raise ValueError(
                        "Native Pick declaration requires a selected native-closing Pick option."
                    )
                continue
            if binding is None:
                raise ValueError(
                    "Native Pick requires an exact occurrence-bound closing declaration."
                )
            needs_scene = (
                option.qualify_fixed_grasp
                or option.adaptive_grasp == "ordinary"
                or (
                    option.native_closing_revision == 8
                    and option.adaptive_grasp == "handover_source"
                )
            )
            if (
                option.native_closing_revision >= 7
                and needs_scene
                and not free_motion_query
            ):
                raise ValueError(
                    "Native Pick precontact qualification requires declared discrete_mesh services."
                )

            if segment.parallel_block is not None or segment.repeat_frames:
                raise ValueError(
                    "Native Pick declarations require non-repeated serial calls."
                )
            if type(linked.call) is Pick:
                entity = linked.call.object.entity_id
            elif (
                type(linked.call) is RegisteredSemanticCall
                and linked.descriptor.skill_id == "pick_up"
            ):
                entity = linked.call.arguments.get("object")
            else:
                raise ValueError(
                    "Native closing declaration must select a canonical Pick target."
                )
            if entity != binding.object_id:
                raise ValueError(
                    "Native Pick closing subject differs from the selected call."
                )
            if binding.opening_place_invocation_id is not None:
                matches = [
                    entry
                    for entry in continuation_routes
                    if entry.route.source_invocation_id == occurrence
                    and entry.route.place_invocation_id
                    == binding.opening_place_invocation_id
                ]
                if (
                    len(matches) != 1
                    or option.adaptive_grasp != "ordinary"
                    or option.native_closing_revision < 3
                    or not option.qualify_downstream_targets
                ):
                    raise ValueError(
                        "Native opening requires its exact ordinary Pick/Place continuation."
                    )
                entry = matches[0]
                if entry.route.object_id != entity or entry.route.requires_full_preview:
                    raise ValueError(
                        "Native opening requires the canonical ordinary on Place."
                    )
                from embodichain.lab.task_program.semantics import (
                    Place,
                    SceneAffordanceRef,
                    SceneObjectRef,
                )

                place = entry.calls[-1]
                parent = None
                if type(place) is Place:
                    if type(place.on) is SceneObjectRef:
                        parent = place.on.entity_id
                    elif type(place.on) is SceneAffordanceRef:
                        affordance = next(
                            value
                            for value in manifest.scene.entries
                            if value.ref.entity_id == place.on.entity_id
                        )
                        parent = (
                            None
                            if affordance.parent is None
                            else affordance.parent.entity_id
                        )
                elif type(place) is RegisteredSemanticCall:
                    parent = place.arguments.get("reference")
                if parent != binding.opening_reference_id:
                    raise ValueError(
                        "Native opening reference differs from its exact on Place owner."
                    )
            if type(binding.opening_object_target) is SceneEntityTarget:
                target = manifest.scene.resolve(
                    binding.opening_object_target.entity_id, path=call.source_path
                )
                if target.entity_id != binding.opening_object_target.entity_id:
                    raise ValueError(
                        "Native opening requires canonical target identities."
                    )
            encountered.add(occurrence)
    if encountered != set(declared):
        raise ValueError(
            "Native Pick declarations contain unknown or unused occurrences."
        )
