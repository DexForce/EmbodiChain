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

"""Observed held-object planning services over shared Scene/Motion/Sensor owners."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from types import MappingProxyType
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.planning_probes import (
    HeldStagingBinding,
    RetainedCommandAuthority,
)
from embodichain.lab.sim.evidence import (
    PhysicalObservationStamp,
    RobotCommandObservation,
)
from embodichain.lab.sim.scene.robot_distal import (
    capture_robot_distal_meshes,
    RobotDistalMeshSnapshot,
)
from embodichain.lab.sim.motion.mesh_collision import RobotArticulationCollisionSnapshot
from embodichain.lab.sim.sensors.command_sampling import (
    capture_robot_commands,
    verify_robot_commands,
)

if TYPE_CHECKING:
    from embodichain.lab.task_program.semantics.profiles import RobotSkillProfile
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions.invocation import ResolvedActionRequest
    from embodichain.lab.sim.atomic_actions.state import (
        PlanningContext,
        RobotObservation,
    )
__all__ = ["SimulationHeldPlanningProvider", "resolve_retained_command_authorities"]


class SimulationHeldPlanningProvider:
    """Bind explicitly validated occurrences to late native planning inputs.

    The assembly owner supplies program-validated bindings and retention
    authorities. Authority metadata alone is not evidence: Atomic still verifies
    the actual held state, unchanged hand commands and emitted motion. This
    adapter neither chooses staging candidates nor creates execution provenance.
    """

    def __init__(
        self,
        robot: Robot,
        simulation: SimulationManager,
        *,
        bindings: Mapping[str, HeldStagingBinding],
        retained_authorities: Mapping[str, RetainedCommandAuthority],
        read_source_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        """Own static declarations without capturing geometry or command arrays.

        Args:
            robot: Exact simulation-owned engine robot.
            simulation: Borrowed B1 physical owner.
            bindings: Host-validated execution occurrence to staging identity.
            retained_authorities: Host-validated monitored predecessor metadata;
                only selected staging occurrences may receive this authority.
            read_source_stamp: Guarded reader of the complete observed source.
        """
        if simulation.num_envs != 1 or simulation.get_robot(robot.uid) is not robot:
            raise ValueError(
                "Held planning requires the exact simulation-owned B1 robot."
            )
        if not callable(read_source_stamp):
            raise TypeError("Held planning requires a native source-stamp reader.")
        selected, authorities = dict(bindings), dict(retained_authorities)
        if any(
            type(key) is not str
            or not key
            or key != key.strip()
            or type(value) is not HeldStagingBinding
            for key, value in selected.items()
        ):
            raise TypeError("Staging requires exact occurrences and typed bindings.")
        if set(authorities) - set(selected) or any(
            type(value) is not RetainedCommandAuthority
            or value.entity_id != selected[key].rigid_entity_id
            for key, value in authorities.items()
        ):
            raise ValueError(
                "Retained authority must match a declared staging subject."
            )
        self._robot, self._simulation = robot, simulation
        self._bindings = MappingProxyType(selected)
        self._authorities = MappingProxyType(authorities)
        self._read_stamp = read_source_stamp

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed engine robot."""
        return self._robot

    def binding(self, request: ResolvedActionRequest) -> HeldStagingBinding | None:
        """Return static ownership without reading robot state.

        Args:
            request: Exact resolved execution occurrence.

        Returns:
            Its explicit binding, or None when no staging was declared.
        """
        return (
            self._bindings.get(request.invocation_id)
            if request.skill_id == "move_held_object"
            else None
        )

    def authority(
        self, request: ResolvedActionRequest
    ) -> RetainedCommandAuthority | None:
        """Return supplied verified declaration metadata, never new evidence.

        Args:
            request: Exact resolved held-object execution occurrence.

        Returns:
            Owned predecessor authority, or None for the original strict guard.
        """
        return (
            self._authorities.get(request.invocation_id)
            if self.binding(request) is not None
            else None
        )

    def _observed(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotObservation:
        if self.binding(request) is None:
            raise ValueError(
                "Held planning occurrence has no declared staging binding."
            )
        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Held planning requires B1/native env0.")
        stamp = context.world_source_stamp
        observed = (
            context.observed_robot
            if context.observed_robot is not None
            else context.robot
        )
        if (
            type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
            or observed.root_pose is None
            or observed.timestamp != context.scene.timestamp
        ):
            raise ValueError(
                "Held planning requires the original root and Default source observation."
            )
        self._check_stamp(stamp)
        return observed

    def _check_stamp(self, stamp: PhysicalObservationStamp) -> None:
        if self._read_stamp() != stamp:
            raise ValueError("Held planning native observation source changed.")

    def _active_part(self, request: ResolvedActionRequest) -> str:
        target = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        if tuple(self.robot.get_joint_ids(target.control_part)) != tuple(
            target.joint_ids
        ):
            raise ValueError("Held planning active joint ownership changed.")
        return target.control_part

    def capture_distal_geometry(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotDistalMeshSnapshot:
        """Capture original-observation geometry through the shared Scene API.

        Args:
            request: Exact occurrence and explicitly bound active control part.
            context: Planning context retaining the original measured robot.

        Returns:
            Owned distal input geometry. Missing/stale observations raise;
            projected command starts never become native expectations.
        """
        observed = self._observed(request, context)
        result = capture_robot_distal_meshes(
            self.robot,
            expected_qpos=observed.qpos,
            expected_root_pose=observed.root_pose,
            active_control_part=self._active_part(request),
        )
        self._check_stamp(context.world_source_stamp)
        return result

    def _check_robot(self, observed: RobotObservation) -> None:
        actual = self.robot.get_qpos()
        root = self.robot.get_local_pose(to_matrix=True)
        if not torch.equal(actual, observed.qpos) or not torch.equal(
            root.to(observed.root_pose), observed.root_pose
        ):
            raise ValueError("Held planning original robot observation changed.")

    def prepare_articulation_query(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotArticulationCollisionSnapshot:
        """Capture the explicitly selected robot/articulation-only query.

        Args:
            request: Bound held-object invocation with articulation scope enabled.
            context: Original observed native world used for this planning call.

        Returns:
            Owned query that the Atomic caller must close. A capture rejected by
            the final guards closes immediately; borrowed native owners survive.
        """
        observed = self._observed(request, context)
        if not self.binding(request).articulation_scene:
            raise ValueError("Articulation staging scope was not declared.")
        self._check_robot(observed)
        query = RobotArticulationCollisionSnapshot(
            self.robot, self._simulation, control_part=self._active_part(request)
        )
        try:
            self._check_robot(observed)
            self._check_stamp(context.world_source_stamp)
        except BaseException:
            query.close()
            raise
        return query

    def capture(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotCommandObservation:
        """Capture actual states and retained drive targets only with authority.

        Args:
            request: Exact selected retained-command occurrence.
            context: Planning context preserving measured positions/velocities.

        Returns:
            Owned command facts. This never promotes planned commands to applied
            commands and does not enable the source's unused revision-two policy.
        """
        if self.authority(request) is None:
            raise ValueError(
                "Held command capture requires explicit verified authority."
            )
        observed = self._observed(request, context)
        return capture_robot_commands(
            self.robot,
            expected_qpos=observed.qpos,
            expected_qvel=observed.qvel,
            timestamp=observed.timestamp,
            source_stamp=context.world_source_stamp,
            read_source_stamp=self._read_stamp,
        )

    def verify_current(self, observation: RobotCommandObservation) -> None:
        """Reject changed command facts without refreshing the original snapshot.

        Args:
            observation: Owned original command acquisition from this provider.
        """
        verify_robot_commands(
            self.robot, observation, read_source_stamp=self._read_stamp
        )


def resolve_retained_command_authorities(
    profile: RobotSkillProfile,
    preset_id: str | None,
    sources: tuple[tuple[str, str, str, str, str], ...],
    *,
    retention_call_id: str = "gen_sim.align_held",
) -> dict[str, RetainedCommandAuthority]:
    """Qualify explicitly bound predecessor declarations using public monitors.

    Args:
        profile: Resolved immutable robot skill profile; no robot is constructed.
        preset_id: Exact policy preset selected by the assembly owner.
        sources: Host-validated rows of current occurrence, predecessor occurrence,
            predecessor call ID, rigid entity ID and resource ID. This function
            does not infer dependencies or select occurrences by task names.
        retention_call_id: Explicit semantic retention call whose monitor applies.

    Returns:
        Typed source-compatible authority metadata for verified composite-effect
        monitors, or an empty/subset mapping when not qualified. These declarations
        are not physical evidence and do not activate unused command-policy modes.
    """
    from embodichain.lab.task_program.semantics import (
        EffectAssurance,
        EffectMonitorRef,
        SkillPolicyPreset,
    )

    if type(sources) is not tuple or any(
        type(row) is not tuple
        or len(row) != 5
        or any(
            type(value) is not str or not value or value != value.strip()
            for value in row
        )
        for row in sources
    ):
        raise TypeError("Retained sources require immutable exact identity rows.")
    if len({row[0] for row in sources}) != len(sources):
        raise ValueError("Retained sources must keep unique occurrence identities.")
    preset = profile.presets.get(preset_id)
    if (
        type(preset) is not SkillPolicyPreset
        or preset.effect_assurance is not EffectAssurance.VERIFIED
    ):
        return {}
    result = {}
    for invocation, predecessor, call, entity, resource in sources:
        monitors = tuple(
            preset.effect_monitors.get(identity)
            for identity in (call, retention_call_id)
        )
        if any(
            type(ref) is not EffectMonitorRef
            or ref.monitor_id != "builtin.composite_effect"
            or ref.revision != "1"
            for ref in monitors
        ):
            continue
        result[invocation] = RetainedCommandAuthority(
            preset_id=preset.preset_id,
            predecessor_call_id=call,
            predecessor_invocation_id=predecessor,
            entity_id=entity,
            resource_id=resource,
            attachment_monitor_id=monitors[0].monitor_id,
            attachment_monitor_revision=monitors[0].revision,
            retention_monitor_id=monitors[1].monitor_id,
            retention_monitor_revision=monitors[1].revision,
        )
    return result


@dataclass(frozen=True, slots=True)
class HeldStagingInvocationBinding:
    """Explicit serial staging selection and optional declared retention source.

    No authority is granted from these strings alone. Program preflight verifies
    the named calls; the existing verified-monitor rule qualifies retention.
    """

    invocation_id: str
    rigid_entity_id: str
    articulation_scene: bool = False
    predecessor_invocation_id: str | None = None
    predecessor_call_id: str | None = None
    resource_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "invocation_id",
            "rigid_entity_id",
            "predecessor_invocation_id",
            "predecessor_call_id",
            "resource_id",
        ):
            value = getattr(self, name)
            if value is None and name not in {"invocation_id", "rigid_entity_id"}:
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} requires an exact identity.")
        if type(self.articulation_scene) is not bool:
            raise TypeError("articulation_scene must be bool.")
        if (
            len(
                {
                    value is None
                    for value in (
                        self.predecessor_invocation_id,
                        self.predecessor_call_id,
                        self.resource_id,
                    )
                }
            )
            != 1
        ):
            raise ValueError(
                "Retained staging source requires all predecessor/resource fields."
            )
        if self.predecessor_invocation_id == self.invocation_id:
            raise ValueError("A staging occurrence cannot be its own predecessor.")


def _declared_authorities(profile, policies, bindings):
    selected = {value.invocation_id: value.preset_id for value in policies}
    result = {}
    for binding in bindings:
        if binding.predecessor_invocation_id is None:
            continue
        row = (
            binding.invocation_id,
            binding.predecessor_invocation_id,
            binding.predecessor_call_id,
            binding.rigid_entity_id,
            binding.resource_id,
        )
        eligible = resolve_retained_command_authorities(
            profile, selected[binding.invocation_id], (row,)
        )
        if not eligible:
            continue
        from embodichain.lab.task_program.semantics import EffectAssurance

        previous = profile.presets[selected[binding.predecessor_invocation_id]]
        monitor = previous.effect_monitors.get(binding.predecessor_call_id)
        if (
            previous.effect_assurance is not EffectAssurance.VERIFIED
            or monitor is None
            or monitor.monitor_id != "builtin.composite_effect"
            or monitor.revision != "1"
        ):
            continue
        result.update(eligible)
    return result


def _validate_program_bindings(compiled, manifest, policies, bindings):
    from embodichain.lab.task_program.semantics.calls import (
        Pick,
        RegisteredSemanticCall,
    )
    from embodichain.lab.sim.atomic_actions.primitives.move_held_object import (
        MoveHeldObjectPlanningRecovery,
    )

    selected = {value.invocation_id: value.preset_id for value in policies}
    declared = {value.invocation_id: value for value in bindings}
    calls = {}
    for segment in compiled.iter_segments():
        for call in segment.calls:
            identity = (
                f"{compiled.program_id}/{segment.segment_id}:{call.segment_call_index}"
            )
            linked = manifest.link_call(
                call.call, preset_id=selected.get(identity), path=call.source_path
            )
            recovery = manifest.robot_profile.presets[
                linked.preset_id
            ].recovery_policy.planning_recovery
            if (
                isinstance(recovery, MoveHeldObjectPlanningRecovery)
                and recovery.staging
                and identity not in declared
            ):
                raise ValueError(
                    "Staging recovery requires an exact declared occurrence."
                )
            calls[identity] = (call.call, segment, len(calls))
    for binding in bindings:
        if binding.invocation_id not in calls:
            raise ValueError("Staging declaration names an unknown occurrence.")
        call, segment, index = calls[binding.invocation_id]
        if (
            segment.parallel_block is not None
            or segment.repeat_frames
            or len(segment.calls) != 1
        ):
            raise ValueError(
                "Staging requires non-repeated serial single-call segments."
            )
        if (
            type(call) is not RegisteredSemanticCall
            or call.call_id != "gen_sim.align_held"
            or call.arguments.get("object") != binding.rigid_entity_id
            or call.arguments.get("preserve_yaw") is not False
            or call.arguments.get("target") == "current_object_pose"
        ):
            raise ValueError(
                "Staging declaration must match its explicit alignment call and subject."
            )
        if binding.predecessor_invocation_id is None:
            continue
        if binding.predecessor_invocation_id not in calls:
            raise ValueError("Staging retention names an unknown predecessor.")
        before, before_segment, before_index = calls[binding.predecessor_invocation_id]
        if (
            before_index >= index
            or before_segment.parallel_block is not None
            or before_segment.repeat_frames
            or len(before_segment.calls) != 1
        ):
            raise ValueError(
                "Staging retention requires an earlier serial predecessor."
            )
        if type(before) is Pick:
            call_id, entity = "pick", before.object.entity_id
        elif type(before) is RegisteredSemanticCall:
            call_id, entity = before.call_id, before.arguments.get("object")
        else:
            call_id, entity = None, None
        if (
            call_id != binding.predecessor_call_id
            or entity != binding.rigid_entity_id
            or before.resources.get("primary") != binding.resource_id
            or call.resources.get("primary") != binding.resource_id
            or not (
                call_id in {"pick", "gen_sim.align_held"}
                or call_id.startswith("gen_sim.pick.")
            )
        ):
            raise ValueError(
                "Staging retention source call/object/resource does not match."
            )
