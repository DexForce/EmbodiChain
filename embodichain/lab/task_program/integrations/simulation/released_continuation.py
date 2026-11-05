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

"""Bind released cleanup and Park authority to exact canonical occurrences."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.atomic_actions.released_continuation import (
    ReleasedHandClearanceRecovery,
    ReleasedHandParkRecovery,
    ReleasedHandContinuationSession,
)
from embodichain.lab.task_program.semantics import (
    Place,
    RegisteredSemanticCall,
    EffectAssurance,
    SemanticIntegrationManifest,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        PlanningContext,
        ResolvedActionRequest,
        ActionPlan,
    )
    from embodichain.lab.sim.atomic_actions.bindings import ActionBinding
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.task_program.compiler import CompiledTaskProgram
    from embodichain.lab.task_program.compiler.lowering import SemanticCallCompiler
    from ..catalog import SimulationTaskProgramRegistration

__all__ = [
    "ReleasedHandContinuationBinding",
    "SimulationReleasedHandContinuationProvider",
]
_RELEASES = {
    "simulation.place_relative",
    "gen_sim.place_upright",
    "gen_sim.stack_place",
}


@dataclass(frozen=True, slots=True)
class ReleasedHandContinuationBinding:
    """Select the adjacent release, cleanup and one or two pure Park occurrences."""

    program_fingerprint: str
    release_invocation_id: str
    clearance_invocation_id: str
    park_invocation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        digest = self.program_fingerprint
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Released continuation requires an exact program SHA256.")
        ids = (
            self.release_invocation_id,
            self.clearance_invocation_id,
            *self.park_invocation_ids,
        )
        if (
            type(self.park_invocation_ids) is not tuple
            or not 1 <= len(self.park_invocation_ids) <= 2
            or len(set(ids)) != len(ids)
            or any(type(v) is not str or not v or v != v.strip() for v in ids)
        ):
            raise ValueError(
                "Released continuation requires distinct exact serial occurrence identities."
            )


@dataclass(frozen=True, slots=True)
class _Bound:
    declaration: ReleasedHandContinuationBinding
    calls: tuple
    presets: tuple[str, ...]


def _bind_released_continuations(program, manifest, policies, declarations):
    from ..invocation_policy import BoundInvocationPolicies, task_program_fingerprint

    fingerprint = task_program_fingerprint(program)
    bound = BoundInvocationPolicies(program, policies)
    bound.validate_manifest(program, manifest)
    presets = {p.invocation_id: p.preset_id for p in policies}
    segments = tuple(program.iter_segments())
    occurrences = {
        f"{program.program_id}/{s.segment_id}:{call.segment_call_index}": (
            i,
            s,
            call.call,
        )
        for i, s in enumerate(segments)
        for call in s.calls
    }
    chosen = []
    seen = set()
    for declaration in declarations:
        if (
            type(declaration) is not ReleasedHandContinuationBinding
            or declaration.program_fingerprint != fingerprint
        ):
            raise ValueError(
                "Released continuation has another declaration/program identity."
            )
        ids = (
            declaration.release_invocation_id,
            declaration.clearance_invocation_id,
            *declaration.park_invocation_ids,
        )
        if seen.intersection(ids):
            raise ValueError("Released continuation occurrences overlap.")
        seen.update(ids)
        if any(
            identity not in occurrences or identity not in presets for identity in ids
        ):
            raise ValueError(
                "Released continuation roles require exact program-bound policies."
            )
        entries = [occurrences[identity] for identity in ids]
        if [e[0] for e in entries] != list(
            range(entries[0][0], entries[0][0] + len(entries))
        ) or any(
            s.parallel_block is not None or s.repeat_frames or len(s.calls) != 1
            for _, s, _ in entries
        ):
            raise ValueError(
                "Released continuation requires consecutive serial non-repeated calls."
            )
        release, clearance, *parks = [entry[2] for entry in entries]
        obj = (
            release.object.entity_id
            if type(release) is Place
            else (
                release.arguments.get("object")
                if type(release) is RegisteredSemanticCall
                and release.call_id in _RELEASES
                else None
            )
        )
        if (
            obj is None
            or type(clearance) is not RegisteredSemanticCall
            or clearance.call_id != "gen_sim.clear_released"
            or set(clearance.arguments) != {"object", "target"}
            or clearance.arguments["object"] != obj
            or set(clearance.resources) != {"primary"}
            or dict(release.resources) != dict(clearance.resources)
            or any(
                type(call) is not RegisteredSemanticCall
                or call.call_id != "simulation.park"
                or call.arguments
                or set(call.resources) != {"primary"}
                for call in parks
            )
            or len({p.resources["primary"] for p in parks}) != len(parks)
            or clearance.resources["primary"]
            not in {p.resources["primary"] for p in parks}
        ):
            raise ValueError(
                "Released continuation object, resources or canonical call roles differ."
            )
        linked = [
            manifest.link_call(call, preset_id=presets[identity])
            for call, identity in zip((release, clearance, *parks), ids, strict=True)
        ]
        selected = [manifest.robot_profile.presets[value.preset_id] for value in linked]
        if (
            selected[0].effect_assurance is not EffectAssurance.VERIFIED
            or release.semantic_id not in selected[0].effect_monitors
        ):
            raise ValueError(
                "Released continuation requires its canonical verified preceding release monitor."
            )
        if (
            type(selected[1].recovery_policy.planning_recovery)
            is not ReleasedHandClearanceRecovery
        ):
            raise ValueError("Released cleanup requires its explicit typed recovery.")
        options = selected[1].action_option_template(clearance.semantic_id)
        motion = selected[1].motion_policy
        if (
            not options.linear_segments
            or motion.strategy != "ik_interp"
            or motion.interpolation_mode != "fixed_grid"
            or motion.ik_seed_mode != "sequential"
            or motion.dynamic_collision_mode != "off"
            or motion.segment_sample_count_retries != (260, 320)
            or motion.segment_continuous_ik_recovery
        ):
            raise ValueError(
                "Released cleanup requires the source fixed-grid Cartesian policy."
            )
        for preset in selected[2:]:
            if (
                type(preset.recovery_policy.planning_recovery)
                is not ReleasedHandParkRecovery
                or preset.effect_assurance is not EffectAssurance.VERIFIED
                or preset.motion_policy.strategy != "ik_interp"
                or preset.motion_policy.dynamic_collision_mode != "off"
            ):
                raise ValueError(
                    "Released Park requires its explicit canonical verified source policy."
                )
        chosen.append(
            _Bound(
                declaration,
                tuple(e[2] for e in entries),
                tuple(v.preset_id for v in linked),
            )
        )
    enabled = set()
    for identity, (_, _, call) in occurrences.items():
        linked = manifest.link_call(call, preset_id=presets.get(identity))
        rule = manifest.robot_profile.presets[
            linked.preset_id
        ].recovery_policy.planning_recovery
        if type(rule) in (ReleasedHandClearanceRecovery, ReleasedHandParkRecovery):
            enabled.add(identity)
    required = {
        identity
        for route in chosen
        for identity in (
            route.declaration.clearance_invocation_id,
            *route.declaration.park_invocation_ids,
        )
    }
    if enabled != required:
        raise ValueError(
            "Every enabled released cleanup/Park requires one exact declared role."
        )
    return tuple(chosen)


class SimulationReleasedHandContinuationProvider:
    """Own canonical compiler authority for one engine lifetime, without a simulator."""

    def __init__(
        self,
        robot: Robot,
        program: CompiledTaskProgram,
        registration: SimulationTaskProgramRegistration,
        *,
        bindings: tuple[ReleasedHandContinuationBinding, ...],
        require_open: Callable[[], None],
        read_source_stamp: Callable[[], PhysicalObservationStamp],
        read_episode: Callable[[], str],
    ) -> None:
        require_open()
        episode = read_episode()
        if type(episode) is not str or not episode or episode != episode.strip():
            raise ValueError(
                "Released continuation requires the actual native episode identity."
            )
        registration.assert_unchanged()
        from ..invocation_policy import task_program_fingerprint

        manifest = SemanticIntegrationManifest(
            scene=registration.catalog.scene,
            robot_profile=registration.robot_profile_binding.declare(),
            call_catalog=registration.call_catalog,
            runtime_preset=program.integration.runtime_preset,
        )
        self._routes = _bind_released_continuations(
            program, manifest, registration.invocation_policy_bindings, bindings
        )
        self._robot, self._program, self._registration = robot, program, registration
        self._fingerprint = task_program_fingerprint(program)
        self._require_owner, self._read_stamp = require_open, read_source_stamp
        self._episode, self._read_episode = episode, read_episode
        self._engine = self._compiler = None
        self._workflows = {}
        self._closed = False

    @property
    def robot(self) -> Robot:
        """Borrow the exact engine robot without inferring an embodiment."""
        return self._robot

    def _open(self):
        self._require_owner()
        if self._closed:
            raise RuntimeError("Released continuation provider is closed.")
        if self._read_episode() != self._episode:
            raise RuntimeError("Released continuation belongs to another episode.")

    def bind_compiler(
        self, engine: AtomicActionEngine, compiler: SemanticCallCompiler
    ) -> None:
        """Bind once to the existing canonical compiler and exact engine owner."""
        self._open()
        if self._compiler is not None:
            raise RuntimeError("Released compiler is already bound.")
        if engine.robot is not self.robot or compiler.integration.engine is not engine:
            raise ValueError("Released compiler/engine ownership differs.")
        self._registration.assert_unchanged()
        from ..invocation_policy import task_program_fingerprint

        if task_program_fingerprint(self._program) != self._fingerprint:
            raise ValueError(
                "Released compiled program changed before compiler binding."
            )
        for entry in self._routes:
            ids = (
                entry.declaration.release_invocation_id,
                entry.declaration.clearance_invocation_id,
                *entry.declaration.park_invocation_ids,
            )
            self._workflows[entry.declaration.clearance_invocation_id] = tuple(
                compiler.analyze(
                    (call,),
                    workflow_id=identity.rsplit(":", 1)[0],
                    preset_ids=(preset,),
                )
                for call, identity, preset in zip(
                    entry.calls, ids, entry.presets, strict=True
                )
            )
        self._engine, self._compiler = engine, compiler

    def _guard(self, request, context):
        self._open()
        if self._compiler is None:
            raise RuntimeError("Released compiler is not bound.")
        self._registration.assert_unchanged()
        if (
            request.binding.owner_id != self._engine.planning_services.binding_owner_id
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.robot.timestamp != context.scene.timestamp
            or context.world_source_stamp is None
            or self._read_stamp() != context.world_source_stamp
        ):
            raise ValueError(
                "Released continuation changed engine rows or native observation scope."
            )

    def _policy(self, request, workflow):
        from ..catalog import _canonical_value

        selected = workflow.calls[0].bound
        preset = selected.preset
        semantic = selected.linked.call.semantic_id
        if any(
            _canonical_value(actual) != _canonical_value(expected)
            for actual, expected in (
                (request.motion_policy, preset.motion_policy),
                (request.recovery_policy, preset.recovery_policy),
                (request.skill_options, preset.action_option_template(semantic)),
            )
        ):
            raise ValueError(
                "Released continuation request differs from its exact selected preset."
            )

    def prepare(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ReleasedHandContinuationSession | None:
        """Resolve an exact current cleanup occurrence, without projecting future state."""
        self._guard(request, context)
        entry = next(
            (
                r
                for r in self._routes
                if r.declaration.clearance_invocation_id == request.invocation_id
            ),
            None,
        )
        if entry is None:
            return None
        self._policy(request, self._workflows[request.invocation_id][1])
        return _Session(self, entry, self._workflows[request.invocation_id])

    def require_park(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> None:
        """Validate exact suffix authority for an actual or projected Park request."""
        self._guard(request, context)
        from embodichain.lab.sim.atomic_actions import JointPositionGoal

        entry = next(
            (
                r
                for r in self._routes
                if request.invocation_id in r.declaration.park_invocation_ids
            ),
            None,
        )
        if (
            entry is None
            or request.skill_id != "move_joints"
            or type(request.goal) is not JointPositionGoal
            or request.goal.target != "park"
        ):
            raise ValueError("Released Park has no exact canonical authority.")
        index = entry.declaration.park_invocation_ids.index(request.invocation_id) + 2
        self._policy(
            request, self._workflows[entry.declaration.clearance_invocation_id][index]
        )
        expected = (
            self._workflows[entry.declaration.clearance_invocation_id][index]
            .calls[0]
            .bound.binding.action_binding
        )
        selected = request.binding.endpoint("primary", "motion")
        reference = expected.endpoint("primary", "motion")
        if (
            selected.target != reference.target
            or selected.resource_id != reference.resource_id
            or selected.task_state_key != reference.task_state_key
            or not selected.commands["park"].equivalent_to(reference.commands["park"])
        ):
            raise ValueError(
                "Released Park endpoint differs from its exact declaration."
            )

    def close(self) -> None:
        """Invalidate borrowed preview sessions without closing native owners."""
        self._closed = True
        self._workflows = {}


class _Session:
    def __init__(self, owner, entry, workflows):
        self.owner, self.entry, self.workflows = owner, entry, workflows

    @property
    def park_bindings(self) -> tuple[ActionBinding, ...]:
        self.owner._open()
        return tuple(
            w.calls[0].bound.binding.action_binding for w in self.workflows[2:]
        )

    def resolve_park(
        self, index: int, context: PlanningContext
    ) -> ResolvedActionRequest:
        self.owner._open()
        if not context.is_preview:
            raise ValueError("Released suffix planning requires a private projection.")
        identity = self.entry.declaration.park_invocation_ids[index]
        grounded = self.owner._compiler.ground(
            self.workflows[index + 2], 0, context, invocation_id=identity
        )
        request = self.owner._engine.resolve(grounded.invocation)
        self.owner.require_park(request, context)
        return request

    def plan_park(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> ActionPlan:
        self.owner.require_park(request, context)
        return self.owner._engine.plan_request(request, context)
