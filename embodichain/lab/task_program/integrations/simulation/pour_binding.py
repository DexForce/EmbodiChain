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

"""Bind declared Pour projection roles to exact program and policy occurrences."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
from embodichain.lab.task_program.semantics import Pick, Place, RegisteredSemanticCall
from embodichain.lab.task_program.semantics.integration import (
    SemanticIntegrationManifest,
)
from embodichain.lab.sim.atomic_actions import PickUpOptions, PourOptions
from .pour_projection import ResolvedPourProjectionRoute
from .targets import AbsolutePoseTarget

if TYPE_CHECKING:
    from embodichain.lab.task_program.compiler import CompiledTaskProgram
    from embodichain.lab.task_program.semantics import (
        RobotSkillProfile,
        SceneManifest,
        SemanticCallCatalog,
    )
    from embodichain.lab.task_program.integrations.extensions import (
        RegisteredSemanticLowererFactory,
    )
    from embodichain.lab.task_program.integrations.invocation_policy import (
        InvocationPolicyBinding,
    )
__all__ = ["PourProjectionBinding", "resolve_pour_projection_routes"]


@dataclass(frozen=True, slots=True)
class PourProjectionBinding:
    """Exact execution roles; geometry and policy operands retain their existing owners."""

    pick_invocation_id: str
    stage_invocation_id: str
    pour_invocation_id: str
    return_invocation_id: str
    place_invocation_id: str

    def __post_init__(self) -> None:
        values = self.invocation_ids
        if (
            any(
                type(value) is not str or not value or value != value.strip()
                for value in values
            )
            or len(set(values)) != 5
        ):
            raise ValueError(
                "Pour projection requires five distinct exact invocation identities."
            )

    @property
    def invocation_ids(self) -> tuple[str, str, str, str, str]:
        """Return declared role identities in their required execution order."""
        return (
            self.pick_invocation_id,
            self.stage_invocation_id,
            self.pour_invocation_id,
            self.return_invocation_id,
            self.place_invocation_id,
        )


def resolve_pour_projection_routes(
    program: CompiledTaskProgram,
    *,
    profile: RobotSkillProfile,
    scene: SceneManifest,
    call_catalog: SemanticCallCatalog,
    factories: tuple[RegisteredSemanticLowererFactory, ...],
    invocation_policies: tuple[InvocationPolicyBinding, ...],
    bindings: tuple[PourProjectionBinding, ...],
) -> tuple[ResolvedPourProjectionRoute, ...]:
    """Resolve five explicit roles without simulation, IK or independent policy defaults.

    Args:
        program: Exact provider-free compiled program.
        profile: Immutable declared robot profile and named presets.
        scene: Immutable canonical semantic scene manifest.
        call_catalog: Registered public call descriptors.
        factories: Declared lowerer factories owning staging and return routes.
        invocation_policies: Program-bound exact occurrence selections.
        bindings: Explicit Pick/stage/Pour/return/Place execution-role declarations.

    Returns:
        Owned runtime operands. Angle/sample count come from the effective future
        Pour preset; return pose and clearance come from existing route owners.
        Invalid chains reject before a native runtime is assembled.
    """
    if not bindings:
        return ()
    from .pour_transport import PourTransportFactory, PourRegionTransportFactory
    from .retained_pour import RetainedPourFactory
    from embodichain.lab.task_program.integrations.invocation_policy import (
        BoundInvocationPolicies,
    )

    policies = BoundInvocationPolicies(program, invocation_policies)
    manifest = SemanticIntegrationManifest(
        scene=scene,
        robot_profile=profile,
        call_catalog=call_catalog,
        runtime_preset=program.integration.runtime_preset,
    )
    policies.validate_manifest(program, manifest)
    selected = {value.invocation_id: value.preset_id for value in invocation_policies}
    segments = tuple(program.iter_segments())
    by_id = {
        f"{program.program_id}/{segment.segment_id}:{call.segment_call_index}": (
            index,
            segment,
            call,
        )
        for index, segment in enumerate(segments)
        for call in segment.calls
    }
    transports = [
        factory for factory in factories if isinstance(factory, PourTransportFactory)
    ]
    pours = [factory for factory in factories if type(factory) is RetainedPourFactory]
    if len(transports) != 1 or len(pours) != 1:
        raise ValueError(
            "Pour projection requires one declared staging/return factory and retained Pour factory."
        )
    factory = transports[0]
    stages = {value.target_id: value for value in factory.staging_routes}
    targets = {value.target_id: value for value in factory.routes}
    result = []
    seen = set()
    for binding in bindings:
        if type(binding) is not PourProjectionBinding:
            raise TypeError("Pour projection bindings must be typed declarations.")
        identities = binding.invocation_ids
        if seen.intersection(identities):
            raise ValueError(
                "Pour projection recipes cannot share execution occurrences."
            )
        seen.update(identities)
        if any(
            identity not in by_id or identity not in selected for identity in identities
        ):
            raise ValueError(
                "Every Pour projection role requires an existing program-bound policy occurrence."
            )
        entries = [by_id[identity] for identity in identities]
        start = entries[0][0]
        if [entry[0] for entry in entries] != list(range(start, start + 5)) or any(
            segment.parallel_block is not None
            or segment.repeat_frames
            or len(segment.calls) != 1
            for _, segment, _ in entries
        ):
            raise ValueError(
                "Pour projection requires five consecutive non-repeated serial calls."
            )
        pick, stage, pour, returned, place = (entry[2].call for entry in entries)
        if (
            type(pick) is not Pick
            or type(stage) is not RegisteredSemanticCall
            or stage.call_id != "simulation.move_held_object"
            or type(pour) is not RegisteredSemanticCall
            or pour.call_id != "simulation.pour"
            or type(returned) is not RegisteredSemanticCall
            or returned.call_id != "simulation.move_held_object"
            or type(place) is not Place
            or place.at is None
        ):
            raise ValueError(
                "Pour projection requires the canonical five-role call sequence."
            )
        subject = pick.object.entity_id
        stage_route = stages.get(stage.arguments.get("target"))
        return_route = targets.get(returned.arguments.get("target"))
        if (
            stage_route is None
            or stage_route.object_id != subject
            or dict(stage.arguments)
            != {
                "object": subject,
                "target": stage_route.target_id,
                "reference": stage_route.receiver_id,
            }
            or dict(pour.arguments) != {"object": subject}
            or set(returned.arguments) != {"object", "target"}
            or returned.arguments["object"] != subject
            or place.object.entity_id != subject
            or return_route is None
            or return_route.object_id != subject
            or type(return_route.pose) is not AbsolutePoseTarget
            or pours[0].object_id != subject
        ):
            raise ValueError(
                "Pour projection participants or existing stage/return route declarations differ."
            )
        if set(pick.resources) != {"primary"} or any(
            dict(call.resources) != dict(pick.resources)
            for call in (stage, pour, returned, place)
        ):
            raise ValueError(
                "Pour projection requires one explicit unchanged resource across all roles."
            )
        linked_pick = manifest.link_call(pick, preset_id=selected[identities[0]])
        pick_options = profile.presets[linked_pick.preset_id].action_option_template(
            pick.semantic_id
        )
        if (
            type(pick_options) is not PickUpOptions
            or pick_options.adaptive_grasp != "pour"
            or pick_options.constrained_grasp
            or not pick_options.qualify_downstream_targets
        ):
            raise ValueError(
                "Pour projection requires the declared unconstrained Pour-purpose Pick option."
            )
        linked_pour = manifest.link_call(pour, preset_id=selected[identities[2]])
        preset = profile.presets[linked_pour.preset_id]
        options = preset.action_option_template(pour.semantic_id)
        if type(options) is not PourOptions:
            raise TypeError("Pour projection requires the selected typed PourOptions.")
        return_pose = return_route.pose.to_matrix()
        if not torch.allclose(
            return_pose,
            place.at.to_matrix(),
            rtol=0,
            atol=8 * torch.finfo(torch.float32).eps,
        ):
            raise ValueError("Pour return transport and Place targets disagree.")
        result.append(
            ResolvedPourProjectionRoute(
                identities[0],
                identities[1],
                identities[2],
                subject,
                stage_route.receiver_id,
                pick.resources["primary"],
                return_pose,
                stage_route.clearance_m,
                float(options.rotate_angle),
                preset.motion_policy.sample_count,
                (
                    factory.proxy_position_tolerance_m
                    if type(factory) is PourRegionTransportFactory
                    else None
                ),
            )
        )
    return tuple(result)
