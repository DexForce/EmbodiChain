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

"""Closed task-specific DemoSkill calls and occurrence-bound source declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from typing import ClassVar

from embodichain.data_pipeline.real2sim import load_demo_skill
from embodichain.data_pipeline.real2sim.demo_skill import _sha256
from embodichain.lab.sim.atomic_actions import (
    ExecuteDemoSkill,
    DemoSkillGoal,
    ActionOptions,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics.calls import RegisteredSemanticCall
from embodichain.lab.task_program.semantics.scene import SceneObjectRef

__all__: list[str] = []


@dataclass(frozen=True)
class _ConfiguredDemoSkillSource:
    call_index: int
    program_fingerprint: str
    asset_manifest: str
    asset_sha256: str
    call_id: str
    contact_sensor_uid: str | None = None
    tcp_contact_links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.call_index) is not int or self.call_index < 0:
            raise ValueError("DemoSkill call_index must be nonnegative")
        _sha256(self.program_fingerprint, "program_fingerprint")
        _sha256(self.asset_sha256, "asset_sha256")
        if not isinstance(self.asset_manifest, str) or not self.asset_manifest:
            raise ValueError("DemoSkill asset_manifest is required")
        _call_id(self.call_id)
        if self.contact_sensor_uid is not None and (
            not isinstance(self.contact_sensor_uid, str) or not self.contact_sensor_uid
        ):
            raise ValueError("contact_sensor_uid must be a nonempty identifier")
        if type(self.tcp_contact_links) is not tuple or any(
            not isinstance(name, str) or not name for name in self.tcp_contact_links
        ):
            raise ValueError("tcp_contact_links must be explicit link names")

    def validate_source(self) -> None:
        artifact = load_demo_skill(self.asset_manifest)
        if artifact.content_sha256 != self.asset_sha256:
            raise ValueError("configured DemoSkill asset content has changed")
        contacts = [
            item
            for item in artifact.spec.constraints
            if item.kind in {"contact", "no_contact"}
        ]
        if contacts and self.contact_sensor_uid is None:
            raise ValueError(
                "DemoSkill contact constraints require a configured sensor"
            )
        if (
            any("tcp" in {item.subject, item.other} for item in contacts)
            and not self.tcp_contact_links
        ):
            raise ValueError(
                "TCP contact constraints require explicit robot contact links"
            )


def _call_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+", value) is None
    ):
        raise ValueError("DemoSkill call_id must be a lowercase dotted identifier")


@dataclass(frozen=True, slots=True)
class _DemoSkillLowererFactory:
    """Immutable allowlisted factory with a task-specific class-owned call ID."""

    asset_sha256: str
    role_ids: tuple[str, ...]
    terminal_held_role: str | None = None
    release_on_exit: bool = False
    released_role: str | None = None
    call_id: ClassVar[str] = "simulation.demo_skill"
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar = ExecuteDemoSkill.descriptor()

    def create(self, *, simulation, robot, scene_registry, engine):
        if engine.robot is not robot:
            raise ValueError("DemoSkill lowerer requires the registered engine robot")
        owner = self

        class _Lowerer(RegisteredSemanticLowerer):
            call_id = owner.call_id
            target_descriptor = ExecuteDemoSkill.descriptor()
            preserves_symbolic_state = (
                owner.terminal_held_role is None and not owner.release_on_exit
            )
            from embodichain.lab.task_program.semantics.effects import (
                SemanticEffectKind,
            )

            effect_contract_kind = (
                SemanticEffectKind.ATTACH
                if owner.terminal_held_role is not None
                else (SemanticEffectKind.RELEASE if owner.release_on_exit else None)
            )

            def lower(self, call, context, *, bound, option_template):
                if type(option_template) is not ActionOptions:
                    raise TypeError("ExecuteDemoSkill uses ActionOptions")
                if set(call.arguments) != set(owner.role_ids):
                    raise ValueError(
                        "DemoSkill arguments must bind exactly its declared roles"
                    )
                roles, uids = {}, {}
                for name, identifier in call.arguments.items():
                    ref = scene_registry.resolve(
                        identifier, expected_type=SceneObjectRef
                    )
                    registration = scene_registry.lookup(ref)
                    entity = registration.state_provider.entity
                    if simulation.get_rigid_object(entity.uid) is not entity:
                        raise ValueError(
                            "DemoSkill roles must belong to the registered simulation"
                        )
                    roles[name], uids[name] = ref.entity_id, entity.uid
                semantics = None
                registered_effect = None
                if owner.terminal_held_role is not None or owner.release_on_exit:
                    from embodichain.lab.task_program.compiler.lowering import (
                        RegisteredSemanticEffect,
                        RegisteredHeldObjectEffect,
                    )
                    from embodichain.lab.task_program.semantics.effects import (
                        SemanticEffectKind,
                        HeldObjectRelation,
                    )

                    role = owner.terminal_held_role
                    if role is None:
                        role = owner.released_role
                        if role is None:
                            raise ValueError(
                                "release_on_exit requires an explicit terminal object role"
                            )
                    if owner.terminal_held_role is not None:
                        semantics = scene_registry.object_semantics(
                            scene_registry.resolve(
                                roles[role], expected_type=SceneObjectRef
                            )
                        )
                    registered_effect = RegisteredSemanticEffect(
                        effect_kind=(
                            SemanticEffectKind.ATTACH
                            if owner.terminal_held_role
                            else SemanticEffectKind.RELEASE
                        ),
                        held_objects=(
                            RegisteredHeldObjectEffect(
                                expectation_id="primary",
                                relation=(
                                    HeldObjectRelation.ATTACHED
                                    if owner.terminal_held_role
                                    else HeldObjectRelation.DETACHED
                                ),
                                object_id=roles[role],
                                slot_id="primary",
                                allow_missing_detached_baseline=owner.release_on_exit,
                            ),
                        ),
                    )
                return SemanticLowering(
                    goal=DemoSkillGoal(owner.asset_sha256, roles, uids, semantics),
                    registered_effect=registered_effect,
                )

        return _Lowerer()


@lru_cache(maxsize=256)
def _lowerer_type(call_id: str) -> type[_DemoSkillLowererFactory]:
    _call_id(call_id)
    # Only the validated call identity varies; no arbitrary import or class path
    # is accepted from configuration. All factories share the same closed code.
    return type(
        "_DemoSkill_" + call_id.replace(".", "__"),
        (_DemoSkillLowererFactory,),
        {"call_id": call_id, "__module__": __name__},
    )


def _decode_demo_skill_lowerer(value: Mapping) -> _DemoSkillLowererFactory:
    if set(value) != {"kind", "call_id", "asset_manifest", "asset_sha256"}:
        raise ValueError(
            "DemoSkill lowerer requires kind, call_id, asset_manifest and asset_sha256"
        )
    artifact = load_demo_skill(value["asset_manifest"])
    if artifact.content_sha256 != value["asset_sha256"]:
        raise ValueError("DemoSkill lowerer asset digest mismatch")
    spec = artifact.spec
    return _lowerer_type(value["call_id"])(
        artifact.content_sha256,
        tuple(role.role_id for role in spec.roles),
        spec.terminal_held_role,
        spec.release_on_exit,
        next(
            (role.role_id for role in spec.roles if role.reference_kind == "held"), None
        ),
    )
