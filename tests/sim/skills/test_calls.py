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

"""Tests for immutable, declarative semantic call values."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    SkillBindingContract,
    SkillDescriptor,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from embodichain.lab.sim.skills.calls import (
    HandOver,
    Pick,
    Place,
    RegisteredSemanticCall,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SemanticCallSpec,
    SemanticPose,
    builtin_semantic_call_catalog,
)
from embodichain.lab.sim.skills.scene import (
    SceneAffordanceRef,
    SceneEntityRef,
    SceneObjectRef,
)


def _identity_pose() -> SemanticPose:
    return SemanticPose((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def _call_descriptor(
    call_id: str,
    spec_type: type[SemanticCallSpec],
) -> SemanticCallDescriptor:
    if spec_type is not RegisteredSemanticCall:
        return builtin_semantic_call_catalog().discover(call_id)
    target = builtin_semantic_call_catalog().discover("pick").target_descriptor
    assert target is not None
    assert target.binding_contract is not None
    return SemanticCallDescriptor(
        call_id=call_id,
        spec_type=spec_type,
        target_descriptor=target,
    )


def test_semantic_pose_owns_inputs_and_returns_independent_tensors() -> None:
    position = torch.tensor([1.0, 2.0, 3.0])
    quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0])
    pose = SemanticPose(position, quaternion)

    position.zero_()
    quaternion.zero_()
    returned_position = pose.position
    returned_quaternion = pose.quaternion_wxyz
    returned_position.fill_(9.0)
    returned_quaternion.fill_(9.0)

    torch.testing.assert_close(pose.position, torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(
        pose.quaternion_wxyz,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )


def test_semantic_pose_normalizes_wxyz_quaternion() -> None:
    pose = SemanticPose((0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 2.0))

    expected = torch.tensor(
        [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        dtype=torch.float32,
    )
    torch.testing.assert_close(pose.quaternion_wxyz, expected)


def test_semantic_pose_converts_to_homogeneous_matrix() -> None:
    pose = SemanticPose((1.0, 2.0, 3.0), (2.0, 0.0, 0.0, 2.0))

    expected = torch.tensor(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    torch.testing.assert_close(pose.to_matrix(), expected, atol=1.0e-6, rtol=1.0e-6)


@pytest.mark.parametrize(
    "factory",
    (
        pytest.param(
            lambda resources: Pick(
                object=SceneObjectRef("cube"),
                resources=resources,
            ),
            id="pick",
        ),
        pytest.param(
            lambda resources: Place(
                object=SceneObjectRef("cube"),
                at=_identity_pose(),
                resources=resources,
            ),
            id="place",
        ),
        pytest.param(
            lambda resources: HandOver(
                object=SceneObjectRef("cube"),
                resources=resources,
            ),
            id="hand-over",
        ),
        pytest.param(
            lambda resources: RegisteredSemanticCall(
                call_id="vendor.navigate",
                resources=resources,
            ),
            id="registered",
        ),
    ),
)
def test_semantic_calls_snapshot_and_freeze_resources(
    factory: Callable[[Mapping[str, str]], SemanticCallSpec],
) -> None:
    source = {"actor": "left_arm"}
    call = factory(source)

    source["actor"] = "right_arm"

    assert call.resources == {"actor": "left_arm"}
    with pytest.raises(TypeError):
        call.resources["actor"] = "right_arm"  # type: ignore[index]


def test_pick_requires_typed_object_and_affordance_references() -> None:
    with pytest.raises(TypeError, match="Pick.object"):
        Pick(object=SceneEntityRef("cube"))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Pick.grasp"):
        Pick(
            object=SceneObjectRef("cube"),
            grasp=SceneObjectRef("cube.grasp"),  # type: ignore[arg-type]
        )


def test_place_requires_exactly_one_destination() -> None:
    object_ref = SceneObjectRef("cube")

    with pytest.raises(ValueError, match="exactly one"):
        Place(object=object_ref)
    with pytest.raises(ValueError, match="exactly one"):
        Place(
            object=object_ref,
            at=_identity_pose(),
            on=SceneObjectRef("table"),
        )


def test_place_snapshots_absolute_destination_pose() -> None:
    destination = _identity_pose()

    call = Place(object=SceneObjectRef("cube"), at=destination)

    assert call.at is not destination
    assert call.at is not None
    torch.testing.assert_close(call.at.to_matrix(), destination.to_matrix())


def test_handover_uses_destination_resource_selection() -> None:
    call = HandOver(
        object=SceneObjectRef("cube"),
        resources={"destination": "right_actor"},
    )

    assert call.resources == {"destination": "right_actor"}


def test_handover_snapshots_optional_final_target() -> None:
    final_target = _identity_pose()

    call = HandOver(
        object=SceneObjectRef("cube"),
        final_target=final_target,
    )

    assert call.final_target is not final_target
    assert call.final_target is not None
    torch.testing.assert_close(call.final_target.to_matrix(), final_target.to_matrix())


def test_registered_call_recursively_snapshots_declarative_arguments() -> None:
    step = {"object": SceneObjectRef("cube")}
    steps = [step]
    pose = _identity_pose()
    arguments = {"steps": steps, "target": pose}

    call = RegisteredSemanticCall(
        call_id="vendor.navigate",
        arguments=arguments,
    )
    step["object"] = SceneObjectRef("changed")
    steps.append({"object": SceneObjectRef("extra")})

    saved_steps = call.arguments["steps"]
    assert isinstance(saved_steps, tuple)
    assert len(saved_steps) == 1
    assert saved_steps[0] == {"object": SceneObjectRef("cube")}
    saved_target = call.arguments["target"]
    assert isinstance(saved_target, SemanticPose)
    assert saved_target is not pose
    with pytest.raises(TypeError):
        call.arguments["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "unsafe_value",
    (
        pytest.param(lambda: None, id="callable"),
        pytest.param(torch.tensor([1.0]), id="tensor"),
        pytest.param(object(), id="live-object"),
    ),
)
def test_registered_call_rejects_executable_or_live_payloads(
    unsafe_value: object,
) -> None:
    with pytest.raises(TypeError, match="non-declarative"):
        RegisteredSemanticCall(
            call_id="vendor.navigate",
            arguments={"unsafe": unsafe_value},
        )


def test_registered_call_rejects_non_finite_payload_numbers() -> None:
    with pytest.raises(ValueError, match="finite"):
        RegisteredSemanticCall(
            call_id="vendor.navigate",
            arguments={"speed": float("nan")},
        )


@pytest.mark.parametrize(
    "call_id",
    (".", "vendor.", ".inspect", "vendor..inspect", "Vendor.inspect"),
)
def test_registered_call_rejects_malformed_namespace(call_id: str) -> None:
    with pytest.raises(ValueError, match="segments"):
        RegisteredSemanticCall(call_id=call_id)


def test_registered_call_rejects_cyclic_payload() -> None:
    payload: dict[str, object] = {}
    payload["self"] = payload

    with pytest.raises(ValueError, match="cyclic"):
        RegisteredSemanticCall(
            call_id="vendor.inspect",
            arguments=payload,
        )


def test_registered_call_rejects_string_subclass_identifier() -> None:
    class LiveString(str):
        live_handle = object()

    with pytest.raises(ValueError, match="non-empty string"):
        RegisteredSemanticCall(call_id=LiveString("vendor.inspect"))


def test_semantic_call_catalog_discovers_without_mutable_runtime_state() -> None:
    pick_descriptor = _call_descriptor(Pick.call_kind, Pick)
    catalog = SemanticCallCatalog([pick_descriptor])

    assert catalog.discover("pick") is pick_descriptor
    assert catalog.discover(Pick(object=SceneObjectRef("cube"))) is pick_descriptor
    with pytest.raises(TypeError):
        catalog.descriptors["other"] = pick_descriptor  # type: ignore[index]


def test_semantic_call_catalog_extension_does_not_mutate_original() -> None:
    pick_descriptor = _call_descriptor(Pick.call_kind, Pick)
    extension = _call_descriptor("vendor.navigate", RegisteredSemanticCall)
    original = SemanticCallCatalog([pick_descriptor])

    extended = original.with_descriptor(extension)

    with pytest.raises(KeyError, match="Unknown semantic call"):
        original.discover("vendor.navigate")
    assert (
        extended.discover(RegisteredSemanticCall(call_id="vendor.navigate"))
        is extension
    )


def test_semantic_call_catalog_rejects_duplicate_ids() -> None:
    descriptor = _call_descriptor(Pick.call_kind, Pick)

    with pytest.raises(ValueError, match="Duplicate semantic call ID"):
        SemanticCallCatalog([descriptor, descriptor])


def test_catalog_rejects_executable_call_subclasses() -> None:
    class UnsafeRegisteredCall(RegisteredSemanticCall):
        pass

    with pytest.raises(TypeError, match="exactly"):
        SemanticCallDescriptor(
            call_id="vendor.unsafe",
            spec_type=UnsafeRegisteredCall,
        )


def test_registered_payload_rejects_value_subclasses() -> None:
    class LiveInteger(int):
        live_handle = object()

    with pytest.raises(TypeError, match="non-declarative"):
        RegisteredSemanticCall(
            call_id="vendor.unsafe",
            arguments={"value": LiveInteger(1)},
        )


def test_builtin_descriptor_target_cannot_be_remapped() -> None:
    target = builtin_semantic_call_catalog().discover("pick").target_descriptor
    assert target is not None
    remapped = SkillDescriptor(
        skill_id="move_joints",
        goal_type=target.goal_type,
        options_type=target.options_type,
        binding_contract=SkillBindingContract(),
    )

    with pytest.raises(ValueError, match="exact curated"):
        SemanticCallDescriptor(
            call_id=Pick.call_kind,
            spec_type=Pick,
            target_descriptor=remapped,
        )


def test_catalog_rejects_descriptor_subclass_with_live_state() -> None:
    class LiveDescriptor(SemanticCallDescriptor):
        live_handle = object()

    source = _call_descriptor("vendor.inspect", RegisteredSemanticCall)
    descriptor = LiveDescriptor(
        call_id=source.call_id,
        spec_type=source.spec_type,
        target_descriptor=source.target_descriptor,
    )

    with pytest.raises(TypeError, match="exact SemanticCallDescriptor"):
        SemanticCallCatalog((descriptor,))


def test_descriptor_rejects_runtime_bearing_binding_contract_subclasses() -> None:
    class LiveSlot(SkillResourceSlot):
        live_handle = object()

    target = builtin_semantic_call_catalog().discover("pick").target_descriptor
    assert target is not None
    endpoint = SkillEndpointRequirement("motion")
    contract = SkillBindingContract(slots=(LiveSlot("primary", (endpoint,)),))
    remapped_target = SkillDescriptor(
        skill_id=target.skill_id,
        goal_type=target.goal_type,
        options_type=target.options_type,
        binding_contract=contract,
    )

    with pytest.raises(TypeError, match="exact SkillResourceSlot"):
        SemanticCallDescriptor(
            call_id="vendor.inspect",
            spec_type=RegisteredSemanticCall,
            target_descriptor=remapped_target,
        )


def test_descriptor_rejects_target_descriptor_subclass() -> None:
    class LiveTarget(SkillDescriptor):
        live_handle = object()

    target = builtin_semantic_call_catalog().discover("pick").target_descriptor
    assert target is not None
    live_target = LiveTarget(
        skill_id=target.skill_id,
        goal_type=target.goal_type,
        options_type=target.options_type,
        agent_visible=target.agent_visible,
        binding_contract=target.binding_contract,
    )
    assert target.binding_contract is not None

    with pytest.raises(TypeError, match="exactly SkillDescriptor"):
        SemanticCallDescriptor(
            call_id="vendor.inspect",
            spec_type=RegisteredSemanticCall,
            target_descriptor=live_target,
        )
