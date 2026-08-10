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

"""Tests for authoritative semantic-scene registrations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    AntipodalAffordance,
    EntityState,
    SceneSnapshot,
)
from embodichain.lab.sim.skills import (
    AmbiguousSceneAffordanceError,
    GRASP_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneEntityRegistration,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
    UnsupportedSceneAffordanceError,
)


class _StateProvider:
    """Return one fixed identity pose for registration validation."""

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp
        return EntityState(torch.eye(4).repeat(env_ids.numel(), 1, 1))


class _GeometryProvider:
    """Return one opaque collision-geometry descriptor."""

    def get_geometry(self) -> object:
        return {"kind": "box"}


class _EmptyGeometryProvider:
    """Satisfy the geometry protocol but fail to materialize a descriptor."""

    def get_geometry(self) -> object:
        return None


class _MutableStateProvider:
    """Expose a mutable pose while recording provider calls."""

    def __init__(self, pose: torch.Tensor) -> None:
        self.pose = pose
        self.calls = 0

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp, env_ids
        self.calls += 1
        return EntityState(self.pose)


class _MotionGenerator:
    """Minimal dynamic-collision integration surface."""

    def __init__(
        self,
        *,
        entity_ids: tuple[str, ...],
        world_entity_ids: tuple[str, ...] | None = None,
        supports_updates: bool = True,
        batch_mode: str | None = "per_env",
    ) -> None:
        self.dynamic_collision_entity_ids = entity_ids
        self.collision_world_entity_ids = (
            entity_ids if world_entity_ids is None else world_entity_ids
        )
        self.supports_dynamic_collision_world = supports_updates
        self.collision_world_batch_mode = batch_mode


class _ExternalSceneProvider:
    """External provider with an explicit concrete collision declaration."""

    def __init__(self, entity_ids: tuple[str, ...]) -> None:
        self.collision_entity_ids = entity_ids

    def snapshot(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> SceneSnapshot:
        del timestamp, env_ids
        raise NotImplementedError


class _SimulationEntity:
    """Simulation entity pose source used by the opt-in adapter tests."""

    def __init__(self, pose: torch.Tensor) -> None:
        self.pose = pose

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix is True
        return self.pose


class _Simulation:
    """Minimal simulation lookup surface with selected and unselected assets."""

    def __init__(self) -> None:
        self.rigid_objects = {
            "sim_cube": _SimulationEntity(torch.eye(4)),
            "ignored": _SimulationEntity(torch.eye(4) * 2.0),
        }
        self.articulations = {
            "sim_drawer": _SimulationEntity(torch.eye(4)),
        }

    def get_rigid_object(self, uid: str) -> _SimulationEntity | None:
        return self.rigid_objects.get(uid)

    def get_articulation(self, uid: str) -> _SimulationEntity | None:
        return self.articulations.get(uid)


class _CopyTrackedAffordance(AntipodalAffordance):
    """Count payload copies so metadata projection can prove it performs none."""

    copies = 0

    def __deepcopy__(self, memo: dict[int, object]) -> _CopyTrackedAffordance:
        del memo
        type(self).copies += 1
        return _CopyTrackedAffordance()


class _SelfCopyAffordance(AntipodalAffordance):
    """Malicious payload that violates deepcopy ownership."""

    def __deepcopy__(self, memo: dict[int, object]) -> _SelfCopyAffordance:
        del memo
        return self


@pytest.mark.parametrize("entity_id", ["", " cube", "cube "])
def test_scene_entity_ref_rejects_non_exact_identifier(entity_id: str) -> None:
    with pytest.raises(ValueError, match="entity_id"):
        SceneObjectRef(entity_id)


def test_scene_entity_refs_are_typed_and_immutable() -> None:
    object_ref = SceneObjectRef("cube")

    assert object_ref != SceneArticulationRef("cube")
    with pytest.raises(FrozenInstanceError):
        object_ref.entity_id = "other"  # type: ignore[misc]


def test_registration_normalizes_self_alias_without_rewriting_names() -> None:
    registration = SceneEntityRegistration(
        ref=SceneObjectRef("cube"),
        state_provider=_StateProvider(),
        aliases=("cube", "sim_cube"),
    )

    assert registration.aliases == ("sim_cube",)


def test_registration_rejects_duplicate_aliases() -> None:
    with pytest.raises(ValueError, match="aliases"):
        SceneEntityRegistration(
            ref=SceneObjectRef("cube"),
            state_provider=_StateProvider(),
            aliases=("sim_cube", "sim_cube"),
        )


def test_registration_rejects_string_as_alias_collection() -> None:
    with pytest.raises(TypeError, match="aliases.*not a string"):
        SceneEntityRegistration(
            ref=SceneObjectRef("cube"),
            state_provider=_StateProvider(),
            aliases="sim_cube",  # type: ignore[arg-type]
        )


def test_root_registration_requires_explicit_state_provider() -> None:
    with pytest.raises(ValueError, match="state_provider"):
        SceneEntityRegistration(ref=SceneObjectRef("cube"))


def test_link_registration_requires_parent_and_native_name() -> None:
    with pytest.raises(ValueError, match="parent and native_name"):
        SceneEntityRegistration(
            ref=SceneLinkRef("drawer_handle_link"),
            state_provider=_StateProvider(),
        )


def test_affordance_registration_owns_parent_relation_and_pose() -> None:
    relative_pose = torch.eye(4)
    registration = SceneEntityRegistration(
        ref=SceneAffordanceRef("drawer_handle"),
        parent=SceneLinkRef("drawer_handle_link"),
        native_name="handle",
        affordance=Affordance(),
        relative_pose=relative_pose,
    )
    relative_pose.fill_(4.0)

    assert registration.relative_pose is not None
    assert torch.equal(registration.relative_pose, torch.eye(4))


def test_affordance_registration_rejects_two_pose_sources() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SceneEntityRegistration(
            ref=SceneAffordanceRef("drawer_handle"),
            state_provider=_StateProvider(),
            parent=SceneLinkRef("drawer_handle_link"),
            native_name="handle",
            affordance=Affordance(),
            relative_pose=torch.eye(4),
        )


def test_grasp_capability_requires_typed_versioned_payload() -> None:
    object_ref = SceneObjectRef("cube")
    common = {
        "ref": SceneAffordanceRef("cube_grasp"),
        "parent": object_ref,
        "native_name": "grasp",
        "relative_pose": torch.eye(4),
        "affordance_capabilities": frozenset({GRASP_AFFORDANCE_CAPABILITY}),
    }

    with pytest.raises(TypeError, match="AntipodalAffordance"):
        SceneEntityRegistration(
            **common,
            affordance=Affordance(),
            affordance_revision="v1",
        )
    with pytest.raises(ValueError, match="affordance_revision"):
        SceneEntityRegistration(
            **common,
            affordance=AntipodalAffordance(),
        )


def test_registry_selects_only_explicit_scoped_affordance_default() -> None:
    object_ref = SceneObjectRef("cube")
    first = SceneAffordanceRef("first_grasp")
    second = SceneAffordanceRef("second_grasp")

    def registrations(*, with_default: bool) -> tuple[SceneEntityRegistration, ...]:
        return (
            SceneEntityRegistration(
                ref=object_ref,
                state_provider=_StateProvider(),
                default_affordances=(
                    {GRASP_AFFORDANCE_CAPABILITY: second} if with_default else {}
                ),
            ),
            *tuple(
                SceneEntityRegistration(
                    ref=ref,
                    parent=object_ref,
                    native_name=ref.entity_id,
                    affordance=AntipodalAffordance(),
                    affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                    affordance_revision="v1",
                    relative_pose=torch.eye(4),
                )
                for ref in (first, second)
            ),
        )

    ambiguous = SceneRegistry(registrations(with_default=False))
    with pytest.raises(AmbiguousSceneAffordanceError, match="multiple"):
        ambiguous.resolve_affordance(
            object_ref,
            capability=GRASP_AFFORDANCE_CAPABILITY,
        )
    with pytest.raises(UnsupportedSceneAffordanceError, match="no affordance"):
        ambiguous.resolve_affordance(
            object_ref,
            capability="affordance.unknown",
        )

    registry = SceneRegistry(registrations(with_default=True))
    assert (
        registry.resolve_affordance(
            object_ref,
            capability=GRASP_AFFORDANCE_CAPABILITY,
        )
        == second
    )
    assert (
        registry.resolve_affordance(
            object_ref,
            capability=GRASP_AFFORDANCE_CAPABILITY,
            explicit=first,
        )
        == first
    )


def test_registry_metadata_projection_does_not_copy_affordance_payload() -> None:
    object_ref = SceneObjectRef("cube")
    _CopyTrackedAffordance.copies = 0
    registry = SceneRegistry(
        (
            SceneEntityRegistration(ref=object_ref, state_provider=_StateProvider()),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("cube_grasp"),
                parent=object_ref,
                native_name="grasp",
                affordance=_CopyTrackedAffordance(),
                affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                affordance_revision="v1",
                relative_pose=torch.eye(4),
            ),
        )
    )
    _CopyTrackedAffordance.copies = 0

    metadata = registry.entity_metadata

    assert metadata[1].affordance_payload_type is _CopyTrackedAffordance
    assert _CopyTrackedAffordance.copies == 0


def test_registry_rejects_affordance_that_cannot_produce_owned_copy() -> None:
    cube = SceneObjectRef("cube")

    with pytest.raises(TypeError, match="distinct value"):
        SceneRegistry(
            (
                SceneEntityRegistration(ref=cube, state_provider=_StateProvider()),
                SceneEntityRegistration(
                    ref=SceneAffordanceRef("cube_grasp"),
                    parent=cube,
                    native_name="grasp",
                    affordance=_SelfCopyAffordance(),
                    relative_pose=torch.eye(4),
                ),
            )
        )


def test_registry_builds_owned_object_semantics_from_direct_child() -> None:
    cube = SceneObjectRef("cube")
    table = SceneObjectRef("table")
    cube_grasp = SceneAffordanceRef("cube_grasp")
    table_grasp = SceneAffordanceRef("table_grasp")
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=cube,
                state_provider=_StateProvider(),
                semantic_type="cube",
            ),
            SceneEntityRegistration(ref=table, state_provider=_StateProvider()),
            SceneEntityRegistration(
                ref=cube_grasp,
                parent=cube,
                native_name="grasp",
                affordance=AntipodalAffordance(),
                relative_pose=torch.eye(4),
            ),
            SceneEntityRegistration(
                ref=table_grasp,
                parent=table,
                native_name="grasp",
                affordance=AntipodalAffordance(),
                relative_pose=torch.eye(4),
            ),
        )
    )

    first = registry.object_semantics(cube, affordance=cube_grasp)
    second = registry.object_semantics("cube", affordance="cube_grasp")

    assert first.entity_id == "cube"
    assert first.label == "cube"
    assert first.affordance is not second.affordance
    first.affordance.custom_config["mutated"] = True
    assert "mutated" not in second.affordance.custom_config
    with pytest.raises(ValueError, match="not a direct child"):
        registry.object_semantics(cube, affordance=table_grasp)


def test_collision_registration_requires_geometry_provider() -> None:
    with pytest.raises(ValueError, match="geometry_provider"):
        SceneEntityRegistration(
            ref=SceneObjectRef("obstacle"),
            state_provider=_StateProvider(),
            collision_role=SceneCollisionRole.DYNAMIC,
        )

    registration = SceneEntityRegistration(
        ref=SceneObjectRef("obstacle"),
        state_provider=_StateProvider(),
        geometry_provider=_GeometryProvider(),
        collision_role=SceneCollisionRole.DYNAMIC,
    )
    assert registration.geometry_provider is not None


def test_registry_resolves_aliases_to_typed_canonical_refs() -> None:
    cube_ref = SceneObjectRef("cube")
    drawer_ref = SceneArticulationRef("drawer")
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=cube_ref,
                state_provider=_StateProvider(),
                aliases=("sim_cube",),
            ),
            SceneEntityRegistration(
                ref=drawer_ref,
                state_provider=_StateProvider(),
                aliases=("sim_drawer",),
            ),
        )
    )

    assert registry.resolve("sim_cube", expected_type=SceneObjectRef) is cube_ref
    assert registry.lookup("sim_drawer").ref is drawer_ref
    assert registry.aliases == {
        "sim_cube": "cube",
        "sim_drawer": "drawer",
    }

    with pytest.raises(TypeError, match="SceneObjectRef"):
        registry.resolve("sim_cube", expected_type=SceneArticulationRef)
    with pytest.raises(TypeError, match="SceneArticulationRef"):
        registry.resolve(SceneArticulationRef("cube"))


def test_registry_enforces_one_flat_global_id_namespace() -> None:
    registrations = (
        SceneEntityRegistration(
            ref=SceneObjectRef("shared"),
            state_provider=_StateProvider(),
        ),
        SceneEntityRegistration(
            ref=SceneArticulationRef("shared"),
            state_provider=_StateProvider(),
        ),
    )

    with pytest.raises(ValueError, match="Duplicate canonical"):
        SceneRegistry(registrations)


def test_registry_rejects_alias_collision_with_canonical_id() -> None:
    with pytest.raises(ValueError, match="collides with canonical"):
        SceneRegistry(
            (
                SceneEntityRegistration(
                    ref=SceneObjectRef("cube"),
                    state_provider=_StateProvider(),
                    aliases=("drawer",),
                ),
                SceneEntityRegistration(
                    ref=SceneArticulationRef("drawer"),
                    state_provider=_StateProvider(),
                ),
            )
        )


def test_registry_rejects_ambiguous_aliases_across_types() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        SceneRegistry(
            (
                SceneEntityRegistration(
                    ref=SceneObjectRef("cube"),
                    state_provider=_StateProvider(),
                    aliases=("legacy",),
                ),
                SceneEntityRegistration(
                    ref=SceneArticulationRef("drawer"),
                    state_provider=_StateProvider(),
                    aliases=("legacy",),
                ),
            )
        )


def test_registry_requires_registered_exact_typed_parent() -> None:
    link_registration = SceneEntityRegistration(
        ref=SceneLinkRef("drawer_link"),
        parent=SceneArticulationRef("drawer"),
        native_name="link",
        state_provider=_StateProvider(),
    )

    with pytest.raises(ValueError, match="unregistered parent"):
        SceneRegistry((link_registration,))
    with pytest.raises(TypeError, match="registered as SceneObjectRef"):
        SceneRegistry(
            (
                SceneEntityRegistration(
                    ref=SceneObjectRef("drawer"),
                    state_provider=_StateProvider(),
                ),
                link_registration,
            )
        )


@pytest.mark.parametrize(
    "ref_type",
    [SceneLinkRef, SceneAffordanceRef],
)
def test_registry_rejects_duplicate_parent_native_member(ref_type: type) -> None:
    parent = SceneArticulationRef("drawer")

    def member_registration(entity_id: str) -> SceneEntityRegistration:
        if ref_type is SceneLinkRef:
            return SceneEntityRegistration(
                ref=SceneLinkRef(entity_id),
                parent=parent,
                native_name="handle",
                state_provider=_StateProvider(),
            )
        return SceneEntityRegistration(
            ref=SceneAffordanceRef(entity_id),
            parent=parent,
            native_name="handle",
            affordance=Affordance(),
            relative_pose=torch.eye(4),
        )

    with pytest.raises(ValueError, match="native_name.*already registered"):
        SceneRegistry(
            (
                SceneEntityRegistration(
                    ref=parent,
                    state_provider=_StateProvider(),
                ),
                member_registration("first"),
                member_registration("second"),
            )
        )


def test_registry_is_structurally_immutable_and_owns_relative_pose() -> None:
    parent = SceneObjectRef("drawer")
    relative_pose = torch.eye(4)
    affordance_registration = SceneEntityRegistration(
        ref=SceneAffordanceRef("handle"),
        parent=parent,
        native_name="handle",
        affordance=Affordance(),
        relative_pose=relative_pose,
    )
    registrations = [
        SceneEntityRegistration(
            ref=parent,
            state_provider=_StateProvider(),
        ),
        affordance_registration,
    ]
    registry = SceneRegistry(registrations)

    registrations.clear()
    relative_pose.fill_(3.0)
    assert len(registry) == 2
    returned_pose = registry.lookup("handle").relative_pose
    assert returned_pose is not None
    assert torch.equal(returned_pose, torch.eye(4))
    returned_pose.fill_(5.0)
    assert torch.equal(registry.lookup("handle").relative_pose, torch.eye(4))
    with pytest.raises(TypeError):
        registry.aliases["new"] = "drawer"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registry.collision_world_mode = SceneCollisionWorldMode.SHARED  # type: ignore[misc]


def test_registry_owns_and_defensively_copies_affordance_metadata() -> None:
    parent = SceneObjectRef("drawer")
    affordance = Affordance(custom_config={"limits": {"opening": 0.3}})
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=parent,
                state_provider=_StateProvider(),
            ),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("handle"),
                parent=parent,
                native_name="handle",
                affordance=affordance,
                relative_pose=torch.eye(4),
            ),
        )
    )

    affordance.custom_config["limits"]["opening"] = 0.8
    public_affordance = registry.lookup("handle").affordance
    assert public_affordance is not None
    assert public_affordance.custom_config["limits"]["opening"] == 0.3

    public_affordance.custom_config["limits"]["opening"] = 1.0
    second_read = registry.lookup("handle").affordance
    assert second_read is not None
    assert second_read.custom_config["limits"]["opening"] == 0.3


def test_registry_provider_uses_canonical_ids_and_derives_relative_pose() -> None:
    parent_pose = torch.eye(4).repeat(2, 1, 1)
    parent_pose[:, 0, 3] = torch.tensor([1.0, 2.0])
    relative_pose = torch.eye(4)
    relative_pose[1, 3] = 0.25
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("drawer"),
                state_provider=_MutableStateProvider(parent_pose),
                aliases=("sim_drawer",),
            ),
            SceneEntityRegistration(
                ref=SceneAffordanceRef("handle"),
                parent=SceneObjectRef("drawer"),
                native_name="handle",
                affordance=Affordance(),
                relative_pose=relative_pose,
            ),
        )
    )

    snapshot = registry.make_scene_provider().snapshot(
        timestamp=0.0,
        env_ids=torch.tensor([10, 20], dtype=torch.long),
    )

    assert set(snapshot.entities) == {"drawer", "handle"}
    assert "sim_drawer" not in snapshot.entities
    assert torch.equal(
        snapshot.entities["handle"].pose,
        torch.matmul(parent_pose, relative_pose),
    )


def test_registry_providers_have_independent_revisions() -> None:
    state_provider = _MutableStateProvider(torch.eye(4))
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=state_provider,
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )
    first_provider = registry.make_scene_provider()
    second_provider = registry.make_scene_provider()
    env_ids = torch.tensor([0, 1], dtype=torch.long)
    first_provider.snapshot(timestamp=0.0, env_ids=env_ids)
    moved = torch.eye(4).repeat(2, 1, 1)
    moved[1, 0, 3] = 0.1
    state_provider.pose = moved

    changed = first_provider.snapshot(timestamp=1.0, env_ids=env_ids)
    independent_initial = second_provider.snapshot(timestamp=1.0, env_ids=env_ids)

    assert changed.version == 1
    assert changed.collision_world_revisions(2) == (0, 1)
    assert independent_initial.version == 0
    assert independent_initial.collision_world_revisions(2) == (0, 0)


def test_registry_provider_accumulates_subthreshold_motion_per_row() -> None:
    pose = torch.eye(4).repeat(2, 1, 1)
    state_provider = _MutableStateProvider(pose)
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=state_provider,
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )
    provider = registry.make_scene_provider(translation_threshold=0.01)
    env_ids = torch.tensor([4, 8], dtype=torch.long)
    provider.snapshot(timestamp=0.0, env_ids=env_ids)
    first_motion = pose.clone()
    first_motion[1, 0, 3] = 0.006
    state_provider.pose = first_motion

    below_threshold = provider.snapshot(timestamp=1.0, env_ids=env_ids)
    second_motion = first_motion.clone()
    second_motion[1, 0, 3] = 0.012
    state_provider.pose = second_motion
    accumulated_change = provider.snapshot(timestamp=2.0, env_ids=env_ids)

    assert below_threshold.version == 0
    assert below_threshold.collision_world_revisions(2) == (0, 0)
    assert accumulated_change.version == 1
    assert accumulated_change.collision_world_revisions(2) == (0, 1)


def test_multi_env_dynamic_collision_requires_explicit_mode_before_observation() -> (
    None
):
    state_provider = _MutableStateProvider(torch.eye(4))
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=state_provider,
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        )
    )

    with pytest.raises(ValueError, match="explicit collision_world_mode"):
        registry.make_scene_provider(batch_size=2)
    provider = registry.make_scene_provider()
    with pytest.raises(ValueError, match="explicit collision_world_mode"):
        provider.snapshot(
            timestamp=0.0,
            env_ids=torch.tensor([0, 1], dtype=torch.long),
        )
    assert state_provider.calls == 0


def test_single_env_dynamic_collision_defaults_to_shared_mode() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        )
    )
    provider = registry.make_scene_provider(batch_size=1)

    assert provider.collision_world_mode is SceneCollisionWorldMode.SHARED

    snapshot = provider.snapshot(
        timestamp=0.0,
        env_ids=torch.tensor([0], dtype=torch.long),
    )

    assert provider.collision_entity_ids == ("cube",)
    assert snapshot.collision_world_revisions(1) == (0,)


def test_collision_integration_requires_exact_canonical_ids_and_mode() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                aliases=("sim_cube",),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )

    assert (
        registry.validate_collision_integration(
            _MotionGenerator(entity_ids=("cube",)),  # type: ignore[arg-type]
            batch_size=2,
            scene_provider=_ExternalSceneProvider(("cube",)),
        )
        is SceneCollisionWorldMode.PER_ENV
    )
    with pytest.raises(ValueError, match="authoritative registry IDs"):
        registry.validate_collision_integration(
            _MotionGenerator(entity_ids=("sim_cube",)),  # type: ignore[arg-type]
            batch_size=2,
        )
    with pytest.raises(ValueError, match="does not support"):
        registry.validate_collision_integration(
            _MotionGenerator(  # type: ignore[arg-type]
                entity_ids=("cube",),
                supports_updates=False,
            ),
            batch_size=2,
        )
    with pytest.raises(ValueError, match="mode mismatch"):
        registry.validate_collision_integration(
            _MotionGenerator(  # type: ignore[arg-type]
                entity_ids=("cube",),
                batch_mode="shared",
            ),
            batch_size=2,
        )


def test_collision_integration_requires_exact_full_world_ids() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
            SceneEntityRegistration(
                ref=SceneObjectRef("table"),
                aliases=("legacy_table",),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.STATIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )

    assert registry.collision_world_entity_ids == ("cube", "table")
    assert (
        registry.validate_collision_integration(
            _MotionGenerator(
                entity_ids=("cube",),
                world_entity_ids=("cube", "table"),
            ),  # type: ignore[arg-type]
            batch_size=2,
        )
        is SceneCollisionWorldMode.PER_ENV
    )
    with pytest.raises(ValueError, match="Collision world.*authoritative registry IDs"):
        registry.validate_collision_integration(
            _MotionGenerator(
                entity_ids=("cube",),
                world_entity_ids=("cube", "legacy_table"),
            ),  # type: ignore[arg-type]
            batch_size=2,
        )


def test_static_only_collision_world_does_not_require_dynamic_updates() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("table"),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.STATIC,
            ),
        )
    )

    assert (
        registry.validate_collision_integration(
            _MotionGenerator(
                entity_ids=(),
                world_entity_ids=("table",),
                supports_updates=False,
                batch_mode=None,
            ),  # type: ignore[arg-type]
            batch_size=2,
        )
        is None
    )


def test_collision_integration_rejects_external_provider_id_drift() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )

    with pytest.raises(ValueError, match="provider.*authoritative registry IDs"):
        registry.validate_collision_integration(
            _MotionGenerator(entity_ids=("cube",)),  # type: ignore[arg-type]
            batch_size=2,
            scene_provider=_ExternalSceneProvider(("legacy_cube",)),
        )


def test_planning_provider_factory_validates_before_returning_provider() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.PER_ENV,
    )

    provider = registry.make_planning_scene_provider(
        _MotionGenerator(entity_ids=("cube",)),  # type: ignore[arg-type]
        batch_size=2,
    )
    assert provider.collision_entity_ids == ("cube",)

    with pytest.raises(ValueError, match="entity mismatch"):
        registry.make_planning_scene_provider(
            _MotionGenerator(entity_ids=("other",)),  # type: ignore[arg-type]
            batch_size=2,
        )

    with pytest.raises(ValueError, match="configured batch_size=2"):
        provider.snapshot(
            timestamp=0.0,
            env_ids=torch.tensor([0], dtype=torch.long),
        )


def test_collision_geometry_is_materialized_under_canonical_ids() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("dynamic_cube"),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
            SceneEntityRegistration(
                ref=SceneObjectRef("static_table"),
                state_provider=_StateProvider(),
                geometry_provider=_GeometryProvider(),
                collision_role=SceneCollisionRole.STATIC,
            ),
        ),
        collision_world_mode=SceneCollisionWorldMode.SHARED,
    )

    all_geometry = registry.collision_geometry_by_id()
    dynamic_geometry = registry.collision_geometry_by_id(SceneCollisionRole.DYNAMIC)

    assert set(all_geometry) == {"dynamic_cube", "static_table"}
    assert set(dynamic_geometry) == {"dynamic_cube"}
    with pytest.raises(TypeError):
        all_geometry["other"] = {}  # type: ignore[index]


def test_collision_integration_rejects_empty_dynamic_geometry() -> None:
    registry = SceneRegistry(
        (
            SceneEntityRegistration(
                ref=SceneObjectRef("cube"),
                state_provider=_StateProvider(),
                geometry_provider=_EmptyGeometryProvider(),
                collision_role=SceneCollisionRole.DYNAMIC,
            ),
        )
    )

    with pytest.raises(ValueError, match="scene entity 'cube'.*None"):
        registry.validate_collision_integration(
            _MotionGenerator(  # type: ignore[arg-type]
                entity_ids=("cube",),
                batch_mode="shared",
            ),
            batch_size=1,
        )


def test_from_simulation_is_explicit_and_uses_uid_only_as_alias() -> None:
    simulation = _Simulation()

    registry = SceneRegistry.from_simulation(
        simulation,  # type: ignore[arg-type]
        rigid_objects={"cube": "sim_cube"},
    )
    snapshot = registry.make_scene_provider().snapshot(
        timestamp=0.0,
        env_ids=torch.tensor([0], dtype=torch.long),
    )

    assert len(registry) == 1
    assert registry.resolve("sim_cube") == SceneObjectRef("cube")
    assert registry.lookup("cube").collision_role is SceneCollisionRole.NONE
    assert registry.dynamic_collision_entity_ids == ()
    assert registry.collision_geometry_by_id() == {}
    assert set(snapshot.entities) == {"cube"}
    assert "ignored" not in snapshot.entities


def test_from_simulation_derives_live_geometry_only_for_explicit_collision_role() -> (
    None
):
    simulation = _Simulation()
    registry = SceneRegistry.from_simulation(
        simulation,  # type: ignore[arg-type]
        rigid_objects={"cube": "sim_cube"},
        collision_roles={"cube": SceneCollisionRole.DYNAMIC},
        collision_world_mode=SceneCollisionWorldMode.SHARED,
    )

    assert registry.dynamic_collision_entity_ids == ("cube",)
    assert registry.collision_geometry_by_id() == {
        "cube": simulation.rigid_objects["sim_cube"]
    }


def test_from_simulation_allows_geometry_provider_override() -> None:
    registry = SceneRegistry.from_simulation(
        _Simulation(),  # type: ignore[arg-type]
        rigid_objects={"cube": "sim_cube"},
        collision_roles={"cube": SceneCollisionRole.STATIC},
        geometry_providers={"cube": _GeometryProvider()},
    )

    assert registry.collision_geometry_by_id() == {"cube": {"kind": "box"}}


def test_from_simulation_requires_selected_uid_to_exist() -> None:
    with pytest.raises(KeyError, match="missing"):
        SceneRegistry.from_simulation(
            _Simulation(),  # type: ignore[arg-type]
            articulations={"drawer": "missing"},
        )
