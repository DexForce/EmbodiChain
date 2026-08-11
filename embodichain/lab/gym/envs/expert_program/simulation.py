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

"""Explicit simulation bindings for declarative Expert Programs.

The values in this module bridge task-owned, executable-free declarations to
the existing :class:`SceneRegistry` and :class:`RobotSkillProfile` contracts.
They deliberately do not scan the simulation or infer semantic capabilities
from names. Every simulation entity, articulation member, control part, and
semantic command is selected explicitly and validated while the binding is
built.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ArticulationOperationAffordance,
    ArticulationOperationTarget,
    ControlPartCommandProfile,
    EntityState,
)
from embodichain.lab.sim.skills.profiles import (
    ControlPartEndpoint,
    ResourceEndpoint,
    ResourceBinding,
    RobotResource,
    RobotSkillProfile,
    SkillPolicyPreset,
)
from embodichain.lab.sim.skills.integration import SceneEntityManifest, SceneManifest
from embodichain.lab.sim.skills.scene import (
    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY,
    GRASP_AFFORDANCE_CAPABILITY,
    SceneAffordanceRef,
    SceneArticulationRef,
    SceneCollisionRole,
    SceneCollisionWorldMode,
    SceneDynamics,
    SceneEntityRegistration,
    SceneGeometryProvider,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
)
from embodichain.toolkits.graspkit.pg_grasp import GraspGeneratorCfg
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager


_IDENTITY_POSE = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _identifier(value: str, *, field_name: str) -> str:
    """Return one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _optional_identifier(value: str | None, *, field_name: str) -> str | None:
    """Validate one optional identifier."""
    if value is not None:
        _identifier(value, field_name=field_name)
    return value


def _identifier_tuple(
    values: tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Own a duplicate-free tuple of exact identifiers."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of identifiers.")
    normalized = tuple(values)
    for value in normalized:
        _identifier(value, field_name=field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique identifiers.")
    return normalized


def _finite(value: float, *, field_name: str) -> float:
    """Return one finite non-boolean float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _pose_tuple(
    values: tuple[float, ...],
    *,
    field_name: str,
) -> tuple[float, ...]:
    """Own and validate one flattened SE(3) matrix."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must contain 16 finite numbers.")
    normalized = tuple(
        _finite(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )
    if len(normalized) != 16:
        raise ValueError(f"{field_name} must contain exactly 16 numbers.")
    pose = torch.tensor(normalized, dtype=torch.float64).reshape(4, 4)
    bottom = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=torch.float64)
    if not torch.allclose(pose[3], bottom, atol=1.0e-6, rtol=0.0):
        raise ValueError(f"{field_name} must have bottom row [0, 0, 0, 1].")
    rotation = pose[:3, :3]
    if not torch.allclose(
        rotation.T @ rotation,
        torch.eye(3, dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    ) or not torch.isclose(
        torch.linalg.det(rotation),
        torch.tensor(1.0, dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError(f"{field_name} must contain a proper SE(3) rotation.")
    return normalized


def _pose_tensor(values: tuple[float, ...]) -> torch.Tensor:
    """Materialize an owned float32 pose matrix."""
    return torch.tensor(values, dtype=torch.float32).reshape(4, 4)


def _validate_scene_classification(
    dynamics: SceneDynamics,
    collision_role: SceneCollisionRole,
) -> None:
    """Validate exact scene-enum values."""
    if not isinstance(dynamics, SceneDynamics):
        raise TypeError("dynamics must be a SceneDynamics value.")
    if not isinstance(collision_role, SceneCollisionRole):
        raise TypeError("collision_role must be a SceneCollisionRole value.")


@dataclass(frozen=True, slots=True)
class SimulationRigidObjectBinding:
    """Explicit binding for one simulation rigid object."""

    entity_id: str
    simulation_uid: str
    aliases: tuple[str, ...] = ()
    dynamics: SceneDynamics = SceneDynamics.UNKNOWN
    collision_role: SceneCollisionRole = SceneCollisionRole.NONE
    semantic_type: str | None = None
    default_grasp_affordance: str | None = None
    geometry_provider: SceneGeometryProvider | None = None

    def __post_init__(self) -> None:
        _identifier(self.entity_id, field_name="entity_id")
        _identifier(self.simulation_uid, field_name="simulation_uid")
        object.__setattr__(
            self,
            "aliases",
            _identifier_tuple(self.aliases, field_name="aliases"),
        )
        _validate_scene_classification(self.dynamics, self.collision_role)
        _optional_identifier(self.semantic_type, field_name="semantic_type")
        _optional_identifier(
            self.default_grasp_affordance,
            field_name="default_grasp_affordance",
        )


@dataclass(frozen=True, slots=True)
class SimulationArticulationBinding:
    """Explicit binding for one simulation articulation."""

    entity_id: str
    simulation_uid: str
    aliases: tuple[str, ...] = ()
    dynamics: SceneDynamics = SceneDynamics.UNKNOWN
    collision_role: SceneCollisionRole = SceneCollisionRole.NONE
    semantic_type: str | None = None
    default_operation_affordance: str | None = None
    geometry_provider: SceneGeometryProvider | None = None

    def __post_init__(self) -> None:
        _identifier(self.entity_id, field_name="entity_id")
        _identifier(self.simulation_uid, field_name="simulation_uid")
        object.__setattr__(
            self,
            "aliases",
            _identifier_tuple(self.aliases, field_name="aliases"),
        )
        _validate_scene_classification(self.dynamics, self.collision_role)
        _optional_identifier(self.semantic_type, field_name="semantic_type")
        _optional_identifier(
            self.default_operation_affordance,
            field_name="default_operation_affordance",
        )


@dataclass(frozen=True, slots=True)
class SimulationArticulationLinkBinding:
    """Explicit canonical link backed by one native articulation link."""

    entity_id: str
    articulation_id: str
    native_link_name: str
    aliases: tuple[str, ...] = ()
    dynamics: SceneDynamics = SceneDynamics.UNKNOWN
    semantic_type: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.entity_id, field_name="entity_id")
        _identifier(self.articulation_id, field_name="articulation_id")
        _identifier(self.native_link_name, field_name="native_link_name")
        object.__setattr__(
            self,
            "aliases",
            _identifier_tuple(self.aliases, field_name="aliases"),
        )
        if not isinstance(self.dynamics, SceneDynamics):
            raise TypeError("dynamics must be a SceneDynamics value.")
        _optional_identifier(self.semantic_type, field_name="semantic_type")


@dataclass(frozen=True, slots=True)
class AntipodalGraspAffordanceBinding:
    """Build one antipodal grasp affordance from a selected rigid-object mesh."""

    entity_id: str
    object_id: str
    native_name: str
    revision: str
    aliases: tuple[str, ...] = ()
    relative_pose: tuple[float, ...] = _IDENTITY_POSE
    mesh_env_id: int = 0
    generator_cfg: GraspGeneratorCfg | None = None
    gripper_collision_cfg: GripperCollisionCfg | None = None
    force_reannotate: bool = False

    def __post_init__(self) -> None:
        for field_name in ("entity_id", "object_id", "native_name", "revision"):
            _identifier(getattr(self, field_name), field_name=field_name)
        object.__setattr__(
            self,
            "aliases",
            _identifier_tuple(self.aliases, field_name="aliases"),
        )
        object.__setattr__(
            self,
            "relative_pose",
            _pose_tuple(self.relative_pose, field_name="relative_pose"),
        )
        if (
            isinstance(self.mesh_env_id, bool)
            or not isinstance(self.mesh_env_id, int)
            or self.mesh_env_id < 0
        ):
            raise ValueError("mesh_env_id must be a non-negative integer.")
        if self.generator_cfg is not None and not isinstance(
            self.generator_cfg,
            GraspGeneratorCfg,
        ):
            raise TypeError("generator_cfg must be GraspGeneratorCfg or None.")
        if self.gripper_collision_cfg is not None and not isinstance(
            self.gripper_collision_cfg,
            GripperCollisionCfg,
        ):
            raise TypeError(
                "gripper_collision_cfg must be GripperCollisionCfg or None."
            )
        if not isinstance(self.force_reannotate, bool):
            raise TypeError("force_reannotate must be a bool.")
        object.__setattr__(self, "generator_cfg", deepcopy(self.generator_cfg))
        object.__setattr__(
            self,
            "gripper_collision_cfg",
            deepcopy(self.gripper_collision_cfg),
        )


@dataclass(frozen=True, slots=True)
class ArticulationOperationTargetBinding:
    """Declarative named target for one articulation operation."""

    target_position: float
    displacement: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_position",
            _finite(self.target_position, field_name="target_position"),
        )
        object.__setattr__(
            self,
            "displacement",
            _finite(self.displacement, field_name="displacement"),
        )

    def build(self) -> ArticulationOperationTarget:
        """Build the existing atomic-action target value."""
        return ArticulationOperationTarget(
            target_position=self.target_position,
            displacement=self.displacement,
        )


@dataclass(frozen=True, slots=True)
class ArticulationOperationAffordanceBinding:
    """Bind one handle operation to an explicit native link and joint."""

    entity_id: str
    articulation_id: str
    link_id: str
    joint_id: str
    revision: str
    semantic_targets: Mapping[str, ArticulationOperationTargetBinding]
    aliases: tuple[str, ...] = ()
    handle_pose_offset: tuple[float, ...] = _IDENTITY_POSE
    approach_offset: tuple[float, ...] = _IDENTITY_POSE
    contact_offset: tuple[float, ...] = _IDENTITY_POSE
    operation_offset: tuple[float, ...] = _IDENTITY_POSE
    retract_offset: tuple[float, ...] = _IDENTITY_POSE
    operation_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    position_scale: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "entity_id",
            "articulation_id",
            "link_id",
            "joint_id",
            "revision",
        ):
            _identifier(getattr(self, field_name), field_name=field_name)
        object.__setattr__(
            self,
            "aliases",
            _identifier_tuple(self.aliases, field_name="aliases"),
        )
        for field_name in (
            "handle_pose_offset",
            "approach_offset",
            "contact_offset",
            "operation_offset",
            "retract_offset",
        ):
            object.__setattr__(
                self,
                field_name,
                _pose_tuple(getattr(self, field_name), field_name=field_name),
            )
        axis = tuple(
            _finite(value, field_name=f"operation_axis[{index}]")
            for index, value in enumerate(self.operation_axis)
        )
        if len(axis) != 3 or math.sqrt(sum(value * value for value in axis)) <= 0.0:
            raise ValueError("operation_axis must contain three non-zero values.")
        object.__setattr__(self, "operation_axis", axis)
        position_scale = _finite(self.position_scale, field_name="position_scale")
        if position_scale <= 0.0:
            raise ValueError("position_scale must be positive.")
        object.__setattr__(self, "position_scale", position_scale)
        if not isinstance(self.semantic_targets, Mapping):
            raise TypeError("semantic_targets must be a mapping.")
        targets: dict[str, ArticulationOperationTargetBinding] = {}
        for target_id, target in self.semantic_targets.items():
            _identifier(target_id, field_name="semantic target IDs")
            if type(target) is not ArticulationOperationTargetBinding:
                raise TypeError(
                    "semantic_targets values must be exact "
                    "ArticulationOperationTargetBinding values."
                )
            targets[target_id] = target
        object.__setattr__(self, "semantic_targets", MappingProxyType(targets))


@dataclass(frozen=True, slots=True)
class _SimulationArticulationLinkStateProvider:
    """Read one selected native link pose with an optional local offset."""

    articulation: Any
    native_link_name: str
    local_offset: torch.Tensor = field(repr=False)

    def observe(
        self,
        *,
        timestamp: float,
        env_ids: torch.Tensor,
    ) -> EntityState:
        del timestamp
        getter = getattr(self.articulation, "get_link_pose", None)
        if not callable(getter):
            raise TypeError("Simulation articulation must provide get_link_pose().")
        pose = getter(
            self.native_link_name,
            env_ids=env_ids.detach().to("cpu").tolist(),
            to_matrix=True,
        )
        if not isinstance(pose, torch.Tensor):
            raise TypeError(
                "Simulation articulation get_link_pose() must return a tensor."
            )
        offset = self.local_offset.to(device=pose.device, dtype=pose.dtype)
        return EntityState(torch.matmul(pose, offset))


def _require_native_entity(
    simulation: SimulationManager,
    *,
    getter_name: str,
    registry_id: str,
    simulation_uid: str,
) -> Any:
    """Resolve one explicitly selected native entity or fail closed."""
    getter = getattr(simulation, getter_name, None)
    if not callable(getter):
        raise TypeError(f"simulation must provide {getter_name}().")
    entity = getter(simulation_uid)
    if entity is None:
        raise KeyError(
            f"Simulation UID {simulation_uid!r} selected for registry entity "
            f"{registry_id!r} was not found."
        )
    return entity


def _native_names(entity: Any, *, attribute: str, owner: str) -> tuple[str, ...]:
    """Read and validate one existing native-name collection."""
    values = getattr(entity, attribute, None)
    if values is None:
        raise TypeError(f"{owner} must expose {attribute}.")
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{owner}.{attribute} must be an iterable of names.")
    try:
        names = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{owner}.{attribute} must be an iterable of names.") from exc
    for name in names:
        _identifier(name, field_name=f"{owner}.{attribute}")
    if len(set(names)) != len(names):
        raise ValueError(f"{owner}.{attribute} must contain unique names.")
    return names


def _mesh_tensor(
    entity: Any,
    *,
    getter_name: str,
    mesh_env_id: int,
    vertices: bool,
) -> torch.Tensor:
    """Read one explicitly selected mesh row with strict shape validation."""
    getter = getattr(entity, getter_name, None)
    if not callable(getter):
        raise TypeError(f"Simulation rigid object must provide {getter_name}().")
    if vertices:
        value = getter(env_ids=[mesh_env_id], scale=True)
    else:
        value = getter(env_ids=[mesh_env_id])
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Simulation rigid object {getter_name}() must return a tensor."
        )
    if value.dim() != 3 or value.shape[0] != 1 or value.shape[2] != 3:
        raise ValueError(
            f"Simulation rigid object {getter_name}() must return shape (1, N, 3)."
        )
    selected = value[0].detach().clone()
    if selected.shape[0] == 0:
        raise ValueError(f"Simulation rigid object {getter_name}() returned no data.")
    if vertices:
        if not selected.is_floating_point() or not torch.isfinite(selected).all():
            raise ValueError("Antipodal mesh vertices must be finite floating values.")
    elif selected.dtype == torch.bool or selected.is_floating_point():
        raise TypeError("Antipodal mesh triangles must use an integer dtype.")
    return selected


def _antipodal_affordance(
    binding: AntipodalGraspAffordanceBinding,
    entity: Any,
) -> AntipodalAffordance:
    """Build and validate one owned antipodal affordance payload."""
    vertices = _mesh_tensor(
        entity,
        getter_name="get_vertices",
        mesh_env_id=binding.mesh_env_id,
        vertices=True,
    )
    triangles = _mesh_tensor(
        entity,
        getter_name="get_triangles",
        mesh_env_id=binding.mesh_env_id,
        vertices=False,
    )
    if bool((triangles < 0).any()) or int(triangles.max().item()) >= vertices.shape[0]:
        raise ValueError("Antipodal mesh triangles reference invalid vertex indices.")
    return AntipodalAffordance(
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        generator_cfg=deepcopy(binding.generator_cfg),
        gripper_collision_cfg=deepcopy(binding.gripper_collision_cfg),
        force_reannotate=binding.force_reannotate,
    )


@dataclass(frozen=True, slots=True)
class SimulationSceneBinding:
    """Build one authoritative registry from explicit simulation bindings."""

    registry_id: str
    rigid_objects: tuple[SimulationRigidObjectBinding, ...] = ()
    articulations: tuple[SimulationArticulationBinding, ...] = ()
    links: tuple[SimulationArticulationLinkBinding, ...] = ()
    antipodal_grasps: tuple[AntipodalGraspAffordanceBinding, ...] = ()
    articulation_operations: tuple[ArticulationOperationAffordanceBinding, ...] = ()
    collision_world_mode: SceneCollisionWorldMode | None = None

    def __post_init__(self) -> None:
        _identifier(self.registry_id, field_name="registry_id")
        expected_types = {
            "rigid_objects": SimulationRigidObjectBinding,
            "articulations": SimulationArticulationBinding,
            "links": SimulationArticulationLinkBinding,
            "antipodal_grasps": AntipodalGraspAffordanceBinding,
            "articulation_operations": ArticulationOperationAffordanceBinding,
        }
        all_ids: list[str] = []
        for field_name, expected_type in expected_types.items():
            values = tuple(getattr(self, field_name))
            if not all(type(value) is expected_type for value in values):
                raise TypeError(
                    f"{field_name} must contain exact {expected_type.__name__} values."
                )
            object.__setattr__(self, field_name, values)
            all_ids.extend(value.entity_id for value in values)
        duplicates = sorted(
            entity_id for entity_id in set(all_ids) if all_ids.count(entity_id) > 1
        )
        if duplicates:
            raise ValueError(f"Scene binding entity IDs must be unique: {duplicates}.")
        if self.collision_world_mode is not None and not isinstance(
            self.collision_world_mode,
            SceneCollisionWorldMode,
        ):
            raise TypeError(
                "collision_world_mode must be SceneCollisionWorldMode or None."
            )

    def declare(self) -> SceneManifest:
        """Project the complete provider-free scene declaration.

        Canonical topology errors are rejected here, before a simulation is
        constructed. Native simulation UIDs, mesh data, link names, and joint
        names remain live validation owned by :meth:`build`.
        """
        objects = {item.entity_id: item for item in self.rigid_objects}
        articulations = {item.entity_id: item for item in self.articulations}
        links = {item.entity_id: item for item in self.links}
        entries: list[SceneEntityManifest] = []

        for binding in self.rigid_objects:
            native_aliases = (
                ()
                if binding.simulation_uid == binding.entity_id
                else (binding.simulation_uid,)
            )
            defaults = (
                {}
                if binding.default_grasp_affordance is None
                else {
                    GRASP_AFFORDANCE_CAPABILITY: SceneAffordanceRef(
                        binding.default_grasp_affordance
                    )
                }
            )
            entries.append(
                SceneEntityManifest(
                    ref=SceneObjectRef(binding.entity_id),
                    aliases=(*native_aliases, *binding.aliases),
                    dynamics=binding.dynamics,
                    collision_role=binding.collision_role,
                    semantic_type=binding.semantic_type,
                    default_affordances=defaults,
                )
            )

        for binding in self.articulations:
            native_aliases = (
                ()
                if binding.simulation_uid == binding.entity_id
                else (binding.simulation_uid,)
            )
            defaults = (
                {}
                if binding.default_operation_affordance is None
                else {
                    ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY: (
                        SceneAffordanceRef(binding.default_operation_affordance)
                    )
                }
            )
            entries.append(
                SceneEntityManifest(
                    ref=SceneArticulationRef(binding.entity_id),
                    aliases=(*native_aliases, *binding.aliases),
                    dynamics=binding.dynamics,
                    collision_role=binding.collision_role,
                    semantic_type=binding.semantic_type,
                    default_affordances=defaults,
                )
            )

        for binding in self.links:
            if binding.articulation_id not in articulations:
                raise KeyError(
                    f"Link {binding.entity_id!r} references unbound articulation "
                    f"{binding.articulation_id!r}."
                )
            entries.append(
                SceneEntityManifest(
                    ref=SceneLinkRef(binding.entity_id),
                    aliases=binding.aliases,
                    parent=SceneArticulationRef(binding.articulation_id),
                    native_name=binding.native_link_name,
                    dynamics=binding.dynamics,
                    semantic_type=binding.semantic_type,
                )
            )

        for binding in self.antipodal_grasps:
            if binding.object_id not in objects:
                raise KeyError(
                    f"Grasp affordance {binding.entity_id!r} references unbound "
                    f"object {binding.object_id!r}."
                )
            entries.append(
                SceneEntityManifest(
                    ref=SceneAffordanceRef(binding.entity_id),
                    aliases=binding.aliases,
                    parent=SceneObjectRef(binding.object_id),
                    native_name=binding.native_name,
                    affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                    affordance_payload_type=AntipodalAffordance,
                    affordance_revision=binding.revision,
                    relative_pose=binding.relative_pose,
                )
            )

        for binding in self.articulation_operations:
            if binding.articulation_id not in articulations:
                raise KeyError(
                    f"Operation affordance {binding.entity_id!r} references "
                    f"unbound articulation {binding.articulation_id!r}."
                )
            link = links.get(binding.link_id)
            if link is None:
                raise KeyError(
                    f"Operation affordance {binding.entity_id!r} references "
                    f"unbound link {binding.link_id!r}."
                )
            if link.articulation_id != binding.articulation_id:
                raise ValueError(
                    f"Operation affordance {binding.entity_id!r} and link "
                    f"{binding.link_id!r} select different articulations."
                )
            entries.append(
                SceneEntityManifest(
                    ref=SceneAffordanceRef(binding.entity_id),
                    aliases=binding.aliases,
                    parent=SceneArticulationRef(binding.articulation_id),
                    native_name=link.native_link_name,
                    affordance_capabilities=frozenset(
                        {ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY}
                    ),
                    affordance_payload_type=ArticulationOperationAffordance,
                    affordance_revision=binding.revision,
                )
            )

        return SceneManifest(entries)

    def build(self, simulation: SimulationManager) -> SceneRegistry:
        """Build the existing authoritative scene registry.

        Args:
            simulation: Live simulation used only for explicitly named lookups.

        Returns:
            Immutable registry with typed roots, links, and affordances.
        """
        objects = {item.entity_id: item for item in self.rigid_objects}
        articulations = {item.entity_id: item for item in self.articulations}
        geometry = {
            item.entity_id: item.geometry_provider
            for item in (*self.rigid_objects, *self.articulations)
            if item.geometry_provider is not None
        }
        roles = {
            item.entity_id: item.collision_role
            for item in (*self.rigid_objects, *self.articulations)
        }
        base = SceneRegistry.from_simulation(
            simulation,
            rigid_objects={
                item.entity_id: item.simulation_uid for item in self.rigid_objects
            },
            articulations={
                item.entity_id: item.simulation_uid for item in self.articulations
            },
            collision_roles=roles,
            geometry_providers=geometry,
            collision_world_mode=self.collision_world_mode,
        )

        registrations: list[SceneEntityRegistration] = []
        for registration in base.registrations:
            entity_id = registration.ref.entity_id
            if isinstance(registration.ref, SceneObjectRef):
                binding = objects[entity_id]
                defaults = (
                    {}
                    if binding.default_grasp_affordance is None
                    else {
                        GRASP_AFFORDANCE_CAPABILITY: SceneAffordanceRef(
                            binding.default_grasp_affordance
                        )
                    }
                )
            else:
                binding = articulations[entity_id]
                defaults = (
                    {}
                    if binding.default_operation_affordance is None
                    else {
                        ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY: (
                            SceneAffordanceRef(binding.default_operation_affordance)
                        )
                    }
                )
            registrations.append(
                replace(
                    registration,
                    aliases=(*registration.aliases, *binding.aliases),
                    dynamics=binding.dynamics,
                    semantic_type=binding.semantic_type,
                    default_affordances=defaults,
                )
            )

        native_articulations: dict[str, Any] = {}
        links: dict[str, SimulationArticulationLinkBinding] = {}
        for binding in self.links:
            articulation_binding = articulations.get(binding.articulation_id)
            if articulation_binding is None:
                raise KeyError(
                    f"Link {binding.entity_id!r} references unbound articulation "
                    f"{binding.articulation_id!r}."
                )
            articulation = native_articulations.setdefault(
                binding.articulation_id,
                _require_native_entity(
                    simulation,
                    getter_name="get_articulation",
                    registry_id=binding.articulation_id,
                    simulation_uid=articulation_binding.simulation_uid,
                ),
            )
            native_links = _native_names(
                articulation,
                attribute="link_names",
                owner=f"articulation {binding.articulation_id!r}",
            )
            if binding.native_link_name not in native_links:
                raise KeyError(
                    f"Native link {binding.native_link_name!r} selected for "
                    f"{binding.entity_id!r} was not found; available links are "
                    f"{sorted(native_links)}."
                )
            links[binding.entity_id] = binding
            registrations.append(
                SceneEntityRegistration(
                    ref=SceneLinkRef(binding.entity_id),
                    state_provider=_SimulationArticulationLinkStateProvider(
                        articulation,
                        binding.native_link_name,
                        _pose_tensor(_IDENTITY_POSE),
                    ),
                    aliases=binding.aliases,
                    parent=SceneArticulationRef(binding.articulation_id),
                    native_name=binding.native_link_name,
                    dynamics=binding.dynamics,
                    semantic_type=binding.semantic_type,
                )
            )

        native_objects: dict[str, Any] = {}
        for binding in self.antipodal_grasps:
            object_binding = objects.get(binding.object_id)
            if object_binding is None:
                raise KeyError(
                    f"Grasp affordance {binding.entity_id!r} references unbound "
                    f"object {binding.object_id!r}."
                )
            entity = native_objects.setdefault(
                binding.object_id,
                _require_native_entity(
                    simulation,
                    getter_name="get_rigid_object",
                    registry_id=binding.object_id,
                    simulation_uid=object_binding.simulation_uid,
                ),
            )
            registrations.append(
                SceneEntityRegistration(
                    ref=SceneAffordanceRef(binding.entity_id),
                    aliases=binding.aliases,
                    parent=SceneObjectRef(binding.object_id),
                    native_name=binding.native_name,
                    affordance=_antipodal_affordance(binding, entity),
                    affordance_capabilities=frozenset({GRASP_AFFORDANCE_CAPABILITY}),
                    affordance_revision=binding.revision,
                    relative_pose=_pose_tensor(binding.relative_pose),
                )
            )

        for binding in self.articulation_operations:
            articulation_binding = articulations.get(binding.articulation_id)
            if articulation_binding is None:
                raise KeyError(
                    f"Operation affordance {binding.entity_id!r} references "
                    f"unbound articulation {binding.articulation_id!r}."
                )
            link_binding = links.get(binding.link_id)
            if link_binding is None:
                raise KeyError(
                    f"Operation affordance {binding.entity_id!r} references "
                    f"unbound link {binding.link_id!r}."
                )
            if link_binding.articulation_id != binding.articulation_id:
                raise ValueError(
                    f"Operation affordance {binding.entity_id!r} and link "
                    f"{binding.link_id!r} select different articulations."
                )
            articulation = native_articulations[binding.articulation_id]
            native_joints = _native_names(
                articulation,
                attribute="joint_names",
                owner=f"articulation {binding.articulation_id!r}",
            )
            if binding.joint_id not in native_joints:
                raise KeyError(
                    f"Native joint {binding.joint_id!r} selected for "
                    f"{binding.entity_id!r} was not found; available joints are "
                    f"{sorted(native_joints)}."
                )
            payload = ArticulationOperationAffordance(
                joint_id=binding.joint_id,
                approach_offset=_pose_tensor(binding.approach_offset),
                contact_offset=_pose_tensor(binding.contact_offset),
                operation_offset=_pose_tensor(binding.operation_offset),
                retract_offset=_pose_tensor(binding.retract_offset),
                operation_axis=torch.tensor(
                    binding.operation_axis,
                    dtype=torch.float32,
                ),
                position_scale=binding.position_scale,
                semantic_targets={
                    target_id: target.build()
                    for target_id, target in binding.semantic_targets.items()
                },
            )
            registrations.append(
                SceneEntityRegistration(
                    ref=SceneAffordanceRef(binding.entity_id),
                    state_provider=_SimulationArticulationLinkStateProvider(
                        articulation,
                        link_binding.native_link_name,
                        _pose_tensor(binding.handle_pose_offset),
                    ),
                    aliases=binding.aliases,
                    parent=SceneArticulationRef(binding.articulation_id),
                    native_name=link_binding.native_link_name,
                    affordance=payload,
                    affordance_capabilities=frozenset(
                        {ARTICULATION_OPERATION_AFFORDANCE_CAPABILITY}
                    ),
                    affordance_revision=binding.revision,
                )
            )

        return SceneRegistry(
            registrations,
            collision_world_mode=self.collision_world_mode,
        )


@dataclass(frozen=True, slots=True)
class ControlPartCommandPreset:
    """Named one-dimensional joint commands for one exact control part."""

    preset_id: str
    control_part: str
    commands: Mapping[str, tuple[float, ...]]

    def __post_init__(self) -> None:
        _identifier(self.preset_id, field_name="preset_id")
        _identifier(self.control_part, field_name="control_part")
        if not isinstance(self.commands, Mapping):
            raise TypeError("commands must be a mapping.")
        commands: dict[str, tuple[float, ...]] = {}
        for command_id, positions in self.commands.items():
            _identifier(command_id, field_name="command IDs")
            if isinstance(positions, (str, bytes)):
                raise TypeError("command positions must be an iterable of numbers.")
            normalized = tuple(
                _finite(value, field_name=f"commands[{command_id!r}][{index}]")
                for index, value in enumerate(positions)
            )
            if not normalized:
                raise ValueError("command positions must not be empty.")
            commands[command_id] = normalized
        object.__setattr__(self, "commands", MappingProxyType(commands))

    def build(self, *, control_dof: int) -> ControlPartCommandProfile:
        """Build a command profile after validating the native control width."""
        for command_id, positions in self.commands.items():
            if len(positions) != control_dof:
                raise ValueError(
                    f"Command {command_id!r} in preset {self.preset_id!r} has "
                    f"{len(positions)} positions, but control part "
                    f"{self.control_part!r} has {control_dof} joints."
                )
        return ControlPartCommandProfile.joint_positions(
            **{
                command_id: torch.tensor(positions, dtype=torch.float32)
                for command_id, positions in self.commands.items()
            }
        )

    def declare(self) -> ControlPartCommandProfile:
        """Build a provider-free command profile from declared tuple widths."""
        widths = {len(positions) for positions in self.commands.values()}
        if len(widths) > 1:
            raise ValueError(
                f"Command preset {self.preset_id!r} declares inconsistent command "
                f"widths {sorted(widths)}."
            )
        return ControlPartCommandProfile.joint_positions(
            **{
                command_id: torch.tensor(positions, dtype=torch.float32)
                for command_id, positions in self.commands.items()
            }
        )


def _require_control_part_dof(robot: Robot, control_part: str) -> int:
    """Validate one native joint-backed control part and return its width."""
    control_parts = getattr(robot, "control_parts", None)
    if not isinstance(control_parts, Mapping):
        raise TypeError("robot must expose a control_parts mapping.")
    get_joint_ids = getattr(robot, "get_joint_ids", None)
    if not callable(get_joint_ids):
        raise TypeError("robot must provide get_joint_ids().")
    if control_part not in control_parts:
        raise KeyError(
            f"Robot control part {control_part!r} was not found; available "
            f"control parts are {sorted(str(key) for key in control_parts)}."
        )
    joint_ids = tuple(get_joint_ids(name=control_part))
    if not joint_ids:
        raise ValueError(f"Robot control part {control_part!r} contains no joints.")
    if not all(
        isinstance(joint_id, int) and not isinstance(joint_id, bool) and joint_id >= 0
        for joint_id in joint_ids
    ):
        raise ValueError(
            f"Robot control part {control_part!r} returned invalid joint IDs."
        )
    if len(set(joint_ids)) != len(joint_ids):
        raise ValueError(
            f"Robot control part {control_part!r} contains duplicate joint IDs."
        )
    return len(joint_ids)


@runtime_checkable
class SimulationResourceEndpointBinding(Protocol):
    """Build one typed resource endpoint from an explicitly selected robot.

    Implementations are reusable robot-integration declarations. They may
    validate embodiment-specific controller surfaces, but must only return an
    owned :class:`ResourceEndpoint`; live controller handles remain in the
    endpoint adapter and runtime transport.
    """

    @property
    def endpoint_id(self) -> str:
        """Return the stable endpoint ID within its containing resource."""

    def build(self, robot: Robot) -> ResourceEndpoint:
        """Build and validate one endpoint declaration for ``robot``."""

    def declare(self) -> ResourceEndpoint:
        """Return the provider-free endpoint declaration."""


@runtime_checkable
class SimulationRobotResourceBinding(Protocol):
    """Build one leaf or composite resource in the robot resource DAG."""

    @property
    def resource_id(self) -> str:
        """Return the stable resource ID."""

    @property
    def members(self) -> tuple[str, ...]:
        """Return explicitly declared child resource IDs."""

    def build(self, robot: Robot) -> RobotResource:
        """Build and validate one owned robot resource declaration."""

    def declare(self) -> RobotResource:
        """Return the provider-free resource declaration."""


@dataclass(frozen=True, slots=True)
class RobotResourceBinding:
    """Generic simulation binding for arbitrary typed resource endpoints.

    This is the direct configuration path for mobile bases, whole-body
    controllers, tools, and other non-joint transports. Endpoint-specific
    validation remains in the registered :class:`ResourceEndpointAdapter`;
    this value owns the declaration and preserves the resource DAG exactly.
    """

    resource_id: str
    endpoints: Mapping[str, ResourceEndpoint] = field(default_factory=dict)
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        resource = RobotResource(
            resource_id=self.resource_id,
            endpoints=self.endpoints,
            members=self.members,
        )
        object.__setattr__(self, "endpoints", resource.endpoints)
        object.__setattr__(self, "members", resource.members)

    def build(self, robot: Robot) -> RobotResource:
        """Build an independently owned resource without assuming robot joints."""
        del robot
        return RobotResource(
            resource_id=self.resource_id,
            endpoints=self.endpoints,
            members=self.members,
        )

    def declare(self) -> RobotResource:
        """Return an independently owned provider-free resource."""
        return RobotResource(
            resource_id=self.resource_id,
            endpoints=self.endpoints,
            members=self.members,
        )


@dataclass(frozen=True, slots=True)
class ControlPartEndpointBinding:
    """Profile endpoint backed by one explicit robot control part."""

    endpoint_id: str
    control_part: str
    capabilities: frozenset[str]
    command_preset: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.endpoint_id, field_name="endpoint_id")
        _identifier(self.control_part, field_name="control_part")
        if isinstance(self.capabilities, (str, bytes)):
            raise TypeError("capabilities must be an iterable of identifiers.")
        capabilities = frozenset(self.capabilities)
        for capability in capabilities:
            _identifier(capability, field_name="capabilities")
        object.__setattr__(self, "capabilities", capabilities)
        _optional_identifier(self.command_preset, field_name="command_preset")

    def build(self, robot: Robot) -> ResourceEndpoint:
        """Build a joint-backed endpoint after native control-part validation."""
        _require_control_part_dof(robot, self.control_part)
        return ControlPartEndpoint(
            control_part=self.control_part,
            command_profile=self.command_preset,
            capabilities=self.capabilities,
        )

    def declare(self) -> ResourceEndpoint:
        """Return the endpoint contract without reading a robot."""
        return ControlPartEndpoint(
            control_part=self.control_part,
            command_profile=self.command_preset,
            capabilities=self.capabilities,
        )


@dataclass(frozen=True, slots=True)
class ControlPartResourceBinding:
    """Joint-backed robot resource containing control-part endpoints."""

    resource_id: str
    endpoints: tuple[ControlPartEndpointBinding, ...] = ()
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.resource_id, field_name="resource_id")
        endpoints = tuple(self.endpoints)
        if not all(
            type(endpoint) is ControlPartEndpointBinding for endpoint in endpoints
        ):
            raise TypeError(
                "endpoints must contain exact ControlPartEndpointBinding values."
            )
        endpoint_ids = [endpoint.endpoint_id for endpoint in endpoints]
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise ValueError("endpoint_id values must be unique within a resource.")
        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(
            self,
            "members",
            _identifier_tuple(self.members, field_name="members"),
        )

    def build(self, robot: Robot) -> RobotResource:
        """Build a resource containing strictly validated control-part endpoints."""
        endpoints: dict[str, ResourceEndpoint] = {}
        for binding in self.endpoints:
            endpoint = binding.build(robot)
            if type(endpoint) is not ControlPartEndpoint:
                raise TypeError(
                    "ControlPartEndpointBinding.build() must return exactly "
                    "ControlPartEndpoint."
                )
            endpoints[binding.endpoint_id] = endpoint
        return RobotResource(
            resource_id=self.resource_id,
            endpoints=endpoints,
            members=self.members,
        )

    def declare(self) -> RobotResource:
        """Return the resource graph without reading native control parts."""
        return RobotResource(
            resource_id=self.resource_id,
            endpoints={
                binding.endpoint_id: binding.declare() for binding in self.endpoints
            },
            members=self.members,
        )


def _owned_nested_identifier_mapping(
    values: Mapping[str, Mapping[str, str]],
    *,
    field_name: str,
) -> Mapping[str, Mapping[str, str]]:
    """Own a strict two-level identifier mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    outer: dict[str, Mapping[str, str]] = {}
    for key, nested in values.items():
        _identifier(key, field_name=f"{field_name} keys")
        if not isinstance(nested, Mapping):
            raise TypeError(f"{field_name}[{key!r}] must be a mapping.")
        normalized: dict[str, str] = {}
        for nested_key, nested_value in nested.items():
            _identifier(nested_key, field_name=f"{field_name} slot IDs")
            _identifier(nested_value, field_name=f"{field_name} resource IDs")
            normalized[nested_key] = nested_value
        outer[key] = MappingProxyType(normalized)
    return MappingProxyType(outer)


def _owned_identifier_mapping(
    values: Mapping[str, str],
    *,
    field_name: str,
) -> Mapping[str, str]:
    """Own one strict identifier mapping."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        _identifier(key, field_name=f"{field_name} keys")
        _identifier(value, field_name=f"{field_name} values")
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class SimulationRobotSkillProfileBinding:
    """Build a profile from typed resources with strict native validation."""

    profile_id: str
    resources: tuple[SimulationRobotResourceBinding, ...]
    command_presets: tuple[ControlPartCommandPreset, ...] = ()
    defaults: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    presets: tuple[SkillPolicyPreset, ...] = ()
    default_preset: str | None = None
    skill_presets: Mapping[str, str] = field(default_factory=dict)
    grounding_providers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.profile_id, field_name="profile_id")
        resources = tuple(self.resources)
        if not all(
            isinstance(resource, SimulationRobotResourceBinding)
            for resource in resources
        ):
            raise TypeError("resources must implement SimulationRobotResourceBinding.")
        for resource in resources:
            _identifier(resource.resource_id, field_name="resource_id")
            _identifier_tuple(resource.members, field_name="resource members")
        resource_ids = [resource.resource_id for resource in resources]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("resource_id values must be unique.")
        object.__setattr__(self, "resources", resources)
        command_presets = tuple(self.command_presets)
        if not all(
            type(preset) is ControlPartCommandPreset for preset in command_presets
        ):
            raise TypeError(
                "command_presets must contain exact ControlPartCommandPreset values."
            )
        command_preset_ids = [preset.preset_id for preset in command_presets]
        if len(set(command_preset_ids)) != len(command_preset_ids):
            raise ValueError("command preset IDs must be unique.")
        object.__setattr__(self, "command_presets", command_presets)
        object.__setattr__(
            self,
            "defaults",
            _owned_nested_identifier_mapping(self.defaults, field_name="defaults"),
        )
        presets = tuple(self.presets)
        if not all(type(preset) is SkillPolicyPreset for preset in presets):
            raise TypeError("presets must contain exact SkillPolicyPreset values.")
        preset_ids = [preset.preset_id for preset in presets]
        if len(set(preset_ids)) != len(preset_ids):
            raise ValueError("policy preset IDs must be unique.")
        object.__setattr__(self, "presets", presets)
        _optional_identifier(self.default_preset, field_name="default_preset")
        object.__setattr__(
            self,
            "skill_presets",
            _owned_identifier_mapping(
                self.skill_presets,
                field_name="skill_presets",
            ),
        )
        object.__setattr__(
            self,
            "grounding_providers",
            _owned_identifier_mapping(
                self.grounding_providers,
                field_name="grounding_providers",
            ),
        )

    def build(self, robot: Robot) -> RobotSkillProfile:
        """Build the existing profile after validating every typed resource.

        Args:
            robot: Live robot selected by the simulation factory.

        Returns:
            Reusable, engine-independent robot skill profile.
        """
        control_dofs: dict[str, int] = {}

        def require_control_part(control_part: str) -> int:
            if control_part not in control_dofs:
                control_dofs[control_part] = _require_control_part_dof(
                    robot,
                    control_part,
                )
            return control_dofs[control_part]

        command_presets = {preset.preset_id: preset for preset in self.command_presets}
        command_profiles: dict[str, ControlPartCommandProfile] = {}
        for preset in self.command_presets:
            command_profiles[preset.preset_id] = preset.build(
                control_dof=require_control_part(preset.control_part)
            )

        resources: dict[str, RobotResource] = {}
        for resource_binding in self.resources:
            resource = resource_binding.build(robot)
            if type(resource) is not RobotResource:
                raise TypeError(
                    f"Resource binding {resource_binding.resource_id!r} must build "
                    "exactly RobotResource."
                )
            if resource.resource_id != resource_binding.resource_id:
                raise ValueError(
                    f"Resource binding {resource_binding.resource_id!r} built "
                    f"resource ID {resource.resource_id!r}."
                )
            if resource.members != tuple(resource_binding.members):
                raise ValueError(
                    f"Resource binding {resource_binding.resource_id!r} changed its "
                    "declared resource members while building."
                )
            for endpoint_id, endpoint in resource.endpoints.items():
                if not isinstance(endpoint, ControlPartEndpoint):
                    continue
                require_control_part(endpoint.control_part)
                profile_id = (
                    endpoint.control_part
                    if endpoint.command_profile is None
                    else endpoint.command_profile
                )
                command_preset = command_presets.get(profile_id)
                if endpoint.command_profile is not None and command_preset is None:
                    raise KeyError(
                        f"Endpoint {resource.resource_id!r}.{endpoint_id!r} "
                        "references unknown command "
                        f"preset {profile_id!r}."
                    )
                if (
                    command_preset is not None
                    and command_preset.control_part != endpoint.control_part
                ):
                    raise ValueError(
                        f"Endpoint {resource.resource_id!r}.{endpoint_id!r} uses "
                        "control part "
                        f"{endpoint.control_part!r}, but command preset "
                        f"{profile_id!r} targets "
                        f"{command_preset.control_part!r}."
                    )
            resources[resource.resource_id] = resource

        return RobotSkillProfile(
            profile_id=self.profile_id,
            resources=resources,
            command_profiles=command_profiles,
            defaults={
                skill_id: ResourceBinding(resources=bindings)
                for skill_id, bindings in self.defaults.items()
            },
            presets={preset.preset_id: preset for preset in self.presets},
            default_preset=self.default_preset,
            skill_presets=self.skill_presets,
            grounding_providers=self.grounding_providers,
        )

    def declare(self) -> RobotSkillProfile:
        """Project the complete provider-free robot skill profile."""
        resources: dict[str, RobotResource] = {}
        for binding in self.resources:
            resource = binding.declare()
            if type(resource) is not RobotResource:
                raise TypeError(
                    f"Resource binding {binding.resource_id!r} must declare "
                    "exactly RobotResource."
                )
            if resource.resource_id != binding.resource_id:
                raise ValueError(
                    f"Resource binding {binding.resource_id!r} declared "
                    f"resource ID {resource.resource_id!r}."
                )
            if resource.members != tuple(binding.members):
                raise ValueError(
                    f"Resource binding {binding.resource_id!r} changed its "
                    "declared resource members."
                )
            resources[resource.resource_id] = resource

        command_presets = {preset.preset_id: preset for preset in self.command_presets}
        for resource in resources.values():
            for endpoint_id, endpoint in resource.endpoints.items():
                if not isinstance(endpoint, ControlPartEndpoint):
                    continue
                preset_id = endpoint.command_profile
                if preset_id is None:
                    continue
                preset = command_presets.get(preset_id)
                if preset is None:
                    raise KeyError(
                        f"Endpoint {resource.resource_id!r}.{endpoint_id!r} "
                        f"references unknown command preset {preset_id!r}."
                    )
                if preset.control_part != endpoint.control_part:
                    raise ValueError(
                        f"Endpoint {resource.resource_id!r}.{endpoint_id!r} uses "
                        f"control part {endpoint.control_part!r}, but command "
                        f"preset {preset_id!r} targets {preset.control_part!r}."
                    )

        return RobotSkillProfile(
            profile_id=self.profile_id,
            resources=resources,
            command_profiles={
                preset.preset_id: preset.declare() for preset in self.command_presets
            },
            defaults={
                skill_id: ResourceBinding(resources=bindings)
                for skill_id, bindings in self.defaults.items()
            },
            presets={preset.preset_id: preset for preset in self.presets},
            default_preset=self.default_preset,
            skill_presets=self.skill_presets,
            grounding_providers=self.grounding_providers,
        )


__all__ = [
    "AntipodalGraspAffordanceBinding",
    "ArticulationOperationAffordanceBinding",
    "ArticulationOperationTargetBinding",
    "ControlPartCommandPreset",
    "ControlPartEndpointBinding",
    "ControlPartResourceBinding",
    "RobotResourceBinding",
    "SimulationArticulationBinding",
    "SimulationArticulationLinkBinding",
    "SimulationRigidObjectBinding",
    "SimulationResourceEndpointBinding",
    "SimulationRobotResourceBinding",
    "SimulationRobotSkillProfileBinding",
    "SimulationSceneBinding",
]
