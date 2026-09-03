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

"""Allowlisted live services for configured Task Program integrations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, ClassVar

import torch

from embodichain.utils.math import axis_angle_to_rotation_matrix

from .extensions import (
    ControlPartEvidenceProviderFactory,
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    Affordance,
    AtomicActionEngine,
    AxisAlign,
    AxisAlignAffordance,
    AxisAlignGoal,
    AxisAlignOptions,
    CoordinatedPickGoal,
    CoordinatedPickment,
    CoordinatedPickmentOptions,
    HeldObjectPoseGoal,
    JointPositionGoal,
    MoveHeldObject,
    MoveHeldObjectOptions,
    MoveJoints,
    MoveJointsOptions,
    ObjectSemantics,
    PARK_COMMAND,
    Place,
    PlaceGoal,
    PlaceOptions,
    PlanningContext,
    Pour,
    PourGoal,
    PourOptions,
    PushObject,
    PushObjectGoal,
    PushObjectOptions,
    SceneEntityPose,
    SceneProvider,
    SkillDescriptor,
    Slide,
    SlideAffordance,
    SlideGoal,
    SlideOptions,
    sample_initial_articulation_geometry,
)
from embodichain.lab.task_program.semantics import (
    BinaryEffectEvidenceQuery,
    BinaryEffectObservation,
    BinaryEvidenceKind,
    BoundSemanticCall,
    CONSTRAINT_EFFECT_CHANNEL,
    CONTROL_PART_EVIDENCE_PROVIDER_ID,
    CONTROL_PART_EVIDENCE_PROVIDER_REVISION,
    ControlPartEvidenceAddress,
    ControlPartSimulationEvidenceProvider,
    EffectEvidenceCollectionContext,
    EffectEvidenceProvider,
    GRASP_AFFORDANCE_CAPABILITY,
    HeldObjectRelation,
    HeldObjectStateExpectation,
    RegisteredSemanticCall,
    SceneArticulationRef,
    SceneLinkRef,
    SceneObjectRef,
    SceneRegistry,
    SemanticEffectKind,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredHeldObjectEffect,
    RegisteredSemanticLowerer,
    RegisteredSemanticEffect,
    SemanticLowering,
    SemanticObjectTarget,
)

if TYPE_CHECKING:
    from embodichain.toolkits.graspkit import GraspPoseGenerator

__all__: list[str] = []

_ARTICULATION_LINK_SLIDE_CALL_ID = "simulation.articulation_link_slide"
_AXIS_ALIGN_CALL_ID = "simulation.axis_align"
_COORDINATED_TRANSPORT_CALL_ID = "simulation.coordinated_transport"
_MOVE_HELD_OBJECT_CALL_ID = "simulation.move_held_object"
_PARK_CALL_ID = "simulation.park"
_PLACE_RELATIVE_CALL_ID = "simulation.place_relative"
_POUR_CALL_ID = "simulation.pour"
_PUSH_OBJECT_CALL_ID = "simulation.push_object"
_SLIDE_TARGET_POSE_MODES = frozenset({"live", "snapshot"})


def _identifier(value: object, *, field_name: str) -> str:
    """Validate one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _axis(value: tuple[float, float, float]) -> tuple[float, float, float]:
    """Validate one legacy finite non-zero three-dimensional axis fallback."""
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("translation_axis must be an exact three-value tuple.")
    normalized = tuple(float(item) for item in value)
    if (
        not all(math.isfinite(item) for item in normalized)
        or math.sqrt(sum(item * item for item in normalized)) <= 1.0e-6
    ):
        raise ValueError("translation_axis must contain three finite non-zero values.")
    return normalized


def _pose(value: tuple[float, ...]) -> tuple[float, ...]:
    """Validate and own one flattened SE(3) transform."""
    if type(value) is not tuple or len(value) != 16:
        raise TypeError("relative_pose must be an exact 16-value tuple.")
    normalized = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError("relative_pose must contain only finite values.")
    pose = torch.tensor(normalized, dtype=torch.float64).reshape(4, 4)
    if not torch.allclose(
        pose[3],
        torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=torch.float64),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("relative_pose must have bottom row [0, 0, 0, 1].")
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
        raise ValueError("relative_pose must contain a proper SE(3) rotation.")
    return normalized


def _world_displacement(
    value: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Validate one finite non-zero world-frame displacement."""
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("world_displacement must be an exact three-value tuple.")
    normalized = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in normalized):
        raise ValueError("world_displacement must contain only finite values.")
    if math.sqrt(sum(item * item for item in normalized)) <= 1.0e-6:
        raise ValueError("world_displacement must be non-zero.")
    return normalized


def _slide_target_pose_mode(value: object) -> str:
    """Validate how a configured Slide resolves its target pose."""
    if type(value) is not str or value not in _SLIDE_TARGET_POSE_MODES:
        raise ValueError(
            f"target_pose_mode must be one of {sorted(_SLIDE_TARGET_POSE_MODES)}."
        )
    return value


class _ParkLowerer(RegisteredSemanticLowerer):
    """Lower one resource-scoped semantic park request."""

    call_id: ClassVar[str] = _PARK_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Construct a named joint goal whose posture stays in the profile."""
        del context, bound
        if type(option_template) is not MoveJointsOptions:
            raise TypeError(
                "Configured semantic park requires an exact "
                "MoveJointsOptions template."
            )
        if dict(call.arguments):
            raise ValueError(
                f"{self.call_id} arguments must be empty; the embodiment profile "
                "owns the parked posture."
            )
        return SemanticLowering(goal=JointPositionGoal(PARK_COMMAND))


class _AxisAlignLowerer(RegisteredSemanticLowerer):
    """Lower one configured object-upright request to ``AxisAlign``."""

    call_id: ClassVar[str] = _AXIS_ALIGN_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = AxisAlign.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.ATTACH

    def __init__(self, semantics: tuple[ObjectSemantics, ...]) -> None:
        if type(semantics) is not tuple or not semantics:
            raise ValueError("AxisAlign semantics must be a non-empty exact tuple.")
        self._semantics = {
            value.entity_id: value
            for value in semantics
            if isinstance(value.entity_id, str)
        }
        if len(self._semantics) != len(semantics):
            raise ValueError("AxisAlign semantics require unique scene entity IDs.")

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Construct one typed axis-alignment goal and attach effect."""
        del context, bound
        if type(option_template) is not AxisAlignOptions:
            raise TypeError(
                "Configured semantic axis alignment requires an exact "
                "AxisAlignOptions template."
            )
        arguments = dict(call.arguments)
        if set(arguments) != {"object"}:
            raise ValueError(f"{self.call_id} arguments must contain only 'object'.")
        object_id = arguments["object"]
        semantics = self._semantics.get(object_id)
        if semantics is None:
            raise ValueError(f"{self.call_id} does not declare object {object_id!r}.")
        return SemanticLowering(
            goal=AxisAlignGoal(semantics=semantics),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.ATTACH,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.ATTACHED,
                        object_id=object_id,
                        slot_id="primary",
                    ),
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _AxisAlignLowererFactory(RegisteredSemanticLowererFactory):
    """Create an axis-alignment lowerer from configured scene objects."""

    call_id: ClassVar[str] = _AXIS_ALIGN_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = AxisAlign.descriptor()

    object_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.object_ids) is not tuple or not self.object_ids:
            raise ValueError("AxisAlign object_ids must be a non-empty exact tuple.")
        normalized = tuple(
            _identifier(value, field_name=f"object_ids[{index}]")
            for index, value in enumerate(self.object_ids)
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("AxisAlign object_ids must be unique.")
        object.__setattr__(self, "object_ids", normalized)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Resolve exact axis-aware grasp semantics for every allowed object."""
        del simulation
        if engine.robot is not robot:
            raise ValueError("AxisAlign lowerer requires the engine's exact robot.")
        semantics: list[ObjectSemantics] = []
        for object_id in self.object_ids:
            object_ref = scene_registry.resolve(
                object_id,
                expected_type=SceneObjectRef,
            )
            grasp_ref = scene_registry.resolve_affordance(
                object_ref,
                capability=GRASP_AFFORDANCE_CAPABILITY,
            )
            object_semantics = scene_registry.object_semantics(
                object_ref,
                affordance=grasp_ref,
            )
            if type(object_semantics.affordance) is not AxisAlignAffordance:
                raise TypeError(
                    "Configured AxisAlign requires an AxisAlignAffordance grasp "
                    f"payload for {object_ref.entity_id!r}."
                )
            semantics.append(object_semantics)
        return _AxisAlignLowerer(tuple(semantics))


@dataclass(frozen=True, slots=True)
class _ParkLowererFactory(RegisteredSemanticLowererFactory):
    """Create the stateless profile-bound Park lowerer."""

    call_id: ClassVar[str] = _PARK_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Validate runtime ownership and return one fresh Park lowerer."""
        del simulation, scene_registry
        if engine.robot is not robot:
            raise ValueError("Park lowerer requires the engine's exact robot.")
        return _ParkLowerer()


@dataclass(frozen=True, slots=True)
class _AntipodalGraspPoseGeneratorFactory:
    """Own executable-free values and lazily create one fresh grasp service."""

    model_id: str
    min_opening_width: float
    max_opening_width: float
    finger_length: float
    finger_width: float
    finger_thickness: float
    palm_depth: float
    sample_count: int | None = None
    approach_deviation_angle: float | None = None
    approach_direction_samples: int | None = None
    max_candidates: int | None = None
    opening_margin: float | None = None
    point_sample_density: float | None = None
    filter_ground_collision: bool | None = None
    force_refresh: bool | None = None

    def __call__(self) -> GraspPoseGenerator:
        """Create a fresh configured antipodal grasp-pose generator."""
        from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
        from embodichain.toolkits.graspkit.pg_grasp import (
            AntipodalGraspPoseGenerator,
            AntipodalGraspPoseGeneratorCfg,
            GraspAnnotationCfg,
            ParallelJawGraspCollisionCfg,
        )

        algorithm_kwargs: dict[str, object] = {}
        if self.sample_count is not None:
            algorithm_kwargs["sample_count"] = self.sample_count
        if self.approach_deviation_angle is not None:
            algorithm_kwargs["approach_deviation_angle"] = self.approach_deviation_angle
        if self.approach_direction_samples is not None:
            algorithm_kwargs["approach_direction_samples"] = (
                self.approach_direction_samples
            )
        if self.max_candidates is not None:
            algorithm_kwargs["max_candidates"] = self.max_candidates

        collision_kwargs: dict[str, object] = {}
        if self.opening_margin is not None:
            collision_kwargs["opening_margin"] = self.opening_margin
        if self.point_sample_density is not None:
            collision_kwargs["point_sample_density"] = self.point_sample_density
        if self.filter_ground_collision is not None:
            collision_kwargs["filter_ground_collision"] = self.filter_ground_collision

        annotation_kwargs: dict[str, object] = {}
        if self.force_refresh is not None:
            annotation_kwargs["force_refresh"] = self.force_refresh

        return AntipodalGraspPoseGenerator(
            ParallelJawGripperModelCfg(
                model_id=self.model_id,
                min_opening_width=self.min_opening_width,
                max_opening_width=self.max_opening_width,
                finger_length=self.finger_length,
                finger_width=self.finger_width,
                finger_thickness=self.finger_thickness,
                palm_depth=self.palm_depth,
            ),
            algorithm_cfg=AntipodalGraspPoseGeneratorCfg(**algorithm_kwargs),
            collision_cfg=ParallelJawGraspCollisionCfg(**collision_kwargs),
            annotation_cfg=GraspAnnotationCfg(**annotation_kwargs),
        )


class _ArticulationLinkSlideLowerer(RegisteredSemanticLowerer):
    """Lower one configured articulation-link call to the built-in Slide skill."""

    call_id: ClassVar[str] = _ARTICULATION_LINK_SLIDE_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = Slide.descriptor()

    def __init__(
        self,
        semantics: ObjectSemantics,
        link_entity_id: str,
        *,
        target_pose_mode: str = "live",
    ) -> None:
        if not isinstance(semantics, ObjectSemantics):
            raise TypeError("semantics must be ObjectSemantics.")
        if not isinstance(semantics.affordance, SlideAffordance):
            raise TypeError("Articulation-link Slide requires a SlideAffordance.")
        self._semantics = semantics
        self._link_entity_id = _identifier(
            link_entity_id,
            field_name="link_entity_id",
        )
        self._target_pose_mode = _slide_target_pose_mode(target_pose_mode)

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Validate the configured payload and construct one typed Slide goal."""
        del bound
        if type(option_template) is not SlideOptions:
            raise TypeError(
                "Articulation-link Slide requires an exact SlideOptions template."
            )
        arguments = dict(call.arguments)
        configured_arguments = {"handle": self._link_entity_id}
        if arguments != configured_arguments:
            raise ValueError(
                f"{self.call_id} arguments must name only the configured handle; "
                "motion options belong to the selected policy preset."
            )
        target_pose: torch.Tensor | SceneEntityPose
        if self._target_pose_mode == "live":
            target_pose = SceneEntityPose(self._link_entity_id)
        else:
            try:
                observed_pose = context.scene.entities[self._link_entity_id].pose
            except KeyError as exc:
                raise KeyError(
                    "Articulation-link Slide snapshot target is absent from the "
                    f"planning scene: {self._link_entity_id!r}."
                ) from exc
            if not isinstance(observed_pose, torch.Tensor):
                raise TypeError(
                    "Articulation-link Slide snapshot target pose must be a tensor."
                )
            target_pose = observed_pose.clone()
        return SemanticLowering(
            goal=SlideGoal(
                semantics=self._semantics,
                target_pose=target_pose,
            )
        )


@dataclass(frozen=True, slots=True)
class _ArticulationLinkSlideLowererFactory(RegisteredSemanticLowererFactory):
    """Create Slide semantics from one configured articulation-link geometry."""

    call_id: ClassVar[str] = _ARTICULATION_LINK_SLIDE_CALL_ID
    revision: ClassVar[str] = "4"
    target_descriptor: ClassVar[SkillDescriptor] = Slide.descriptor()

    articulation_id: str
    articulation_simulation_uid: str
    link_entity_id: str
    translation_axis: tuple[float, float, float] | None = None
    """Optional compatibility axis that preserves the mesh-only legacy path."""

    target_pose_mode: str = "live"

    def __post_init__(self) -> None:
        for field_name in (
            "articulation_id",
            "articulation_simulation_uid",
            "link_entity_id",
        ):
            _identifier(getattr(self, field_name), field_name=field_name)
        if self.translation_axis is not None:
            object.__setattr__(self, "translation_axis", _axis(self.translation_axis))
        _slide_target_pose_mode(self.target_pose_mode)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Build one fresh lowerer from the configured live link."""
        if engine.robot is not robot:
            raise ValueError("Slide lowerer requires the engine's exact robot.")
        registration = scene_registry.lookup(
            self.link_entity_id,
            expected_type=SceneLinkRef,
        )
        expected_parent = SceneArticulationRef(self.articulation_id)
        if registration.parent != expected_parent:
            raise ValueError(
                f"Configured link {self.link_entity_id!r} must belong to "
                f"articulation {self.articulation_id!r}."
            )
        native_link_name = registration.native_name
        if type(native_link_name) is not str or not native_link_name:
            raise RuntimeError(
                f"Configured link {self.link_entity_id!r} has no native link name."
            )
        get_articulation = getattr(simulation, "get_articulation", None)
        if not callable(get_articulation):
            raise TypeError("simulation must provide get_articulation().")
        articulation = get_articulation(self.articulation_simulation_uid)
        if articulation is None:
            raise RuntimeError(
                "Configured Slide requires articulation "
                f"{self.articulation_simulation_uid!r}."
            )
        link_names = getattr(articulation, "link_names", ())
        if native_link_name not in link_names:
            raise RuntimeError(
                f"Articulation must expose link {native_link_name!r}; available "
                f"links are {sorted(link_names)}."
            )
        get_link_vert_face = getattr(articulation, "get_link_vert_face", None)
        if not callable(get_link_vert_face):
            raise TypeError("Articulation must provide get_link_vert_face().")
        vertices, triangles = get_link_vert_face(native_link_name)
        geometry: dict[str, object] = {}
        if self.translation_axis is None:
            # Explicit axes own the legacy mesh-only path: sampled geometry
            # would both override their sign and reject scaled articulations.
            geometry = sample_initial_articulation_geometry(
                articulation,
                native_link_name,
                initial_qpos=articulation.cfg.init_qpos,
                initial_qpos_joint_names=articulation.joint_names,
                body_scale=articulation.cfg.body_scale,
            ).to_object_geometry()
        affordance_kwargs: dict[str, object] = {}
        if self.translation_axis is not None:
            affordance_kwargs["translation_axis"] = torch.tensor(
                self.translation_axis,
                dtype=torch.float32,
                device=engine.device,
            )
        semantics = ObjectSemantics(
            label="articulation_link",
            entity_id=self.link_entity_id,
            geometry=geometry,
            affordance=SlideAffordance(
                mesh_vertices=torch.as_tensor(
                    vertices,
                    dtype=torch.float32,
                    device=engine.device,
                ),
                mesh_triangles=torch.as_tensor(
                    triangles,
                    dtype=torch.long,
                    device=engine.device,
                ),
                **affordance_kwargs,
            ),
        )
        return _ArticulationLinkSlideLowerer(
            semantics,
            self.link_entity_id,
            target_pose_mode=self.target_pose_mode,
        )


class _MoveHeldObjectLowerer(RegisteredSemanticLowerer):
    """Lower one configured live-relative target to ``MoveHeldObject``."""

    call_id: ClassVar[str] = _MOVE_HELD_OBJECT_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()

    def __init__(
        self,
        target_id: str,
        reference_entity_id: str,
        relative_pose: tuple[float, ...],
    ) -> None:
        self._target_id = _identifier(target_id, field_name="target_id")
        self._reference_entity_id = _identifier(
            reference_entity_id,
            field_name="reference_entity_id",
        )
        self._relative_pose = _pose(relative_pose)

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Construct one late-bound object-space transport goal."""
        del context, bound
        if type(option_template) is not MoveHeldObjectOptions:
            raise TypeError(
                "Configured held-object transport requires an exact "
                "MoveHeldObjectOptions template."
            )
        self._validate_call(call)
        return SemanticLowering(goal=HeldObjectPoseGoal(self._target_pose()))

    def pick_lookahead_targets(
        self,
        call: RegisteredSemanticCall,
        *,
        picked_object: SceneObjectRef,
        bound: BoundSemanticCall,
        previous_target: SemanticObjectTarget | None,
    ) -> tuple[SemanticObjectTarget, ...] | None:
        """Expose transport reachability while retaining the picked object."""
        del picked_object, bound, previous_target
        self._validate_call(call)
        return (SemanticObjectTarget(pose=self._target_pose()),)

    def _validate_call(self, call: RegisteredSemanticCall) -> None:
        """Require the configured immutable target selector."""
        arguments = dict(call.arguments)
        if arguments != {"target": self._target_id}:
            raise ValueError(
                f"{self.call_id} arguments must select configured target "
                f"{self._target_id!r}."
            )

    def _target_pose(self) -> SceneEntityPose:
        """Return one independently owned late-bound transport target."""
        return SceneEntityPose(
            self._reference_entity_id,
            relative_pose=torch.tensor(
                self._relative_pose,
                dtype=torch.float32,
            ).reshape(4, 4),
        )


@dataclass(frozen=True, slots=True)
class _MoveHeldObjectLowererFactory(RegisteredSemanticLowererFactory):
    """Create one configured live-relative ``MoveHeldObject`` lowerer."""

    call_id: ClassVar[str] = _MOVE_HELD_OBJECT_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveHeldObject.descriptor()

    target_id: str
    reference_entity_id: str
    relative_pose: tuple[float, ...]

    def __post_init__(self) -> None:
        _identifier(self.target_id, field_name="target_id")
        _identifier(self.reference_entity_id, field_name="reference_entity_id")
        object.__setattr__(self, "relative_pose", _pose(self.relative_pose))

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Build one lowerer after validating its live reference entity."""
        del simulation
        if engine.robot is not robot:
            raise ValueError(
                "MoveHeldObject lowerer requires the engine's exact robot."
            )
        scene_registry.lookup(self.reference_entity_id)
        return _MoveHeldObjectLowerer(
            self.target_id,
            self.reference_entity_id,
            self.relative_pose,
        )


@dataclass(frozen=True, slots=True)
class _RelativePlaceRoute:
    """One configured object relation expressed in the world frame."""

    object_id: str
    reference_entity_id: str
    relation: str
    world_displacement: tuple[float, float, float]

    def __post_init__(self) -> None:
        for field_name in ("object_id", "reference_entity_id", "relation"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name=field_name),
            )
        if self.relation not in {
            "above",
            "behind",
            "front_of",
            "left_of",
            "on",
            "right_of",
        }:
            raise ValueError(
                "Relative Place relation must be one of above, behind, front_of, "
                "left_of, on, or right_of."
            )
        object.__setattr__(
            self,
            "world_displacement",
            _world_displacement(self.world_displacement),
        )

    @property
    def selector(self) -> tuple[str, str, str]:
        """Return the semantic arguments selecting this immutable route."""
        return self.object_id, self.reference_entity_id, self.relation


class _RelativePlaceLowerer(RegisteredSemanticLowerer):
    """JIT-ground one object relation to the canonical ``Place`` skill."""

    call_id: ClassVar[str] = _PLACE_RELATIVE_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = Place.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.RELEASE

    def __init__(self, routes: tuple[_RelativePlaceRoute, ...]) -> None:
        if type(routes) is not tuple or not routes:
            raise ValueError("Relative Place routes must be a non-empty exact tuple.")
        self._routes = {route.selector: route for route in routes}
        if len(self._routes) != len(routes):
            raise ValueError("Relative Place routes must be unique.")

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Resolve fresh object/reference poses and preserve the held grasp."""
        if type(option_template) is not PlaceOptions:
            raise TypeError(
                "Configured semantic relative placement requires an exact "
                "PlaceOptions template."
            )
        route = self._resolve_route(call)

        resource = bound.binding.resources.get("primary")
        if resource is None:
            raise ValueError("Relative Place requires a bound primary resource.")
        motion_endpoint = resource.endpoints.get("motion")
        if motion_endpoint is None:
            raise ValueError("Relative Place requires a primary motion endpoint.")
        task_state_key = motion_endpoint.task_state_key
        if not isinstance(task_state_key, str):
            raise TypeError("Relative Place motion endpoint has no task-state key.")
        held = context.task.get_held_object(task_state_key)
        if held is None or held.semantics.entity_id != route.object_id:
            raise ValueError(
                f"Relative Place requires verified object {route.object_id!r} "
                f"held under task-state key {task_state_key!r}."
            )

        object_pose = self._observed_pose(context, route.object_id)
        reference_pose = self._observed_pose(context, route.reference_entity_id)
        object_target = object_pose.clone()
        displacement = torch.tensor(
            route.world_displacement,
            dtype=object_target.dtype,
            device=object_target.device,
        )
        object_target[:, :3, 3] = reference_pose[:, :3, 3] + displacement
        object_to_eef = held.object_to_eef.to(
            dtype=object_target.dtype,
            device=object_target.device,
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).expand(
                context.batch_size,
                -1,
                -1,
            )
        if object_to_eef.shape != (context.batch_size, 4, 4):
            raise ValueError(
                "Relative Place held-object transform must match the planning batch."
            )
        return SemanticLowering(
            goal=PlaceGoal(xpos=torch.bmm(object_target, object_to_eef)),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.RELEASE,
                held_objects=(
                    RegisteredHeldObjectEffect(
                        expectation_id="primary",
                        relation=HeldObjectRelation.DETACHED,
                        object_id=route.object_id,
                        slot_id="primary",
                    ),
                ),
            ),
        )

    def pick_lookahead_targets(
        self,
        call: RegisteredSemanticCall,
        *,
        picked_object: SceneObjectRef,
        bound: BoundSemanticCall,
        previous_target: SemanticObjectTarget | None,
    ) -> tuple[SemanticObjectTarget, ...] | None:
        """Expose the live-relative release target to an earlier Pick."""
        del previous_target
        route = self._resolve_route(call)
        if picked_object.entity_id != route.object_id:
            return None
        options = bound.preset.action_option_template(call.semantic_id)
        if type(options) is not PlaceOptions:
            raise TypeError(
                "Relative Place look-ahead requires an exact PlaceOptions template."
            )
        return (
            SemanticObjectTarget(
                pose=SceneEntityPose(
                    route.reference_entity_id,
                    world_displacement=torch.tensor(
                        route.world_displacement,
                        dtype=torch.float32,
                    ),
                ),
                preserve_current_object_orientation=(
                    options.preserve_current_object_orientation
                ),
            ),
        )

    def _resolve_route(self, call: RegisteredSemanticCall) -> _RelativePlaceRoute:
        """Validate one call and return its exact configured route."""
        arguments = dict(call.arguments)
        if set(arguments) != {"object", "reference", "relation"}:
            raise ValueError(
                f"{self.call_id} arguments must contain only 'object', "
                "'reference', and 'relation'."
            )
        selector = (
            arguments["object"],
            arguments["reference"],
            arguments["relation"],
        )
        route = self._routes.get(selector)
        if route is None:
            raise ValueError(
                f"{self.call_id} does not declare relation route {selector!r}."
            )
        return route

    @staticmethod
    def _observed_pose(context: PlanningContext, entity_id: str) -> torch.Tensor:
        """Return one positive-confidence entity pose in the planning batch."""
        try:
            observed = context.scene.entities[entity_id]
        except KeyError as exc:
            raise KeyError(
                f"Relative Place references unobserved entity {entity_id!r}."
            ) from exc
        if observed.confidence <= 0.0:
            raise ValueError(
                f"Relative Place requires positive confidence for {entity_id!r}."
            )
        pose = observed.pose.to(
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        if pose.shape == (4, 4):
            return pose.unsqueeze(0).expand(context.batch_size, -1, -1).clone()
        if pose.shape != (context.batch_size, 4, 4):
            raise ValueError(
                f"Relative Place entity {entity_id!r} pose must match the "
                "planning batch."
            )
        return pose.clone()


@dataclass(frozen=True, slots=True)
class _RelativePlaceLowererFactory(RegisteredSemanticLowererFactory):
    """Create fresh relative-placement lowerers from canonical scene refs."""

    call_id: ClassVar[str] = _PLACE_RELATIVE_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = Place.descriptor()

    routes: tuple[_RelativePlaceRoute, ...]

    def __post_init__(self) -> None:
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError("Relative Place routes must be a non-empty exact tuple.")
        if not all(type(route) is _RelativePlaceRoute for route in self.routes):
            raise TypeError("Relative Place routes must be _RelativePlaceRoute values.")
        if len({route.selector for route in self.routes}) != len(self.routes):
            raise ValueError("Relative Place routes must be unique.")

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Canonicalize all object/reference IDs before constructing a lowerer."""
        del simulation
        if engine.robot is not robot:
            raise ValueError(
                "Relative Place lowerer requires the engine's exact robot."
            )
        routes: list[_RelativePlaceRoute] = []
        for route in self.routes:
            object_ref = scene_registry.resolve(
                route.object_id,
                expected_type=SceneObjectRef,
            )
            reference_ref = scene_registry.resolve(
                route.reference_entity_id,
                expected_type=SceneObjectRef,
            )
            routes.append(
                _RelativePlaceRoute(
                    object_id=object_ref.entity_id,
                    reference_entity_id=reference_ref.entity_id,
                    relation=route.relation,
                    world_displacement=route.world_displacement,
                )
            )
        return _RelativePlaceLowerer(tuple(routes))


@dataclass(frozen=True, slots=True)
class _CoordinatedTransportRoute:
    """One configured absolute target or fresh world-frame displacement."""

    object_id: str
    target_id: str
    reference_entity_id: str | None = None
    relative_pose: tuple[float, ...] | None = None
    world_displacement: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_id",
            _identifier(self.object_id, field_name="object_id"),
        )
        object.__setattr__(
            self,
            "target_id",
            _identifier(self.target_id, field_name="target_id"),
        )
        has_reference = self.reference_entity_id is not None
        has_pose = self.relative_pose is not None
        has_displacement = self.world_displacement is not None
        if has_reference != has_pose or has_reference == has_displacement:
            raise ValueError(
                "Coordinated transport route must declare exactly one of "
                "reference_entity_id with relative_pose or world_displacement."
            )
        if has_reference:
            assert self.reference_entity_id is not None
            assert self.relative_pose is not None
            object.__setattr__(
                self,
                "reference_entity_id",
                _identifier(
                    self.reference_entity_id,
                    field_name="reference_entity_id",
                ),
            )
            object.__setattr__(self, "relative_pose", _pose(self.relative_pose))
        else:
            assert self.world_displacement is not None
            object.__setattr__(
                self,
                "world_displacement",
                _world_displacement(self.world_displacement),
            )


def _coordinated_transport_route(
    value: _CoordinatedTransportRoute | tuple[str, str, str, tuple[float, ...]],
    *,
    index: int,
) -> _CoordinatedTransportRoute:
    """Normalize the legacy private tuple used by direct lowerer tests."""
    if type(value) is _CoordinatedTransportRoute:
        return value
    if type(value) is not tuple or len(value) != 4:
        raise TypeError(
            f"Coordinated transport routes[{index}] must be a route or an "
            "exact four-value compatibility tuple."
        )
    return _CoordinatedTransportRoute(
        object_id=value[0],
        target_id=value[1],
        reference_entity_id=value[2],
        relative_pose=value[3],
    )


class _CoordinatedTransportLowerer(RegisteredSemanticLowerer):
    """Lower one configured dual-arm object transport and release route."""

    call_id: ClassVar[str] = _COORDINATED_TRANSPORT_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = CoordinatedPickment.descriptor()
    effect_contract_kind: ClassVar[SemanticEffectKind] = SemanticEffectKind.RELEASE

    def __init__(
        self,
        routes: tuple[
            _CoordinatedTransportRoute | tuple[str, str, str, tuple[float, ...]],
            ...,
        ],
        semantics: tuple[ObjectSemantics, ...],
    ) -> None:
        if len(routes) != len(semantics):
            raise ValueError(
                "Coordinated transport routes and semantics must have equal length."
            )
        normalized = tuple(
            _coordinated_transport_route(route, index=index)
            for index, route in enumerate(routes)
        )
        self._routes = {
            (route.object_id, route.target_id): (route, object_semantics)
            for route, object_semantics in zip(normalized, semantics, strict=True)
        }

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Construct one late-bound coordinated object target."""
        del bound
        if type(option_template) is not CoordinatedPickmentOptions:
            raise TypeError(
                "Configured coordinated transport requires an exact "
                "CoordinatedPickmentOptions template."
            )
        if not option_template.release:
            raise ValueError(
                "Configured coordinated transport must enable coordinated release."
            )
        arguments = dict(call.arguments)
        if set(arguments) != {"object", "target"}:
            raise ValueError(
                f"{self.call_id} arguments must contain only 'object' and 'target'."
            )
        route = (arguments["object"], arguments["target"])
        resolved = self._routes.get(route)
        if resolved is None:
            raise ValueError(
                f"{self.call_id} does not declare object-target route {route!r}."
            )
        route_cfg, semantics = resolved
        if route_cfg.world_displacement is not None:
            try:
                observed = context.scene.entities[route_cfg.object_id]
            except KeyError as exc:
                raise KeyError(
                    "Coordinated transport world displacement references "
                    f"unobserved object {route_cfg.object_id!r}."
                ) from exc
            if observed.confidence <= 0.0:
                raise ValueError(
                    "Coordinated transport requires positive observation "
                    f"confidence for {route_cfg.object_id!r}."
                )
            object_pose = observed.pose.to(
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            )
            if object_pose.dim() == 2:
                object_pose = object_pose.unsqueeze(0).expand(
                    context.batch_size,
                    -1,
                    -1,
                )
            object_target_pose: torch.Tensor | SceneEntityPose = object_pose.clone()
            displacement = torch.tensor(
                route_cfg.world_displacement,
                dtype=object_target_pose.dtype,
                device=object_target_pose.device,
            )
            object_target_pose[:, :3, 3] += displacement
        else:
            assert route_cfg.reference_entity_id is not None
            assert route_cfg.relative_pose is not None
            object_target_pose = SceneEntityPose(
                route_cfg.reference_entity_id,
                relative_pose=torch.tensor(
                    route_cfg.relative_pose,
                    dtype=torch.float32,
                ).reshape(4, 4),
            )
        return SemanticLowering(
            goal=CoordinatedPickGoal(
                semantics=semantics,
                object_target_pose=object_target_pose,
            ),
            registered_effect=RegisteredSemanticEffect(
                effect_kind=SemanticEffectKind.RELEASE,
                held_objects=tuple(
                    RegisteredHeldObjectEffect(
                        expectation_id=slot_id,
                        relation=HeldObjectRelation.DETACHED,
                        object_id=route_cfg.object_id,
                        slot_id=slot_id,
                        allow_missing_detached_baseline=True,
                    )
                    for slot_id in ("left", "right")
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class _CoordinatedTransportLowererFactory(RegisteredSemanticLowererFactory):
    """Create configured dual-arm transport routes from canonical scene refs."""

    call_id: ClassVar[str] = _COORDINATED_TRANSPORT_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = CoordinatedPickment.descriptor()

    routes: tuple[
        _CoordinatedTransportRoute | tuple[str, str, str, tuple[float, ...]],
        ...,
    ]

    def __post_init__(self) -> None:
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError(
                "Coordinated transport routes must be a non-empty exact tuple."
            )
        normalized = [
            _coordinated_transport_route(route, index=index)
            for index, route in enumerate(self.routes)
        ]
        selectors = [(route.object_id, route.target_id) for route in normalized]
        if len(set(selectors)) != len(selectors):
            raise ValueError("Coordinated transport routes must be unique.")
        object.__setattr__(self, "routes", tuple(normalized))

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Resolve grasp semantics and validate all live target references."""
        del simulation
        if engine.robot is not robot:
            raise ValueError(
                "Coordinated transport lowerer requires the engine's exact robot."
            )
        canonical_routes: list[_CoordinatedTransportRoute] = []
        semantics: list[ObjectSemantics] = []
        for route in self.routes:
            object_ref = scene_registry.resolve(
                route.object_id,
                expected_type=SceneObjectRef,
            )
            grasp_ref = scene_registry.resolve_affordance(
                object_ref,
                capability=GRASP_AFFORDANCE_CAPABILITY,
            )
            if route.world_displacement is not None:
                canonical_routes.append(
                    _CoordinatedTransportRoute(
                        object_id=object_ref.entity_id,
                        target_id=route.target_id,
                        world_displacement=route.world_displacement,
                    )
                )
            else:
                assert route.reference_entity_id is not None
                assert route.relative_pose is not None
                reference = scene_registry.lookup(route.reference_entity_id)
                canonical_routes.append(
                    _CoordinatedTransportRoute(
                        object_id=object_ref.entity_id,
                        target_id=route.target_id,
                        reference_entity_id=reference.ref.entity_id,
                        relative_pose=route.relative_pose,
                    )
                )
            semantics.append(
                scene_registry.object_semantics(
                    object_ref,
                    affordance=grasp_ref,
                )
            )
        return _CoordinatedTransportLowerer(
            tuple(canonical_routes),
            tuple(semantics),
        )


class _PourLowerer(RegisteredSemanticLowerer):
    """Lower one configured held-object pouring request."""

    call_id: ClassVar[str] = _POUR_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = Pour.descriptor()

    def __init__(
        self,
        object_id: str,
        internal_axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    ) -> None:
        self._object_id = _identifier(object_id, field_name="object_id")
        axis = torch.tensor(internal_axis, dtype=torch.float32)
        if axis.shape != (3,) or not torch.isfinite(axis).all():
            raise ValueError("internal_axis must contain three finite values.")
        magnitude = torch.linalg.vector_norm(axis)
        if magnitude <= 1.0e-6:
            raise ValueError("internal_axis must be non-zero.")
        self._internal_axis = axis / magnitude

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Construct the typed goal while policy presets own the tilt angle."""
        del context, bound
        if type(option_template) is not PourOptions:
            raise TypeError("Configured Pour requires an exact PourOptions template.")
        self._validate_call(call)
        return SemanticLowering(goal=PourGoal())

    def pick_lookahead_targets(
        self,
        call: RegisteredSemanticCall,
        *,
        picked_object: SceneObjectRef,
        bound: BoundSemanticCall,
        previous_target: SemanticObjectTarget | None,
    ) -> tuple[SemanticObjectTarget, ...] | None:
        """Expose the configured tilt and return for pickup feasibility."""
        self._validate_call(call)
        if picked_object.entity_id != self._object_id:
            return None
        if (
            previous_target is None
            or type(previous_target.pose) is not SceneEntityPose
            or previous_target.pose.relative_pose is None
        ):
            return ()
        options = bound.preset.action_option_template(call.semantic_id)
        if type(options) is not PourOptions:
            raise TypeError(
                "Configured Pour look-ahead requires an exact PourOptions template."
            )
        source = previous_target.pose
        tilted_relative = source.relative_pose.clone()
        local_rotation = axis_angle_to_rotation_matrix(
            self._internal_axis.to(tilted_relative) * options.rotate_angle
        )
        tilted_relative[:3, :3] = torch.matmul(
            tilted_relative[:3, :3],
            local_rotation,
        )
        tilted = SemanticObjectTarget(
            pose=SceneEntityPose(
                source.entity_id,
                relative_pose=tilted_relative,
                minimum_confidence=source.minimum_confidence,
            )
        )
        return (tilted, previous_target)

    def _validate_call(self, call: RegisteredSemanticCall) -> None:
        """Require the configured immutable held-object selector."""
        arguments = dict(call.arguments)
        if arguments != {"object": self._object_id}:
            raise ValueError(
                f"{self.call_id} arguments must name only configured object "
                f"{self._object_id!r}."
            )


@dataclass(frozen=True, slots=True)
class _PourLowererFactory(RegisteredSemanticLowererFactory):
    """Create a pouring lowerer for one axis-aware grasp object."""

    call_id: ClassVar[str] = _POUR_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = Pour.descriptor()

    object_id: str

    def __post_init__(self) -> None:
        _identifier(self.object_id, field_name="object_id")

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Validate the object's selected grasp semantics before construction."""
        del simulation
        if engine.robot is not robot:
            raise ValueError("Pour lowerer requires the engine's exact robot.")
        object_ref = scene_registry.resolve(
            self.object_id,
            expected_type=SceneObjectRef,
        )
        grasp_ref = scene_registry.resolve_affordance(
            object_ref,
            capability=GRASP_AFFORDANCE_CAPABILITY,
        )
        semantics = scene_registry.object_semantics(
            object_ref,
            affordance=grasp_ref,
        )
        if not isinstance(semantics.affordance, AxisAlignAffordance):
            raise TypeError(
                "Configured Pour requires an AxisAlignAffordance grasp payload."
            )
        internal_axis = semantics.affordance.internal_axis.detach().cpu().tolist()
        return _PourLowerer(
            self.object_id,
            tuple(float(value) for value in internal_axis),
        )


class _PushObjectLowerer(RegisteredSemanticLowerer):
    """Lower configured object-target routes to the built-in planar push."""

    call_id: ClassVar[str] = _PUSH_OBJECT_CALL_ID
    target_descriptor: ClassVar[SkillDescriptor] = PushObject.descriptor()

    def __init__(
        self,
        routes: tuple[tuple[str, str], ...],
        semantics: tuple[ObjectSemantics, ...],
    ) -> None:
        if len(routes) != len(semantics):
            raise ValueError("PushObject routes and semantics must have equal length.")
        self._routes = {
            route: object_semantics
            for route, object_semantics in zip(routes, semantics, strict=True)
        }

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Construct a late-bound object and target push goal."""
        del context, bound
        if type(option_template) is not PushObjectOptions:
            raise TypeError(
                "Configured planar pushing requires an exact PushObjectOptions "
                "template."
            )
        arguments = dict(call.arguments)
        if set(arguments) != {"object", "target"}:
            raise ValueError(
                f"{self.call_id} arguments must contain only 'object' and 'target'."
            )
        route = (arguments["object"], arguments["target"])
        semantics = self._routes.get(route)
        if semantics is None:
            raise ValueError(
                f"{self.call_id} does not declare object-target route {route!r}."
            )
        return SemanticLowering(
            goal=PushObjectGoal(
                semantics=semantics,
                target_pose=SceneEntityPose(route[1]),
            )
        )


@dataclass(frozen=True, slots=True)
class _PushObjectLowererFactory(RegisteredSemanticLowererFactory):
    """Create one lowerer for an immutable set of rigid-object push routes."""

    call_id: ClassVar[str] = _PUSH_OBJECT_CALL_ID
    revision: ClassVar[str] = "1"
    target_descriptor: ClassVar[SkillDescriptor] = PushObject.descriptor()

    routes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.routes) is not tuple or not self.routes:
            raise ValueError("PushObject routes must be a non-empty exact tuple.")
        normalized: list[tuple[str, str]] = []
        for index, route in enumerate(self.routes):
            if type(route) is not tuple or len(route) != 2:
                raise TypeError(
                    f"PushObject routes[{index}] must be an exact two-value tuple."
                )
            normalized.append(
                (
                    _identifier(route[0], field_name=f"routes[{index}].object_id"),
                    _identifier(
                        route[1],
                        field_name=f"routes[{index}].target_entity_id",
                    ),
                )
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("PushObject routes must be unique.")
        if len({object_id for object_id, _ in normalized}) != len(normalized):
            raise ValueError("PushObject routes must select each object at most once.")
        object.__setattr__(self, "routes", tuple(normalized))

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Validate configured scene entities and create one fresh lowerer."""
        del simulation
        if engine.robot is not robot:
            raise ValueError("PushObject lowerer requires the engine's exact robot.")
        canonical_routes: list[tuple[str, str]] = []
        semantics: list[ObjectSemantics] = []
        for object_id, target_entity_id in self.routes:
            object_registration = scene_registry.lookup(
                object_id,
                expected_type=SceneObjectRef,
            )
            target_registration = scene_registry.lookup(target_entity_id)
            canonical_routes.append(
                (
                    object_registration.ref.entity_id,
                    target_registration.ref.entity_id,
                )
            )
            semantics.append(
                ObjectSemantics(
                    affordance=Affordance(),
                    geometry={},
                    label=object_registration.semantic_type or "object",
                    entity_id=object_registration.ref.entity_id,
                )
            )
        return _PushObjectLowerer(tuple(canonical_routes), tuple(semantics))


class _JointPositionConstraintObserver:
    """Observe configured gripper constraints from measured joint displacement."""

    def __init__(
        self,
        robot: object,
        *,
        control_parts: tuple[str, ...],
        object_ids: tuple[str, ...],
        open_qpos: tuple[float, ...],
        minimum_displacement: float,
    ) -> None:
        self._robot = robot
        self._control_parts = frozenset(control_parts)
        self._object_ids = frozenset(object_ids)
        self._open_qpos = open_qpos
        self._minimum_displacement = minimum_displacement

    def __call__(
        self,
        query: BinaryEffectEvidenceQuery,
        context: EffectEvidenceCollectionContext,
    ) -> BinaryEffectObservation:
        """Return true for requested rows whose configured gripper has closed."""
        if type(query) is not BinaryEffectEvidenceQuery:
            raise TypeError(
                "Joint-position constraint evidence requires a binary query."
            )
        if query.clause.evidence_kind is not BinaryEvidenceKind.CONSTRAINT:
            raise ValueError(
                "Joint-position constraint evidence serves only constraint queries."
            )
        address = query.source.address
        if (
            type(address) is not ControlPartEvidenceAddress
            or address.control_part not in self._control_parts
            or address.channel != CONSTRAINT_EFFECT_CHANNEL
        ):
            raise ValueError(
                "Joint-position constraint evidence requires one configured "
                "control-part constraint route."
            )
        expectation = query.expectation
        if (
            type(expectation) is not HeldObjectStateExpectation
            or expectation.object_id not in self._object_ids
        ):
            raise ValueError(
                "Joint-position constraint evidence requires one configured "
                "held-object expectation."
            )

        get_qpos = getattr(self._robot, "get_qpos", None)
        if not callable(get_qpos):
            raise TypeError("Constraint evidence requires robot.get_qpos().")
        qpos = get_qpos(name=address.control_part)
        if not isinstance(qpos, torch.Tensor) or qpos.dim() != 2 or qpos.shape[1] == 0:
            raise ValueError("Control-part qpos must have non-empty shape (N, J).")
        if qpos.shape[1] != len(self._open_qpos):
            raise ValueError(
                f"Control-part qpos width {qpos.shape[1]} does not match configured "
                f"open_qpos width {len(self._open_qpos)}."
            )
        rows = context.env_ids.to(device=qpos.device)
        if bool((rows < 0).any()) or int(rows.max().item()) >= qpos.shape[0]:
            raise ValueError("Evidence env_ids must address valid robot rows.")
        measured = qpos.index_select(0, rows)
        open_qpos = torch.tensor(
            self._open_qpos,
            dtype=measured.dtype,
            device=measured.device,
        )
        displacement = torch.amax(torch.abs(measured - open_qpos), dim=1)
        return BinaryEffectObservation(
            values=(displacement >= self._minimum_displacement).to(
                device=context.env_ids.device
            )
        )


@dataclass(frozen=True, slots=True)
class _JointPositionConstraintEvidenceProviderFactory(
    ControlPartEvidenceProviderFactory
):
    """Create control-part evidence scoped to graspable scene objects."""

    provider_id: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_REVISION

    control_parts: tuple[str, ...]
    object_ids: tuple[str, ...] | None
    open_qpos: tuple[float, ...]
    minimum_displacement: float

    def __post_init__(self) -> None:
        if type(self.control_parts) is not tuple or not self.control_parts:
            raise ValueError("control_parts must be a non-empty exact tuple.")
        for control_part in self.control_parts:
            _identifier(control_part, field_name="control_parts")
        if len(set(self.control_parts)) != len(self.control_parts):
            raise ValueError("control_parts must not contain duplicates.")
        if self.object_ids is not None:
            if type(self.object_ids) is not tuple or not self.object_ids:
                raise ValueError("object_ids must be a non-empty exact tuple or None.")
            for object_id in self.object_ids:
                _identifier(object_id, field_name="object_ids")
            if len(set(self.object_ids)) != len(self.object_ids):
                raise ValueError("object_ids must not contain duplicates.")
        if type(self.open_qpos) is not tuple or not self.open_qpos:
            raise ValueError("open_qpos must be a non-empty exact tuple.")
        if not all(math.isfinite(value) for value in self.open_qpos):
            raise ValueError("open_qpos must contain only finite values.")
        if (
            not math.isfinite(self.minimum_displacement)
            or self.minimum_displacement <= 0.0
        ):
            raise ValueError("minimum_displacement must be finite and positive.")

    def _resolve_object_ids(self, scene_registry: SceneRegistry) -> tuple[str, ...]:
        """Resolve an optional allowlist against graspable scene objects."""
        if type(scene_registry) is not SceneRegistry:
            raise TypeError("scene_registry must be exactly SceneRegistry.")
        object_ids = self.object_ids
        if object_ids is None:
            object_ids = tuple(
                ref.entity_id
                for ref in scene_registry.entity_refs
                if type(ref) is SceneObjectRef
                and scene_registry.affordances(
                    ref,
                    capability=GRASP_AFFORDANCE_CAPABILITY,
                )
            )
        if not object_ids:
            raise ValueError(
                "Joint-position constraint evidence requires at least one "
                "graspable scene object."
            )
        resolved_object_ids: list[str] = []
        for object_id in object_ids:
            object_ref = scene_registry.resolve(
                object_id,
                expected_type=SceneObjectRef,
            )
            if not scene_registry.affordances(
                object_ref,
                capability=GRASP_AFFORDANCE_CAPABILITY,
            ):
                raise ValueError(
                    "Joint-position constraint evidence object "
                    f"{object_id!r} must expose a grasp affordance."
                )
            resolved_object_ids.append(object_ref.entity_id)
        if len(set(resolved_object_ids)) != len(resolved_object_ids):
            raise ValueError(
                "Joint-position constraint evidence object_ids must resolve to "
                "distinct scene objects."
            )
        return tuple(resolved_object_ids)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
        scene_provider: SceneProvider,
    ) -> EffectEvidenceProvider:
        """Bind a fresh evidence provider to the exact assembled robot."""
        from embodichain.lab.sim.objects import Robot

        del simulation
        if not isinstance(robot, Robot):
            raise TypeError("Constraint evidence requires a simulation Robot.")
        if engine.robot is not robot:
            raise ValueError("Constraint evidence requires the engine's exact robot.")
        object_ids = self._resolve_object_ids(scene_registry)
        return ControlPartSimulationEvidenceProvider(
            robot,
            scene_provider=scene_provider,
            constraint_observer=_JointPositionConstraintObserver(
                robot,
                control_parts=self.control_parts,
                object_ids=object_ids,
                open_qpos=self.open_qpos,
                minimum_displacement=self.minimum_displacement,
            ),
        )
