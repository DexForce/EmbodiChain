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

"""Allowlisted live services for configured Expert Program runtimes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, ClassVar

import torch

from embodichain.lab.gym.envs.expert_program.extensions import (
    ControlPartEvidenceProviderFactory,
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    ObjectSemantics,
    PlanningContext,
    SceneEntityPose,
    SceneProvider,
    SkillDescriptor,
    Slide,
    SlideAffordance,
    SlideGoal,
    SlideOptions,
)
from embodichain.lab.sim.skills import (
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
    HeldObjectStateExpectation,
    RegisteredSemanticCall,
    RegisteredSemanticLowerer,
    SceneArticulationRef,
    SceneLinkRef,
    SceneRegistry,
    SemanticLowering,
)

if TYPE_CHECKING:
    from embodichain.toolkits.graspkit import GraspPoseGenerator

__all__: list[str] = []

_ARTICULATION_LINK_SLIDE_CALL_ID = "simulation.articulation_link_slide"
_SLIDE_TARGET_POSE_MODES = frozenset({"live", "snapshot"})


def _identifier(value: object, *, field_name: str) -> str:
    """Validate one exact non-empty identifier."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{field_name} must be a non-empty string without outer whitespace."
        )
    return value


def _axis(value: tuple[float, float, float]) -> tuple[float, float, float]:
    """Validate one finite non-zero three-dimensional axis."""
    if type(value) is not tuple or len(value) != 3:
        raise TypeError("translation_axis must be an exact three-value tuple.")
    normalized = tuple(float(item) for item in value)
    if (
        not all(math.isfinite(item) for item in normalized)
        or math.sqrt(sum(item * item for item in normalized)) <= 1.0e-6
    ):
        raise ValueError("translation_axis must contain three finite non-zero values.")
    return normalized


def _slide_target_pose_mode(value: object) -> str:
    """Validate how a configured Slide resolves its target pose."""
    if type(value) is not str or value not in _SLIDE_TARGET_POSE_MODES:
        raise ValueError(
            f"target_pose_mode must be one of {sorted(_SLIDE_TARGET_POSE_MODES)}."
        )
    return value


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
    approach_direction_samples: int | None = None
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
        if self.approach_direction_samples is not None:
            algorithm_kwargs["approach_direction_samples"] = (
                self.approach_direction_samples
            )

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
    """Create Slide semantics from one configured articulation-link mesh."""

    call_id: ClassVar[str] = _ARTICULATION_LINK_SLIDE_CALL_ID
    revision: ClassVar[str] = "2"

    articulation_id: str
    articulation_simulation_uid: str
    link_entity_id: str
    translation_axis: tuple[float, float, float]
    target_pose_mode: str = "live"

    def __post_init__(self) -> None:
        for field_name in (
            "articulation_id",
            "articulation_simulation_uid",
            "link_entity_id",
        ):
            _identifier(getattr(self, field_name), field_name=field_name)
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
        semantics = ObjectSemantics(
            label="articulation_link",
            entity_id=self.link_entity_id,
            geometry={},
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
                translation_axis=torch.tensor(
                    self.translation_axis,
                    dtype=torch.float32,
                    device=engine.device,
                ),
            ),
        )
        return _ArticulationLinkSlideLowerer(
            semantics,
            self.link_entity_id,
            target_pose_mode=self.target_pose_mode,
        )


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
    """Create built-in control-part evidence from configured joint positions."""

    provider_id: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_ID
    revision: ClassVar[str] = CONTROL_PART_EVIDENCE_PROVIDER_REVISION

    control_parts: tuple[str, ...]
    object_ids: tuple[str, ...]
    open_qpos: tuple[float, ...]
    minimum_displacement: float

    def __post_init__(self) -> None:
        for field_name in ("control_parts", "object_ids"):
            values = getattr(self, field_name)
            if type(values) is not tuple or not values:
                raise ValueError(f"{field_name} must be a non-empty exact tuple.")
            for value in values:
                _identifier(value, field_name=field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must not contain duplicates.")
        if type(self.open_qpos) is not tuple or not self.open_qpos:
            raise ValueError("open_qpos must be a non-empty exact tuple.")
        if not all(math.isfinite(value) for value in self.open_qpos):
            raise ValueError("open_qpos must contain only finite values.")
        if (
            not math.isfinite(self.minimum_displacement)
            or self.minimum_displacement <= 0.0
        ):
            raise ValueError("minimum_displacement must be finite and positive.")

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

        del simulation, scene_registry
        if not isinstance(robot, Robot):
            raise TypeError("Constraint evidence requires a simulation Robot.")
        if engine.robot is not robot:
            raise ValueError("Constraint evidence requires the engine's exact robot.")
        return ControlPartSimulationEvidenceProvider(
            robot,
            scene_provider=scene_provider,
            constraint_observer=_JointPositionConstraintObserver(
                robot,
                control_parts=self.control_parts,
                object_ids=self.object_ids,
                open_qpos=self.open_qpos,
                minimum_displacement=self.minimum_displacement,
            ),
        )
