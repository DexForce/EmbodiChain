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

"""Open Drawer through a registered Expert Program call lowered to Slide."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    RegisteredSemanticLowererFactory,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationExpertProgramAdapterFactory,
    SimulationExpertProgramRegistration,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
    AtomicActionEngine,
    ObjectSemantics,
    PlanningContext,
    SceneEntityPose,
    SkillDescriptor,
    Slide,
    SlideAffordance,
    SlideGoal,
    SlideOptions,
)
from embodichain.lab.sim.skills import (
    BoundSemanticCall,
    RegisteredSemanticCall,
    RegisteredSemanticLowerer,
    SceneDynamics,
    SceneRegistry,
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SemanticLowering,
    builtin_semantic_call_catalog,
)

from ._common import (
    HAND_CONTROL_PART,
    create_parallel_jaw_grasp_pose_generator,
    create_ur5_skill_profile_binding,
    load_bundled_expert_program,
)

__all__ = [
    "ExpertProgramOpenDrawerEnv",
    "OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION",
    "create_open_drawer_robot_profile_binding",
    "create_open_drawer_scene_binding",
]

ENV_ID = "ExpertProgramOpenDrawer-v1"
PROGRAM_FILENAME = "open_drawer.yaml"
OPEN_DRAWER_CALL_ID = "embodichain_tasks.open_drawer"
DRAWER_ENTITY_ID = "drawer"
HANDLE_ENTITY_ID = "drawer_handle"
HANDLE_LINK_NAME = "large_handle_bar"
ROBOT_PROFILE_ID = "expert_program_ur5_slide"
SCENE_REGISTRY_ID = "expert_program_open_drawer"
SAFE_MOTION_SAMPLE_COUNT = 140
DEFAULT_GRASP_SAMPLES = 10_000
DEFAULT_TRANSLATION_AXIS = (0.0, 1.0, 0.0)

_SLIDE_DESCRIPTOR = Slide.descriptor()


def _axis(value: object) -> tuple[float, float, float]:
    """Return one finite non-zero three-dimensional axis."""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("translation_axis must be a three-value sequence.")
    values = tuple(float(item) for item in value)
    if (
        len(values) != 3
        or not all(math.isfinite(item) for item in values)
        or math.sqrt(sum(item * item for item in values)) <= 1.0e-6
    ):
        raise ValueError("translation_axis must contain three finite non-zero values.")
    return values


class _OpenDrawerSlideLowerer(RegisteredSemanticLowerer):
    """Lower the task's safe declarative payload to the built-in Slide action."""

    call_id: ClassVar[str] = OPEN_DRAWER_CALL_ID
    schema_version: ClassVar[int] = 1
    target_descriptor: ClassVar[SkillDescriptor] = _SLIDE_DESCRIPTOR

    def __init__(self, semantics: ObjectSemantics) -> None:
        if not isinstance(semantics, ObjectSemantics):
            raise TypeError("semantics must be ObjectSemantics.")
        if not isinstance(semantics.affordance, SlideAffordance):
            raise TypeError("Open Drawer semantics require a SlideAffordance.")
        self._semantics = semantics

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        """Validate the registered payload and construct one typed Slide goal."""
        del context, bound
        if type(option_template) is not SlideOptions:
            raise TypeError("Open Drawer requires an exact SlideOptions template.")
        arguments = dict(call.arguments)
        task_arguments = {"handle": HANDLE_ENTITY_ID}
        legacy_arguments = {
            **task_arguments,
            "direction": option_template.direction,
            "hand_interp_steps": option_template.hand_interp_steps,
            "approach_distance": option_template.approach_distance,
            "translation_distance": option_template.translation_distance,
        }
        if arguments not in (task_arguments, legacy_arguments):
            raise ValueError(
                f"{OPEN_DRAWER_CALL_ID} arguments must name only the canonical "
                "handle; legacy option fields, when present, must exactly match "
                "the selected policy preset."
            )
        handle = arguments["handle"]
        assert type(handle) is str
        return SemanticLowering(
            goal=SlideGoal(
                semantics=self._semantics,
                target_pose=SceneEntityPose(handle),
            )
        )


@dataclass(frozen=True, slots=True)
class _OpenDrawerSlideLowererFactory(RegisteredSemanticLowererFactory):
    """Create live Slide semantics from the registered drawer handle mesh."""

    call_id: ClassVar[str] = OPEN_DRAWER_CALL_ID
    revision: ClassVar[str] = "1"
    articulation_id: str = DRAWER_ENTITY_ID
    handle_entity_id: str = HANDLE_ENTITY_ID
    handle_link_name: str = HANDLE_LINK_NAME
    translation_axis: tuple[float, float, float] = DEFAULT_TRANSLATION_AXIS

    def __post_init__(self) -> None:
        for name in ("articulation_id", "handle_entity_id", "handle_link_name"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact non-empty identifier.")
        object.__setattr__(self, "translation_axis", _axis(self.translation_axis))

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Build one fresh lowerer from the exact live registered handle."""
        if engine.robot is not robot:
            raise ValueError("Open Drawer requires the engine's exact robot.")
        scene_registry.resolve(self.handle_entity_id)
        get_articulation = getattr(simulation, "get_articulation", None)
        if not callable(get_articulation):
            raise TypeError("simulation must provide get_articulation().")
        drawer = get_articulation(self.articulation_id)
        if drawer is None:
            raise RuntimeError(
                f"{ENV_ID} requires articulation {self.articulation_id!r}."
            )
        link_names = getattr(drawer, "link_names", ())
        if self.handle_link_name not in link_names:
            raise RuntimeError(
                f"Drawer must expose link {self.handle_link_name!r}; available "
                f"links are {sorted(link_names)}."
            )
        get_link_vert_face = getattr(drawer, "get_link_vert_face", None)
        if not callable(get_link_vert_face):
            raise TypeError("Drawer articulation must provide get_link_vert_face().")
        vertices, triangles = get_link_vert_face(self.handle_link_name)
        semantics = ObjectSemantics(
            label="drawer_handle",
            entity_id=self.handle_entity_id,
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
        return _OpenDrawerSlideLowerer(semantics)


def _open_drawer_call_catalog() -> SemanticCallCatalog:
    """Extend the built-in semantic catalog with the task-owned Slide call."""
    return builtin_semantic_call_catalog().with_descriptor(
        SemanticCallDescriptor(
            call_id=OPEN_DRAWER_CALL_ID,
            spec_type=RegisteredSemanticCall,
            target_descriptor=_SLIDE_DESCRIPTOR,
        )
    )


def create_open_drawer_scene_binding() -> SimulationSceneBinding:
    """Declare the provider-free drawer and handle scene integration.

    Returns:
        Immutable scene binding used by the Open Drawer registration.
    """
    return SimulationSceneBinding(
        registry_id=SCENE_REGISTRY_ID,
        articulations=(
            SimulationArticulationBinding(
                entity_id=DRAWER_ENTITY_ID,
                simulation_uid=DRAWER_ENTITY_ID,
                dynamics=SceneDynamics.DYNAMIC,
                semantic_type="drawer",
            ),
        ),
        links=(
            SimulationArticulationLinkBinding(
                entity_id=HANDLE_ENTITY_ID,
                articulation_id=DRAWER_ENTITY_ID,
                native_link_name=HANDLE_LINK_NAME,
                dynamics=SceneDynamics.DYNAMIC,
                semantic_type="drawer_handle",
            ),
        ),
    )


def create_open_drawer_robot_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Declare the provider-free UR5 Slide profile.

    Returns:
        Immutable robot-profile binding used by the Open Drawer registration.
    """
    return create_ur5_skill_profile_binding(
        None,
        profile_id=ROBOT_PROFILE_ID,
        sample_count=SAFE_MOTION_SAMPLE_COUNT,
        skill_ids=("slide",),
        action_option_templates={
            OPEN_DRAWER_CALL_ID: SlideOptions(
                direction="pull",
                hand_interp_steps=12,
                approach_distance=0.10,
                translation_distance=0.18,
            )
        },
    )


#: Immutable standard-path registration for ``ExpertProgramOpenDrawer-v1``.
OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION = SimulationExpertProgramRegistration(
    scene_binding=create_open_drawer_scene_binding(),
    robot_profile_binding=create_open_drawer_robot_profile_binding(),
    call_catalog=_open_drawer_call_catalog(),
    registered_semantic_lowerer_factories=(_OpenDrawerSlideLowererFactory(),),
)


def _create_open_drawer_grasp_pose_generator():
    """Create the task's fresh parallel-jaw grasp service."""
    return create_parallel_jaw_grasp_pose_generator(
        sample_count=DEFAULT_GRASP_SAMPLES,
        opening_margin=0.03,
    )


_OPEN_DRAWER_EXPERT_PROGRAM_ADAPTER_FACTORY = SimulationExpertProgramAdapterFactory(
    OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION,
    grasp_pose_generator_factories={
        HAND_CONTROL_PART: _create_open_drawer_grasp_pose_generator,
    },
)


@register_env(
    ENV_ID,
    max_episode_steps=600,
    expert_program_adapter_factory=_OPEN_DRAWER_EXPERT_PROGRAM_ADAPTER_FACTORY,
)
class ExpertProgramOpenDrawerEnv(EmbodiedEnv):
    """Open a passive drawer through Expert Program and the atomic Slide skill.

    ``Slide`` owns only approach/grasp/translation motion and intentionally has
    no drawer-state effect. The segment's standard articulation-joint validator
    reads the passive drawer state after settling.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize the registration-owned Expert Program adapter.

        Args:
            cfg: Environment configuration. The bundled program is installed
                when ``cfg.expert_program`` is unset.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.

        """
        if cfg.expert_program is None:
            cfg.expert_program = load_bundled_expert_program(PROGRAM_FILENAME)

        kwargs.setdefault(
            "expert_program_adapter_factory",
            _OPEN_DRAWER_EXPERT_PROGRAM_ADAPTER_FACTORY,
        )
        super().__init__(cfg, **kwargs)
