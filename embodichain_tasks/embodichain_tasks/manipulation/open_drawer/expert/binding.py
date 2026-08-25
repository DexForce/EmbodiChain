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

"""Runtime bindings for the Open Drawer Expert Program."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import ClassVar

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramEnvironmentAdapter,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationExpertProgramFactory,
    SimulationSceneBinding,
)
from embodichain.lab.sim.atomic_actions import (
    ActionOptions,
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
    SemanticCallCatalog,
    SemanticCallDescriptor,
    SemanticLowering,
    builtin_semantic_call_catalog,
)

from ..._expert import (
    HAND_CONTROL_PART,
    create_parallel_jaw_grasp_pose_generator,
    create_ur5_skill_profile_binding,
)

__all__ = ["create_open_drawer_expert_program_adapter"]

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


def _open_drawer_call_catalog() -> SemanticCallCatalog:
    """Extend the built-in semantic catalog with the task-owned Slide call."""
    return builtin_semantic_call_catalog().with_descriptor(
        SemanticCallDescriptor(
            call_id=OPEN_DRAWER_CALL_ID,
            spec_type=RegisteredSemanticCall,
            target_descriptor=_SLIDE_DESCRIPTOR,
        )
    )


def create_open_drawer_expert_program_adapter(
    environment: EmbodiedEnv,
    *,
    translation_axis: object = DEFAULT_TRANSLATION_AXIS,
) -> ExpertProgramEnvironmentAdapter:
    """Bind the configured drawer scene to the semantic Slide runtime.

    Args:
        environment: Initialized Open Drawer environment.
        translation_axis: Drawer-handle translation axis from task extensions.

    Returns:
        Adapter that compiles and executes the Open Drawer Expert Program.

    Raises:
        RuntimeError: If the configured drawer or handle link is absent.
    """
    normalized_axis = _axis(translation_axis)
    drawer = environment.sim.get_articulation(DRAWER_ENTITY_ID)
    if drawer is None:
        raise RuntimeError(
            "ExpertProgramOpenDrawer-v1 requires articulation " f"{DRAWER_ENTITY_ID!r}."
        )
    if HANDLE_LINK_NAME not in drawer.link_names:
        raise RuntimeError(
            f"Drawer must expose link {HANDLE_LINK_NAME!r}; available links "
            f"are {sorted(drawer.link_names)}."
        )
    vertices, triangles = drawer.get_link_vert_face(HANDLE_LINK_NAME)
    grasp_pose_generator = create_parallel_jaw_grasp_pose_generator(
        sample_count=DEFAULT_GRASP_SAMPLES,
        opening_margin=0.03,
    )
    semantics = ObjectSemantics(
        label="drawer_handle",
        entity_id=HANDLE_ENTITY_ID,
        geometry={},
        affordance=SlideAffordance(
            mesh_vertices=torch.as_tensor(
                vertices, dtype=torch.float32, device=environment.device
            ),
            mesh_triangles=torch.as_tensor(
                triangles, dtype=torch.long, device=environment.device
            ),
            translation_axis=torch.tensor(
                normalized_axis,
                dtype=torch.float32,
                device=environment.device,
            ),
        ),
    )
    scene_binding = SimulationSceneBinding(
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
    profile_binding = create_ur5_skill_profile_binding(
        environment.robot,
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
    factory = SimulationExpertProgramFactory.from_environment(
        environment,
        scene_binding=scene_binding,
        robot_profile_binding=profile_binding,
        grasp_pose_generators={HAND_CONTROL_PART: grasp_pose_generator},
    )
    return factory.create_adapter(
        call_catalog=_open_drawer_call_catalog(),
        registered_lowerers=(_OpenDrawerSlideLowerer(semantics),),
    )
