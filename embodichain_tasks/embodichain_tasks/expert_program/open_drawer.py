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
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    ExpertProgramEnvironmentAdapter,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationExpertProgramFactory,
    SimulationSceneBinding,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
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

from ._common import (
    HAND_CONTROL_PART,
    create_parallel_jaw_grasp_pose_generator,
    create_ur5_skill_profile_binding,
    load_bundled_expert_program,
)

__all__ = ["ExpertProgramOpenDrawerEnv"]

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


def _finite_float(value: object, *, field_name: str, positive: bool) -> float:
    """Return one finite declarative number."""
    if type(value) not in (int, float):
        raise TypeError(f"{field_name} must be an int or float.")
    normalized = float(value)
    if not math.isfinite(normalized) or (positive and normalized <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{field_name} must be {qualifier}.")
    return normalized


def _positive_int(value: object, *, field_name: str) -> int:
    """Return one exact positive integer."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


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
    ) -> SemanticLowering:
        """Validate the registered payload and construct one typed Slide goal."""
        del context, bound
        expected_keys = {
            "handle",
            "direction",
            "hand_interp_steps",
            "approach_distance",
            "translation_distance",
        }
        if set(call.arguments) != expected_keys:
            raise ValueError(
                f"{OPEN_DRAWER_CALL_ID} arguments must be exactly "
                f"{sorted(expected_keys)}."
            )
        handle = call.arguments["handle"]
        if type(handle) is not str or handle != HANDLE_ENTITY_ID:
            raise ValueError(
                f"handle must be exactly the canonical ID {HANDLE_ENTITY_ID!r}."
            )
        direction = call.arguments["direction"]
        if type(direction) is not str or direction not in ("pull", "push"):
            raise ValueError("direction must be exactly 'pull' or 'push'.")
        hand_interp_steps = _positive_int(
            call.arguments["hand_interp_steps"],
            field_name="hand_interp_steps",
        )
        approach_distance = _finite_float(
            call.arguments["approach_distance"],
            field_name="approach_distance",
            positive=False,
        )
        if approach_distance < 0.0:
            raise ValueError("approach_distance must be non-negative.")
        translation_distance = _finite_float(
            call.arguments["translation_distance"],
            field_name="translation_distance",
            positive=True,
        )
        return SemanticLowering(
            goal=SlideGoal(
                semantics=self._semantics,
                target_pose=SceneEntityPose(HANDLE_ENTITY_ID),
            ),
            skill_options=SlideOptions(
                direction=direction,
                hand_interp_steps=hand_interp_steps,
                approach_distance=approach_distance,
                translation_distance=translation_distance,
            ),
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


@register_env(ENV_ID, max_episode_steps=600)
class ExpertProgramOpenDrawerEnv(EmbodiedEnv):
    """Open a passive drawer through Expert Program and the atomic Slide skill.

    ``Slide`` owns only approach/grasp/translation motion and intentionally has
    no drawer-state effect. The segment's standard articulation-joint validator
    reads the passive drawer state after settling.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize live drawer semantics and the registered-call adapter.

        Args:
            cfg: Environment configuration. The bundled program is installed
                when ``cfg.expert_program`` is unset.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.

        Raises:
            TypeError: If task extensions have invalid types.
            ValueError: If numeric settings or the selected drawer joint are
                invalid.
            RuntimeError: If the configured drawer or handle link is absent.
        """
        if cfg.expert_program is None:
            cfg.expert_program = load_bundled_expert_program(PROGRAM_FILENAME)

        extensions = self._extensions(cfg.extensions)
        translation_axis = _axis(
            extensions.get("translation_axis", DEFAULT_TRANSLATION_AXIS)
        )

        super().__init__(cfg, **kwargs)

        drawer = self.sim.get_articulation(DRAWER_ENTITY_ID)
        if drawer is None:
            raise RuntimeError(f"{ENV_ID} requires articulation {DRAWER_ENTITY_ID!r}.")
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
                    vertices, dtype=torch.float32, device=self.device
                ),
                mesh_triangles=torch.as_tensor(
                    triangles, dtype=torch.long, device=self.device
                ),
                translation_axis=torch.tensor(
                    translation_axis,
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
        )
        lowerer = _OpenDrawerSlideLowerer(semantics)
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
            self.robot,
            profile_id=ROBOT_PROFILE_ID,
            sample_count=SAFE_MOTION_SAMPLE_COUNT,
            skill_ids=("slide",),
        )
        factory = SimulationExpertProgramFactory.from_environment(
            self,
            scene_binding=scene_binding,
            robot_profile_binding=profile_binding,
            grasp_pose_generators={HAND_CONTROL_PART: grasp_pose_generator},
        )
        self._expert_program_adapter = factory.create_adapter(
            call_catalog=_open_drawer_call_catalog(),
            registered_lowerers=(lowerer,),
        )

    @staticmethod
    def _extensions(value: object) -> Mapping[str, object]:
        """Return the task extension mapping without accepting loose values."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("cfg.extensions must be a mapping.")
        return value

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the exact production adapter used by ``EmbodiedEnv``.

        Returns:
            Adapter that compiles and executes the configured Expert Program.
        """
        return self._expert_program_adapter
