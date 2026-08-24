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

"""Open the CobotMagic drawer through a declarative Expert Program.

The task-owned registered call is lowered to the built-in ``Slide`` action.
``Slide`` owns approach, grasp, and pull motion; the packaged program owns the
physical drawer-joint validator that decides task success.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.expert_program import (
    ControlPartCommandPreset,
    ControlPartEndpointBinding,
    ControlPartResourceBinding,
    ExpertProgramEnvironmentAdapter,
    SimulationArticulationBinding,
    SimulationArticulationLinkBinding,
    SimulationExpertProgramFactory,
    SimulationRobotSkillProfileBinding,
    SimulationSceneBinding,
    load_expert_program,
)
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    CARTESIAN_POSE_CAPABILITY,
    GRASP_CAPABILITY,
    MotionPolicy,
    ObjectSemantics,
    RecoveryPolicy,
    SlideAffordance,
)
from embodichain.lab.sim.skills import SceneCollisionRole, SceneDynamics
from embodichain.lab.sim.skills.profiles import SkillPolicyPreset
from embodichain.toolkits.graspkit import GraspPoseGenerator
from embodichain_tasks.configs import get_config_path
from embodichain_tasks.expert_program.open_drawer import (
    _OpenDrawerSlideLowerer,
    _open_drawer_call_catalog,
)

__all__ = [
    "OpenDrawerEnv",
    "create_open_drawer_robot_profile_binding",
    "create_open_drawer_scene_binding",
]

DRAWER_SCENE_REGISTRY_ID = "open_drawer_v2"
DRAWER_ROBOT_PROFILE_ID = "cobot_magic_right_manipulator_v2"
DRAWER_UID = "drawer"
DRAWER_HANDLE_LINK_ID = "drawer_handle"
DRAWER_HANDLE_AFFORDANCE_ID = DRAWER_HANDLE_LINK_ID
DRAWER_NATIVE_HANDLE_LINK = "handle_xpos"
DRAWER_NATIVE_MESH_LINK = "inner_box"
DRAWER_NATIVE_SLIDE_JOINT = "slide_rails"
# The original PR accepted its 0.11 m target with a 0.02 m effect tolerance.
# The explicit validator preserves that physical acceptance boundary.
DRAWER_OPEN_POSITION = 0.09
DRAWER_OPEN_DISPLACEMENT = 0.12
DRAWER_EXPERT_PROGRAM_PATH = Path("expert_program/tableware/open_drawer.json")
RIGHT_MANIPULATOR_RESOURCE_ID = "right_manipulator"
RIGHT_ARM_CONTROL_PART = "right_arm"
RIGHT_EEF_CONTROL_PART = "right_eef"
SAFE_MOTION_SAMPLE_COUNT = 190
_HANDLE_MESH_MIN_PARENT_X = 0.10
_HANDLE_ORIGIN_IN_PARENT = (0.105, 0.0, 0.10)
# The 1 cm +Z depth keeps the physical handle inside the jaws while preserving
# the historical pull endpoint when paired with the 0.12 m Slide distance.
_CALIBRATED_HANDLE_GRASP_POSE = (
    (-0.023958006, -0.999453075, -0.022793945, 0.00049425),
    (0.999712744, -0.023966955, 0.000119456, -0.00441209),
    (-0.000665692, -0.022784535, 0.999740177, 0.01492312),
    (0.0, 0.0, 0.0, 1.0),
)
_CALIBRATED_HANDLE_OPENING_WIDTH = 0.01

# Positive is approach/push. In the handle frame, +Z maps to the parent's -X,
# so the registered ``pull`` call moves along the drawer joint's opening +X.
DEFAULT_TRANSLATION_AXIS = (0.0, 0.0, 1.0)


class _CalibratedDrawerHandleGraspPoseGenerator(GraspPoseGenerator):
    """Ground the legacy drawer's known handle-to-TCP calibration."""

    @staticmethod
    def _grasp_poses(obj_poses: torch.Tensor) -> torch.Tensor:
        """Apply the calibrated local TCP pose to a handle-pose batch."""
        if (
            not isinstance(obj_poses, torch.Tensor)
            or not obj_poses.is_floating_point()
            or obj_poses.dim() != 3
            or obj_poses.shape[0] == 0
            or obj_poses.shape[1:] != (4, 4)
            or not bool(torch.isfinite(obj_poses).all().item())
        ):
            raise ValueError(
                "obj_poses must be a non-empty finite floating tensor with "
                "shape (B, 4, 4)."
            )
        local_pose = obj_poses.new_tensor(_CALIBRATED_HANDLE_GRASP_POSE)
        return torch.matmul(obj_poses, local_pose)

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        object_part: str = "center",
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return the single calibrated candidate for each handle pose."""
        del mesh_vertices, mesh_triangles, approach_direction, object_part
        return [
            (
                pose.unsqueeze(0),
                torch.zeros(1, dtype=torch.float32, device=obj_poses.device),
            )
            for pose in self._grasp_poses(obj_poses)
        ]

    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return successful calibrated poses and physical handle widths."""
        del mesh_vertices, mesh_triangles, approach_direction
        poses = self._grasp_poses(obj_poses)
        batch_size = poses.shape[0]
        return (
            torch.ones(batch_size, dtype=torch.bool, device=poses.device),
            poses,
            torch.full(
                (batch_size,),
                _CALIBRATED_HANDLE_OPENING_WIDTH,
                dtype=torch.float32,
                device=poses.device,
            ),
        )


def _extract_handle_mesh(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the visual handle and express it in the locator-link frame.

    The legacy drawer asset models ``handle_xpos`` as a meshless locator. Its
    physical handle is the only part of the parent ``inner_box`` visual beyond
    X=0.10 m. The fixed joint places the locator at ``(0.105, 0, 0.10)`` with a
    -90-degree Y rotation, so this helper crops and transforms that geometry.

    Args:
        vertices: Parent-link-local visual vertices with shape ``(N, 3)``.
        triangles: Parent-link-local triangle indices with shape ``(M, 3)``.

    Returns:
        Compact handle-local vertices and remapped triangle indices.

    Raises:
        RuntimeError: If the expected visual handle geometry is absent.
    """
    handle_mask = torch.all(
        vertices[triangles, 0] > _HANDLE_MESH_MIN_PARENT_X,
        dim=1,
    )
    handle_triangles = triangles[handle_mask]
    if handle_triangles.numel() == 0:
        raise RuntimeError(
            f"Drawer link {DRAWER_NATIVE_MESH_LINK!r} does not contain the "
            "expected handle visual geometry."
        )

    vertex_ids, remapped = torch.unique(
        handle_triangles.reshape(-1),
        sorted=True,
        return_inverse=True,
    )
    parent_vertices = vertices.index_select(0, vertex_ids)
    origin = parent_vertices.new_tensor(_HANDLE_ORIGIN_IN_PARENT)
    relative = parent_vertices - origin
    handle_vertices = torch.stack(
        (relative[:, 2], relative[:, 1], -relative[:, 0]),
        dim=1,
    )
    return handle_vertices, remapped.reshape(-1, 3)


def create_open_drawer_scene_binding() -> SimulationSceneBinding:
    """Declare the canonical drawer and native handle-link identities.

    Returns:
        Scene binding for the passive drawer articulation and handle link.
    """
    return SimulationSceneBinding(
        registry_id=DRAWER_SCENE_REGISTRY_ID,
        articulations=(
            SimulationArticulationBinding(
                entity_id=DRAWER_UID,
                simulation_uid=DRAWER_UID,
                dynamics=SceneDynamics.DYNAMIC,
                collision_role=SceneCollisionRole.NONE,
                semantic_type="sliding_drawer",
            ),
        ),
        links=(
            SimulationArticulationLinkBinding(
                entity_id=DRAWER_HANDLE_LINK_ID,
                articulation_id=DRAWER_UID,
                native_link_name=DRAWER_NATIVE_HANDLE_LINK,
                dynamics=SceneDynamics.DYNAMIC,
                semantic_type="drawer_handle",
            ),
        ),
    )


def create_open_drawer_robot_profile_binding() -> SimulationRobotSkillProfileBinding:
    """Declare the CobotMagic right arm and parallel-gripper resource.

    Returns:
        Skill-profile binding whose primary resource supports ``Slide``.
    """
    return SimulationRobotSkillProfileBinding(
        profile_id=DRAWER_ROBOT_PROFILE_ID,
        resources=(
            ControlPartResourceBinding(
                resource_id=RIGHT_MANIPULATOR_RESOURCE_ID,
                endpoints=(
                    ControlPartEndpointBinding(
                        endpoint_id="motion",
                        control_part=RIGHT_ARM_CONTROL_PART,
                        capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
                    ),
                    ControlPartEndpointBinding(
                        endpoint_id="grasp",
                        control_part=RIGHT_EEF_CONTROL_PART,
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        command_preset="right_parallel_gripper",
                    ),
                ),
            ),
        ),
        command_presets=(
            ControlPartCommandPreset(
                preset_id="right_parallel_gripper",
                control_part=RIGHT_EEF_CONTROL_PART,
                commands={
                    "open": (0.05, 0.05),
                    "grasp": (0.0, 0.0),
                },
            ),
        ),
        defaults={"slide": {"primary": RIGHT_MANIPULATOR_RESOURCE_ID}},
        presets=(
            SkillPolicyPreset(
                "safe",
                motion_policy=MotionPolicy(sample_count=SAFE_MOTION_SAMPLE_COUNT),
                recovery_policy=RecoveryPolicy(tracking_error_threshold=0.10),
            ),
        ),
        default_preset="safe",
    )


def _load_default_expert_program():
    """Decode the packaged Open Drawer program."""
    return load_expert_program(get_config_path(DRAWER_EXPERT_PROGRAM_PATH))


@register_env("OpenDrawer-v1", max_episode_steps=300)
class OpenDrawerEnv(EmbodiedEnv):
    """Open the passive drawer through the shared semantic runtime."""

    def __init__(self, cfg: EmbodiedEnvCfg, **kwargs: Any) -> None:
        """Initialize live Slide semantics and the Expert Program adapter.

        Args:
            cfg: Declarative environment configuration.
            **kwargs: Additional arguments forwarded to :class:`EmbodiedEnv`.

        Raises:
            RuntimeError: If the drawer or its handle link is absent.
        """
        if cfg.expert_program is None:
            cfg.expert_program = _load_default_expert_program()
        super().__init__(cfg, **kwargs)

        drawer = self.sim.get_articulation(DRAWER_UID)
        if drawer is None:
            raise RuntimeError(f"OpenDrawer-v1 requires articulation {DRAWER_UID!r}.")
        required_links = {DRAWER_NATIVE_HANDLE_LINK, DRAWER_NATIVE_MESH_LINK}
        missing_links = required_links.difference(drawer.link_names)
        if missing_links:
            raise RuntimeError(
                f"Drawer is missing required links {sorted(missing_links)}; "
                f"available links are {sorted(drawer.link_names)}."
            )
        vertices, triangles = drawer.get_link_vert_face(DRAWER_NATIVE_MESH_LINK)
        handle_vertices, handle_triangles = _extract_handle_mesh(
            torch.as_tensor(vertices, dtype=torch.float32, device=self.device),
            torch.as_tensor(triangles, dtype=torch.long, device=self.device),
        )
        semantics = ObjectSemantics(
            label="drawer_handle",
            entity_id=DRAWER_HANDLE_LINK_ID,
            geometry={},
            affordance=SlideAffordance(
                mesh_vertices=handle_vertices,
                mesh_triangles=handle_triangles,
                translation_axis=torch.tensor(
                    DEFAULT_TRANSLATION_AXIS,
                    dtype=torch.float32,
                    device=self.device,
                ),
                joint_name=DRAWER_NATIVE_SLIDE_JOINT,
            ),
        )
        grasp_pose_generator = _CalibratedDrawerHandleGraspPoseGenerator()
        factory = SimulationExpertProgramFactory.from_environment(
            self,
            scene_binding=create_open_drawer_scene_binding(),
            robot_profile_binding=create_open_drawer_robot_profile_binding(),
            grasp_pose_generators={
                RIGHT_EEF_CONTROL_PART: grasp_pose_generator,
            },
        )
        self._expert_program_adapter = factory.create_adapter(
            call_catalog=_open_drawer_call_catalog(),
            registered_lowerers=(_OpenDrawerSlideLowerer(semantics),),
        )

    @property
    def expert_program_adapter(self) -> ExpertProgramEnvironmentAdapter:
        """Return the production adapter assembled for this environment.

        Returns:
            Adapter that compiles and executes the configured Expert Program.
        """
        return self._expert_program_adapter
