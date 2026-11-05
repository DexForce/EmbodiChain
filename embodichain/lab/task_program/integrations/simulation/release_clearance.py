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

"""Released-hand target binding with explicit nominal hand reserves."""

from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from typing import ClassVar
import math
import torch
from embodichain.compute.geometry.hand_clearance import (
    released_hand_targets,
    released_hand_height,
)
from embodichain.toolkits.graspkit.hand_envelope import NominalHandEnvelope
from embodichain.lab.sim.atomic_actions import (
    MoveEndEffector,
    MoveEndEffectorOptions,
    EndEffectorPoseGoal,
    ActionOptions,
    AtomicActionEngine,
    PlanningContext,
    SkillDescriptor,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    BoundSemanticCall,
    SceneRegistry,
    SceneManifest,
    SceneObjectRef,
    GRASP_AFFORDANCE_CAPABILITY,
)
from ..extensions import RegisteredSemanticLowererFactory
from .targets import _identifier

__all__ = ["ReleasedHandRoute", "NominalHandBinding", "ClearReleasedLowererFactory"]


@dataclass(frozen=True, slots=True)
class ReleasedHandRoute:
    """Exact semantic object/target and minimum clearance height."""

    object_id: str
    target_id: str
    minimum_height: float

    def __post_init__(self) -> None:
        _identifier(self.object_id, field_name="object_id")
        _identifier(self.target_id, field_name="target_id")
        if type(self.minimum_height) not in (int, float) or not math.isfinite(
            self.minimum_height
        ):
            raise ValueError("Minimum clearance height must be finite.")


@dataclass(frozen=True, slots=True)
class NominalHandBinding:
    """Bind caller-supplied nominal dimensions to one exact model/target pair."""

    grasp_target_id: str
    model_id: str
    envelope: NominalHandEnvelope

    def __post_init__(self) -> None:
        _identifier(self.grasp_target_id, field_name="grasp_target_id")
        _identifier(self.model_id, field_name="model_id")
        if type(self.envelope) is not NominalHandEnvelope:
            raise TypeError("Nominal hand envelope must be typed.")


class _ClearReleasedLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = "gen_sim.clear_released"
    target_descriptor: ClassVar[SkillDescriptor] = MoveEndEffector.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(
        self,
        routes,
        robot,
        *,
        retreat_distance,
        object_vertices,
        gripper_models,
        hand_envelopes,
    ):
        self._routes = {(r.object_id, r.target_id): r.minimum_height for r in routes}
        self._robot, self._retreat, self._vertices, self._grippers, self._envelopes = (
            robot,
            retreat_distance,
            object_vertices,
            gripper_models,
            hand_envelopes,
        )

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: ActionOptions,
    ) -> SemanticLowering:
        if type(option_template) is not MoveEndEffectorOptions:
            raise TypeError("Released-hand clearance requires MoveEndEffectorOptions.")
        args = dict(call.arguments)
        if set(args) != {"object", "target"}:
            raise ValueError("Released-hand clearance requires only object and target.")
        selector = (args["object"], args["target"])
        if selector not in self._routes:
            raise ValueError("Released-hand clearance has no declared route.")
        endpoint = bound.binding.resources["primary"].endpoints["motion"]
        held = context.task.get_held_object(endpoint.task_state_key)
        if (
            held is not None
            and context.task.held_object_mask(endpoint.task_state_key).any()
        ):
            raise ValueError("Released-hand clearance cannot move a held object.")
        motion = endpoint.runtime_target
        current = self._robot.compute_fk(
            qpos=context.robot.qpos[:, list(motion.joint_ids)],
            name=motion.control_part,
            to_matrix=True,
        )
        root = self._robot.get_link_pose(
            link_name=self._robot.cfg.solver_cfg[motion.control_part].root_link_name,
            env_ids=context.env_ids.tolist(),
            to_matrix=True,
        ).to(current)
        observed = context.scene.entities[args["object"]]
        if observed.confidence <= 0 or not torch.isfinite(observed.pose).all():
            raise ValueError(
                "Released-hand clearance requires a finite observed object pose."
            )
        pose = observed.pose.to(current)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0).expand(current.shape[0], -1, -1)
        grasp = bound.binding.resources["primary"].endpoints["grasp"].runtime_target
        envelope = self._envelopes[grasp.target_id]
        height = released_hand_height(
            current,
            pose,
            self._vertices[args["object"]],
            opening_width=self._grippers[grasp.target_id].max_opening_width,
            finger_length=envelope.finger_length,
            finger_width=envelope.finger_width,
            finger_thickness=envelope.finger_thickness,
            palm_depth=envelope.palm_depth,
        )
        height = torch.clamp_min(height, self._routes[selector])
        return SemanticLowering(
            goal=EndEffectorPoseGoal(
                released_hand_targets(
                    current, root, height=height, retreat=self._retreat
                )
            )
        )


@dataclass(frozen=True, slots=True)
class ClearReleasedLowererFactory(RegisteredSemanticLowererFactory):
    """Create released-hand target lowering without replacing shared motion."""

    call_id: ClassVar[str] = "gen_sim.clear_released"
    revision: ClassVar[str] = "declared-envelope-1"
    target_descriptor: ClassVar[SkillDescriptor] = MoveEndEffector.descriptor()
    routes: tuple[ReleasedHandRoute, ...]
    hands: tuple[NominalHandBinding, ...]
    retreat_distance: float = 0.15

    def __post_init__(self) -> None:
        for values, cls, label in (
            (self.routes, ReleasedHandRoute, "routes"),
            (self.hands, NominalHandBinding, "hands"),
        ):
            if (
                type(values) is not tuple
                or not values
                or any(type(value) is not cls for value in values)
            ):
                raise TypeError(
                    f"Clearance {label} require explicit typed declarations."
                )
        if len({(r.object_id, r.target_id) for r in self.routes}) != len(self.routes):
            raise ValueError("Clearance routes must be unique.")
        if len({h.grasp_target_id for h in self.hands}) != len(self.hands):
            raise ValueError("Clearance hands must be unique.")
        if (
            type(self.retreat_distance) not in (int, float)
            or not math.isfinite(self.retreat_distance)
            or self.retreat_distance <= 0
        ):
            raise ValueError("Retreat distance must be finite and positive.")

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Resolve declared source objects without native geometry acquisition.

        Args:
            scene: Immutable canonical declaration.
        """
        for route in self.routes:
            scene.resolve(route.object_id, expected_type=SceneObjectRef)

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicActionEngine,
    ) -> RegisteredSemanticLowerer:
        """Bind exact object mesh and hand-model inputs to shared numeric targets.

        Args:
            simulation: Borrowed physical owner; no stepping or planning occurs.
            robot: Exact engine robot.
            scene_registry: Existing object/grasp semantic registry.
            engine: Ordinary Atomic engine with installed grasp generators.

        Returns:
            Fresh lowerer with independently owned source meshes and model inputs.
        """
        if engine.robot is not robot:
            raise ValueError("Released-hand clearance must bind the factory's robot.")
        vertices = {}
        for route in self.routes:
            ref = scene_registry.resolve(route.object_id, expected_type=SceneObjectRef)
            grasp = scene_registry.resolve_affordance(
                ref, capability=GRASP_AFFORDANCE_CAPABILITY
            )
            vertices[route.object_id] = (
                scene_registry.object_semantics(ref, affordance=grasp)
                .affordance.mesh_vertices.detach()
                .clone()
            )
        models = {}
        for hand in self.hands:
            model = engine.grasp_pose_generators[hand.grasp_target_id].gripper_model
            if model.model_id != hand.model_id:
                raise ValueError(
                    "Nominal hand reserve belongs to a different gripper model."
                )
            models[hand.grasp_target_id] = deepcopy(model)
        return _ClearReleasedLowerer(
            self.routes,
            robot,
            retreat_distance=self.retreat_distance,
            object_vertices=vertices,
            gripper_models=models,
            hand_envelopes={h.grasp_target_id: h.envelope for h in self.hands},
        )
