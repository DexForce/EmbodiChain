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

"""Runtime-bound control call grounding over explicit bridge-owned operations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import re
import math
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionGoal,
    JointPositionCommand,
    JointPositionTarget,
    MoveJoints,
    MoveJointsOptions,
    MoveEndEffector,
    MoveEndEffectorOptions,
    EndEffectorPoseGoal,
    ObjectSemantics,
    Press,
    PressAffordance,
    PressGoal,
    PressOptions,
    Twist,
    TwistOptions,
    TwistGoal,
    TwistJointTarget,
    TwistAffordance,
    ActionControlOverrides,
    SceneEntityPose,
)
from embodichain.lab.sim.atomic_actions.articulation_evidence import PressEventTarget
from embodichain.lab.sim.atomic_actions.control_validation import (
    press_start_is_released,
)
from embodichain.lab.sim.scene.press_inputs import inspect_press
from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control
from embodichain.lab.sim.utility.hand_tip import (
    closed_native_hand_tip_offset,
    HandCollisionTipCalibration,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations.runtime_services import (
    TaskProgramRuntimeServiceContext,
)
from embodichain.lab.task_program.semantics import SceneArticulationRef, SceneLinkRef

__all__ = [
    "specialize_control_factory",
    "PressCallDeclaration",
    "PressPrepareFactory",
    "PressFactory",
    "PressApproachFactory",
    "TwistCallDeclaration",
    "TwistPrepareFactory",
    "TwistFactory",
]


@dataclass(frozen=True, slots=True)
class PressCallDeclaration:
    """Explicit control identity and command variant, independent of task indices."""

    operation_id: str
    entity_id: str
    variant: str = "calibrated"
    extra_press_distance: float = 0.010

    def __post_init__(self) -> None:
        for value in (self.operation_id, self.entity_id):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Control call identities must be exact.")
        if self.variant not in (
            "original",
            "limits_only",
            "calibrated",
            "approach_only",
        ):
            raise ValueError("Unknown Press command variant.")
        if (
            type(self.extra_press_distance) not in (int, float)
            or not math.isfinite(self.extra_press_distance)
            or not 0 <= self.extra_press_distance <= 0.010
        ):
            raise ValueError("Extra Press distance must be within [0, 0.010] metres.")


class _PressPrepareLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = "gen_sim.press_prepare"
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(
        self,
        declaration: PressCallDeclaration | TwistCallDeclaration,
        runtime_services: TaskProgramRuntimeServiceContext,
        engine: Any,
    ) -> None:
        self.declaration, self.services, self.engine = (
            declaration,
            runtime_services,
            engine,
        )

    def _operation(self) -> Any:
        return self.services.control_operation(self.declaration.operation_id)

    def _arguments(self, operation: Any) -> dict[str, object]:
        return {"object": operation.binding.object_id}

    def _check(self, call: Any, context: Any, bound: Any) -> Any:
        operation = self._operation()
        if dict(call.arguments) != self._arguments(operation):
            raise ValueError("Press call must reference its exact declared object.")
        endpoint = bound.binding.action_binding.endpoint("primary", "motion")
        if (
            context.task.get_held_object(endpoint.task_state_key) is not None
            or context.task.coordinated_held_objects
        ):
            raise ValueError("Control calls require an empty hand.")
        target = endpoint.require_target(JointPositionTarget)
        if target.control_part != operation.geometry.motion_part:
            raise ValueError(
                "Control call motion binding differs from declared geometry."
            )
        return target

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        target = self._check(call, context, bound)
        if type(option_template) is not MoveJointsOptions:
            raise TypeError("Press preparation requires MoveJointsOptions.")
        return SemanticLowering(
            goal=JointPositionGoal(
                context.robot.qpos[:, list(target.joint_ids)].clone()
            )
        )

    def _hand_geometry(self, context: Any, bound: Any) -> Any:
        operation = self._operation()
        grasp = bound.binding.action_binding.endpoint("primary", "grasp")
        hand = grasp.require_target(JointPositionTarget)
        cfg = self.engine.hand_geometry_cfgs[hand.target_id]
        if (
            cfg.motion_part != operation.geometry.motion_part
            or cfg.hand_part != hand.control_part
        ):
            raise ValueError("Control hand calibration differs from bound endpoints.")
        for command, expected in (
            (OPEN_COMMAND, cfg.open_qpos),
            (GRASP_COMMAND, cfg.closed_qpos),
        ):
            actual = grasp.joint_positions(
                command,
                num_envs=context.batch_size,
                device=context.robot.qpos.device,
                dtype=context.robot.qpos.dtype,
            )
            if not torch.equal(actual, actual.new_tensor(expected).reshape(1, -1)):
                raise ValueError(
                    "Control hand commands differ from declared calibration."
                )
        return cfg


class _PressLowerer(_PressPrepareLowerer):
    call_id: ClassVar[str] = "gen_sim.press"
    target_descriptor = Press.descriptor()

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        self._check(call, context, bound)
        operation = self._operation()
        b = operation.binding
        q = operation.articulation.get_qpos()[
            0, operation.articulation.joint_names.index(b.joint)
        ]
        if not press_start_is_released(
            float(q), PressEventTarget(b.released_position, b.pressed_position)
        ):
            raise ValueError("Button is not released before Press.")
        if type(option_template) is not PressOptions or not math.isclose(
            option_template.press_distance,
            b.stroke + self.declaration.extra_press_distance,
        ):
            raise ValueError("Press options differ from declared command distance.")
        offset = closed_native_hand_tip_offset(
            operation.robot,
            self._hand_geometry(context, bound),
            context.robot.qpos,
            finger_link_names=operation.geometry.finger_link_names,
        )
        point = tuple(p - offset * a for p, a in zip(b.press_position, b.axis))
        return SemanticLowering(
            goal=PressGoal(
                ObjectSemantics(
                    entity_id=self.declaration.entity_id,
                    geometry={},
                    affordance=PressAffordance(
                        press_axis=torch.tensor(b.axis, device=self.engine.device),
                        press_position=point,
                    ),
                ),
                SceneEntityPose(self.declaration.entity_id),
            )
        )


class _PressApproachLowerer(_PressPrepareLowerer):
    call_id: ClassVar[str] = "gen_sim.press"
    target_descriptor = MoveEndEffector.descriptor()

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        self._check(call, context, bound)
        if type(option_template) is not MoveEndEffectorOptions:
            raise TypeError("Approach-only control requires MoveEndEffectorOptions.")
        operation = self._operation()
        b = operation.binding
        affordance = PressAffordance(
            press_axis=torch.tensor(b.axis, device=self.engine.device),
            press_position=b.press_position,
        )
        pose = affordance.get_press_pose(
            operation.articulation.get_link_pose(b.link, to_matrix=True)
        )
        pose[:, :3, 3] -= 0.08 * pose[:, :3, 2]
        return SemanticLowering(goal=EndEffectorPoseGoal(pose))


@dataclass(frozen=True, slots=True)
class PressPrepareFactory:
    """Ground a fresh prepare lowerer using the actual bridge operation."""

    declaration: PressCallDeclaration
    call_id: ClassVar[str] = "gen_sim.press_prepare"
    revision: ClassVar[str] = "1"
    lowerer_type: ClassVar[type] = _PressPrepareLowerer
    target_descriptor = MoveJoints.descriptor()

    def __post_init__(self) -> None:
        if type(self.declaration) is not PressCallDeclaration:
            raise TypeError("Press factory requires an exact typed declaration.")
        if (
            issubclass(self.lowerer_type, _PressLowerer)
            and self.declaration.variant == "approach_only"
        ):
            raise ValueError("Approach-only requires its MoveEndEffector factory.")
        if (
            issubclass(self.lowerer_type, _PressApproachLowerer)
            and self.declaration.variant != "approach_only"
        ):
            raise ValueError(
                "Approach factory requires the explicit approach-only variant."
            )

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> RegisteredSemanticLowerer:
        """Reject assembly without the actual bridge-owned operation context."""
        raise ValueError("Control lowerers require actual runtime services.")

    def create_with_runtime_services(
        self,
        *,
        simulation: Any,
        robot: Any,
        scene_registry: Any,
        engine: Any,
        runtime_services: TaskProgramRuntimeServiceContext,
    ) -> RegisteredSemanticLowerer:
        """Validate source/scene ownership and bind one fresh lowerer."""
        operation = runtime_services.control_operation(self.declaration.operation_id)
        if (
            operation.family != "press"
            or operation.entity_id != self.declaration.entity_id
            or operation.robot is not robot
            or engine.robot is not robot
            or simulation.num_envs != 1
        ):
            raise ValueError("Press factory differs from its actual bridge operation.")
        payload = operation.route_payload
        if (
            payload.get("variant") != self.declaration.variant
            or payload.get("extra_press_distance", 0.010)
            != self.declaration.extra_press_distance
        ):
            raise ValueError("Press declaration differs from operation provenance.")
        b, art = operation.binding, operation.articulation
        if simulation.get_articulation(b.object_id) is not art:
            raise ValueError("Control articulation belongs to another simulation.")
        entry = scene_registry.lookup(
            self.declaration.entity_id, expected_type=SceneLinkRef
        )
        if (
            entry.parent != SceneArticulationRef(b.object_id)
            or entry.native_name != b.link
        ):
            raise ValueError("Control link ownership changed.")
        actual = inspect_press(
            {
                "uid": art.uid,
                "fpath": art.cfg.fpath,
                "fix_base": art.cfg.root_props.fixed_base,
                "body_scale": art.cfg.body_scale,
            },
            joint=b.joint,
            link=b.link,
            collision_mesh=b.collision_mesh,
            released_position=b.released_position,
        )
        if replace(actual, source_path="") != replace(b, source_path=""):
            raise ValueError("Press source or geometry changed.")
        chain = [
            j for j in art.get_parent_joint_chain(b.link) if j.joint_type != "fixed"
        ]
        if (
            len(chain) != 1
            or chain[0].name != b.joint
            or chain[0].parent_link_name != b.parent
        ):
            raise ValueError("Press runtime topology changed.")
        if self.declaration.variant != "original" and not torch.allclose(
            torch.tensor(chain[0].joint_limits),
            torch.tensor(b.limits),
            atol=1e-7,
            rtol=1e-5,
        ):
            raise ValueError("Press runtime limits differ from scaled travel.")
        return self.lowerer_type(self.declaration, runtime_services, engine)


@dataclass(frozen=True, slots=True)
class PressFactory(PressPrepareFactory):
    """Ground the calibrated Press command without injecting action options."""

    call_id: ClassVar[str] = "gen_sim.press"
    lowerer_type: ClassVar[type] = _PressLowerer
    target_descriptor = Press.descriptor()


@dataclass(frozen=True, slots=True)
class PressApproachFactory(PressPrepareFactory):
    """Ground the explicitly declared approach-only negative control."""

    call_id: ClassVar[str] = "gen_sim.press"
    lowerer_type: ClassVar[type] = _PressApproachLowerer
    target_descriptor = MoveEndEffector.descriptor()


@dataclass(frozen=True, slots=True)
class TwistCallDeclaration:
    """Explicit Twist contact variant and bounded command declaration."""

    operation_id: str
    entity_id: str
    variant: str = "source"
    grasp_depth: float = 0.008
    chunk_angle: float = math.radians(5.0)
    settle_steps: int = 5
    fixed_roll: float | None = None

    def __post_init__(self) -> None:
        for value in (self.operation_id, self.entity_id):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Control call identities must be exact.")
        if self.variant not in ("source", "no_twist", "centroid", "closed_tip"):
            raise ValueError("Unknown Twist command variant.")
        if (
            type(self.grasp_depth) not in (int, float)
            or not math.isfinite(self.grasp_depth)
            or not 0 < self.grasp_depth <= 0.01
        ):
            raise ValueError("Twist grasp depth must be within (0, 0.01] metres.")
        if (
            type(self.chunk_angle) not in (int, float)
            or not math.isfinite(self.chunk_angle)
            or not 0 < self.chunk_angle <= math.pi / 2
        ):
            raise ValueError("Twist chunk angle must be within (0, pi/2].")
        if type(self.settle_steps) is not int or not 0 <= self.settle_steps <= 100:
            raise ValueError("Twist settle samples must be within [0, 100].")
        if self.fixed_roll is not None and (
            type(self.fixed_roll) not in (int, float)
            or not math.isfinite(self.fixed_roll)
            or abs(self.fixed_roll) > math.pi
        ):
            raise ValueError("Twist fixed roll must be within [-pi, pi].")


class _TwistPrepareLowerer(_PressPrepareLowerer):
    call_id: ClassVar[str] = "gen_sim.twist_prepare"
    target_descriptor = MoveJoints.descriptor()

    def _arguments(self, operation: Any) -> dict[str, object]:
        return {
            "object": operation.binding.object_id,
            "setting": operation.binding.target_setting,
        }

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        target = self._check(call, context, bound)
        if type(option_template) is not MoveJointsOptions:
            raise TypeError("Twist preparation requires MoveJointsOptions.")
        self._operation().start_startup()
        return SemanticLowering(
            goal=JointPositionGoal(
                context.robot.qpos[:, list(target.joint_ids)].clone()
            )
        )


class _TwistLowerer(_TwistPrepareLowerer):
    call_id: ClassVar[str] = "gen_sim.twist"
    target_descriptor = Twist.descriptor()

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        self._check(call, context, bound)
        operation = self._operation()
        b, declaration = operation.binding, self.declaration
        bite = min(declaration.grasp_depth, b.grip_depth / 2)
        if (
            type(option_template) is not TwistOptions
            or option_template.rotation_enabled != (declaration.variant != "no_twist")
            or option_template.contact_bite_depth != bite
            or option_template.max_twist_angle != declaration.chunk_angle
            or option_template.contact_roll != declaration.fixed_roll
            or option_template.contact_settle_steps != declaration.settle_steps
        ):
            raise ValueError(
                "Twist options differ from the explicit bounded declaration."
            )
        cfg = self._hand_geometry(context, bound)
        rest = float(
            operation.articulation.get_link_physical_attr(b.link)[0].rest_offset
        )
        pad_rest = max(
            float(value.rest_offset)
            for value in operation.robot.get_link_physical_attr(cfg.pad_link_names)
        )
        calibration = HandCollisionTipCalibration(
            operation.robot,
            cfg,
            context.robot.qpos,
            finger_link_names=operation.geometry.finger_link_names,
        )
        offset, command = calibration.fit_opening(b.grip_width + 2 * (rest + pad_rest))
        if declaration.variant == "closed_tip":
            offset = calibration.closed_tip_offset()
        shift = b.grip_depth / 2 if declaration.variant == "centroid" else bite - offset
        point = tuple(p + shift * a for p, a in zip(b.outer_point, b.axis))
        return SemanticLowering(
            goal=TwistGoal(
                ObjectSemantics(
                    entity_id=declaration.entity_id,
                    geometry={},
                    affordance=TwistAffordance(
                        grasp_position=point,
                        axis_origin=b.origin,
                        twist_axis=torch.tensor(b.axis),
                        joint_name=b.joint,
                        joint_limits=b.limits,
                    ),
                ),
                SceneEntityPose(declaration.entity_id),
                contact_reference=point,
                joint_target=TwistJointTarget(
                    b.object_id, b.joint, b.target_qpos, axis_sign=int(b.axis_sign)
                ),
            ),
            control_overrides=ActionControlOverrides(
                endpoints={
                    "primary": {"grasp": {GRASP_COMMAND: JointPositionCommand(command)}}
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class TwistPrepareFactory:
    """Bind preparation to the same operation that owns startup observation."""

    declaration: TwistCallDeclaration
    call_id: ClassVar[str] = "gen_sim.twist_prepare"
    revision: ClassVar[str] = "1"
    lowerer_type: ClassVar[type] = _TwistPrepareLowerer
    target_descriptor = MoveJoints.descriptor()

    def __post_init__(self) -> None:
        if type(self.declaration) is not TwistCallDeclaration:
            raise TypeError("Twist factory requires an exact typed declaration.")

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> RegisteredSemanticLowerer:
        """Reject assembly without actual bridge-owned operation services."""
        raise ValueError("Control lowerers require actual runtime services.")

    def create_with_runtime_services(
        self,
        *,
        simulation: Any,
        robot: Any,
        scene_registry: Any,
        engine: Any,
        runtime_services: TaskProgramRuntimeServiceContext,
    ) -> RegisteredSemanticLowerer:
        """Check source ownership and bind a fresh runtime lowerer."""
        operation = runtime_services.control_operation(self.declaration.operation_id)
        if (
            operation.family != "twist"
            or operation.entity_id != self.declaration.entity_id
            or operation.robot is not robot
            or engine.robot is not robot
            or simulation.num_envs != 1
        ):
            raise ValueError("Twist factory differs from its actual bridge operation.")
        payload = operation.route_payload
        if any(
            payload.get(name) != getattr(self.declaration, name)
            for name in (
                "variant",
                "grasp_depth",
                "chunk_angle",
                "settle_steps",
                "fixed_roll",
            )
        ):
            raise ValueError("Twist declaration differs from operation provenance.")
        b, art = operation.binding, operation.articulation
        if simulation.get_articulation(b.object_id) is not art:
            raise ValueError("Control articulation belongs to another simulation.")
        entry = scene_registry.lookup(
            self.declaration.entity_id, expected_type=SceneLinkRef
        )
        if (
            entry.parent != SceneArticulationRef(b.object_id)
            or entry.native_name != b.link
        ):
            raise ValueError("Control link ownership changed.")
        actual = inspect_revolute_control(
            {
                "uid": art.uid,
                "fpath": art.cfg.fpath,
                "fix_base": art.cfg.root_props.fixed_base,
                "body_scale": art.cfg.body_scale,
            },
            b.target_setting,
        )
        if actual != b:
            raise ValueError("Twist source-qualified geometry changed.")
        return self.lowerer_type(self.declaration, runtime_services, engine)


@dataclass(frozen=True, slots=True)
class TwistFactory(TwistPrepareFactory):
    """Ground a typed absolute joint target and calibrated contact reference."""

    call_id: ClassVar[str] = "gen_sim.twist"
    lowerer_type: ClassVar[type] = _TwistLowerer
    target_descriptor = Twist.descriptor()


@lru_cache(maxsize=1024)
def _specialized_types(base: type, suffix: str) -> type:
    call_id = f"{base.call_id}.{suffix}"
    digest = hashlib.sha256(call_id.encode()).hexdigest()
    lowerer = type(
        f"{base.lowerer_type.__name__}_{digest}",
        (base.lowerer_type,),
        {"__module__": __name__, "call_id": call_id, "__slots__": ()},
    )
    return type(
        f"{base.__name__}_{digest}",
        (base,),
        {
            "__module__": __name__,
            "call_id": call_id,
            "lowerer_type": lowerer,
            "__slots__": (),
        },
    )


def specialize_control_factory(
    factory: PressPrepareFactory | TwistPrepareFactory, operation_suffix: str
) -> PressPrepareFactory | TwistPrepareFactory:
    """Declare an operation-suffixed call ID on immutable factory/lowerer types.

    Specialization retains the exact descriptor and declaration. It does not
    patch instances, replace registered skills, or retain live runtime state.
    An empty suffix preserves the established unsuffixed public call IDs.
    """
    if type(factory) not in (
        PressPrepareFactory,
        PressFactory,
        PressApproachFactory,
        TwistPrepareFactory,
        TwistFactory,
    ):
        raise TypeError("Only concrete control factories may be specialized.")
    if (
        type(operation_suffix) is not str
        or operation_suffix
        and not re.fullmatch(r"[a-z0-9_]+(?:[.-][a-z0-9_]+)*", operation_suffix)
    ):
        raise ValueError(
            "Control call suffix must be a canonical lowercase identifier."
        )
    if not operation_suffix:
        return factory
    if operation_suffix != factory.declaration.operation_id:
        raise ValueError("Control call suffix must match its declared operation.")
    return _specialized_types(type(factory), operation_suffix)(factory.declaration)
