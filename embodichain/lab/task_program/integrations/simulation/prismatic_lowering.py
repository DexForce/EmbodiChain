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

"""Typed source-bound prismatic Slide, withdrawal and Park semantic calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
import torch

from embodichain.lab.sim.atomic_actions import (
    EndEffectorPoseGoal,
    JointPositionCommand,
    JointPositionGoal,
    JointPositionTarget,
    MoveEndEffector,
    MoveEndEffectorOptions,
    MoveJoints,
    MoveJointsOptions,
    ObjectSemantics,
    PARK_COMMAND,
    SceneEntityPose,
    Slide,
    SlideAffordance,
    SlideGoal,
    SlideJointTarget,
    SlideOptions,
)
from embodichain.lab.sim.atomic_actions.articulation_continuation import (
    SlidePlanningRecovery,
    ArticulationWithdrawalPlanningRecovery,
    ArticulationParkPlanningRecovery,
    articulation_withdrawal_targets,
    require_articulation_withdrawal_state,
)
from embodichain.lab.sim.scene.articulation_inputs import (
    PrismaticBinding,
    handle_mesh,
    inspect_prismatic,
    inspect_prismatic_part,
)
from embodichain.lab.sim.scene.prismatic_control import (
    synchronize_prismatic_joint_limits,
    validate_prismatic_placement,
    measure_native_support_top,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.semantics import (
    SceneArticulationRef,
    SceneLinkRef,
    SceneManifest,
)

__all__ = [
    "PrismaticControlBinding",
    "ArticulationSlideFactory",
    "ArticulationWithdrawFactory",
    "ArticulationParkFactory",
]


@dataclass(frozen=True, slots=True)
class PrismaticControlBinding:
    """Explicit semantic link backed by one immutable qualified physical source."""

    entity_id: str
    source: PrismaticBinding

    def __post_init__(self) -> None:
        if (
            type(self.entity_id) is not str
            or not self.entity_id
            or self.entity_id != self.entity_id.strip()
            or type(self.source) is not PrismaticBinding
        ):
            raise ValueError(
                "Prismatic control requires an exact link identity and source binding."
            )

    @property
    def selection_key(self) -> str:
        """Return the source object/optional-part argument key, not a control-part name."""
        b = self.source
        return b.object_id if not b.part_id else f"{b.object_id}::{b.part_id}"


def _bind(bindings, table_uid, simulation, registry, engine):
    if simulation.num_envs != 1:
        raise ValueError("Source prismatic controls require one environment.")
    table = simulation.get_rigid_object(table_uid)
    if table is None:
        raise ValueError("Prismatic controls require their declared measured support.")
    table_top = measure_native_support_top(table)
    result = {}
    for declaration in bindings:
        b = declaration.source
        entry = registry.lookup(declaration.entity_id, expected_type=SceneLinkRef)
        if (
            entry.parent != SceneArticulationRef(b.object_id)
            or entry.native_name != b.link
        ):
            raise ValueError("Prismatic link ownership differs from its declaration.")
        art = simulation.get_articulation(b.object_id)
        if art is None:
            raise ValueError("Declared prismatic articulation is absent.")
        config = dict(
            uid=b.object_id,
            fpath=art.cfg.fpath,
            fix_base=art.cfg.root_props.fixed_base,
            body_scale=art.cfg.body_scale,
        )
        checked = (
            inspect_prismatic_part(config, b.part_id)
            if b.part_id
            else inspect_prismatic(config)
        )
        if checked != b or b.joint not in art.joint_names:
            raise ValueError("Prismatic source or native joint binding changed.")
        validate_prismatic_placement(
            b,
            dict(
                fpath=art.cfg.fpath,
                body_scale=art.cfg.body_scale,
                init_local_pose=art.cfg.init_local_pose,
                init_pos=art.cfg.init_pos,
                init_rot=art.cfg.init_rot,
            ),
            table_top,
        )
        synchronize_prismatic_joint_limits(b, art)
        vertices, faces = handle_mesh(b, art.cfg.fpath)
        semantics = ObjectSemantics(
            entity_id=declaration.entity_id,
            geometry={},
            affordance=SlideAffordance(
                mesh_vertices=torch.as_tensor(
                    vertices, dtype=torch.float32, device=engine.device
                ),
                mesh_triangles=torch.as_tensor(
                    faces, dtype=torch.long, device=engine.device
                ),
                translation_axis=torch.tensor(b.axis, device=engine.device)
                * b.axis_sign,
                joint_name=b.joint,
                joint_limits=b.limits,
            ),
        )
        result[declaration.selection_key] = declaration, semantics, art
    return result


class _SlideLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = "gen_sim.articulation_slide"
    target_descriptor = Slide.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, bindings: dict, robot: Any) -> None:
        self.bindings, self.robot = bindings, robot

    def _selection(self, call, context, bound, *, require_grasp=True):
        args = dict(call.arguments)
        if set(args) not in ({"object", "state"}, {"object", "state", "part"}):
            raise ValueError(
                "Articulation calls require object/state and optional part."
            )
        key = (
            args["object"]
            if "part" not in args
            else f"{args['object']}::{args['part']}"
        )
        if key not in self.bindings:
            raise ValueError("Articulation call references an undeclared part.")
        declaration, semantics, art = self.bindings[key]
        target = declaration.source.target(args["state"])
        endpoint = bound.binding.action_binding.endpoint("primary", "motion")
        resource = endpoint.task_state_key
        if (
            require_grasp
            and bound.binding.action_binding.endpoint("primary", "grasp").task_state_key
            != resource
        ):
            raise ValueError("Articulation motion/grasp must share one state resource.")
        held = context.get_held_object(resource)
        if held is not None and (held.env_mask is None or held.env_mask.any()):
            raise ValueError(
                "Articulation interaction requires its resource to be free."
            )
        if any(
            resource in resources and (value.env_mask is None or value.env_mask.any())
            for resources, value in context.task.coordinated_held_objects.items()
        ):
            raise ValueError(
                "Articulation interaction cannot reuse a coordinated holding."
            )
        return declaration, semantics, art, target, endpoint

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        if type(option_template) is not SlideOptions:
            raise TypeError("Articulation Slide requires SlideOptions.")
        declaration, semantics, _, target, _ = self._selection(call, context, bound)
        b = declaration.source
        return SemanticLowering(
            goal=SlideGoal(
                semantics,
                SceneEntityPose(declaration.entity_id),
                joint_target=SlideJointTarget(
                    b.object_id, b.joint, target, axis_sign=b.axis_sign
                ),
            )
        )


class _WithdrawLowerer(_SlideLowerer):
    call_id: ClassVar[str] = "gen_sim.articulation_withdraw"
    target_descriptor = MoveEndEffector.descriptor()

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        if type(option_template) is not MoveEndEffectorOptions:
            raise TypeError("Articulation withdrawal requires MoveEndEffectorOptions.")
        declaration, _, art, target, endpoint = self._selection(
            call, context, bound, require_grasp=False
        )
        b = declaration.source
        qpos = art.get_qpos()[context.env_ids.tolist(), art.joint_names.index(b.joint)]
        primary = bound.binding.resources.get("primary")
        hand = None if primary is None else primary.endpoints.get("grasp")
        observed = expected = None
        if hand is not None:
            command = hand.commands.get("open")
            if type(command) is not JointPositionCommand:
                raise ValueError(
                    "Articulation withdrawal requires a declared hand-open command."
                )
            ids = list(hand.runtime_target.joint_ids)
            observed = context.robot.qpos[:, ids]
            expected = command.resolve(
                num_envs=observed.shape[0],
                control_dof=len(ids),
                device=observed.device,
                dtype=observed.dtype,
            )
        require_articulation_withdrawal_state(
            qpos,
            target=target,
            joint_span=b.limits[1] - b.limits[0],
            observed_hand=observed,
            open_hand=expected,
        )
        motion = endpoint.require_target(JointPositionTarget)
        current = self.robot.compute_fk(
            qpos=context.robot.qpos[:, list(motion.joint_ids)],
            name=motion.control_part,
            to_matrix=True,
        )
        link = art.get_link_pose(
            b.link, env_ids=context.env_ids.tolist(), to_matrix=True
        )
        outward = -(link[:, :3, :3] @ current.new_tensor(b.axis)) * b.axis_sign
        return SemanticLowering(
            goal=EndEffectorPoseGoal(articulation_withdrawal_targets(current, outward))
        )


class _ParkLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = "gen_sim.articulation_park"
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        if type(option_template) is not MoveJointsOptions:
            raise TypeError("Articulation Park requires MoveJointsOptions.")
        if dict(call.arguments):
            raise ValueError("Articulation Park arguments must be empty.")
        return SemanticLowering(goal=JointPositionGoal(PARK_COMMAND))


@dataclass(frozen=True, slots=True)
class ArticulationSlideFactory:
    """Create source-qualified prismatic lowerers without task-owned planners."""

    bindings: tuple[PrismaticControlBinding, ...]
    table_uid: str
    call_id: ClassVar[str] = "gen_sim.articulation_slide"
    revision: ClassVar[str] = "2"
    continuation_role: ClassVar[str] = "slide"
    recovery_type: ClassVar[type] = SlidePlanningRecovery
    lowerer_type: ClassVar[type] = _SlideLowerer
    target_descriptor = Slide.descriptor()

    def __post_init__(self) -> None:
        if (
            type(self.bindings) is not tuple
            or not self.bindings
            or any(
                type(value) is not PrismaticControlBinding for value in self.bindings
            )
        ):
            raise ValueError("Prismatic factories require immutable typed bindings.")
        if len({value.selection_key for value in self.bindings}) != len(
            self.bindings
        ) or len({value.entity_id for value in self.bindings}) != len(self.bindings):
            raise ValueError(
                "Prismatic bindings must have unique source/link identities."
            )
        if (
            type(self.table_uid) is not str
            or not self.table_uid
            or self.table_uid != self.table_uid.strip()
        ):
            raise ValueError("Prismatic factories require an explicit support UID.")

    def validate_scene_declaration(self, scene: SceneManifest) -> None:
        """Validate canonical parent/link ownership without reading the simulator."""
        for value in self.bindings:
            entry = scene.lookup(value.entity_id, expected_type=SceneLinkRef)
            if (
                entry.parent != SceneArticulationRef(value.source.object_id)
                or entry.native_name != value.source.link
            ):
                raise ValueError(
                    "Prismatic declared scene ownership differs from its source."
                )

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> RegisteredSemanticLowerer:
        """Bind a fresh lowerer to the exact managed robot and declared sources."""
        if engine.robot is not robot:
            raise ValueError("Articulation lowerer must use its factory's robot.")
        return self.lowerer_type(
            _bind(self.bindings, self.table_uid, simulation, scene_registry, engine),
            robot,
        )


@dataclass(frozen=True, slots=True)
class ArticulationWithdrawFactory(ArticulationSlideFactory):
    """Create the observed-state empty-hand cleanup lowerer."""

    call_id: ClassVar[str] = "gen_sim.articulation_withdraw"
    continuation_role: ClassVar[str] = "withdraw"
    recovery_type: ClassVar[type] = ArticulationWithdrawalPlanningRecovery
    lowerer_type: ClassVar[type] = _WithdrawLowerer
    target_descriptor = MoveEndEffector.descriptor()


@dataclass(frozen=True, slots=True)
class ArticulationParkFactory:
    """Create the named-posture cleanup call; continuation authority is separate."""

    call_id: ClassVar[str] = "gen_sim.articulation_park"
    revision: ClassVar[str] = "1"
    continuation_role: ClassVar[str] = "park"
    recovery_type: ClassVar[type] = ArticulationParkPlanningRecovery
    target_descriptor = MoveJoints.descriptor()

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> RegisteredSemanticLowerer:
        """Return one fresh lowerer without altering the shared skill registry."""
        if engine.robot is not robot:
            raise ValueError("Articulation Park requires the exact factory robot.")
        return _ParkLowerer()


def _validate_prismatic_source_bindings(scene_binding, factories):
    """Check source/native identity where the registration owns both declarations."""
    articulations = {value.entity_id: value for value in scene_binding.articulations}
    links = {value.entity_id: value for value in scene_binding.links}
    for factory in factories:
        if not isinstance(factory, ArticulationSlideFactory):
            continue
        for value in factory.bindings:
            b = value.source
            articulation, link = articulations.get(b.object_id), links.get(
                value.entity_id
            )
            if (
                articulation is None
                or articulation.simulation_uid != b.object_id
                or link is None
                or link.articulation_id != b.object_id
                or link.native_link_name != b.link
            ):
                raise ValueError(
                    "Prismatic source differs from registered native scene identity."
                )
