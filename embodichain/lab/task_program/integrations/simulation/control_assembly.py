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

"""Explicit native assembly for declared Press/Twist control operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from uuid import uuid4
from typing import TYPE_CHECKING

from embodichain.lab.sim.atomic_actions.primitives.twist_recovery import (
    TwistRecoveryScope,
)
from embodichain.lab.sim.scene.press_inputs import PressBinding
from embodichain.lab.sim.scene.revolute_inputs import RevoluteControlBinding
from .control_lowering import PressCallDeclaration, TwistCallDeclaration
from .control_observation import ControlContactActors
from .control_policy import ControlOperationRuntime
from .twist_session import TwistOperationSession

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions.state import PlanningContext
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
    from embodichain.lab.sim.motion.revolute_query import RevolutePathQueryFactory

__all__ = [
    "ControlOperationDeclaration",
    "ControlOperationAssembly",
    "assemble_control_operation",
]


@dataclass(frozen=True, slots=True)
class ControlOperationDeclaration:
    """Immutable source, contact and model bindings for one control operation.

    ``arm_label`` preserves historical route provenance only. Motion and finger
    selection always use explicit calibration and the full declared link roster.
    The configured host owns sensor creation and the ordinary substep suite.
    """

    call: PressCallDeclaration | TwistCallDeclaration
    binding: PressBinding | RevoluteControlBinding
    sensor_uid: str
    grasp_target_id: str
    finger_link_names: tuple[str, ...]
    arm_label: str
    moving_mass: float | None = None
    table_uid: str | None = None
    chunk_count: int | None = None
    success_criterion: str = "contact_backed_coarse_turn"

    def __post_init__(self) -> None:
        from embodichain.lab.task_program.integrations.extensions import (
            validate_immutable_extension_declaration,
        )

        validate_immutable_extension_declaration(self, field_name="control_operation")
        press = type(self.call) is PressCallDeclaration
        twist = type(self.call) is TwistCallDeclaration
        if not (press or twist) or type(self.binding) is not (
            PressBinding if press else RevoluteControlBinding
        ):
            raise TypeError(
                "Control operation requires matching exact call/source types."
            )
        for value in (self.sensor_uid, self.grasp_target_id, self.arm_label):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Control model and sensor identities must be exact.")
        if (
            type(self.finger_link_names) is not tuple
            or not self.finger_link_names
            or len(set(self.finger_link_names)) != len(self.finger_link_names)
            or any(
                type(name) is not str or not name or name != name.strip()
                for name in self.finger_link_names
            )
        ):
            raise ValueError(
                "Control fingers require a complete immutable named roster."
            )
        if press:
            if (
                type(self.moving_mass) not in (int, float)
                or not math.isfinite(self.moving_mass)
                or self.moving_mass <= 0
            ):
                raise ValueError(
                    "Press provenance requires positive source moving mass."
                )
            if self.table_uid is not None or self.chunk_count is not None:
                raise ValueError("Press cannot declare Twist table/chunk services.")
        else:
            if self.moving_mass is not None:
                raise ValueError("Twist mass is qualified from its source lineage.")
            if (
                type(self.table_uid) is not str
                or not self.table_uid
                or self.table_uid != self.table_uid.strip()
            ):
                raise ValueError("Twist requires an exact declared table UID.")
            if type(self.chunk_count) is not int or not 1 <= self.chunk_count <= 32:
                raise ValueError("Twist requires a bounded declared chunk count.")
            if self.success_criterion not in (
                "contact_backed_coarse_turn",
                "calibrated_setting",
            ):
                raise ValueError("Unknown Twist physical event criterion.")

    def source_route_payload(self) -> dict[str, object]:
        """Return the original route projection for evidence provenance only."""
        binding = asdict(self.binding)
        call = self.call
        if type(call) is PressCallDeclaration:
            binding.pop("source_path")
            return dict(
                binding=binding,
                moving_mass=self.moving_mass,
                arm=self.arm_label,
                variant=call.variant,
                extra_press_distance=call.extra_press_distance,
                operation_id=call.operation_id,
            )
        return dict(
            binding=binding,
            arm=self.arm_label,
            variant=call.variant,
            grasp_depth=call.grasp_depth,
            settle_steps=call.settle_steps,
            chunk_count=self.chunk_count,
            chunk_angle=call.chunk_angle,
            fixed_roll=call.fixed_roll,
            feedback_enabled=False,
            grip_extra_m=0.0,
            operation_id=call.operation_id,
            success_criterion=self.success_criterion,
        )


@dataclass(frozen=True, slots=True)
class ControlOperationAssembly:
    """Own one operation's observer/cursor/query resources, borrowing native owners."""

    operation: ControlOperationRuntime
    twist_session: TwistOperationSession | None
    geometry_queries: RevolutePathQueryFactory | None

    def close(self) -> None:
        """Detach owned observation and query resources without native destruction."""
        try:
            self.operation.close()
        finally:
            if self.geometry_queries is not None:
                self.geometry_queries.close()


def assemble_control_operation(
    declaration: ControlOperationDeclaration,
    *,
    simulation: SimulationManager,
    robot: Robot,
    sensor_suite: Mapping[str, object],
    hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg],
    program_fingerprint: str,
    episode_id: str,
    invocation_ids: Sequence[str],
    control_dt: float,
    validate_scope: Callable[[], None],
    validate_observation: Callable[[PlanningContext], None],
) -> ControlOperationAssembly:
    """Resolve exact native bindings and create one isolated operation instance.

    This helper does not step simulation, create sensors, install planners, bind
    clocks, or select policies. The host passes its real episode identity, exact
    compiled Twist occurrences and current scope/observation guards. It must close
    the returned assembly when the bridge ends, including failed assembly paths.
    """
    from embodichain.lab.sim.sensors.contact.sampling import (
        ContactEvidenceSampler,
        native_contact_revision,
    )
    from embodichain.lab.sim.sensors.contact.sensor import ContactSensor
    from embodichain.lab.sim.scene.control_measurement import NativeControlGeometry
    from embodichain.lab.sim.scene.revolute_geometry import load_twist_geometry
    from embodichain.lab.sim.scene.control_source import capture_revolute_source_seal
    from embodichain.lab.sim.motion.revolute_query import RevolutePathQueryFactory

    if type(declaration) is not ControlOperationDeclaration:
        raise TypeError("Control assembly requires an exact immutable declaration.")
    validate_scope()
    if (
        simulation.num_envs != 1
        or robot.num_instances != 1
        or simulation.get_robot(robot.uid) is not robot
    ):
        raise ValueError("Control assembly requires its exact managed B1 robot.")
    if not simulation.is_default_backend:
        raise ValueError("Source control qualification requires Default physics.")
    b, call = declaration.binding, declaration.call
    articulation = simulation.get_articulation(b.object_id)
    if articulation is None:
        raise ValueError("Declared control articulation is missing.")
    from dataclasses import replace
    from embodichain.lab.sim.scene.press_inputs import inspect_press
    from embodichain.lab.sim.scene.revolute_inputs import inspect_revolute_control

    source = dict(
        uid=articulation.uid,
        fpath=articulation.cfg.fpath,
        fix_base=articulation.cfg.root_props.fixed_base,
        body_scale=articulation.cfg.body_scale,
    )
    if type(b) is PressBinding:
        actual = inspect_press(
            source,
            joint=b.joint,
            link=b.link,
            collision_mesh=b.collision_mesh,
            released_position=b.released_position,
        )
        unchanged = replace(actual, source_path="") == replace(b, source_path="")
    else:
        unchanged = inspect_revolute_control(source, b.target_setting) == b
    if not unchanged:
        raise ValueError("Control source binding changed before runtime assembly.")
    sensor = simulation.get_sensor(declaration.sensor_uid)
    if (
        not isinstance(sensor, ContactSensor)
        or sensor_suite.get(declaration.sensor_uid) is not sensor
    ):
        raise ValueError(
            "Control sensor must be the exact ordinary substep-suite sensor."
        )
    twist = type(call) is TwistCallDeclaration
    selected_articulations = tuple(sensor.cfg.articulation_cfg_list)
    if (
        sensor.cfg.max_contacts_per_env != 512
        or sensor.cfg.filter_need_both_actor is not (not twist)
        or sensor.cfg.rigid_uid_list
        or len(selected_articulations) != 2
        or {value.articulation_uid for value in selected_articulations}
        != {robot.uid, b.object_id}
        or any(value.link_name_list for value in selected_articulations)
    ):
        raise ValueError(
            "Control sensor must retain the declared full-actor source query and 512-contact capacity."
        )
    cfg = hand_geometry_cfgs[declaration.grasp_target_id]
    if cfg.hand_part != declaration.grasp_target_id:
        raise ValueError("Control calibration differs from its grasp target.")
    ids = tuple(invocation_ids)
    twist = type(call) is TwistCallDeclaration
    if (twist and len(ids) != declaration.chunk_count) or (not twist and ids):
        raise ValueError(
            "Control occurrences differ from the declared Twist chunk count."
        )

    def actors(uid: str, names: Sequence[str] | None = None) -> tuple[int, ...]:
        values = sensor.get_actor_ids(uid, names)
        expected = 1 if names is None else len(names)
        if values.shape != (1, expected):
            raise ValueError("Control actor IDs must preserve the exact B1 roster.")
        return tuple(int(value) for value in values[0].tolist())

    target, parent = actors(b.object_id, (b.link, b.parent))
    contact_actors = ControlContactActors(
        target, parent, actors(robot.uid, declaration.finger_link_names)
    )
    geometry = queries = session = None
    try:
        if twist:
            geometry = load_twist_geometry(articulation.cfg.fpath, b)
            seal = capture_revolute_source_seal(
                articulation.cfg.fpath,
                b,
                body_scale_mass_policy=getattr(
                    articulation.cfg, "body_scale_mass_policy", "native"
                ),
            )
            queries = RevolutePathQueryFactory(
                simulation,
                robot,
                binding=b,
                geometry=geometry,
                source_seal=seal,
                hand_geometry_cfg=cfg,
                table_uid=declaration.table_uid,
            )
            session = TwistOperationSession(
                robot,
                queries,
                TwistRecoveryScope(
                    program_fingerprint, call.operation_id, episode_id, 0
                ),
                invocation_ids=ids,
                parent_entity_id=b.object_id,
                child_entity_id=call.entity_id,
                joint_name=b.joint,
                validate_observation=validate_observation,
                validate_execution_scope=validate_scope,
            )
        native_geometry = NativeControlGeometry(
            robot,
            articulation,
            finger_link_names=declaration.finger_link_names,
            motion_part=cfg.motion_part,
            press_binding=None if twist else b,
            twist_geometry=geometry,
        )
        operation = ControlOperationRuntime(
            robot,
            articulation,
            b,
            native_geometry,
            ContactEvidenceSampler(
                sensor,
                native_revision_reader=lambda: native_contact_revision(simulation),
                topology_revision_reader=lambda: int(
                    simulation.spawn_result.topology_revision
                ),
            ),
            contact_actors,
            entity_id=call.entity_id,
            operation_id=call.operation_id,
            program_fingerprint=program_fingerprint,
            episode_id=episode_id,
            observer_id=f"control:{uuid4()}",
            route_payload=declaration.source_route_payload(),
            control_dt=control_dt,
            physics_dt=simulation.sim_config.physics_dt,
            simulation_time=lambda: simulation.simulation_time,
            validate_scope=validate_scope,
            twist_session=session,
            table_actor=actors(declaration.table_uid)[0] if twist else None,
            robot_actors=actors(robot.uid, robot.link_names) if twist else (),
            twist_criterion=declaration.success_criterion,
        )
        return ControlOperationAssembly(operation, session, queries)
    except Exception:
        if session is not None:
            session.close()
        if queries is not None:
            queries.close()
        raise


def _validate_control_registration(declarations, scene_binding, presets, factories):
    """Cross-check immutable registration wiring before native assembly."""
    from .control_lowering import PressPrepareFactory, TwistPrepareFactory
    from .control_policy import SimulationControlConditionCfg

    if type(declarations) is not tuple or any(
        type(value) is not ControlOperationDeclaration for value in declarations
    ):
        raise TypeError(
            "control_operations must be an exact tuple of typed declarations."
        )
    operations = {value.call.operation_id: value for value in declarations}
    if len(operations) != len(declarations):
        raise ValueError("Control operation IDs must be unique.")
    articulations = {value.entity_id: value for value in scene_binding.articulations}
    links = {value.entity_id: value for value in scene_binding.links}
    phases = {key: set() for key in operations}
    for declaration in declarations:
        b, call = declaration.binding, declaration.call
        articulation, link = articulations.get(b.object_id), links.get(call.entity_id)
        if (
            articulation is None
            or articulation.simulation_uid != b.object_id
            or link is None
            or link.articulation_id != b.object_id
            or link.native_link_name != b.link
        ):
            raise ValueError(
                "Control source identity differs from declared scene bindings."
            )
    for preset in presets.values():
        if type(preset) is not SimulationControlConditionCfg:
            continue
        declaration = operations.get(preset.operation_id)
        if (
            declaration is None
            or preset.entity != declaration.call.entity_id
            or preset.family
            != ("press" if type(declaration.call) is PressCallDeclaration else "twist")
        ):
            raise ValueError(
                "Control policy requires matching control-operation declarations."
            )
        phases[preset.operation_id].add(preset.phase)
    calls = []
    roles = {key: [] for key in operations}
    for factory in factories:
        if not isinstance(factory, (PressPrepareFactory, TwistPrepareFactory)):
            continue
        declaration = operations.get(factory.declaration.operation_id)
        if declaration is None or factory.declaration != declaration.call:
            raise ValueError(
                "Control lowerer requires its matching operation declaration."
            )
        prepare = factory.target_descriptor.skill_id == "move_joints"
        roles[declaration.call.operation_id].append(prepare)
        calls.append((factory.call_id, declaration, prepare))
    for key, declaration in operations.items():
        required = (
            {"ready", "released"}
            if type(declaration.call) is PressCallDeclaration
            else {"ready", "chunk", "released"}
        )
        if sorted(roles[key]) != [False, True] or not required <= phases[key]:
            raise ValueError(
                "Control operation requires one prepare/action factory and complete physical policy phases."
            )
    return tuple(calls)


def _bind_control_post_policies(
    program, bindings, control_calls, presets, cleanup_call_ids
):
    """Preflight explicit post occurrences against complete typed control recipes."""
    from ..invocation_policy import BoundPostPolicyPresets
    from .control_policy import SimulationControlConditionCfg

    bound = BoundPostPolicyPresets(program, bindings)
    operations = {value.call.operation_id: value for _, value, _ in control_calls}
    prepare_ids = {
        key: {
            call_id
            for call_id, value, prepare in control_calls
            if prepare and value.call.operation_id == key
        }
        for key in operations
    }
    action_ids = {
        key: {
            call_id
            for call_id, value, prepare in control_calls
            if not prepare and value.call.operation_id == key
        }
        for key in operations
    }
    records = {key: [] for key in operations}
    for segment in program.iter_segments():
        for policy in segment.post_policies:
            selected = bound.preset_for(policy, segment=segment)
            original = presets.get(policy.cfg.preset)
            if type(original) is not SimulationControlConditionCfg:
                if selected is not None:
                    raise ValueError(
                        "Post-policy preset binding currently requires a control condition."
                    )
                continue
            if selected is None:
                raise ValueError(
                    "Every control post-policy occurrence requires an explicit binding."
                )
            effective = presets.get(selected)
            if type(effective) is not SimulationControlConditionCfg:
                raise ValueError(
                    "Bound post-policy requires a declared control preset."
                )
            if (
                effective.operation_id != original.operation_id
                or effective.family != original.family
                or effective.phase != original.phase
            ):
                raise ValueError(
                    "Bound control post-policy cannot change operation, family or phase."
                )
            declaration = operations.get(effective.operation_id)
            if declaration is None or effective.entity != declaration.call.entity_id:
                raise ValueError(
                    "Bound control condition differs from its physical operation."
                )
            if policy.entity.entity_id not in {
                declaration.binding.object_id,
                declaration.call.entity_id,
            }:
                raise ValueError("Bound control policy has a foreign logical entity.")
            if (
                segment.parallel_block is not None
                or len(segment.calls) != 1
                or len(segment.post_policies) != 1
            ):
                raise ValueError(
                    "Bound control policies require serial single-call/policy segments."
                )
            records[effective.operation_id].append((segment, effective))
    for operation_id, declaration in operations.items():
        rows = records[operation_id]
        count = (
            declaration.chunk_count
            if type(declaration.call) is TwistCallDeclaration
            else 1
        )
        if len(rows) != count + 2:
            raise ValueError(
                "Bound control post-policy coverage differs from the complete operation."
            )
        resource = dict(rows[0][0].calls[0].call.resources)
        if set(resource) != {"primary"}:
            raise ValueError(
                "Bound control operations require one explicit primary resource."
            )
        for index, (segment, cfg) in enumerate(rows):
            call = segment.calls[0].call
            call_id = getattr(call, "call_id", None)
            if dict(call.resources) != resource:
                raise ValueError(
                    "Bound control operation resources differ across occurrences."
                )
            terminal = index == len(rows) - 1
            if cfg.terminal is not terminal:
                raise ValueError(
                    "Control terminal policy must bind exactly its cleanup occurrence."
                )
            if index == 0:
                valid = call_id in prepare_ids[operation_id] and cfg.phase == "ready"
            elif terminal:
                valid = (
                    call_id in cleanup_call_ids
                    and not call.arguments
                    and cfg.phase == "released"
                )
            else:
                phase = "chunk" if count > 1 and index < count else "released"
                valid = call_id in action_ids[operation_id] and cfg.phase == phase
            if not valid:
                raise ValueError(
                    "Bound control policy does not match its declared prepare/action/cleanup occurrence."
                )
    return bound
