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

"""GenSim E9 lowerers and physical-substep evidence; Gym owns all execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, ClassVar

import torch
from embodichain.utils import logger

from embodichain.lab.sim.atomic_actions import (
    JointPositionGoal,
    JointPositionTarget,
    MoveJoints,
    MoveJointsOptions,
    MoveEndEffector,
    MoveEndEffectorOptions,
    EndEffectorPoseGoal,
    GRASP_COMMAND,
    ObjectSemantics,
    Press,
    PressAffordance,
    PressGoal,
    PressOptions,
    RecoveryPolicy,
    SceneEntityPose,
)
from embodichain.lab.sim.sensors import (
    ContactSensor,
    ContactSensorCfg,
    ArticulationContactFilterCfg,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations.extensions import (
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.task_program.semantics import (
    SceneArticulationRef,
    SceneLinkRef,
    SkillPolicyPreset,
)

from .press import PressSample, evaluate_press_event, inspect_press
from .press_binding import PREPARE_CALL, PRESS_CALL, PRESS_REVISION, PressRoute
from .articulation_binding import PARK_CALL

__all__: list[str] = []
SENSOR_UID = "gen_sim_press_evidence"


class GenSimPress(Press):
    """Keep public Press planning and validate its final arm/hand command grid."""

    skill_id = Press.skill_id
    GoalType = Press.GoalType
    OptionsType = Press.OptionsType
    binding_contract = Press.binding_contract

    def _plan(self, request: Any, context: Any) -> Any:
        from .motion import _joint_velocity_limits, _velocity_validity

        plan = super()._plan(request, context)
        if plan.plan_success.any():
            trajectory = plan.joint_trajectory
            limits = _joint_velocity_limits(self.robot, None).to(trajectory.positions)
            if not _velocity_validity(
                trajectory.positions, trajectory.dt, limits
            ).all():
                return self.failed_plan(
                    request,
                    context,
                    message="E9 Press exceeds final resampled joint velocity limits.",
                )
        return plan


def with_press_options(profile: Any, routes: tuple[PressRoute, ...]) -> Any:
    if not routes:
        return profile
    if len(routes) != 1:
        raise ValueError("E9 MVP requires one exact press route.")
    route = routes[0]
    presets = []
    for preset in profile.presets:
        presets.append(
            SkillPolicyPreset(
                preset.preset_id,
                required_planner=preset.required_planner,
                action_option_templates={
                    **preset.action_option_templates,
                    PREPARE_CALL: MoveJointsOptions(),
                    PRESS_CALL: (
                        MoveEndEffectorOptions()
                        if route.variant == "approach_only"
                        else PressOptions(
                            approach_distance=0.08,
                            press_distance=route.binding.stroke
                            + route.extra_press_distance,
                            hand_interp_steps=12,
                        )
                    ),
                },
                effect_assurance=preset.effect_assurance,
                effect_monitors=preset.effect_monitors,
                motion_policy=replace(
                    preset.motion_policy,
                    strategy="ik_interp",
                    sample_count=max(260, preset.motion_policy.sample_count),
                ),
                tracking_policy=preset.tracking_policy,
                recovery_policy=RecoveryPolicy(max_replans=0, max_action_retries=0),
                workflow_recovery_policy=replace(
                    preset.workflow_recovery_policy, max_recovery_attempts=0
                ),
                runner_cfg=preset.runner_cfg,
            )
        )
    return replace(profile, presets=tuple(presets))


def _bind(route: PressRoute, simulation: Any, registry: Any, engine: Any) -> Any:
    if simulation.num_envs != 1:
        raise ValueError("E9 MVP requires one environment.")
    b = route.binding
    entry = registry.lookup(route.link_id, expected_type=SceneLinkRef)
    if entry.parent != SceneArticulationRef(b.object_id) or entry.native_name != b.link:
        raise ValueError("E9 link ownership changed.")
    art = simulation.get_articulation(b.object_id)
    if art is None:
        raise ValueError("E9 articulation is missing.")
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
        raise ValueError("E9 source hash or bound geometry changed.")
    chain = [j for j in art.get_parent_joint_chain(b.link) if j.joint_type != "fixed"]
    if (
        len(chain) != 1
        or chain[0].name != b.joint
        or chain[0].parent_link_name != b.parent
    ):
        raise ValueError("E9 runtime joint topology changed.")
    if route.variant != "original" and not torch.allclose(
        torch.tensor(chain[0].joint_limits),
        torch.tensor(b.limits),
        atol=1e-7,
        rtol=1e-5,
    ):
        raise ValueError("E9 runtime limits do not match the declared scaled travel.")
    return art


class _PrepareLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = PREPARE_CALL
    target_descriptor = MoveJoints.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, route: PressRoute, art: Any, engine: Any) -> None:
        self.route, self.art, self.engine = route, art, engine

    def _check(self, call: Any, context: Any, bound: Any) -> Any:
        if dict(call.arguments) != {"object": self.route.binding.object_id}:
            raise ValueError("E9 call must reference its exact declared button.")
        endpoint = bound.binding.action_binding.endpoint("primary", "motion")
        if (
            context.task.get_held_object(endpoint.task_state_key) is not None
            or context.task.coordinated_held_objects
        ):
            raise ValueError("E9 cannot press while holding another object.")
        return endpoint.require_target(JointPositionTarget)

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        target = self._check(call, context, bound)
        if type(option_template) is not MoveJointsOptions:
            raise TypeError("E9 preparation requires MoveJointsOptions.")
        return SemanticLowering(
            goal=JointPositionGoal(
                context.robot.qpos[:, list(target.joint_ids)].clone()
            )
        )


class _PressLowerer(_PrepareLowerer):
    call_id: ClassVar[str] = PRESS_CALL
    target_descriptor = Press.descriptor()

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        self._check(call, context, bound)
        b = self.route.binding
        q = self.art.get_qpos()[0, self.art.joint_names.index(b.joint)]
        if (
            not torch.isfinite(q)
            or abs(float(q) - b.released_position) > b.stroke * 0.1
        ):
            raise ValueError("E9 button is not released before Press.")
        if type(option_template) is not PressOptions or not math.isclose(
            option_template.press_distance, b.stroke + self.route.extra_press_distance
        ):
            raise ValueError(
                "E9 options differ from the declared press command distance."
            )
        offset = _closed_finger_tip_offset(self.engine.robot, context, bound)
        point = tuple(p - offset * a for p, a in zip(b.press_position, b.axis))
        logger.log_info(f"E9 measured closed-finger tool offset: {offset:.6f} m")
        return SemanticLowering(
            goal=PressGoal(
                ObjectSemantics(
                    entity_id=self.route.link_id,
                    geometry={},
                    affordance=PressAffordance(
                        press_axis=torch.tensor(b.axis, device=self.engine.device),
                        press_position=point,
                    ),
                ),
                SceneEntityPose(self.route.link_id),
            )
        )


def _closed_finger_tip_offset(robot: Any, context: Any, bound: Any) -> float:
    """Express the closed hand's leading geometry relative to its grasp TCP."""
    motion = bound.binding.action_binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    grasp = bound.binding.action_binding.endpoint("primary", "grasp")
    hand = grasp.require_target(JointPositionTarget)
    qpos = context.robot.qpos.clone()
    qpos[:, list(hand.joint_ids)] = grasp.joint_positions(
        GRASP_COMMAND, num_envs=context.batch_size, device=qpos.device, dtype=qpos.dtype
    )
    prefix = motion.control_part.removesuffix("_arm") + "_"
    links = [
        name
        for name in robot.link_names
        if name.startswith(prefix) and "finger" in name
    ]
    solver = robot.cfg.solver_cfg[motion.control_part]
    poses = robot.compute_fk(
        qpos=qpos,
        link_names=[solver.end_link_name, *links],
        qpos_joint_names=robot.joint_names,
    )
    tcp = poses[0, 0] @ poses.new_tensor(solver.tcp)
    inverse = torch.linalg.inv(tcp)
    leading = []
    for index, name in enumerate(links, start=1):
        vertices, _ = robot.get_link_vert_face(name)
        transform = inverse @ poses[0, index]
        points = vertices.to(transform) @ transform[:3, :3].T + transform[:3, 3]
        leading.append(points[:, 2].max())
    offset = float(torch.stack(leading).max())
    if not math.isfinite(offset) or not 0 <= offset <= 0.15:
        raise ValueError("E9 closed-finger press-tip calibration is unavailable.")
    return offset


@dataclass(frozen=True, slots=True)
class PressPrepareFactory(RegisteredSemanticLowererFactory):
    route: PressRoute
    call_id: ClassVar[str] = PREPARE_CALL
    revision: ClassVar[str] = PRESS_REVISION
    target_descriptor = MoveJoints.descriptor()

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> Any:
        if engine.robot is not robot:
            raise ValueError("E9 must use the integration-owned robot.")
        return _PrepareLowerer(
            self.route, _bind(self.route, simulation, scene_registry, engine), engine
        )


@dataclass(frozen=True, slots=True)
class PressFactory(PressPrepareFactory):
    call_id: ClassVar[str] = PRESS_CALL

    @property
    def target_descriptor(self) -> Any:
        return (
            MoveEndEffector.descriptor()
            if self.route.variant == "approach_only"
            else Press.descriptor()
        )

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> Any:
        if engine.robot is not robot:
            raise ValueError("E9 must use the integration-owned robot.")
        lowerer = (
            _ApproachLowerer if self.route.variant == "approach_only" else _PressLowerer
        )
        return lowerer(
            self.route, _bind(self.route, simulation, scene_registry, engine), engine
        )


class _ApproachLowerer(_PrepareLowerer):
    """Declared negative control: reach the approach pose without touching."""

    call_id: ClassVar[str] = PRESS_CALL
    target_descriptor = MoveEndEffector.descriptor()

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        self._check(call, context, bound)
        if type(option_template) is not MoveEndEffectorOptions:
            raise TypeError(
                "The approach-only control requires MoveEndEffectorOptions."
            )
        b = self.route.binding
        affordance = PressAffordance(
            press_axis=torch.tensor(b.axis, device=self.engine.device),
            press_position=b.press_position,
        )
        pose = affordance.get_press_pose(self.art.get_link_pose(b.link, to_matrix=True))
        pose[:, :3, 3] -= 0.08 * pose[:, :3, 2]
        return SemanticLowering(goal=EndEffectorPoseGoal(pose))


class PressContactSensor(ContactSensor):
    """Record actual joint/contact substeps, including transient press events."""

    def __init__(self, config: ContactSensorCfg, device: Any, *, owner: Any) -> None:
        # A single button has few contact rows. Keep their reduction on CPU
        # rather than synchronizing CUDA separately for each scalar predicate.
        super().__init__(config, torch.device("cpu"), owner=owner)

    def configure(self, route: PressRoute, robot: Any) -> None:
        self.route, self.robot = route, robot
        self.art = self._sim.get_articulation(route.binding.object_id)
        self.joint_index = self.art.joint_names.index(route.binding.joint)
        self.target_actor = int(
            self.get_actor_ids(self.art.uid, [route.binding.link])[0, 0]
        )
        self.parent_actor = int(
            self.get_actor_ids(self.art.uid, [route.binding.parent])[0, 0]
        )
        names = [
            n
            for n in robot.link_names
            if n.startswith(f"{route.arm}_") and "finger" in n
        ]
        self.finger_names = tuple(names)
        self.finger_actors = set(self.get_actor_ids(robot.uid, names)[0].tolist())
        self.contact_envelopes = {}
        for obj, links in (
            (self.art, (route.binding.parent, route.binding.link)),
            (robot, names),
        ):
            for link in links:
                attr = obj.get_link_physical_attr(link)[0]
                self.contact_envelopes[f"{obj.uid}/{link}"] = {
                    "contact_offset": float(attr.contact_offset),
                    "rest_offset": float(attr.rest_offset),
                }
        self.clock = 0.0
        self.armed = False
        self.phase = "press"
        self.samples: list[PressSample] = []
        self.trace: list[dict[str, Any]] = []
        self.acceptance: dict[str, Any] = {}

    @property
    def requires_substep_update(self) -> bool:
        return hasattr(self, "route")

    def update_physics_step(self, dt: float) -> None:
        super().update_physics_step(dt)
        self.clock += dt
        if self.armed:
            self.capture(self.phase)

    def capture(self, phase: str) -> PressSample:
        data = self.get_data()
        mask = data["is_valid"][0]
        pairs = data["user_ids"][0, mask].tolist()
        impulse = data["impulse"][0, mask].tolist()
        valid = self.dropped_contacts == 0 and all(
            bool(torch.isfinite(data[key][0, mask]).all())
            for key in ("impulse", "position", "normal", "distance")
        )
        contact = any(
            self.target_actor in pair
            and any(a in self.finger_actors for a in pair)
            and force > 0
            for pair, force in zip(pairs, impulse)
        )
        q = float(self.art.get_qpos()[0, self.joint_index])
        sample = PressSample(self.clock, q, phase, contact if valid else None, valid)
        self.samples.append(sample)
        self.trace.append(
            {
                **asdict(sample),
                "contact_pairs": pairs,
                "impulses": impulse,
                "dropped_contacts": self.dropped_contacts,
                "parent_contact": (
                    any(
                        self.parent_actor in pair
                        and any(a in self.finger_actors for a in pair)
                        and force > 0
                        for pair, force in zip(pairs, impulse)
                    )
                    if valid
                    else None
                ),
            }
        )
        if len(self.trace) % 200 == 0:
            logger.log_info(
                f"E9 observed substeps={len(self.trace)}, qpos={q:.6f}, contact={contact}, valid={valid}"
            )
        return sample

    def arm(self) -> None:
        self.samples.clear()
        self.trace.clear()
        self.acceptance = {}
        self.phase = "press"
        self.update()
        try:
            self.capture("prepare")
        except Exception as exc:
            self.acceptance = {
                "accepted": False,
                "phase": "observation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            raise
        self.armed = True

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        if hasattr(self, "route"):
            self.armed = False


def ensure_sensor(simulation: Any, robot: Any, route: PressRoute) -> PressContactSensor:
    sensor = simulation.get_sensor(SENSOR_UID)
    if sensor is not None:
        if not isinstance(sensor, PressContactSensor) or sensor.route != route:
            raise ValueError(
                "E9 sensor identity conflicts with the current integration."
            )
        return sensor
    # Instance-local sensor construction; do not mutate the shared sensor registry.
    factories = simulation.SUPPORTED_SENSOR_TYPES
    simulation.SUPPORTED_SENSOR_TYPES = {
        **factories,
        "GenSimPressContact": PressContactSensor,
    }
    try:
        sensor = simulation.add_sensor(
            ContactSensorCfg(
                uid=SENSOR_UID,
                sensor_type="GenSimPressContact",
                max_contacts_per_env=512,
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(articulation_uid=robot.uid),
                    ArticulationContactFilterCfg(
                        articulation_uid=route.binding.object_id
                    ),
                ],
            )
        )
    finally:
        simulation.SUPPORTED_SENSOR_TYPES = factories
    sensor.configure(route, robot)
    return sensor


class PressAcceptancePort:
    """Task Program post-policies consume evidence; they do not execute Press."""

    def __init__(
        self,
        delegate: Any,
        simulation: Any,
        robot: Any,
        route: PressRoute,
        step_dt: float,
    ) -> None:
        self.delegate, self.robot, self.route, self.dt = delegate, robot, route, step_dt
        self.sensor = ensure_sensor(simulation, robot, route)
        self.results: dict[int, Any] = {}
        self.metadata: dict[int, Any] = {}

    def validate_policy(self, policy: Any, *, segment: Any) -> None:
        if policy.cfg.preset not in {
            self.route.preset("ready"),
            self.route.preset("pressed"),
        }:
            return self.delegate.validate_policy(policy, segment=segment)
        if (
            policy.cfg.kind != "wait_stable"
            or policy.entity.entity_id != self.route.binding.object_id
        ):
            raise ValueError("E9 post-policy entity does not match its button.")

    def actions(self, policy: Any, *, segment: Any, active_mask: torch.Tensor) -> Any:
        self.validate_policy(policy, segment=segment)
        ready = policy.cfg.preset == self.route.preset("ready")
        if not ready and policy.cfg.preset != self.route.preset("pressed"):
            yield from self.delegate.actions(
                policy, segment=segment, active_mask=active_mask
            )
            return
        b, sensor = self.route.binding, self.sensor
        hold = self.robot.get_qpos().clone()
        if ready:
            sensor.armed = False
            values = []
            for _ in range(math.ceil(0.5 / self.dt)):
                yield hold.clone()
                values.append(float(sensor.art.get_qpos()[0, sensor.joint_index]))
            sensor.update()
            sensor.arm()
            good = all(
                math.isfinite(v) and abs(v - b.released_position) <= b.stroke * 0.1
                for v in values
            )
            good = (
                good
                and sensor.samples[0].valid
                and sensor.samples[0].target_contact is False
            )
            good = good and torch.allclose(
                sensor.art.get_qpos_limits()[0, sensor.joint_index].cpu(),
                torch.tensor(b.limits),
                atol=1e-7,
                rtol=1e-5,
            )
            result = {
                "accepted": good,
                "phase": "released_stable",
                "joint_positions": values,
            }
            sensor.acceptance = {
                "accepted": False,
                "phase": "ready_only",
                "released_stable": good,
                "joint_positions": values,
            }
            logger.log_info(
                f"E9 released-state check: object={b.object_id}, accepted={good}, qpos={values[-1]}"
            )
            if not good:
                sensor.armed = False
        else:
            for _ in range(math.ceil(0.2 / self.dt)):
                yield hold.clone()
            event = evaluate_press_event(b, sensor.samples)
            contact_pose = sensor.art.get_link_pose(b.link, to_matrix=True)[0]
            point = (
                contact_pose[:3, :3] @ contact_pose.new_tensor(b.press_position)
                + contact_pose[:3, 3]
            )
            inward = contact_pose[:3, :3] @ contact_pose.new_tensor(b.axis)
            tcp = self.robot.compute_fk(
                qpos=self.robot.get_qpos(name=f"{self.route.arm}_arm"),
                name=f"{self.route.arm}_arm",
                to_matrix=True,
            )[0, :3, 3]
            tcp_clearance = float(torch.dot(point - tcp, inward))
            clearances = []
            for name in sensor.finger_names:
                pose = self.robot.get_link_pose(name, to_matrix=True)[0]
                vertices, _ = self.robot.get_link_vert_face(name)
                world = vertices.to(pose) @ pose[:3, :3].T + pose[:3, 3]
                clearances.append(float(((point - world) @ inward).min()))
            clearance = min(clearances)
            no_contact = bool(sensor.samples) and all(
                s.target_contact is False and s.valid
                for s in sensor.samples[
                    -max(1, math.ceil(0.15 / sensor._sim.sim_config.physics_dt)) :
                ]
            )
            terminal = any(call.call.semantic_id == PARK_CALL for call in segment.calls)
            qpos = float(sensor.art.get_qpos()[0, sensor.joint_index])
            good = bool(event["accepted"] and no_contact and clearance >= 0.04)
            result = {
                **event,
                "accepted": good,
                "event_accepted": event["accepted"],
                "retreat_clearance": clearance,
                "tcp_clearance": tcp_clearance,
                "contact_released": no_contact,
                "terminal": terminal,
                "observed_joint_position": qpos,
            }
            sensor.acceptance = deepcopy(result)
            logger.log_info(f"E9 press acceptance: {result}")
            sensor.phase = "cleanup"
            sensor.armed = good and not terminal
        self.results[id(policy)] = torch.full_like(active_mask, good) & active_mask
        self.metadata[id(policy)] = result

    def post_policy_result(self, policy: Any, *, segment: Any) -> Any:
        if id(policy) in self.results:
            return self.results[id(policy)].clone()
        return self.delegate.post_policy_result(policy, segment=segment)

    def post_policy_metadata(self, policy: Any, *, segment: Any) -> Any:
        if id(policy) in self.metadata:
            return deepcopy(self.metadata[id(policy)])
        return self.delegate.post_policy_metadata(policy, segment=segment)
