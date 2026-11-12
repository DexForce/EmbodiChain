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

"""Material-coordinate grounding and measured contact for configured programs.

This service owns no task sequence, controller, motion generator or simulator
step. Program calls use the ordinary Atomic Skills and Gym command bridge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import ClassVar, Any

import torch

from embodichain.compute.geometry.cable import sample_cable_material, cable_endpoints
from embodichain.compute.geometry._cable_route import _measure_cable_route
from embodichain.lab.sim.atomic_skills import (
    AtomicSkillEngine,
    SkillControlOverrides,
    JointPositionCommand,
    SkillDescriptor,
    PlanningContext,
    SkillOptions,
)
from embodichain.lab.sim.atomic_skills.primitives.pick_up import (
    PickUp,
    ContactGraspGoal,
)
from embodichain.lab.sim.atomic_skills.primitives.move_end_effector import (
    MoveEndEffector,
    EndEffectorPoseGoal,
)
from embodichain.lab.sim.atomic_skills.primitives.move_joints import (
    MoveJoints,
    JointPositionGoal,
)
from embodichain.lab.sim.motion.planners.utils import interpolate_xpos_batched
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.language.semantic_contracts import (
    SemanticCallContract,
)
from embodichain.lab.task_program.semantics import (
    RegisteredSemanticCall,
    BoundSemanticCall,
    SceneRegistry,
)
from ..extensions import RegisteredSemanticLowererFactory

__all__: list[str] = []

from embodichain.compute.geometry._cable_route import _CABLE_METRICS as _METRICS


@dataclass(frozen=True, slots=True)
class _CableTarget:
    target_id: str
    kind: str
    material_distance: float | None = None
    position: tuple[float | None, float | None, float | None] = (None, None, None)
    speed: float = 0.05
    tolerance: float = 0.02
    orientation_tolerance: float = 0.25
    port_screen: bool = False
    firm: bool = False
    stop_when_seated: bool = False
    maximum_correction: float = 0.005
    maximum_z_correction: float = 0.003
    minimum_z: float | None = None
    maximum_y: float | None = None
    samples: int = 2
    finger_position: float = 0.0475
    condition: str | None = None
    release_part: str | None = None


@dataclass(frozen=True, slots=True)
class _CableService:
    entity_id: str
    simulation_uid: str
    targets: tuple[_CableTarget, ...]
    geometry: tuple[tuple[str, object], ...]
    motion_parts: tuple[str, ...]
    slip_limit: float = 0.012

    def target(self, target_id: str) -> _CableTarget:
        for target in self.targets:
            if target.target_id == target_id:
                return target
        raise ValueError(f"Unknown cable target {target_id!r}.")


class _CableRuntime:
    """One bridge's references and metrics, in explicit native arena order."""

    def __init__(
        self, cfg: _CableService, simulation: Any, robot: Any, env_ids: torch.Tensor
    ) -> None:
        self.cfg, self.simulation, self.robot = cfg, simulation, robot
        self.env_ids = env_ids.clone()
        self.cable = simulation.get_cable_object(cfg.simulation_uid)
        if self.cable is None:
            raise ValueError(f"Cable {cfg.simulation_uid!r} is absent.")
        arenas = tuple(self.cable.data.batch.arena_indices)
        by_arena = {int(value): row for row, value in enumerate(arenas)}
        try:
            selected_arenas = [
                simulation.spawn_result.arenas.index(
                    robot.body_data.entities[int(value)].arena_name
                )
                for value in env_ids.tolist()
            ]
            rows = [by_arena[value] for value in selected_arenas]
        except KeyError as exc:
            raise ValueError(
                "The cable does not cover the selected environment rows."
            ) from exc
        self.rows = torch.tensor(rows, device=env_ids.device)
        points = torch.as_tensor(
            self.cable.cfg.points, device=env_ids.device, dtype=torch.float32
        )
        self.lengths = points.diff(dim=0).norm(dim=-1)
        origins = torch.as_tensor(
            simulation.spawn_result.arenas.root_offsets,
            device=env_ids.device,
            dtype=torch.float32,
        )
        self.origins = origins.index_select(
            0, torch.tensor(selected_arenas, device=env_ids.device)
        )
        self.geometry = dict(cfg.geometry)
        self._scope_guard = lambda: None
        self.home = {part: self.tool(part).clone() for part in cfg.motion_parts}
        self.pending_pick: tuple[str, _CableTarget] | None = None
        self.hand_to_motion: dict[str, str] = {}
        self.grasps: dict[str, tuple[float, torch.Tensor]] = {}
        self.max_slip = torch.zeros(len(rows), device=env_ids.device)
        self.grasp_slip: dict[str, torch.Tensor] = {}
        self.last_targets: dict[str, torch.Tensor] = {}
        self.pick_initial: dict[str, tuple[float, torch.Tensor]] = {}
        self.port_yaw: dict[str, torch.Tensor] = {}
        self.closed_reference_samples: list[dict[str, object]] = []
        self._metric_cache_time: float | None = None
        self._metric_cache: dict[str, torch.Tensor] = {}
        for target in cfg.targets:
            if target.material_distance is not None:
                # Configuration errors fail before the first command.
                sample_cable_material(
                    self.state(), self.lengths, target.material_distance
                )

    def state(self) -> torch.Tensor:
        self._scope_guard()
        return self.cable.data.segment_state_w.index_select(0, self.rows)

    def tool(
        self, part: str, *, commanded: bool = False, world: bool = False
    ) -> torch.Tensor:
        self._scope_guard()
        joints = self.robot.get_joint_ids(name=part)
        qpos = self.robot.get_qpos(target=commanded).index_select(0, self.env_ids)[
            :, joints
        ]
        pose = self.robot.compute_fk(
            qpos, name=part, to_matrix=True, env_ids=self.env_ids
        ).clone()
        if world:
            pose[:, :3, 3] += self.origins
        return pose

    def point(self, distance: float) -> tuple[torch.Tensor, torch.Tensor, int]:
        return sample_cable_material(self.state(), self.lengths, distance)

    def point_in_tool(self, part: str, distance: float) -> torch.Tensor:
        point = self.point(distance)[0]
        tool = self.tool(part, world=True)
        return (
            tool[:, :3, :3].transpose(1, 2) @ (point - tool[:, :3, 3]).unsqueeze(-1)
        ).squeeze(-1)

    def observe(self) -> torch.Tensor:
        for part, (distance, baseline) in self.grasps.items():
            slip = (self.point_in_tool(part, distance) - baseline).norm(dim=-1)
            self.grasp_slip[part] = torch.maximum(self.grasp_slip[part], slip)
            self.max_slip = torch.maximum(self.max_slip, slip)
        state = self.state()
        invalid = ~torch.isfinite(state).all(dim=(1, 2)) | ~(
            state[..., 3:7].norm(dim=-1) > 1e-8
        ).all(-1)
        return (self.max_slip > self.cfg.slip_limit) | invalid

    def metrics(self) -> dict[str, torch.Tensor]:
        self._scope_guard()
        timestamp = getattr(self.simulation, "simulation_time", None)
        if timestamp is not None and timestamp == self._metric_cache_time:
            return {key: value.clone() for key, value in self._metric_cache.items()}
        self.observe()
        hands = torch.stack(
            [self.tool(part, world=True)[:, :3, 3] for part in self.cfg.motion_parts], 1
        )
        result = _measure_cable_route(
            self.state(),
            self.lengths,
            hands,
            self.origins,
            self.geometry,
            float(self.cable.cfg.radius),
        )
        result["max_grasp_slip"] = self.max_slip.clone()
        if timestamp is not None:
            self._metric_cache_time = timestamp
            self._metric_cache = {key: value.clone() for key, value in result.items()}
        return result

    def seated(self) -> torch.Tensor:
        measurements = self.metrics()
        return measurements["release_ready"] & torch.stack(
            [
                measurements[key] <= 0.001
                for key in (
                    "port_penetration",
                    "bench_penetration",
                    "spool_penetration",
                )
            ]
        ).all(0)

    def after_action(self, action: Any, runtime: Any) -> tuple[torch.Tensor, str]:
        """Capture only after Gym acknowledges the final physical close step."""
        self._scope_guard()
        phase = (
            runtime._last_sent_command_phase()
            if hasattr(runtime, "_last_sent_command_phase")
            else None
        )
        if (
            self.pending_pick is not None
            and action.metadata.get("bridge_action_kind") == "runtime_command"
            and phase is not None
        ):
            if phase[0] == "close" and phase[1]:
                part, target = self.pending_pick
                distance = target.material_distance
                assert distance is not None
                self.grasps[part] = (
                    distance,
                    self.point_in_tool(part, distance).clone(),
                )
                self.grasp_slip[part] = torch.zeros_like(self.max_slip)
                self.pending_pick = None
                self.closed_reference_samples.append(
                    {
                        "part": part,
                        "material_distance": distance,
                        "phase": "close",
                        "sample_index": phase[2],
                    }
                )
        return self.observe(), "cable_material_state_or_slip_guard_failed"

    def _material_pose(self, target: _CableTarget, part: str) -> torch.Tensor:
        assert target.material_distance is not None
        point, tangent, _ = self.point(target.material_distance)
        yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        candidates = []
        joints = self.robot.get_joint_ids(name=part)
        seed = self.robot.get_qpos().index_select(0, self.env_ids)[:, joints]
        for branch, angle in enumerate((yaw, yaw + math.pi)):
            pose = torch.eye(4, device=point.device).repeat(len(point), 1, 1)
            c, s = torch.cos(angle), torch.sin(angle)
            pose[:, 0, 0], pose[:, 0, 1] = c, s
            pose[:, 1, 0], pose[:, 1, 1] = s, -c
            pose[:, 2, 2] = -1
            pose[:, :3, 3] = point - self.origins
            if target.position[2] is not None:
                pose[:, 2, 3] = target.position[2]
            success, qpos = self.robot.compute_ik(
                pose, joint_seed=seed, name=part, env_ids=self.env_ids
            )
            if target.port_screen:
                stage = pose.clone()
                port_angle = math.pi / 2 if branch == 0 else -math.pi / 2
                stage[:, 0, 0], stage[:, 0, 1] = math.cos(port_angle), math.sin(
                    port_angle
                )
                stage[:, 1, 0], stage[:, 1, 1] = math.sin(port_angle), -math.cos(
                    port_angle
                )
                stage[:, :3, 3] = point.new_tensor(self.geometry["port_stage"])
                later, _ = self.robot.compute_ik(
                    stage, joint_seed=qpos, name=part, env_ids=self.env_ids
                )
                success = success.reshape(-1).bool() & later.reshape(-1).bool()
            candidates.append((success.reshape(-1).bool(), qpos, pose))
        first, second = candidates
        choose = second[0] & (
            ~first[0]
            | ((second[1] - seed).square().sum(-1) < (first[1] - seed).square().sum(-1))
        )
        if not bool((first[0] | second[0]).all()):
            raise ValueError("No reachable wrist branch for the cable material target.")
        if target.port_screen:
            self.port_yaw[part] = torch.where(choose, -math.pi / 2, math.pi / 2)
        return torch.where(choose[:, None, None], second[2], first[2])

    def pose(self, target: _CableTarget, part: str) -> torch.Tensor:
        if target.kind == "material":
            return self._material_pose(target, part)
        measured = self.tool(part)
        if target.kind == "home":
            return self.home[part].clone()
        pose = self.tool(part, commanded=target.kind in {"command", "tip_align"})
        if target.kind == "port_stage":
            angle = self.port_yaw[part]
            c, s = torch.cos(angle), torch.sin(angle)
            pose[:, 0, 0], pose[:, 0, 1] = c, s
            pose[:, 1, 0], pose[:, 1, 1] = s, -c
            pose[:, 2, 2] = -1
        if target.kind in {"yaw_align", "ray_align"}:
            assert target.material_distance is not None
            point = self.point(target.material_distance)[0] - self.origins
            end = cable_endpoints(self.state(), self.lengths)[1][:, -1] - self.origins
            direction = end - point
            if target.kind == "yaw_align":
                angle = torch.atan2(direction[:, 0], direction[:, 1])
                c, s = torch.cos(angle), torch.sin(angle)
                rotation = torch.eye(3, device=pose.device).repeat(len(pose), 1, 1)
                rotation[:, 0, 0], rotation[:, 0, 1] = c, -s
                rotation[:, 1, 0], rotation[:, 1, 1] = s, c
                pose[:, :3, :3] = rotation @ pose[:, :3, :3]
            else:
                if not bool((direction[:, 1] > 1e-6).all()):
                    raise ValueError(
                        "The distal cable span points away from the channel."
                    )
                mouth_x, mouth_y = self.geometry["port_mouth"]
                meets = point[:, 0] + direction[:, 0] / direction[:, 1] * (
                    mouth_y - point[:, 1]
                )
                pose[:, 0, 3] += mouth_x - meets
        elif target.kind == "tip_align":
            end = cable_endpoints(self.state(), self.lengths)[1][:, -1] - self.origins
            for axis, value in enumerate(target.position):
                if value is not None:
                    pose[:, axis, 3] += (value - end[:, axis]).clamp(
                        -(
                            target.maximum_z_correction
                            if axis == 2
                            else target.maximum_correction
                        ),
                        (
                            target.maximum_z_correction
                            if axis == 2
                            else target.maximum_correction
                        ),
                    )
            if target.maximum_y is not None:
                pose[:, 1, 3].clamp_(max=target.maximum_y)
            if target.minimum_z is not None:
                pose[:, 2, 3].clamp_(min=target.minimum_z)
        else:
            for axis, value in enumerate(target.position):
                if value is not None:
                    pose[:, axis, 3] = value
        if target.stop_when_seated:
            pose = torch.where(self.seated()[:, None, None], measured, pose)
        return pose


class _CableLowerer(RegisteredSemanticLowerer):
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, cfg: _CableService) -> None:
        self.cfg = cfg
        self.runtime: _CableRuntime | None = None

    def _target(
        self, call: RegisteredSemanticCall
    ) -> tuple[_CableRuntime, _CableTarget]:
        if dict(call.arguments).keys() != {"target"}:
            raise ValueError(
                "Cable calls require exactly one declared target identifier."
            )
        if self.runtime is None:
            raise RuntimeError("Cable service was not bound to this runtime assembly.")
        return self.runtime, self.cfg.target(call.arguments["target"])

    @staticmethod
    def _part(bound: BoundSemanticCall, endpoint: str = "motion") -> str:
        return (
            bound.binding.resources["primary"]
            .endpoints[endpoint]
            .runtime_target.control_part
        )


class _CablePickLowerer(_CableLowerer):
    call_id: ClassVar[str] = "simulation.cable_pick"
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        service, target = self._target(call)
        if target.kind != "material":
            raise ValueError("Cable pickup requires a material target.")
        part = self._part(bound)
        service.pending_pick = (part, target)
        service.hand_to_motion[self._part(bound, "grasp")] = part
        assert target.material_distance is not None
        service.pick_initial[part] = (
            target.material_distance,
            service.point(target.material_distance)[0][:, 2].clone(),
        )
        overrides = SkillControlOverrides()
        if target.firm:
            grasp_part = self._part(bound, "grasp")
            count = len(service.robot.get_joint_ids(name=grasp_part))
            overrides = SkillControlOverrides(
                endpoints={
                    "primary": {
                        "grasp": {
                            "grasp": JointPositionCommand(
                                torch.zeros(count, device=context.env_ids.device)
                            )
                        }
                    }
                }
            )
        return SemanticLowering(
            goal=ContactGraspGoal(service.pose(target, part)),
            control_overrides=overrides,
        )


class _CableMoveLowerer(_CableLowerer):
    call_id: ClassVar[str] = "simulation.cable_move"
    target_descriptor: ClassVar[SkillDescriptor] = MoveEndEffector.descriptor()

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        service, target = self._target(call)
        part = self._part(bound)
        start, end = service.tool(part), service.pose(target, part)
        distance = (start[:, :3, 3] - end[:, :3, 3]).norm(dim=-1).max().item()
        trace = (
            (start[:, :3, :3].transpose(1, 2) @ end[:, :3, :3])
            .diagonal(dim1=-2, dim2=-1)
            .sum(-1)
        )
        angle = torch.acos(((trace - 1) / 2).clamp(-1, 1)).max().item()
        count = max(
            3,
            math.ceil(
                max(distance / target.speed, angle / 0.5) / context.require_control_dt()
            )
            + 1,
        )
        service.last_targets[part] = end.clone()
        return SemanticLowering(
            goal=EndEffectorPoseGoal(interpolate_xpos_batched(start, end, count)[:, 1:])
        )


class _CableJointLowerer(_CableLowerer):
    call_id: ClassVar[str] = "simulation.cable_joint"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        service, target = self._target(call)
        part = self._part(bound)
        ids = service.robot.get_joint_ids(name=part)
        start = service.robot.get_qpos(target=True).index_select(0, context.env_ids)[
            :, ids
        ]
        path = start[:, None, :].repeat(1, target.samples, 1)
        if target.kind == "open":
            # Only release the grasp whose configured gripper is being opened.
            motion_part = target.release_part
            if (
                motion_part not in self.cfg.motion_parts
                or service.hand_to_motion.get(part) != motion_part
            ):
                raise ValueError("Opening a hand requires its declared motion part.")
            service.grasps.pop(motion_part, None)
            service.pending_pick = None
            path = torch.lerp(
                start[:, None],
                torch.full_like(start[:, None], target.finger_position),
                torch.linspace(0, 1, target.samples, device=start.device)[
                    None, :, None
                ],
            )
        elif target.kind != "hold":
            raise ValueError("Cable joint targets must be open or hold declarations.")
        return SemanticLowering(goal=JointPositionGoal(path))


class _CableCheckLowerer(_CableLowerer):
    call_id: ClassVar[str] = "simulation.cable_check"
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()

    def lower(
        self,
        call: RegisteredSemanticCall,
        *,
        context: PlanningContext,
        bound: BoundSemanticCall,
        option_template: SkillOptions,
    ) -> SemanticLowering:
        service, target = self._target(call)
        part = self._part(bound)
        passed = ~service.observe()
        details = {}
        if target.condition == "pose":
            expected, actual = service.last_targets[part], service.tool(part)
            error = (actual[:, :3, 3] - expected[:, :3, 3]).norm(dim=-1)
            trace = (
                (actual[:, :3, :3].transpose(1, 2) @ expected[:, :3, :3])
                .diagonal(dim1=-2, dim2=-1)
                .sum(-1)
            )
            angle = torch.acos(((trace - 1) / 2).clamp(-1, 1))
            passed &= (error <= target.tolerance) & (
                angle <= target.orientation_tolerance
            )
            if target.stop_when_seated:
                passed |= service.seated() & ~service.observe()
            details = {
                "position_error_m": error.tolist(),
                "orientation_error_rad": angle.tolist(),
            }
        elif target.condition == "lift":
            distance, before = service.pick_initial[part]
            lift = service.point(distance)[0][:, 2] - before
            passed &= (lift > 0.008) & (part in service.grasps)
            details = {"material_lift_m": lift.tolist()}
        elif target.condition == "seated":
            passed &= service.seated()
        else:
            raise ValueError("Unknown declared cable precondition.")
        if not bool(passed.all()):
            raise ValueError(
                f"Cable condition {target.target_id!r} failed in envs {context.env_ids[~passed].tolist()}: {details}; max_slip_m={service.max_slip.tolist()}"
            )
        ids = service.robot.get_joint_ids(name=part)
        qpos = service.robot.get_qpos(target=True).index_select(0, context.env_ids)[
            :, ids
        ]
        return SemanticLowering(goal=JointPositionGoal(qpos[:, None].repeat(1, 2, 1)))


@dataclass(frozen=True, slots=True)
class _CableFactory(RegisteredSemanticLowererFactory):
    cfg: _CableService
    revision: ClassVar[str] = "1"
    lowerer_type: ClassVar[type[_CableLowerer]]

    @property
    def argument_contract(self) -> SemanticCallContract:
        return SemanticCallContract(
            required={
                "target": tuple(
                    target.target_id
                    for target in self.cfg.targets
                    if self._supports(target)
                )
            },
            optional={},
        )

    def _supports(self, target: _CableTarget) -> bool:
        if self.call_id == "simulation.cable_pick":
            return target.kind == "material"
        if self.call_id == "simulation.cable_joint":
            return target.kind in {"open", "hold"}
        if self.call_id == "simulation.cable_check":
            return target.kind == "check"
        return target.kind not in {"open", "hold", "check"}

    def create(
        self,
        *,
        simulation: object,
        robot: object,
        scene_registry: SceneRegistry,
        engine: AtomicSkillEngine,
    ) -> RegisteredSemanticLowerer:
        if engine.robot is not robot:
            raise ValueError("Cable services must use the engine's exact robot.")
        lowerer = self.lowerer_type(self.cfg)
        selected = engine.planning_services.env_ids
        ids = (
            torch.arange(robot.get_qpos().shape[0], device=engine.device)
            if selected is None
            else torch.tensor(selected, dtype=torch.long, device=engine.device)
        )
        lowerer.runtime = _CableRuntime(self.cfg, simulation, robot, ids)
        return lowerer


@dataclass(frozen=True, slots=True)
class _CablePickFactory(_CableFactory):
    call_id: ClassVar[str] = _CablePickLowerer.call_id
    target_descriptor: ClassVar[SkillDescriptor] = PickUp.descriptor()
    lowerer_type: ClassVar[type[_CableLowerer]] = _CablePickLowerer


@dataclass(frozen=True, slots=True)
class _CableMoveFactory(_CableFactory):
    call_id: ClassVar[str] = _CableMoveLowerer.call_id
    target_descriptor: ClassVar[SkillDescriptor] = MoveEndEffector.descriptor()
    lowerer_type: ClassVar[type[_CableLowerer]] = _CableMoveLowerer


@dataclass(frozen=True, slots=True)
class _CableJointFactory(_CableFactory):
    call_id: ClassVar[str] = _CableJointLowerer.call_id
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()
    lowerer_type: ClassVar[type[_CableLowerer]] = _CableJointLowerer


@dataclass(frozen=True, slots=True)
class _CableCheckFactory(_CableFactory):
    call_id: ClassVar[str] = _CableCheckLowerer.call_id
    target_descriptor: ClassVar[SkillDescriptor] = MoveJoints.descriptor()
    lowerer_type: ClassVar[type[_CableLowerer]] = _CableCheckLowerer


def _decode_cable_service(value: object) -> _CableService:
    from ..configured import _mapping, _identifier, _real, _integer, _boolean, _sequence

    config = _mapping(
        value,
        path="cable_material",
        required=frozenset(
            {"entity_id", "simulation_uid", "motion_parts", "geometry", "targets"}
        ),
        optional=frozenset({"slip_limit"}),
    )
    targets = []
    fields = {
        field
        for field in _CableTarget.__dataclass_fields__
        if field not in {"target_id", "kind"}
    }
    for name, item in _mapping(
        config["targets"],
        path="cable_material.targets",
        required=frozenset(),
        optional=frozenset(config["targets"]),
    ).items():
        path = f"cable_material.targets.{name}"
        row = _mapping(
            item, path=path, required=frozenset({"kind"}), optional=frozenset(fields)
        )
        kind = _identifier(row["kind"], path=path)
        motion_fields = {"speed", "tolerance", "orientation_tolerance"}
        allowed = {
            "material": motion_fields
            | {"material_distance", "position", "port_screen", "firm"},
            "measured": motion_fields | {"position", "stop_when_seated"},
            "command": motion_fields | {"position", "stop_when_seated"},
            "home": motion_fields,
            "port_stage": motion_fields | {"position"},
            "yaw_align": motion_fields | {"material_distance"},
            "ray_align": motion_fields | {"material_distance"},
            "tip_align": motion_fields
            | {
                "position",
                "maximum_correction",
                "maximum_z_correction",
                "minimum_z",
                "maximum_y",
                "stop_when_seated",
            },
            "hold": {"samples"},
            "open": {"samples", "finger_position", "release_part"},
            "check": {
                "condition",
                "tolerance",
                "orientation_tolerance",
                "stop_when_seated",
            },
        }
        if kind in allowed and set(row) - {"kind"} - allowed[kind]:
            raise ValueError(f"Unsupported fields for cable target kind {kind!r}.")
        if kind not in {
            "material",
            "measured",
            "command",
            "home",
            "port_stage",
            "yaw_align",
            "ray_align",
            "tip_align",
            "hold",
            "open",
            "check",
        }:
            raise ValueError(f"Unsupported cable target kind {kind!r}")
        kwargs = {"target_id": _identifier(name, path=path), "kind": kind}
        for key, val in row.items():
            if key == "kind":
                continue
            if key == "position":
                values = _sequence(val, path=path)
                if len(values) != 3:
                    raise ValueError("Cable position needs exactly three axes.")
                kwargs[key] = tuple(
                    None if v is None else _real(v, path=path) for v in values
                )
            elif key in {"port_screen", "firm", "stop_when_seated"}:
                kwargs[key] = _boolean(val, path=path)
            elif key == "samples":
                kwargs[key] = _integer(val, path=path, minimum=1)
            elif key in {"condition", "release_part"}:
                kwargs[key] = _identifier(val, path=path)
            else:
                kwargs[key] = _real(val, path=path)
        target = _CableTarget(**kwargs)
        if (
            target.speed <= 0
            or target.tolerance <= 0
            or target.orientation_tolerance <= 0
            or target.maximum_correction <= 0
        ):
            raise ValueError("Cable motion limits must be positive.")
        if kind in {"material", "yaw_align", "ray_align"} and (
            target.material_distance is None or target.material_distance < 0
        ):
            raise ValueError("A cable material coordinate must be nonnegative.")
        if kind == "open" and target.release_part is None:
            raise ValueError("An opening target requires its explicit release_part.")
        if kind == "check" and target.condition not in {"pose", "lift", "seated"}:
            raise ValueError("Unknown cable check condition.")
        targets.append(target)
    geometry = _mapping(
        config["geometry"],
        path="cable_material.geometry",
        required=frozenset(
            {
                "bench_height",
                "port_center",
                "port_stage",
                "port_mouth",
                "seats",
                "cylinders",
                "boxes",
                "strict_region",
                "release_region",
            }
        ),
    )

    def freeze(value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return _real(value, path="cable_material.geometry")

    frozen_geometry = {key: freeze(value) for key, value in geometry.items()}
    expected_widths = {"port_center": 2, "port_stage": 3, "port_mouth": 2}
    for key, width in expected_widths.items():
        if (
            not isinstance(frozen_geometry[key], tuple)
            or len(frozen_geometry[key]) != width
        ):
            raise ValueError(f"Cable geometry {key!r} requires {width} coordinates.")
    for key in ("strict_region", "release_region"):
        bounds = frozen_geometry[key]
        if (
            not isinstance(bounds, tuple)
            or len(bounds) != 2
            or any(not isinstance(row, tuple) or len(row) != 3 for row in bounds)
            or any(lo > hi for lo, hi in zip(*bounds))
        ):
            raise ValueError(f"Cable geometry {key!r} requires ordered XYZ bounds.")
    for key, width in (("seats", 3), ("cylinders", 5)):
        if not isinstance(frozen_geometry[key], tuple) or any(
            not isinstance(row, tuple) or len(row) != width
            for row in frozen_geometry[key]
        ):
            raise ValueError(f"Cable geometry {key!r} has invalid rows.")
    if any(
        radius <= 0 or z0 >= z1 for _, _, radius, z0, z1 in frozen_geometry["cylinders"]
    ):
        raise ValueError("Cable obstacle cylinders require positive radii and heights.")
    if len(frozen_geometry["seats"]) != 2:
        raise ValueError(
            "The two-seat cable route measurements require exactly two seats."
        )
    if not isinstance(frozen_geometry["boxes"], tuple) or any(
        not isinstance(row, tuple)
        or len(row) != 2
        or any(not isinstance(vector, tuple) or len(vector) != 3 for vector in row)
        or any(v <= 0 for v in row[1])
        for row in frozen_geometry["boxes"]
    ):
        raise ValueError(
            "Cable obstacle boxes require centres and positive half sizes."
        )
    parts = tuple(
        _identifier(part, path="cable_material.motion_parts")
        for part in _sequence(
            config["motion_parts"], path="cable_material.motion_parts"
        )
    )
    if not parts or len(set(parts)) != len(parts):
        raise ValueError("Cable motion parts must be nonempty and unique.")
    return _CableService(
        entity_id=_identifier(config["entity_id"], path="cable_material.entity_id"),
        simulation_uid=_identifier(
            config["simulation_uid"], path="cable_material.simulation_uid"
        ),
        targets=tuple(targets),
        geometry=tuple(frozen_geometry.items()),
        motion_parts=parts,
        slip_limit=_real(
            config.get("slip_limit", 0.012),
            path="cable_material.slip_limit",
            minimum=0.0,
            strict_minimum=True,
        ),
    )
