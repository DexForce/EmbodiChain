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

"""Resolve symbolic bindings from live simulator state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions import (
    CoordinatedPickmentTarget,
    CoordinatedPlacementTarget,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectState,
    HeldObjectPoseTarget,
    JointPositionTarget,
    ObjectSemantics,
    WorldState,
)
from embodichain.utils.math import pose_inv

from .models import ExecutionProgram, GroundedAction, SemanticStep
from .motion_policy import resolve_motion_policy

__all__ = ["ActionGrounder", "LiveArrangementPlan", "LivePlacementPlan"]


def _batched_pose(value: Any, env: Any) -> torch.Tensor:
    pose = torch.as_tensor(value, dtype=torch.float32, device=env.device)
    if pose.shape == (4, 4):
        pose = pose.unsqueeze(0).repeat(int(env.num_envs), 1, 1)
    if pose.shape != (int(env.num_envs), 4, 4):
        raise ValueError(
            "Live pose must have shape (4, 4) or "
            f"({int(env.num_envs)}, 4, 4), got {tuple(pose.shape)}."
        )
    return pose


def _object(env: Any, uid: str) -> Any:
    entity = env.sim.get_rigid_object(uid)
    if entity is None:
        raise ValueError(f"Unknown rigid object {uid!r}.")
    return entity


def _live_pose(env: Any, uid: str) -> torch.Tensor:
    return _batched_pose(_object(env, uid).get_local_pose(to_matrix=True), env)


def _local_vertices(entity: Any, env: Any, env_id: int = 0) -> torch.Tensor:
    value = entity.get_vertices(env_ids=[env_id], scale=True)
    if isinstance(value, (list, tuple)):
        value = value[0]
    vertices = torch.as_tensor(value, dtype=torch.float32, device=env.device)
    if vertices.ndim == 3 and vertices.shape[0] == 1:
        vertices = vertices[0]
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("Rigid-object mesh vertices must have shape (N, 3).")
    return vertices


def _world_vertices(entity: Any, env: Any, env_id: int) -> torch.Tensor:
    vertices = _local_vertices(entity, env, env_id)
    pose = _batched_pose(entity.get_local_pose(to_matrix=True), env)[env_id]
    return vertices @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]


@dataclass(frozen=True)
class _Geometry:
    radius: torch.Tensor
    half_height: torch.Tensor


class LiveArrangementPlan:
    """Materialize collision-aware line slots independently in every env."""

    def __init__(
        self,
        env: Any,
        steps: Sequence[SemanticStep],
        *,
        slot_margin: float = 0.08,
        minimum_spacing: float = 0.07,
        clearance: float = 0.025,
    ) -> None:
        if not steps:
            raise ValueError("An arrangement plan requires at least one step.")
        self.env = env
        self.steps = tuple(steps)
        self.step_by_id = {step.id: step for step in steps}
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.slot_count = len(steps)
        self.axis = str(steps[0].goal.get("axis", "world_x"))
        self.clearance = float(clearance)

        table = _object(env, "table")
        bounds = []
        for env_id in range(self.num_envs):
            vertices = _world_vertices(table, env, env_id)
            bounds.append(
                torch.stack((vertices.min(dim=0).values, vertices.max(dim=0).values))
            )
        self.table_bounds = torch.stack(bounds)
        self.table_center = self.table_bounds.mean(dim=1)
        self.table_top = self.table_bounds[:, 1, 2]
        if self.axis == "table_long_axis":
            mean_extent = (
                self.table_bounds[:, 1, :2] - self.table_bounds[:, 0, :2]
            ).mean(dim=0)
            self.axis_index = int(torch.argmax(mean_extent).item())
        else:
            self.axis_index = 0 if self.axis in {"x", "world_x"} else 1
        self.perpendicular_index = 1 - self.axis_index
        self.geometry = {
            step.id: self._geometry(step.object_uid) for step in self.steps
        }
        diameters = torch.stack(
            [self.geometry[step.id].radius * 2.0 for step in self.steps],
            dim=1,
        )
        self.spacing = torch.maximum(
            diameters.max(dim=1).values + float(slot_margin),
            torch.full(
                (self.num_envs,),
                float(minimum_spacing),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        self.positions = self._make_slots()
        self.assignments = {
            step.id: torch.full(
                (self.num_envs,),
                int(step.goal.get("nominal_slot_index", index)),
                dtype=torch.long,
                device=self.device,
            )
            for index, step in enumerate(self.steps)
        }
        order_by = str(self.steps[0].goal.get("order_by", "explicit"))
        direction = str(self.steps[0].goal.get("order_direction", "given"))
        if order_by == "size":
            for env_id in range(self.num_envs):
                ordered = sorted(
                    self.steps,
                    key=lambda step: float(self.geometry[step.id].radius[env_id]),
                    reverse=direction != "ascending",
                )
                for slot_id, step in enumerate(ordered):
                    self.assignments[step.id][env_id] = slot_id
        self.completed = {
            step.id: torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            for step in self.steps
        }

    def _geometry(self, uid: str) -> _Geometry:
        entity = _object(self.env, uid)
        radii = []
        heights = []
        for env_id in range(self.num_envs):
            vertices = _local_vertices(entity, self.env, env_id)
            half_extent = (
                vertices.max(dim=0).values - vertices.min(dim=0).values
            ) * 0.5
            # Use the two largest local extents so line slots remain safe even
            # when an orientation policy lays a tall object onto its side.
            radii.append(torch.linalg.vector_norm(torch.topk(half_extent, k=2).values))
            heights.append((vertices[:, 2].max() - vertices[:, 2].min()) * 0.5)
        return _Geometry(torch.stack(radii), torch.stack(heights))

    def _make_slots(self) -> torch.Tensor:
        offsets = (
            torch.arange(self.slot_count, device=self.device, dtype=torch.float32)
            - (self.slot_count - 1) / 2.0
        )
        slots = torch.empty(
            self.num_envs,
            self.slot_count,
            3,
            dtype=torch.float32,
            device=self.device,
        )
        radii = torch.stack(
            [self.geometry[step.id].radius for step in self.steps],
            dim=1,
        )
        # Free slot rematching allows any remaining object to occupy any slot.
        # Size every slot for the largest member in that environment rather
        # than accidentally baking the nominal object order into geometry.
        slot_radii = radii.max(dim=1).values[:, None].repeat(1, self.slot_count)
        obstacles = self._obstacle_bounds()
        search_offsets = [0.0]
        for index in range(1, 11):
            search_offsets.extend((0.025 * index, -0.025 * index))
        for env_id in range(self.num_envs):
            chosen = None
            for perpendicular in search_offsets:
                candidate = self.table_center[env_id].repeat(self.slot_count, 1)
                candidate[:, self.axis_index] += self.spacing[env_id] * offsets
                candidate[:, self.perpendicular_index] += perpendicular
                candidate[:, 2] = self.table_top[env_id]
                if self._safe(
                    candidate,
                    slot_radii[env_id],
                    self.table_bounds[env_id],
                    obstacles[env_id],
                ):
                    chosen = candidate
                    break
            if chosen is None:
                raise ValueError(
                    f"Environment {env_id} has no collision-free arrangement row."
                )
            slots[env_id] = chosen
        return slots

    def _obstacle_bounds(
        self,
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        result: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_envs)
        ]
        getter = getattr(self.env.sim, "get_rigid_object_uid_list", None)
        if not callable(getter):
            return result
        movable = {step.object_uid for step in self.steps}
        for uid in getter():
            if uid == "table" or uid in movable:
                continue
            entity = self.env.sim.get_rigid_object(uid)
            if entity is None:
                continue
            for env_id in range(self.num_envs):
                vertices = _world_vertices(entity, self.env, env_id)
                if float(vertices[:, 2].max()) < float(
                    self.table_top[env_id] - self.clearance
                ):
                    continue
                result[env_id].append(
                    (
                        vertices[:, :2].min(dim=0).values,
                        vertices[:, :2].max(dim=0).values,
                    )
                )
        return result

    def _safe(
        self,
        slots: torch.Tensor,
        radii: torch.Tensor,
        table_bounds: torch.Tensor,
        obstacles: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        lower = table_bounds[0, :2] + radii[:, None] + self.clearance
        upper = table_bounds[1, :2] - radii[:, None] - self.clearance
        if bool(((slots[:, :2] < lower) | (slots[:, :2] > upper)).any()):
            return False
        for center, radius in zip(slots[:, :2], radii):
            for obstacle_lower, obstacle_upper in obstacles:
                closest = torch.maximum(
                    obstacle_lower,
                    torch.minimum(center, obstacle_upper),
                )
                if float(torch.linalg.vector_norm(center - closest)) <= float(
                    radius + self.clearance
                ):
                    return False
        return True

    def target(
        self,
        step: SemanticStep,
        object_pose: torch.Tensor,
        *,
        phase: str,
        policy: Mapping[str, Any],
    ) -> torch.Tensor:
        """Return a live final or collision-clear staging object pose."""
        if phase not in {"staging", "final"}:
            raise ValueError(f"Unsupported arrangement phase {phase!r}.")
        target = object_pose.clone()
        env_ids = torch.arange(self.num_envs, device=self.device)
        slot_ids = self.assignments[step.id]
        target[:, :2, 3] = self.positions[env_ids, slot_ids, :2]
        final_z = (
            self.table_top
            + self.geometry[step.id].half_height
            + float(policy.get("surface_clearance", 0.003))
        )
        target[:, 2, 3] = final_z
        if phase == "staging":
            target[:, 2, 3] = final_z + float(policy.get("transport_clearance", 0.10))
        return target

    def mark_completed(self, step_id: str, success: torch.Tensor) -> None:
        self.completed[step_id] |= success.to(self.device, dtype=torch.bool)

    def remaining(self, env_id: int) -> list[str]:
        return [
            step.id for step in self.steps if not bool(self.completed[step.id][env_id])
        ]

    def available_slots(self, env_id: int) -> list[int]:
        occupied = {
            int(self.assignments[step.id][env_id])
            for step in self.steps
            if bool(self.completed[step.id][env_id])
        }
        return [index for index in range(self.slot_count) if index not in occupied]

    def assign(self, env_id: int, assignment: Mapping[str, int]) -> None:
        for step_id, slot_id in assignment.items():
            self.assignments[step_id][env_id] = int(slot_id)


class LivePlacementPlan:
    """Allocate non-overlapping live slots for one shared container."""

    def __init__(
        self,
        env: Any,
        steps: Sequence[SemanticStep],
        *,
        clearance: float = 0.012,
    ) -> None:
        if not steps:
            raise ValueError("A placement plan requires at least one step.")
        references = {step.goal.get("reference_object") for step in steps}
        if len(references) != 1 or not isinstance(next(iter(references)), str):
            raise ValueError("Placement-plan steps must share one reference object.")
        self.env = env
        self.steps = tuple(steps)
        self.reference_uid = str(next(iter(references)))
        self.num_envs = int(env.num_envs)
        self.clearance = float(clearance)
        self.positions = self._make_slots()

    def _make_slots(self) -> dict[str, torch.Tensor]:
        container = _object(self.env, self.reference_uid)
        positions = {
            step.id: torch.empty(
                self.num_envs,
                3,
                dtype=torch.float32,
                device=self.env.device,
            )
            for step in self.steps
        }
        named_slots = [str(step.goal.get("slot", "auto")) for step in self.steps]
        for slot in named_slots:
            if slot not in {"auto", "left", "center", "right"}:
                raise ValueError(f"Unsupported container slot {slot!r}.")

        for env_id in range(self.num_envs):
            vertices = _world_vertices(container, self.env, env_id)
            lower = vertices.min(dim=0).values
            upper = vertices.max(dim=0).values
            center = (lower + upper) * 0.5
            extent = upper[:2] - lower[:2]
            axis = int(torch.argmax(extent).item())
            radii = []
            for step in self.steps:
                moved_vertices = _local_vertices(
                    _object(self.env, step.object_uid),
                    self.env,
                    env_id,
                )
                half = (
                    moved_vertices.max(dim=0).values - moved_vertices.min(dim=0).values
                )[:2] * 0.5
                radii.append(float(torch.linalg.vector_norm(half)))
            radius = max(radii)
            usable_span = float(extent[axis]) - 2.0 * (radius + self.clearance)
            required_span = 2.0 * radius * max(len(self.steps) - 1, 0)
            if usable_span + 1.0e-6 < required_span:
                raise ValueError(
                    f"Environment {env_id} container {self.reference_uid!r} "
                    "has no non-overlapping slot plan."
                )
            offsets = torch.linspace(
                -required_span * 0.5,
                required_span * 0.5,
                len(self.steps),
                device=self.env.device,
            )
            named_offsets = {
                "left": required_span * 0.5,
                "center": 0.0,
                "right": -required_span * 0.5,
            }
            used: list[float] = []
            for index, step in enumerate(self.steps):
                slot = named_slots[index]
                offset = (
                    float(offsets[index]) if slot == "auto" else named_offsets[slot]
                )
                if any(abs(offset - item) < 2.0 * radius for item in used):
                    raise ValueError(
                        f"Container slot {slot!r} overlaps another requested slot."
                    )
                used.append(offset)
                target = center.clone()
                target[axis] += offset
                target[2] = lower[2]
                positions[step.id][env_id] = target
        return positions

    def target(
        self,
        step: SemanticStep,
        object_pose: torch.Tensor,
        rotation: torch.Tensor,
        *,
        surface_clearance: float,
    ) -> torch.Tensor:
        """Return a slot pose corrected for the rotated object mesh bottom."""
        target = object_pose.clone()
        target[:, :3, :3] = rotation
        target[:, :2, 3] = self.positions[step.id][:, :2]
        entity = _object(self.env, step.object_uid)
        for env_id in range(self.num_envs):
            bottom = (
                _local_vertices(entity, self.env, env_id)
                @ rotation[env_id].transpose(0, 1)
            )[:, 2].min()
            target[env_id, 2, 3] = (
                self.positions[step.id][env_id, 2] + surface_clearance - bottom
            )
        return target


class ActionGrounder:
    """Translate one symbolic action into a public typed atomic-action target."""

    def __init__(
        self,
        program: ExecutionProgram,
        env: Any,
        semantics_factory: Callable[[str], ObjectSemantics],
        arrangement: (
            LiveArrangementPlan | Mapping[str, LiveArrangementPlan] | None
        ) = None,
        placements: Mapping[str, LivePlacementPlan] | None = None,
    ) -> None:
        self.program = program
        self.env = env
        self.semantics_factory = semantics_factory
        self.robot_profile = str(getattr(env, "agent_robot_profile", "dual_ur10"))
        if isinstance(arrangement, Mapping):
            self.arrangements = dict(arrangement)
        elif arrangement is None:
            self.arrangements = {}
        else:
            self.arrangements = {
                step.id: arrangement
                for step in program.semantic_steps
                if step.operator in {"arrange_line", "place_in_line"}
            }
        self.placements = dict(placements or {})

    def policy(self, action: Mapping[str, Any]) -> dict[str, Any]:
        name = str(action.get("motion_policy", "default_transport"))
        inline = action.get("motion_policy_config", action.get("cfg"))
        return resolve_motion_policy(
            self.robot_profile,
            name,
            program_overrides=self.program.motion_policies.get(name),
            inline_overrides=inline if isinstance(inline, Mapping) else None,
        )

    def ground(
        self,
        action: Mapping[str, Any],
        step: SemanticStep,
        *,
        arm: str,
        state: WorldState,
        reference_eef_pose: torch.Tensor | None = None,
    ) -> GroundedAction:
        action_class = str(action["atomic_action_class"])
        control = str(action.get("control", "arm"))
        binding = action.get("target_binding", {})
        if not isinstance(binding, Mapping):
            raise ValueError("target_binding must be a mapping.")
        kind = str(binding.get("kind", ""))
        policy = self.policy(action)
        object_pose = _live_pose(self.env, step.object_uid)
        if step.operator == "orient_object":
            policy["upright_local_axis"] = self._upright_local_axis(step)
            if action_class == "PickUp":
                policy["obj_upright_direction"] = self._upright_local_direction(step)
        reference_pose = self._reference_pose(step)
        target_object_pose = None

        if kind == "object":
            semantics = self.semantics_factory(
                str(binding.get("object", step.object_uid))
            )
            if action_class == "PickUp":
                target: Any = GraspTarget(semantics=semantics)
            elif action_class == "CoordinatedPickment":
                target_object_pose = self._semantic_target(
                    step, object_pose, reference_pose, policy, phase="final"
                )
                left_to_eef, right_to_eef = self._coordinated_grasps(
                    semantics, object_pose
                )
                target = CoordinatedPickmentTarget(
                    object_target_pose=target_object_pose,
                    object_semantics=semantics,
                    left_object_to_eef=left_to_eef,
                    right_object_to_eef=right_to_eef,
                    object_initial_pose=object_pose,
                )
            elif action_class == "Press":
                target_object_pose = object_pose.clone()
                target = EndEffectorPoseTarget(
                    xpos=self._press_pose(
                        arm,
                        step.object_uid,
                        object_pose,
                        policy,
                    )
                )
            else:
                raise ValueError(
                    f"{action_class} does not support object target bindings."
                )
        elif kind in {"semantic_goal", "coordinated_goal"}:
            phase = str(binding.get("phase", "final"))
            target_object_pose = self._semantic_target(
                step, object_pose, reference_pose, policy, phase=phase
            )
            if action_class == "CoordinatedPickment":
                semantics = self.semantics_factory(step.object_uid)
                left_to_eef, right_to_eef = self._coordinated_grasps(
                    semantics, object_pose
                )
                target = CoordinatedPickmentTarget(
                    object_target_pose=target_object_pose,
                    object_semantics=semantics,
                    left_object_to_eef=left_to_eef,
                    right_object_to_eef=right_to_eef,
                    object_initial_pose=object_pose,
                )
            elif action_class == "Press":
                # Press moves the TCP, not the target object. Keep the object's
                # live pose as the postcondition reference while grounding a
                # downward contact point from its current surface geometry.
                target_object_pose = object_pose.clone()
                target = EndEffectorPoseTarget(
                    xpos=self._press_pose(
                        arm,
                        step.object_uid,
                        object_pose,
                        policy,
                    )
                )
            else:
                target = HeldObjectPoseTarget(object_target_pose=target_object_pose)
        elif kind == "coordinated_placement_goal":
            support_uid = binding.get(
                "support_object",
                step.goal.get("support_object"),
            )
            placing_uid = binding.get("placing_object", step.object_uid)
            if not isinstance(placing_uid, str) or not placing_uid:
                raise ValueError("coordinated_placement_goal requires placing_object.")
            if not isinstance(support_uid, str) or not support_uid:
                raise ValueError("coordinated_placement_goal requires support_object.")
            support_pose = _live_pose(self.env, support_uid)
            target_object_pose = self._semantic_target(
                step,
                object_pose,
                support_pose,
                policy,
                phase="final",
            )
            target = CoordinatedPlacementTarget(
                placing_object_target_pose=target_object_pose,
                support_object_target_pose=support_pose,
                placing_held_object=self._held_state_from_live_pose(
                    placing_uid,
                    "left_arm",
                ),
                support_held_object=self._held_state_from_live_pose(
                    support_uid,
                    "right_arm",
                ),
                release=bool(step.goal.get("release", True)),
            )
        elif kind == "current_held_pose":
            if state.held_object is None:
                raise ValueError("Place requires a held object from a prior PickUp.")
            target = EndEffectorPoseTarget(
                xpos=(
                    reference_eef_pose
                    if reference_eef_pose is not None
                    else self._current_eef_pose(arm)
                )
            )
        elif kind == "policy_pose":
            target = EndEffectorPoseTarget(
                xpos=self._retreat_pose(arm, policy, reference_eef_pose)
            )
        elif kind == "joint_state":
            target = JointPositionTarget(
                qpos=self._joint_target(
                    arm,
                    control,
                    str(binding.get("source", "initial")),
                    binding,
                )
            )
        elif kind in {"eef_pose", "pose"}:
            target = EndEffectorPoseTarget(
                xpos=self._explicit_pose(binding, object_pose)
            )
        else:
            raise ValueError(f"Unsupported target binding kind {kind!r}.")
        return GroundedAction(
            action_class=action_class,
            arm=arm,
            control=control,
            target=target,
            cfg=policy,
            object_pose=object_pose,
            reference_pose=reference_pose,
            target_object_pose=target_object_pose,
            motion_policy=policy,
        )

    def _reference_pose(self, step: SemanticStep) -> torch.Tensor | None:
        uid = step.goal.get("reference_object", step.goal.get("support_object"))
        if not isinstance(uid, str) or not uid:
            return None
        if step.goal.get("reference_state") == "initial":
            initial = getattr(self.env, "agent_initial_object_poses", {}).get(uid)
            if initial is None:
                raise ValueError(f"Initial pose for {uid!r} is unavailable.")
            return _batched_pose(initial, self.env)
        return _live_pose(self.env, uid)

    def _semantic_target(
        self,
        step: SemanticStep,
        object_pose: torch.Tensor,
        reference_pose: torch.Tensor | None,
        policy: Mapping[str, Any],
        *,
        phase: str,
    ) -> torch.Tensor:
        if step.operator in {"arrange_line", "place_in_line"}:
            arrangement = self.arrangements.get(step.id)
            if arrangement is None:
                raise ValueError("arrange_line requires a live arrangement plan.")
            target = arrangement.target(
                step,
                object_pose,
                phase=phase,
                policy=policy,
            )
            target[:, :3, :3] = self._target_rotation(step, object_pose)
            moved = _object(self.env, step.object_uid)
            for env_id in range(int(self.env.num_envs)):
                bottom = self._rotated_local_z_min(
                    moved,
                    target[env_id, :3, :3],
                    env_id,
                )
                target[env_id, 2, 3] = (
                    arrangement.table_top[env_id]
                    + float(policy.get("surface_clearance", 0.003))
                    - bottom
                )
            if phase == "staging":
                target[:, 2, 3] += float(policy.get("transport_clearance", 0.10))
            return target
        placement = self.placements.get(step.id)
        if placement is not None:
            target = placement.target(
                step,
                object_pose,
                self._target_rotation(step, object_pose),
                surface_clearance=float(policy.get("surface_clearance", 0.003)),
            )
            if phase == "staging":
                target[:, 2, 3] += float(policy.get("transport_clearance", 0.10))
            return target
        if step.operator == "orient_object":
            initial = None
            if step.goal.get("position_anchor", "initial_xy") == "initial_xy":
                initial = getattr(self.env, "agent_initial_object_poses", {}).get(
                    step.object_uid
                )
            target = (
                _batched_pose(initial, self.env).clone()
                if initial is not None
                else object_pose.clone()
            )
            target[:, :3, :3] = self._target_rotation(step, target)
            support_uid = str(step.goal.get("support_object", "table"))
            support = _object(self.env, support_uid)
            moved = _object(self.env, step.object_uid)
            for env_id in range(int(self.env.num_envs)):
                support_top = _world_vertices(support, self.env, env_id)[:, 2].max()
                bottom = self._rotated_local_z_min(
                    moved,
                    target[env_id, :3, :3],
                    env_id,
                )
                target[env_id, 2, 3] = (
                    support_top + float(policy.get("surface_clearance", 0.003)) - bottom
                )
            if phase == "staging":
                target[:, 2, 3] += float(policy.get("staging_lift_height", 0.12))
            return target
        target = object_pose.clone()
        if reference_pose is not None:
            target[:, :3, 3] = reference_pose[:, :3, 3]
        # Operators without a relational goal (for example press or a
        # direction-only coordinated transport) must preserve the live origin
        # instead of being silently projected onto a synthetic table support.
        relation = str(step.goal.get("relation", "none"))
        distance = float(policy.get("relation_distance", 0.16))
        offsets = {
            "left": (0.0, distance, 0.0),
            "left_of": (0.0, distance, 0.0),
            "right": (0.0, -distance, 0.0),
            "right_of": (0.0, -distance, 0.0),
            "front": (distance, 0.0, 0.0),
            "front_of": (distance, 0.0, 0.0),
            "in_front_of": (distance, 0.0, 0.0),
            "behind": (-distance, 0.0, 0.0),
            "back": (-distance, 0.0, 0.0),
            "front_left": (distance, distance, 0.0),
            "front_left_of": (distance, distance, 0.0),
            "front_right": (distance, -distance, 0.0),
            "front_right_of": (distance, -distance, 0.0),
            "back_left": (-distance, distance, 0.0),
            "back_left_of": (-distance, distance, 0.0),
            "back_right": (-distance, -distance, 0.0),
            "back_right_of": (-distance, -distance, 0.0),
            "above": (0.0, 0.0, float(policy.get("hover_height", 0.10))),
            "held_above_initial": (
                0.0,
                0.0,
                float(policy.get("hover_height", 0.10)),
            ),
        }
        offset = offsets.get(relation)
        if offset is not None:
            target[:, :3, 3] += torch.tensor(
                offset,
                dtype=target.dtype,
                device=target.device,
            )
        direction = str(step.goal.get("direction", "none"))
        direction_offsets = {
            "world_x": (distance, 0.0, 0.0),
            "world_y": (0.0, distance, 0.0),
            "left": (0.0, distance, 0.0),
            "right": (0.0, -distance, 0.0),
            "front": (distance, 0.0, 0.0),
            "back": (-distance, 0.0, 0.0),
            "front_left": (distance, distance, 0.0),
            "front_right": (distance, -distance, 0.0),
            "back_left": (-distance, distance, 0.0),
            "back_right": (-distance, -distance, 0.0),
            "up": (0.0, 0.0, distance),
            "down": (0.0, 0.0, -distance),
        }
        if direction in direction_offsets:
            target[:, :3, 3] += torch.tensor(
                direction_offsets[direction],
                dtype=target.dtype,
                device=target.device,
            )

        root_stack_layer = (
            step.operator == "build_stack"
            and int(step.goal.get("layer_index", 0)) == 0
            and reference_pose is None
        )
        if root_stack_layer:
            table = _object(self.env, "table")
            for env_id in range(int(self.env.num_envs)):
                vertices = _world_vertices(table, self.env, env_id)
                target[env_id, :2, 3] = (
                    vertices[:, :2].min(dim=0).values
                    + vertices[:, :2].max(dim=0).values
                ) * 0.5

        target[:, :3, :3] = self._target_rotation(step, object_pose)
        if relation in {"on", "on_top", "on_top_of"} or root_stack_layer:
            support_uid = (
                step.goal.get("reference_object")
                or step.goal.get("support_object")
                or "table"
            )
            support = _object(self.env, str(support_uid))
            moved = _object(self.env, step.object_uid)
            for env_id in range(int(self.env.num_envs)):
                support_top = _world_vertices(support, self.env, env_id)[:, 2].max()
                bottom = self._rotated_local_z_min(
                    moved,
                    target[env_id, :3, :3],
                    env_id,
                )
                target[env_id, 2, 3] = (
                    support_top + float(policy.get("surface_clearance", 0.003)) - bottom
                )
        elif relation == "inside" and reference_pose is not None:
            # Preserve the object's live height while centering it in container XY.
            target[:, 2, 3] = object_pose[:, 2, 3]
        if phase == "staging":
            # Staging is a runtime waypoint, not a persisted coordinate. This
            # keeps in-place orientation robust to the object's live height.
            target[:, 2, 3] += float(policy.get("transport_clearance", 0.10))
        return target

    def _upright_local_direction(self, step: SemanticStep) -> torch.Tensor:
        axis = self._upright_local_axis(step)
        entity = _object(self.env, step.object_uid)
        vertices = _local_vertices(entity, self.env, 0)
        extents = vertices.max(dim=0).values - vertices.min(dim=0).values
        if axis == "long_axis":
            axis_index = int(torch.argmax(extents).item())
        else:
            axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        direction = torch.zeros(3, dtype=torch.float32, device=self.env.device)
        direction[axis_index] = 1.0
        return direction

    @staticmethod
    def _upright_local_axis(step: SemanticStep) -> str:
        axis = str(step.goal.get("upright_local_axis", "auto"))
        return "long_axis" if axis == "auto" else axis

    def _target_rotation(
        self,
        step: SemanticStep,
        object_pose: torch.Tensor,
    ) -> torch.Tensor:
        goal = str(step.goal.get("orientation_goal", "preserve"))
        if goal == "preserve":
            return object_pose[:, :3, :3].clone()
        if goal not in {"upright", "lay_flat", "axis_align"}:
            raise ValueError(f"Unsupported orientation_goal {goal!r}.")

        entity = _object(self.env, step.object_uid)
        rotations = []
        for env_id in range(int(self.env.num_envs)):
            vertices = _local_vertices(entity, self.env, env_id)
            extents = vertices.max(dim=0).values - vertices.min(dim=0).values
            longest_to_shortest = torch.argsort(
                extents,
                descending=True,
            ).tolist()
            if goal == "upright":
                upright_axis = self._upright_local_axis(step)
                vertical_axis = (
                    int(longest_to_shortest[0])
                    if upright_axis == "long_axis"
                    else {"x": 0, "y": 1, "z": 2}[upright_axis]
                )
                horizontal_axis = next(
                    int(axis)
                    for axis in longest_to_shortest
                    if int(axis) != vertical_axis
                )
            elif goal == "lay_flat":
                vertical_axis = int(longest_to_shortest[-1])
                horizontal_axis = int(longest_to_shortest[0])
            else:
                horizontal_axis = self._aligned_local_axis(
                    step,
                    longest_to_shortest,
                )
                vertical_axis = next(
                    int(axis)
                    for axis in reversed(longest_to_shortest)
                    if int(axis) != horizontal_axis
                )
            direction = self._horizontal_orientation(
                step,
                object_pose,
                env_id,
                horizontal_axis,
            )
            rotations.append(
                self._world_aligned_rotation(
                    direction,
                    horizontal_axis=horizontal_axis,
                    vertical_axis=vertical_axis,
                )
            )
        return torch.stack(rotations)

    @staticmethod
    def _aligned_local_axis(
        step: SemanticStep,
        longest_to_shortest: Sequence[int],
    ) -> int:
        axis = str(step.goal.get("orientation_axis", "long_axis"))
        if axis == "x":
            return 0
        if axis == "y":
            return 1
        if axis == "long_axis":
            return int(longest_to_shortest[0])
        if axis == "short_axis":
            return int(longest_to_shortest[-1])
        raise ValueError(f"Unsupported axis_align orientation_axis {axis!r}.")

    def _horizontal_orientation(
        self,
        step: SemanticStep,
        object_pose: torch.Tensor,
        env_id: int,
        local_axis: int,
    ) -> torch.Tensor:
        align_to = step.goal.get("orientation_reference_object")
        if isinstance(align_to, str) and align_to:
            reference = _object(self.env, align_to)
            vertices = _local_vertices(reference, self.env, env_id)
            extents = vertices.max(dim=0).values - vertices.min(dim=0).values
            ordered = torch.argsort(extents, descending=True)
            requested = str(step.goal.get("orientation_axis", "long_axis"))
            reference_axis = int(
                ordered[-1] if requested == "short_axis" else ordered[0]
            )
            reference_pose = _live_pose(self.env, align_to)
            direction = reference_pose[env_id, :3, reference_axis].clone()
        elif step.operator in {"arrange_line", "place_in_line"}:
            arrangement = self.arrangements.get(step.id)
            axis_index = 0 if arrangement is None else arrangement.axis_index
            direction = torch.zeros(
                3,
                dtype=object_pose.dtype,
                device=object_pose.device,
            )
            direction[axis_index] = 1.0
        elif str(step.goal.get("orientation_axis", "")) in {"y", "world_y"}:
            direction = object_pose.new_tensor([0.0, 1.0, 0.0])
        elif str(step.goal.get("orientation_axis", "")) in {"x", "world_x"}:
            direction = object_pose.new_tensor([1.0, 0.0, 0.0])
        else:
            direction = object_pose[env_id, :3, local_axis].clone()
        direction[2] = 0.0
        norm = torch.linalg.vector_norm(direction)
        if float(norm) < 1.0e-6:
            return object_pose.new_tensor([1.0, 0.0, 0.0])
        return direction / norm

    @staticmethod
    def _world_aligned_rotation(
        horizontal_direction: torch.Tensor,
        *,
        horizontal_axis: int,
        vertical_axis: int,
    ) -> torch.Tensor:
        world_up = horizontal_direction.new_tensor([0.0, 0.0, 1.0])
        remaining_axis = ({0, 1, 2} - {horizontal_axis, vertical_axis}).pop()
        columns = [torch.zeros_like(world_up) for _ in range(3)]
        columns[horizontal_axis] = horizontal_direction
        columns[vertical_axis] = world_up
        columns[remaining_axis] = torch.linalg.cross(
            world_up,
            horizontal_direction,
        )
        rotation = torch.stack(columns, dim=1)
        if float(torch.linalg.det(rotation)) < 0.0:
            rotation[:, remaining_axis] *= -1.0
        return rotation

    def _rotated_local_z_min(
        self,
        entity: Any,
        rotation: torch.Tensor,
        env_id: int,
    ) -> torch.Tensor:
        vertices = _local_vertices(entity, self.env, env_id)
        return (vertices @ rotation.transpose(0, 1))[:, 2].min()

    def _current_eef_pose(self, arm: str) -> torch.Tensor:
        """Return the live TCP pose for one logical Action Engine arm."""
        if arm not in {"left_arm", "right_arm"}:
            raise ValueError(f"Expected a physical arm, got {arm!r}.")
        if hasattr(self.env, "get_current_xpos_agent"):
            left, right = self.env.get_current_xpos_agent()
            value = left if arm == "left_arm" else right
            if value is not None:
                return _batched_pose(value, self.env)

        is_left = arm == "left_arm"
        if not hasattr(self.env, "get_agent_arm_control_part"):
            raise ValueError("Coordinated placement requires live TCP poses.")
        part = self.env.get_agent_arm_control_part(is_left)
        qpos = self._arm_qpos(arm)
        return _batched_pose(
            self.env.robot.compute_fk(qpos=qpos, name=part, to_matrix=True),
            self.env,
        )

    def _press_pose(
        self,
        arm: str,
        uid: str,
        object_pose: torch.Tensor,
        policy: Mapping[str, Any],
    ) -> torch.Tensor:
        """Ground a top-surface contact while retaining the live TCP rotation."""
        target = self._current_eef_pose(arm).clone()
        target[:, :2, 3] = object_pose[:, :2, 3]
        entity = _object(self.env, uid)
        depth = float(policy.get("press_depth", 0.004))
        for env_id in range(int(self.env.num_envs)):
            top = _world_vertices(entity, self.env, env_id)[:, 2].max()
            target[env_id, 2, 3] = top - depth
        return target

    def _held_state_from_live_pose(
        self,
        uid: str,
        arm: str,
    ) -> HeldObjectState:
        """Describe the live object-to-TCP relationship for coordinated placement.

        ``CoordinatedPlacement`` receives both held-object states in its typed
        target. Keeping this computation in the grounder ensures that no pose
        from the symbolic program is trusted as runtime geometry.
        """
        object_pose = _live_pose(self.env, uid)
        eef_pose = self._current_eef_pose(arm)
        return HeldObjectState(
            semantics=self.semantics_factory(uid),
            object_to_eef=torch.bmm(pose_inv(object_pose), eef_pose),
            grasp_xpos=eef_pose,
        )

    def _retreat_pose(
        self,
        arm: str,
        policy: Mapping[str, Any],
        reference: torch.Tensor | None,
    ) -> torch.Tensor:
        pose = reference
        if pose is None and hasattr(self.env, "get_current_xpos_agent"):
            left, right = self.env.get_current_xpos_agent()
            pose = left if arm == "left_arm" else right
        if pose is None:
            raise ValueError("Retreat grounding requires a live end-effector pose.")
        target = _batched_pose(pose, self.env).clone()
        desired = float(policy.get("retreat_height", 0.10))
        ceiling = float(policy.get("maximum_eef_height", 0.80))
        height = torch.clamp(ceiling - target[:, 2, 3], min=0.0, max=desired)
        target[:, 2, 3] += height
        return target

    def _joint_target(
        self,
        arm: str,
        control: str,
        source: str,
        binding: Mapping[str, Any],
    ) -> torch.Tensor:
        if source in {"gripper_closed", "gripper_open"}:
            value = (
                getattr(self.env, "close_state")
                if source == "gripper_closed"
                else getattr(self.env, "open_state")
            )
            return torch.as_tensor(
                value,
                dtype=torch.float32,
                device=self.env.device,
            )
        if source == "joint_delta":
            current = self._arm_qpos(arm).clone()
            index = int(binding["joint_index"])
            current[:, index] += torch.deg2rad(
                torch.tensor(
                    float(binding.get("delta_degrees", 0.0)),
                    device=current.device,
                )
            )
            return current
        initial = getattr(self.env, "init_qpos", self.env.robot.get_qpos())
        joint_ids = self._joint_ids(arm, control)
        return torch.as_tensor(initial, device=self.env.device)[:, joint_ids]

    def _arm_qpos(self, arm: str) -> torch.Tensor:
        if hasattr(self.env, "get_current_qpos_agent"):
            left, right = self.env.get_current_qpos_agent()
            return torch.as_tensor(
                left if arm == "left_arm" else right,
                dtype=torch.float32,
                device=self.env.device,
            )
        return self.env.robot.get_qpos()[:, self._joint_ids(arm, "arm")]

    def _joint_ids(self, arm: str, control: str) -> list[int]:
        side = "left" if arm == "left_arm" else "right"
        key = f"{side}_{'eef' if control == 'hand' else 'arm'}_joints"
        return list(getattr(self.env, key, ()))

    def _explicit_pose(
        self,
        binding: Mapping[str, Any],
        object_pose: torch.Tensor,
    ) -> torch.Tensor:
        reference = str(binding.get("reference", "absolute"))
        target = object_pose.clone()
        if reference == "absolute":
            values = binding.get("position_by_env", binding.get("position"))
            position = torch.as_tensor(
                values,
                dtype=target.dtype,
                device=target.device,
            )
            if position.ndim == 1:
                position = position.unsqueeze(0).repeat(int(self.env.num_envs), 1)
            target[:, :3, 3] = position
            return target
        offset = torch.as_tensor(
            binding.get("offset", (0.0, 0.0, 0.0)),
            dtype=target.dtype,
            device=target.device,
        )
        target[:, :3, 3] += offset
        return target

    def _coordinated_grasps(
        self,
        semantics: ObjectSemantics,
        object_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a deterministic opposing pair along the object's longest XY axis."""
        vertices = semantics.geometry.get("mesh_vertices")
        vertices = torch.as_tensor(
            vertices,
            dtype=torch.float32,
            device=self.env.device,
        )
        lower = vertices.min(dim=0).values
        upper = vertices.max(dim=0).values
        axis = int(torch.argmax(upper[:2] - lower[:2]).item())
        center = (lower + upper) * 0.5
        inset = max(0.01, float((upper[axis] - lower[axis]) * 0.15))
        left = torch.eye(4, dtype=torch.float32, device=self.env.device)
        right = left.clone()
        left[:3, 3] = center
        right[:3, 3] = center
        left[axis, 3] = lower[axis] + inset
        right[axis, 3] = upper[axis] - inset
        # Keep TCP z horizontal and facing the object from opposite sides.
        if axis == 0:
            left[:3, :3] = torch.tensor(
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                device=self.env.device,
            )
            right[:3, :3] = torch.tensor(
                [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                device=self.env.device,
            )
        else:
            left[:3, :3] = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
                device=self.env.device,
            )
            right[:3, :3] = torch.tensor(
                [[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
                device=self.env.device,
            )
        batch = int(self.env.num_envs)
        return left.unsqueeze(0).repeat(batch, 1, 1), right.unsqueeze(0).repeat(
            batch, 1, 1
        )
