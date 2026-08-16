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
from dataclasses import dataclass, replace
from typing import Any

import torch

from embodichain.gen_sim.action_engine.capabilities import (
    build_atomic_capability_registry,
)
from embodichain.gen_sim.action_engine.config import (
    RuntimePolicyCfg,
    default_runtime_policy,
)
from embodichain.gen_sim.action_engine.domain import normalize_placement_relation
from embodichain.gen_sim.action_engine.orientation import (
    AlignAxisConstraint,
    MatchRotationConstraint,
    OrientationConstraint,
    compile_orientation_constraint,
)
from embodichain.lab.sim.atomic_actions import (
    CoordinatedPickGoal,
    CoordinatedPlacementGoal,
    EndEffectorPoseGoal,
    GraspGoal,
    HeldObjectPoseGoal,
    JointPositionGoal,
    ObjectSemantics,
    PlaceGoal,
    PressGoal,
)
from .frames import arm_base_poses, relation_offset, robot_frame_axes
from .models import ExecutionProgram, GroundedAction, SemanticStep
from .motion_policy import resolve_motion_policy, with_motion_modifiers
from .robot_parts import arm_control_part
from .state import ExecutionState

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
        slot_margin: float | None = None,
        minimum_spacing: float | None = None,
        clearance: float | None = None,
        row_search_step: float | None = None,
        row_search_radius: float | None = None,
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
        profile = str(getattr(env, "agent_robot_profile", "dual_ur10"))
        defaults = default_runtime_policy(profile).grounding["arrangement"]
        slot_margin = defaults["slot_margin"] if slot_margin is None else slot_margin
        minimum_spacing = (
            defaults["minimum_spacing"] if minimum_spacing is None else minimum_spacing
        )
        self.clearance = float(
            defaults["layout_clearance"] if clearance is None else clearance
        )
        self.row_search_step = float(
            defaults["row_search_step"] if row_search_step is None else row_search_step
        )
        self.row_search_radius = float(
            defaults["row_search_radius"]
            if row_search_radius is None
            else row_search_radius
        )

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
        self.geometry = {step.id: self._geometry(step) for step in self.steps}
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
        self.reassignment_reason: list[str | None] = [None] * self.num_envs
        self.reassignment_cost = torch.full(
            (self.num_envs,),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.assignments = self._initial_slot_assignments()
        order_by = str(self.steps[0].goal.get("order_by", "explicit"))
        direction = str(self.steps[0].goal.get("order_direction", "given"))
        if order_by == "size" and not any(
            step.goal.get("slot_constraint") == "free_reassignable"
            for step in self.steps
        ):
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

    def _initial_slot_assignments(self) -> dict[str, torch.Tensor]:
        """Match free-order objects to slots in their current spatial order."""
        assignments = {
            step.id: torch.full(
                (self.num_envs,),
                int(step.goal.get("nominal_slot_index", index)),
                dtype=torch.long,
                device=self.device,
            )
            for index, step in enumerate(self.steps)
        }
        free_steps = [
            step
            for step in self.steps
            if step.goal.get("slot_constraint") == "free_reassignable"
        ]
        if not free_steps:
            return assignments
        required_slots = {
            int(step.goal.get("nominal_slot_index", index))
            for index, step in enumerate(self.steps)
            if step.goal.get("slot_constraint") != "free_reassignable"
        }
        available_slots = [
            slot_id
            for slot_id in range(self.slot_count)
            if slot_id not in required_slots
        ]
        if len(available_slots) != len(free_steps):
            raise ValueError(
                "Arrangement slot constraints do not define a one-to-one assignment."
            )
        axis_positions = {
            step.id: _live_pose(self.env, step.object_uid)[:, self.axis_index, 3]
            for step in free_steps
        }
        for env_id in range(self.num_envs):
            ordered_steps = sorted(
                free_steps,
                key=lambda step: (
                    float(axis_positions[step.id][env_id]),
                    int(step.goal.get("nominal_slot_index", 0)),
                    step.id,
                ),
            )
            ordered_slots = sorted(
                available_slots,
                key=lambda slot_id: (
                    float(self.positions[env_id, slot_id, self.axis_index]),
                    slot_id,
                ),
            )
            matching_cost = 0.0
            changed = False
            for step, slot_id in zip(ordered_steps, ordered_slots):
                nominal = int(step.goal.get("nominal_slot_index", 0))
                assignments[step.id][env_id] = slot_id
                changed |= slot_id != nominal
                matching_cost += abs(
                    float(axis_positions[step.id][env_id])
                    - float(self.positions[env_id, slot_id, self.axis_index])
                )
            if changed:
                self.reassignment_reason[env_id] = (
                    "free arrangement initialized from live spatial order"
                )
                self.reassignment_cost[env_id] = matching_cost
        return assignments

    def _geometry(self, step: SemanticStep) -> _Geometry:
        entity = _object(self.env, step.object_uid)
        radii = []
        heights = []
        for env_id in range(self.num_envs):
            vertices = _local_vertices(entity, self.env, env_id)
            half_extent = (
                vertices.max(dim=0).values - vertices.min(dim=0).values
            ) * 0.5
            if step.goal.get("orientation_goal", "none") in {"none", "preserve"}:
                rotation = _live_pose(self.env, step.object_uid)[env_id, :3, :3]
                rotated = vertices @ rotation.transpose(0, 1)
                radii.append(torch.linalg.vector_norm(rotated[:, :2], dim=-1).max())
            else:
                # A non-preserve target may rotate the longest local dimension
                # into the table plane, so retain the conservative bound.
                radii.append(
                    torch.linalg.vector_norm(torch.topk(half_extent, k=2).values)
                )
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
        steps = int(self.row_search_radius / self.row_search_step)
        for index in range(1, steps + 1):
            offset = self.row_search_step * index
            search_offsets.extend((offset, -offset))
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
            + float(policy["surface_clearance"])
        )
        target[:, 2, 3] = final_z
        if phase == "staging":
            target[:, 2, 3] = final_z + float(policy["transport_clearance"])
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

    def metadata(self, step: SemanticStep, env_id: int) -> dict[str, Any]:
        """Describe the live slot resolution used by one environment."""
        nominal = int(step.goal.get("nominal_slot_index", 0))
        resolved = int(self.assignments[step.id][env_id])
        return {
            "nominal_slot_index": nominal,
            "resolved_slot_index": resolved,
            "slot_constraint": str(step.goal.get("slot_constraint", "required")),
            "slot_reassigned": resolved != nominal,
            "reassignment_reason": self.reassignment_reason[env_id],
            "matching_cost": (
                float(self.reassignment_cost[env_id])
                if torch.isfinite(self.reassignment_cost[env_id])
                else None
            ),
            "spacing": float(self.spacing[env_id]),
            "resolved_slot_position": self.positions[env_id, resolved].tolist(),
        }


class LivePlacementPlan:
    """Allocate non-overlapping live slots for one shared container."""

    def __init__(
        self,
        env: Any,
        steps: Sequence[SemanticStep],
        *,
        clearance: float | None = None,
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
        profile = str(getattr(env, "agent_robot_profile", "dual_ur10"))
        default_clearance = default_runtime_policy(profile).grounding["placement"][
            "clearance"
        ]
        self.clearance = float(default_clearance if clearance is None else clearance)
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
        runtime_policy: RuntimePolicyCfg | None = None,
        capability_registry: Any | None = None,
    ) -> None:
        self.program = program
        self.env = env
        self.semantics_factory = semantics_factory
        self.capabilities = capability_registry or build_atomic_capability_registry()
        self.robot_profile = str(getattr(env, "agent_robot_profile", "dual_ur10"))
        self.runtime_policy = runtime_policy or default_runtime_policy(
            self.robot_profile
        )
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

    def policy(
        self,
        action: Mapping[str, Any],
        *,
        extra_modifiers: tuple[tuple[str, str], ...] = (),
    ) -> dict[str, Any]:
        action_class = str(action.get("atomic_action_class", ""))
        capability = self.capabilities.get(action_class)
        motion_base = capability.motion_base or capability.name
        policy_spec = action.get("motion_policy", {"modifiers": []})
        if extra_modifiers:
            policy_spec = with_motion_modifiers(policy_spec, *extra_modifiers)
        inline = action.get("motion_policy_config", action.get("cfg"))
        return resolve_motion_policy(
            self.robot_profile,
            motion_base,
            policy_spec,
            motion_defaults=self.runtime_policy.motion_defaults,
            motion_modifiers=self.runtime_policy.motion_modifiers,
            inline_overrides=inline if isinstance(inline, Mapping) else None,
        )

    def _policy_value(self, policy: Mapping[str, Any], key: str) -> Any:
        defaults = self.runtime_policy.grounding["semantic_defaults"]
        return policy[key] if key in policy else defaults[key]

    def ground(
        self,
        action: Mapping[str, Any],
        step: SemanticStep,
        *,
        arm: str,
        state: ExecutionState,
        reference_eef_pose: torch.Tensor | None = None,
        orientation_reference_pose: torch.Tensor | None = None,
        _handover_workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> GroundedAction:
        action_class = str(action["atomic_action_class"])
        capability = self.capabilities.require_executable(action_class)
        self.capabilities.validate_binding(action)
        control = str(action.get("control", "arm"))
        binding = action.get("target_binding", {})
        if not isinstance(binding, Mapping):
            raise ValueError("target_binding must be a mapping.")
        kind = str(binding.get("kind", ""))
        orientation = compile_orientation_constraint(step.goal)
        is_handover_continuation = self._is_handover_continuation(step)
        uses_handover_staging = (
            kind == "handover_staging"
            and capability.target_materializer == "semantic_held_object"
        )
        use_upright_yaw_search = (
            is_handover_continuation or uses_handover_staging
        ) and self._uses_upright_yaw_search(
            step,
            orientation,
        )
        extra_modifiers: tuple[tuple[str, str], ...] = ()
        if (
            is_handover_continuation
            and use_upright_yaw_search
            and capability.target_materializer
            in {
                "semantic_held_object",
                "current_held_pose",
                "eef_pose",
            }
        ):
            extra_modifiers = (("orientation", "upright"),)
        policy = self.policy(action, extra_modifiers=extra_modifiers)
        if kind == "joint_state":
            joint_defaults = self.runtime_policy.grounding["joint_state"]
            source = binding.get("source")
            if source == "gripper_closed":
                policy["sample_interval"] = int(
                    joint_defaults["hand_close_sample_interval"]
                )
            elif source == "gripper_open":
                policy["sample_interval"] = int(
                    joint_defaults["hand_open_sample_interval"]
                )
            elif source == "initial" and control == "arm":
                # Returning home after release is a safety motion. If the
                # collision-aware planner cannot find a route, do not silently
                # replace it with collision-unaware joint interpolation.
                policy["collision_safety"] = "required"
        if uses_handover_staging and use_upright_yaw_search:
            # Handover consumes the live payload pose immediately after this
            # move.  Use the existing upright-yaw feasibility search instead
            # of the generic transport orientation heuristic, which can tilt
            # a payload while moving it to the exchange point.
            policy["upright_yaw_samples"] = max(
                int(policy.get("upright_yaw_samples", 1)),
                8,
            )
        object_pose = _live_pose(self.env, step.object_uid)
        if step.operator == "orient_object":
            policy["upright_local_axis"] = self._upright_local_axis(step)
            if capability.target_materializer == "object_grasp":
                policy["obj_upright_direction"] = self._upright_local_direction(step)
        reference_pose = self._reference_pose(step)
        target_object_pose = None

        if capability.target_materializer_hook is not None:
            grounded = capability.target_materializer_hook(
                grounder=self,
                action=action,
                step=step,
                arm=arm,
                state=state,
                binding=binding,
                policy=policy,
                object_pose=object_pose,
                reference_pose=reference_pose,
                reference_eef_pose=reference_eef_pose,
                orientation_reference_pose=orientation_reference_pose,
            )
            if not isinstance(grounded, GroundedAction):
                raise TypeError(
                    f"AtomicAction {action_class!r} target materializer must "
                    "return GroundedAction."
                )
            return grounded

        if kind == "object":
            semantics = self.semantics_factory(
                str(binding.get("object", step.object_uid))
            )
            if capability.target_materializer == "object_grasp":
                target: Any = GraspGoal(semantics=semantics)
            elif capability.target_materializer == "coordinated_pickment":
                target_object_pose = self._semantic_target(
                    step,
                    object_pose,
                    reference_pose,
                    policy,
                    phase="final",
                    orientation_reference_pose=orientation_reference_pose,
                )
                target = CoordinatedPickGoal(
                    object_target_pose=target_object_pose,
                    semantics=semantics,
                    object_initial_pose=object_pose,
                )
            elif capability.target_materializer == "press":
                target_object_pose = object_pose.clone()
                target = PressGoal(
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
                step,
                object_pose,
                reference_pose,
                policy,
                phase=phase,
                orientation_reference_pose=orientation_reference_pose,
            )
            if capability.target_materializer == "coordinated_pickment":
                semantics = self.semantics_factory(step.object_uid)
                target = CoordinatedPickGoal(
                    object_target_pose=target_object_pose,
                    semantics=semantics,
                    object_initial_pose=object_pose,
                )
            elif capability.target_materializer == "press":
                # Press moves the TCP, not the target object. Keep the object's
                # live pose as the postcondition reference while grounding a
                # downward contact point from its current surface geometry.
                target_object_pose = object_pose.clone()
                target = PressGoal(
                    xpos=self._press_pose(
                        arm,
                        step.object_uid,
                        object_pose,
                        policy,
                    )
                )
            elif capability.target_materializer == "semantic_held_object":
                target = HeldObjectPoseGoal(object_target_pose=target_object_pose)
            else:
                raise ValueError(
                    f"Target materializer {capability.target_materializer!r} cannot "
                    f"resolve {kind!r}."
                )
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
                orientation_reference_pose=orientation_reference_pose,
            )
            target = CoordinatedPlacementGoal(
                placing_object_target_pose=target_object_pose,
                support_object_target_pose=support_pose,
                release=bool(step.goal.get("release", True)),
            )
        elif kind == "current_held_pose":
            if state.get_held_object(arm_control_part(self.env, arm)) is None:
                raise ValueError("Place requires a held object from a prior PickUp.")
            target = PlaceGoal(
                xpos=(
                    reference_eef_pose
                    if reference_eef_pose is not None
                    else self._current_eef_pose(arm)
                )
            )
        elif kind == "policy_pose":
            source = binding.get("source")
            retreat_reference = self._retreat_reference_pose(
                arm,
                reference_eef_pose,
            )
            if binding.get("operation") == "retreat":
                policy["retreat_reachability_search"] = True
                policy["retreat_reference_pose"] = retreat_reference.clone()
            if source in {"release", "handover"}:
                policy["clearance_object_uid"] = step.object_uid
                policy["collision_safety"] = "required"
                contact_uids = [step.object_uid]
                reference_uid = step.goal.get("reference_object")
                if isinstance(reference_uid, str) and reference_uid:
                    contact_uids.append(reference_uid)
                policy["collision_exclusion_uids"] = list(dict.fromkeys(contact_uids))
            if source == "handover":
                policy.update(self.runtime_policy.grounding["handover"])
                policy["transfer_arm"] = arm
                policy["transfer_role_axis"] = self._handover_role_axis(
                    arm,
                    dtype=object_pose.dtype,
                    device=object_pose.device,
                )
            target = EndEffectorPoseGoal(
                xpos=self._retreat_pose(
                    arm,
                    policy,
                    retreat_reference,
                    clear_exchange=source == "handover",
                )
            )
        elif kind == "visual_constraint":
            visual_pose = self._visual_target(binding, arm)
            if capability.target_materializer == "semantic_held_object":
                target_object_pose = object_pose.clone()
                target_object_pose[:, :3, 3] = visual_pose[:, :3, 3]
                target = HeldObjectPoseGoal(object_target_pose=target_object_pose)
            elif capability.target_materializer == "eef_pose":
                target = EndEffectorPoseGoal(xpos=visual_pose)
            else:
                raise ValueError(
                    f"Target materializer {capability.target_materializer!r} "
                    "cannot resolve a visual_constraint."
                )
        elif kind == "joint_state":
            target = JointPositionGoal(
                target=self._joint_target(
                    arm,
                    control,
                    str(binding.get("source", "initial")),
                    binding,
                )
            )
        elif kind in {"eef_pose", "pose"}:
            target = EndEffectorPoseGoal(xpos=self._explicit_pose(binding, object_pose))
        elif kind == "handover_goal":
            target, target_object_pose, policy = self._handover_target(
                step,
                binding,
                object_pose,
                reference_pose,
                policy,
                state,
                orientation_reference_pose=orientation_reference_pose,
                workspace=_handover_workspace,
            )
        elif kind == "handover_staging":
            transfer_arm = str(binding.get("transfer_arm", "left_arm"))
            receive_arm = str(binding.get("receive_arm", "right_arm"))
            middle, _ = self._handover_workspace_poses(
                object_pose,
                transfer_arm=transfer_arm,
                receive_arm=receive_arm,
                policy=policy,
                step=step,
                orientation_reference_pose=orientation_reference_pose,
            )
            middle[:, :3, :3] = self._target_rotation(
                step,
                object_pose,
                orientation_reference_pose=orientation_reference_pose,
            )
            target_object_pose = middle
            target = HeldObjectPoseGoal(object_target_pose=middle)
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
            object_uid=step.object_uid,
        )

    def _handover_role_axis(
        self,
        transfer_arm: str,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the world-space axis from the receiver base to transfer base."""
        if transfer_arm not in {"left_arm", "right_arm"}:
            raise ValueError(f"Unknown handover arm {transfer_arm!r}.")
        _, lateral = robot_frame_axes(self.env)
        horizontal = lateral if transfer_arm == "left_arm" else -lateral
        return torch.cat(
            (
                horizontal.to(dtype=dtype, device=device),
                torch.zeros(
                    (int(self.env.num_envs), 1),
                    dtype=dtype,
                    device=device,
                ),
            ),
            dim=1,
        )

    def ground_candidates(
        self,
        action: Mapping[str, Any],
        step: SemanticStep,
        *,
        arm: str,
        state: ExecutionState,
        reference_eef_pose: torch.Tensor | None = None,
        orientation_reference_pose: torch.Tensor | None = None,
    ) -> tuple[GroundedAction, ...]:
        """Return deterministic grounding candidates for an opt-in capability."""
        binding = action.get("target_binding", {})
        if not isinstance(binding, Mapping):
            return (
                self.ground(
                    action,
                    step,
                    arm=arm,
                    state=state,
                    reference_eef_pose=reference_eef_pose,
                    orientation_reference_pose=orientation_reference_pose,
                ),
            )
        placement_support_uid = self._placement_support_uid(step)
        placement_relation = (
            normalize_placement_relation(step.goal.get("relation", "on"))
            if step.operator == "place_relative"
            else str(step.goal.get("relation", "none"))
        )
        is_on_placement = (
            binding.get("kind") == "semantic_goal"
            and binding.get("phase", "final") != "staging"
            and placement_relation in {"on", "on_top", "on_top_of"}
            and placement_support_uid is not None
        )
        if is_on_placement:
            base = self.ground(
                action,
                step,
                arm=arm,
                state=state,
                reference_eef_pose=reference_eef_pose,
                orientation_reference_pose=orientation_reference_pose,
            )
            return self._placement_grounding_candidates(
                base,
                step,
                support_uid=placement_support_uid,
            )
        if binding.get("kind") != "handover_goal":
            return (
                self.ground(
                    action,
                    step,
                    arm=arm,
                    state=state,
                    reference_eef_pose=reference_eef_pose,
                    orientation_reference_pose=orientation_reference_pose,
                ),
            )
        policy = self.policy(action)
        object_pose = _live_pose(self.env, step.object_uid)
        rotation = self._target_rotation(
            step,
            object_pose,
            orientation_reference_pose=orientation_reference_pose,
        )
        workspaces = self._handover_workspace_candidates(
            step,
            object_pose,
            transfer_arm=str(binding.get("transfer_arm", "left_arm")),
            receive_arm=str(binding.get("receive_arm", "right_arm")),
            policy=policy,
            rotation=rotation,
        )
        return tuple(
            self.ground(
                action,
                step,
                arm=arm,
                state=state,
                reference_eef_pose=reference_eef_pose,
                orientation_reference_pose=orientation_reference_pose,
                _handover_workspace=workspace,
            )
            for workspace in workspaces
        )

    def _placement_grounding_candidates(
        self,
        base: GroundedAction,
        step: SemanticStep,
        *,
        support_uid: str,
    ) -> tuple[GroundedAction, ...]:
        """Sample bounded support-relative poses from live object geometry."""
        if base.target_object_pose is None or not isinstance(
            base.target, HeldObjectPoseGoal
        ):
            return (base,)
        support = _object(self.env, support_uid)
        moved = _object(self.env, step.object_uid)
        placement = self.runtime_policy.grounding["placement"]
        count = int(placement["candidate_count"])
        fraction = float(placement["candidate_offset_fraction"])
        margin = float(placement["support_margin"])
        patterns = (
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        )[:count]
        candidates: list[GroundedAction] = []
        seen_offsets: list[torch.Tensor] = []
        for candidate_index, pattern in enumerate(patterns):
            target_pose = base.target_object_pose.clone()
            offsets = target_pose.new_zeros((int(self.env.num_envs), 2))
            for env_id in range(int(self.env.num_envs)):
                support_vertices = _world_vertices(support, self.env, env_id)
                moved_local = _local_vertices(moved, self.env, env_id)
                rotated = moved_local @ target_pose[env_id, :3, :3].transpose(0, 1)
                support_lower = support_vertices[:, :2].min(dim=0).values
                support_upper = support_vertices[:, :2].max(dim=0).values
                moved_lower = rotated[:, :2].min(dim=0).values
                moved_upper = rotated[:, :2].max(dim=0).values
                allowed_lower = support_lower + margin - moved_lower
                allowed_upper = support_upper - margin - moved_upper
                if bool(torch.all(allowed_lower <= allowed_upper)):
                    base_xy = target_pose[env_id, :2, 3].clone()
                    center = torch.minimum(
                        torch.maximum(base_xy, allowed_lower),
                        allowed_upper,
                    )
                    direction = target_pose.new_tensor(pattern)
                    room = torch.where(
                        direction >= 0.0,
                        allowed_upper - center,
                        center - allowed_lower,
                    )
                    candidate_xy = center + direction * room * fraction
                    offsets[env_id] = candidate_xy - base_xy
                    target_pose[env_id, :2, 3] = candidate_xy

                footprint_lower = target_pose[env_id, :2, 3] + moved_lower
                footprint_upper = target_pose[env_id, :2, 3] + moved_upper
                local_mask = torch.all(
                    (support_vertices[:, :2] >= footprint_lower - margin)
                    & (support_vertices[:, :2] <= footprint_upper + margin),
                    dim=1,
                )
                if bool(local_mask.any()):
                    support_height = support_vertices[local_mask, 2].max()
                else:
                    distances = torch.linalg.vector_norm(
                        support_vertices[:, :2] - target_pose[env_id, :2, 3],
                        dim=1,
                    )
                    nearest_count = min(8, int(support_vertices.shape[0]))
                    nearest = torch.topk(
                        distances,
                        nearest_count,
                        largest=False,
                    ).indices
                    support_height = support_vertices[nearest, 2].max()
                target_pose[env_id, 2, 3] = (
                    support_height
                    + float(self._policy_value(base.motion_policy, "surface_clearance"))
                    - rotated[:, 2].min()
                )
            if any(torch.allclose(offsets, prior) for prior in seen_offsets):
                continue
            seen_offsets.append(offsets)
            candidates.append(
                replace(
                    base,
                    target=replace(base.target, object_target_pose=target_pose),
                    target_object_pose=target_pose,
                    motion_policy={
                        **base.motion_policy,
                        "placement_candidate_index": candidate_index,
                        "placement_xy_offset": offsets,
                    },
                )
            )
        return tuple(candidates) or (base,)

    @staticmethod
    def _placement_support_uid(step: SemanticStep) -> str | None:
        value = step.goal.get("reference_object", step.goal.get("support_object"))
        if isinstance(value, str) and value:
            return value
        if (
            step.postcondition.get("type") == "stack_layer_supported"
            and int(step.goal.get("layer_index", -1)) == 0
        ):
            return "table"
        return None

    def _is_handover_continuation(self, step: SemanticStep) -> bool:
        if step.operator != "place_relative":
            return False
        predecessors = {
            candidate.id: candidate for candidate in self.program.semantic_steps
        }
        return any(
            (predecessor := predecessors.get(dependency)) is not None
            and predecessor.operator == "handover"
            and predecessor.object_uid == step.object_uid
            for dependency in step.depends_on
        )

    def _visual_target(
        self,
        binding: Mapping[str, Any],
        arm: str,
    ) -> torch.Tensor:
        """Unproject one normalized image keypoint using live camera depth."""
        camera_uid = str(binding.get("camera_uid", ""))
        sensor = self.env.sim.get_sensor(camera_uid)
        if sensor is None:
            raise ValueError(f"Unknown visual-constraint camera {camera_uid!r}.")
        keypoint_value = binding.get("normalized_keypoint")
        if keypoint_value is None:
            bbox = binding.get("normalized_bbox")
            if isinstance(bbox, Sequence) and len(bbox) == 4:
                keypoint_value = [
                    (float(bbox[0]) + float(bbox[2])) * 0.5,
                    (float(bbox[1]) + float(bbox[3])) * 0.5,
                ]
        if keypoint_value is None:
            raise ValueError(
                "visual_constraint requires a normalized keypoint or bbox in [0, 1]."
            )
        keypoint = torch.as_tensor(
            keypoint_value,
            dtype=torch.float32,
            device=self.env.device,
        ).flatten()
        if keypoint.numel() != 2 or bool(
            ((~torch.isfinite(keypoint)) | (keypoint < 0.0) | (keypoint > 1.0)).any()
        ):
            raise ValueError(
                "visual_constraint requires a normalized keypoint or bbox in [0, 1]."
            )
        data = sensor.get_data()
        if "depth" not in data:
            raise ValueError(
                f"Camera {camera_uid!r} must provide depth for visual Grounding."
            )
        depth = torch.as_tensor(data["depth"], device=self.env.device).squeeze(-1)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0).repeat(int(self.env.num_envs), 1, 1)
        if depth.ndim != 3 or depth.shape[0] != int(self.env.num_envs):
            raise ValueError("Camera depth must have shape (N, H, W) or (N, H, W, 1).")
        height, width = depth.shape[-2:]
        pixel_x = min(max(int(round(float(keypoint[0]) * (width - 1))), 0), width - 1)
        pixel_y = min(max(int(round(float(keypoint[1]) * (height - 1))), 0), height - 1)
        distance = depth[:, pixel_y, pixel_x].to(torch.float32)
        if bool((~torch.isfinite(distance) | (distance <= 0.0)).any()):
            raise ValueError("visual_constraint keypoint has no valid live depth.")
        intrinsics = torch.as_tensor(
            sensor.get_intrinsics(),
            dtype=torch.float32,
            device=self.env.device,
        )
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0).repeat(int(self.env.num_envs), 1, 1)
        camera_pose = torch.as_tensor(
            sensor.get_arena_pose(to_matrix=True),
            dtype=torch.float32,
            device=self.env.device,
        )
        if camera_pose.ndim == 2:
            camera_pose = camera_pose.unsqueeze(0).repeat(int(self.env.num_envs), 1, 1)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        point = torch.stack(
            (
                (float(pixel_x) - cx) * distance / fx,
                (float(pixel_y) - cy) * distance / fy,
                distance,
                torch.ones_like(distance),
            ),
            dim=1,
        )
        world = torch.bmm(camera_pose, point.unsqueeze(-1)).squeeze(-1)
        target = self._current_eef_pose(arm).clone()
        target[:, :3, 3] = world[:, :3]
        return target

    def _handover_target(
        self,
        step: SemanticStep,
        binding: Mapping[str, Any],
        object_pose: torch.Tensor,
        reference_pose: torch.Tensor | None,
        policy: Mapping[str, Any],
        state: ExecutionState,
        *,
        orientation_reference_pose: torch.Tensor | None,
        workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[GraspGoal, torch.Tensor, dict[str, Any]]:
        transfer_arm = str(binding.get("transfer_arm", "left_arm"))
        receive_arm = str(
            binding.get(
                "receive_arm",
                "right_arm" if transfer_arm == "left_arm" else "left_arm",
            )
        )
        if transfer_arm == receive_arm or {transfer_arm, receive_arm} != {
            "left_arm",
            "right_arm",
        }:
            raise ValueError("HandOver requires distinct left_arm/right_arm roles.")
        transfer_part = arm_control_part(self.env, transfer_arm)
        held = state.get_held_object(transfer_part)
        if held is None:
            raise ValueError(
                f"HandOver requires {transfer_arm} to hold {step.object_uid!r}."
            )

        del reference_pose
        if workspace is None:
            middle, final = self._handover_workspace_poses(
                object_pose,
                transfer_arm=transfer_arm,
                receive_arm=receive_arm,
                policy=policy,
                step=step,
                orientation_reference_pose=orientation_reference_pose,
            )
        else:
            middle, final = (item.clone() for item in workspace)
        rotation = self._target_rotation(
            step,
            object_pose,
            orientation_reference_pose=orientation_reference_pose,
        )
        middle[:, :3, :3] = rotation
        final[:, :3, :3] = rotation
        semantics = self.semantics_factory(step.object_uid)
        grounded_policy = dict(policy)
        grounded_policy.update(
            {
                "transfer_arm": transfer_arm,
                "receive_arm": receive_arm,
                "middle_object_pose": middle,
                "final_object_pose": final,
            }
        )
        return (
            GraspGoal(semantics=semantics),
            middle,
            grounded_policy,
        )

    def _handover_workspace_poses(
        self,
        object_pose: torch.Tensor,
        *,
        transfer_arm: str,
        receive_arm: str,
        policy: Mapping[str, Any],
        step: SemanticStep,
        orientation_reference_pose: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose the highest-ranked collision-aware handover workspace."""
        rotation = self._target_rotation(
            step,
            object_pose,
            orientation_reference_pose=orientation_reference_pose,
        )
        candidates = self._handover_workspace_candidates(
            step,
            object_pose,
            transfer_arm=transfer_arm,
            receive_arm=receive_arm,
            policy=policy,
            rotation=rotation,
        )
        return candidates[0]

    def _handover_workspace_candidates(
        self,
        step: SemanticStep,
        object_pose: torch.Tensor,
        *,
        transfer_arm: str,
        receive_arm: str,
        policy: Mapping[str, Any],
        rotation: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Rank exchange poses inside the two arm workspaces and above obstacles."""
        if transfer_arm == receive_arm or {transfer_arm, receive_arm} != {
            "left_arm",
            "right_arm",
        }:
            raise ValueError("Handover workspace requires distinct arm roles.")
        table = self.env.sim.get_rigid_object("table")
        if table is not None and hasattr(table, "get_vertices"):
            centers = []
            tops = []
            bounds = []
            for env_id in range(int(self.env.num_envs)):
                vertices = _world_vertices(table, self.env, env_id)
                lower = vertices[:, :2].min(dim=0).values
                upper = vertices[:, :2].max(dim=0).values
                centers.append((lower + upper) * 0.5)
                tops.append(vertices[:, 2].max())
                bounds.append(torch.stack((lower, upper)))
            center = torch.stack(centers)
            table_top = torch.stack(tops)
            table_bounds = torch.stack(bounds)
        else:
            left = self._current_eef_pose("left_arm")
            right = self._current_eef_pose("right_arm")
            center = (left[:, :2, 3] + right[:, :2, 3]) * 0.5
            table_top = object_pose[:, 2, 3]
            extent = float(policy.get("exchange_candidate_offset", 0.16)) * 2.0
            table_bounds = torch.stack((center - extent, center + extent), dim=1)

        forward, lateral = robot_frame_axes(self.env)
        left_base, right_base = arm_base_poses(self.env)
        transfer_base = left_base if transfer_arm == "left_arm" else right_base
        receive_base = right_base if receive_arm == "right_arm" else left_base
        base_midpoint = (transfer_base[:, :2, 3] + receive_base[:, :2, 3]) * 0.5
        table_forward = torch.sum((center - base_midpoint) * forward, dim=1)
        shared_center = base_midpoint + forward * table_forward[:, None]
        offset = float(policy.get("exchange_candidate_offset", 0.16))
        obstacle_clearance = float(policy.get("exchange_obstacle_clearance", 0.04))
        tool_horizontal_envelope = float(
            policy.get("exchange_gripper_horizontal_envelope", 0.035)
        ) + float(policy.get("exchange_wrist_horizontal_envelope", 0.055))
        tool_vertical_envelope = float(
            policy.get("exchange_gripper_vertical_envelope", 0.025)
        ) + float(policy.get("exchange_wrist_vertical_envelope", 0.04))
        minimum_reach = float(policy.get("exchange_minimum_reach", 0.10))
        maximum_reach = float(policy.get("exchange_maximum_reach", 1.00))
        if not 0.0 <= minimum_reach < maximum_reach:
            raise ValueError("Handover reach bounds require 0 <= minimum < maximum.")
        requested_count = max(1, int(policy.get("exchange_candidate_count", 4)))
        object_clearance = float(policy.get("exchange_clearance", 0.06))
        if (
            min(
                obstacle_clearance,
                tool_horizontal_envelope,
                tool_vertical_envelope,
                object_clearance,
            )
            < 0.0
        ):
            raise ValueError("Handover geometry clearances must be non-negative.")
        xy_coefficients = (
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (2.0, 0.0),
            (-2.0, 0.0),
            (0.0, 0.5),
            (0.0, -0.5),
        )
        ranked_by_env: list[list[tuple[float, torch.Tensor]]] = []
        moved = _object(self.env, step.object_uid)
        obstacle_uids = (
            self.env.sim.get_rigid_object_uid_list()
            if hasattr(self.env.sim, "get_rigid_object_uid_list")
            else []
        )
        for env_id in range(int(self.env.num_envs)):
            local_vertices = _local_vertices(moved, self.env, env_id)
            rotated = local_vertices @ rotation[env_id].transpose(0, 1)
            half_xy = (
                rotated[:, :2].max(dim=0).values - rotated[:, :2].min(dim=0).values
            ) * 0.5
            bottom = rotated[:, 2].min()
            margin = half_xy + obstacle_clearance + tool_horizontal_envelope
            lower_limit = table_bounds[env_id, 0] + margin
            upper_limit = table_bounds[env_id, 1] - margin
            options: list[tuple[float, torch.Tensor]] = []
            for forward_scale, lateral_scale in xy_coefficients:
                xy = (
                    shared_center[env_id]
                    + forward[env_id] * (offset * forward_scale)
                    + lateral[env_id] * (offset * lateral_scale)
                )
                if bool(((xy < lower_limit) | (xy > upper_limit)).any()):
                    continue
                transfer_distance = torch.linalg.vector_norm(
                    xy - transfer_base[env_id, :2, 3]
                )
                receive_distance = torch.linalg.vector_norm(
                    xy - receive_base[env_id, :2, 3]
                )
                if not (
                    minimum_reach <= float(transfer_distance) <= maximum_reach
                    and minimum_reach <= float(receive_distance) <= maximum_reach
                ):
                    continue
                obstacle_score, nearby_obstacle_top = self._handover_obstacle_metrics(
                    xy,
                    env_id=env_id,
                    object_uid=step.object_uid,
                    obstacle_uids=obstacle_uids,
                    half_xy=half_xy,
                    clearance=obstacle_clearance + tool_horizontal_envelope,
                )
                center_cost = float(
                    torch.linalg.vector_norm(xy - shared_center[env_id])
                )
                pose = object_pose[env_id].clone()
                pose[:3, :3] = rotation[env_id]
                pose[:2, 3] = xy
                safety_floor = torch.maximum(table_top[env_id], nearby_obstacle_top)
                safe_z = (
                    safety_floor + object_clearance + tool_vertical_envelope - bottom
                )
                pose[2, 3] = torch.maximum(object_pose[env_id, 2, 3], safe_z)
                lift_cost = max(
                    0.0,
                    float(pose[2, 3] - object_pose[env_id, 2, 3]),
                )
                options.append(
                    (obstacle_score + center_cost * 0.25 + lift_cost * 0.1, pose)
                )
            if not options:
                raise ValueError(
                    "No handover exchange pose lies inside the table bounds and "
                    "the reachable intersection of both arm bases."
                )
            options.sort(key=lambda item: item[0])
            ranked_by_env.append(options[:requested_count])

        candidate_count = min(
            requested_count,
            max(len(options) for options in ranked_by_env),
        )
        candidates = []
        for candidate_index in range(candidate_count):
            middle = object_pose.clone()
            for env_id, options in enumerate(ranked_by_env):
                middle[env_id] = options[min(candidate_index, len(options) - 1)][1]
            # The built-in HandOver primitive plans its final transfer/receiver
            # phase concurrently. An exchange-to-exchange target makes that
            # receiver path stationary; graph-level retreat/home nodes then
            # clear the transfer arm before any receiver-side continuation.
            final = middle.clone()
            candidates.append((middle, final))
        return tuple(candidates)

    def _handover_obstacle_metrics(
        self,
        xy: torch.Tensor,
        *,
        env_id: int,
        object_uid: str,
        obstacle_uids: Sequence[str],
        half_xy: torch.Tensor,
        clearance: float,
    ) -> tuple[float, torch.Tensor]:
        score = 0.0
        highest_top = torch.tensor(
            -torch.inf,
            dtype=xy.dtype,
            device=xy.device,
        )
        for uid in obstacle_uids:
            if uid in {"table", object_uid}:
                continue
            obstacle = self.env.sim.get_rigid_object(uid)
            if obstacle is None or not hasattr(obstacle, "get_vertices"):
                continue
            vertices = _world_vertices(obstacle, self.env, env_id)
            lower = vertices[:, :2].min(dim=0).values - half_xy - clearance
            upper = vertices[:, :2].max(dim=0).values + half_xy + clearance
            outside = torch.maximum(
                torch.maximum(lower - xy, xy - upper),
                torch.zeros_like(xy),
            )
            if bool((outside > 0.0).any()):
                distance = float(torch.linalg.vector_norm(outside))
                score += 1.0 / max(distance, 1.0e-3)
            else:
                score += 1.0e3
                highest_top = torch.maximum(highest_top, vertices[:, 2].max())
        return score, highest_top

    def _handover_receiver_exit(
        self,
        middle: torch.Tensor,
        receive_arm: str,
        policy: Mapping[str, Any],
    ) -> torch.Tensor:
        final = middle.clone()
        receive_pose = self._current_eef_pose(receive_arm)
        direction = receive_pose[:, :2, 3] - middle[:, :2, 3]
        norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        fallback = direction.new_zeros(direction.shape)
        fallback[:, 1] = -1.0 if receive_arm == "right_arm" else 1.0
        direction = torch.where(
            norm > 1.0e-6, direction / norm.clamp_min(1.0e-6), fallback
        )
        final[:, :2, 3] += direction * min(
            0.12,
            float(self._policy_value(policy, "relation_distance")) * 0.5,
        )
        return final

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
        orientation_reference_pose: torch.Tensor | None = None,
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
            target[:, :3, :3] = self._target_rotation(
                step,
                object_pose,
                orientation_reference_pose=orientation_reference_pose,
            )
            moved = _object(self.env, step.object_uid)
            for env_id in range(int(self.env.num_envs)):
                bottom = self._rotated_local_z_min(
                    moved,
                    target[env_id, :3, :3],
                    env_id,
                )
                target[env_id, 2, 3] = (
                    arrangement.table_top[env_id]
                    + float(policy["surface_clearance"])
                    - bottom
                )
            if phase == "staging":
                target[:, 2, 3] += float(policy["transport_clearance"])
            return target
        placement = self.placements.get(step.id)
        if placement is not None:
            target = placement.target(
                step,
                object_pose,
                self._target_rotation(
                    step,
                    object_pose,
                    orientation_reference_pose=orientation_reference_pose,
                ),
                surface_clearance=float(policy["surface_clearance"]),
            )
            if phase == "staging":
                target[:, 2, 3] += float(policy["transport_clearance"])
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
            target[:, :3, :3] = self._target_rotation(
                step,
                target,
                orientation_reference_pose=orientation_reference_pose,
            )
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
                    support_top + float(policy["surface_clearance"]) - bottom
                )
            if phase == "staging":
                target[:, 2, 3] += float(policy["staging_lift_height"])
            return target
        target = object_pose.clone()
        if reference_pose is not None:
            target[:, :3, 3] = reference_pose[:, :3, 3]
        # Operators without a relational goal (for example press or a
        # direction-only coordinated transport) must preserve the live origin
        # instead of being silently projected onto a synthetic table support.
        relation = (
            normalize_placement_relation(step.goal.get("relation", "on"))
            if step.operator == "place_relative"
            else str(step.goal.get("relation", "none"))
        )
        distance = float(self._policy_value(policy, "relation_distance"))
        relation_frame = str(step.goal.get("relation_frame", "world"))
        forward_distance = distance
        lateral_distance = distance
        if relation_frame == "robot" and reference_pose is not None:
            nominal = float(policy.get("robot_relative_distance", 0.10))
            clearance = float(policy.get("relation_clearance", 0.02))
            reference_uid = str(step.goal.get("reference_object", ""))
            if reference_uid:
                forward_axis, lateral_axis = robot_frame_axes(self.env)
                forward_distance = self._relative_object_spacing(
                    step.object_uid,
                    reference_uid,
                    axis=forward_axis,
                    nominal=nominal,
                    clearance=clearance,
                )
                lateral_distance = self._relative_object_spacing(
                    step.object_uid,
                    reference_uid,
                    axis=lateral_axis,
                    nominal=nominal,
                    clearance=clearance,
                )
        directional_offset = relation_offset(
            self.env,
            relation,
            frame=relation_frame,
            forward_distance=forward_distance,
            lateral_distance=lateral_distance,
            dtype=target.dtype,
            device=target.device,
        )
        offsets = {
            "above": (0.0, 0.0, float(self._policy_value(policy, "hover_height"))),
            "held_above_initial": (
                0.0,
                0.0,
                float(self._policy_value(policy, "hover_height")),
            ),
        }
        if directional_offset is not None:
            target[:, :3, 3] += directional_offset
        elif (offset := offsets.get(relation)) is not None:
            target[:, :3, 3] += torch.tensor(
                offset,
                dtype=target.dtype,
                device=target.device,
            )
        slot = str(step.goal.get("slot", "auto"))
        if relation in {"on", "on_top", "on_top_of", "inside"} and slot in {
            "left",
            "right",
        }:
            slot_offset = relation_offset(
                self.env,
                slot,
                frame=relation_frame,
                forward_distance=forward_distance,
                lateral_distance=lateral_distance,
                dtype=target.dtype,
                device=target.device,
            )
            if slot_offset is not None:
                target[:, :3, 3] += slot_offset
        direction = str(step.goal.get("direction", "none"))
        direction_offsets = {
            "world_x": (distance, 0.0, 0.0),
            "world_y": (0.0, distance, 0.0),
            "up": (0.0, 0.0, distance),
            "down": (0.0, 0.0, -distance),
        }
        planar_direction_offset = relation_offset(
            self.env,
            direction,
            frame=relation_frame,
            forward_distance=distance,
            lateral_distance=distance,
            dtype=target.dtype,
            device=target.device,
        )
        if planar_direction_offset is not None:
            target[:, :3, 3] += planar_direction_offset
        elif direction in direction_offsets:
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

        target[:, :3, :3] = self._target_rotation(
            step,
            object_pose,
            orientation_reference_pose=orientation_reference_pose,
        )
        if (
            step.operator == "coordinated_transport"
            and relation not in {"on", "on_top", "on_top_of", "inside"}
            and direction not in {"up", "down"}
        ):
            release = str(step.goal.get("terminal_behavior", "hold")) == "place"
            if not release:
                target[:, 2, 3] = object_pose[:, 2, 3] + float(
                    self._policy_value(policy, "transport_clearance")
                )
            else:
                table = _object(self.env, "table")
                moved = _object(self.env, step.object_uid)
                clearance = float(self._policy_value(policy, "surface_clearance"))
                for env_id in range(int(self.env.num_envs)):
                    table_top = _world_vertices(table, self.env, env_id)[:, 2].max()
                    bottom = self._rotated_local_z_min(
                        moved,
                        target[env_id, :3, :3],
                        env_id,
                    )
                    target[env_id, 2, 3] = table_top + clearance - bottom
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
                    support_top
                    + float(self._policy_value(policy, "surface_clearance"))
                    - bottom
                )
        elif relation == "inside" and reference_pose is not None:
            # Grounding the final move happens after the staging lift. Preserve
            # the pre-pick supported height rather than the lifted live height.
            supported_pose = orientation_reference_pose
            if supported_pose is None:
                supported_pose = object_pose
            supported_pose = _batched_pose(supported_pose, self.env)
            target[:, 2, 3] = supported_pose[:, 2, 3]
        if phase == "staging":
            # Staging is a runtime waypoint, not a persisted coordinate. This
            # keeps in-place orientation robust to the object's live height.
            target[:, 2, 3] += float(self._policy_value(policy, "transport_clearance"))
        elif self._is_handover_continuation(step) and relation not in {
            "on",
            "on_top",
            "on_top_of",
            "inside",
        }:
            # A handover can leave the live rigid-body center a few centimetres
            # below the original table-supported height.  Reusing that drifted
            # height for the lateral placement target makes the can intersect
            # the table during release and it may tip or slide.  Preserve the
            # predecessor's supported height for the final held-object pose.
            supported_pose = orientation_reference_pose
            if supported_pose is None:
                supported_pose = object_pose
            supported_pose = _batched_pose(supported_pose, self.env)
            target[:, 2, 3] = torch.maximum(
                target[:, 2, 3],
                supported_pose[:, 2, 3],
            )
        return target

    def _relative_object_spacing(
        self,
        moved_uid: str,
        reference_uid: str,
        *,
        axis: int | torch.Tensor,
        nominal: float,
        clearance: float,
    ) -> float:
        """Return deterministic center spacing from live object extents."""
        moved = _object(self.env, moved_uid)
        reference = _object(self.env, reference_uid)
        required = float(nominal)
        for env_id in range(int(self.env.num_envs)):
            moved_vertices = _world_vertices(moved, self.env, env_id)
            reference_vertices = _world_vertices(reference, self.env, env_id)
            if isinstance(axis, torch.Tensor):
                direction = axis[env_id].to(
                    dtype=moved_vertices.dtype,
                    device=moved_vertices.device,
                )
                moved_axis = moved_vertices[:, :2] @ direction
                reference_axis = reference_vertices[:, :2] @ direction
            else:
                moved_axis = moved_vertices[:, axis]
                reference_axis = reference_vertices[:, axis]
            moved_half = (moved_axis.max() - moved_axis.min()) * 0.5
            reference_half = (reference_axis.max() - reference_axis.min()) * 0.5
            required = max(
                required,
                float(moved_half + reference_half) + float(clearance),
            )
        return required

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

    def _uses_upright_yaw_search(
        self,
        step: SemanticStep,
        constraint: OrientationConstraint,
    ) -> bool:
        """Preserve a live upright state as a planning preference.

        Explicit full-frame matching cannot admit yaw search. With no hard
        orientation terms, yaw search is enabled only when the live object's
        long axis is already upright, so a preceding upright operation remains
        stable without turning that state into a sticky acceptance constraint.
        """
        if constraint.allows_upright_yaw_search:
            return True
        if (
            constraint.terms
            or constraint.planning_preference != "minimize_rotation_from_current"
        ):
            return False
        entity = _object(self.env, step.object_uid)
        vertices = _local_vertices(entity, self.env, 0)
        extents = vertices.max(dim=0).values - vertices.min(dim=0).values
        axis_index = int(torch.argmax(extents).item())
        pose = _live_pose(self.env, step.object_uid)
        cosine = pose[:, 2, axis_index].abs().clamp(0.0, 1.0)
        tolerance = float(self.runtime_policy.predicate_fallbacks["upright_max_tilt"])
        return bool(torch.all(torch.arccos(cosine) <= tolerance).item())

    @staticmethod
    def _upright_local_axis(step: SemanticStep) -> str:
        align_terms = tuple(
            term
            for term in compile_orientation_constraint(step.goal).terms
            if isinstance(term, AlignAxisConstraint)
        )
        if align_terms:
            return align_terms[0].local_axis
        axis = str(step.goal.get("upright_local_axis", "auto"))
        return "long_axis" if axis == "auto" else axis

    def _target_rotation(
        self,
        step: SemanticStep,
        object_pose: torch.Tensor,
        *,
        orientation_reference_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        constraint = compile_orientation_constraint(step.goal)
        if not constraint.terms:
            return object_pose[:, :3, :3].clone()
        if (
            len(constraint.terms) == 1
            and isinstance(constraint.terms[0], MatchRotationConstraint)
            and constraint.terms[0].reference == "step_start"
        ):
            if orientation_reference_pose is not None:
                reference = _batched_pose(orientation_reference_pose, self.env)
                return reference[:, :3, :3].clone()
            return object_pose[:, :3, :3].clone()
        goal = str(step.goal.get("orientation_goal", "none"))
        align_term = next(
            (
                term
                for term in constraint.terms
                if isinstance(term, AlignAxisConstraint)
            ),
            None,
        )
        if align_term is not None:
            goal = "upright"
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
                upright_axis = (
                    align_term.local_axis
                    if align_term is not None
                    else self._upright_local_axis(step)
                )
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
        depth = float(self._policy_value(policy, "press_depth"))
        for env_id in range(int(self.env.num_envs)):
            top = _world_vertices(entity, self.env, env_id)[:, 2].max()
            target[env_id, 2, 3] = top - depth
        return target

    def _retreat_pose(
        self,
        arm: str,
        policy: Mapping[str, Any],
        reference: torch.Tensor | None,
        *,
        clear_exchange: bool = False,
    ) -> torch.Tensor:
        target = self._retreat_reference_pose(arm, reference).clone()
        desired = float(self._policy_value(policy, "retreat_height"))
        if clear_exchange:
            _, lateral = robot_frame_axes(self.env)
            direction = lateral if arm == "left_arm" else -lateral
            target[:, :2, 3] += direction.to(
                dtype=target.dtype,
                device=target.device,
            ) * float(policy.get("retreat_distance", 0.10))
            desired = max(
                desired,
                float(self._policy_value(policy, "minimum_retreat_height")),
            )
        ceiling = float(self._policy_value(policy, "maximum_eef_height"))
        height = torch.clamp(ceiling - target[:, 2, 3], min=0.0, max=desired)
        target[:, 2, 3] += height
        return target

    def _retreat_reference_pose(
        self,
        arm: str,
        reference: torch.Tensor | None,
    ) -> torch.Tensor:
        """Resolve the live or speculative TCP pose from which retreat starts."""
        pose = reference
        if pose is None and hasattr(self.env, "get_current_xpos_agent"):
            left, right = self.env.get_current_xpos_agent()
            pose = left if arm == "left_arm" else right
        if pose is None:
            raise ValueError("Retreat grounding requires a live end-effector pose.")
        return _batched_pose(pose, self.env)

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
        grasp_policy = self.runtime_policy.grounding["coordinated_grasp"]
        inset = max(
            float(grasp_policy["minimum_inset"]),
            float((upper[axis] - lower[axis]) * grasp_policy["inset_fraction"]),
        )
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
