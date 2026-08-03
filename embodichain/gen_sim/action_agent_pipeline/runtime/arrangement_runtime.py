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

"""Materialize coordinate-free arrangement slots from each live environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from embodichain.gen_sim.action_agent_pipeline.config.defaults import defaults_section
from embodichain.gen_sim.action_agent_pipeline.runtime.pose_utils import (
    _object_mesh_vertices,
    _object_world_vertices,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.motion_policy import (
    resolve_motion_policy,
)

__all__ = ["ArrangementRuntimePlan"]

_DEFAULTS = defaults_section("arrangement")
_SLOT_MARGIN = float(_DEFAULTS["slot_margin"])
_MIN_SLOT_SPACING = float(_DEFAULTS["min_slot_spacing"])
_LAYOUT_CLEARANCE = float(_DEFAULTS["layout_clearance"])
_TRANSPORT_CLEARANCE = float(_DEFAULTS["transport_clearance"])
_ROW_SEARCH_STEP = float(_DEFAULTS["row_search_step"])
_ROW_SEARCH_RADIUS = float(_DEFAULTS["row_search_radius"])


@dataclass(frozen=True)
class _ObjectGeometry:
    xy_radius: torch.Tensor
    half_height: torch.Tensor


class ArrangementRuntimePlan:
    """Per-environment slot materialization for one immutable arrangement Seed."""

    def __init__(
        self,
        *,
        env: Any,
        semantic_steps: Sequence[Any],
    ) -> None:
        if not semantic_steps:
            raise ValueError("Arrangement runtime planning requires semantic steps.")
        self.env = env
        self.semantic_steps = tuple(semantic_steps)
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.step_by_id = {step.id: step for step in self.semantic_steps}
        self.slot_count = len(self.semantic_steps)
        self.axis = str(self.semantic_steps[0].goal["axis"])
        self.axis_index = 0 if self.axis in {"world_x", "x"} else 1
        self.perpendicular_index = 1 - self.axis_index
        # Slot indices always increase along the positive world axis. The
        # order_direction field selects which objects bind to those slots; it
        # does not reverse the spatial coordinate system.
        self.direction = 1.0

        table = env.sim.get_rigid_object("table")
        if table is None:
            raise ValueError(
                "Runtime arrangement grounding requires the live `table` object."
            )
        table_bounds = []
        for env_id in range(self.num_envs):
            vertices = _object_world_vertices(table, self.device, env_id=env_id)
            table_bounds.append(
                torch.stack([vertices.min(dim=0).values, vertices.max(dim=0).values])
            )
        self.table_bounds = torch.stack(table_bounds)
        self.table_center = self.table_bounds.mean(dim=1)
        self.table_top_z = self.table_bounds[:, 1, 2]

        self.object_geometry = {
            step.id: self._object_geometry(step.object_uid)
            for step in self.semantic_steps
        }
        self.initial_object_z = {
            step.id: env.sim.get_rigid_object(step.object_uid)
            .get_local_pose(to_matrix=True)[:, 2, 3]
            .clone()
            for step in self.semantic_steps
        }
        max_diameter = (
            torch.stack(
                [geometry.xy_radius * 2.0 for geometry in self.object_geometry.values()]
            )
            .max(dim=0)
            .values
        )
        self.spacing = torch.maximum(
            max_diameter + _SLOT_MARGIN,
            torch.full_like(max_diameter, _MIN_SLOT_SPACING),
        )
        self.slot_positions = self._make_slots()
        self.reassignment_reason: list[str | None] = [None] * self.num_envs
        self.reassignment_cost = torch.full(
            (self.num_envs,),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.resolved_slots = self._initial_slot_assignments()
        self.completed = {
            step.id: torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            for step in self.semantic_steps
        }

    def _initial_slot_assignments(self) -> dict[str, torch.Tensor]:
        """Match free-order objects to slots without crossing their live order."""
        resolved = {
            step.id: torch.full(
                (self.num_envs,),
                int(step.goal["nominal_slot_index"]),
                dtype=torch.long,
                device=self.device,
            )
            for step in self.semantic_steps
        }
        free_steps = [
            step
            for step in self.semantic_steps
            if step.goal.get("slot_constraint") == "free_reassignable"
        ]
        if not free_steps:
            return resolved

        required_slots = {
            int(step.goal["nominal_slot_index"])
            for step in self.semantic_steps
            if step.goal.get("slot_constraint") != "free_reassignable"
        }
        available_slots = [
            slot for slot in range(self.slot_count) if slot not in required_slots
        ]
        if len(available_slots) != len(free_steps):
            raise ValueError(
                "Arrangement slot constraints do not define a one-to-one assignment."
            )

        object_axis_positions = {
            step.id: self.env.sim.get_rigid_object(step.object_uid)
            .get_local_pose(to_matrix=True)[:, self.axis_index, 3]
            .clone()
            for step in free_steps
        }
        for env_id in range(self.num_envs):
            ordered_steps = sorted(
                free_steps,
                key=lambda step: (
                    float(object_axis_positions[step.id][env_id].item()),
                    int(step.goal["nominal_slot_index"]),
                    step.id,
                ),
            )
            ordered_slots = sorted(
                available_slots,
                key=lambda slot: (
                    float(self.slot_positions[env_id, slot, self.axis_index].item()),
                    slot,
                ),
            )
            matching_cost = 0.0
            changed = False
            for step, slot in zip(ordered_steps, ordered_slots):
                resolved[step.id][env_id] = slot
                changed |= slot != int(step.goal["nominal_slot_index"])
                matching_cost += abs(
                    float(object_axis_positions[step.id][env_id].item())
                    - float(self.slot_positions[env_id, slot, self.axis_index].item())
                )
            if changed:
                self.reassignment_reason[env_id] = (
                    "free arrangement initialized from live spatial order"
                )
                self.reassignment_cost[env_id] = matching_cost
        return resolved

    def target_positions(
        self,
        step: Any,
        *,
        object_pose: torch.Tensor,
        phase: str,
        policy: Mapping[str, Any],
    ) -> torch.Tensor:
        """Resolve one step's current slot into final or staging object positions."""
        if phase not in {"staging", "final"}:
            raise ValueError(f"Unsupported arrangement grounding phase: {phase!r}.")
        target = object_pose[:, :3, 3].clone()
        env_ids = torch.arange(self.num_envs, device=self.device)
        slots = self.resolved_slots[step.id]
        target[:, :2] = self.slot_positions[env_ids, slots, :2]
        geometry = self.object_geometry[step.id]
        surface_clearance = float(policy.get("surface_clearance", 0.0))
        final_z = self.table_top_z + geometry.half_height + surface_clearance
        target[:, 2] = final_z
        if phase == "staging":
            clearance = float(policy.get("transport_clearance", _TRANSPORT_CLEARANCE))
            staging_lift = float(policy.get("staging_lift_height", clearance))
            target[:, 2] = torch.maximum(
                self.initial_object_z[step.id] + staging_lift,
                final_z + clearance,
            )
        return target

    def metadata(self, step: Any) -> list[dict[str, Any]]:
        """Return serializable runtime layout metadata for every environment."""
        result = []
        nominal = int(step.goal["nominal_slot_index"])
        object_pose = self.env.sim.get_rigid_object(step.object_uid).get_local_pose(
            to_matrix=True
        )
        policy = resolve_motion_policy(
            str(getattr(self.env, "agent_robot_profile", "")),
            "default_transport",
        )
        staging = self.target_positions(
            step,
            object_pose=object_pose,
            phase="staging",
            policy=policy,
        )
        final = self.target_positions(
            step,
            object_pose=object_pose,
            phase="final",
            policy=policy,
        )
        for env_id in range(self.num_envs):
            resolved = int(self.resolved_slots[step.id][env_id].item())
            result.append(
                {
                    "nominal_slot_index": nominal,
                    "resolved_slot_index": resolved,
                    "slot_constraint": str(step.goal["slot_constraint"]),
                    "slot_reassigned": resolved != nominal,
                    "reassignment_reason": self.reassignment_reason[env_id],
                    "matching_cost": (
                        float(self.reassignment_cost[env_id].item())
                        if torch.isfinite(self.reassignment_cost[env_id])
                        else None
                    ),
                    "table_center": [
                        float(value)
                        for value in self.table_center[env_id].detach().cpu().tolist()
                    ],
                    "spacing": float(self.spacing[env_id].item()),
                    "resolved_slot_position": [
                        float(value)
                        for value in self.slot_positions[env_id, resolved]
                        .detach()
                        .cpu()
                        .tolist()
                    ],
                    "staging_position": [
                        float(value)
                        for value in staging[env_id].detach().cpu().tolist()
                    ],
                    "final_position": [
                        float(value) for value in final[env_id].detach().cpu().tolist()
                    ],
                }
            )
        return result

    def set_assignment(
        self,
        env_id: int,
        assignment: Mapping[str, int],
        *,
        reason: str,
        cost: float,
    ) -> None:
        """Install one complete matching for the remaining steps in one env."""
        for step_id, slot in assignment.items():
            self.resolved_slots[step_id][env_id] = int(slot)
        self.reassignment_reason[env_id] = reason
        self.reassignment_cost[env_id] = float(cost)

    def mark_completed(self, step_id: str, success: torch.Tensor) -> None:
        """Freeze successfully verified step-to-slot bindings."""
        self.completed[step_id] |= success.to(device=self.device, dtype=torch.bool)

    def remaining_step_ids(self, env_id: int) -> list[str]:
        return [
            step.id
            for step in self.semantic_steps
            if not bool(self.completed[step.id][env_id].item())
        ]

    def occupied_slots(self, env_id: int) -> set[int]:
        return {
            int(self.resolved_slots[step.id][env_id].item())
            for step in self.semantic_steps
            if bool(self.completed[step.id][env_id].item())
        }

    def _object_geometry(self, object_uid: str) -> _ObjectGeometry:
        obj = self.env.sim.get_rigid_object(object_uid)
        if obj is None:
            raise ValueError(f"Unknown arrangement object {object_uid!r}.")
        radii = []
        half_heights = []
        for env_id in range(self.num_envs):
            vertices = _object_mesh_vertices(obj, self.device, env_id=env_id)
            radii.append(torch.linalg.norm(vertices[:, :2], dim=-1).max())
            height = vertices[:, 2].max() - vertices[:, 2].min()
            half_heights.append(height * 0.5)
        return _ObjectGeometry(
            xy_radius=torch.stack(radii),
            half_height=torch.stack(half_heights),
        )

    def _make_slots(self) -> torch.Tensor:
        slot_offsets = (
            torch.arange(
                self.slot_count,
                dtype=torch.float32,
                device=self.device,
            )
            - (self.slot_count - 1) / 2.0
        )
        radii = torch.stack(
            [self.object_geometry[step.id].xy_radius for step in self.semantic_steps],
            dim=1,
        )
        obstacle_bounds = self._hard_obstacle_bounds()
        slots = torch.empty(
            (self.num_envs, self.slot_count, 3),
            dtype=torch.float32,
            device=self.device,
        )
        for env_id in range(self.num_envs):
            selected = None
            for perpendicular_offset in _row_search_offsets():
                candidate = self.table_center[env_id].repeat(self.slot_count, 1)
                candidate[:, self.axis_index] += (
                    self.direction * self.spacing[env_id] * slot_offsets
                )
                candidate[:, self.perpendicular_index] += perpendicular_offset
                candidate[:, 2] = self.table_top_z[env_id]
                if self._slots_are_safe(
                    candidate,
                    radii=radii[env_id],
                    table_bounds=self.table_bounds[env_id],
                    obstacle_bounds=obstacle_bounds[env_id],
                ):
                    selected = candidate
                    break
            if selected is None:
                raise ValueError(
                    f"Environment {env_id} has no collision-free arrangement row "
                    "within the live table bounds."
                )
            slots[env_id] = selected
        return slots

    def _hard_obstacle_bounds(self) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        result: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_envs)
        ]
        uid_getter = getattr(self.env.sim, "get_rigid_object_uid_list", None)
        if uid_getter is None:
            return result
        movable = {step.object_uid for step in self.semantic_steps}
        for uid in uid_getter():
            if uid == "table" or uid in movable:
                continue
            obj = self.env.sim.get_rigid_object(uid)
            if obj is None:
                continue
            for env_id in range(self.num_envs):
                vertices = _object_world_vertices(obj, self.device, env_id=env_id)
                if float(vertices[:, 2].max()) < float(
                    self.table_top_z[env_id] - _LAYOUT_CLEARANCE
                ):
                    continue
                result[env_id].append(
                    (
                        vertices[:, :2].min(dim=0).values,
                        vertices[:, :2].max(dim=0).values,
                    )
                )
        return result

    @staticmethod
    def _slots_are_safe(
        slots: torch.Tensor,
        *,
        radii: torch.Tensor,
        table_bounds: torch.Tensor,
        obstacle_bounds: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> bool:
        lower = table_bounds[0, :2] + radii[:, None] + _LAYOUT_CLEARANCE
        upper = table_bounds[1, :2] - radii[:, None] - _LAYOUT_CLEARANCE
        if bool(((slots[:, :2] < lower) | (slots[:, :2] > upper)).any()):
            return False
        for center, radius in zip(slots[:, :2], radii):
            for obstacle_lower, obstacle_upper in obstacle_bounds:
                closest = torch.maximum(
                    obstacle_lower,
                    torch.minimum(center, obstacle_upper),
                )
                if float(torch.linalg.norm(center - closest)) <= float(
                    radius + _LAYOUT_CLEARANCE
                ):
                    return False
        return True


def _row_search_offsets() -> list[float]:
    offsets = [0.0]
    steps = int(_ROW_SEARCH_RADIUS / _ROW_SEARCH_STEP)
    for index in range(1, steps + 1):
        value = index * _ROW_SEARCH_STEP
        offsets.extend([value, -value])
    return offsets
