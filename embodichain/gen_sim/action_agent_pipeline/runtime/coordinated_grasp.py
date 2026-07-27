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

"""Orchestrate dual-arm grasp generation, collision filtering, and IK selection."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from embodichain.gen_sim.action_agent_pipeline.runtime.coordinated_grasp_geometry import (
    _coordinated_grasp_pair_candidates,
    _coordinated_grasp_pair_world_y_angle_degrees,
    _filter_coordinated_payload_collision_candidates,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.coordinated_grasp_ik import (
    _has_coordinated_ik_api,
    _select_ik_feasible_coordinated_grasp_pair,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.pose_utils import (
    _ensure_batched_pose_tensor,
)
from embodichain.lab.sim.atomic_actions import ObjectSemantics
from embodichain.utils.logger import log_warning

__all__ = ["_default_coordinated_object_to_eef"]


def _default_coordinated_object_to_eef(
    semantics: ObjectSemantics,
    device,
    object_initial_pose: torch.Tensor,
    *,
    object_label: str | None = None,
    object_target_pose: torch.Tensor | None = None,
    pre_grasp_distance: float = 0.10,
    lift_height: float = 0.08,
    sample_interval: int = 120,
    hand_interp_steps: int = 10,
    hold_steps: int = 4,
    object_motion_keyframes: int = 6,
    max_grasp_separation_angle_to_world_y_degrees: float | None = None,
    payload_uids: Sequence[str] = (),
    env=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_was_batched = object_initial_pose.ndim == 3
    object_initial_pose = _ensure_batched_pose_tensor(object_initial_pose, device)
    if object_target_pose is not None:
        object_target_pose = _ensure_batched_pose_tensor(object_target_pose, device)
        if object_target_pose.shape[0] != object_initial_pose.shape[0]:
            raise ValueError(
                "CoordinatedPickment initial and target pose batches must match."
            )
    vertices = semantics.geometry.get("mesh_vertices")
    if vertices is None:
        vertices = semantics.entity.get_vertices(env_ids=[0], scale=True)[0]
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    if vertices.ndim != 2 or vertices.shape[-1] != 3 or vertices.numel() == 0:
        raise ValueError("CoordinatedPickment mesh_vertices must have shape (N, 3).")
    left_transforms = []
    right_transforms = []
    for env_id in range(object_initial_pose.shape[0]):
        env_context = "" if object_initial_pose.shape[0] == 1 else f" in env {env_id}"
        initial_pose = object_initial_pose[env_id]
        target_pose = None if object_target_pose is None else object_target_pose[env_id]
        candidates = _coordinated_grasp_pair_candidates(
            vertices=vertices,
            object_initial_pose=initial_pose,
            object_label=object_label,
            max_grasp_separation_angle_to_world_y_degrees=(
                max_grasp_separation_angle_to_world_y_degrees
            ),
            env=env,
            device=device,
            env_id=env_id,
        )
        if not candidates:
            if max_grasp_separation_angle_to_world_y_degrees is not None:
                raise ValueError(
                    "No CoordinatedPickment grasp candidate satisfies the configured "
                    f"world-Y separation-angle constraint in env {env_id}."
                )
            raise ValueError(
                f"No CoordinatedPickment grasp candidate was generated in env {env_id}."
            )
        candidates = _filter_coordinated_payload_collision_candidates(
            candidates,
            payload_uids=payload_uids,
            object_initial_pose=initial_pose,
            env=env,
            device=device,
            env_id=env_id,
        )
        if not candidates:
            raise ValueError(
                "No CoordinatedPickment grasp candidate avoids the declared "
                f"payloads in env {env_id}."
            )
        selected = _select_ik_feasible_coordinated_grasp_pair(
            candidates,
            object_initial_pose=initial_pose,
            object_target_pose=target_pose,
            pre_grasp_distance=pre_grasp_distance,
            lift_height=lift_height,
            sample_interval=sample_interval,
            hand_interp_steps=hand_interp_steps,
            hold_steps=hold_steps,
            object_motion_keyframes=object_motion_keyframes,
            env=env,
            device=device,
            env_id=env_id,
        )
        if selected is not None:
            if max_grasp_separation_angle_to_world_y_degrees is not None:
                selected_angle = _coordinated_grasp_pair_world_y_angle_degrees(
                    selected,
                    object_initial_pose=initial_pose,
                )
                if selected_angle > 1e-3:
                    log_warning(
                        "Exact world-Y CoordinatedPickment grasp is unavailable"
                        f"{env_context}; using {selected_angle:.1f} degrees within "
                        "the configured "
                        f"{max_grasp_separation_angle_to_world_y_degrees:.1f}-degree "
                        "limit."
                    )
            elif selected.axis_kind != "long_axis":
                log_warning(
                    "Preferred long-axis CoordinatedPickment grasp is unavailable"
                    f"{env_context}; using {selected.axis_kind} fallback."
                )
        else:
            if _has_coordinated_ik_api(env):
                if max_grasp_separation_angle_to_world_y_degrees is not None:
                    raise ValueError(
                        "No IK-feasible CoordinatedPickment grasp candidate satisfies "
                        "the configured world-Y separation-angle constraint in "
                        f"env {env_id}."
                    )
                log_warning(
                    "No IK-feasible CoordinatedPickment grasp candidate found in "
                    f"env {env_id}; falling back to the best heuristic candidate."
                )
            selected = min(candidates, key=lambda pair: (pair.priority, pair.score))
        left_transforms.append(selected.left_object_to_eef)
        right_transforms.append(selected.right_object_to_eef)

    left_result = torch.stack(left_transforms)
    right_result = torch.stack(right_transforms)
    if not input_was_batched:
        return left_result[0], right_result[0]
    return left_result, right_result
