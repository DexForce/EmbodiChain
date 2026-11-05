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

"""Per-query native proposal views; no task state or skill registry replacement."""

from __future__ import annotations
from typing import Any
import torch
from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
from embodichain.toolkits.graspkit.pg_grasp import AntipodalGraspPoseGenerator
from embodichain.toolkits.graspkit.pg_grasp.proposals import (
    unqualified_center_proposals,
)
from embodichain.lab.sim.atomic_actions.source_native_contact import NativeClosingQuery

__all__: list[str] = []


class NativeGraspView(ParallelJawGraspPoseGenerator):
    def __init__(
        self, delegate: ParallelJawGraspPoseGenerator, query: NativeClosingQuery
    ) -> None:
        super().__init__(delegate.gripper_model)
        self._delegate, self._query = delegate, query

    @property
    def native_proposal_limit(self) -> int | None:
        return (
            self._delegate.algorithm_cfg.max_candidates
            if self._query.revision in (8, 9)
            else None
        )

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices,
        mesh_triangles,
        obj_poses,
        approach_direction,
        obj_longest_axis=None,
        is_positive_part=True,
        axis_region_selector=None,
    ):
        if (
            self._query.revision in (8, 9)
            and obj_longest_axis is None
            and axis_region_selector is None
        ):
            if not isinstance(self._delegate, AntipodalGraspPoseGenerator):
                raise ValueError(
                    "Deferred native proposals require an Antipodal provider."
                )
            rows = unqualified_center_proposals(
                self._delegate,
                mesh_vertices=mesh_vertices,
                mesh_triangles=mesh_triangles,
                obj_poses=obj_poses,
                approach_direction=approach_direction,
            )
            self._query.record_deferred_proposal()
        else:
            rows = self._delegate.get_valid_grasp_poses(
                mesh_vertices=mesh_vertices,
                mesh_triangles=mesh_triangles,
                obj_poses=obj_poses,
                approach_direction=approach_direction,
                obj_longest_axis=obj_longest_axis,
                is_positive_part=is_positive_part,
                **(
                    {"axis_region_selector": axis_region_selector}
                    if axis_region_selector is not None
                    else {}
                ),
            )
        if obj_poses.shape != (len(rows), 4, 4):
            raise ValueError("Native proposal rows must match observed object poses.")
        filtered = []
        for row, (poses, costs) in enumerate(rows):
            if (
                poses.ndim != 3
                or poses.shape[1:] != (4, 4)
                or costs.shape != poses.shape[:1]
            ):
                raise ValueError(
                    "Native proposals require matching pose and cost rows."
                )
            costs = costs.to(poses.device)
            valid = torch.isfinite(costs)
            filtered.append(
                (poses[valid], costs[valid])
                if valid.any()
                else (
                    (
                        poses[:1] if len(poses) else obj_poses[row : row + 1].to(poses)
                    ).clone(),
                    poses.new_full((1,), torch.inf),
                )
            )
        return filtered

    def get_best_grasp_poses(self, **kwargs: Any):
        return self._delegate.get_best_grasp_poses(**kwargs)

    def get_grasp_candidates(self, **kwargs: Any):
        return self._delegate.get_grasp_candidates(**kwargs)

    def get_dual_arm_valid_grasp_poses(self, **kwargs: Any):
        return self._delegate.get_dual_arm_valid_grasp_poses(**kwargs)
