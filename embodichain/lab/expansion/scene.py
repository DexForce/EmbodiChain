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

"""Preserve scene/workspace joint candidates through native cohort preparation."""

from __future__ import annotations
from pathlib import Path
from collections.abc import Mapping
import torch
from embodichain.lab.sim.scene.feasibility import (
    JointSceneCandidate,
    load_feasibility_atlas,
)
from embodichain.lab.sim.scene import SceneView
from embodichain.lab.sim.scene.operations import SceneChangeSet, apply_changes_batch

__all__ = ["JointSceneSource", "apply_joint_scene_candidates"]


class JointSceneSource:
    """Read immutable, jointly sampled layouts and robot installations.

    Args:
        path: Content-verified #912 feasibility atlas. Completed observations
            provide proposals; their past success never substitutes new execution.
    """

    def __init__(self, path: str | Path) -> None:
        atlas = load_feasibility_atlas(path)
        self.candidates = tuple(
            observation.candidate
            for observation in atlas.observations
            if observation.completed
        )
        if not self.candidates:
            raise ValueError("scene atlas has no completed candidate observations")
        self.request_id = atlas.request.request_id

    def candidate(self, ordinal: int) -> JointSceneCandidate:
        """Select a complete joint proposal without independently mixing its axes."""
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("ordinal must be nonnegative")
        return self.candidates[ordinal % len(self.candidates)]


def apply_joint_scene_candidates(
    view: SceneView, candidates: Mapping[int, JointSceneCandidate]
) -> None:
    """Apply a compatible cohort before settling and refreshing observations.

    Args:
        view: Exactly the candidate physical rows, with fixed asset topology.
        candidates: Joint proposals retaining both scene and robot pose identity.
    Raises:
        ValueError: For missing rows, assets, unsupported robots or malformed poses.
    """
    if set(candidates) != set(view.env_ids):
        raise ValueError("joint candidates must cover the selected rows")
    bindings = {uid: (kind, entity) for kind, uid, entity in view.entities()}
    roots = {}
    changes = {}
    # Validate every row and all resource identities before the first write.
    from embodichain.lab.sim.scene.operations import _preflight_changes

    for row, candidate in candidates.items():
        if not isinstance(candidate, JointSceneCandidate):
            raise TypeError("candidate must be JointSceneCandidate")
        poses = {
            change.entity_uid: change.arena_pose
            for change in candidate.scene.pose_changes
        }
        changes[row] = SceneChangeSet(poses, tuple(poses))
        _preflight_changes(SceneView(view.sim, env_ids=(row,)), changes[row])
        for uid, pose in candidate.deployment.poses.items():
            kind, robot = bindings.get(uid, (None, None))
            if kind != "robot" or not callable(getattr(robot, "set_local_pose", None)):
                raise ValueError(f"joint candidate robot {uid!r} is not bound")
            if not robot.cfg.root_props.fixed_base:
                raise ValueError("installation expansion requires a fixed robot base")
            group = roots.setdefault(uid, (robot, [], []))
            group[1].append(row)
            group[2].append(pose)
    apply_changes_batch(view, changes)
    for robot, rows, poses in roots.values():
        robot.set_local_pose(
            torch.tensor(poses, dtype=torch.float32, device=view.sim.device),
            env_ids=rows,
        )
    view.sync_render_state()
