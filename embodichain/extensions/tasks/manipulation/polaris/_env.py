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

"""PolaRiS DROID evaluation in six captured scenes; scores are per-episode latches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.utils.math import matrix_from_quat

# Object, target, minimum height of the object's collision geometry above the table.
PLACEMENTS = {
    "move_latte_cup": (("latte_art_cup", "cutting_board", 0.04),),
    "food_bussing": (
        ("ice_cream", "yellow_bowl", 0.06),
        ("grapes", "yellow_bowl", 0.06),
    ),
    "pan_clean": (("sponge", "frying_pan", 0.03),),
    "organize_tools": (("scissor", "container_00", 0.04),),
    "tape_into_container": (("tape_00", "container_02", 0.04),),
    "block_stack_kitchen": (
        ("green_cube", "green_tray", 0.03),
        ("wood_cube", "green_tray", 0.03),
    ),
}


def _within_xy(
    source: torch.Tensor, target: torch.Tensor, fraction: float
) -> torch.Tensor:
    # (arena, lower/upper, xyz), at each object's current orientation.
    overlap = (
        torch.minimum(source[:, 1, :2], target[:, 1, :2])
        - torch.maximum(source[:, 0, :2], target[:, 0, :2])
    ).clamp_min(0)
    area = (source[:, 1, :2] - source[:, 0, :2]).prod(-1)
    return overlap.prod(-1) >= fraction * area


class PolarisRubric:
    """Latch each physical criterion once, in dependency order within a control step."""

    def __init__(self, scene: str, num_envs: int, device: torch.device):
        self.placements = PLACEMENTS[scene]
        self.stacking = scene == "block_stack_kitchen"
        self.reached = torch.zeros(
            (num_envs, 3 * len(self.placements) + int(self.stacking)),
            device=device,
            dtype=torch.bool,
        )

    def reset(self, env_ids) -> None:
        self.reached[env_ids] = False

    def update(self, gripper_position, finger_angle, positions, bounds) -> None:
        opened = finger_angle.abs() < 0.1
        for index, (obj, target, height) in enumerate(self.placements):
            reached, lifted, placed = self.reached[:, 3 * index : 3 * index + 3].unbind(
                1
            )
            reached.logical_or_(
                torch.linalg.vector_norm(gripper_position - positions[obj], dim=-1)
                < 0.2
            )
            lifted.logical_or_(reached & (bounds[obj][:, 0, 2] > height))
            placed.logical_or_(
                lifted & opened & _within_xy(bounds[obj], bounds[target], 0.8)
            )
        if self.stacking:
            self.reached[:, 6] |= (
                self.reached[:, 2]
                & self.reached[:, 5]
                & opened
                & _within_xy(bounds["green_cube"], bounds["wood_cube"], 0.5)
            )


class PolarisDroidEnv(EmbodiedEnv):
    """DROID joint-position policy environment with an independent measured rubric."""

    scene: ClassVar[str]
    _gripper_link = "robotiq_arg2f_base_link"

    def _initialize_physical_objective(self) -> None:
        super()._initialize_physical_objective()
        scene = self.scene
        self.rubric = PolarisRubric(scene, self.num_envs, self.device)
        self._scored_names = {
            name for obj, target, _ in self.rubric.placements for name in (obj, target)
        }
        self._objects = {
            cfg.uid: self.sim.get_rigid_object(cfg.uid) for cfg in self.cfg.rigid_object
        }
        bundle = "PolarisDroid" + "".join(word.title() for word in scene.split("_"))
        source = Path(get_data_path(f"{bundle}/initial_conditions.json"))
        rows = json.loads(source.read_text())["poses"]
        names = (self.cfg.extensions or {}).get("initial_condition_names", {})
        self._pose_sets = {}
        for name in self._objects:
            upstream = names.get(name, name)
            if upstream in rows[0]:
                poses = torch.tensor(
                    [row[upstream] for row in rows],
                    device=self.device,
                    dtype=torch.float32,
                )
                # Isaac Lab stores wxyz; DexSim accepts xyzw.
                self._pose_sets[name] = poses[:, [0, 1, 2, 4, 5, 6, 3]]
        self._pose_count = len(rows)
        if self._gripper_link not in self.robot.link_names:
            raise ValueError(
                f"PolaRiS requires {self._gripper_link!r}; available links: {self.robot.link_names}"
            )
        self._finger_joint = self.robot.joint_names.index("finger_joint")
        self._vertices = {}
        for name, obj in self._objects.items():
            vertices = []
            for shape in obj.get_collision_shapes():
                if shape.vertices is None:
                    raise ValueError(
                        f"PolaRiS expects cooked mesh collision geometry for {name!r}."
                    )
                rotation = shape.local_pose[:3, :3]
                vertices.append(shape.vertices @ rotation.T + shape.local_pose[:3, 3])
            if not vertices:
                raise ValueError(f"PolaRiS object {name!r} has no collision shapes.")
            self._vertices[name] = torch.cat(vertices).to(self.device, torch.float32)

    def _initialize_episode(self, env_ids, **kwargs):
        super()._initialize_episode(env_ids, **kwargs)
        index = kwargs.get(
            "initial_condition", 0 if self.cfg.task_program is not None else None
        )
        if index is not None and (
            type(index) is not int or not 0 <= index < self._pose_count
        ):
            raise ValueError(
                f"initial_condition must be an integer in [0, {self._pose_count}); got {index!r}."
            )
        for name, obj in self._objects.items():
            if index is not None and name in self._pose_sets:
                pose = self._pose_sets[name][index].expand(len(env_ids), -1).clone()
            else:
                pose = obj.get_local_pose()[env_ids].clone()
            # Bring the configured orientation down to the tabletop using the same
            # cooked geometry as the lift check, without advancing other arenas.
            rotation = matrix_from_quat(pose[:, 3:])
            bottom = (
                (self._vertices[name] @ rotation[:, 2, :, None]).amin(dim=1).squeeze(-1)
            )
            pose[:, 2] = -bottom
            obj.set_local_pose(pose, env_ids=env_ids)

    def _update_physical_objective(self) -> None:
        super()._update_physical_objective()
        positions, bounds = {}, {}
        for name in self._scored_names:
            obj = self._objects[name]
            pose = obj.get_local_pose(to_matrix=True)
            positions[name] = pose[:, :3, 3]
            points = self._vertices[name] @ pose[:, :3, :3].transpose(1, 2)
            bounds[name] = (
                torch.stack((points.amin(1), points.amax(1)), dim=1)
                + positions[name][:, None]
            )
        self.rubric.update(
            self.robot.get_link_pose(self._gripper_link)[:, :3],
            self.robot.get_qpos()[:, self._finger_joint],
            positions,
            bounds,
        )

    def _reset_physical_objective(self, env_ids) -> None:
        super()._reset_physical_objective(env_ids)
        self.rubric.reset(env_ids)

    def compute_task_state(self, **kwargs):
        reached = self.rubric.reached
        success = self.is_task_success(**kwargs)
        return (
            success,
            torch.zeros_like(success),
            {"progress": reached.float().mean(dim=1), "criteria": reached.clone()},
        )

    def is_task_success(self, **kwargs) -> torch.Tensor:
        measured = self.rubric.reached.all(dim=1)
        if self.cfg.task_program is not None:
            return measured & super().is_task_success(**kwargs)
        return measured
