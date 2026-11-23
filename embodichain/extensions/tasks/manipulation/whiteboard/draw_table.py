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

"""Draw the RoboWorld table outline using a physically gripped marker."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.sensors.contact.sensor import (
    ArticulationContactFilterCfg,
    ContactSensor,
    ContactSensorCfg,
)

__all__: list[str] = []

# The board coordinates are right/up as seen by the robot. Dimensions and
# acceptance thresholds follow the RoboWorld table task.
_LINES = (
    ("top", (-0.30, 0.18), (0.30, 0.18), 0.003),
    ("bottom", (0.30, -0.18), (-0.30, -0.18), 0.003),
    ("left", (-0.30, -0.18), (-0.30, 0.18), 0.003),
    ("right", (0.30, 0.18), (0.30, -0.18), 0.003),
    ("header", (-0.30, 0.12), (0.30, 0.12), 0.0015),
    ("column_left", (-0.10, 0.18), (-0.10, -0.18), 0.0015),
    ("column_right", (0.10, -0.18), (0.10, 0.18), 0.0015),
)
_CENTER = (0.50, 0.0, 1.28)
_GRID = 0.001
_TIP = 0.063
_TCP_FROM_MARKER_ORIGIN = 0.0355
_PRESSURE = 0.052
_WITHDRAWAL_CLEARANCE = 0.010


def _distance_to_line(points: np.ndarray, start, end) -> np.ndarray:
    start = np.asarray(start)
    delta = np.asarray(end) - start
    along = np.clip((points - start) @ delta / (delta @ delta), 0, 1)
    return np.linalg.norm(points - (start + along[:, None] * delta), axis=1)


class _Ink:
    """Store only raster cells marked by measured tip-board contacts."""

    def __init__(self) -> None:
        self.cells: set[tuple[int, int]] = set()
        self._line_samples = [
            np.linspace(
                start, end, round(np.linalg.norm(np.subtract(end, start)) / _GRID) + 1
            )
            for _, start, end, _ in _LINES
        ]
        self.last: np.ndarray | None = None
        self._measurement: tuple[list[float], list[float], float] | None = None

    def stamp(self, point: np.ndarray | None) -> None:
        if point is None:
            self.last = None
            return
        path = (point,)
        if self.last is not None:
            distance = np.linalg.norm(point - self.last)
            if distance <= 0.006:
                path = np.linspace(
                    self.last, point, max(2, int(np.ceil(distance / _GRID)) + 1)
                )
        previous_size = len(self.cells)
        for p in path:
            index = np.rint(p / _GRID).astype(int)
            for a, b in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                self.cells.add((int(index[0]) + a, int(index[1]) + b))
        if len(self.cells) != previous_size:
            self._measurement = None
        self.last = point

    def measure(self) -> tuple[list[float], list[float], float]:
        """Reuse the drawing measurements until new raster cells are stamped."""
        if self._measurement is None:
            coverage, gaps = self.line_results()
            self._measurement = (coverage, gaps, self.stray_area())
        return self._measurement

    def line_results(self) -> tuple[list[float], list[float]]:
        coverages, gaps = [], []
        for points in self._line_samples:
            hits = []
            for u, v in np.rint(points / _GRID).astype(int):
                hits.append(
                    any(
                        (u + a, v + b) in self.cells
                        for a in range(-2, 3)
                        for b in range(-2, 3)
                        if a * a + b * b <= 4
                    )
                )
            missing = np.flatnonzero(np.diff(np.r_[False, np.logical_not(hits), False]))
            gap = np.diff(missing.reshape(-1, 2), axis=1).max(initial=0) * _GRID
            coverages.append(float(np.mean(hits)))
            gaps.append(float(gap))
        return coverages, gaps

    def stray_area(self) -> float:
        if not self.cells:
            return 0.0
        centers = np.asarray(sorted(self.cells)) * _GRID
        in_band = np.zeros(len(centers), dtype=bool)
        for _, start, end, half_width in _LINES:
            in_band |= _distance_to_line(centers, start, end) <= half_width + 0.002
        return float(np.count_nonzero(~in_band) * _GRID**2)


@register_env("WhiteboardDrawTableFranka-v1", max_episode_steps=4800)
class _WhiteboardDrawTableEnv(EmbodiedEnv):
    """A single native workcell with a friction-held marker and measured ink."""

    def _initialize_physical_objective(self) -> None:
        super()._initialize_physical_objective()
        if self.num_envs != 1:
            raise ValueError("WhiteboardDrawTableFranka-v1 requires num_envs=1.")
        self._pen = self.sim.get_articulation("pen")
        self._arm = self.robot.get_joint_ids("arm")
        self._hand = self.robot.get_joint_ids("eef")
        self._sensor = ContactSensor(
            ContactSensorCfg(
                uid="whiteboard_nib_contact",
                rigid_uid_list=["board"],
                articulation_cfg_list=[
                    ArticulationContactFilterCfg(
                        articulation_uid="pen", link_name_list=["nib"]
                    )
                ],
                filter_need_both_actor=True,
                max_contacts_per_env=32,
            ),
            self.sim.device,
            owner=self.sim,
        )
        self._ink = _Ink()
        self._pressure = 0.0
        self._previous_sample = -1
        self._ready = False
        self._withdrawn = False
        self._dropped = False
        self._last_drawing_goal = None
        self._drawing_offset = torch.zeros(2, device=self.device)
        self._writing_rotation = torch.as_tensor(
            (
                Rotation.from_euler("y", 90, degrees=True)
                * Rotation.from_euler("z", 45, degrees=True)
            ).as_matrix(),
            dtype=torch.float32,
            device=self.device,
        )

    def _initialize_episode(self, env_ids=None, **kwargs: Any) -> None:
        super()._initialize_episode(env_ids, **kwargs)
        self._ink = _Ink()
        self._previous_sample = -1
        self._withdrawn = False
        self._pressure = 0.0
        self._dropped = False
        self._last_drawing_goal = None
        self._drawing_offset.zero_()
        self._ready = True
        current = self.robot.get_qpos().clone()
        # The task starts with the uncapped pen between the closed fingers. Its
        # pose is written here, before the episode. The expert only sends robot commands.
        start = self._tip_goal((-0.032, 0.0, 0.0))
        current[:, self._arm] = self._solve(start, current[:, self._arm])
        current[:, self._hand] = 0.0076
        self.robot.set_qpos(current, env_ids=[0])
        tcp = self._tcp()
        marker = tcp.clone()
        marker[:, :3, 3] += _TCP_FROM_MARKER_ORIGIN * tcp[:, :3, 2]
        self._pen.set_local_pose(marker, env_ids=[0])

    def _tcp(self) -> torch.Tensor:
        return self.robot.compute_fk(
            self.robot.get_qpos()[:, self._arm], name="arm", to_matrix=True
        )

    def _tip_goal(self, at: tuple[float, float, float]) -> torch.Tensor:
        depth, u, v = at
        pose = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0)
        pose[:, :3, :3] = self._writing_rotation
        tip = pose.new_tensor((_CENTER[0] + depth, _CENTER[1] - u, _CENTER[2] + v))
        pose[:, :3, 3] = tip - (_TCP_FROM_MARKER_ORIGIN + _TIP) * pose[:, :3, 2]
        return pose

    def _solve(self, pose: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        valid, qpos = self.robot.compute_ik(pose, joint_seed=seed, name="arm")
        if not valid.all():
            raise ValueError(
                f"Unreachable whiteboard TCP pose {pose[0, :3, 3].tolist()}."
            )
        return qpos

    def _marker_tip(self) -> torch.Tensor:
        nib = self._pen.get_link_pose("nib", to_matrix=True)
        return nib[:, :3, 3] + 0.00375 * nib[:, :3, 2]

    def _sample_ink(self) -> None:
        step = int(self._elapsed_steps[0])
        if step == self._previous_sample:
            return
        self._previous_sample = step
        marker = self._pen.get_local_pose(to_matrix=True)[0]
        tcp = self._tcp()[0]
        axis = marker[:3, 2]
        tip = self._marker_tip()[0]
        self._withdrawn = bool(tip[0] <= _CENTER[0] - _WITHDRAWAL_CLEARANCE)
        held_origin = tcp[:3, 3] + _TCP_FROM_MARKER_ORIGIN * tcp[:3, 2]
        translation_error = (marker[:3, 3] - held_origin).norm()
        axis_alignment = axis @ tcp[:3, 2]
        self._dropped = bool(translation_error > 0.025 or axis_alignment < 0.85)
        self._sensor.update()
        sample = self._sensor.get_data()
        contacts = sample["position"][0]
        near_tip = (contacts - tip).norm(dim=-1) <= 0.012
        loaded = sample["is_valid"][0] & sample["impulse_valid"][0] & near_tip
        force = (
            sample["impulse"][0, loaded] * sample["normal"][0, loaded, 0].abs()
        ).sum()
        self._pressure = float(force) / (self.step_dt / self.cfg.sim_steps_per_control)
        writes = (
            bool(loaded.any())
            and self._pressure > 0
            and float(axis[0]) > 0.20
            and not self._dropped
        )
        point = (
            np.array((_CENTER[1] - float(tip[1]), float(tip[2]) - _CENTER[2]))
            if writes
            else None
        )
        self._ink.stamp(point)

    def _acceptance(self) -> tuple[bool, dict[str, Any]]:
        coverage, gaps, stray = self._ink.measure()
        accepted = (
            min(coverage) >= 0.9 and max(gaps) <= 0.010 and stray <= 0.0001 + 1e-12
        )
        return accepted, {
            "line_coverage": torch.tensor([coverage], device=self.device),
            "longest_line_gap": torch.tensor([gaps], device=self.device),
            "stray_ink_area": torch.tensor([stray], device=self.device),
        }

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Grade measured markings once the held tip clears the drawing surface."""
        if not getattr(self, "_ready", False):
            return super().compute_task_state(**kwargs)
        self._sample_ink()
        accepted, metrics = (
            self._acceptance() if self._withdrawn and self._ink.cells else (False, {})
        )
        metrics["nib_force"] = torch.tensor([self._pressure], device=self.device)
        return (
            torch.tensor([accepted and not self._dropped], device=self.device),
            torch.tensor([self._dropped], device=self.device),
            metrics,
        )

    def _accepted_drawing(self) -> bool:
        success, _, _ = self.compute_task_state()
        return bool(success[0])

    def _action(
        self, at: tuple[float, float, float], *, drawing: bool = False
    ) -> ControllerAction:
        qpos = self.robot.get_qpos().clone()
        pose = self._tip_goal(at)
        tcp = self._tcp()
        marker = self._pen.get_local_pose(to_matrix=True)
        tip = self._marker_tip()
        if drawing:
            if self._last_drawing_goal is not None:
                self._drawing_offset.add_(
                    0.22 * (self._last_drawing_goal - tip[0, 1:])
                ).clamp_(-0.010, 0.010)
            self._last_drawing_goal = pose.new_tensor(
                (_CENTER[1] - at[1], _CENTER[2] + at[2])
            )
            pose[0, 1:3, 3] += self._drawing_offset
        else:
            self._last_drawing_goal = None
            self._drawing_offset.zero_()
        # Measure deflection tangentially. Along the surface normal, use the
        # uncompressed marker and let the pressure loop regulate insertion.
        tip[:, 0] = marker[:, 0, 3] + _TIP * marker[:, 0, 2]
        measured_offset = tip - tcp[:, :3, 3]
        calibrated_offset = torch.bmm(
            tcp[:, :3, :3].transpose(1, 2), measured_offset[..., None]
        )
        pose[:, :3, 3] += (_TCP_FROM_MARKER_ORIGIN + _TIP) * pose[:, :3, 2]
        pose[:, :3, 3] -= torch.bmm(pose[:, :3, :3], calibrated_offset).squeeze(-1)
        qpos[:, self._arm] = self._solve(pose, qpos[:, self._arm])
        qpos[:, self._hand] = 0
        return ControllerAction(qpos)

    def _stroke(
        self, start, end, previous: tuple[float, float], last: bool
    ) -> Iterator[ControllerAction]:
        for u, v in np.linspace(
            previous,
            start,
            max(10, int(np.linalg.norm(np.subtract(start, previous)) / 0.0045)),
        ):
            yield self._action((-0.025, u, v))
        depth = -0.025
        for depth in np.linspace(depth, 0.003, 55):
            if self._pressure >= _PRESSURE * 0.7:
                break
            yield self._action((depth, *start), drawing=True)
        length = np.linalg.norm(np.subtract(end, start))
        for u, v in np.linspace(start, end, int(np.ceil(length / 0.0013)) + 1):
            depth = float(
                np.clip(
                    depth
                    + np.clip((_PRESSURE - self._pressure) * 0.0012, -0.0004, 0.0004),
                    -0.003,
                    0.004,
                )
            )
            yield self._action((depth, u, v), drawing=True)
        for depth in np.linspace(depth, -0.025, 16):
            yield self._action((depth, *end))
        if last:
            yield self._action((-0.025, *end))

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Execute one closed-loop contact trajectory per table line."""
        previous = (0.0, 0.0)
        for index, (name, start, end, _) in enumerate(_LINES):
            last = index == len(_LINES) - 1
            yield DemoSegment(
                name=name,
                target_uid="board",
                instruction=f"Draw the {name.replace('_', ' ')} line.",
                actions=self._stroke(start, end, previous, last),
                validator=self._accepted_drawing if last else None,
            )
            previous = end
