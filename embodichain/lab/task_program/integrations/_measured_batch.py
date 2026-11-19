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


"""Batch measured predicate inputs while retaining native event state machines."""

from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np

__all__: list[str] = []


class _MeasuredChecks:
    """Read one packed physical boundary instead of synchronizing per predicate.

    Native monitors still own ordered events, consecutive counts, receipts and
    terminal reports. Only their predicate arithmetic is shared across rows;
    unsupported fast paths continue through the original native predicate.
    The cache exists solely inside an explicit measured-boundary scope.
    """

    def __init__(self, env, configuration, monitors):
        self.env, self.monitors = env, monitors
        conditions = [*configuration["initial_checks"], *configuration["final_checks"]]
        for event in configuration["events"]:
            conditions.extend(event["conditions"])
        self.conditions = {self.key(c): c for c in conditions}
        self.entities = tuple(
            dict.fromkeys(
                c[name] for c in conditions for name in ("entity", "other") if name in c
            )
        )
        self.values = None
        self.reads = 0

    @staticmethod
    def key(condition):
        return json.dumps(condition, sort_keys=True, separators=(",", ":"))

    @contextmanager
    def boundary(self):
        import torch

        env = self.env
        parts = []
        for uid in self.entities:
            obj = env.sim.get_rigid_object(uid)
            if obj is None:
                raise ValueError("Predicate entity is absent from the physical scene")
            parts.append(
                torch.cat(
                    (
                        obj.get_local_pose()[:, :3],
                        obj.body_state[:, 7:10].norm(dim=-1, keepdim=True),
                        obj.body_state[:, 10:13].norm(dim=-1, keepdim=True),
                    ),
                    dim=-1,
                )
            )
        parts.append(env.robot.get_qpos())
        # One D2H boundary, independent of the number of predicates and rows.
        packed = torch.cat(parts, dim=-1).detach().cpu().numpy()
        self.reads += 1
        states = {
            uid: packed[:, index * 5 : (index + 1) * 5]
            for index, uid in enumerate(self.entities)
        }
        joints = packed[:, len(self.entities) * 5 :]
        values = {}
        for key, condition in self.conditions.items():
            kind = condition["kind"]
            if kind == "joint_range":
                index = env.robot.joint_names.index(condition["joint"])
                value = joints[:, index].astype(np.float64)
                result = (value >= condition["lower"]) & (value <= condition["upper"])
            elif kind in ("pose_near", "lifted", "speed_below", "axis_relation"):
                state = states[condition["entity"]]
                position = state[:, :3].astype(np.float64)
                if kind == "pose_near":
                    result = (
                        np.linalg.norm(position - condition["position"], axis=-1)
                        <= condition["tolerance_m"]
                    )
                elif kind == "lifted":
                    initial = np.full(env.num_envs, np.nan)
                    for row, monitor in self.monitors.items():
                        initial[row] = monitor.initial_heights.get(
                            condition["entity"], np.nan
                        )
                    result = position[:, 2] - initial >= condition["height_m"]
                elif kind == "axis_relation":
                    axis = condition["axis"]
                    delta = position[:, axis] - states[condition["other"]][:, axis]
                    result = (delta >= condition["lower"]) & (
                        delta <= condition["upper"]
                    )
                else:
                    # Native scalar comparisons convert the float32 norms to
                    # Python floats before comparing authored thresholds.
                    result = (
                        state[:, 3].astype(np.float64) <= condition["linear_m_s"]
                    ) & (state[:, 4].astype(np.float64) <= condition["angular_rad_s"])
            else:
                continue
            values[key] = ~result if condition.get("negate", False) else result
        previous = {
            row: getattr(monitor, "_batch_predicates", None)
            for row, monitor in self.monitors.items()
        }
        for monitor in self.monitors.values():
            monitor._batch_predicates = values
        try:
            yield
        finally:
            for row, monitor in self.monitors.items():
                monitor._batch_predicates = previous[row]
