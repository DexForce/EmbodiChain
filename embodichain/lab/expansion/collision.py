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

"""Shared scene-feasibility collision checks on measured cohort states."""

from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from embodichain.utils.math import matrix_from_quat
from embodichain.lab.task_program.integrations.feasibility import (
    build_feasibility_host,
    MeasuredTaskMonitor,
)
from embodichain.utils.utility import load_config

__all__: list[str] = []


class _MeasuredFrameBody:
    def __init__(self, states, shapes):
        self.body_state = torch.tensor(
            [(*state.pose, *state.velocity) for state in states], dtype=torch.float32
        )
        self._poses = torch.eye(4).repeat(len(states), 1, 1)
        self._poses[:, :3, :3] = matrix_from_quat(self.body_state[:, 3:7])
        self._poses[:, :3, 3] = self.body_state[:, :3]
        self.shapes = shapes

    def get_local_pose(self, *, to_matrix=False):
        return self._poses if to_matrix else self.body_state[:, :7]

    def get_collision_shapes(self, env_id=0):
        return self.shapes[env_id]


def _measured_frame(env, snapshot, shapes):
    if snapshot.env_ids != tuple(range(env.num_envs)):
        raise ValueError("measured validation requires a complete World snapshot")
    entities = {state.uid: [] for state in snapshot.batches[0]}
    for batch in snapshot.batches:
        for state in batch:
            entities[state.uid].append(state)
    bodies = {
        uid: _MeasuredFrameBody(states, shapes[uid])
        for uid, states in entities.items()
        if states[0].kind == "rigid_object"
    }
    qpos = torch.tensor(
        [state.qpos for state in entities[env.robot.uid]], dtype=torch.float32
    )
    return SimpleNamespace(
        robot=SimpleNamespace(joint_names=env.robot.joint_names, get_qpos=lambda: qpos),
        sim=SimpleNamespace(
            get_rigid_object_uid_list=lambda: tuple(bodies),
            get_rigid_object=lambda uid: bodies[uid],
        ),
    )


class _MeasuredCollision:
    def __init__(self, env, profile: str, output: Path):
        _, host = build_feasibility_host(
            load_config(profile),
            config_directory=Path(profile).parent,
            output_directory=output,
        )
        self.checker = host._collision_factory(env)
        from embodichain.lab.task_program.integrations.simulation.feasibility_geometry import (
            collision_bodies_from_simulation,
        )

        self._geometry = collision_bodies_from_simulation(env)
        config = load_config(profile)["task_program"]
        self.monitors = {
            row: MeasuredTaskMonitor(
                config["final_checks"], config.get("events", ()), env_id=row
            )
            for row in range(env.num_envs)
        }
        self.initial_conditions = tuple(config["initial_semantics"]) + tuple(
            config["initial_geometry"]
        )
        self.initial = {}
        self.terminal = {}
        self.phase = host._runtime_phase
        self.env = env
        self.moving = tuple(
            uid
            for uid in env.sim.get_rigid_object_uid_list()
            if env.sim.get_rigid_object(uid).body_type == "dynamic"
        )
        self.results = {}

    def prepare(self, snapshot=None):
        from embodichain.lab.task_program.integrations.simulation.feasibility_geometry import (
            collision_bodies_from_simulation,
        )

        for row in range(self.env.num_envs):
            if collision_bodies_from_simulation(self.env, env_id=row) != self._geometry:
                raise ValueError(
                    "cohort collision geometry differs; create a compatible world"
                )
        self.results.clear()
        self.initial.clear()
        self.terminal.clear()
        self.shapes = {
            uid: tuple(
                self.env.sim.get_rigid_object(uid).get_collision_shapes(env_id=row)
                for row in range(self.env.num_envs)
            )
            for uid in self.env.sim.get_rigid_object_uid_list()
        }
        measured = (
            self.env
            if snapshot is None
            else _measured_frame(self.env, snapshot, self.shapes)
        )
        for row, monitor in self.monitors.items():
            monitor.reset(measured)
            self.initial[row] = all(
                monitor.predicate(measured, c) for c in self.initial_conditions
            ) and not all(monitor.predicate(measured, c) for c in monitor.final_checks)

    def sample(self, ids, action, snapshot=None):
        env = self.env
        measured = (
            env if snapshot is None else _measured_frame(env, snapshot, self.shapes)
        )
        frames = {
            f"{env.robot.uid}/{name}": env.robot.get_link_pose(
                link_name=name, to_matrix=True
            )
            .detach()
            .cpu()
            .numpy()
            for name in env.robot.link_names
        }
        frames.update(
            {
                uid: measured.sim.get_rigid_object(uid)
                .get_local_pose(to_matrix=True)
                .detach()
                .cpu()
                .numpy()
                for uid in env.sim.get_rigid_object_uid_list()
            }
        )
        for row in ids:
            self.monitors[row].observe(measured, action)
            self.terminal[row] = self.monitors[row].validation(measured)
            metadata = action.metadata["row_actions"][str(row)]
            phase = self.phase(env, SimpleNamespace(metadata=metadata), None)
            row_frames = {key: value[row] for key, value in frames.items()}
            row_frames["default_plane"] = np.eye(4)
            report, _ = self.checker.check(
                row_frames,
                allowed_contacts=phase.allowed_contacts,
                moving_objects=self.moving,
            )
            if row not in self.results or report.status != "passed":
                self.results[row] = report

    def close(self):
        self.checker.close()
