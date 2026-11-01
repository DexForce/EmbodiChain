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
from embodichain.lab.task_program.integrations.feasibility import (
    build_feasibility_host,
    MeasuredTaskMonitor,
)
from embodichain.utils.utility import load_config

__all__: list[str] = []


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

    def prepare(self):
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
        for row, monitor in self.monitors.items():
            monitor.reset(self.env)
            self.initial[row] = all(
                monitor.predicate(self.env, c) for c in self.initial_conditions
            ) and not all(monitor.predicate(self.env, c) for c in monitor.final_checks)

    def sample(self, ids, action):
        env = self.env
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
                uid: env.sim.get_rigid_object(uid)
                .get_local_pose(to_matrix=True)
                .detach()
                .cpu()
                .numpy()
                for uid in env.sim.get_rigid_object_uid_list()
            }
        )
        for row in ids:
            self.monitors[row].observe(env, action)
            self.terminal[row] = self.monitors[row].validation(env)
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
