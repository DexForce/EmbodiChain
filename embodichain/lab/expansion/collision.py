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
        num_envs=env.num_envs,
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

    def prepare(self, snapshot=None, *, env_ids=None):
        from embodichain.lab.task_program.integrations.simulation.feasibility_geometry import (
            collision_bodies_from_simulation,
        )

        for row in range(self.env.num_envs):
            if collision_bodies_from_simulation(self.env, env_id=row) != self._geometry:
                raise ValueError(
                    "cohort collision geometry differs; create a compatible world"
                )
        ids = tuple(self.monitors) if env_ids is None else tuple(env_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(row not in self.monitors for row in ids)
        ):
            raise ValueError("prepare requires unique physical monitor rows")
        for row in ids:
            self.results.pop(row, None)
            self.initial.pop(row, None)
            self.terminal.pop(row, None)
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
        for row in ids:
            self.monitors[row].reset(measured)
        with MeasuredTaskMonitor.batch_scope(
            measured,
            tuple(self.monitors.values()),
            extra_conditions=self.initial_conditions,
        ):
            for row in ids:
                monitor = self.monitors[row]
                self.initial[row] = all(
                    monitor.predicate(measured, c) for c in self.initial_conditions
                ) and not all(
                    monitor.predicate(measured, c) for c in monitor.final_checks
                )

    def sample(self, ids, action, snapshot=None):
        env = self.env
        measured = (
            env if snapshot is None else _measured_frame(env, snapshot, self.shapes)
        )
        poses = env.robot.body_data.body_link_pose
        matrices = torch.eye(4, device=poses.device, dtype=poses.dtype).repeat(
            env.num_envs, len(env.robot.link_names), 1, 1
        )
        matrices[..., :3, :3] = matrix_from_quat(poses[..., 3:])
        matrices[..., :3, 3] = poses[..., :3]
        matrices = matrices.detach().cpu().numpy()
        frames = {
            f"{env.robot.uid}/{name}": matrices[:, index]
            for index, name in enumerate(env.robot.link_names)
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
        frames["default_plane"] = np.broadcast_to(np.eye(4), (env.num_envs, 4, 4))
        with MeasuredTaskMonitor.batch_scope(measured, tuple(self.monitors.values())):
            for row in ids:
                self.monitors[row].observe(measured, action)
                self.terminal[row] = self.monitors[row].validation(measured)
        groups = {}
        for row in ids:
            metadata = action.metadata["row_actions"][str(row)]
            phase = self.phase(env, SimpleNamespace(metadata=metadata), None)
            groups.setdefault(tuple(phase.allowed_contacts), []).append(row)
        for allowed, rows in groups.items():
            reports = self.checker.check_batch(
                frames,
                allowed_contacts=allowed,
                moving_objects=self.moving,
                env_ids=rows,
            )
            for row, report in zip(rows, reports):
                if row not in self.results or report.status != "passed":
                    self.results[row] = report

    def close(self):
        self.checker.close()


class _SampledRigidCollision:
    """Measured box/plane adapter using the existing full-robot collision checker."""

    def __init__(self, env, profile, output):
        from .contracts import ValidationCheck, ValidationResult
        from embodichain.lab.sim.motion.expansion.real2sim_collision import (
            BoxCollisionBody,
        )

        self.config = _decode_sampled_rigid_profile(load_config(profile))
        self.env = env
        self.monitors = {
            row: MeasuredTaskMonitor(
                self.config["final_checks"], self.config["events"], env_id=row
            )
            for row in range(env.num_envs)
        }
        self.bodies = tuple(
            BoxCollisionBody.from_rigid_object(env.sim.get_rigid_object(uid))
            for uid in env.sim.get_rigid_object_uid_list()
        )
        if (
            env.sim.get_articulation_uid_list()
            or env.sim.get_deformable_object_uid_list()
            or env.sim.get_rigid_object_group_uid_list()
        ):
            raise ValueError(
                "sampled rigid validation cannot omit unsupported scene entities"
            )
        self.checkers = ()
        self.initial, self.terminal, self.results = {}, {}, {}

    def prepare(self, snapshot=None):
        del snapshot
        from embodichain.lab.sim.motion.expansion.real2sim_collision import (
            BoxCollisionBody,
            PlaneCollisionBody,
            SampledRobotCollisionChecker,
        )

        self.initial.clear()
        self.terminal.clear()
        self.results.clear()
        for row, monitor in self.monitors.items():
            for body in self.bodies:
                actual = BoxCollisionBody.from_rigid_object(
                    self.env.sim.get_rigid_object(body.uid), env_id=row
                )
                if actual.geometry_sha256 != body.geometry_sha256:
                    raise ValueError("physical row collision geometry differs")
            monitor.reset(self.env)
            self.initial[row] = all(
                monitor.predicate(self.env, item)
                for item in self.config["initial_checks"]
            ) and not all(
                monitor.predicate(self.env, item)
                for item in self.config["final_checks"]
            )

        self.checkers = tuple(
            SampledRobotCollisionChecker(
                self.env.robot.cfg.fpath,
                joint_names=self.env.robot.joint_names,
                bodies=(*self.bodies, PlaneCollisionBody("ground", np.eye(4))),
                world_from_base=self.env.robot.get_local_pose(to_matrix=True)[row]
                .detach()
                .cpu()
                .numpy(),
            )
            for row in range(self.env.num_envs)
        )

    def sample(self, ids, action, snapshot=None):
        from embodichain.lab.sim.motion.expansion.real2sim_collision import (
            CollisionPhase,
        )
        from types import SimpleNamespace

        contacts = {
            tuple(item["pair"]): item["depth_m"]
            for item in self.config["allowed_contacts"]
        }
        for row in ids:
            self.monitors[row].observe(self.env, action)
            self.terminal[row] = self.monitors[row].validation(self.env)
            qpos = self.env.robot.get_qpos()[row : row + 1].detach().cpu().numpy()
            poses = {
                body.uid: self.env.sim.get_rigid_object(body.uid)
                .get_local_pose(to_matrix=True)[row : row + 1]
                .detach()
                .cpu()
                .numpy()
                for body in self.bodies
            }
            poses["ground"] = np.eye(4)[None]
            report = self.checkers[row].validate(
                qpos,
                body_poses=poses,
                phases=(CollisionPhase("measured", 0, 1, contacts),),
            )
            if row not in self.results or not report.accepted:
                self.results[row] = SimpleNamespace(
                    status="passed" if report.accepted else "failed",
                    reason=f"{report.reason}; pair={report.first_failure_pair}; penetration_m={report.first_failure_penetration_m}",
                )

    def close(self):
        """Release adapter-owned CPU geometry; the host owns the World."""
        self.checkers = ()


def _decode_sampled_rigid_profile(value):
    if (
        type(value) is not dict
        or set(value)
        != {"kind", "initial_checks", "final_checks", "events", "allowed_contacts"}
        or value["kind"] != "sampled_rigid_v1"
    ):
        raise ValueError("invalid sampled_rigid_v1 collision profile")
    if not value["initial_checks"] or not value["final_checks"]:
        raise ValueError(
            "sampled rigid validation requires independent entry and terminal predicates"
        )
    monitor = MeasuredTaskMonitor(value["final_checks"], value["events"])
    for item in value["initial_checks"]:
        monitor._validate(item)
    from embodichain.lab.sim.motion.expansion.real2sim_collision import CollisionPhase

    contacts = {}
    for item in value["allowed_contacts"]:
        if type(item) is not dict or set(item) != {"pair", "depth_m"}:
            raise ValueError("allowed contacts require exact pair and depth_m")
        pair = tuple(sorted(item["pair"]))
        if pair in contacts:
            raise ValueError("duplicate allowed contact pair")
        contacts[pair] = item["depth_m"]
    CollisionPhase("profile", 0, 1, contacts)
    return value


def _create_measured_collision(env, profile, output):
    value = load_config(profile)
    if isinstance(value, dict) and value.get("kind") == "sampled_rigid_v1":
        return _SampledRigidCollision(env, profile, output)
    return _MeasuredCollision(env, profile, output)
