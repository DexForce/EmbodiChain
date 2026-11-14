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

"""Shared measured geometry/contact acceptance without task-owned World control."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, TYPE_CHECKING
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import torch
from embodichain.data_pipeline.recording.media import _atomic_json as atomic_json

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.types import ControllerAction
    from embodichain.lab.sim.scene import SceneState

__all__ = ["MeasuredExpansionValidator"]


class _GeometryRow:
    """Independently check measured task predicates and complete physical geometry."""

    def _initialize_task_checks(self, env, configuration, output):
        from embodichain.lab.task_program.integrations.feasibility import (
            MeasuredTaskMonitor,
        )

        self.env, self.configuration, self.output = env, configuration, output
        self.monitor = MeasuredTaskMonitor(
            configuration["final_checks"], configuration["events"], env_id=self.env_id
        )
        for condition in configuration["initial_checks"]:
            self.monitor._validate(condition)
        self.initial, self.terminal, self.results = {}, {}, {}

    @staticmethod
    def contact_permissions(env, configuration, known):
        """Resolve authored contacts and mechanically adjacent robot assemblies."""
        # Collapse fixed URDF assemblies before deciding mechanical adjacency.
        model = ET.parse(env.robot.cfg.fpath).getroot()
        joints = model.findall("joint")
        links = {link.get("name") for link in model.findall("link")}
        groups = {link: link for link in links}

        def group(link):
            while groups[link] != link:
                link = groups[link]
            return link

        for joint in joints:
            if joint.get("type") == "fixed":
                parent, child = joint.find("parent").get("link"), joint.find(
                    "child"
                ).get("link")
                groups[group(child)] = group(parent)
        adjacent = {
            frozenset(
                (
                    group(j.find("parent").get("link")),
                    group(j.find("child").get("link")),
                )
            )
            for j in joints
        }
        physical_links = sorted(
            link for link in links if f"{env.robot.uid}/{link}" in known
        )
        ignored = [
            (f"{env.robot.uid}/{left}", f"{env.robot.uid}/{right}")
            for i, left in enumerate(physical_links)
            for right in physical_links[i + 1 :]
            if group(left) == group(right)
            or frozenset((group(left), group(right))) in adjacent
        ]
        children = {j.get("name"): j.find("child").get("link") for j in joints}
        root_groups = {group(link) for link in links - set(children.values())}
        roots = {link for link in physical_links if group(link) in root_groups}

        def expand(name):
            if name == "@root":
                return [
                    f"{env.robot.uid}/{link}"
                    for link in sorted(roots)
                    if f"{env.robot.uid}/{link}" in known
                ]
            if name.startswith("@"):
                return [
                    f"{env.robot.uid}/{children[joint]}"
                    for joint in env.robot.control_parts[name[1:]]
                ]
            return [name]

        def pairs(values):
            return tuple(
                (a, b)
                for left, right in values
                for a in expand(left)
                for b in expand(right)
            )

        ignored.extend(pairs(configuration["mounting_contacts"]))
        allowed = pairs(configuration["allowed_contacts"])
        if any(not set(pair) <= known for pair in (*ignored, *allowed)):
            raise ValueError("Contact permissions reference an absent physical frame")
        # Match the physical model's authored self-collision policy. Keeping
        # robot/scene checks does not require all internal robot shape pairs.
        if env.robot.cfg.root_props.self_collision_enabled is False:
            robot_frames = sorted(
                frame for frame in known if frame.startswith(env.robot.uid + "/")
            )
            ignored.extend(
                (a, b)
                for i, a in enumerate(robot_frames)
                for b in robot_frames[i + 1 :]
            )
        return ignored, allowed

    def __init__(
        self, env, configuration: dict, output: Path, *, device="cpu", env_id=0
    ):
        self.env_id = env_id
        from embodichain.lab.task_program.integrations.simulation.feasibility_geometry import (
            collision_bodies_from_simulation,
        )
        from embodichain.lab.sim.scene.feasibility import NewtonCollisionChecker

        self._initialize_task_checks(env, configuration, output)
        bodies = collision_bodies_from_simulation(env, env_id=self.env_id)
        known = {body.frame_uid for body in bodies}
        ignored, self.allowed_contacts = self.contact_permissions(
            env, configuration, known
        )
        checker = NewtonCollisionChecker(
            bodies, ignored_pairs=ignored, device=device, max_shapes=512
        )
        self.checkers = [(checker, self.allowed_contacts, known)]
        atomic_json(
            output / "collision-geometry.json",
            {
                "shape_count": len(bodies),
                "frames": sorted(known),
                "query_count": len(self.checkers),
                "self_collision_enabled": env.robot.cfg.root_props.self_collision_enabled,
            },
        )
        self.moving = tuple(
            uid
            for uid in env.sim.get_rigid_object_uid_list()
            if env.sim.get_rigid_object(uid).body_type == "dynamic"
        )

    def prepare(self, snapshot=None):
        self.monitor.reset(self.env)
        self.initial = {
            self.env_id: all(
                self.monitor.predicate(self.env, c)
                for c in self.configuration["initial_checks"]
            )
            and not all(
                self.monitor.predicate(self.env, c)
                for c in self.configuration["final_checks"]
            )
        }
        self.terminal.clear()
        self.results.clear()

    def sample(self, ids, action, snapshot=None):
        from embodichain.utils.math import matrix_from_quat

        self.monitor.observe(self.env, action)
        self.terminal[self.env_id] = self.monitor.validation(self.env)
        # Read the measured link buffer once and convert the whole arm in one
        # batch. get_link_pose() otherwise repeats these allocations and
        # quaternion conversions separately for every physical/visual link.
        robot = self.env.robot
        poses = robot.body_data.body_link_pose[self.env_id]
        matrices = torch.eye(4, dtype=poses.dtype, device=poses.device).repeat(
            len(robot.link_names), 1, 1
        )
        matrices[:, :3, :3] = matrix_from_quat(poses[:, 3:])
        matrices[:, :3, 3] = poses[:, :3]
        matrices = matrices.detach().cpu().numpy()
        frames = {
            f"{robot.uid}/{link}": matrix
            for link, matrix in zip(robot.link_names, matrices)
        }
        frames.update(
            {
                uid: self.env.sim.get_rigid_object(uid)
                .get_local_pose(to_matrix=True)[self.env_id]
                .detach()
                .cpu()
                .numpy()
                for uid in self.env.sim.get_rigid_object_uid_list()
            }
        )
        frames["default_plane"] = np.eye(4)
        for checker, contacts, aliases in self.checkers:
            report, _ = checker.check(
                frames,
                allowed_contacts=contacts,
                moving_objects=tuple(uid for uid in self.moving if uid in aliases),
            )
            if self.env_id not in self.results or report.status != "passed":
                self.results[self.env_id] = SimpleNamespace(
                    status=report.status,
                    reason=f"{report.reason}; pair={checker.last_pair}",
                )
            if report.status != "passed":
                metadata = action.metadata.get("row_actions", {}).get(
                    str(self.env_id), action.metadata
                )
                atomic_json(
                    self.output / "collision-failure.json",
                    {
                        "reason": self.results[self.env_id].reason,
                        "action": {
                            key: str(value)
                            for key, value in metadata.items()
                            if isinstance(value, (str, int, float, bool))
                        },
                    },
                )
                break

    def close(self):
        for checker, _, _ in self.checkers:
            checker.close()
        self.checkers.clear()


class _ContactRow(_GeometryRow):
    """Inspect actual physics contacts, including every control substep.

    Unlike the independent Newton query, this backend retains native triangle
    meshes. Unknown actors, unavailable impulses and dropped rows fail closed.
    The optional final hold audit additionally requires both physical fingers
    throughout the final measured control samples and bounds object drift.
    """

    def __init__(self, env, configuration, output, *, sensor, env_id=0):
        self.env_id = env_id
        self._initialize_task_checks(env, configuration, output)
        self.active, self._bad, self.substeps, self._interval_samples = (
            False,
            None,
            0,
            0,
        )
        self.hold = configuration.get("contact_hold")
        count = (
            round(self.hold["duration_s"] * env.control_frequency) + 1
            if self.hold
            else 1
        )
        self.samples = deque(maxlen=count)
        self.sensor = sensor
        self.frames = {}
        for link, actor in zip(
            env.robot.link_names,
            self.sensor.get_actor_ids(env.robot.uid, env.robot.link_names)[
                self.env_id
            ].tolist(),
        ):
            self.frames[actor] = f"{env.robot.uid}/{link}"
        for uid in (*env.sim.get_rigid_object_uid_list(), "default_plane"):
            self.frames[int(self.sensor.get_actor_ids(uid)[self.env_id, 0])] = uid
        ignored, allowed = self.contact_permissions(
            env, configuration, set(self.frames.values())
        )
        self.allowed = {frozenset(pair) for pair in (*ignored, *allowed)}
        self.moving = {f"{env.robot.uid}/{link}" for link in env.robot.link_names} | {
            uid
            for uid in env.sim.get_rigid_object_uid_list()
            if env.sim.get_rigid_object(uid).body_type == "dynamic"
        }
        atomic_json(
            output / "collision-geometry.json",
            {
                "backend": "native_contacts",
                "geometry": "native physical shapes, including triangle meshes",
                "frames": sorted(self.frames.values()),
                "substep_sampling": True,
                "max_contacts_per_env": 8192,
                "self_collision_enabled": env.robot.cfg.root_props.self_collision_enabled,
            },
        )

    def reset(self, env_ids=None):
        self.active, self._bad, self.substeps, self._interval_samples = (
            False,
            None,
            0,
            0,
        )
        self.samples.clear()

    def prepare(self, snapshot=None):
        super().prepare(snapshot)
        self.reset()
        self.active = True

    def begin_control_step(self):
        self._bad = None
        self._interval_samples = 0

    def before_sample(self):
        return None

    def _hold_report(self):
        report = {
            "samples": len(self.samples),
            "required_samples": self.samples.maxlen,
            "physics_substeps": self.substeps,
        }
        if len(self.samples) != self.samples.maxlen:
            return {**report, "passed": False}
        poses = np.asarray([sample[0] for sample in self.samples])
        from scipy.spatial.transform import Rotation

        rotations = Rotation.from_quat(poses[:, 3:])
        report.update(
            both_fingers_samples=sum(all(sample[1]) for sample in self.samples),
            max_translation_drift_m=float(
                np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1).max()
            ),
            max_rotation_drift_rad=float(
                (rotations[0].inv() * rotations).magnitude().max()
            ),
        )
        report["passed"] = (
            report["both_fingers_samples"] == self.samples.maxlen
            and report["max_translation_drift_m"]
            <= self.hold["max_translation_drift_m"]
            and report["max_rotation_drift_rad"] <= self.hold["max_rotation_drift_rad"]
        )
        return report

    def sample(self, ids, action, snapshot=None):
        from embodichain.lab.expansion.contracts import (
            ValidationCheck,
            ValidationResult,
        )

        if not self._interval_samples:
            self._bad = "native contact substeps were not sampled"
        if self.env_id not in self.results or self._bad:
            self.results[self.env_id] = SimpleNamespace(
                status="failed" if self._bad else "passed",
                reason=self._bad or "all measured native contacts are permitted",
            )
        if self._bad:
            atomic_json(self.output / "collision-failure.json", {"reason": self._bad})
        self.monitor.observe(self.env, action)
        terminal = self.monitor.validation(self.env)
        if self.hold:
            data = self.sensor.get_data()
            pairs = data["user_ids"][self.env_id][
                data["is_valid"][self.env_id] & data["impulse_valid"][self.env_id]
            ]
            book = int(self.sensor.get_actor_ids(self.hold["entity"])[self.env_id, 0])
            fingers = self.sensor.get_actor_ids(self.env.robot.uid, self.hold["links"])[
                self.env_id
            ].tolist()
            touching = [
                bool(
                    ((pairs == book).any(dim=-1) & (pairs == finger).any(dim=-1)).any()
                )
                for finger in fingers
            ]
            pose = (
                self.env.sim.get_rigid_object(self.hold["entity"])
                .get_local_pose()[self.env_id]
                .tolist()
            )
            self.samples.append((pose, touching))
            report = (
                self._hold_report()
                if terminal.accepted
                else {"passed": False, "reason": "awaiting terminal hold"}
            )
            terminal = ValidationResult(
                (
                    *terminal.checks,
                    ValidationCheck(
                        "task.contact_hold",
                        "passed" if report["passed"] else "failed",
                        str(report),
                    ),
                )
            )
            if terminal.accepted:
                atomic_json(self.output / "contact-hold.json", report)
        self.terminal[self.env_id] = terminal

    def close(self):
        if self.hold and self.samples:
            obj = self.env.sim.get_rigid_object(self.hold["entity"])
            atomic_json(
                self.output / "terminal-diagnostics.json",
                {
                    "pose": obj.get_local_pose()[self.env_id].tolist(),
                    "body_state": obj.body_state[self.env_id].tolist(),
                    "predicates": [
                        {
                            "condition": condition,
                            "passed": self.monitor.predicate(self.env, condition),
                        }
                        for condition in self.configuration["final_checks"]
                    ],
                    "hold": self._hold_report(),
                    "events": self.monitor.receipts,
                },
            )
        self.active = False


class MeasuredExpansionValidator:
    """Use one measured acceptance port for geometry or complete native contacts.

    Args:
        environment: Caller-owned native Gym environment.
        configuration: Task-owned initial_checks/final_checks/events, declared
            allowed_contacts/mounting_contacts, and optional contact_hold with
            entity, finger links, duration and translation/rotation drift limits.
        output: Artifact directory for diagnostic and terminal evidence.
        backend: geometry (independent Newton query) or native_contacts (actual
            physical substeps including triangle meshes). Unavailable data rejects.
        device: Numerical geometry-query device; never changes physics.

    The World, stepping and acceptance thresholds remain with the host/task.
    Selected-row preparation leaves live peers' monitors and evidence intact.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        configuration: Mapping[str, Any],
        output: str | Path,
        *,
        backend: str = "geometry",
        device: str = "cpu",
    ) -> None:
        if backend not in ("geometry", "native_contacts"):
            raise ValueError("backend must be geometry or native_contacts")
        from embodichain.lab.gym.envs._json import json_safe_copy

        configuration = json_safe_copy(
            configuration, field_name="measured acceptance configuration"
        )
        self.env, self.rows, self.sensor = environment, {}, None
        self._observer_registered = False
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.initial, self.terminal, self.results = {}, {}, {}
        self._closed = False
        try:
            if backend == "native_contacts":
                from embodichain.lab.sim.sensors import (
                    ContactSensorCfg,
                    ArticulationContactFilterCfg,
                    ContactPermissionChecker,
                )

                self.sensor_uid = "expansion_native_contacts"
                if self.sensor_uid in environment.sensors:
                    raise ValueError(
                        "the measured contact validator already owns a sensor"
                    )
                self.sensor = environment.sim.add_sensor(
                    ContactSensorCfg(
                        uid=self.sensor_uid,
                        rigid_uid_list=environment.sim.get_rigid_object_uid_list(),
                        articulation_cfg_list=[
                            ArticulationContactFilterCfg(
                                articulation_uid=environment.robot.uid
                            )
                        ],
                        filter_need_both_actor=False,
                        max_contacts_per_env=8192,
                    )
                )
                environment.sensors[self.sensor_uid] = self.sensor
            for row in range(environment.num_envs):
                path = output if environment.num_envs == 1 else output / f"row-{row}"
                path.mkdir(parents=True, exist_ok=True)
                self.rows[row] = (
                    _GeometryRow(
                        environment, configuration, path, device=device, env_id=row
                    )
                    if self.sensor is None
                    else _ContactRow(
                        environment, configuration, path, sensor=self.sensor, env_id=row
                    )
                )
            if self.sensor is not None:
                permissions = self.rows[0].allowed
                self.checker = ContactPermissionChecker(
                    self.sensor,
                    [self.rows[row].frames for row in range(environment.num_envs)],
                    allowed_contacts=tuple(
                        (
                            tuple(pair)
                            if len(pair) == 2
                            else (next(iter(pair)), next(iter(pair)))
                        )
                        for pair in permissions
                    ),
                    moving_frames=tuple(self.rows[0].moving),
                )
                self.sensor.add_sampling_observer("expansion", self)
                self._observer_registered = True
        except BaseException:
            self.close()
            raise

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset only the sensor-selected monitor rows."""
        ids = tuple(self.rows) if env_ids is None else tuple(int(i) for i in env_ids)
        for row in ids:
            self.rows[row].reset() if self.sensor is not None else None

    def prepare(
        self,
        snapshot: SceneState | None = None,
        *,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Capture initial qualification after physical preparation/settling."""
        ids = tuple(self.rows) if env_ids is None else tuple(env_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(type(row) is not int or row not in self.rows for row in ids)
        ):
            raise ValueError("prepare requires unique physical rows")
        for row in ids:
            check = self.rows[row]
            check.prepare(snapshot)
            self.initial[row] = check.initial[row]
            self.terminal.pop(row, None)
            self.results.pop(row, None)

    def before_sample(self):
        return None

    def after_sample(self, dt: float, token: object) -> None:
        """Inspect every physical sample once, reducing all rows together."""
        active = [check for check in self.rows.values() if check.active]
        if not active:
            return
        summaries = self.checker.check().cpu().tolist()
        for check in active:
            check.substeps += 1
            check._interval_samples += 1
            code, actor_a, actor_b = summaries[check.env_id]
            if code:
                check._bad = f"native contact evidence rejected: code={code}; actors={actor_a},{actor_b}"

    def begin_control_step(self) -> None:
        """Start a physical interval without discarding prior failure results."""
        for check in self.rows.values():
            if self.sensor is not None and check.active:
                check.begin_control_step()

    def sample(
        self,
        ids: Sequence[int],
        action: ControllerAction,
        snapshot: SceneState | None = None,
    ) -> None:
        """Check measured task state and retain each row's first physical failure."""
        for row in ids:
            check = self.rows[row]
            check.sample((row,), action, snapshot)
            self.terminal[row] = check.terminal[row]
            self.results[row] = check.results[row]
            if self.sensor is not None:
                check._interval_samples = 0

    def close(self) -> None:
        """Detach owned observers before releasing resources; safe on failure."""
        if self._closed:
            return
        self._closed = True
        for check in self.rows.values():
            check.close()
        if self.sensor is not None:
            if self._observer_registered:
                self.sensor.remove_sampling_observer("expansion", expected=self)
            self.env.sensors.pop(self.sensor_uid, None)
            self.env.sim.remove_asset(self.sensor_uid)
