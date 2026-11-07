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

"""Configured rigid Task Program feasibility workflow and measured event adapter."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.scene.feasibility import (
    SceneRobotFeasibilityCfg,
    FeasibilityCheck,
    TaskPhase,
    NewtonCollisionChecker,
    UnsupportedFeasibilityError,
    prepare_feasibility,
    search_feasibility,
    request_from_mapping,
    save_feasibility_atlas,
)
from embodichain.lab.sim.scene.feasibility._values import digest, plain, atomic_json
from embodichain.lab.sim.motion.expansion import ValidationCheck, ValidationResult
from .simulation.feasibility import TaskProgramFeasibilityHost, save_feasibility_episode
from .simulation.feasibility_geometry import collision_bodies_from_simulation

__all__ = ["MeasuredTaskMonitor", "build_feasibility_host", "run_feasibility", "cli"]


def _closed(value: Mapping[str, object], allowed: set[str], name: str) -> dict:
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise ValueError(f"Unknown or invalid {name} fields")
    return dict(value)


class MeasuredTaskMonitor:
    """Measure explicit rigid-body predicates and ordered intermediate events.

    Args:
        final_checks: Required terminal predicates.
        events: Ordered ``event_id``, ``conditions``, ``consecutive_steps`` maps.

    Predicates use physical arena poses/velocities and explicit joint names.
    They never infer task roles or lower thresholds from candidate outcomes.
    """

    def __init__(
        self,
        final_checks: Sequence[Mapping[str, object]],
        events: Sequence[Mapping[str, object]],
        *,
        env_id: int = 0,
    ) -> None:
        if type(env_id) is not int or env_id < 0:
            raise ValueError("env_id must be a nonnegative physical row")
        self.env_id = env_id
        self.final_checks = tuple(self._validate(c) for c in final_checks)
        if not self.final_checks:
            raise ValueError("Measured task acceptance requires terminal checks")
        self.events = []
        for event in events:
            data = _closed(
                event, {"event_id", "conditions", "consecutive_steps"}, "event"
            )
            if not isinstance(data.get("event_id"), str) or not data["event_id"]:
                raise ValueError("event_id must be explicit")
            steps = data.get("consecutive_steps", 1)
            if type(steps) is not int or steps < 1:
                raise ValueError("event consecutive_steps must be positive")
            conditions = tuple(self._validate(c) for c in data["conditions"])
            if not conditions:
                raise ValueError("event conditions cannot be empty")
            self.events.append((data["event_id"], conditions, steps))
        if len({e[0] for e in self.events}) != len(self.events):
            raise ValueError("event IDs must be unique")
        self.initial_heights = {}
        self._index = 0
        self._steps = 0
        self._runs = []
        self._previous = []
        self._order_ok = True
        self.receipts = []

    @staticmethod
    def _validate(value: Mapping[str, object]) -> dict:
        kinds = {
            "pose_near": {"entity", "position", "tolerance_m"},
            "lifted": {"entity", "height_m"},
            "joint_range": {"joint", "lower", "upper"},
            "speed_below": {"entity", "linear_m_s", "angular_rad_s"},
            "axis_relation": {"entity", "other", "axis", "lower", "upper"},
            "support_bounds": {"entity", "lower", "upper"},
        }
        if not isinstance(value, Mapping) or value.get("kind") not in kinds:
            raise UnsupportedFeasibilityError(
                "Unsupported measured rigid-task predicate"
            )
        kind = value["kind"]
        data = _closed(value, kinds[kind] | {"kind", "negate"}, "predicate")
        if (
            not kinds[kind] <= data.keys()
            or type(data.get("negate", False)) is not bool
        ):
            raise ValueError("Predicate is missing required parameters")
        # Validate numbers/shapes before a native environment can be created.
        if kind in ("pose_near", "support_bounds"):
            for key in (("position",) if kind == "pose_near" else ("lower", "upper")):
                values = np.asarray(data[key], dtype=float)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise ValueError(
                        "Predicate coordinates must be three finite values"
                    )
            if kind == "support_bounds" and np.any(
                np.asarray(data["lower"]) > data["upper"]
            ):
                raise ValueError("Support bounds must be ordered")
        for key in ("tolerance_m", "height_m", "linear_m_s", "angular_rad_s"):
            if key in data and (
                type(data[key]) not in (int, float)
                or not np.isfinite(data[key])
                or data[key] < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
        if kind == "joint_range":
            if (
                not np.isfinite([data["lower"], data["upper"]]).all()
                or data["lower"] > data["upper"]
            ):
                raise ValueError("Joint bounds must be finite and ordered")
        if kind == "axis_relation" and (
            data["axis"] not in (0, 1, 2)
            or not np.isfinite([data["lower"], data["upper"]]).all()
            or data["lower"] > data["upper"]
        ):
            raise ValueError(
                "Axis relation requires an arena axis and finite ordered bounds"
            )
        return data

    def reset(self, env: Any) -> None:
        """Seed measured initial heights and reset ordered event history."""
        self.initial_heights = {
            uid: float(env.sim.get_rigid_object(uid).get_local_pose()[self.env_id, 2])
            for uid in env.sim.get_rigid_object_uid_list()
        }
        self._index = self._steps = 0
        self._runs = [0] * len(self.events)
        self._previous = [False] * len(self.events)
        self._order_ok = True
        self.receipts = []

    def predicate(self, env: Any, condition: Mapping[str, object]) -> bool:
        """Read one explicitly declared condition from current physical state.

        Args:
            env: Current environment; reads use the explicit physical row.
            condition: A validated predicate mapping.
        Returns:
            Physical truth value; unknown entities/joints are configuration errors.
        """
        kind = condition["kind"]
        if kind == "joint_range":
            index = env.robot.joint_names.index(condition["joint"])
            value = float(env.robot.get_qpos()[self.env_id, index])
            passed = condition["lower"] <= value <= condition["upper"]
        else:
            obj = env.sim.get_rigid_object(condition["entity"])
            if obj is None:
                raise ValueError("Predicate entity is absent from the physical scene")
            position = (
                obj.get_local_pose()[self.env_id, :3].detach().cpu().double().numpy()
            )
            if kind == "pose_near":
                passed = (
                    np.linalg.norm(position - condition["position"])
                    <= condition["tolerance_m"]
                )
            elif kind == "lifted":
                passed = (
                    position[2] - self.initial_heights[condition["entity"]]
                    >= condition["height_m"]
                )
            elif kind == "speed_below":
                state = obj.body_state[self.env_id]
                # RigidBodyState follows pose(7), linear(3), angular(3).
                passed = (
                    float(state[7:10].norm()) <= condition["linear_m_s"]
                    and float(state[10:13].norm()) <= condition["angular_rad_s"]
                )
            elif kind == "axis_relation":
                other = (
                    env.sim.get_rigid_object(condition["other"])
                    .get_local_pose()[self.env_id, :3]
                    .detach()
                    .cpu()
                    .numpy()
                )
                delta = position[condition["axis"]] - other[condition["axis"]]
                passed = condition["lower"] <= delta <= condition["upper"]
            else:
                # Full physical shape bounds, transformed into the arena. The
                # conservative boxes also reject unsupported overhangs.
                corners = []
                root = (
                    obj.get_local_pose(to_matrix=True)[self.env_id]
                    .detach()
                    .cpu()
                    .double()
                    .numpy()
                )
                for shape in obj.get_collision_shapes(
                    **({"env_id": self.env_id} if self.env_id else {})
                ):
                    if shape.vertices is not None and shape.vertices.numel():
                        points = shape.vertices.cpu().numpy()
                    elif shape.half_extents is not None:
                        half = shape.half_extents.cpu().numpy()
                        points = np.array(
                            [
                                [x, y, z]
                                for x in (-half[0], half[0])
                                for y in (-half[1], half[1])
                                for z in (-half[2], half[2])
                            ]
                        )
                    elif shape.radius is not None:
                        half = np.array(
                            [
                                shape.radius,
                                shape.radius,
                                shape.radius + float(shape.half_height or 0.0),
                            ]
                        )
                        points = np.array(
                            [
                                [x, y, z]
                                for x in (-half[0], half[0])
                                for y in (-half[1], half[1])
                                for z in (-half[2], half[2])
                            ]
                        )
                    else:
                        raise UnsupportedFeasibilityError(
                            "Support validation lacks physical shape bounds"
                        )
                    pose = root @ shape.local_pose.cpu().numpy()
                    corners.append(points @ pose[:3, :3].T + pose[:3, 3])
                if not corners:
                    raise UnsupportedFeasibilityError(
                        "Support validation lacks physical geometry"
                    )
                points = np.concatenate(corners)
                passed = bool(
                    np.all(points >= condition["lower"])
                    and np.all(points <= condition["upper"])
                )
        return bool(not passed if condition.get("negate", False) else passed)

    def observe(self, env: Any, action: Any) -> None:
        """Record consecutive event predicates at actual control boundaries."""
        self._steps += 1
        expected = self._index
        for i, (name, conditions, required) in enumerate(self.events):
            current = all(self.predicate(env, c) for c in conditions)
            self._runs[i] = self._runs[i] + 1 if current else 0
            triggered = self._runs[i] >= required
            if i > expected and triggered and not self._previous[i]:
                self._order_ok = False
            if i == expected and triggered:
                self.receipts.append({"event_id": name, "control_step": self._steps})
                self._index += 1
            self._previous[i] = triggered

    def validation(self, env: Any) -> ValidationResult:
        """Return measured terminal, ordering and intermediate-event checks."""
        return ValidationResult(
            (
                ValidationCheck(
                    "task.terminal",
                    (
                        "passed"
                        if all(self.predicate(env, c) for c in self.final_checks)
                        else "failed"
                    ),
                ),
                ValidationCheck(
                    "task.events",
                    (
                        "passed"
                        if self._index == len(self.events) and self._order_ok
                        else "failed"
                    ),
                    str(self.receipts),
                ),
            )
        )


def build_feasibility_host(
    config: Mapping[str, object],
    *,
    config_directory: str | Path = ".",
    output_directory: str | Path,
) -> tuple[Any, TaskProgramFeasibilityHost]:
    """Build the closed, configuration-defined Task Program search adapter.

    Args:
        config: request, search and task_program mapping. No executable imports.
        config_directory: Base for deployment/config resource paths.
        output_directory: Accepted episode and report destination.
    Returns:
        Prepared joint domain and a closeable canonical Task Program host.
    """
    top = _closed(config, {"request", "search", "task_program"}, "workflow")
    adapter = _closed(
        top["task_program"],
        {
            "deployment",
            "initial_semantics",
            "initial_geometry",
            "final_checks",
            "events",
            "runtime_phase_by_segment",
            "runtime_phase_by_call",
            "ignored_contacts",
            "settle_steps",
            "program",
            "simulation_device",
            "enable_sensor",
            "observation_keys",
            "workspace_proposal",
        },
        "task_program adapter",
    )
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg
    from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
    from embodichain.utils.utility import load_config
    from .simulation.feasibility_geometry import collision_bodies_from_simulation

    deployment = Path(config_directory) / adapter["deployment"]
    deployment_data = load_config(deployment)
    environment_id = deployment_data["id"]
    program_override = (
        None
        if "program" not in adapter
        else Path(config_directory) / adapter["program"]
    )
    physical = config_to_cfg(
        deployment_data,
        source_path=deployment,
        task_program_path_override=program_override,
    )
    if physical.task_program is None:
        raise ValueError("Feasibility workflow requires a configured Task Program")
    event_values = (
        physical.events.to_dict()
        if hasattr(physical.events, "to_dict")
        else physical.events
    )
    if event_values:
        from embodichain.lab.gym.envs.managers.events import (
            wait_for_dynamic_objects_to_settle,
        )

        settle = wait_for_dynamic_objects_to_settle
        allowed = {
            settle,
            settle.__name__,
            settle.__module__ + ":" + settle.__qualname__,
            settle.__module__ + "." + settle.__qualname__,
        }
        terms = (
            vars(physical.events).values()
            if not isinstance(physical.events, Mapping)
            else physical.events.values()
        )
        for term in terms:
            function = (
                term.get("func")
                if isinstance(term, Mapping)
                else getattr(term, "func", None)
            )
            mode = (
                term.get("mode")
                if isinstance(term, Mapping)
                else getattr(term, "mode", None)
            )
            if function not in allowed or mode != "reset":
                raise UnsupportedFeasibilityError(
                    "Configured feasibility permits reset settling only; independent scene/physical randomization requires a trusted Python adapter"
                )
    cfg = SceneRobotFeasibilityCfg.from_mapping(top.get("search", {}))
    request = request_from_mapping(top["request"])
    if len(request.modes) != 1:
        raise UnsupportedFeasibilityError(
            "Configured adapter supports one runtime policy mode; use Python configure_mode for explicit multi-mode binding"
        )
    if adapter.get("workspace_proposal") is not None:
        _closed(
            adapter["workspace_proposal"],
            {
                "phase_id",
                "control_part",
                "variable_name",
                "sample_count",
                "uniform_fraction",
            },
            "workspace proposal",
        )
    monitor = MeasuredTaskMonitor(adapter["final_checks"], adapter.get("events", ()))
    semantics = tuple(monitor._validate(v) for v in adapter["initial_semantics"])
    geometry = tuple(monitor._validate(v) for v in adapter["initial_geometry"])
    if not semantics or not geometry:
        raise ValueError(
            "Explicit initial semantic and geometric conditions are required"
        )
    # Identity incorporates actual thresholds and resolved policy/robot settings.
    from dataclasses import replace
    from importlib.metadata import version

    registration = REGISTERED_ENVS[environment_id]
    factory_fingerprint = (
        registration.task_program_adapter_factory.integration_fingerprint
    )
    request = replace(
        request,
        validator_id=request.validator_id
        + ":"
        + digest(
            (
                adapter,
                "newton-convex-query/v1",
                version("newton"),
                cfg.collision_device,
                cfg.collision_use_cuda_graph,
            )
        ),
        policy_id=request.policy_id
        + ":"
        + digest((physical.task_program.to_dict(), factory_fingerprint)),
        robot_profile_id=request.robot_profile_id
        + ":"
        + digest(physical.robot.to_dict()),
        parent_scene_id=request.parent_scene_id + ":" + digest(physical.to_dict()),
    )
    phases = {p.phase_id: p for p in request.modes[0].phases}
    segments = adapter.get("runtime_phase_by_segment", {})
    calls = adapter.get("runtime_phase_by_call", {})
    if (
        not segments
        and not calls
        or any(v not in phases for v in (*segments.values(), *calls.values()))
    ):
        raise ValueError("Explicit runtime provenance-to-phase mapping is required")

    def factory(candidate):
        import copy
        import torch

        c = copy.deepcopy(physical)
        c.num_envs = c.sim_cfg.num_envs = 1
        if "simulation_device" in adapter:
            c.sim_cfg.device = adapter["simulation_device"]
        c.sim_cfg.headless = True
        if "enable_sensor" in adapter:
            if type(adapter["enable_sensor"]) is not bool:
                raise ValueError("enable_sensor must be boolean")
            c.enable_sensor = adapter["enable_sensor"]
            if not c.enable_sensor:
                c.sim_cfg.render_cfg.renderer = "no-render"
                c.light.direct = []
                c.light.indirect = None
        c.dataset = None
        c.init_rollout_buffer = False
        c.record_trajectory = False
        c.trajectory_auto_save = False
        if candidate.poses:
            if set(candidate.poses) != {c.robot.uid}:
                raise UnsupportedFeasibilityError(
                    "Configured Gym embodiment owns one robot, including composed dual arms"
                )
            matrix = np.asarray(candidate.poses[c.robot.uid])
            c.robot.init_local_pose = torch.tensor(matrix, dtype=torch.float32)
            c.robot.init_pos = tuple(matrix[:3, 3])
            c.robot.init_rot = tuple(
                Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=True)
            )
        return REGISTERED_ENVS[environment_id].make(cfg=c)

    geometry_identity = None

    def initial(env, candidate):
        nonlocal geometry_identity
        import hashlib

        signature = digest(
            (
                hashlib.sha256(Path(env.robot.cfg.fpath).read_bytes()).hexdigest(),
                {
                    uid: env.sim.get_rigid_object(uid).get_collision_shapes()
                    for uid in env.sim.get_rigid_object_uid_list()
                },
            )
        )
        if geometry_identity is None:
            geometry_identity = signature
        elif geometry_identity != signature:
            raise ValueError(
                "Fixed physical collision geometry changed during the search"
            )
        monitor.reset(env)
        result = []
        for name, conditions in (("semantics", semantics), ("geometry", geometry)):
            passed = all(monitor.predicate(env, c) for c in conditions)
            if name == "semantics":
                passed &= not all(
                    monitor.predicate(env, c) for c in monitor.final_checks
                )
            result.append(ValidationCheck(name, "passed" if passed else "failed"))
        return ValidationResult(tuple(result))

    def runtime_phase(env, action, candidate):
        metadata = getattr(action, "metadata", {})
        key = calls.get(str(metadata.get("runtime_call_index")))
        key = key or segments.get(str(metadata.get("program_segment_id")))
        if key not in phases:
            raise UnsupportedFeasibilityError(
                f"Controller action lacks phase mapping: call={metadata.get('runtime_call_index')!r}, segment={metadata.get('program_segment_id')!r}"
            )
        return phases[key]

    bodies_cache = []

    def collision_factory(env):
        if not bodies_cache:
            bodies_cache.extend(collision_bodies_from_simulation(env))
        return NewtonCollisionChecker(
            tuple(bodies_cache),
            ignored_pairs=adapter.get("ignored_contacts", ()),
            device=cfg.collision_device,
            use_cuda_graph=cfg.collision_use_cuda_graph,
        )

    host = TaskProgramFeasibilityHost(
        environment_factory=factory,
        initial_validator=initial,
        measured_validator=lambda env, c: monitor.validation(env),
        collision_factory=collision_factory,
        runtime_phase=runtime_phase,
        observe=monitor.observe,
        sink=lambda episode: save_feasibility_episode(
            episode, Path(output_directory) / "episodes"
        ),
        settle_steps=adapter.get("settle_steps", 0),
        observation_keys=adapter.get("observation_keys", ()),
        workspace_proposal=adapter.get("workspace_proposal"),
    )
    return prepare_feasibility(request, cfg), host


def run_feasibility(config_path: str | Path, output_directory: str | Path) -> Any:
    """Search a configured task, persist exact evidence and write a slice report.

    Args:
        config_path: Closed YAML/JSON workflow configuration.
        output_directory: Output root for atlas, session audit and accepted data.
    Returns:
        A measured atlas; unsupported candidates remain explicit diagnostics.
    """
    from embodichain.utils.utility import load_config
    from embodichain.lab.sim.scene.feasibility.visualization import (
        write_feasibility_report,
    )

    path, root = Path(config_path), Path(output_directory)
    prepared, host = build_feasibility_host(
        load_config(path), config_directory=path.parent, output_directory=root
    )
    try:
        import json

        (root / "diagnostics").mkdir(parents=True, exist_ok=True)

        def checkpoint(observation):
            with (root / "diagnostics" / "attempts.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(plain(observation), allow_nan=False) + "\n")

        atlas = search_feasibility(prepared, host, on_observation=checkpoint)
        save_feasibility_atlas(atlas, root / "atlas.json")
        atomic_json(root / "session.json", host.session_snapshot)
        write_feasibility_report(atlas, root / "report.html")
        return atlas
    finally:
        try:
            atomic_json(root / "session.json", host.session_snapshot)
        finally:
            host.close()


def cli(argv: Sequence[str] | None = None) -> int:
    """Run joint scene/deployment search from a closed workflow configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    atlas = run_feasibility(args.config, args.output)
    print(
        f"regions={len(atlas.regions)} measured_successes={atlas.metrics.get('measured_successes',0)} committed={atlas.metrics.get('committed',0)} stop={atlas.stop_reason}"
    )
    if not any(o.evaluation.accepted for o in atlas.observations):
        raise SystemExit(2)
    return 0
