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

"""Evaluate the official ANYmal-C checkpoint in a DexSim-owned World."""

from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from dexsim.kit.motion_policy import (
    EvaluationFrame,
    JointCommand,
    PolicyContext,
    ResolvedPolicy,
    ResourceResolver,
    RunOptions,
    parse_policy_spec,
    resolve_policy_spec,
    run_motion_policy,
)
from dexsim.kit.motion_policy.evaluator import _adapter_request, _scene_spec
from dexsim.kit.motion_policy.scene import (
    SceneSpec,
    initialize_policy_robot,
    prepare_world,
)
from dexsim.kit.motion_policy.scene_runtime import (
    SceneContext,
    SceneRuntime,
    advance_simulation,
    apply_joint_command,
    apply_spec_control,
    capture_reset_state,
    read_robot_state,
    reset_robot,
    robot_description,
    robot_topology,
)
from dexsim.kit.motion_policy.types import EnvironmentStep

from embodichain.learning.rl.policy_evaluation import MotionProfileRequest

from .profile import AnymalCVelocityAdapter, build_profile

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext


class _BorrowedGround:
    """Present an owner-thread scene to the Motion Policy Kit evaluator."""

    viewer_is_open = False

    def __init__(
        self,
        context: SessionContext,
        scene: SceneSpec,
        policy: ResolvedPolicy,
        staging: Path,
        command: tuple[float, ...],
        *,
        live_path: str | None = None,
    ) -> None:
        self._attached = live_path is not None
        if self._attached:
            result = context.scene
            if result is None:
                raise ValueError("ANYmal-C requires an existing finalized scene")
            robot = result.get_articulation(live_path)
            scene_context = None
            if robot.desc.fixed_base:
                raise ValueError("ANYmal-C policy requires a floating base")
        else:
            prepare_world(context.world, scene)
            staging.mkdir()
            scene_context = SceneContext(
                context.builder, scene, policy.resources, staging
            )
            scene_context.create_flat_ground()
            scene_context.create_lights()
            path = scene_context.load_policy_robot()
            result = context.builder.finalize()
            robot = result.get_articulation(path)
            initialize_policy_robot(robot, scene)
            apply_spec_control(robot, policy)
        topology = robot_topology(robot)
        self._runtime = SceneRuntime(
            context.world,
            robot,
            topology,
            capture_reset_state(robot, topology),
            scene,
            result,
            scene_context,
        )
        self.physics_backend = scene.physics_backend
        steps = policy.runtime.sim_steps_per_control
        self.policy_context = PolicyContext(
            robot=robot_description(self._runtime),
            physics_dt=scene.physics_dt,
            sim_steps_per_control=steps,
            policy_dt=scene.physics_dt * steps,
        )
        self._command = np.asarray(command, dtype=np.float32)
        self._step = 0
        self.current_frame = self._frame()
        self._origin = self.current_frame.robot_state.root_pose[:2, 3].copy()
        self._heading = self.current_frame.robot_state.root_pose[:2, 0].copy()
        self._heading /= np.linalg.norm(self._heading)

    def _frame(self) -> EvaluationFrame:
        timing = self.policy_context
        physics_steps = self._step * timing.sim_steps_per_control
        return EvaluationFrame(
            control_step=self._step,
            policy_time=self._step * timing.policy_dt,
            simulation_step=physics_steps,
            simulation_time=physics_steps * timing.physics_dt,
            robot_state=read_robot_state(self._runtime, None),
            task_state={},
            controls={"command": self._command},
        )

    def reset(self) -> EvaluationFrame:
        if not self._attached:
            reset_robot(self._runtime)
        self._step = 0
        self.current_frame = self._frame()
        self._origin = self.current_frame.robot_state.root_pose[:2, 3].copy()
        self._heading = self.current_frame.robot_state.root_pose[:2, 0].copy()
        self._heading /= np.linalg.norm(self._heading)
        return self.current_frame

    def step(self, action: object) -> EnvironmentStep:
        if not isinstance(action, JointCommand):
            raise TypeError("ANYmal-C expects a joint command")
        apply_joint_command(self._runtime, action)
        for _ in range(self.policy_context.sim_steps_per_control):
            advance_simulation(self._runtime)
        self._step += 1
        self.current_frame = self._frame()
        return EnvironmentStep(self.current_frame, metrics=self.metrics())

    def metrics(self) -> dict[str, float]:
        pose = self.current_frame.robot_state.root_pose
        displacement = pose[:2, 3] - self._origin
        return {
            "base_height": float(pose[2, 3]),
            "displacement_x": float(
                displacement @ self._heading if self._attached else displacement[0]
            ),
            "displacement_y": float(
                displacement @ np.array([-self._heading[1], self._heading[0]])
            ),
            "tilt_degrees": math.degrees(
                math.acos(float(np.clip(pose[2, 2], -1.0, 1.0)))
            ),
        }

    def open_viewer(self, title: str) -> None:
        """The DexSim host owns the viewer."""

    def poll(self) -> str | None:
        return None

    def wait_for_reset_or_close(self) -> str:
        return "viewer closed"

    def close(self) -> None:
        """The DexSim host retains ownership of the populated scene."""


def run(
    context: SessionContext,
    *,
    files: dict[str, Path],
    options: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Run the public velocity policy and save its physical acceptance criteria."""
    started = perf_counter()
    profile = build_profile(
        MotionProfileRequest(
            checkpoint=files["policy"],
            resource_root=files["resource_root"],
            device=torch.device("cpu"),
            renderer="hybrid",
        )
    )
    resolved = resolve_policy_spec(
        parse_policy_spec(profile.policy_spec),
        ResourceResolver(output / "cache", offline=True),
    )
    command = tuple(options.get("command", (0.4, 0.0, 0.0)))
    steps = int(options.get("control_steps", 100))
    evaluation = RunOptions(headless=True, command=command, control_steps=steps)
    scene = _scene_spec(resolved, evaluation)
    adapter = AnymalCVelocityAdapter(_adapter_request(resolved, scene, "cpu"))
    scene_started = perf_counter()
    environment = _BorrowedGround(context, scene, resolved, output / "scene", command)
    # Retain the scene handles for the remainder of the host's lifetime.
    context.state["anymal_policy_environment"] = environment
    rollout_started = perf_counter()
    result = run_motion_policy(
        resolved, evaluation, adapter=adapter, environment=environment
    )
    measured = dict(result.metrics)
    checks = {
        "control_steps": result.control_steps == steps
        and result.reason == "control steps reached",
        "physics_steps": result.simulation_steps
        == steps * resolved.runtime.sim_steps_per_control,
        "forward_distance": measured["displacement_x"]
        >= options.get("minimum_distance_m", 0.4),
        "base_height": measured["base_height"] >= 0.45,
        "tilt": abs(measured["tilt_degrees"]) <= 20.0,
    }
    report = {
        "session_id": context.session_id,
        "control_steps": result.control_steps,
        "simulation_steps": result.simulation_steps,
        "simulation_time_s": result.simulation_time,
        "wall_duration_s": perf_counter() - started,
        "stage_durations_s": {
            "assets_and_policy": scene_started - started,
            "scene_and_binding": rollout_started - scene_started,
            "rollout": perf_counter() - rollout_started,
        },
        "metrics": measured,
        "checks": checks,
        "passed": all(checks.values()),
    }
    path = output / "evaluation.json"
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {**report, "evaluation": str(path)}


class AttachedVelocity:
    """Run a trained joint policy from the bound robot's measured pose."""

    def __init__(self, context, files, robot_path, command, output):
        from embodichain.lab.sim.dexsim_session._motion_jobs import MotionJobs

        self.context = context
        self.revision = context.scene.topology_revision
        diag = context.scene.get_solver_diagnostics() or {}
        if (
            diag.get("solver_type") != "dexuni"
            or not (diag.get("features") or {}).get("mujoco_solve_enabled")
            or not math.isclose(diag.get("dt", 0), 0.005)
        ):
            raise ValueError("ANYmal-C requires a 5ms DexUni joint-enabled World")
        profile = build_profile(
            MotionProfileRequest(
                checkpoint=files["policy"],
                resource_root=files["resource_root"],
                device=torch.device("cpu"),
                renderer="fast-rt",
            )
        )
        spec = resolve_policy_spec(
            parse_policy_spec(profile.policy_spec),
            ResourceResolver(output / "cache", offline=True),
        )
        scene = _scene_spec(
            spec, RunOptions(headless=True, command=command, control_steps=100)
        )
        self.env = _BorrowedGround(
            context, scene, spec, output / "scene", command, live_path=robot_path
        )
        frame = self.env.current_frame.robot_state
        if frame.root_pose[2, 3] < 0.45 or frame.root_pose[2, 2] < math.cos(
            math.radians(20)
        ):
            raise ValueError("ANYmal-C must already be standing")
        self.adapter = AnymalCVelocityAdapter(_adapter_request(spec, scene, "cpu"))
        self.adapter.setup(self.env.policy_context)
        self.jobs = MotionJobs(context, output, self.progress)

    def progress(self):
        return {
            "control_step": self.env._step,
            "physics_step": self.context.scene.get_simulation_step(),
            "simulation_time_s": self.context.scene.get_simulation_time(),
        }

    def start(self, command_id):
        return self.jobs.start(
            command_id,
            "anymal-c",
            {"command": self.env._command.tolist()},
            self._rollout,
        )

    def _rollout(self, folder):
        frame = self.env.reset()
        self.adapter.reset(frame)
        with (folder / "native.jsonl").open("w") as trace:
            for _ in range(100):
                if self.context.scene.topology_revision != self.revision:
                    raise ValueError("Scene topology changed")
                prediction = self.adapter.infer(frame)
                if prediction.termination_reason:
                    break
                frame = self.env.step(prediction.action).frame
                trace.write(
                    json.dumps(
                        {
                            **self.progress(),
                            "root_pose": frame.robot_state.root_pose.tolist(),
                        }
                    )
                    + "\n"
                )
                yield self.progress()
        metrics = self.env.metrics()
        passed = (
            self.env._step == 100
            and metrics["displacement_x"] >= 0.4
            and metrics["base_height"] >= 0.45
            and abs(metrics["tilt_degrees"]) <= 20.0
        )
        return {
            "passed": bool(passed),
            "measurements": metrics,
            "trajectory": str(folder / "native.jsonl"),
        }

    def close(self):
        self.jobs.close()
        self.adapter.close()
