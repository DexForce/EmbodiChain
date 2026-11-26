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
    ) -> None:
        prepare_world(context.world, scene)
        staging.mkdir()
        scene_context = SceneContext(context.builder, scene, policy.resources, staging)
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
        reset_robot(self._runtime)
        self._step = 0
        self.current_frame = self._frame()
        self._origin = self.current_frame.robot_state.root_pose[:2, 3].copy()
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
            "displacement_x": float(displacement[0]),
            "displacement_y": float(displacement[1]),
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
