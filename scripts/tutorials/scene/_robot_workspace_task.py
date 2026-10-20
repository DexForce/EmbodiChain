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

"""Task Program consumer of the independent scene-workspace demonstration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import REGISTERED_ENVS
from embodichain.lab.sim.cfg import LightCfg
from embodichain.lab.sim.motion.expansion import ValidationResult
from embodichain.lab.task_program import (
    CyclicPoseTargetCfg,
    PoseCfg,
    WaitStablePostCfg,
    ObjectNearTargetValidatorCfg,
)
from embodichain.lab.task_program.integrations import (
    SimulationTaskProgramAdapterFactory,
)
from embodichain.lab.task_program.integrations.scene_expansion import (
    execute_scene_variant,
)
from embodichain.lab.task_program.integrations.simulation.scene_expansion import (
    SimulationSceneExpansionHost,
    _validation_result,
)
from embodichain.toolkits.graspkit.pose_generator import GraspPoseGenerator
from embodichain.utils.utility import load_config
from embodichain.utils.config_paths import resolve_config_path

from ._robot_workspace_scene import (
    CUBE_Z,
    GOAL,
    TABLE_Z,
    _fragment,
    _robot_cfg,
    _prepare_candidates,
    _initial_checks,
    _negative,
    _settling_cfg,
)

__all__: list[str] = []


class _PinnedGrasp(GraspPoseGenerator):
    """Bind one generated object-local grasp to every fresh task adapter."""

    def __init__(self, binding: dict[str, torch.Tensor]) -> None:
        self.binding = binding
        self.requests = 0

    def get_valid_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
        obj_longest_axis: torch.Tensor | None = None,
        is_positive_part: bool | torch.Tensor = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self.requests += 1
        local = self.binding["pose"].to(device=obj_poses.device, dtype=obj_poses.dtype)
        return [(pose.unsqueeze(0), pose.new_zeros(1)) for pose in obj_poses @ local]

    def get_best_grasp_poses(
        self,
        *,
        mesh_vertices: torch.Tensor,
        mesh_triangles: torch.Tensor,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.requests += 1
        local = self.binding["pose"].to(device=obj_poses.device, dtype=obj_poses.dtype)
        return (
            torch.ones(len(obj_poses), dtype=torch.bool, device=obj_poses.device),
            obj_poses @ local,
            obj_poses.new_full((len(obj_poses),), 0.052),
        )


class _Evidence:
    def __init__(self) -> None:
        self.reset()

    def reset(self, *, initial_height: float | None = None) -> None:
        self.initial_height = initial_height
        self.steps = 0
        self.lift = 0.0
        self.stable_steps = 0
        self.max_stable_steps = 0
        self.position_error = float("inf")
        self.linear_speed = float("inf")
        self.angular_speed = float("inf")

    def observe(self, env: Any) -> None:
        if self.initial_height is None:
            raise RuntimeError("Lift evidence requires the measured initial height")
        cube = env.sim.get_rigid_object("cube")
        full = cube.body_state[0]
        position = cube.get_local_pose()[0, :3]
        finger = float(env.robot.get_qpos(name="eef")[0, 0])
        self.steps += 1
        if finger > 0.01:
            self.lift = max(self.lift, float(position[2]) - self.initial_height)
        self.position_error = float(
            torch.linalg.vector_norm(position - position.new_tensor(GOAL))
        )
        self.linear_speed = float(torch.linalg.vector_norm(full[7:10]))
        self.angular_speed = float(torch.linalg.vector_norm(full[10:13]))
        stable = (
            finger <= 0.005
            and self.position_error <= 0.04
            and self.linear_speed <= 0.03
            and self.angular_speed <= 0.2
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        self.max_stable_steps = max(self.max_stable_steps, self.stable_steps)

    def validation(self) -> ValidationResult:
        from embodichain.lab.sim.motion.expansion import (
            ValidationCheck,
            ValidationResult,
        )

        if not self.steps:
            return ValidationResult(
                (
                    ValidationCheck(
                        "task.measured_execution",
                        "unavailable",
                        "No executed control steps",
                    ),
                )
            )
        return ValidationResult(
            (
                ValidationCheck(
                    "task.measured_lift",
                    "passed" if self.lift >= 0.05 else "failed",
                    metrics={"lift_m": self.lift},
                ),
                ValidationCheck(
                    "task.released_stable_goal",
                    "passed" if self.stable_steps >= 10 else "failed",
                    metrics={
                        "stable_control_steps": self.stable_steps,
                        "position_error_m": self.position_error,
                        "linear_speed": self.linear_speed,
                        "angular_speed": self.angular_speed,
                    },
                ),
            )
        )


class _ObservedEnv(gym.Wrapper):
    def __init__(self, env: Any, evidence: _Evidence, display: Any) -> None:
        super().__init__(env)
        self.evidence, self.display = evidence, display

    def step(self, action: Any) -> tuple[Any, ...]:
        result = self.env.step(action)
        self.evidence.observe(self.env)
        self.display.control_step(self.env.sim)
        return result


def _task_config(display: Any) -> tuple[Any, Any]:
    path = resolve_config_path(
        "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml",
        base_dir=Path(__file__).parent,
    )
    cfg = config_to_cfg(load_config(path), source_path=path)
    registration = REGISTERED_ENVS[
        "TaskProgramRepeatedPickPlace-v1"
    ].task_program_registration
    cfg.num_envs = 1
    cfg.sim_cfg = display.config()
    cfg.robot = _robot_cfg()
    entities = _fragment().entities
    cfg.rigid_object = [entity for entity in entities if entity.body_type == "dynamic"]
    cfg.background = [entity for entity in entities if entity.body_type == "static"]
    cfg.light.direct = (
        []
        if display.view in ("none", "viser")
        else [
            LightCfg(
                uid="workspace_light",
                light_type="sun",
                intensity=5.0,
                direction=(0.0, 0.0, -1.0),
            )
        ]
    )
    cfg.light.indirect = None
    cfg.sensor = [display.camera_config()] if display.view == "offscreen" else []
    cfg.enable_sensor = bool(cfg.sensor)
    cfg.dataset = None
    cfg.filter_dataset_saving = True
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = True
    cfg.trajectory_auto_save = False
    cfg.max_episode_steps = 2400
    cfg.task_program.program.count = 1
    cfg.task_program.instruction = "Pick the cube from its sampled initial pose and place it in the fixed goal region."
    drop = cfg.task_program.targets["drop_pose"]
    for pose in drop.values:
        pose.position = (*pose.position[:2], pose.position[2] + TABLE_Z)
    cfg.task_program.targets["rest_pose"] = CyclicPoseTargetCfg(
        values=(PoseCfg(position=GOAL, quaternion_xyzw=(0.0, 0.0, 0.0, 1.0)),)
    )
    place = cfg.task_program.program.body.items[-1]
    place.post = (WaitStablePostCfg(entity="cube"),)
    place.validators = (
        ObjectNearTargetValidatorCfg(
            object="cube", target="rest_pose", position_tolerance=0.04
        ),
    )
    cfg.validate()
    return cfg, registration


def _run_task(output: Path, seed: int, count: int, display: Any) -> dict[str, Any]:
    display.stage = "configuration"
    cfg, registration = _task_config(display)
    cfg.seed = seed
    binding: dict[str, torch.Tensor] = {}
    generators: list[_PinnedGrasp] = []

    def grasp_factory() -> _PinnedGrasp:
        generator = _PinnedGrasp(binding)
        generators.append(generator)
        return generator

    factory = SimulationTaskProgramAdapterFactory(
        registration,
        grasp_pose_generator_factories={"eef": grasp_factory},
    )
    display.stage = "environment_setup"
    env = EmbodiedEnv(cfg, task_program_adapter_factory=factory)
    evidence = _Evidence()
    observed = _ObservedEnv(env, evidence, display)
    try:
        env.reset(seed=seed)
        if display.view == "offscreen":
            display.camera = env.sensors["workspace_camera"]
        display.start(env.sim)
        display.stage = "candidate_generation"
        workspace, grasp, candidates, info = _prepare_candidates(
            env.robot, output, seed, count
        )
        binding["pose"] = grasp.detach().clone()
        display.candidates(candidates, grasp, info, sim=env.sim)
        current_seed: tuple[float, ...] | None = None
        initialization_order: list[int] = []

        def initial_validator() -> ValidationResult:
            checks = _initial_checks(env.sim, workspace, grasp, current_seed)
            if evidence.steps == 0:
                if (
                    not initialization_order
                    or initialization_order[-1] != current_index
                ):
                    initialization_order.append(current_index)
                evidence.initial_height = float(
                    env.sim.get_rigid_object("cube").get_local_pose()[0, 2]
                )
            display.status[current_index] = "passed" if checks.passed else "failed"
            display.frame(env.sim, f"candidate_{current_index}_initial", force=True)
            return _validation_result(checks)

        host = SimulationSceneExpansionHost(
            env,
            initial_validator=initial_validator,
            settling=_settling_cfg(),
        )
        rows = []
        (output / "trajectories").mkdir(exist_ok=True)
        current_index = count
        current_seed = None
        evidence.reset()
        negative = execute_scene_variant(
            observed,
            _negative(seed, count),
            prepare_scene=host,
            measured_validator=evidence.validation,
            episode_index=current_index,
            attempt_id=current_index,
        )
        negative_steps = evidence.steps
        if negative.accepted or negative.execution is not None or negative_steps:
            raise RuntimeError("Red start was executed despite failed IK")
        display.frame(env.sim, "negative_rejected", force=True)
        for candidate in candidates:
            current_index = candidate.variant.ordinal
            current_seed = candidate.joint_seed
            evidence.reset()
            result = execute_scene_variant(
                observed,
                candidate.variant,
                prepare_scene=host,
                measured_validator=evidence.validation,
                episode_index=current_index,
                attempt_id=current_index,
            )
            snapshot = host.initial_state
            row = {
                "variant_id": candidate.variant.variant_id,
                "cache_index": candidate.cache_index,
                "original_pose": candidate.original_pose,
                "adjusted_pose": candidate.variant.pose_changes[0].arena_pose,
                "scene": result.to_metadata(),
                "execution": (
                    None if result.execution is None else result.execution.to_metadata()
                ),
                "failure_stage": (
                    "initial_validation"
                    if not result.preparation.validation.accepted
                    else (
                        "program_execution"
                        if not result.program_success
                        else "measured_acceptance" if not result.accepted else None
                    )
                ),
                "measured_lift_m": evidence.lift,
                "measured_initial_height_m": evidence.initial_height,
                "stable_control_steps": evidence.stable_steps,
                "executed_steps": evidence.steps,
            }
            first_matches = False
            if evidence.steps:
                expected = next(
                    entity["pose"]
                    for entity in snapshot.to_metadata()["entities"]
                    if entity["uid"] == "cube"
                )
                first = env._traj_buffer["states", "rigid_objects", "cube", "pose"][
                    0, 0
                ]
                first_matches = bool(
                    torch.allclose(first, first.new_tensor(expected), atol=1e-6, rtol=0)
                )
            row["first_frame_matches_initial_state"] = first_matches
            if evidence.steps and not first_matches:
                raise RuntimeError(
                    "Recorded first frame differs from actual initial state"
                )
            if result.accepted:
                trajectory = (
                    output
                    / "trajectories"
                    / f"candidate_{candidate.variant.ordinal:03d}.pt"
                )
                env.save_trajectory(str(trajectory))
                row["trajectory"] = str(trajectory.relative_to(output))
            if evidence.steps:
                display.frame(env.sim, f"candidate_{current_index}_result", force=True)
            restored = host.restore_initial_state(snapshot)
            row["restoration"] = {
                "initial_state_id": restored.initial_state_id,
                "restored_from_initial_state_id": restored.metadata[
                    "restored_from_initial_state_id"
                ],
                "checks": [
                    {
                        "name": c.check_id,
                        "status": c.status,
                        "metrics": dict(c.metrics),
                    }
                    for c in restored.validation.checks
                ],
                "passed": restored.validation.accepted,
            }
            if not restored.validation.accepted or int(env._traj_steps[0]) != 0:
                raise RuntimeError("Restoration or recording reset failed")
            rows.append(row)
            display.frame(
                env.sim, f"candidate_{candidate.variant.ordinal}_settled", force=True
            )
        return {
            "workspace": info,
            "candidates": rows,
            "negative": negative.to_metadata(),
            "negative_executed_steps": negative_steps,
            "initialization_order": initialization_order,
            "successful_trials": sum(row["scene"]["accepted"] for row in rows),
            "task_grasp_provider_calls": sum(
                generator.requests for generator in generators
            ),
            "accepted": all(
                row["scene"]["initial_validation"]
                and all(
                    c["status"] == "passed" for c in row["scene"]["initial_validation"]
                )
                for row in rows
            )
            and any(row["scene"]["accepted"] for row in rows),
        }
    finally:
        display.native_cloud = None
        env.close(exit_process=False)
