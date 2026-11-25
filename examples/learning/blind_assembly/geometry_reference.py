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

"""Evaluate the geometry-informed DexUni assembly reference controller."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json
import math
from typing import Any

from scipy.spatial.transform import Rotation
import torch

from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.gym.utils.registration import discover_task_packages, make
from embodichain.lab.scripts.run_task import _create_parser
from embodichain.lab.sim import SimulationManager

__all__ = ["main"]


class _AssemblyServo:
    """Center a held part on its fixture and follow the mating geometry."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.assembly = env.assembly_state
        self.scene = self.assembly.scene
        self.part = env.sim.get_rigid_object(self.scene._moving_uid)
        self.fixture = env.sim.get_rigid_object(self.scene._fixture_uid)
        self.action = env.action_manager.get_term(self.assembly.cfg.action_term)
        self.zero = torch.zeros_like(self.action.raw_actions)
        self.arm_base_inverse = torch.linalg.inv(self.assembly._assembly_robot_base)[
            None
        ]
        self.damping = torch.eye(6, device=env.device) * 0.0025
        self.xy_integral = torch.zeros(2, device=env.device)
        self.start_step = self.assembly.cfg.grasp_warmup_steps + 6
        self.initial_yaw = 0.0
        self.commanded_yaw = 0.0
        self.depth = 0.0
        self.capture: tuple[float, float] | None = None

    def action_at(self, step: int) -> torch.Tensor:
        if step < self.start_step:
            return self.zero
        env = self.env
        assembly = self.assembly
        qpos = env.robot.get_qpos(name=assembly.cfg.arm_part)
        tcp_local = assembly._assembly_chain.forward_kinematics(qpos).get_matrix()
        tcp = assembly._assembly_robot_base[None] @ tcp_local
        part = self.part.get_local_pose(to_matrix=True)
        fixture = self.fixture.get_local_pose(to_matrix=True)
        relative = torch.linalg.inv(fixture) @ part
        depth = float(assembly._assembly_metrics["depth"][0])
        if step == self.start_step:
            self.initial_yaw = float(torch.atan2(relative[0, 1, 0], relative[0, 0, 0]))
            self.commanded_yaw = self.initial_yaw
            self.depth = depth

        if self.scene.assembly_kind == "nut":
            geometry = self.scene.assembly_asset.config
            pitch = float(geometry["pitch"]) * float(geometry.get("scale", 1.0))
            lead = pitch * int(geometry.get("n_starts", 1))
            turn = float(assembly._assembly_angle[0])
            if self.capture is None and bool(assembly._assembly_metrics["captured"][0]):
                self.capture = depth, turn
            if self.capture is None:
                self.depth = min(self.depth + 0.009 * env.step_dt, pitch * 0.65)
            else:
                capture_depth, capture_turn = self.capture
                self.depth = capture_depth + lead * (
                    abs(turn - capture_turn) + 0.10
                ) / (2 * math.pi)
            self.commanded_yaw += 0.75 * env.step_dt
            yaw = self.commanded_yaw
        else:
            elapsed = (step - self.start_step) * env.step_dt
            yaw = self.initial_yaw + 0.18 * math.sin(1.55 * elapsed)
            self.depth = min(
                self.scene._fixture_zmax - 0.00035,
                self.depth + 0.014 * env.step_dt,
                depth + 0.0025,
            )
        self.xy_integral = (self.xy_integral - 0.045 * relative[0, :2, 3]).clamp(
            -0.0025, 0.0025
        )
        desired = torch.eye(4, device=env.device).unsqueeze(0)
        desired[0, :3, :3] = torch.as_tensor(
            Rotation.from_rotvec((0.0, 0.0, yaw)).as_matrix(),
            device=env.device,
            dtype=torch.float32,
        )
        desired[0, :2, 3] = self.xy_integral
        desired[0, 2, 3] = (
            self.scene._fixture_zmax - self.scene._moving_zmin - self.depth
        )

        part_in_tcp = torch.linalg.inv(tcp) @ part
        target = (
            self.arm_base_inverse @ fixture @ desired @ torch.linalg.inv(part_in_tcp)
        )
        translation = target[0, :3, 3] - tcp_local[0, :3, 3]
        translation *= min(
            2.0, 0.008 / max(float(torch.linalg.norm(translation)), 1e-9)
        )
        rotation_error = target[0, :3, :3] @ tcp_local[0, :3, :3].T
        axis_angle = Rotation.from_matrix(rotation_error.cpu().numpy()).as_rotvec()
        rotation = (
            1.7 * torch.as_tensor(axis_angle, device=env.device, dtype=torch.float32)
        ).clamp(-0.07, 0.07)
        jacobian = assembly._assembly_chain.jacobian(qpos)[0]
        delta = jacobian.T @ torch.linalg.solve(
            jacobian @ jacobian.T + self.damping, torch.cat((translation, rotation))
        )
        return (delta.unsqueeze(0) / self.action._scale).clamp(-1.0, 1.0)


def main() -> None:
    """Run one reference episode and report measured mating and contact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("gear_meshing", "nut_threading"), default="gear_meshing"
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    seed = (
        args.seed
        if args.seed is not None
        else (213 if args.task == "nut_threading" else 106)
    )
    discover_task_packages()
    path = files("embodichain.extensions.tasks.configs").joinpath(
        f"tasks/manipulation/assembly/{args.task}/task.study.default_assets.dexuni.yaml"
    )
    flags = [
        "--gym-config",
        str(path),
        "--seed",
        str(seed),
        "--num-envs",
        "1",
        "--headless",
        "--renderer",
        "no-render",
        "--filter-dataset-saving",
    ]
    cfg, deployment, _ = build_env_cfg_from_args(_create_parser().parse_args(flags))
    env = make(deployment["id"], cfg=cfg).unwrapped
    try:
        env.reset(seed=seed, options={"save_data": False})
        # Preserve the solved terminal state for the measurements below.
        env._replay_no_auto_reset = True
        servo = _AssemblyServo(env)
        for step in range(env.max_episode_steps):
            _, _, terminated, truncated, _ = env.step(servo.action_at(step))
            if terminated[0] or truncated[0]:
                break
        metrics = env.assembly_state._assembly_metrics
        success = bool(metrics["success"][0])
        result = {
            "task_id": deployment["id"],
            "seed": seed,
            "steps": step + 1,
            "success": success,
            "depth_m": float(metrics["depth"][0]),
            "radial_error_m": float(metrics["radial_error"][0]),
            "tilt_rad": float(metrics["tilt"][0]),
            "grasped": bool(metrics["grasped"][0]),
        }
        if args.task == "nut_threading":
            assembly = env.assembly_state
            result["turn_since_capture_rad"] = float(
                (
                    assembly._assembly_angle[0] - assembly._assembly_capture_angle[0]
                ).abs()
            )
            result["advance_since_capture_m"] = float(
                metrics["depth"][0] - assembly._assembly_capture_depth[0]
            )
        print(json.dumps(result))
        if not success:
            raise SystemExit(1)
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
