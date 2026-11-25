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

"""Run the DexUni CartPole task with a linear state-feedback controller."""

from __future__ import annotations

import argparse
from importlib.resources import files
import json

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete
import torch

from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.gym.utils.registration import discover_task_packages, make
from embodichain.lab.scripts.run_task import _create_parser
from embodichain.lab.sim import SimulationManager

__all__ = ["main"]


def _balance_gain(step_dt: float) -> np.ndarray:
    """Solve the discrete regulator for the CartPole asset and cart drive."""
    # Cart and pole inertial values come from CartPole/cart_pole.urdf.
    pole_mass, cart_mass, center, inertia = 0.1, 0.5, 0.5, 0.01
    cross = pole_mass * center
    joint_inertia = np.array(
        [[cart_mass + pole_mass, cross], [cross, inertia + pole_mass * center**2]]
    )
    dynamics = np.zeros((4, 4))
    dynamics[:2, 2:] = np.eye(2)
    dynamics[2:, 1] = np.linalg.solve(joint_inertia, [0, cross * 9.81])
    control = np.zeros((4, 1))
    control[2:, 0] = np.linalg.solve(joint_inertia, [1, 0])
    # The 0.1 m action scale and 10 N/m drive gain produce 1 N per unit
    # action. The drive also applies 1 N s/m damping to the cart.
    dynamics[2:, 2] = -control[2:, 0]
    a, b, *_ = cont2discrete((dynamics, control, np.eye(4), np.zeros((4, 1))), step_dt)
    cost = np.diag([4.0, 70.0, 1.0, 3.0])
    effort_cost = np.array([[1.0]])
    solution = solve_discrete_are(a, b, cost, effort_cost)
    return np.linalg.solve(
        effort_cost + b.T @ solution @ b, b.T @ solution @ a
    ).reshape(4)


def main() -> None:
    """Evaluate one closed-loop episode and check its final physical state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    discover_task_packages()
    deployment = files("embodichain.extensions.tasks.configs").joinpath(
        "tasks/classic_control/cart_pole/env.dexuni.yaml"
    )
    flags = [
        "--gym-config",
        str(deployment),
        "--num-envs",
        "1",
        "--seed",
        str(args.seed),
        "--filter-dataset-saving",
        "--disable-sensor",
        "--renderer",
        "fast-rt" if args.viewer else "no-render",
        "--no-headless" if args.viewer else "--headless",
    ]
    cfg, data, _ = build_env_cfg_from_args(_create_parser().parse_args(flags))
    env = make(data["id"], cfg=cfg).unwrapped
    try:
        gain = torch.as_tensor(
            _balance_gain(env.step_dt), dtype=torch.float32, device=env.device
        )
        env.reset(seed=args.seed, options={"save_data": False})
        env._replay_no_auto_reset = True
        peak_angle = 0.0
        action = torch.zeros((1, 2), device=env.device)
        for steps in range(1, env.max_episode_steps + 1):
            angles = env.robot.get_qpos()
            velocities = env.robot.get_qvel()
            feedback = torch.cat((angles, velocities), dim=-1)
            action[:, 0] = -(feedback * gain).sum(dim=-1).clamp(-1.0, 1.0)
            _, _, terminated, truncated, info = env.step(action)
            peak_angle = max(peak_angle, abs(float(env.robot.get_qpos()[0, 1])))
            if terminated[0] or truncated[0]:
                break
        success = bool(info["success"][0]) and steps == env.max_episode_steps
        print(
            json.dumps(
                {
                    "success": success,
                    "steps": steps,
                    "peak_pole_angle_rad": peak_angle,
                    "final_pole_angle_rad": float(env.robot.get_qpos()[0, 1]),
                    "final_pole_velocity_rad_s": float(env.robot.get_qvel()[0, 1]),
                }
            )
        )
        if not success:
            raise SystemExit(1)
    finally:
        env.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
