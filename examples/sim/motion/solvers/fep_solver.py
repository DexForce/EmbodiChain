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
"""Move the Franka TCP around a horizontal circle using sequential FEP IK.

Run: python -m examples.sim.motion.solvers.fep_solver --device cuda
Use --headless --max-steps 301 for a finite smoke run, or --headless --viser
for browser visualization. The default radius is 15 cm; set --radius in metres.
The seed keeps q7 fixed by default. Add --redundancy-search to optimize q7;
--arm-angle sets a soft swivel preference in radians (otherwise follow the seed).
Green shows the target circle; orange traces the actual TCP over the last lap.
Position/velocity drives track interpolated commands; this is not collision
planning. The example uses stiffer arm drives to reduce tracking error.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import time

from embodichain.cli.sim import add_sim_args_to_parser


def main() -> None:
    """Run a fixed-orientation TCP circle with the previous solution as seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_sim_args_to_parser(parser)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Stop after this many IK targets."
    )
    parser.add_argument(
        "--radius", type=float, default=0.15, help="Horizontal circle radius in metres."
    )
    parser.add_argument(
        "--redundancy-search", action="store_true", help="Optimize q7 and arm posture."
    )
    parser.add_argument(
        "--arm-angle",
        type=float,
        default=None,
        help="Preferred swivel angle in radians; requires --redundancy-search.",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("--radius must be finite and positive")
    if args.arm_angle is not None and (
        not math.isfinite(args.arm_angle) or not args.redundancy_search
    ):
        parser.error("--arm-angle must be finite and requires --redundancy-search")

    import numpy as np
    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg, physics_cfg_for_backend
    from embodichain.lab.sim.motion.solvers import FEPSolverCfg
    from embodichain.lab.sim.robots import FrankaPandaCfg
    from embodichain.lab.visualization import visualization_cfg_from_args
    from embodichain.lab.visualization.protocol import SceneOverlays, TrajectoryOverlay

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=args.device,
            num_envs=args.num_envs,
            arena_space=args.arena_space,
            physics_dt=0.01,
            physics_cfg=physics_cfg_for_backend(args.physics),
            render_cfg=RenderCfg(renderer=args.renderer),
            visualization=visualization_cfg_from_args(args),
        )
    )
    count = 0
    max_ik_error = 0.0
    max_tracking_error = 0.0
    tracking_error_sum = 0.0
    max_joint_step = 0.0
    try:
        cfg = FrankaPandaCfg.from_dict({})
        cfg.solver_cfg["arm"] = FEPSolverCfg(
            root_link_name="base",
            end_link_name="fr3_hand_tcp",
            redundancy_search=args.redundancy_search,
            arm_angle=args.arm_angle,
            # At 50 Hz this caps each searched joint command at 2 rad/s.
            max_joint_step=0.04 if args.redundancy_search else None,
        )
        # Demo-only tuning: the default 1e4/1e3 stiffness/damping gives
        # visible tracking lag and load deflection. Keep the robot preset intact.
        cfg.joint_drive_props.stiffness["fr3_joint[1-7]"] = 1e5
        robot = sim.add_robot(cfg=cfg)
        sim.prepare()
        seed = robot.get_solver("arm").get_default_qpos_seed()[None]
        # The limit midpoint has q2=0, a shoulder singularity: tiny Cartesian
        # steps can cause large, unavoidable fixed-q7 branch changes there.
        seed[:, 1] = -0.4
        seed = seed.expand(sim.num_envs, -1).clone()
        velocity_limits = robot.get_qvel_limits(name="arm")
        control_dt = 2 * sim.sim_config.physics_dt
        robot.set_qpos(seed, name="arm", target=False)
        robot.set_qvel(torch.zeros_like(seed), name="arm", target=False)
        robot.set_qpos(seed, name="arm")
        robot.set_qvel(torch.zeros_like(seed), name="arm")
        sim.update(step=20)
        start = robot.compute_fk(seed, name="arm", to_matrix=True)
        # Start at the outermost point and extend inward to retain reachability.
        angles = torch.linspace(0, 2 * math.pi, 301, device=sim.device)
        circle = start[:, None, :3, 3].repeat(1, len(angles), 1)
        circle[:, :, 0] += args.radius * (angles.cos() - 1)
        circle[:, :, 1] += args.radius * angles.sin()
        # IK uses arena coordinates; both viewers need world coordinates.
        offsets = sim.arena_offsets
        circle_world = (circle + offsets[:, None]).cpu().numpy()
        actual = robot.compute_fk(
            robot.get_qpos(name="arm"), name="arm", to_matrix=True
        )
        trail = deque([(actual[:, :3, 3] + offsets).cpu().numpy()], maxlen=301)
        target_color = (40, 210, 80)
        actual_color = (255, 140, 30)
        actual_rgba = np.array((*actual_color, 255), dtype=np.float32) / 255
        actual_cloud = None
        if not args.viser:
            points = circle_world.reshape(-1, 3)
            sim.visualize_point_cloud(
                points,
                colors=np.tile(target_color, (len(points), 1)),
                point_size=4.0,
                name="fep_target_circle",
            )
            actual_cloud = sim.get_env().create_point_cloud("fep_actual_trajectory")
            actual_cloud.reserve_points(301 * sim.num_envs)
            actual_cloud.apply_points(
                trail[0], np.tile(actual_rgba, (sim.num_envs, 1)), async_update=False
            )
            actual_cloud.set_point_size(5.0)
        if not args.headless and sim.open_window():
            center = circle_world[0].mean(axis=0)
            sim.get_world().get_windows().set_look_at(
                eye=center + np.array([0.7, -1.2, 0.8], dtype=np.float32),
                look_at=center - np.array([0.0, 0.0, 0.15], dtype=np.float32),
                up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
        print(
            f"Franka FEP on {sim.device}; horizontal circle radius={args.radius:.2f} m. "
            f"Redundancy search={'on' if args.redundancy_search else 'off'}. "
            "Green: target; orange: actual TCP. Ctrl+C to stop.",
            flush=True,
        )

        # Two 10 ms physics steps per command, one circle every six seconds.
        while args.max_steps is None or count < args.max_steps:
            tick = time.perf_counter()
            target = start.clone()
            target[:, :3, 3] = circle[:, count % 300]
            valid, joints = robot.compute_ik(target, joint_seed=seed, name="arm")
            if not bool(valid.all()):
                raise RuntimeError(f"FEP failed at target {count}; stopping motion")
            fk = robot.compute_fk(joints, name="arm", to_matrix=True)
            max_ik_error = max(
                max_ik_error,
                float((fk[:, :3, 3] - target[:, :3, 3]).norm(dim=-1).max()),
            )
            delta = joints - seed
            velocity = delta / control_dt
            if not bool((velocity.abs() <= velocity_limits).all()):
                raise RuntimeError(
                    f"Joint velocity limit exceeded at target {count}; "
                    "stopping instead of executing a discontinuous IK branch"
                )
            max_joint_step = max(max_joint_step, float(delta.abs().max()))
            robot.set_qvel(velocity, name="arm")
            for substep in range(2):
                robot.set_qpos(seed + delta * ((substep + 1) / 2), name="arm")
                sim.update(step=1)
            actual = robot.compute_fk(
                robot.get_qpos(name="arm"), name="arm", to_matrix=True
            )
            tracking_error = (actual[:, :3, 3] - target[:, :3, 3]).norm(dim=-1)
            max_tracking_error = max(max_tracking_error, float(tracking_error.max()))
            tracking_error_sum += float(tracking_error.mean())
            trail.append((actual[:, :3, 3] + offsets).cpu().numpy())
            trace = np.stack(trail)
            if actual_cloud is not None:
                points = trace.reshape(-1, 3)
                colors = np.tile(actual_rgba, (len(points), 1))
                if actual_cloud.apply_points(points, colors, async_update=False) != len(
                    points
                ):
                    raise RuntimeError("Failed to update the TCP trajectory display")
            else:
                sim.set_visualization_overlays(
                    SceneOverlays(
                        trajectories=tuple(
                            overlay
                            for env_id in range(sim.num_envs)
                            for overlay in (
                                TrajectoryOverlay(
                                    overlay_id=f"fep_target_{env_id}",
                                    points=circle_world[env_id],
                                    color=target_color,
                                ),
                                TrajectoryOverlay(
                                    overlay_id=f"fep_actual_{env_id}",
                                    points=trace[:, env_id],
                                    color=actual_color,
                                ),
                            )
                        )
                    )
                )
            seed = joints
            count += 1
            if not args.headless or args.viser:
                time.sleep(max(0.0, control_dt - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        print(
            f"Solved {count} targets; max IK position error={max_ik_error * 1000:.4f} mm; "
            f"max drive-tracking position error={max_tracking_error * 1000:.3f} mm; "
            f"mean tracking error={tracking_error_sum / max(count, 1) * 1000:.3f} mm; "
            f"max joint step={max_joint_step:.4f} rad.",
            flush=True,
        )
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
