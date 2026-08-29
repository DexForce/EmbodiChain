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

"""Move a W1 end effector along a Cartesian straight line with SRS IK."""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import MarkerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import DexforceW1Cfg
from embodichain.lab.visualization import (
    VisualizationCfg,
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)


def main(
    device: str,
    num_steps: int,
    line_offset: tuple[float, float, float],
    physics_steps: int,
    max_joint_step_deg: float,
    visualization: VisualizationCfg | None = None,
) -> None:
    """Run sequential SRS IK targets along a fixed-orientation straight line.

    Args:
        device: Simulation and SRS solver device, either ``cpu`` or ``cuda``.
        num_steps: Number of Cartesian waypoints, including both endpoints.
        line_offset: End-point translation relative to the initial TCP pose, in meters.
        physics_steps: Number of simulation steps displayed per waypoint.
        max_joint_step_deg: Maximum allowed change of any joint per IK waypoint.
        visualization: Optional visualization configuration.
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if physics_steps < 1:
        raise ValueError("physics_steps must be at least 1")
    if max_joint_step_deg <= 0.0:
        raise ValueError("max_joint_step_deg must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device"
        )

    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    sim = SimulationManager(
        SimulationManagerCfg(
            # Keep the native window closed while planning so renderer/window
            # lifecycle events cannot terminate or perturb timed CUDA IK calls.
            headless=True,
            device=device,
            width=2200,
            height=1200,
            visualization=visualization or VisualizationCfg(),
        )
    )
    # Keep pose conversion, IK, and FK error measurement deterministic. In automatic
    # mode the engine thread may advance the robot base between these operations.
    sim.set_manual_update(True)

    try:
        arm_name = "left_arm"
        robot_cfg = DexforceW1Cfg.from_dict({"uid": "dexforce_w1"})
        # The robot default intentionally ignores the elbow joint when ranking IK
        # candidates. For a Cartesian-path tutorial, weight every joint to prevent
        # visually disruptive branch changes between adjacent waypoints.
        robot_cfg.solver_cfg[arm_name].ik_nearest_weight = np.array(
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
        )
        robot: Robot = sim.add_robot(cfg=robot_cfg)
        sim.prepare()
        joint_ids = robot.get_joint_ids(arm_name)
        qpos_seed = torch.tensor(
            [[np.pi / 6, 0.0, 0.0, -np.pi / 2, 0.0, 0.0, np.pi / 6]],
            dtype=torch.float32,
            device=sim.device,
        )
        reference_qpos = qpos_seed.clone()
        robot.set_qpos(qpos_seed, joint_ids=joint_ids, target=False)
        robot.set_qpos(qpos_seed, joint_ids=joint_ids, target=True)
        sim.update(step=physics_steps)

        start_pose = robot.compute_fk(qpos=qpos_seed, name=arm_name, to_matrix=True)
        target_poses = start_pose.repeat(num_steps, 1, 1)
        offset = torch.tensor(
            line_offset, dtype=start_pose.dtype, device=start_pose.device
        )
        interpolation = torch.linspace(
            0.0, 1.0, num_steps, dtype=start_pose.dtype, device=start_pose.device
        )
        target_poses[:, :3, 3] += interpolation.unsqueeze(1) * offset

        # Warm up lazy FK compilation and the selected SRS backend outside timing.
        robot.compute_ik(
            pose=target_poses[:1],
            joint_seed=qpos_seed,
            name=arm_name,
            return_all_solutions=True,
        )
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        solve_times_ms: list[float] = []
        candidate_counts: list[int] = []
        translation_errors_mm: list[float] = []
        waypoint_candidates: list[torch.Tensor] = []
        continuity_weights = torch.tensor(
            [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            dtype=qpos_seed.dtype,
            device=qpos_seed.device,
        )
        max_joint_step = torch.deg2rad(
            torch.tensor(
                max_joint_step_deg, dtype=qpos_seed.dtype, device=qpos_seed.device
            )
        )
        for waypoint, target_pose in enumerate(target_poses):
            start_time = time.perf_counter()
            success, solution = robot.compute_ik(
                pose=target_pose.unsqueeze(0),
                # Use one fixed seed while enumerating candidates. Path continuity
                # is selected globally below instead of greedily changing the
                # candidate set after every waypoint.
                joint_seed=reference_qpos,
                name=arm_name,
                return_all_solutions=True,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            solve_times_ms.append((time.perf_counter() - start_time) * 1000.0)

            if not bool(success[0]):
                raise RuntimeError(
                    f"Trajectory planning failed at waypoint "
                    f"{waypoint + 1}/{num_steps}; nothing was executed."
                )

            candidates = solution[0]
            candidate_counts.append(candidates.shape[0])
            waypoint_candidates.append(candidates)
            print(
                f"Enumerated {waypoint + 1:03d}/{num_steps}: "
                f"solve={solve_times_ms[-1]:.3f} ms, "
                f"candidates={candidates.shape[0]}"
            )

        # Find a minimum-cost path through the layered IK candidate graph. This
        # avoids the greedy failure mode where a locally attractive arm angle has
        # no continuous successor at a later waypoint.
        path_costs: list[torch.Tensor] = []
        predecessors: list[torch.Tensor] = []
        first_delta = torch.atan2(
            torch.sin(waypoint_candidates[0] - reference_qpos),
            torch.cos(waypoint_candidates[0] - reference_qpos),
        )
        first_allowed = first_delta.abs().amax(dim=1) <= max_joint_step
        first_cost = (first_delta.square() * continuity_weights).sum(dim=1)
        first_cost.masked_fill_(~first_allowed, float("inf"))
        path_costs.append(first_cost)
        predecessors.append(torch.full_like(first_cost, -1, dtype=torch.long))

        for waypoint in range(1, num_steps):
            previous_candidates = waypoint_candidates[waypoint - 1]
            candidates = waypoint_candidates[waypoint]
            edge_delta = torch.atan2(
                torch.sin(candidates[:, None, :] - previous_candidates[None, :, :]),
                torch.cos(candidates[:, None, :] - previous_candidates[None, :, :]),
            )
            allowed_edges = edge_delta.abs().amax(dim=2) <= max_joint_step
            transition_cost = (edge_delta.square() * continuity_weights).sum(dim=2)
            transition_cost.masked_fill_(~allowed_edges, float("inf"))
            reference_delta = torch.atan2(
                torch.sin(candidates - reference_qpos),
                torch.cos(candidates - reference_qpos),
            )
            node_cost = 0.05 * (reference_delta.square() * continuity_weights).sum(
                dim=1
            )
            total_cost = transition_cost + path_costs[-1].unsqueeze(0)
            best_cost, best_predecessor = total_cost.min(dim=1)
            best_cost += node_cost
            if not bool(torch.isfinite(best_cost).any()):
                reachable_previous = torch.isfinite(path_costs[-1])
                reachable_edges = edge_delta[:, reachable_previous]
                smallest_step = reachable_edges.abs().amax(dim=2).min()
                raise RuntimeError(
                    f"No globally continuous IK path reaches waypoint "
                    f"{waypoint + 1}/{num_steps}: smallest available maximum "
                    f"joint step is {torch.rad2deg(smallest_step).item():.3f} deg, "
                    f"limit is {max_joint_step_deg:.3f} deg."
                )
            path_costs.append(best_cost)
            predecessors.append(best_predecessor)

        selected_indices = [int(path_costs[-1].argmin())]
        for waypoint in range(num_steps - 1, 0, -1):
            selected_indices.append(int(predecessors[waypoint][selected_indices[-1]]))
        selected_indices.reverse()
        planned_qpos = [
            waypoint_candidates[i][selected_indices[i]].unsqueeze(0)
            for i in range(num_steps)
        ]

        for waypoint, (target_pose, qpos_seed) in enumerate(
            zip(target_poses, planned_qpos, strict=True)
        ):
            actual_pose = robot.compute_fk(
                qpos=qpos_seed, name=arm_name, to_matrix=True
            )
            error_mm = float(
                torch.linalg.vector_norm(
                    actual_pose[0, :3, 3] - target_pose[:3, 3]
                ).item()
                * 1000.0
            )
            translation_errors_mm.append(error_mm)
            print(
                f"Planned {waypoint + 1:03d}/{num_steps}: " f"error={error_mm:.4f} mm"
            )

        print(
            f"SRS {device.upper()} planning summary: {num_steps}/{num_steps} solved, "
            f"median/p95/mean solve={np.median(solve_times_ms):.3f}/"
            f"{np.percentile(solve_times_ms, 95):.3f}/"
            f"{np.mean(solve_times_ms):.3f} ms, "
            f"max translation error="
            f"{max(translation_errors_mm, default=float('nan')):.4f} mm"
        )
        print(
            f"IK candidates per waypoint min/median/max: "
            f"{min(candidate_counts)}/{int(np.median(candidate_counts))}/"
            f"{max(candidate_counts)}"
        )
        if device == "cuda":
            print(
                f"CUDA peak allocated memory: "
                f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB"
            )
        planned_qpos_tensor = torch.cat(planned_qpos)
        wrapped_steps = torch.atan2(
            torch.sin(planned_qpos_tensor[1:] - planned_qpos_tensor[:-1]),
            torch.cos(planned_qpos_tensor[1:] - planned_qpos_tensor[:-1]),
        )
        max_step_flat_index = wrapped_steps.abs().argmax()
        max_step_waypoint = int(max_step_flat_index // wrapped_steps.shape[1]) + 2
        max_step_joint = int(max_step_flat_index % wrapped_steps.shape[1]) + 1
        print(
            f"Max adjacent joint step: "
            f"{torch.rad2deg(wrapped_steps.abs().max()).item():.3f} deg "
            f"at waypoint {max_step_waypoint}, joint {max_step_joint}"
        )
        if wrapped_steps.shape[0] > 1:
            joint_step_changes = wrapped_steps[1:] - wrapped_steps[:-1]
            print(
                f"Max joint step change: "
                f"{torch.rad2deg(joint_step_changes.abs().max()).item():.3f} deg"
            )

        sim.open_window()
        marker_stride = max(1, num_steps // 25)
        marker_indices = torch.arange(0, num_steps, marker_stride, device=sim.device)
        if marker_indices[-1] != num_steps - 1:
            marker_indices = torch.cat(
                (marker_indices, marker_indices.new_tensor([num_steps - 1]))
            )
        sim.draw_marker(
            MarkerCfg(
                name="srs_target_path",
                marker_type="axis",
                axis_xpos=target_poses[marker_indices],
                axis_size=0.0006,
                axis_len=0.008,
                arena_index=0,
            )
        )

        print("Planning completed; executing the joint trajectory...")
        execution_errors_mm: list[float] = []
        previous_qpos = robot.get_qpos(name=arm_name).clone()
        zero_qvel = torch.zeros_like(previous_qpos)
        for waypoint, qpos in enumerate(planned_qpos):
            wrapped_delta = torch.atan2(
                torch.sin(qpos - previous_qpos), torch.cos(qpos - previous_qpos)
            )
            for substep in range(1, physics_steps + 1):
                alpha = substep / physics_steps
                interpolated_qpos = previous_qpos + alpha * wrapped_delta
                robot.set_qpos(interpolated_qpos, joint_ids=joint_ids, target=False)
                robot.set_qpos(interpolated_qpos, joint_ids=joint_ids, target=True)
                robot.set_qvel(zero_qvel, joint_ids=joint_ids, target=False)
                robot.set_qvel(zero_qvel, joint_ids=joint_ids, target=True)
                sim.update(step=1)
            previous_qpos = qpos
            actual_qpos = robot.get_qpos(name=arm_name)
            actual_pose = robot.compute_fk(
                qpos=actual_qpos, name=arm_name, to_matrix=True
            )
            execution_errors_mm.append(
                float(
                    torch.linalg.vector_norm(
                        actual_pose[0, :3, 3] - target_poses[waypoint, :3, 3]
                    ).item()
                    * 1000.0
                )
            )
            if waypoint % marker_stride == 0 or waypoint == num_steps - 1:
                sim.draw_marker(
                    MarkerCfg(
                        name=f"srs_executed_path_{waypoint:03d}",
                        marker_type="axis",
                        axis_xpos=actual_pose,
                        axis_size=0.0012,
                        axis_len=0.004,
                        arena_index=0,
                    )
                )
        print("Trajectory execution completed.")
        print(
            f"Max execution tracking error: {max(execution_errors_mm):.4f} mm. "
            "Long axes show targets; short thick axes show executed samples."
        )
        sim.capture_visualization(force=True)
    finally:
        # Do not use the default os._exit(0) cleanup path: it suppresses Python
        # tracebacks raised during planning and makes failures look like clean exits.
        sim.destroy(exit_process=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu", help="SRS backend."
    )
    parser.add_argument(
        "--num-steps", type=int, default=50, help="Number of line waypoints."
    )
    parser.add_argument(
        "--line-offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, -0.30, 0.0),
        help="TCP line displacement in meters (default: 0 -0.30 0).",
    )
    parser.add_argument(
        "--physics-steps",
        type=int,
        default=20,
        help="Simulation interpolation steps per IK waypoint (default: 20).",
    )
    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=15.0,
        help="Reject an IK branch changing any joint by more than this angle.",
    )
    add_viser_args_to_parser(parser)
    args = parser.parse_args()
    exit_code = 0
    try:
        main(
            device=args.device,
            num_steps=args.num_steps,
            line_offset=tuple(args.line_offset),
            physics_steps=args.physics_steps,
            max_joint_step_deg=args.max_joint_step_deg,
            visualization=visualization_cfg_from_args(args),
        )
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        # Deferred destruction is only safe after main() has unwound and no local
        # Robot/solver wrappers remain live on its Python frame.
        SimulationManager.flush_cleanup_queue()
    sys.exit(exit_code)
