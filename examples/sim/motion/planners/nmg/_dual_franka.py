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

"""Plan and play synchronized FR3 waypoint pairs with the released ONNX actor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import time
from typing import TYPE_CHECKING

from examples.sim.motion.planners.nmg._common import (
    _configure_renderer,
    _demo_parser,
    _finish_recording,
    _free_motion_commands,
    _make_sim,
    _model_path,
    _requested_joint_limits,
    _start_recording,
)

if TYPE_CHECKING:
    import torch

    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.planners import PlanResult
    from embodichain.lab.sim.objects import Robot
    from examples.sim.motion.planners.nmg._visuals import TrajectoryMarkers

_PARTS = ("left_arm", "right_arm")
_LOOK_AT = ((1.64, -1.60, 1.25), (0.34, 0.0, 0.42), (0.0, 0.0, 1.0))
_SETTLE_SECONDS = 0.6
_TRIGGER_RADIUS_M = 0.025
_TRIGGER_HOLD_SECONDS = 0.24
_TRAIL_COLORS = ((0.04, 0.85, 0.91, 1.0), (0.95, 0.24, 0.64, 1.0))
_PLAN_COLORS = ((0.18, 0.39, 0.60, 1.0), (0.55, 0.22, 0.40, 1.0))


class _DualInterlock:
    """Latch a shared signal when both measured TCPs dwell in their sensor zones."""

    def __init__(
        self, sim: SimulationManager, centers: torch.Tensor, display: bool
    ) -> None:
        self.centers = centers
        self.physics_dt = sim.sim_config.physics_dt
        self.required_steps = math.ceil(_TRIGGER_HOLD_SECONDS / self.physics_dt)
        self.consecutive_steps = 0
        self.longest_hold = 0
        self.active = False
        self.indicators = None
        self._display_state = (False, False, False)
        if display:
            self._add_markers(sim)

    def _add_markers(self, sim: SimulationManager) -> None:
        from embodichain.lab.visualization.markers import (
            MarkerGroupCfg,
            MarkerPrototypeCfg,
        )

        pads = self.centers.detach().clone()
        pads[:, 2] -= 0.052
        posts = pads.clone()
        posts[:, 2] *= 0.5
        bases = sim.add_marker_group(
            MarkerGroupCfg(
                name="nmg_interlock_stands",
                prototypes={
                    "post": MarkerPrototypeCfg(
                        shape="cylinder", color=(0.14, 0.19, 0.25, 1.0)
                    )
                },
            )
        )
        bases.update(
            translations=posts.cpu().tolist(),
            scales=[(0.042, 0.042, float(height)) for height in pads[:, 2]],
        )
        status = pads.mean(dim=0)
        status[0] += 0.16
        status[2] = 0.10
        positions = pads.cpu().tolist() + [status.cpu().tolist()]
        self.indicators = sim.add_marker_group(
            MarkerGroupCfg(
                name="nmg_interlock_indicators",
                prototypes={
                    "pad": MarkerPrototypeCfg(shape="cylinder"),
                    "status": MarkerPrototypeCfg(shape="sphere"),
                },
            )
        )
        self.indicators.update(
            translations=positions,
            scales=[(0.070, 0.070, 0.018)] * 2 + [(0.055,) * 3],
            prototype_indices=[0, 0, 1],
        )
        self._publish()

    def _publish(self) -> None:
        ready = (0.12, 0.95, 0.35, 1.0)
        inactive = (*_TRAIL_COLORS, (0.95, 0.30, 0.10, 1.0))
        self.indicators.update(
            colors=[
                ready if state else color
                for state, color in zip(self._display_state, inactive, strict=True)
            ]
        )

    def observe(self, tips: torch.Tensor, *, settle_tick: bool) -> None:
        inside = (tips - self.centers).norm(dim=-1) <= _TRIGGER_RADIUS_M
        left, right = inside.tolist()
        if not (left and right):
            self.consecutive_steps = 0
        elif settle_tick:
            self.consecutive_steps += 1
        self.longest_hold = max(self.longest_hold, self.consecutive_steps)
        self.active |= self.consecutive_steps >= self.required_steps
        display_state = (left, right, self.active)
        if self.indicators is not None and display_state != self._display_state:
            self._display_state = display_state
            self._publish()


def _play_pair(
    sim: SimulationManager,
    robot: Robot,
    plans: Mapping[str, PlanResult],
    visits: list[int],
    phases: Sequence[tuple[torch.Tensor, torch.Tensor]],
    control_dt: float,
    markers: Mapping[str, TrajectoryMarkers],
    interlock: _DualInterlock | None,
    trigger_slot: int | None,
) -> dict[str, object]:
    import torch

    from embodichain.lab.sim.motion.execution import (
        JointTrajectoryPlaybackCfg,
        play_joint_trajectory,
    )

    joint_ids = [joint for part in _PARTS for joint in robot.get_joint_ids(part)]
    cfg = JointTrajectoryPlaybackCfg(
        control_dt=control_dt,
        joint_command_mode="position_velocity",
        preserve_source_grid=True,
    )

    def measured_tips() -> torch.Tensor:
        return torch.stack(
            [
                robot.compute_fk(robot.get_qpos(name=part), name=part)[0, :3]
                for part in _PARTS
            ]
        )

    def observe(_sample: int, _count: int) -> None:
        for marker in markers.values():
            marker.record_tip()
        if interlock is not None:
            interlock.observe(measured_tips(), settle_tick=False)

    duration = 0.0
    tracking = {part: [] for part in _PARTS}
    physics_dt = sim.sim_config.physics_dt
    settle_steps = round(_SETTLE_SECONDS / physics_dt)
    for slot, (end, (commands, command_dt)) in enumerate(
        zip(visits, phases, strict=True)
    ):
        for marker in markers.values():
            marker.show_targets(slot, active=slot)
        executed = play_joint_trajectory(
            sim,
            robot,
            positions=commands,
            dt=command_dt,
            joint_ids=joint_ids,
            cfg=cfg,
            on_step=observe if markers or interlock is not None else None,
        )
        for _ in range(settle_steps):
            sim.update(step=1)
            if interlock is not None:
                interlock.observe(measured_tips(), settle_tick=slot == trigger_slot)
        for part in _PARTS:
            measured = robot.get_qpos(name=part)
            tracking[part].append(
                float((measured - plans[part].positions[:, end]).abs().max())
            )
            if part in markers:
                markers[part].record_tip(measured)
                markers[part].show_targets(slot + 1)
        duration += float(executed.dt.sum()) + settle_steps * physics_dt

    report: dict[str, object] = {
        "simulation_playback": "completed",
        "playback_duration_s": duration,
        "measured_tracking_error_rad": tracking,
    }
    if interlock is not None:
        report["interlock"] = {
            "activated": interlock.active,
            "simultaneous_hold_s": interlock.longest_hold * physics_dt,
            "required_hold_s": interlock.required_steps * physics_dt,
            "sensor_radius_m": _TRIGGER_RADIUS_M,
        }
    return report


def run_dual_franka(
    *,
    description: str,
    offsets: Mapping[str, Sequence[Sequence[float]]],
    modalities: Mapping[str, Sequence[str]],
    trigger_slot: int | None = None,
) -> None:
    """Run a free-space or interlock example with a single dual-arm robot.

    Args:
        description: Description of the runnable example.
        offsets: Reachable joint offsets from home for each arm and target.
        modalities: Matching joint or pose target type sequences.
        trigger_slot: Optional paired Cartesian target with proximity sensors.
    """
    args = _demo_parser(description).parse_args()
    _configure_renderer(args)
    model_path = _model_path("franka", args)
    if args.download_only:
        return
    if args.num_envs != 1:
        raise ValueError(
            "The dual FR3 examples visualize one environment. Use --num-envs 1."
        )

    import torch

    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.motion.planners.neural_planner import (
        NeuralPlanner,
        NeuralPlannerCfg,
        NeuralPlanOptions,
        PlanState,
    )
    from examples.sim.motion.planners.nmg._robots import make_dual_franka_cfg
    from examples.sim.motion.planners.nmg._visuals import TrajectoryMarkers

    sim, device = _make_sim(args)
    try:
        robot = sim.add_robot(cfg=make_dual_franka_cfg())
        sim.prepare()
        if (not args.headless or args.web) and sim.open_viewer() and not args.web:
            sim.get_world().get_windows().set_look_at(*[list(v) for v in _LOOK_AT])
        constraints = _requested_joint_limits(args)
        targets, options, goal_poses, frames = {}, {}, {}, {}
        for part in _PARTS:
            start = robot.get_qpos(name=part).clone()
            goal_qpos = start[:, None] + start.new_tensor(offsets[part])
            limits = robot.get_qpos_limits(name=part)
            if (
                (goal_qpos < limits[:, None, :, 0])
                | (goal_qpos > limits[:, None, :, 1])
            ).any():
                raise ValueError(f"An authored {part} target violates joint limits.")
            goal_poses[part] = robot.compute_batch_fk(
                qpos=goal_qpos, name=part, to_matrix=True
            )
            root = robot.get_solver(part).root_link_name
            frames[part] = torch.linalg.inv(
                robot.get_link_pose(root, to_matrix=True)[0]
            ).tolist()
            targets[part] = [
                (
                    PlanState.from_qpos(goal_qpos[:, slot])
                    if kind == "joint"
                    else PlanState.from_xpos(goal_poses[part][:, slot])
                )
                for slot, kind in enumerate(modalities[part])
            ]
            options[part] = NeuralPlanOptions(
                start_qpos=start,
                waypoint_modalities=list(modalities[part]),
                constraints=constraints,
            )

        count = len(targets[_PARTS[0]])
        control_dt = 4 * sim.sim_config.physics_dt
        planner = NeuralPlanner(
            NeuralPlannerCfg(
                robot_uid=robot.uid,
                sim_instance_id=sim.instance_id,
                onnx_model_path=str(model_path),
                batch_control_parts=True,
                policy_frame_from_world_by_control_part=frames,
                max_steps=40 * count,
                dt=control_dt,
                onnx_providers=(
                    ["CPUExecutionProvider"]
                    if device.type == "cpu"
                    else ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ),
                onnx_cuda_use_tf32=False,
            )
        )
        plans = planner.plan_multi_part(targets, options, synchronize_waypoints=True)
        shared = plans[_PARTS[0]]
        succeeded = all(plan.is_all_success() for plan in plans.values())
        visits = shared.waypoint_indices[0].tolist()
        report: dict[str, object] = {
            "planning_succeeded": succeeded,
            "policy_batching": shared.policy_batching,
            "duration_s": float(shared.duration[0]),
            "target_modalities": modalities,
            "requested_joint_limits": constraints,
            "paired_sample_indices": visits,
            "raw_arrivals": {
                part: plan.policy_waypoint_indices[0].tolist()
                for part, plan in plans.items()
            },
        }
        if succeeded:
            report["paired_arrival_times_s"] = shared.dt[0].cumsum(0)[visits].tolist()
        if succeeded and not args.no_playback:
            phases = _free_motion_commands(
                robot,
                torch.cat([plans[part].positions for part in _PARTS], dim=-1),
                shared.dt,
                visits,
                control_dt,
                constraints,
                control_parts=_PARTS,
            )
            display = bool(
                args.video
                or args.web
                or (not args.headless and sim.has_native_renderer)
            )
            markers = {
                part: TrajectoryMarkers(
                    sim,
                    robot,
                    goal_poses[part][0],
                    plans[part].xpos_list[0],
                    visits,
                    modalities[part],
                    control_part=part,
                    planned_color=plan_color,
                    measured_color=trail_color,
                )
                for part, plan_color, trail_color in zip(
                    _PARTS, _PLAN_COLORS, _TRAIL_COLORS, strict=True
                )
                if display
            }
            interlock = (
                None
                if trigger_slot is None
                else _DualInterlock(
                    sim,
                    torch.stack(
                        [goal_poses[part][0, trigger_slot, :3, 3] for part in _PARTS]
                    ),
                    display,
                )
            )
            if args.video:
                physics_dt = sim.sim_config.physics_dt
                settle = round(_SETTLE_SECONDS / physics_dt) * physics_dt
                _start_recording(
                    sim,
                    args.video,
                    sum(float(dt.sum()) + settle for _, dt in phases),
                    _LOOK_AT,
                )
            report.update(
                _play_pair(
                    sim,
                    robot,
                    plans,
                    visits,
                    phases,
                    control_dt,
                    markers,
                    interlock,
                    trigger_slot,
                )
            )
            if args.video:
                report["video"] = _finish_recording(sim, args.video)
            succeeded = interlock is None or interlock.active
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        if not succeeded:
            raise RuntimeError(
                "The dual FR3 example failed. See the paired arrival report."
            )
        if args.keep_open:
            while sim.is_viewer_open:
                sim.poll_viewer()
                time.sleep(0.01)
    finally:
        if sim.is_window_recording():
            sim.stop_window_record()
        sim.wait_window_record_saves()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
