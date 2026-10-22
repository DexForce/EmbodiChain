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

"""Shape-feedback expert for the anchored, two-spool cable routing board.

Before running the YAM deployment, prepare its robot assets from the repository
root with ``python scripts/tools/prepare_yam_i2rt_assets.py``. The script downloads
the pinned source files, generates and verifies the URDF/collision meshes, and
reuses a verified cache at ``~/.cache/embodichain_data/YamI2rt/`` by default.
Set ``EMBODICHAIN_DATA_ROOT`` before preparation and execution for another cache
root. See the task config directory's README for dependencies and offline usage.

Geometry and routine adapted from robosimstudio's Apache-2.0 route_the_cable
example at 73739abe212f963e8a9f5862b27bca370df091a7. Robot assets retain MIT.
https://github.com/robosimstudio/robosimstudio/tree/main/examples/08_demos/route_the_cable
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any
import torch
from tensordict import TensorDict
from embodichain.compute.geometry.cable import sample_cable_material, cable_endpoints
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg, DemoSegment
from embodichain.lab.gym.envs.expert_trajectory import prepare_expert_joint_trajectory
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ActionInvocation,
    ActionControlOverrides,
    JointPositionCommand,
    ControlPartCommandProfile,
    MotionPolicy,
    SceneSnapshot,
)
from embodichain.lab.sim.atomic_actions.primitives.grasp_cable import (
    CableGraspGoal,
    GraspCableOptions,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import build_pose_plan_states
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenerator,
    MotionGenCfg,
    MotionGenOptions,
)
from embodichain.lab.sim.motion.planners import PlanResult, TrapezoidalPlannerCfg
from embodichain.lab.sim.motion.planners.utils import interpolate_xpos_batched
from embodichain.utils import logger

__all__ = ["RouteTheCableEnv", "measure_route"]


def measure_route(
    state: torch.Tensor,
    lengths: torch.Tensor,
    hands_w: torch.Tensor,
    origins: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure the reference board goal and sampled geometry diagnostics.

    Args:
        state: World cable segment states ``(B,50,13)``.
        lengths: Capsule axis rest lengths ``(50,)``.
        hands_w: World tool centres ``(B,2,3)``.
        origins: World Arena translations ``(B,3)``.

    Returns:
        Per-environment counts, speed, clearance and routing acceptance.
    """
    p = state[..., :3] - origins[:, None, :]
    starts, ends = cable_endpoints(state, lengths)
    start_local = starts - origins[:, None, :]
    end_local = ends - origins[:, None, :]
    valid = torch.isfinite(state).all(dim=(1, 2))
    out = {}
    for i, (y, side) in enumerate(((-0.26, -1), (-0.02, 1))):
        lo, hi = sorted((0.525 + side * 0.003, 0.525 + side * 0.045))
        inside = (
            (p[:, :, 0] >= lo)
            & (p[:, :, 0] <= hi)
            & ((p[:, :, 1] - y).abs() <= 0.035)
            & (p[:, :, 2] >= 0.75)
            & (p[:, :, 2] <= 0.90)
        )
        out[f"seat_{i}"] = inside.sum(-1)
    in_port = (
        ((p[:, :, 0] - 0.586).abs() <= 0.045)
        & ((p[:, :, 1] - 0.08).abs() <= 0.045)
        & (p[:, :, 2] >= 0.75)
        & (p[:, :, 2] <= 0.90)
    )
    end = end_local[:, -1]
    out["end_position"] = end
    out["port_count"] = in_port.sum(-1)
    out["end_in_port"] = (
        ((end[:, :2] - end.new_tensor([0.586, 0.08])).abs() <= 0.045).all(-1)
        & (end[:, 2] >= 0.75)
        & (end[:, 2] <= 0.90)
    )
    out["hands_clearance"] = torch.cdist(hands_w, state[..., :3]).amin(dim=(1, 2))
    out["max_speed"] = torch.linalg.vector_norm(state[..., 7:10], dim=-1).amax(-1)
    out["bench_penetration"] = (
        0.75 + 0.0035 - torch.minimum(start_local[..., 2], end_local[..., 2]).amin(-1)
    ).clamp_min(0)
    # Five points per capsule axis, using the exact capped-cylinder SDF.
    axis_samples = (
        start_local[:, :, None, :]
        + torch.linspace(0, 1, 5, device=state.device)[None, None, :, None]
        * (end_local - start_local)[:, :, None, :]
    )
    penetration = torch.zeros(state.shape[0], device=state.device)
    for y in (-0.26, -0.02):
        for radius, z0, z1 in (
            (0.0125, 0.0, 0.0025),
            (0.00475, 0.0025, 0.019679),
            (0.0125, 0.019679, 0.0235),
        ):
            radial = (
                torch.linalg.vector_norm(
                    axis_samples[..., :2] - state.new_tensor([0.525, y]), dim=-1
                )
                - radius
            )
            vertical = (axis_samples[..., 2] - (0.75 + (z0 + z1) / 2)).abs() - (
                z1 - z0
            ) / 2
            q = torch.stack((radial, vertical), -1)
            sdf = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(
                -1
            ).clamp_max(0)
            penetration = torch.maximum(
                penetration, (0.0035 - sdf).clamp_min(0).amax(dim=(1, 2))
            )
    out["spool_penetration"] = penetration
    port_penetration = torch.zeros_like(penetration)
    for center, half_size in (
        ((0.5755, 0.08, 0.75625), (0.0055, 0.016, 0.00625)),
        ((0.5965, 0.08, 0.75625), (0.0055, 0.016, 0.00625)),
        ((0.586, 0.0928, 0.75625), (0.005, 0.0032, 0.00625)),
    ):
        q = (axis_samples - state.new_tensor(center)).abs() - state.new_tensor(
            half_size
        )
        sdf = torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.amax(-1).clamp_max(0)
        port_penetration = torch.maximum(
            port_penetration, (0.0035 - sdf).clamp_min(0).amax(dim=(1, 2))
        )
    out["port_penetration"] = port_penetration
    out["strict_port"] = (
        ((end[:, 0] - 0.586).abs() <= 0.0015)
        & (end[:, 1] >= 0.064)
        & (end[:, 1] <= 0.0896)
        & (end[:, 2] >= 0.753)
        & (end[:, 2] <= 0.759)
    )
    out["release_ready"] = valid & (
        out["strict_port"]
        | (
            ((end[:, 0] - 0.586).abs() <= 0.001)
            & (end[:, 1] >= 0.073)
            & (end[:, 1] <= 0.085)
            & (end[:, 2] >= 0.753)
            & (end[:, 2] <= 0.766)
        )
    )
    out["reference_routed"] = (
        valid
        & (out["seat_0"] >= 2)
        & (out["seat_1"] >= 2)
        & (out["port_count"] >= 3)
        & out["end_in_port"]
        & (out["hands_clearance"] > 0.05)
        & (out["max_speed"] <= 0.05)
    )
    return out


@register_env("RouteTheCable-YAM-v1", max_episode_steps=6000)
class RouteTheCableEnv(EmbodiedEnv):
    """Route with physical pinches and freshly observed material geometry.

    Only the board's proximal end is fixed. No runtime latch, body teleport or
    physics-state clear is used inside the expert. Every action uses env.step.
    """

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        if cfg is None:
            raise ValueError("RouteTheCable requires a configured 50-segment cable.")
        cables = [obj for obj in cfg.cable_object if obj.uid == "cable"]
        if len(cables) != 1:
            raise ValueError("RouteTheCable requires exactly one cable named 'cable'.")
        cables[0].validate()
        points = torch.as_tensor(cables[0].points)
        if tuple(points.shape) != (51, 3):
            raise ValueError("RouteTheCable requires the 50-segment reference cable.")
        self._grasp = None
        self._command = None
        self._max_slip = None
        self._episode_slip = None
        super().__init__(cfg, **kwargs)
        try:
            self._initialize_routing()
        except Exception:
            try:
                self.close(exit_process=False)
            except Exception as cleanup_error:
                logger.log_warning(
                    f"Failed to close routing environment after initialization error: {cleanup_error!r}"
                )
            raise

    def _initialize_routing(self) -> None:
        """Bind task services only after the shared environment is ready."""
        self.cable = self.sim.get_cable_object("cable")
        if self.cable is None or self.cable.data.num_segments != 50:
            raise ValueError("RouteTheCable requires the 50-segment reference cable.")
        self._lengths = torch.linalg.vector_norm(
            torch.as_tensor(
                self.cable.cfg.points, device=self.device, dtype=torch.float32
            ).diff(dim=0),
            dim=-1,
        )
        self._origins = torch.as_tensor(
            self.sim.spawn_result.arenas.root_offsets,
            device=self.device,
            dtype=torch.float32,
        )[list(self.cable.data.batch.arena_indices)]
        self._motion = MotionGenerator(
            MotionGenCfg(planner_cfg=TrapezoidalPlannerCfg(robot_uid=self.robot.uid))
        )
        self._engine = AtomicActionEngine(
            self._motion,
            control_profiles={
                f"{side}_eef": ControlPartCommandProfile.joint_positions(
                    open=torch.full((2,), 0.0475, device=self.device),
                    # A light distal pinch avoids wedging the tiny cable
                    # against the fingertips and lifting the compliant arm.
                    grasp=torch.full(
                        (2,),
                        max(float(self.cable.cfg.radius) - 0.0005, 0.0),
                        device=self.device,
                    ),
                )
                for side in ("right", "left")
            },
        )
        self._max_slip = torch.zeros(self.num_envs, device=self.device)
        self._episode_slip = torch.zeros_like(self._max_slip)
        self._home = self.robot.get_qpos().clone()
        self._home_poses = {
            side: self._tool(side).clone() for side in ("right", "left")
        }
        self._command = self._home.clone()

    def _extend_obs(self, obs, **kwargs):
        obs = super()._extend_obs(obs, **kwargs)
        cable = self.sim.get_cable_object("cable")
        if cable is not None and cable.data is not None:
            obs["cable_state_w"] = cable.data.segment_state_w
        return obs

    def _initialize_episode(self, env_ids, **kwargs) -> None:
        super()._initialize_episode(env_ids, **kwargs)
        self._grasp = None
        if self._max_slip is not None:
            self._max_slip[env_ids] = 0
            self._episode_slip[env_ids] = 0
        self._command = self.robot.get_qpos(target=True).clone()

    def _tool(self, side: str, *, world: bool = False) -> torch.Tensor:
        ids = self.robot.get_joint_ids(name=side + "_arm")
        pose = self.robot.compute_fk(
            self.robot.get_qpos()[:, ids], name=side + "_arm", to_matrix=True
        )
        if world:
            pose = pose.clone()
            pose[:, :3, 3] += self._origins
        return pose

    def _commanded_tool(self, side: str) -> torch.Tensor:
        ids = self.robot.get_joint_ids(name=side + "_arm")
        return self.robot.compute_fk(
            self._command[:, ids], name=side + "_arm", to_matrix=True
        ).clone()

    def _point(self, distance: float):
        return sample_cable_material(
            self.cable.data.segment_state_w, self._lengths, distance
        )

    def _point_in_tool(self, side: str, distance: float) -> torch.Tensor:
        point = self._point(distance)[0]
        tool = self._tool(side, world=True)
        return torch.bmm(
            tool[:, :3, :3].transpose(1, 2), (point - tool[:, :3, 3]).unsqueeze(-1)
        ).squeeze(-1)

    def _update_sim_state(self, **kwargs) -> None:
        super()._update_sim_state(**kwargs)
        if self._grasp is not None:
            side, distance, initial = self._grasp
            slip = torch.linalg.vector_norm(
                self._point_in_tool(side, distance) - initial, dim=-1
            )
            self._max_slip = torch.maximum(self._max_slip, slip)
            self._episode_slip = torch.maximum(self._episode_slip, slip)

    def route_metrics(self) -> dict[str, torch.Tensor]:
        """Return measured routing and expert-quality metrics.

        Returns:
            Per-environment reference goal, strict channel, penetration and
            grasp-slip measurements, including the combined success flag.
        """
        hands = torch.stack(
            [self._tool(s, world=True)[:, :3, 3] for s in ("right", "left")], 1
        )
        out = measure_route(
            self.cable.data.segment_state_w, self._lengths, hands, self._origins
        )
        out["max_grasp_slip"] = self._episode_slip.clone()
        out["success"] = (
            out["reference_routed"]
            & out["strict_port"]
            & (out["bench_penetration"] <= 0.001)
            & (out["spool_penetration"] <= 0.001)
            & (out["port_penetration"] <= 0.001)
            & (out["max_grasp_slip"] <= 0.012)
        )
        return out

    def is_task_success(self) -> torch.Tensor:
        """Return the physical routing goal combined with episode grasp quality.

        Returns:
            Boolean success flags for each environment.
        """
        if not hasattr(self, "_origins"):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self.route_metrics()["success"]

    def _validate_route(self) -> torch.Tensor:
        metrics = self.route_metrics()
        success = metrics["success"]
        if not bool(success.all()):
            values = {name: value.cpu().tolist() for name, value in metrics.items()}
            logger.log_warning(f"Cable final route rejected: {values}.")
        return success

    def _target(
        self, point: torch.Tensor, yaw: torch.Tensor, height: float
    ) -> torch.Tensor:
        pose = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        c, s = torch.cos(yaw), torch.sin(yaw)
        pose[:, 0, 0] = c
        pose[:, 0, 1] = s
        pose[:, 1, 0] = s
        pose[:, 1, 1] = -c
        pose[:, 2, 2] = -1
        pose[:, :3, 3] = point - self._origins
        pose[:, 2, 3] = 0.75 + height
        return pose

    def _grasp_pose(
        self, distance: float, side: str, height: float, *, port: bool = False
    ) -> torch.Tensor:
        point, tangent, _ = self._point(distance)
        yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
        # Choose the reachable wrist branch without changing cable material ID.
        candidates = []
        for branch, angle in enumerate((yaw, yaw + math.pi)):
            pose = self._target(point, angle, height)
            success, qpos = self.robot.compute_ik(
                pose,
                joint_seed=self.robot.get_qpos()[
                    :, self.robot.get_joint_ids(name=side + "_arm")
                ],
                name=side + "_arm",
            )
            if port:
                stage = self._target(
                    point,
                    torch.full_like(yaw, math.pi / 2 if branch == 0 else -math.pi / 2),
                    0.065,
                )
                stage[:, 0, 3] = 0.586
                stage[:, 1, 3] = 0.047
                later, _ = self.robot.compute_ik(
                    stage, joint_seed=qpos, name=side + "_arm"
                )
                success = success.reshape(-1).bool() & later.reshape(-1).bool()
            candidates.append((success.reshape(-1).bool(), qpos, pose))
        first, second = candidates
        distance0 = (
            (
                first[1]
                - self.robot.get_qpos()[:, self.robot.get_joint_ids(name=side + "_arm")]
            )
            .square()
            .sum(-1)
        )
        distance1 = (
            (
                second[1]
                - self.robot.get_qpos()[:, self.robot.get_joint_ids(name=side + "_arm")]
            )
            .square()
            .sum(-1)
        )
        choose = second[0] & (~first[0] | (distance1 < distance0))
        if port:
            self._port_yaw = torch.where(
                choose,
                torch.full_like(yaw, -math.pi / 2),
                torch.full_like(yaw, math.pi / 2),
            )
        return torch.where(choose[:, None, None], second[2], first[2])

    def _execute(
        self,
        trajectory: torch.Tensor,
        joint_ids: list[int],
        *,
        closed_at: int | None = None,
        grasp_ref=None,
        stop_when: Callable[[], torch.Tensor] | None = None,
    ) -> Iterable[ControllerAction]:
        mode = self.cfg.expert_trajectory.joint_command_mode
        velocities = None
        if mode == "position_velocity":
            # Include the current command as t=0 so velocity estimation spans
            # the first executed interval on the fixed control grid.
            positions = torch.cat((self._command[:, None, :], trajectory), dim=1)
            intervals = torch.full(
                positions.shape[:2],
                self.step_dt,
                dtype=torch.float64,
                device=positions.device,
            )
            intervals[:, 0] = 0
            prepared = prepare_expert_joint_trajectory(
                PlanResult(success=True, positions=positions, dt=intervals),
                control_dt=self.step_dt,
                joint_command_mode=mode,
            )
            trajectory = prepared.positions[:, 1:]
            velocities = prepared.velocities[:, 1:]
        for index, frame in enumerate(trajectory.unbind(1), start=1):
            if stop_when is not None and bool(stop_when().all()):
                return
            if self._grasp is not None and bool((self._max_slip > 0.012).any()):
                return
            self._command[:, joint_ids] = frame[:, joint_ids]
            if mode == "position_velocity":
                velocity = torch.zeros_like(self._command)
                velocity[:, joint_ids] = velocities[:, index - 1, joint_ids]
                yield ControllerAction(
                    TensorDict(
                        {"qpos": self._command, "qvel": velocity},
                        batch_size=[self.num_envs],
                    )
                )
            else:
                yield ControllerAction(self._command)
            if closed_at == index:
                side, distance = grasp_ref
                self._max_slip.zero_()
                self._grasp = (
                    side,
                    distance,
                    self._point_in_tool(side, distance).clone(),
                )

    def _move(
        self,
        side: str,
        target: torch.Tensor,
        name: str,
        *,
        speed: float = 0.05,
        tolerance: float = 0.02,
        goal_reached: Callable[[], torch.Tensor] | None = None,
    ) -> DemoSegment:
        start = self._tool(side)
        distance = (
            torch.linalg.vector_norm(start[:, :3, 3] - target[:, :3, 3], dim=-1)
            .max()
            .item()
        )
        angle = (
            torch.acos(
                (
                    (
                        (start[:, :3, :3].transpose(1, 2) @ target[:, :3, :3])
                        .diagonal(dim1=-2, dim2=-1)
                        .sum(-1)
                        - 1
                    )
                    / 2
                ).clamp(-1, 1)
            )
            .max()
            .item()
        )
        count = max(3, math.ceil(max(distance / speed, angle / 0.5) / self.step_dt) + 1)
        ids = list(self.robot.get_joint_ids(name=side + "_arm"))
        poses = interpolate_xpos_batched(start, target, count)[:, 1:]
        result = self._motion.generate(
            build_pose_plan_states(poses),
            options=MotionGenOptions(
                strategy="ik_interp",
                start_qpos=self.robot.get_qpos()[:, ids],
                control_part=side + "_arm",
                sample_count=count,
                interpolation_dt=self.step_dt,
                is_linear=True,
                preserve_cartesian_samples=True,
            ),
        )
        # The planner may lengthen the control grid to meet joint limits.
        planned_count = count if result.positions is None else result.positions.shape[1]
        full = self._command[:, None, :].repeat(1, planned_count, 1)
        planned = (
            torch.as_tensor(result.success, device=self.device)
            .reshape(-1)
            .bool()
            .clone()
        )
        if result.positions is None:
            planned.zero_()
        else:
            full[:, :, ids] = result.positions

        def actions():
            if bool(planned.all()):
                yield from self._execute(full, ids, stop_when=goal_reached)
                yield from self._hold(0.25, stop_when=goal_reached)
                if goal_reached is not None:
                    # Contact can delay tracking after an insertion correction.
                    # Allow bounded physical settling, with the same target
                    # and acceptance limits, before declaring a failed move.
                    def settled():
                        return goal_reached() | (
                            torch.linalg.vector_norm(
                                self._tool(side)[:, :3, 3] - target[:, :3, 3],
                                dim=-1,
                            )
                            <= tolerance
                        )

                    yield from self._hold(1.0, stop_when=settled)

        def validate():
            position_error = torch.linalg.vector_norm(
                self._tool(side)[:, :3, 3] - target[:, :3, 3], dim=-1
            )
            reached = position_error <= tolerance
            if goal_reached is not None:
                reached = reached | goal_reached()
            accepted = planned & reached & (self._max_slip <= 0.012)
            if not bool(accepted.all()):
                logger.log_warning(
                    f"Cable motion segment {name!r} rejected: "
                    f"planning_success={planned.cpu().tolist()}, "
                    f"position_error_m={position_error.cpu().tolist()} "
                    f"(tolerance={tolerance}), "
                    f"max_grasp_slip_m={self._max_slip.cpu().tolist()} "
                    "(limit=0.012)."
                )
            return accepted

        return DemoSegment(
            actions=actions(),
            name=name,
            target_uid="cable",
            metadata={"side": side, "planning_success": planned.cpu().tolist()},
            validator=validate,
        )

    def _hold(
        self, seconds: float, *, stop_when: Callable[[], torch.Tensor] | None = None
    ) -> Iterable[ControllerAction]:
        for _ in range(max(1, math.ceil(seconds / self.step_dt))):
            if stop_when is not None and bool(stop_when().all()):
                return
            if self._grasp is not None and bool((self._max_slip > 0.012).any()):
                return
            if self.cfg.expert_trajectory.joint_command_mode == "position_velocity":
                yield ControllerAction(
                    TensorDict(
                        {
                            "qpos": self._command,
                            "qvel": torch.zeros_like(self._command),
                        },
                        batch_size=[self.num_envs],
                    )
                )
            else:
                yield ControllerAction(self._command)

    def _pinch(
        self, side: str, distance: float, *, port: bool = False, firm: bool = False
    ) -> Iterable[DemoSegment]:
        yield self._move(
            side,
            self._grasp_pose(distance, side, 0.06, port=port),
            f"{side}_approach_{distance}",
            speed=0.12,
        )
        before = self._point(distance)[0].clone()
        goal = CableGraspGoal(
            "cable", distance, self._grasp_pose(distance, side, 0.0115, port=port)
        )
        binding = self._engine.bind_control_parts(
            "grasp_cable",
            {"primary": {"motion": side + "_arm", "grasp": side + "_eef"}},
        )
        plan = self._engine.plan(
            ActionInvocation(
                skill_id="grasp_cable",
                goal=goal,
                binding=binding,
                motion_policy=MotionPolicy(sample_count=73),
                skill_options=GraspCableOptions(close_hold_duration=0.35),
                control_overrides=(
                    ActionControlOverrides(
                        endpoints={
                            "primary": {
                                "grasp": {
                                    "grasp": JointPositionCommand(
                                        torch.zeros(2, device=self.device)
                                    )
                                }
                            }
                        }
                    )
                    if firm
                    else ActionControlOverrides()
                ),
            ),
            self._engine.initial_context(
                scene=SceneSnapshot(timestamp=0.0, version=0, entities={}),
                control_dt=self.step_dt,
            ),
        )
        trajectory = plan.joint_trajectory.positions
        closed_at = next(
            segment.stop for segment in plan.segments if segment.name == "close"
        )
        ids = list(self.robot.get_joint_ids(name=side + "_arm")) + list(
            self.robot.get_joint_ids(name=side + "_eef")
        )
        yield DemoSegment(
            actions=(
                self._execute(
                    trajectory, ids, closed_at=closed_at, grasp_ref=(side, distance)
                )
                if bool(plan.plan_success.all())
                else ()
            ),
            name=f"{side}_pinch_lift_{distance}",
            target_uid="cable",
            metadata={
                "material_distance": distance,
                "firm_grasp": firm,
                "planning_success": plan.plan_success.cpu().tolist(),
                "phase_lengths": {
                    segment.name: segment.stop - segment.start
                    for segment in plan.segments
                },
            },
            validator=lambda: plan.plan_success
            & (self._point(distance)[0][:, 2] - before[:, 2] > 0.008)
            & (self._max_slip <= 0.012),
        )

    def _open(
        self, side: str, name: str, *, finger_position: float = 0.0475
    ) -> DemoSegment:
        self._grasp = None
        self._max_slip.zero_()
        ids = list(self.robot.get_joint_ids(name=side + "_eef"))
        start = self._command.clone()
        count = 36
        trajectory = start[:, None, :].repeat(1, count, 1)
        trajectory[:, :, ids] = torch.lerp(
            start[:, None, ids],
            torch.full_like(start[:, None, ids], finger_position),
            torch.linspace(0, 1, count, device=self.device)[None, :, None],
        )
        return DemoSegment(
            actions=self._execute(trajectory, ids),
            name=name,
            target_uid="cable",
            validator=lambda: torch.isfinite(self.cable.data.segment_state_w).all(
                dim=(1, 2)
            ),
        )

    def create_demo_segments(
        self, *, milestone: str = "full", **kwargs: Any
    ) -> Iterable[DemoSegment]:
        """Yield freshly planned stages; milestone=pinch qualifies only grasping.

        Args:
            milestone: ``pinch``, ``first_spool`` or ``full``.
            **kwargs: Reserved collector options.

        Yields:
            Bounded action segments with measured row-wise validators.
        """
        if milestone not in {"pinch", "first_spool", "full"}:
            raise ValueError("Unknown routing milestone.")
        self._command = self.robot.get_qpos(target=True).clone()
        for stage, (distance, x, y, height) in enumerate(
            (
                (0.162, 0.500, -0.255, 0.03),
                (0.282, 0.528, -0.140, 0.03),
                # Clear the second rim without the extra 15 mm lift that
                # exhausted the physical pinch's 12 mm slip budget.
                (0.438, 0.565, 0.008, 0.05),
            )
        ):
            yield from self._pinch("right", distance, firm=stage in {0, 2})
            if milestone == "pinch":
                return
            target = self._tool("right").clone()
            target[:, 2, 3] = 0.75 + height
            yield self._move("right", target, f"raise_{stage}")
            target = target.clone()
            target[:, 0, 3] = x
            target[:, 1, 3] = y
            yield self._move("right", target, f"carry_{stage}")
            target = target.clone()
            target[:, 2, 3] = 0.7645
            yield self._move("right", target, f"lower_{stage}")
            # Opening fully beside the spool sweeps a finger through its rim.
            # First clear the 7 mm cable with a 16 mm jaw aperture, then lift
            # above the rim before opening through the full finger stroke.
            yield self._open("right", f"release_{stage}", finger_position=0.008)
            target = target.clone()
            target[:, 2, 3] = 0.82
            yield self._move("right", target, f"retreat_{stage}", speed=0.08)
            yield self._open("right", f"open_clear_of_spool_{stage}")
            if milestone == "first_spool":
                return
        yield self._move("right", self._home_poses["right"], "home_right", speed=0.15)
        yield from self._place_free_end()

    def _place_free_end(self) -> Iterable[DemoSegment]:
        """Observe, align and release the distal span after routing both spools."""
        yield from self._pinch("left", 0.570, port=True, firm=True)
        # Stage directly above the mouth. Feeding forward at the old lower
        # staging height folded the free span before it reached the channel.
        target = self._target(self._point(0.570)[0], self._port_yaw, 0.065)
        target[:, :3, 3] = target.new_tensor([0.586, 0.047, 0.815])
        yield self._move("left", target, "stage_free_end")
        for correction in range(2):
            state = self.cable.data.segment_state_w
            point = self._point(0.570)[0]
            end = cable_endpoints(state, self._lengths)[1][:, -1]
            direction = end - point
            angle = torch.atan2(direction[:, 0], direction[:, 1])
            target = self._tool("left").clone()
            c, s = torch.cos(angle), torch.sin(angle)
            rotation = torch.eye(3, device=self.device).repeat(self.num_envs, 1, 1)
            rotation[:, 0, 0] = c
            rotation[:, 0, 1] = -s
            rotation[:, 1, 0] = s
            rotation[:, 1, 1] = c
            target[:, :3, :3] = rotation @ target[:, :3, :3]
            yield self._move("left", target, f"align_yaw_{correction}")
            point = self._point(0.570)[0] - self._origins
            end = (
                cable_endpoints(self.cable.data.segment_state_w, self._lengths)[1][
                    :, -1
                ]
                - self._origins
            )
            direction = end - point
            if not bool((direction[:, 1] > 1e-6).all()):
                yield DemoSegment(
                    actions=(),
                    name="reject_backward_free_end",
                    target_uid="cable",
                    validator=lambda: torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    ),
                )
                return
            meets = point[:, 0] + direction[:, 0] / direction[:, 1].clamp_min(1e-6) * (
                0.064 - point[:, 1]
            )
            target = self._tool("left").clone()
            target[:, 0, 3] += 0.586 - meets
            yield self._move("left", target, f"align_lateral_{correction}", speed=0.03)
        # Correct the actual tip between bounded lowering stages. Accumulate
        # corrections on the previous command so contact-load bias is retained.
        for level, height in enumerate((0.790, 0.780, 0.772, 0.7645)):
            if bool(self._end_release_ready().all()):
                break
            target = self._commanded_tool("left")
            target[:, 2, 3] = height
            yield self._move(
                "left",
                target,
                f"lower_channel_{level}",
                speed=0.025,
                tolerance=0.006,
                goal_reached=self._end_release_ready,
            )
            for correction in range(2):
                if bool(self._end_release_ready().all()):
                    break
                end = (
                    cable_endpoints(self.cable.data.segment_state_w, self._lengths)[1][
                        :, -1
                    ]
                    - self._origins
                )
                target = self._commanded_tool("left")
                target[:, :2, 3] += (end.new_tensor([0.586, 0.077]) - end[:, :2]).clamp(
                    -0.005, 0.005
                )
                target[:, 1, 3].clamp_(max=0.047)
                if level == 3:
                    target[:, 2, 3] += (0.754 - end[:, 2]).clamp(-0.003, 0.003)
                    target[:, 2, 3].clamp_(min=0.762)
                yield self._move(
                    "left",
                    target,
                    f"align_tip_{level}_{correction}",
                    speed=0.02,
                    tolerance=0.006,
                    goal_reached=self._end_release_ready,
                )

        def seated():
            return self._end_release_ready() & (self._max_slip <= 0.012)

        yield DemoSegment(
            actions=self._hold(self.step_dt),
            name="verify_release_alignment",
            target_uid="cable",
            validator=seated,
        )
        yield self._open("left", "release_free_end")
        target = target.clone()
        target[:, 1, 3] = 0.030
        target[:, 2, 3] = 0.82
        yield self._move("left", target, "retreat_free_end", speed=0.08)
        yield self._move("left", self._home_poses["left"], "home_left", speed=0.15)
        yield DemoSegment(
            actions=self._hold(2.0),
            name="settle_and_validate",
            target_uid="cable",
            validator=self._validate_route,
        )

    def _end_release_ready(self) -> torch.Tensor:
        metrics = self.route_metrics()
        return (
            metrics["release_ready"]
            & (metrics["port_penetration"] <= 0.001)
            & (metrics["bench_penetration"] <= 0.001)
            & (metrics["spool_penetration"] <= 0.001)
        )
