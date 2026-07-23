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

"""Stateless trajectory builder utilities for atomic actions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.lab.sim.planners import PlanState, PlanResult, MoveType
from embodichain.lab.sim.planners.motion_generator import MotionGenOptions
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator


def _resolve_runtime_device(device: torch.device | str) -> torch.device:
    """Resolve an indexless CUDA device to the active concrete GPU index."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return resolved


class TrajectoryBuilder:
    """Stateless trajectory utilities shared by every atomic action.

    Holds a reference to the motion generator (and through it, the robot and
    device) so callers don't have to thread those through each helper call.
    All methods are pure: no per-call state is kept on the builder.
    """

    def __init__(self, motion_generator: MotionGenerator) -> None:
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = _resolve_runtime_device(self.robot.device)

    # ------------------------------------------------------------------
    # Success / shape helpers
    # ------------------------------------------------------------------

    def all_envs_success(self, is_success: bool | torch.Tensor) -> bool:
        """Return true only when all environments report success."""
        if isinstance(is_success, torch.Tensor):
            return bool(torch.all(is_success).item())
        return bool(is_success)

    def resolve_pose_target(self, target: torch.Tensor, *, n_envs: int) -> torch.Tensor:
        """Resolve an end-effector pose target into batched homogeneous transforms.

        Accepts the following shapes for ``target``:

        - ``(4, 4)`` — broadcast to ``(n_envs, 4, 4)`` (single waypoint).
        - ``(n_envs, 4, 4)`` — single waypoint, validated and passed through.
        - ``(n_envs, n_waypoint, 4, 4)`` — a multi-waypoint trajectory; each
          waypoint is visited in order. ``n_waypoint`` may be 1.

        Returns a 3D tensor for single-waypoint inputs and a 4D tensor for
        multi-waypoint inputs.
        """
        if not isinstance(target, torch.Tensor):
            logger.log_error(
                f"target must be torch.Tensor of shape (4, 4), ({n_envs}, 4, 4), "
                f"or ({n_envs}, n_waypoint, 4, 4)",
                TypeError,
            )
        target = target.to(device=self.device, dtype=torch.float32)
        if target.shape == (4, 4):
            target = target.unsqueeze(0).repeat(n_envs, 1, 1)
        if target.dim() == 3:
            if target.shape != (n_envs, 4, 4):
                logger.log_error(
                    f"target tensor must have shape (4, 4) or ({n_envs}, 4, 4), "
                    f"but got {target.shape}",
                    ValueError,
                )
        elif target.dim() == 4:
            if target.shape[0] != n_envs or target.shape[2:] != (4, 4):
                logger.log_error(
                    f"multi-waypoint target tensor must have shape "
                    f"({n_envs}, n_waypoint, 4, 4), but got {target.shape}",
                    ValueError,
                )
            if target.shape[1] == 0:
                logger.log_error(
                    "multi-waypoint target tensor has zero waypoints (shape[1] == 0); "
                    "at least one waypoint is required.",
                    ValueError,
                )
        else:
            logger.log_error(
                f"target tensor must be (4, 4), ({n_envs}, 4, 4), or "
                f"({n_envs}, n_waypoint, 4, 4), but got {target.shape}",
                ValueError,
            )
        return target

    def resolve_start_qpos(
        self,
        start_qpos: torch.Tensor | None,
        *,
        n_envs: int,
        arm_dof: int,
        control_part: str,
    ) -> torch.Tensor:
        """Resolve planning start joint positions into batched arm joint positions."""
        if start_qpos is None:
            start_qpos = self.robot.get_qpos(name=control_part)
        if start_qpos.shape == (arm_dof,):
            start_qpos = start_qpos.unsqueeze(0).repeat(n_envs, 1)
        if start_qpos.shape != (n_envs, arm_dof):
            logger.log_error(
                f"start_qpos must have shape ({n_envs}, {arm_dof}), "
                f"but got {start_qpos.shape}",
                ValueError,
            )
        return start_qpos

    def resolve_joint_target(
        self,
        target_qpos: torch.Tensor,
        *,
        n_envs: int,
        joint_dof: int,
        control_part: str,
    ) -> torch.Tensor:
        """Resolve a joint-space target into batched control-part joint positions.

        Accepts the following shapes for ``target_qpos``:

        - ``(joint_dof,)`` — broadcast to ``(n_envs, joint_dof)`` (single waypoint).
        - ``(n_envs, joint_dof)`` — single waypoint, validated and passed through.
        - ``(n_envs, n_waypoint, joint_dof)`` — a multi-waypoint trajectory; each
          waypoint is visited in order. ``n_waypoint`` may be 1.

        Returns a 2D tensor for single-waypoint inputs and a 3D tensor for
        multi-waypoint inputs, leaving downstream planners to treat the trailing
        axis as the joint dimension.
        """
        if not isinstance(target_qpos, torch.Tensor):
            logger.log_error(
                f"target qpos for '{control_part}' must be a torch.Tensor with shape "
                f"({joint_dof},), ({n_envs}, {joint_dof}), or "
                f"({n_envs}, n_waypoint, {joint_dof})",
                TypeError,
            )
        target_qpos = target_qpos.to(device=self.device, dtype=torch.float32)
        if target_qpos.shape == (joint_dof,):
            target_qpos = target_qpos.unsqueeze(0).repeat(n_envs, 1)
        if target_qpos.dim() == 2:
            if target_qpos.shape != (n_envs, joint_dof):
                logger.log_error(
                    f"target qpos for '{control_part}' must have shape ({joint_dof},) "
                    f"or ({n_envs}, {joint_dof}), but got {target_qpos.shape}",
                    ValueError,
                )
        elif target_qpos.dim() == 3:
            if target_qpos.shape[0] != n_envs or target_qpos.shape[2] != joint_dof:
                logger.log_error(
                    f"multi-waypoint target qpos for '{control_part}' must have shape "
                    f"({n_envs}, n_waypoint, {joint_dof}), but got {target_qpos.shape}",
                    ValueError,
                )
            if target_qpos.shape[1] == 0:
                logger.log_error(
                    f"multi-waypoint target qpos for '{control_part}' has zero waypoints "
                    f"(shape[1] == 0); at least one waypoint is required.",
                    ValueError,
                )
        else:
            logger.log_error(
                f"target qpos for '{control_part}' must be 1D, 2D, or 3D with "
                f"trailing dim {joint_dof}, but got {target_qpos.shape}",
                ValueError,
            )
        return target_qpos

    # ------------------------------------------------------------------
    # Pose math
    # ------------------------------------------------------------------

    def apply_local_offset(
        self, pose: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Apply a world-frame translational offset to a batched pose.

        Despite the historical method name, ``offset`` is added directly to the
        translation column and is not rotated by each pose's orientation.
        """
        if not (pose.dim() == 3 and pose.shape[1:] == (4, 4)):
            logger.log_error("pose must have shape [N, 4, 4]", ValueError)
        offset = offset.to(device=pose.device, dtype=pose.dtype)
        if offset.dim() == 1:
            offset = offset.unsqueeze(0)
        if not (offset.dim() == 2 and offset.shape[1] == 3):
            logger.log_error("offset must have shape [N, 3] or [3]", ValueError)
        if offset.shape[0] not in (1, pose.shape[0]):
            logger.log_error(
                f"offset batch size must be 1 or match pose batch size {pose.shape[0]}, "
                f"but got {offset.shape[0]}",
                ValueError,
            )
        result = pose.clone()
        result[:, :3, 3] += offset
        return result

    # ------------------------------------------------------------------
    # IK / FK convenience
    # ------------------------------------------------------------------

    def ik_solve(
        self,
        target_pose: torch.Tensor,
        *,
        control_part: str,
        qpos_seed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Solve IK for a single (unbatched) target pose."""
        if qpos_seed is None:
            qpos_seed = self.robot.get_qpos(name=control_part)[0]
        elif qpos_seed.dim() == 2:
            qpos_seed = qpos_seed[0]
        elif qpos_seed.dim() != 1:
            logger.log_error(
                f"qpos_seed must be 1D or 2D, but got shape {qpos_seed.shape}",
                ValueError,
            )
        success, qpos = self.robot.compute_ik(
            pose=target_pose.unsqueeze(0),
            joint_seed=qpos_seed.unsqueeze(0),
            name=control_part,
            env_ids=[0],
        )
        if not success.all():
            logger.log_error(f"IK failed for target pose: {target_pose}", RuntimeError)
        return qpos.squeeze(0)

    def fk_compute(self, qpos: torch.Tensor, *, control_part: str) -> torch.Tensor:
        """Compute forward kinematics for a joint configuration."""
        is_unbatched = qpos.dim() == 1
        if is_unbatched:
            qpos = qpos.unsqueeze(0)
        xpos = self.robot.compute_fk(qpos=qpos, name=control_part, to_matrix=True)
        return xpos.squeeze(0) if is_unbatched else xpos

    # ------------------------------------------------------------------
    # Waypoint splitting
    # ------------------------------------------------------------------

    def split_three_phase(
        self,
        sample_interval: int,
        hand_interp_steps: int,
        *,
        first_phase_ratio: float = 0.6,
        first_phase_name: str = "first",
        third_phase_name: str = "third",
    ) -> tuple[int, int, int]:
        """Split total sample interval into motion, hand-interp, and motion phases."""
        first = int(np.round(sample_interval - hand_interp_steps) * first_phase_ratio)
        if first < 2:
            logger.log_error(
                f"Not enough waypoints for {first_phase_name} trajectory. "
                "Increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        second = hand_interp_steps
        third = sample_interval - first - second
        if third < 2:
            logger.log_error(
                f"Not enough waypoints for {third_phase_name} trajectory. "
                "Increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        return first, second, third

    # ------------------------------------------------------------------
    # Arm trajectory planning
    # ------------------------------------------------------------------

    def plan_arm_traj(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
        cfg: "ActionCfg | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan batched arm trajectories for all environments.

        Returns ``(success:(B,), trajectory:(B, n_waypoints, arm_dof))``.
        ``cfg.motion_source`` selects 'ik_interp' (default) or 'motion_gen'.
        """
        motion_source = (
            getattr(cfg, "motion_source", "ik_interp") if cfg else "ik_interp"
        )
        if motion_source == "motion_gen":
            return self._plan_motion_gen(
                target_states_list,
                start_qpos,
                n_waypoints,
                control_part=control_part,
                arm_dof=arm_dof,
                cfg=cfg,
            )
        return self._plan_ik_interp(
            target_states_list,
            start_qpos,
            n_waypoints,
            control_part=control_part,
            arm_dof=arm_dof,
        )

    def _plan_ik_interp(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched IK + interpolation fallback trajectory source."""
        n_envs = start_qpos.shape[0]
        n_state = len(target_states_list[0])
        xpos_traj = torch.zeros(
            (n_envs, n_state, 4, 4), dtype=torch.float32, device=self.device
        )
        for i, target_states in enumerate(target_states_list):
            for j, target_state in enumerate(target_states):
                xpos_traj[i, j] = target_state.xpos

        trajectory = torch.zeros(
            (n_envs, n_state, arm_dof), dtype=torch.float32, device=self.device
        )
        success = torch.ones(n_envs, dtype=torch.bool, device=self.device)
        qpos_seed = start_qpos
        for j in range(n_state):
            is_success, qpos = self.robot.compute_ik(
                pose=xpos_traj[:, j], name=control_part, joint_seed=qpos_seed
            )
            if not self.all_envs_success(is_success):
                logger.log_warning(
                    f"Failed to compute IK for target state {j} in some environments."
                )
                success = success & is_success
            trajectory[:, j] = qpos
            qpos_seed = qpos
        trajectory = torch.concatenate([start_qpos.unsqueeze(1), trajectory], dim=1)
        # Failed envs: hold start qpos across all waypoints
        if not success.all():
            held = start_qpos.unsqueeze(1).repeat(1, trajectory.shape[1], 1)
            trajectory = torch.where(success[:, None, None], trajectory, held)
        interp = interpolate_with_distance(
            trajectory=trajectory, interp_num=n_waypoints, device=self.device
        )
        return success, interp

    def _plan_motion_gen(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
        cfg: "ActionCfg | None",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Motion-generator trajectory source for Cartesian (EEF) targets."""
        if self.motion_generator is None:
            logger.log_error(
                "motion_source='motion_gen' requires a MotionGenerator on the engine",
                ValueError,
            )
        self._validate_planner_type(cfg)
        n_envs = start_qpos.shape[0]
        plan_states = self._to_batched_plan_states(target_states_list, n_envs)
        plan_opts = self._build_plan_opts(cfg, n_waypoints)
        result: PlanResult = self.motion_generator.generate(
            plan_states,
            options=MotionGenOptions(
                start_qpos=start_qpos,
                control_part=control_part,
                plan_opts=plan_opts,
                is_interpolate=self.motion_generator.planner.preinterpolate_targets,
            ),
        )
        return self._process_motion_gen_result(result, start_qpos, n_waypoints, arm_dof)

    def _validate_planner_type(self, cfg: "ActionCfg | None") -> None:
        """Reject actions whose requested planner differs from the engine's."""
        actual_type = self.motion_generator.planner.cfg.planner_type
        requested_type = getattr(cfg, "planner_type", None)
        if requested_type != actual_type:
            logger.log_error(
                f"Action requested planner_type={requested_type!r}, but "
                f"MotionGenerator owns {actual_type!r}.",
                ValueError,
            )

    def _process_motion_gen_result(
        self,
        result: PlanResult,
        start_qpos: torch.Tensor,
        n_waypoints: int,
        arm_dof: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate a MotionGenerator PlanResult and apply sample/hold policy."""
        success = (
            result.success
            if isinstance(result.success, torch.Tensor)
            else torch.tensor(result.success, device=self.device)
        )
        positions = result.positions
        n_envs = start_qpos.shape[0]
        if positions is None or positions.ndim != 3:
            logger.log_error(
                "MotionGenerator returned no (B, N, controlled_dof) positions",
                ValueError,
            )
        if positions.shape[0] != n_envs or positions.shape[2] != arm_dof:
            logger.log_error(
                f"MotionGenerator returned incompatible trajectory shape "
                f"{tuple(positions.shape)}; expected (..., {arm_dof}) on "
                f"{n_envs} envs.",
                ValueError,
            )
        if positions.device != self.device or not torch.isfinite(positions).all():
            logger.log_error(
                "MotionGenerator returned non-finite or wrong-device positions",
                ValueError,
            )
        if not self.motion_generator.planner.preserve_plan_samples:
            if positions.shape[1] != n_waypoints:
                positions = interpolate_with_distance(
                    trajectory=positions, interp_num=n_waypoints, device=self.device
                )
        positions = positions.to(self.device)
        # Failed envs hold start qpos across all waypoints.
        if not success.all():
            held = start_qpos.unsqueeze(1).repeat(1, positions.shape[1], 1)
            positions = torch.where(success[:, None, None], positions, held)
        return success, positions

    def _to_batched_plan_states(
        self, target_states_list: list[list[PlanState]], n_envs: int
    ) -> list[PlanState]:
        """Convert per-env PlanState lists into a batched list[PlanState].

        Each output PlanState carries a leading batch dim ``B`` so it matches
        the planner contract.
        """
        n_state = len(target_states_list[0])
        batched: list[PlanState] = []
        for j in range(n_state):
            sample = target_states_list[0][j]
            if sample.xpos is not None:
                xpos = torch.stack(
                    [target_states_list[i][j].xpos for i in range(n_envs)]
                )  # (B, 4, 4)
                batched.append(
                    PlanState(
                        xpos=xpos,
                        move_type=MoveType.EEF_MOVE,
                        move_part=sample.move_part,
                    )
                )
            else:
                qpos = torch.stack(
                    [target_states_list[i][j].qpos for i in range(n_envs)]
                )  # (B, DOF)
                batched.append(
                    PlanState(
                        qpos=qpos,
                        move_type=MoveType.JOINT_MOVE,
                        move_part=sample.move_part,
                    )
                )
        return batched

    def _build_plan_opts(self, cfg: "ActionCfg | None", n_waypoints: int):
        """Build planner options from action configuration (three-way factory)."""
        planner_type = getattr(cfg, "planner_type", None)
        if planner_type == "toppra":
            constraints: dict = {}
            vl = getattr(cfg, "velocity_limit", None)
            al = getattr(cfg, "acceleration_limit", None)
            constraints["velocity"] = vl if vl is not None else 0.2
            constraints["acceleration"] = al if al is not None else 0.5
            return ToppraPlanOptions(
                sample_method=TrajectorySampleMethod.QUANTITY,
                sample_interval=n_waypoints,
                constraints=constraints,
            )
        if planner_type == "neural":
            from embodichain.lab.sim.planners.neural_planner import NeuralPlanOptions

            return NeuralPlanOptions()
        if planner_type == "curobo":
            from embodichain.lab.sim.planners.curobo.curobo_planner import (
                CuroboPlanOptions,
            )

            return CuroboPlanOptions(max_attempts=getattr(cfg, "max_attempts", None))
        logger.log_error(
            f"Unknown planner_type {planner_type!r} for motion_source='motion_gen'.",
            ValueError,
        )

    def plan_joint_motion(
        self,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
        cfg: "ActionCfg | None" = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan a joint-space trajectory through one or more target waypoints.

        For ``motion_source='motion_gen'``, this delegates only when the
        selected backend advertises ``supports_joint_move``. Cartesian-only
        backends (such as the neural planner) retain the deterministic local
        interpolation for joint-only phases. ``motion_source='ik_interp'``
        always uses that local interpolation.

        Returns:
            ``(success:(B,), trajectory:(B, N, arm_dof))``.
        """
        motion_source = (
            getattr(cfg, "motion_source", "ik_interp") if cfg else "ik_interp"
        )
        if motion_source == "motion_gen":
            if self.motion_generator is None:
                logger.log_error(
                    "motion_source='motion_gen' requires a MotionGenerator on the engine",
                    ValueError,
                )
            self._validate_planner_type(cfg)
            if self.motion_generator.planner.supports_joint_move:
                if target_qpos.dim() == 2:
                    target_qpos = target_qpos.unsqueeze(1)  # (B, 1, D)
                plan_states = [
                    PlanState(qpos=target_qpos[:, j], move_type=MoveType.JOINT_MOVE)
                    for j in range(target_qpos.shape[1])
                ]
                plan_opts = self._build_plan_opts(cfg, n_waypoints)
                result: PlanResult = self.motion_generator.generate(
                    plan_states,
                    options=MotionGenOptions(
                        start_qpos=start_qpos,
                        control_part=control_part,
                        plan_opts=plan_opts,
                        is_interpolate=self.motion_generator.planner.preinterpolate_targets,
                    ),
                )
                return self._process_motion_gen_result(
                    result, start_qpos, n_waypoints, arm_dof
                )
        success = torch.ones(start_qpos.shape[0], dtype=torch.bool, device=self.device)
        trajectory = self.plan_joint_traj(start_qpos, target_qpos, n_waypoints)
        return success, trajectory

    def plan_joint_traj(
        self,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate a joint-space trajectory through one or more target waypoints.

        ``start_qpos`` has shape ``(n_envs, joint_dof)``. ``target_qpos`` is
        either a single waypoint ``(n_envs, joint_dof)`` or a sequence of
        waypoints ``(n_envs, n_waypoint, joint_dof)``. The start configuration is
        prepended to the target waypoints to build the keyframe sequence
        ``(n_envs, 1 + n_waypoint, joint_dof)``, which is then resampled to
        ``n_waypoints`` output samples by cumulative-distance piecewise-linear
        interpolation — so each consecutive waypoint pair is traversed in turn.
        """
        if target_qpos.dim() == 2:
            target_qpos = target_qpos.unsqueeze(1)
        keyframes = torch.cat([start_qpos.unsqueeze(1), target_qpos], dim=1)
        return interpolate_with_distance(
            trajectory=keyframes, interp_num=n_waypoints, device=self.device
        )

    # ------------------------------------------------------------------
    # Hand qpos helpers
    # ------------------------------------------------------------------

    def expand_hand_qpos(
        self, hand_qpos: torch.Tensor, *, n_envs: int, hand_dof: int
    ) -> torch.Tensor:
        """Resolve hand qpos to batched shape ``(n_envs, hand_dof)``."""
        hand_qpos = hand_qpos.to(device=self.device, dtype=torch.float32)
        if hand_qpos.shape == (hand_dof,):
            return hand_qpos.unsqueeze(0).repeat(n_envs, 1)
        if hand_qpos.shape == (n_envs, hand_dof):
            return hand_qpos
        logger.log_error(
            f"hand_qpos must have shape ({hand_dof},) or ({n_envs}, {hand_dof}), "
            f"but got {hand_qpos.shape}",
            ValueError,
        )
        raise AssertionError("unreachable")  # logger.log_error already raised

    def broadcast_hand_qpos_to_waypoints(
        self,
        hand_qpos: torch.Tensor,
        *,
        n_envs: int,
        hand_dof: int,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Expand hand qpos to (n_envs, n_waypoints, hand_dof) by broadcasting the per-env value across all waypoints."""
        return (
            self.expand_hand_qpos(hand_qpos, n_envs=n_envs, hand_dof=hand_dof)
            .unsqueeze(1)
            .repeat(1, n_waypoints, 1)
        )

    def interpolate_hand_qpos(
        self,
        start_hand_qpos: torch.Tensor,
        end_hand_qpos: torch.Tensor,
        *,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate hand joint positions between two gripper states."""
        is_unbatched = start_hand_qpos.dim() == 1 and end_hand_qpos.dim() == 1
        start_hand_qpos = start_hand_qpos.to(self.device)
        end_hand_qpos = end_hand_qpos.to(self.device)
        if start_hand_qpos.dim() == 1:
            start_hand_qpos = start_hand_qpos.unsqueeze(0)
        if end_hand_qpos.dim() == 1:
            end_hand_qpos = end_hand_qpos.unsqueeze(0)
        weights = torch.linspace(
            0, 1, steps=n_waypoints, device=self.device, dtype=start_hand_qpos.dtype
        )
        result = torch.lerp(
            start_hand_qpos.unsqueeze(1),
            end_hand_qpos.unsqueeze(1),
            weights[None, :, None],
        )
        if is_unbatched:
            return result.squeeze(0)
        return result


__all__ = ["TrajectoryBuilder"]
