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
from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
import numpy as np
import torch

from embodichain.lab.sim.planners.base_planner import (
    BasePlanner,
    BasePlannerCfg,
    PlanOptions,
    _infer_batch_size,
    validate_plan_options,
)
from embodichain.lab.sim.planners.utils import MoveType, PlanResult, PlanState
from embodichain.utils import configclass, logger
from embodichain.utils.math import convert_quat, quat_error_magnitude, quat_from_matrix

__all__ = [
    "NeuralPlanner",
    "NeuralPlannerCfg",
    "NeuralPlanOptions",
]


class _OnnxPolicy:
    """Small ONNX Runtime wrapper that keeps NMG model code out of EmbodiChain."""

    def __init__(self, path: Path, providers: list[str] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "NeuralPlanner requires onnxruntime. Install EmbodiChain with "
                "the 'nmg' optional dependency."
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "NMG ONNX policy must expose exactly one input and one output; "
                f"got {len(inputs)} inputs and {len(outputs)} outputs."
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        input_shape = inputs[0].shape
        output_shape = outputs[0].shape
        if len(input_shape) != 2 or not isinstance(input_shape[1], int):
            raise ValueError(
                f"Expected ONNX input shape [batch, obs], got {input_shape}."
            )
        if len(output_shape) != 2 or output_shape[1] != 7:
            raise ValueError(
                f"Expected ONNX output shape [batch, 7], got {output_shape}."
            )
        self.obs_dim = int(input_shape[1])
        self.fixed_batch_size = (
            int(input_shape[0]) if isinstance(input_shape[0], int) else None
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if self.fixed_batch_size is not None and obs.shape[0] != self.fixed_batch_size:
            raise ValueError(
                f"ONNX policy has fixed batch size {self.fixed_batch_size}, "
                f"but received {obs.shape[0]}."
            )
        obs_np = np.ascontiguousarray(obs.detach().cpu().numpy(), dtype=np.float32)
        action = self.session.run([self.output_name], {self.input_name: obs_np})[0]
        return torch.as_tensor(action, dtype=torch.float32, device=obs.device)


def _waypoint_obs_dim(num_waypoints: int, use_relative_obs: bool) -> int:
    """Return the unified NMG constraint-observation width."""
    n = int(num_waypoints)
    dim = 7 + 7 + n * (3 + 4 + 7 + 5) + 7 + n
    if use_relative_obs:
        dim += 7 + n * (3 + 4 + 7)
    return dim


def _quat_inverse_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Return the inverse of a unit quaternion stored as xyzw."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply quaternion tensors stored as xyzw."""
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _canonicalize_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Map equivalent quaternion signs to the training-time ``w >= 0`` half."""
    return torch.where(q[..., 3:4] < 0.0, -q, q)


@configclass
class NeuralPlannerCfg(BasePlannerCfg):
    planner_type: str = "neural"

    onnx_model_path: str = MISSING
    """Path to a standalone NMG ONNX policy, including observation normalization."""

    control_part: str | None = None
    """Robot control part used for FK and qpos, e.g. 'left_arm'."""

    max_steps: int = 240
    """Maximum closed-loop kinematic rollout steps."""

    action_scale: float = 0.2
    """Delta joint scaling factor in radians."""

    num_arm_joints: int = 7
    """Number of arm joints controlled by the APG policy."""

    num_waypoints: int = 8
    """Number of constraint slots encoded by the ONNX policy."""

    use_relative_obs: bool = True
    """Whether the exported policy uses the unified relative-observation blocks."""

    canonicalize_quat_obs: bool = True
    """Whether quaternion observations use the training-time ``w >= 0`` convention."""

    intermediate_orientation: bool = True
    """Whether every Cartesian waypoint requires its orientation constraint."""

    pos_eps: float = 0.01
    """Waypoint position threshold in metres."""

    rot_eps: float = 0.1
    """Waypoint rotation threshold in radians."""

    joint_eps: float = 0.02
    """Waypoint joint-position threshold in radians."""

    onnx_providers: list[str] | None = None
    """Optional ONNX Runtime execution-provider priority list."""

    policy_frame_from_world: list[list[float]] | None = None
    """Optional left transform ``policy_T_world`` for runtime poses."""

    runtime_tcp_from_policy_tcp: list[list[float]] | None = None
    """Optional right transform ``runtime_tcp_T_policy_tcp`` for runtime poses.

    Together, pose conversion is
    ``policy_T_policy_tcp = policy_T_world @ world_T_runtime_tcp
    @ runtime_tcp_T_policy_tcp``.
    """

    dt: float = 0.01
    """Nominal timestep reported in PlanResult."""


@configclass
class NeuralPlanOptions(PlanOptions):
    control_part: str | None = None
    start_qpos: torch.Tensor | None = None
    max_steps: int | None = None


class NeuralPlanner(BasePlanner):
    r"""Neural motion planner based on an APG waypoint transformer policy.

    The planner loads a standalone ONNX waypoint policy and rolls it out in
    closed loop to drive the arm toward a sequence of end-effector and/or
    joint-position waypoints. Velocities and accelerations in the returned
    :class:`PlanResult` are estimated via finite differences over the generated
    position trajectory.

    Args:
        cfg: Configuration for the neural planner.

    Raises:
        ValueError: If ``onnx_model_path`` is missing or invalid.
        FileNotFoundError: If the ONNX model file does not exist.
        ImportError: If ONNX Runtime is not installed.
    """

    supported_move_types = frozenset({MoveType.EEF_MOVE, MoveType.JOINT_MOVE})
    preserve_plan_samples = True
    """Keep native closed-loop states instead of distance-resampling them."""
    preserve_failed_plan_positions = True
    """Keep closed-loop rollout samples even when not all waypoints converge."""

    def __init__(self, cfg: NeuralPlannerCfg):
        super().__init__(cfg)

        self.cfg: NeuralPlannerCfg = cfg
        if cfg.onnx_model_path is MISSING or not str(cfg.onnx_model_path):
            logger.log_error("onnx_model_path is required", ValueError)
        self._load_onnx_model(Path(cfg.onnx_model_path))

    def default_plan_options(self) -> NeuralPlanOptions:
        return NeuralPlanOptions()

    def with_motion_context(
        self,
        options: PlanOptions,
        *,
        start_qpos: torch.Tensor | None,
        control_part: str | None,
    ) -> NeuralPlanOptions:
        """Forward MotionGenerator context into :class:`NeuralPlanOptions`."""
        if not isinstance(options, NeuralPlanOptions):
            logger.log_error("NeuralPlanner requires NeuralPlanOptions", TypeError)
        if options.control_part is None:
            options.control_part = control_part
        if options.start_qpos is None:
            options.start_qpos = start_qpos
        return options

    def _load_onnx_model(self, model_path: Path) -> None:
        if not model_path.exists():
            logger.log_error(f"ONNX policy not found: {model_path}", FileNotFoundError)
        if model_path.suffix.lower() != ".onnx":
            raise ValueError(
                "NeuralPlanner only accepts standalone .onnx policies; "
                f"got {model_path}."
            )

        self._num_waypoints = int(self.cfg.num_waypoints)
        self._use_relative_obs = bool(self.cfg.use_relative_obs)
        self._canonicalize_quat_obs = bool(self.cfg.canonicalize_quat_obs)
        self._action_dim = int(self.cfg.num_arm_joints)
        if self._action_dim != 7:
            raise ValueError(
                f"NMG ONNX policy controls 7 arm joints, got {self._action_dim}."
            )
        self._max_steps = int(self.cfg.max_steps)
        self._pos_eps = float(self.cfg.pos_eps)
        self._rot_eps = float(self.cfg.rot_eps)
        self._joint_eps = float(self.cfg.joint_eps)
        self._intermediate_orientation = bool(self.cfg.intermediate_orientation)
        self._policy = _OnnxPolicy(model_path, self.cfg.onnx_providers)
        self._obs_dim = self._policy.obs_dim
        self._policy_frame_from_world = self._as_transform(
            self.cfg.policy_frame_from_world, "policy_frame_from_world"
        )
        self._runtime_tcp_from_policy_tcp = self._as_transform(
            self.cfg.runtime_tcp_from_policy_tcp, "runtime_tcp_from_policy_tcp"
        )
        expected_obs_dim = _waypoint_obs_dim(
            self._num_waypoints, self._use_relative_obs
        )
        if self._obs_dim != expected_obs_dim:
            raise ValueError(
                f"ONNX input has obs dim {self._obs_dim}, but the configured "
                f"unified constraint layout requires {expected_obs_dim}."
            )

    def _as_transform(self, value: list[list[float]] | None, name: str) -> torch.Tensor:
        """Convert an optional homogeneous-transform config to a device tensor."""
        if value is None:
            return torch.eye(4, dtype=torch.float32, device=self.device)
        transform = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if transform.shape != (4, 4):
            raise ValueError(f"{name} must have shape (4, 4), got {transform.shape}.")
        return transform

    def _to_policy_frame(self, xpos: torch.Tensor) -> torch.Tensor:
        """Map runtime-world TCP poses to the NMG training base and TCP frame."""
        left = self._policy_frame_from_world.expand(xpos.shape[0], -1, -1)
        right = self._runtime_tcp_from_policy_tcp.expand(xpos.shape[0], -1, -1)
        return torch.bmm(torch.bmm(left, xpos), right)

    @validate_plan_options(options_cls=NeuralPlanOptions)
    @torch.no_grad()
    def plan(
        self,
        target_states: list[PlanState],
        options: NeuralPlanOptions = NeuralPlanOptions(),
    ) -> PlanResult:
        r"""Execute neural trajectory planning.

        Runs the waypoint transformer policy in closed loop for each environment
        until all waypoints are reached or ``max_steps`` is exhausted.

        Args:
            target_states: List of :class:`PlanState` waypoints. Each entry uses
                :attr:`MoveType.EEF_MOVE` with ``xpos`` shape ``(B, 4, 4)`` or
                :attr:`MoveType.JOINT_MOVE` with ``qpos`` shape ``(B, 7)``.
            options: :class:`NeuralPlanOptions` with ``control_part``,
                ``start_qpos``, and ``max_steps`` overrides.

        Returns:
            :class:`PlanResult` containing the planned trajectory. All tensor
            fields are env-batched with leading dim ``B``: ``success`` ``(B,)``,
            ``positions``/``velocities``/``accelerations`` ``(B, N, DOF)``,
            ``xpos_list`` ``(B, N, 4, 4)``, ``dt`` ``(B, N)``, and
            ``duration`` ``(B,)``. Velocities and accelerations are computed
            via finite differences and are therefore approximate.

        Raises:
            ValueError: If ``control_part`` is not provided, a target state is
                unsupported, or ``start_qpos`` has too few joints.
        """
        if not target_states:
            return PlanResult(
                success=torch.zeros(0, dtype=torch.bool, device=self.device),
                positions=None,
            )

        control_part = options.control_part or self.cfg.control_part
        if control_part is None:
            logger.log_error(
                "control_part is required for NeuralPlanner",
                ValueError,
            )

        (
            waypoints_pos,
            waypoints_quat,
            waypoints_joint,
            valid_mask,
            pos_mask,
            rot_mask,
            joint_mask,
            episode_k,
        ) = self._parse_waypoints(target_states)
        qpos = self._initial_qpos(control_part, options.start_qpos)
        b = qpos.shape[0]
        limits = self.robot.get_qpos_limits(name=control_part)[0].to(self.device)
        lower = limits[: self._action_dim, 0]
        upper = limits[: self._action_dim, 1]

        last_action = torch.zeros(b, self._action_dim, device=self.device)
        active_idx = torch.zeros(b, dtype=torch.long, device=self.device)
        positions = [qpos.clone()]
        xpos_list = [self._fk_matrix(qpos, control_part)]
        converged = torch.zeros(b, dtype=torch.bool, device=self.device)
        max_steps = int(options.max_steps or self._max_steps)

        with torch.no_grad():
            for _ in range(max_steps):
                ee_pose = self._fk_pose_xyzw(qpos, control_part)
                obs = self._build_obs(
                    qpos[:, : self._action_dim],
                    ee_pose,
                    waypoints_pos,
                    waypoints_quat,
                    waypoints_joint,
                    valid_mask,
                    pos_mask,
                    rot_mask,
                    joint_mask,
                    active_idx,
                    last_action,
                )
                action = self._policy(obs).clamp(-1.0, 1.0)
                # Hold converged envs: zero their action so qpos does not drift.
                # `converged` reflects state up to the end of the previous step, so
                # once an env converged at the end of step N its action is masked
                # from step N+1 onward.
                not_converged = ~converged
                action = torch.where(
                    not_converged.unsqueeze(-1), action, torch.zeros_like(action)
                )
                qpos[:, : self._action_dim] += action * float(self.cfg.action_scale)
                qpos[:, : self._action_dim] = torch.clamp(
                    qpos[:, : self._action_dim], lower, upper
                )
                last_action = torch.where(
                    not_converged.unsqueeze(-1), action, last_action
                )
                positions.append(qpos.clone())
                xpos_list.append(self._fk_matrix(qpos, control_part))

                ee_pose = self._fk_pose_xyzw(qpos, control_part)
                reached = self._is_active_reached(
                    qpos[:, : self._action_dim],
                    ee_pose,
                    waypoints_pos,
                    waypoints_quat,
                    waypoints_joint,
                    pos_mask,
                    rot_mask,
                    joint_mask,
                    active_idx,
                )
                active_idx = torch.where(reached, active_idx + 1, active_idx)
                converged = converged | (active_idx >= episode_k)
                if converged.all():
                    break

        positions_t = torch.stack(positions)
        xpos_t = torch.stack(xpos_list)
        dt = torch.full(
            (positions_t.shape[0],),
            float(self.cfg.dt),
            dtype=torch.float32,
            device=self.device,
        )
        dt = dt.unsqueeze(0).expand(b, -1)
        positions_t = positions_t.permute(1, 0, 2)
        xpos_t = xpos_t.permute(1, 0, 2, 3)
        velocities_t, accelerations_t = self._compute_vel_acc_via_finite_diff(
            positions_t, dt
        )
        success = active_idx >= episode_k
        return PlanResult(
            success=success,
            positions=positions_t,
            velocities=velocities_t,
            accelerations=accelerations_t,
            xpos_list=xpos_t,
            dt=dt,
            duration=torch.full(
                (b,),
                float(max(positions_t.shape[1] - 1, 0) * self.cfg.dt),
                device=self.device,
            ),
        )

    def _parse_waypoints(self, target_states: list[PlanState]) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        if len(target_states) > self._num_waypoints:
            logger.log_error(
                f"Received {len(target_states)} waypoints, but the ONNX policy supports "
                f"at most {self._num_waypoints}.",
                ValueError,
            )
        b = _infer_batch_size(target_states) or 1
        waypoint_pos = torch.zeros(b, self._num_waypoints, 3, device=self.device)
        waypoint_quat = torch.zeros(b, self._num_waypoints, 4, device=self.device)
        waypoint_quat[..., 3] = 1.0
        waypoint_joint = torch.zeros(
            b, self._num_waypoints, self._action_dim, device=self.device
        )
        valid_mask = torch.zeros(b, self._num_waypoints, device=self.device)
        pos_mask = torch.zeros_like(valid_mask)
        rot_mask = torch.zeros_like(valid_mask)
        joint_mask = torch.zeros_like(valid_mask)
        for idx, target in enumerate(target_states):
            if target.move_type == MoveType.EEF_MOVE and target.xpos is not None:
                xpos = torch.as_tensor(
                    target.xpos, dtype=torch.float32, device=self.device
                )
                if xpos.dim() == 2:
                    xpos = xpos.unsqueeze(0)
                policy_xpos = self._to_policy_frame(xpos)
                waypoint_pos[:, idx] = policy_xpos[:, :3, 3]
                quat_xyzw = convert_quat(
                    quat_from_matrix(policy_xpos[:, :3, :3]), to="xyzw"
                )
                waypoint_quat[:, idx] = (
                    _canonicalize_quat_xyzw(quat_xyzw)
                    if getattr(self, "_canonicalize_quat_obs", False)
                    else quat_xyzw
                )
                pos_mask[:, idx] = 1.0
                rot_mask[:, idx] = 1.0
            elif target.move_type == MoveType.JOINT_MOVE and target.qpos is not None:
                qpos = torch.as_tensor(
                    target.qpos, dtype=torch.float32, device=self.device
                )
                if qpos.dim() == 1:
                    qpos = qpos.unsqueeze(0)
                if qpos.shape != (b, self._action_dim):
                    logger.log_error(
                        "NeuralPlanner JOINT_MOVE qpos must have shape "
                        f"({b}, {self._action_dim}), got {tuple(qpos.shape)}.",
                        ValueError,
                    )
                waypoint_joint[:, idx] = qpos
                joint_mask[:, idx] = 1.0
            else:
                logger.log_error(
                    "NeuralPlanner expects EEF_MOVE entries with xpos or "
                    "JOINT_MOVE entries with qpos.",
                    ValueError,
                )
            valid_mask[:, idx] = 1.0
        if not self._intermediate_orientation:
            final_mask = torch.zeros_like(rot_mask)
            final_mask[:, len(target_states) - 1] = 1.0
            rot_mask *= final_mask
        return (
            waypoint_pos,
            waypoint_quat,
            waypoint_joint,
            valid_mask,
            pos_mask,
            rot_mask,
            joint_mask,
            len(target_states),
        )

    def _initial_qpos(
        self, control_part: str, start_qpos: torch.Tensor | None
    ) -> torch.Tensor:
        if start_qpos is None:
            qpos = self.robot.get_qpos(name=control_part)
        else:
            qpos = torch.as_tensor(start_qpos, dtype=torch.float32, device=self.device)
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.shape[-1] < self._action_dim:
            logger.log_error(
                f"start_qpos has {qpos.shape[-1]} joints, but policy expects "
                f"{self._action_dim}.",
                ValueError,
            )
        return qpos.to(self.device).clone()

    def _fk_matrix(self, qpos: torch.Tensor, control_part: str) -> torch.Tensor:
        return self.robot.compute_fk(qpos=qpos, name=control_part, to_matrix=True)

    def _fk_pose_xyzw(self, qpos: torch.Tensor, control_part: str) -> torch.Tensor:
        fk = self._to_policy_frame(self._fk_matrix(qpos, control_part))
        pos = fk[:, :3, 3]
        quat_xyzw = convert_quat(quat_from_matrix(fk[:, :3, :3]), to="xyzw")
        if getattr(self, "_canonicalize_quat_obs", False):
            quat_xyzw = _canonicalize_quat_xyzw(quat_xyzw)
        return torch.cat([pos, quat_xyzw], dim=-1)

    def _build_obs(
        self,
        joint_pos: torch.Tensor,
        ee_pose: torch.Tensor,
        waypoint_pos: torch.Tensor,
        waypoint_quat: torch.Tensor,
        waypoint_joint: torch.Tensor,
        valid_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        rot_mask: torch.Tensor,
        joint_mask: torch.Tensor,
        active_idx: torch.Tensor,
        last_action: torch.Tensor,
    ) -> torch.Tensor:
        b = joint_pos.shape[0]
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)
        active_onehot = torch.zeros(b, self._num_waypoints, device=self.device)
        active_onehot.scatter_(1, active_idx_clamped.unsqueeze(1), 1.0)
        pos_block = waypoint_pos * pos_mask.unsqueeze(-1)
        joint_block = waypoint_joint * joint_mask.unsqueeze(-1)
        identity = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=waypoint_quat.dtype,
            device=self.device,
        )
        quat_block = torch.where(
            rot_mask.unsqueeze(-1) > 0.5,
            waypoint_quat,
            identity.view(1, 1, 4),
        )
        obs_parts = [
            joint_pos,
            ee_pose,
            pos_block.reshape(b, self._num_waypoints * 3),
            quat_block.reshape(b, self._num_waypoints * 4),
            joint_block.reshape(b, self._num_waypoints * self._action_dim),
            active_onehot,
            valid_mask,
            pos_mask,
            rot_mask,
            joint_mask,
            last_action,
        ]
        if self._use_relative_obs:
            idx = torch.arange(b, device=self.device)
            active_pos = pos_block[idx, active_idx_clamped]
            active_quat = quat_block[idx, active_idx_clamped]
            active_joint = joint_block[idx, active_idx_clamped]
            inv_eef = _quat_inverse_xyzw(ee_pose[:, 3:7])
            active_rel_quat = _quat_mul_xyzw(active_quat, inv_eef)
            if getattr(self, "_canonicalize_quat_obs", False):
                active_rel_quat = _canonicalize_quat_xyzw(active_rel_quat)
            active_cart_rel = torch.cat(
                [active_pos - ee_pose[:, :3], active_rel_quat], dim=-1
            )
            active_rel = torch.where(
                (joint_mask[idx, active_idx_clamped] > 0.5).unsqueeze(-1),
                active_joint - joint_pos,
                active_cart_rel,
            )
            obs_parts.append(active_rel)
            rel_pos = (pos_block - ee_pose[:, None, :3]) * pos_mask.unsqueeze(-1)
            rel_quat = _quat_mul_xyzw(
                quat_block,
                inv_eef[:, None, :].expand_as(quat_block),
            )
            if getattr(self, "_canonicalize_quat_obs", False):
                rel_quat = _canonicalize_quat_xyzw(rel_quat)
            rel_quat = torch.where(
                rot_mask.unsqueeze(-1) > 0.5,
                rel_quat,
                identity.view(1, 1, 4),
            )
            joint_err = (joint_block - joint_pos[:, None]) * joint_mask.unsqueeze(-1)
            obs_parts.extend(
                [
                    rel_pos.reshape(b, self._num_waypoints * 3),
                    rel_quat.reshape(b, self._num_waypoints * 4),
                    joint_err.reshape(b, self._num_waypoints * self._action_dim),
                ]
            )
        waypoint_type = torch.zeros(
            b,
            self._num_waypoints,
            dtype=joint_pos.dtype,
            device=self.device,
        )
        waypoint_type = torch.where(
            (valid_mask > 0.5) & (rot_mask < 0.5),
            torch.ones_like(waypoint_type),
            waypoint_type,
        )
        waypoint_type = torch.where(
            joint_mask > 0.5,
            torch.full_like(waypoint_type, 2.0),
            waypoint_type,
        )
        obs_parts.append(waypoint_type)
        obs = torch.cat(obs_parts, dim=-1)
        if obs.shape[-1] != self._obs_dim:
            raise ValueError(
                f"Built obs dim {obs.shape[-1]}, expected {self._obs_dim}."
            )
        return obs

    def _is_active_reached(
        self,
        joint_pos: torch.Tensor,
        ee_pose: torch.Tensor,
        waypoint_pos: torch.Tensor,
        waypoint_quat: torch.Tensor,
        waypoint_joint: torch.Tensor,
        pos_mask: torch.Tensor,
        rot_mask: torch.Tensor,
        joint_mask: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> torch.Tensor:
        b = ee_pose.shape[0]
        idx = torch.arange(b, device=self.device)
        active_idx_clamped = torch.clamp(active_idx, max=self._num_waypoints - 1)
        active_pos = waypoint_pos[idx, active_idx_clamped]
        active_quat_xyzw = waypoint_quat[idx, active_idx_clamped]
        active_joint = waypoint_joint[idx, active_idx_clamped]
        active_pos_mask = pos_mask[idx, active_idx_clamped] > 0.5
        active_rot_mask = rot_mask[idx, active_idx_clamped] > 0.5
        active_joint_mask = joint_mask[idx, active_idx_clamped] > 0.5
        pos_dist = (ee_pose[:, :3] - active_pos).norm(dim=-1)
        ee_quat_wxyz = convert_quat(ee_pose[:, 3:7], to="wxyz")
        active_quat_wxyz = convert_quat(active_quat_xyzw, to="wxyz")
        rot_dist = quat_error_magnitude(ee_quat_wxyz, active_quat_wxyz)
        rot_ok = torch.where(
            active_rot_mask,
            rot_dist < self._rot_eps,
            torch.ones_like(rot_dist, dtype=torch.bool),
        )
        pos_ok = torch.where(
            active_pos_mask,
            pos_dist < self._pos_eps,
            torch.ones_like(pos_dist, dtype=torch.bool),
        )
        joint_dist = torch.amax(torch.abs(joint_pos - active_joint), dim=-1)
        joint_ok = torch.where(
            active_joint_mask,
            joint_dist < self._joint_eps,
            torch.ones_like(joint_dist, dtype=torch.bool),
        )
        reached = pos_ok & rot_ok & joint_ok
        return reached

    @staticmethod
    def _compute_vel_acc_via_finite_diff(
        positions: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Estimate velocities and accelerations via finite differences.

        Uses a second-order central difference for interior points and one-sided
        differences at the boundaries. The estimates are approximate because the
        neural policy does not produce velocities or accelerations directly.

        Args:
            positions: Joint positions of shape ``(B, N, DOF)``.
            dt: Per-point time deltas of shape ``(B, N)``. ``dt[:, t]`` is the
                interval used to reach point ``t`` from point ``t - 1``.

        Returns:
            Tuple of ``(velocities, accelerations)``, each of shape
            ``(B, N, DOF)``.
        """
        b, n, dof = positions.shape
        if n == 1:
            zeros = torch.zeros_like(positions)
            return zeros, zeros

        # Forward difference for the first point: (p[1] - p[0]) / dt[1]
        v_first = (positions[:, 1] - positions[:, 0]) / dt[:, 1].unsqueeze(-1)
        # Backward difference for the last point: (p[N-1] - p[N-2]) / dt[N-1]
        v_last = (positions[:, -1] - positions[:, -2]) / dt[:, -1].unsqueeze(-1)

        if n == 2:
            velocities = torch.stack([v_first, v_last], dim=1)
            return velocities, torch.zeros_like(velocities)

        # Central difference for interior points:
        # (p[i+1] - p[i-1]) / (dt[i] + dt[i+1])
        p_next = positions[:, 2:]
        p_prev = positions[:, :-2]
        dt_sum = (dt[:, 1:-1] + dt[:, 2:]).unsqueeze(-1)
        v_interior = (p_next - p_prev) / dt_sum.clamp_min(1e-12)
        velocities = torch.cat(
            [v_first.unsqueeze(1), v_interior, v_last.unsqueeze(1)], dim=1
        )

        # Acceleration via second-order finite differences.
        # Boundary points use a one-sided stencil; interior points use
        # (p[i+1] - 2*p[i] + p[i-1]) / dt[i]^2
        a_first = (positions[:, 2] - 2.0 * positions[:, 1] + positions[:, 0]) / (
            dt[:, 1].unsqueeze(-1) ** 2
        )
        a_last = (positions[:, -1] - 2.0 * positions[:, -2] + positions[:, -3]) / (
            dt[:, -1].unsqueeze(-1) ** 2
        )
        a_interior = (
            positions[:, 2:] - 2.0 * positions[:, 1:-1] + positions[:, :-2]
        ) / (dt[:, 1:-1].unsqueeze(-1) ** 2)
        accelerations = torch.cat(
            [a_first.unsqueeze(1), a_interior, a_last.unsqueeze(1)], dim=1
        )

        return velocities, accelerations
