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

"""Manager functors implementing robot-held assembly task semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as functional

from embodichain.lab.gym.envs.managers.cfg import FunctorCfg
from embodichain.lab.gym.envs.managers.manager_base import Functor
from embodichain.lab.gym.envs.objectives.conditions import ConditionResult
from embodichain.utils.math import euler_xyz_from_quat, quat_from_matrix
from ._defaults import assembly_defaults

_PARAMETERS = assembly_defaults("parameters")

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = [
    "AssemblyProprioception",
    "assembly_privileged_state",
    "assembly_contact_labels",
    "reset_assembly",
    "assembly_success",
    "assembly_failure",
    "assembly_milestone_reward",
]


class AssemblyProprioception(Functor):
    """Build per-environment proprioceptive history for a blind assembly actor.

    Current comes first, followed by chronological history including current.
    A selective reset reinitializes only the requested rows.

    Args:
        cfg: Functor configuration containing the history length and layout.
        env: Assembly environment supplying joint features and actions.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        from embodichain.learning.rl.policies import ObservationHistory

        dimension = env.assembly_state.frame_dim
        self._history = ObservationHistory(
            env.num_envs,
            dimension,
            history_length=int(
                cfg.params.get(
                    "history_length", _PARAMETERS["observation_history_length"]
                )
            ),
            device=env.device,
            action_slice=(dimension - env.assembly_state.arm_dof, dimension),
        )
        self._revision = -1
        self._needs_reset = True
        self._reset_rows = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Any = None) -> None:
        """Mark selected histories for reinitialization on the next observation.

        Args:
            env_ids: Environment indices to reset, or ``None`` for every row.
        """
        if env_ids is None:
            self._reset_rows.fill_(True)
        else:
            self._reset_rows[env_ids] = True
        self._needs_reset = True

    def __call__(
        self,
        env: EmbodiedEnv,
        obs: Any,
        history_length: int = _PARAMETERS["observation_history_length"],
    ) -> torch.Tensor:
        """Return history with the current frame without duplicating reads.

        Args:
            env: Assembly environment providing the current frame.
            obs: Existing observations, unused by this functor.
            history_length: Length also specified in the initialization config.

        Returns:
            Batched proprioceptive history flattened per environment.
        """
        changed = self._revision != env.assembly_state._assembly_physics_revision
        if changed or self._needs_reset:
            frame = _proprioceptive_frame(env.assembly_state)
            if changed:
                self._history.append(frame, reset_mask=self._reset_rows)
            else:
                # Async reset may request observations twice in one physical
                # step. Replace only reset rows; do not shift other histories.
                self._history.reset(frame, env_ids=self._reset_rows)
            self._reset_rows.zero_()
            self._needs_reset = False
            self._revision = env.assembly_state._assembly_physics_revision
        return self._history.observation


def assembly_privileged_state(env: EmbodiedEnv, obs: Any) -> torch.Tensor:
    """Build privileged mating features intended for the training critic.

    The policy configuration must keep this observation out of the actor group.

    Args:
        env: Assembly environment containing the task state and geometry.
        obs: Existing observations, unused by this functor.

    Returns:
        Flattened batched pose, geometry and contact-class features.
    """
    env.assembly_state.refresh_assembly_state()
    classes = 4 if env.assembly_state.scene.assembly_kind == "nut" else 3
    labels = (
        functional.one_hot(env.assembly_state._assembly_labels, num_classes=classes)
        .flatten(1)
        .float()
    )
    metrics = env.assembly_state._assembly_metrics
    fields = [
        _proprioceptive_frame(env.assembly_state),
        metrics["relative_pose"],
        torch.stack(
            [metrics[name] for name in ("radial_error", "tilt", "depth")], dim=-1
        ),
        labels,
    ]
    if env.assembly_state.scene.assembly_kind == "nut":
        config = env.assembly_state.scene.assembly_asset.config
        geometry = torch.tensor(
            [config["major_diameter"], config["pitch"], config["nut_height"]],
            device=env.device,
        )
        fields.append(geometry.expand(env.num_envs, -1))
    return torch.cat(fields, dim=-1)


def assembly_contact_labels(env: EmbodiedEnv, obs: Any) -> torch.Tensor:
    """Return pre-action mating-feature class labels for auxiliary training.

    Args:
        env: Assembly environment containing current feature labels.
        obs: Existing observations, unused by this functor.

    Returns:
        Independent batched integer labels for the mating features.
    """
    env.assembly_state.refresh_assembly_state()
    return env.assembly_state._assembly_labels.clone()


def _assembly_kinematic_chain(
    robot_cfg: Any, device: torch.device, arm_part: str
) -> Any:
    """Include the application's solver TCP in encoder-only FK and Jacobians."""
    chain = robot_cfg.build_pk_serial_chain(device=torch.device("cpu"))[arm_part]
    tcp = np.asarray(robot_cfg.solver_cfg[arm_part].tcp)
    if not np.array_equal(tcp, np.eye(4)):
        import pytorch_kinematics as pk

        end_frame = chain.find_frame(chain.get_frame_names(False)[-1])
        name = end_frame.name + "_assembly_tcp"
        end_frame.add_child(
            pk.Frame(
                name=name,
                link=pk.Link(name=name),
                joint=pk.Joint(
                    offset=pk.Transform3d(
                        matrix=torch.as_tensor(tcp, dtype=chain.dtype)
                    ),
                    joint_type="fixed",
                ),
            )
        )
        chain = pk.SerialChain(chain, name, dtype=chain.dtype)
    return chain.to(device=device)


def _proprioceptive_frame(state: Any) -> torch.Tensor:
    """Encoder-only 33/36-value frame using FK and Jacobian times qvel."""
    cached = getattr(state, "_assembly_frame_cache", None)
    if cached is not None and cached[0] == state._assembly_revision:
        return cached[1]
    physics_revision = getattr(
        state, "_assembly_physics_revision", state._assembly_revision
    )
    pending = getattr(state, "_assembly_frame_reset_ids", None)
    partial = (
        cached is not None
        and getattr(state, "_assembly_frame_physics_revision", -1) == physics_revision
        and pending is not None
        and 0 < len(pending) < state.env.num_envs
    )
    q = state.env.robot.get_qpos(name=state.cfg.arm_part)
    qvel = state.env.robot.get_qvel(name=state.cfg.arm_part)
    if partial:
        q, qvel = q[pending], qvel[pending]
    count = q.shape[0]
    jacobian, transform = state._assembly_chain.jacobian(q, ret_eef_pose=True)
    transform = state._assembly_robot_base @ transform
    position = transform[:, :3, 3]
    rotation = transform[:, :3, :3]
    orientation = (
        rotation[:, :, :2].transpose(1, 2).reshape(count, 6)
        if state.cfg.orientation_representation == "rotation6d"
        else torch.stack(euler_xyz_from_quat(quat_from_matrix(rotation)), dim=-1)
    )
    twist = (jacobian @ qvel.unsqueeze(-1)).squeeze(-1)
    base_rotation = state._assembly_robot_base[:3, :3]
    linear = twist[:, :3] @ base_rotation.T
    angular_world = twist[:, 3:] @ base_rotation.T
    angular_local = (rotation.transpose(1, 2) @ angular_world.unsqueeze(-1)).squeeze(-1)
    twist = torch.cat([linear, angular_local], dim=-1)
    q = (q - state._assembly_q_center) / state._assembly_q_half_range
    qvel = qvel / state._assembly_velocity_scale
    previous = (
        state.env.action_manager.get_term(state.cfg.action_term).raw_actions
        if state.env.action_manager is not None
        else None
    )
    if previous is None:
        previous = torch.zeros_like(q)
    elif partial:
        previous = previous[pending]
    frame = torch.cat([q, qvel, position, orientation, twist, previous], dim=-1)
    if partial:
        full_frame = cached[1].clone()
        full_frame[pending] = frame
        frame = full_frame
    state._assembly_frame_cache = (state._assembly_revision, frame)
    state._assembly_frame_physics_revision = physics_revision
    state._assembly_frame_reset_ids = None
    return frame


def reset_assembly(env: EmbodiedEnv, env_ids: torch.Tensor) -> None:
    """Initialize selected physical grasps without stepping other arenas."""
    state = env.assembly_state
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=state.env.device)
    n = len(ids)
    if n == 0:
        return
    q = state.env.robot.get_qpos(name=state.cfg.arm_part)[ids]
    tcp = (
        state._assembly_robot_base[None]
        @ state._assembly_chain.forward_kinematics(q).get_matrix()
    )
    turn = torch.diag(torch.tensor([1.0, -1.0, -1.0], device=state.env.device))
    rotation = tcp[:, :3, :3] @ turn
    # Keep the upper grip section between the physical pads. Gripper
    # closure remains a force-limited drive; no fixed grasp constraint.
    grip_z = (
        (
            float(state.scene.assembly_asset.config["face_width"]) / 2
            + state.cfg.gear_grip_offsets[0]
            + state.cfg.gear_grip_offsets[1]
        )
        if state.scene.assembly_kind == "gear"
        else state.scene._moving_zmax - state.cfg.grip_depth
    )
    part = torch.eye(4, device=state.env.device).repeat(n, 1, 1)
    part[:, :3, :3] = rotation
    part[:, :3, 3] = tcp[:, :3, 3] - (
        rotation
        @ torch.tensor([0.0, 0.0, grip_z], device=state.env.device)
        .expand(n, -1)
        .unsqueeze(-1)
    ).squeeze(-1)
    grasp_jitter = state.cfg.grasp_position_range
    # Along-pad displacement is along z; lateral x does not change the
    # closing aperture. A y displacement would create an invalid initial
    # one-sided clamp rather than the paper's along-pad randomization.
    part[:, :3, 3] += (
        rotation
        @ (
            (torch.rand((n, 3), device=state.env.device) - 0.5)
            * torch.tensor([0.0, 0.0, 2 * grasp_jitter], device=state.env.device)
        ).unsqueeze(-1)
    ).squeeze(-1)
    fixture = part.clone()
    initial_offset = state.cfg.initial_position_range
    offset = (torch.rand((n, 3), device=state.env.device) - 0.5) * torch.tensor(
        [2 * initial_offset, 2 * initial_offset, 0.0], device=state.env.device
    )
    offset[:, 2] = state.scene._moving_zmin - state.scene._fixture_zmax
    fixture[:, :3, 3] += (rotation @ offset.unsqueeze(-1)).squeeze(-1)
    state.env.sim.get_rigid_object(state.scene._moving_uid).set_local_pose(
        part, env_ids=ids
    )
    state.env.sim.get_rigid_object(state.scene._moving_uid).set_velocity(
        lin_vel=torch.zeros((n, 3), device=state.env.device),
        ang_vel=torch.zeros((n, 3), device=state.env.device),
        env_ids=ids,
    )
    state.env.sim.get_rigid_object(state.scene._fixture_uid).set_local_pose(
        fixture, env_ids=ids
    )
    if state.scene.assembly_kind == "gear":
        # Every fixture entity is transformed rigidly with the shaft.
        anchor = state.env.sim.get_articulation(state.scene._hinge_uid)
        source_shaft = np.asarray(state.scene._fixture_definition.position)
        source_anchor = next(
            b
            for b in state.scene.assembly_asset.bodies
            if b.name == state.scene._hinge_uid
        )
        local = torch.as_tensor(
            np.asarray(source_anchor.position) - source_shaft,
            dtype=torch.float32,
            device=state.env.device,
        )
        anchor_pose = fixture.clone()
        anchor_pose[:, :3, 3] += (rotation @ local.expand(n, -1).unsqueeze(-1)).squeeze(
            -1
        )
        anchor.set_local_pose(anchor_pose, env_ids=ids)
        anchor.set_qpos(
            torch.rand((n, 1), device=state.env.device) * 2 * torch.pi,
            env_ids=ids,
            target=False,
        )
    grip_ids, grip_values = _grasp_joint_reset(
        state.env.robot,
        state.cfg.gripper_part,
        state.scene._grasp_reset_joint,
        n,
        state.env.device,
    )
    state.env.robot.set_qpos(
        grip_values,
        joint_ids=grip_ids,
        env_ids=ids,
        target=False,
    )
    state.env.robot.set_qpos(
        torch.full(
            (n, len(state.env.robot.get_joint_ids(name=state.cfg.gripper_part))),
            state.scene._grasp_target_joint,
            device=state.env.device,
        ),
        name=state.cfg.gripper_part,
        env_ids=ids,
    )
    state._assembly_reset_rows[ids] = True
    state._assembly_success_hold[ids] = 0
    state._assembly_milestones[ids] = False
    state._assembly_new_milestones[ids] = False
    state._assembly_angle[ids] = 0
    state._assembly_previous_yaw[ids] = 0
    state._assembly_capture_angle[ids] = float("nan")
    state._assembly_capture_depth[ids] = float("nan")
    for field in ("_assembly_pending_reset_ids", "_assembly_frame_reset_ids"):
        previous = getattr(state, field, None)
        setattr(
            state,
            field,
            (
                ids.clone()
                if previous is None
                else torch.unique(torch.cat((previous, ids)))
            ),
        )
    state._assembly_revision += 1


def assembly_milestone_reward(
    env: EmbodiedEnv,
    obs: Any,
    action: torch.Tensor,
    info: dict[str, Any],
    weights: list[float],
) -> torch.Tensor:
    """Weight the newly reached milestones in this control transition.

    Args:
        env: Host containing the shared assembly measurement state.
        obs: Current manager observations.
        action: Policy action for this transition.
        info: Current environment outcomes.
        weights: Four finite capture/depth/success/reserved milestone weights.

    Returns:
        Batched reward; the state records each milestone once per episode.
    """
    env.assembly_state.refresh_assembly_state()
    if len(weights) != 4 or not np.isfinite(weights).all():
        raise ValueError("Assembly milestone weights must contain four finite values.")
    return (
        env.assembly_state._assembly_new_milestones.float()
        * torch.tensor(weights, device=env.device)
    ).sum(dim=-1)


def _grasp_joint_reset(
    robot: Any, gripper_part: str, value: float, rows: int, device: torch.device
) -> tuple[list[int], torch.Tensor]:
    """Expand the EEF leaders through source mimic relations without name rules."""
    values = {int(i): value for i in robot.get_joint_ids(name=gripper_part)}
    relations = list(
        zip(
            robot.mimic_ids,
            robot.mimic_parents,
            robot.mimic_multipliers,
            robot.mimic_offsets,
        )
    )
    for _ in range(len(relations) + 1):
        for child, parent, multiplier, offset in relations:
            if parent in values:
                values[int(child)] = values[parent] * multiplier + offset
    indices = sorted(values)
    return indices, torch.tensor(
        [values[i] for i in indices], dtype=torch.float32, device=device
    ).expand(rows, -1)


def assembly_success(env: EmbodiedEnv) -> ConditionResult:
    """Read physical success from the configured interval measurement snapshot.

    Args:
        env: Environment with startup-bound assembly state.

    Returns:
        Physical success, validity and shared geometric diagnostics.
    """
    state = env.assembly_state
    state.refresh_assembly_state()
    values = state._assembly_metrics
    metrics = {
        k: v for k, v in values.items() if k not in {"success", "fail", "relative_pose"}
    }
    valid = torch.isfinite(values["relative_pose"]).all(dim=-1)
    return ConditionResult(values["success"], valid, metrics)


def assembly_failure(env: EmbodiedEnv) -> ConditionResult:
    """Read failure from the snapshot consumed by success and rewards.

    Args:
        env: Environment with startup-bound assembly state.

    Returns:
        Failure and measurement-validity masks.
    """
    state = env.assembly_state
    state.refresh_assembly_state()
    values = state._assembly_metrics
    return ConditionResult(
        values["fail"], torch.isfinite(values["relative_pose"]).all(dim=-1), {}
    )
