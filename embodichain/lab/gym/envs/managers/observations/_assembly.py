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
"""Blind-assembly observations and training-only contact labels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import numpy as np
from embodichain.utils.math import euler_xyz_from_quat, quat_from_matrix
import torch.nn.functional as functional

from embodichain.lab.gym.envs.managers.cfg import FunctorCfg
from embodichain.lab.gym.envs.managers.manager_base import Functor

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = [
    "AssemblyProprioception",
    "assembly_privileged_state",
    "assembly_contact_labels",
]


class AssemblyProprioception(Functor):
    """Build per-environment proprioceptive history for a blind assembly actor.

    The configured layout determines whether the current frame comes before or
    after the history. A selective reset reinitializes only the requested rows.

    Args:
        cfg: Functor configuration containing the history length and layout.
        env: Assembly environment supplying joint features and actions.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        from embodichain.learning.rl.policies import ObservationHistory

        dimension = 36 if env.assembly_state.scene.assembly_kind == "nut" else 33
        self._history = ObservationHistory(
            env.num_envs,
            dimension,
            history_length=int(cfg.params.get("history_length", 30)),
            device=env.device,
            action_slice=(dimension - 7, dimension),
            layout=cfg.params.get("history_layout", "previous_current"),
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
        history_length: int = 30,
        history_layout: str = "previous_current",
    ) -> torch.Tensor:
        """Return history with the current frame without duplicating reads.

        Args:
            env: Assembly environment providing the current frame.
            obs: Existing observations, unused by this functor.
            history_length: Length also specified in the initialization config.
            history_layout: Layout also specified in the initialization config.

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


def _assembly_kinematic_chain(robot_cfg: Any, device: torch.device) -> Any:
    """Include the application's solver TCP in encoder-only FK and Jacobians."""
    chain = robot_cfg.build_pk_serial_chain(device=torch.device("cpu"))["arm"]
    tcp = np.asarray(robot_cfg.solver_cfg["arm"].tcp)
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


def _proprioceptive_frame(state) -> torch.Tensor:
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
    q = state.env.robot.get_qpos(name="arm")
    qvel = state.env.robot.get_qvel(name="arm")
    if partial:
        q, qvel = q[pending], qvel[pending]
    count = q.shape[0]
    jacobian, transform = state._assembly_chain.jacobian(q, ret_eef_pose=True)
    official = state._assembly_observation_convention == "official_v1"
    if official:
        transform = state._assembly_robot_base @ transform
    position = transform[:, :3, 3]
    rotation = transform[:, :3, :3]
    orientation = (
        rotation[:, :, :2].transpose(1, 2).reshape(count, 6)
        if state.scene.assembly_kind == "nut"
        else torch.stack(euler_xyz_from_quat(quat_from_matrix(rotation)), dim=-1)
    )
    twist = (jacobian @ qvel.unsqueeze(-1)).squeeze(-1)
    if official:
        base_rotation = state._assembly_robot_base[:3, :3]
        linear = twist[:, :3] @ base_rotation.T
        angular_world = twist[:, 3:] @ base_rotation.T
        angular_local = (
            rotation.transpose(1, 2) @ angular_world.unsqueeze(-1)
        ).squeeze(-1)
        twist = torch.cat([linear, angular_local], dim=-1)
        q = (q - state._assembly_q_center) / state._assembly_q_half_range
        qvel = qvel / state._assembly_velocity_scale
    previous = (
        state.env.action_manager.get_term("arm_delta").raw_actions
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
