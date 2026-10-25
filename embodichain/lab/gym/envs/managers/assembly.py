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
"""Reusable blind-assembly observations, reset events and sparse rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import torch
import torch.nn.functional as functional

from .manager_base import Functor
from .cfg import FunctorCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.assembly_env import BlindAssemblyEnv

__all__ = [
    "AssemblyProprioception",
    "assembly_privileged_state",
    "assembly_contact_labels",
    "reset_assembly",
    "assembly_milestone_reward",
]


class AssemblyProprioception(Functor):
    """Keep encoder-only history with selective reset and idempotent reads."""

    def __init__(self, cfg: FunctorCfg, env: BlindAssemblyEnv) -> None:
        super().__init__(cfg, env)
        from embodichain.learning.rl.policies import ObservationHistory

        dimension = 36 if env.assembly_kind == "nut" else 33
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
        if env_ids is None:
            self._reset_rows.fill_(True)
        else:
            self._reset_rows[env_ids] = True
        self._needs_reset = True

    def __call__(
        self,
        env: BlindAssemblyEnv,
        obs: Any,
        history_length: int = 30,
        history_layout: str = "previous_current",
    ) -> torch.Tensor:
        env.refresh_assembly_state()
        changed = self._revision != env._assembly_physics_revision
        if changed or self._needs_reset:
            frame = env.proprioceptive_frame()
            if changed:
                self._history.append(frame, reset_mask=self._reset_rows)
            else:
                # Async reset may request observations twice in one physical
                # step. Replace only reset rows; do not shift other histories.
                self._history.reset(frame, env_ids=self._reset_rows)
            self._reset_rows.zero_()
            env._assembly_reset_rows.zero_()
            self._needs_reset = False
            self._revision = env._assembly_physics_revision
        return self._history.observation


def assembly_privileged_state(env: BlindAssemblyEnv, obs: Any) -> torch.Tensor:
    """Expose true mating pose and contact classes solely to the critic."""
    env.refresh_assembly_state()
    classes = 4 if env.assembly_kind == "nut" else 3
    labels = (
        functional.one_hot(env._assembly_labels, num_classes=classes).flatten(1).float()
    )
    metrics = env._assembly_metrics
    fields = [
        env.proprioceptive_frame(),
        metrics["relative_pose"],
        torch.stack(
            [metrics[name] for name in ("radial_error", "tilt", "depth")], dim=-1
        ),
        labels,
    ]
    if env.assembly_kind == "nut":
        config = env.assembly_asset.config
        geometry = torch.tensor(
            [config["major_diameter"], config["pitch"], config["nut_height"]],
            device=env.device,
        )
        fields.append(geometry.expand(env.num_envs, -1))
    return torch.cat(fields, dim=-1)


def assembly_contact_labels(env: BlindAssemblyEnv, obs: Any) -> torch.Tensor:
    """Return pre-action mating-feature class labels for auxiliary training."""
    env.refresh_assembly_state()
    return env._assembly_labels.clone()


def reset_assembly(env: BlindAssemblyEnv, env_ids: torch.Tensor) -> None:
    """Initialize selected physical grasps without stepping other arenas."""
    env.reset_assembly_rows(env_ids)


def assembly_milestone_reward(
    env: BlindAssemblyEnv, obs: Any, action: torch.Tensor, info: dict[str, Any]
) -> torch.Tensor:
    """Pay each capture/mesh/seat milestone once in the current episode."""
    env.refresh_assembly_state()
    weights = (
        [1.0, 0.0, 1.0, 0.0] if env.assembly_kind == "peg" else [1.0, 1.0, 1.0, 0.0]
    )
    return (
        env._assembly_new_milestones.float() * torch.tensor(weights, device=env.device)
    ).sum(dim=-1)
